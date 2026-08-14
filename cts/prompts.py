"""Everything the vision model is ever told, plus the schema of what it must say back.

This file is the highest-leverage thing in the project. No downstream component can
recover a detail the vision pass failed to record, so the prompt below is written to
be exhaustive rather than short.

Three rules govern this module:

* The vision model NEVER sees a card name, oracle text, set, artist, or any other
  metadata. `VISION_PROMPT` is a constant: nothing is interpolated into it, ever.
  Alternate arts of one commander depict different scenes, and priming the model with
  the name collapses every printing toward the same generic description.
* `PROMPT_VERSION` is stamped onto every `descriptions` row. Bump it on any change to
  `VISION_PROMPT`, `VISION_SCHEMA`, or `VISION_OPTIONS` — anything that changes what
  comes back — so `python -m cts describe --backfill-stale` can find the stale rows.
* The two layers have opposite epistemic rules and must never be merged. See SPEC.md
  Phase 5.
"""

from __future__ import annotations

# Bump on ANY change to the prompt text, the schema, or the generation options.
PROMPT_VERSION: int = 1


# --- Shape of the output -------------------------------------------------
# Named here so the JSON schema and describe.py's validator cannot drift apart.

PRIMARY_SUBJECT_KEYS: tuple[str, ...] = (
    "species",
    "facial_hair",
    "held_objects",
    "clothing",
    "pose",
)

SLOT_KEYS: tuple[str, ...] = (
    "primary_subject",
    "other_figures",
    "figure_count",
    "setting",
    "time_of_day",
    "palette",
    "art_style",
    "composition",
)

TOP_LEVEL_KEYS: tuple[str, ...] = (
    "literal",
    "literal_propositions",
    "slots",
    "interpretive",
    "interpretive_propositions",
)

# The prompt asks for 10-20 literal and 8-15 interpretive propositions. The schema
# floor sits slightly below the ask so a grammar-constrained decoder is never forced
# to pad with filler, and the validator rejects anything under the floor.
LITERAL_MIN, LITERAL_MAX = 8, 20
INTERPRETIVE_MIN, INTERPRETIVE_MAX = 6, 15

# Values that mean "I looked, and the attribute is absent". Slot fold-ins with one of
# these values are dropped rather than indexed: a proposition reading "primary subject
# facial hair: none" embeds uncomfortably close to a query for beards, so writing the
# absence into the search index actively hurts precision. The absence still lives in
# descriptions.slots, where structured filters can read it.
ABSENT_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "n/a",
        "na",
        "nil",
        "none",
        "not applicable",
        "not present",
        "not visible",
        "nothing",
        "null",
        "unknown",
        "unspecified",
    }
)


def _obj(properties: dict, required: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


# Key order below is the order the spec writes it in, and the order we want the model
# to think in: record the facts, then read your own facts and interpret them. Ollama
# converts this schema to a decoding grammar and may not preserve property order, so
# the order is a hint, not a guarantee — which is why each layer's rules are stated
# self-sufficiently in the prompt rather than relying on sequence.
VISION_SCHEMA: dict = _obj(
    {
        "literal": {
            "type": "string",
            "description": "Dense factual paragraph. Only what is physically visible.",
        },
        "literal_propositions": {
            "type": "array",
            "minItems": LITERAL_MIN,
            "maxItems": LITERAL_MAX,
            "items": {"type": "string"},
            "description": "10-20 atomic, self-contained, verifiable statements.",
        },
        "slots": _obj(
            {
                "primary_subject": _obj(
                    {
                        "species": {"type": "string"},
                        "facial_hair": {"type": "string"},
                        "held_objects": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 6,
                            "items": _obj(
                                {
                                    "object": {"type": "string"},
                                    "is_weapon": {"type": "boolean"},
                                },
                                ("object", "is_weapon"),
                            ),
                        },
                        "clothing": {"type": "string"},
                        "pose": {"type": "string"},
                    },
                    PRIMARY_SUBJECT_KEYS,
                ),
                "other_figures": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": _obj(
                        {"species": {"type": "string"}, "role": {"type": "string"}},
                        ("species", "role"),
                    ),
                },
                "figure_count": {"type": "integer"},
                "setting": {"type": "string"},
                "time_of_day": {"type": "string"},
                "palette": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 8,
                    "items": {"type": "string"},
                },
                "art_style": {"type": "string"},
                "composition": {"type": "string"},
            },
            SLOT_KEYS,
        ),
        "interpretive": {
            "type": "string",
            "description": "Paragraph on what the image conveys. Impressions, not facts.",
        },
        "interpretive_propositions": {
            "type": "array",
            "minItems": INTERPRETIVE_MIN,
            "maxItems": INTERPRETIVE_MAX,
            "items": {"type": "string"},
        },
    },
    TOP_LEVEL_KEYS,
)


# Generation options for the vision call.
#
# num_ctx is not decoration. The prompt below is ~3.5k tokens, an art crop costs
# anywhere from ~400 tokens (patch-merging models) to ~6.4k (tiled models such as
# llama3.2-vision at four tiles), and the response runs 1.2k to 1.8k. That total blows
# straight past Ollama's default context, and the failure mode is silent: the head of
# the prompt — every rule in it — is dropped, and the model returns schema-valid mush
# for hours. 16k covers the worst case with room to spare; lower it only if VRAM is
# tight, and re-read a few descriptions afterwards if you do.
#
# Temperature is low because the literal layer is the layer that must not drift; the
# interpretive layer gets its range from the explicit checklist in the prompt, not
# from sampling noise. No seed is set, so the single retry has fresh entropy.
VISION_OPTIONS: dict = {
    "temperature": 0.3,
    "top_p": 0.9,
    "num_ctx": 16384,
    "num_predict": 3000,
}


VISION_PROMPT: str = """\
You are a visual description instrument for an art search index.

You are shown ONE cropped illustration. There is no card frame, no border, no title,
no caption, and nobody will ever tell you what this picture is, what it is called, or
where it came from. That is deliberate. Describe the image in front of you and nothing
else.

WHY THIS MATTERS

Your description is the only record of this image the search system will ever have.
The picture itself is never looked at again. If you do not write something down, then
as far as this system is concerned it is not there: an unmentioned lantern cannot be
found by anyone searching for lanterns, and an unmentioned feeling of loneliness
cannot be found by anyone searching for lonely art.

People search this index with the whole range at once: "figures with beards",
"holding something that isn't a weapon", "a single figure against a huge empty
background", "painterly, muted, almost watercolor", "looks lonely", "the moment right
before a betrayal", "would fit on a black metal album cover". One description has to
serve all of it. That is why you write two layers.

THE TWO LAYERS HAVE OPPOSITE RULES

  LITERAL       Only what a camera records. No inference, no story, no mood, no
                names. Every statement must be one that two careful strangers looking
                at this image would both agree is true. If they could reasonably
                disagree, it does not belong in this layer.

  INTERPRETIVE  Only what a camera cannot record. Mood, implied story, power, genre,
                register, analogy. Here you are permitted to be wrong. You are not
                permitted to be vague. A confident, specific reading that turns out
                debatable is useful; "the mood is interesting" is worthless.

Never mix them. No feeling belongs in the literal layer. No new physical fact belongs
in the interpretive layer. The literal layer records the evidence. The interpretive
layer says what the evidence adds up to. A reader must be able to disagree with your
entire interpretation and still trust every literal statement you made.

RULES FOR THE LITERAL LAYER

1. Describe only what is visible in this image. Not what is probably just outside the
   frame, not what usually accompanies such a scene.
2. Name nothing. No character names, no place names, no story, no world, no franchise,
   no artist, no title. If you think you recognize the picture, describe it anyway as
   though you have never seen it before. Recognition is the single most common way
   this task is failed: it replaces what is actually painted with what you remember.
3. Never state an emotion, intention, or relationship. State the visible evidence for
   it instead. Not "looks furious" but "brows drawn low, teeth bared, fist clenched".
   Not "protecting the child" but "stands between the child and the open doorway, arm
   extended sideways".
4. Be specific in a way a search can use. Name colors precisely ("desaturated slate
   blue", "warm ochre", "ember orange"), not "colorful" or "dark". Name materials and
   textures ("hammered bronze", "wet fur", "cracked lacquer"). Count things exactly
   when there are ten or fewer, and estimate above that ("roughly twenty spear tips").
   State where things are: foreground, midground, background, left, right, behind,
   above, in front of.
5. Describe light explicitly: where it comes from, its color, how hard the shadows
   are, whether anything glows.
6. If there is writing or a symbol in the image, say where it is and what it looks
   like. Transcribe it only if it is short and unmistakably legible. Never guess at
   what illegible marks say.

THE PRIMARY SUBJECT

Exactly one figure is the primary subject: the one the composition is about, normally
the largest, most central, most sharply rendered, or most brightly lit. Everything
else that lives is an "other figure". A bearded villager standing behind a dragon does
not make the dragon bearded, and that confusion is the single worst failure available
to you, so decide who the primary subject is before you fill in anything else.

If two figures genuinely compete, choose the one that occupies more of the frame or is
in sharper focus, and record the other in other_figures with a role.

If nothing living is depicted at all — a landscape, an object, an empty building — set
species to "none" and the other primary_subject fields to "none", set figure_count to
0, and carry the whole description in setting, composition, and the propositions.

FILLING THE SLOTS

Every slot key must be present in your answer. Never omit one. If an attribute is
genuinely absent, write the exact word "none" — that means "I looked, and it is not
there". If you looked but genuinely cannot see it, say so and why, briefly
("obscured by a deep hood"). A missing key is the one thing that breaks this system,
because then absence and inattention look the same.

  species        Plain visual vocabulary for what kind of being it is: "human",
                 "elf-like humanoid", "dragon", "skeletal humanoid", "armored
                 insectoid", "wolf". Never a named species from any fiction. If
                 uncertain, say "humanoid" and add the distinguishing features.
  facial_hair    Be exact; people search this directly. "full gray beard", "short
                 black stubble", "long white beard braided with gold rings",
                 "mustache only". Use "clean-shaven" when a bare humanlike face is
                 visible, "none" when the creature has no such feature at all, and
                 name the obstruction when the face is hidden.
  held_objects   Only what the primary subject is actually holding or supporting —
                 in hands, claws, talons, tendrils. A sheathed sword on a belt or a
                 bow slung across the back is worn, not held: put it in clothing or a
                 proposition instead. Set is_weapon true only when the object's
                 purpose is to injure (sword, axe, spear, bow, dagger, mace, firearm).
                 A lantern, book, staff, goblet, instrument, tool, banner, chain,
                 severed head, infant or animal is not a weapon, even in a violent
                 scene. If the hands are empty, emit exactly one entry with object
                 "none" and is_weapon false.
  clothing       Garments, armor, headwear, jewelry: materials, colors, layers,
                 condition. "none" if unclothed or if the subject is a bare creature.
  pose           What the body is doing: stance, what each limb does, where the head
                 and gaze point, whether the subject is still or mid-motion, and
                 whether it faces the viewer or away.
  other_figures  One entry per other distinguishable being or group. Roles are like
                 "background", "midground", "companion", "mount", "opponent",
                 "crowd", "silhouette", "victim". If there are none, emit exactly one
                 entry with species "none" and role "none".
  figure_count   Every living being in the image including the primary subject and
                 including animals. Count exactly up to ten and estimate above
                 that. 0 if none.
  setting        Where this is: terrain, structures, interior or exterior, weather,
                 notable objects in the environment.
  time_of_day    "day", "dawn", "dusk", "night", "indoors, artificial light", or
                 "indeterminate" — plus the light evidence in a few words.
  palette        Three to eight precise color terms, dominant first.
  art_style      Medium, technique and finish, since people search this directly:
                 "oil painting, heavy impasto, warm varnish", "digital painting,
                 painterly, visible brushwork", "ink linework with flat cel shading",
                 "woodcut-like hard black shapes", "photoreal rendering, smooth
                 gradients", "muted watercolor with wet edges".
  composition    Framing (close-up, medium, full figure, wide), where the subject sits
                 in the frame, depth layers, negative space, camera height, symmetry,
                 leading lines. Phrases like "one small figure against a vast empty
                 sky" are exactly what people type.

LITERAL PROPOSITIONS: 10 TO 20 OF THEM

Each one is indexed and retrieved on its own, so each must survive alone.

* Atomic. One fact per statement.
* Self-contained. Name what you are talking about: "the primary figure", "a
  background figure", "the sky", "the left hand". Never use he, she, it, they, this,
  that, the same, or also. "has a full gray beard" is fine; "he has one" is useless.
* No cross-references. Each statement stands without the one before it.
* No repetition. Every statement must add information the others do not carry.
* Collectively exhaustive. Between them, cover: the primary subject's body, face and
  expression-evidence, hair, held objects, clothing and armor; each other figure;
  animals; architecture and terrain; the sky and weather; the light source and its
  direction; notable small objects; anything unusual, damaged, or out of place.
* No names, no lore, no mood, no judgment of quality.

INTERPRETIVE PROPOSITIONS: 8 TO 15 OF THEM

Now invert everything above. Do not restate a single fact here. Cover, at minimum:

* Emotional register and mood — what the image feels like.
* Implied narrative — what has just happened, what is about to happen.
* Power dynamic — who holds it, who does not; when a figure is alone, the dynamic
  between that figure and the world around it.
* Genre and tonal register — horror, heroic fantasy, folk tale, tragedy, comedy,
  noir, elegy, propaganda poster, quiet slice of life.
* What kind of story this looks like a frame from, and what part of it: an opening,
  a turning point, an aftermath.
* At least two analogies. Film, music, an album sleeve, a period or art movement, a
  kind of book. This is what makes queries like "would fit on a black metal album
  cover" or "feels like a Ghibli character" reachable at all.
* Whether the image is loud or quiet, still or kinetic, intimate or vast.

Write in the register of these: "feels isolated and resigned"; "reads as the calm
immediately before violence"; "romantic-era heroic framing"; "would sit comfortably on
a doom metal record sleeve"; "the threat is implied and offscreen rather than shown".

Commit. Do not hedge with "possibly", "perhaps", "may suggest". Every impression must
be caused by something in the picture, but none of them has to be provable — that is
the entire difference between this layer and the one above it. Being wrong here is
recoverable. Being vague here is not, because a vague impression matches no query ever
typed.

A WORKED EXAMPLE

For an imaginary painting: a woman in a red oilskin coat stands at the end of a wooden
pier at dusk holding a lit paper lantern, a large black dog sitting on the planks
behind her, fog over the water, an unlit ferris wheel on the far shore.

Good literal propositions:
  "the primary figure wears a red oilskin coat that reaches mid-calf"
  "the primary figure holds a lit paper lantern in the right hand at waist height"
  "a large black dog sits on the planks roughly two meters behind the primary figure"
  "the pier is built from weathered gray planks with visible gaps between them"
  "an unlit ferris wheel stands on the far shore, muted to gray by fog"
  "the sky graduates from pale apricot at the horizon to slate blue overhead"

Bad literal propositions, and why:
  "she is waiting for someone"          - a story, not a fact, and starts with a pronoun
  "the pier feels abandoned"            - a feeling; belongs in the other layer
  "a beautiful seaside scene"           - a judgment, and carries no information
  "holds it up high"                    - meaningless on its own

Abbreviated slots for the same image (yours must be complete):
  "primary_subject": {"species": "human", "facial_hair": "none",
    "held_objects": [{"object": "lit paper lantern", "is_weapon": false}],
    "clothing": "red oilskin coat over dark trousers, bare head",
    "pose": "standing still at the pier's end, facing away from the viewer, head
     turned slightly right"},
  "other_figures": [{"species": "dog", "role": "companion, midground"}],
  "figure_count": 2,
  "time_of_day": "dusk, low warm light from the left, long soft shadows",
  "palette": ["slate blue", "pale apricot", "signal red", "wet gray"],
  "composition": "wide shot, small figure left of center in the lower third, large
   empty sky, the pier a strong horizontal leading away from the viewer"

Good interpretive propositions:
  "feels quiet and unresolved rather than sad"
  "reads as the last few minutes of someone's patience"
  "the animal's presence emphasizes the solitude rather than relieving it"
  "would sit comfortably on the sleeve of a melancholy synth-pop record"
  "framed like the closing shot of a coming-of-age film"
  "no violence is implied; the whole tension is about arrival and absence"

Bad interpretive propositions, and why:
  "there is a dog behind her"           - a literal fact in the wrong layer
  "the mood is interesting"             - vague; matches no query
  "possibly somewhat melancholy"        - hedged into uselessness

That example is a different picture, shown only to demonstrate the two registers.
Nothing from it — no pier, no dog, no lantern, no fog — belongs in your answer unless
it is genuinely in the image in front of you.

OUTPUT

Return exactly one JSON object and nothing else. No preamble, no commentary, no
markdown, no code fences. Begin with { and end with }.

  literal                     one paragraph, roughly 90 to 160 words, facts only
  literal_propositions        10 to 20 strings
  slots                       every key present, "none" where an attribute is absent
  interpretive                one paragraph, roughly 70 to 130 words, impressions only
  interpretive_propositions   8 to 15 strings

Write in English. Look at the whole image once more before you answer, including the
edges and the background, and then describe it.
"""


# Appended to the prompt for the single retry after a rejected response. Terse on
# purpose: the failure is almost always structural, and restating the whole rule set
# only pushes the image further from the instructions.
VISION_RETRY_NOTE: str = """\

YOUR PREVIOUS ANSWER WAS REJECTED: {reason}

Return one JSON object only, starting with {{ and ending with }}, with every key
present — literal, literal_propositions, slots (primary_subject with species,
facial_hair, held_objects, clothing, pose; other_figures; figure_count; setting;
time_of_day; palette; art_style; composition), interpretive,
interpretive_propositions. Write "none" for absent attributes; never drop a key.
Facts only in the literal layer, impressions only in the interpretive layer.
"""


def vision_prompt(retry_reason: str | None = None) -> str:
    """The vision prompt, optionally with the one-shot corrective note appended."""
    if not retry_reason:
        return VISION_PROMPT
    return VISION_PROMPT + VISION_RETRY_NOTE.format(reason=retry_reason)
