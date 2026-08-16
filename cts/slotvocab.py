"""Phase 7b: make structured slot filters match the vocabulary that is actually stored.

The vision pass is name-blind by design. It never writes "goblin"; it writes
"green-skinned humanoid with pointed ears". The router, reasoning about a Magic query,
writes "goblin". Compared as SQL strings those two never intersect, so every slot filter
the router emitted was dropped as matching nothing and the whole structured layer
contributed exactly zero.

Three layers close that gap, cheapest first:

1. NORMALIZATION. Case, punctuation, underscores and simple plurals are noise.
   "low_angle_shot" and "low-angle shot" are the same constraint.
2. RELAXED MATCHING. A stored slot is a descriptive noun phrase, not an enum member,
   so `equals` means "this phrase is about that", not "these strings are identical":
   the query's content tokens appearing in the phrase (in any order) is a match.
3. TERM EXPANSION. What is left is a genuine vocabulary difference, and it is closed
   by *mining the corpus itself*: every artwork belongs to a card whose type line
   already names its creature types, so counting which descriptive species phrases
   co-occur with which creature type recovers "goblin -> green-skinned humanoid"
   from the data, with no hand-written table and no second vision pass. Association is
   kept only where it is statistically real (support and lift thresholds), which is
   what makes the layer degrade honestly: "dwarf" is simply not something the vision
   pass recorded, so almost nothing is mined for it, and the filter still drops.

Everything here is derived from data already in the database. Nothing is re-described,
re-embedded or written back; the whole structure costs well under a second to build and
is cached per process.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache

# ------------------------------------------------------------------- normalization

_NON_WORD = re.compile(r"[^a-z0-9]+")

# Suffixes where a trailing "s" is part of the word, not a plural marker.
_NOT_PLURAL = ("ss", "us", "is", "os", "as")


# Matching walks every distinct phrase of a slot — up to ~8,500 for palette — once per
# term, so the same handful of strings is normalized thousands of times per query.
@lru_cache(maxsize=100_000)
def normalize(text: str) -> str:
    """Lowercase, punctuation and underscores to single spaces, trimmed."""
    return _NON_WORD.sub(" ", str(text or "").lower()).strip()


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(_NOT_PLURAL):
        return token[:-1]
    return token


@lru_cache(maxsize=100_000)
def tokens(text: str) -> frozenset[str]:
    """Content tokens, singularized. Order is deliberately thrown away."""
    return frozenset(_singular(t) for t in normalize(text).split() if t)


# --------------------------------------------------------------------------- paths

NUMERIC_PATHS = ("figure_count",)

# Paths whose values name a kind of being, and so share the mined creature-type map.
SPECIES_PATHS = ("primary_subject.species", "other_figures")

# Magic subtypes that name a job, not a kind of being. They tell you nothing about what
# the artwork depicts, so mining them produces noise rather than vocabulary.
CLASS_SUBTYPES = frozenset(
    """advisor archer artificer assassin barbarian bard berserker citizen cleric coward
    detective drone druid employee gamer guest hero jester juggernaut knight mercenary
    minion monk mount ninja noble nomad peasant performer pilot pirate praetor processor
    rebel rigger rogue samurai scientist scout servo shaman soldier spellshaper spy
    surrakar survivor synth tactician time warlock warrior wizard""".split()
)

# Card types and supertypes. A double-faced card's type_line holds both faces separated
# by "//", so splitting on the first dash otherwise leaves the second face's
# "Legendary Creature" in with the subtypes.
CARD_TYPES = frozenset(
    """artifact aura background basic battle class creature enchantment equipment
    instant kindred land legendary ongoing planeswalker saga snow sorcery token tribal
    vehicle world""".split()
)

# Concepts the corpus cannot supply a mapping for, because they are not creature types:
# camera geometry, viewer-relative direction, and one state word the vision prompt never
# uses. Deliberately small — everything species-shaped is mined, not listed.
CURATED_ALIASES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # (concept, expansion terms, paths it applies to; empty = every text path)
    ("undead", ("skeletal", "skeleton", "rotting", "decaying", "decayed", "corpse",
                "mummified", "zombie", "undead"), ()),
    ("seen from behind", ("seen from behind", "from behind", "facing away",
                          "back to the viewer", "back turned", "rear view",
                          "turned away"), ("primary_subject.pose", "composition")),
    ("back turned", ("back turned", "facing away", "back to the viewer",
                     "from behind", "turned away"), ("primary_subject.pose", "composition")),
    ("walking away", ("walking away", "striding away", "facing away",
                      "back to the viewer", "moving away"), ("primary_subject.pose",)),
    ("low angle", ("low angle", "from below", "looking up at", "worm s eye",
                   "upward angle"), ("composition",)),
    ("high angle", ("high angle", "from above", "looking down at", "bird s eye",
                    "overhead", "downward angle"), ("composition",)),
    ("close up", ("close up", "tight crop", "head and shoulders", "headshot"), ("composition",)),
    ("wide shot", ("wide shot", "wide framing", "expansive", "full figure"), ("composition",)),
    ("negative space", ("negative space", "empty space", "empty sky", "vast empty"),
     ("composition",)),
)

# Mining thresholds. A phrase becomes vocabulary for a creature type only when it
# co-occurs at least MIN_SUPPORT times, is that type at least MIN_PRECISION of the time,
# and is at least MIN_LIFT times more likely under that type than overall — which keeps
# "green-skinned humanoid" for goblin (lift 15) and rejects "humanoid" (lift 0.2).
# A pair seen only twice is then held to a higher bar, since two co-occurrences at
# 18% precision is noise ("humanoid creature" is not goblin vocabulary) while two at
# 100% is a real if rare phrasing ("goblin-like humanoid").
MIN_SUPPORT = 2
MIN_PRECISION = 0.10
MIN_LIFT = 6.0
STRONG_SUPPORT = 3
CONFIDENT_PRECISION = 0.5
MAX_ALIASES_PER_TERM = 14


# ------------------------------------------------------------------------ extraction


def _compact(value) -> str:
    """The exact text SQLite's json_extract returns for a list-valued slot."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def slot_phrases(slots: dict) -> dict[str, list[str]]:
    """Every searchable phrase in one artwork's slots, keyed by filter path.

    List slots contribute both their compact JSON (so a filter written against the
    serialized shape still works exactly as it did) and each field value on its own
    (so a filter written against the *meaning* works too).
    """
    out: dict[str, list[str]] = {}

    def add(path: str, value) -> None:
        text = str(value or "").strip()
        if text:
            out.setdefault(path, []).append(text)

    subject = slots.get("primary_subject") or {}
    for key in ("species", "facial_hair", "clothing", "pose"):
        add(f"primary_subject.{key}", subject.get(key))

    held = subject.get("held_objects")
    if isinstance(held, list):
        add("primary_subject.held_objects", _compact(held))
        for entry in held:
            if isinstance(entry, dict):
                add("primary_subject.held_objects", entry.get("object"))

    others = slots.get("other_figures")
    if isinstance(others, list):
        add("other_figures", _compact(others))
        for entry in others:
            if isinstance(entry, dict):
                add("other_figures", entry.get("species"))
                add("other_figures", entry.get("role"))

    palette = slots.get("palette")
    if isinstance(palette, list):
        add("palette", _compact(palette))
        for colour in palette:
            add("palette", colour)
    else:
        add("palette", palette)

    for key in ("setting", "time_of_day", "art_style", "composition"):
        add(key, slots.get(key))

    return out


def _subtypes(type_line: str) -> set[str]:
    """Creature types from a Magic type line: what follows the dash, on every face."""
    found: set[str] = set()
    for face in str(type_line or "").split("//"):
        for dash in ("—", "–", "--"):
            if dash in face:
                face = face.split(dash, 1)[1]
                break
        else:
            continue
        found |= {w.lower() for w in re.findall(r"[A-Za-z']{3,}", face)}
    return found - CLASS_SUBTYPES - CARD_TYPES


# --------------------------------------------------------------------------- matching


def _value_variants(value: str) -> list[str]:
    """The filter's value, plus the plain string inside any JSON scaffolding.

    The router is told held_objects and other_figures serialize as JSON, so it writes
    '"species":"dog"'. That is a legitimate way to ask, and so is "dog"; both are tried.
    """
    variants = [value]
    for _, literal in re.findall(r'"([A-Za-z_]+)"\s*:\s*"([^"]*)"', value):
        if literal.strip():
            variants.append(literal.strip())
    return variants


# Token-subset matching throws word order away, which is right for a short noun phrase
# ("winged humanoid" is "winged humanoid female") and wrong for a whole descriptive
# sentence, where three common words co-occurring somewhere means nothing. Long phrases
# — pose, setting, composition, art_style, and the serialized lists — are matched on
# contiguous text instead.
SUBSET_MAX_PHRASE_TOKENS = 8


def _phrase_match(op: str, term: str, phrase: str) -> bool:
    """Does one stored phrase satisfy one term under `op`?

    Stored slots are descriptive text, not enum members, so `equals` cannot mean string
    identity: "is a human" is answered by "human female" and "is an angel" by "winged
    humanoid". Both ops therefore accept the term appearing whole inside the phrase.
    They differ in one thing only: `contains` also matches inside a word and inside the
    serialized JSON of a list slot, which is exactly what the old SQL LIKE did and what
    filters like '"is_weapon":false' are written against.
    """
    term_tokens = tokens(term)
    if not term_tokens:
        return False
    phrase_tokens = tokens(phrase)

    if op == "equals":
        if normalize(term) == normalize(phrase):
            return True
    elif term.lower() in phrase.lower():
        return True

    # A multi-word term is required to appear contiguously; a single word would
    # otherwise match any longer word it happens to be a prefix of.
    if len(term_tokens) > 1 and normalize(term) in normalize(phrase):
        return True

    return len(phrase_tokens) <= SUBSET_MAX_PHRASE_TOKENS and term_tokens <= phrase_tokens


# ----------------------------------------------------------------------- the vocabulary


@dataclass
class SlotVocabulary:
    """Every distinct stored slot phrase, who has it, and the mined vocabulary map."""

    phrases: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    numbers: dict[str, dict[str, float]] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    all_ids: set[str] = field(default_factory=set)

    # ---- expansion

    def expand(self, value: str, path: str, op: str) -> list[tuple[str, str]]:
        """(match mode, term) pairs: the filter value plus what the corpus says it means.

        The three sources match differently on purpose. The value itself is matched
        under the requested op. A mined phrase is matched only as a whole, because it
        was mined as a whole stored value — expanding "goblin" to "humanoid creature"
        and then letting that match by loose token containment would drag in every
        horned, grotesque and dragon-like humanoid creature in the corpus. A curated
        term is a fragment of prose and is looked for inside the sentence.
        """
        terms: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(mode: str, text: str) -> None:
            key = (mode, normalize(text))
            if key[1] and key not in seen:
                seen.add(key)
                terms.append((mode, text))

        variants = _value_variants(value)
        for variant in variants:
            add(op, variant)

        for variant in variants:
            variant_tokens = tokens(variant)
            if path in SPECIES_PATHS:
                for phrase in self.aliases.get(normalize(variant), ()):
                    add("exact", phrase)
            for concept, expansions, paths in CURATED_ALIASES:
                if paths and path not in paths:
                    continue
                concept_tokens = tokens(concept)
                # The value must name the whole concept, or be a multi-word rephrasing
                # of it. A single stray word ("up", "shot") must not trigger a concept.
                if concept_tokens <= variant_tokens or (
                    len(variant_tokens) > 1 and variant_tokens <= concept_tokens
                ):
                    for text in expansions:
                        add("contains", text)
        return terms

    # ---- matching

    def matching_phrases(self, path: str, op: str, value: str, *, expand: bool = True) -> set[str]:
        stored = self.phrases.get(path)
        if not stored:
            return set()
        if expand:
            terms = self.expand(value, path, op)
        else:
            terms = [(op, v) for v in _value_variants(value)]
        return {
            p
            for p in stored
            if any(
                normalize(t) == normalize(p) if mode == "exact" else _phrase_match(mode, t, p)
                for mode, t in terms
            )
        }

    def _numeric(self, path: str, op: str, value: str) -> set[str]:
        column = self.numbers.get(path) or {}
        try:
            target = float(value)
        except (TypeError, ValueError):
            return set()
        if op == "gte":
            keep = lambda n: n >= target       # noqa: E731
        elif op == "lte":
            keep = lambda n: n <= target       # noqa: E731
        elif op == "equals":
            keep = lambda n: n == target       # noqa: E731
        elif op == "contains":
            keep = lambda n: n == target       # noqa: E731
        else:  # not_contains
            keep = lambda n: n != target       # noqa: E731
        return {ill for ill, n in column.items() if keep(n)}

    def match(self, flt: dict) -> set[str]:
        """Illustration ids satisfying one slot filter."""
        path, op, value = flt["path"], flt["op"], flt["value"]
        if path in NUMERIC_PATHS:
            return self._numeric(path, op, value)

        if op in ("gte", "lte"):
            # Ordering is only defined on the numeric slot; treat it as a containment
            # ask rather than silently matching nothing.
            op = "contains"

        if op == "not_contains":
            # Negation is never expanded. An over-broad synonym set on a positive
            # filter costs precision; on a negation it deletes correct answers.
            hit: set[str] = set()
            for phrase in self.matching_phrases(path, "contains", value, expand=False):
                hit |= self.phrases[path][phrase]
            return self.all_ids - hit

        matched: set[str] = set()
        for phrase in self.matching_phrases(path, op, value):
            matched |= self.phrases[path][phrase]
        return matched


# ------------------------------------------------------------------------- construction


def _mine_aliases(species_by_ill: dict[str, list[str]], types_by_ill: dict[str, set[str]]) -> dict[str, list[str]]:
    """Creature type -> descriptive species phrases, learned from the corpus.

    For every (type, phrase) pair this measures support, precision P(type | phrase) and
    lift, and keeps only pairs that clear all three thresholds. Nothing is hardcoded:
    if the vision pass never recorded a distinguishing phrase for a type — as with
    "dwarf" — nothing survives, and a filter on that type correctly still matches
    nothing rather than matching a guess.
    """
    total = len(species_by_ill) or 1
    phrase_count: dict[str, int] = {}
    type_count: dict[str, int] = {}
    pair_count: dict[tuple[str, str], int] = {}

    for ill, phrases in species_by_ill.items():
        seen_phrases = {normalize(p) for p in phrases if normalize(p)}
        seen_types = types_by_ill.get(ill) or set()
        for phrase in seen_phrases:
            phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
        for kind in seen_types:
            type_count[kind] = type_count.get(kind, 0) + 1
            for phrase in seen_phrases:
                pair_count[(kind, phrase)] = pair_count.get((kind, phrase), 0) + 1

    scored: dict[str, list[tuple[int, str]]] = {}
    for (kind, phrase), support in pair_count.items():
        if support < MIN_SUPPORT:
            continue
        precision = support / phrase_count[phrase]
        prior = type_count[kind] / total
        if precision < MIN_PRECISION or prior <= 0:
            continue
        if precision / prior < MIN_LIFT:
            continue
        if support < STRONG_SUPPORT and precision < CONFIDENT_PRECISION:
            continue
        scored.setdefault(kind, []).append((support, phrase))

    return {
        kind: [phrase for _, phrase in sorted(hits, key=lambda h: (-h[0], h[1]))][:MAX_ALIASES_PER_TERM]
        for kind, hits in scored.items()
    }


def build(conn: sqlite3.Connection) -> SlotVocabulary:
    """Read every stored slots blob and derive the whole structure. Read-only."""
    vocab = SlotVocabulary()
    species_by_ill: dict[str, list[str]] = {}

    for row in conn.execute("SELECT illustration_id, slots FROM descriptions"):
        ill = row["illustration_id"]
        try:
            slots = json.loads(row["slots"] or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(slots, dict):
            continue
        vocab.all_ids.add(ill)

        for path, phrases in slot_phrases(slots).items():
            bucket = vocab.phrases.setdefault(path, {})
            for phrase in phrases:
                bucket.setdefault(phrase, set()).add(ill)

        count = slots.get("figure_count")
        try:
            vocab.numbers.setdefault("figure_count", {})[ill] = float(count)
        except (TypeError, ValueError):
            pass

        subject = slots.get("primary_subject") or {}
        if str(subject.get("species") or "").strip():
            species_by_ill[ill] = [str(subject["species"]).strip()]

    types_by_ill: dict[str, set[str]] = {}
    for row in conn.execute(
        "SELECT a.illustration_id AS ill, c.type_line AS type_line "
        "FROM arts a JOIN cards c ON c.oracle_id = a.oracle_id"
    ):
        kinds = _subtypes(row["type_line"])
        if kinds:
            types_by_ill[row["ill"]] = kinds

    vocab.aliases = _mine_aliases(species_by_ill, types_by_ill)
    return vocab


ROUTER_HINT_SPECIES = 16
ROUTER_HINT_TYPES = 60


def router_hint(vocab: SlotVocabulary) -> str:
    """What the router needs to know about the vocabulary that actually exists.

    Two facts it cannot guess: the species slot holds descriptive phrases rather than
    Magic creature types, and the mined map covers some creature types and not others.
    Naming the ones that resolve is what lets the router stop guessing at the rest.
    """
    stored = vocab.phrases.get("primary_subject.species") or {}
    common = sorted(stored.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    examples = [p for p, _ in common if normalize(p) != "none"][:ROUTER_HINT_SPECIES]

    resolvable = sorted(vocab.aliases)[:ROUTER_HINT_TYPES]
    lines = []
    if examples:
        lines.append(
            "primary_subject.species holds phrases like: " + "; ".join(examples) + "."
        )
    if resolvable:
        lines.append(
            "Magic creature type names are accepted for primary_subject.species and "
            "other_figures — the engine maps them onto those phrases. These resolve: "
            + ", ".join(resolvable)
            + ". A creature type NOT in that list was never recorded distinguishably; "
            "do not filter on it, leave it to the query text."
        )
    lines.append(
        "pose, setting, composition and art_style are whole descriptive sentences: "
        "filter them with contains and one or two plain words, never with equals."
    )
    return "\n".join(lines)


_CACHE: dict[tuple[str, int], SlotVocabulary] = {}


def load(conn: sqlite3.Connection) -> SlotVocabulary:
    """Cached build, keyed on the database file and its description count."""
    path = ""
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main":
            path = row[2] or ""
            break
    count = int(conn.execute("SELECT COUNT(*) FROM descriptions").fetchone()[0])
    key = (path, count)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = build(conn)
    return _CACHE[key]
