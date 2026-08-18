"""Oracle-corpus result dicts -> Discord message JSON. Stdlib only, like `render.py`.

Same property as `serve/render.py` and for the same reason: **pure functions
returning plain dicts matching Discord's embed JSON, importing nothing but the
standard library**, so `tests/test_oracle_render.py` needs nothing installed but
pytest. `tests/test_render.py`'s docstring calls that "the reason the suite is
worth running", and it is not being lost for a second surface.

This module currently renders `/search` — one named card, or a disambiguation
list. `/oracle`'s ranked mechanical results are rendered here too when they land,
and one decision is worth writing down before they do: **an `/oracle` result
embed carries no image and no thumbnail, ever.** A picture would invite the
reader to judge a *ranked mechanical result* on its art, and the whole feature
exists to separate those two questions. That guarantee used to rest on the oracle
database holding no image URLs at all; `cards.image_normal` exists now for
`/search`, so the guarantee has weakened from *impossible* to *tested* and is
re-declared here rather than left to be discovered.

`/search` renders an image and that is principled rather than inconsistent: it
returns a single card the user named outright, where the image is that card's
primary identifier and there is no ranking for it to bias.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- constants

COLOR_CARD = 0x9B59B6        # one named card: the oracle corpus's own accent
COLOR_AMBIGUOUS = 0x95A5A6   # a list of candidates is not an answer, and looks it

MAX_CONTENT = 2000
MAX_TITLE = 256
MAX_FIELD_VALUE = 1024
MAX_FOOTER = 2048
# Discord allows 4096. The oracle text is the whole point of the embed, so it gets
# nearly all of that rather than `render.MAX_DESCRIPTION`'s one-sentence budget.
MAX_DESCRIPTION = 4000

MAX_CANDIDATES_SHOWN = 10

# Which formats are worth naming, in the order a player thinks about them. The
# corpus stores every format Scryfall publishes; listing all twenty in an embed
# field is noise, and this is the subset with a competitive scene behind it.
LEGALITY_FORMATS = (
    ("commander", "Commander"),
    ("standard", "Standard"),
    ("pioneer", "Pioneer"),
    ("modern", "Modern"),
    ("legacy", "Legacy"),
    ("vintage", "Vintage"),
    ("pauper", "Pauper"),
)

LINK_LABELS = (
    ("scryfall", "Scryfall"),
    ("edhrec", "EDHREC"),
    ("tcgplayer", "TCGplayer"),
)


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# ------------------------------------------------------------------------- disclosure


def resolution_note(layer: str | None, typed: str, name: str, distance: Any = None) -> str:
    """The footer sentence for a resolution that reinterpreted the input.

    L0-L2 matched what the user actually typed and need no comment. L3, L4 and L5
    changed the interpretation, and silently correcting input is how a reader ends
    up looking at the wrong card's text and never noticing.
    """
    if layer in (None, "L0", "L1", "L2"):
        return ""
    tail = ""
    if layer == "L5" and distance is not None:
        tail = f" (edit distance {distance})"
    return f'matched "{name}" from your input "{typed}"{tail}'


# ------------------------------------------------------------------------ card pieces


def oracle_block(card: dict) -> str:
    """The card's full rules text, verbatim, in a code block.

    A code block rather than prose because oracle text is full of `{T}`, `{1}{W}`
    and `+1/+1`, all of which Discord's markdown would otherwise eat. Multi-face
    cards keep Scryfall's own separation: the ingest joins faces with "\\n//\\n"
    and that is what is shown.
    """
    text = str(card.get("oracle_text") or "").strip()
    if not text:
        # Vanilla creatures and most basic lands genuinely have no rules text.
        # Saying so is better than an empty box.
        return "_(no rules text)_"
    return "```\n" + _truncate(text, MAX_DESCRIPTION - 10) + "\n```"


def _mana_field(card: dict) -> dict:
    cost = str(card.get("mana_cost") or "").strip()
    cmc = card.get("cmc")
    try:
        value = f"{float(cmc):g}"
    except (TypeError, ValueError):
        value = "?"
    label = f"{cost} · MV {value}" if cost else f"MV {value}"
    return {"name": "Cost", "value": _truncate(label, MAX_FIELD_VALUE), "inline": True}


def legality_line(legalities: dict | None) -> str:
    """Legal formats, then bans and restrictions named explicitly.

    A banned card is not "not legal" in the same way an un-set card is, and
    collapsing the two would hide the one a deckbuilder needs to know.
    """
    if not isinstance(legalities, dict) or not legalities:
        return "unknown"
    legal = [label for key, label in LEGALITY_FORMATS if legalities.get(key) == "legal"]
    banned = [label for key, label in LEGALITY_FORMATS if legalities.get(key) == "banned"]
    restricted = [
        label for key, label in LEGALITY_FORMATS if legalities.get(key) == "restricted"
    ]
    parts = [", ".join(legal) if legal else "not legal in the major formats"]
    if banned:
        parts.append("Banned in " + ", ".join(banned))
    if restricted:
        parts.append("Restricted in " + ", ".join(restricted))
    return " · ".join(parts)


def price_line(card: dict, refreshed_at: str | None) -> str:
    """`$1.60 · foil $12.00 (as of 2026-08-17)`, never presented as current.

    Prices are a weekly snapshot, so they are up to seven days old, and Scryfall
    updates them only once a day in the first place. Their own footer disclaims
    that "absolutely no guarantee is made for any price information", and this
    embed does not make a stronger claim than its source does.
    """
    bits: list[str] = []
    for key, label in (("price_usd", "${:.2f}"), ("price_usd_foil", "foil ${:.2f}")):
        value = card.get(key)
        try:
            bits.append(label.format(float(value)))
        except (TypeError, ValueError):
            continue
    if not bits:
        return ""
    stamp = str(refreshed_at or "")[:10]
    suffix = f" (as of {stamp})" if stamp else " (weekly snapshot)"
    return " · ".join(bits) + suffix


def link_line(links: dict | None) -> str:
    """`[Scryfall](…) · [EDHREC](…)`. An absent link is omitted, never guessed.

    EDHREC and TCGplayer URLs are Scryfall's own `related_uris` / `purchase_uris`
    values, stored at ingest. Nothing here slugifies a card name into a URL and
    hopes — `cts/links.py`'s standing rule is that a reference which cannot be
    built is omitted rather than emitted broken.
    """
    if not isinstance(links, dict):
        return ""
    return " · ".join(
        f"[{label}]({links[key]})" for key, label in LINK_LABELS if links.get(key)
    )


def card_embed(
    card: dict,
    *,
    layer: str | None = None,
    typed: str = "",
    distance: Any = None,
    refreshed_at: str | None = None,
) -> dict:
    """One named card -> one Discord embed dict."""
    name = str(card.get("name") or "unknown card")

    fields: list[dict] = [
        _mana_field(card),
        {
            "name": "Type",
            "value": _truncate(str(card.get("type_line") or "—"), MAX_FIELD_VALUE),
            "inline": True,
        },
        {
            "name": "Set (one printing)",
            "value": _truncate(
                f"{str(card.get('set_code') or '?').upper()} · "
                f"{str(card.get('rarity') or 'unknown')}",
                MAX_FIELD_VALUE,
            ),
            "inline": True,
        },
        {
            "name": "Legal",
            "value": _truncate(legality_line(card.get("legalities")), MAX_FIELD_VALUE),
            "inline": False,
        },
    ]

    price = price_line(card, refreshed_at)
    if price:
        fields.append({"name": "USD", "value": _truncate(price, MAX_FIELD_VALUE),
                       "inline": True})

    links = link_line(card.get("links"))
    if links:
        fields.append({"name": "Links", "value": _truncate(links, MAX_FIELD_VALUE),
                       "inline": False})

    embed: dict = {
        "title": _truncate(name, MAX_TITLE),
        "color": COLOR_CARD,
        "description": _truncate(oracle_block(card), MAX_DESCRIPTION),
        "fields": fields,
    }

    if card.get("scryfall_uri"):
        embed["url"] = str(card["scryfall_uri"])
    # The image URI is a layout trap handled at ingest: top-level `image_uris` for
    # `normal` and `adventure` cards, per-face for `transform` cards, which have
    # none at the top level at all. `image_normal` is whichever one exists.
    if card.get("image_normal"):
        embed["image"] = {"url": str(card["image_normal"])}

    footer = resolution_note(layer, typed, name, distance)
    if not footer:
        stamp = str(refreshed_at or "")[:10]
        footer = f"oracle corpus, refreshed {stamp}" if stamp else "local oracle corpus"
    embed["footer"] = {"text": _truncate(footer, MAX_FOOTER)}
    return embed


# ---------------------------------------------------------------------- disambiguation


def candidate_lines(candidates: Iterable[dict]) -> list[str]:
    return [
        " · ".join(
            part
            for part in (
                f"**{str(c.get('name') or '?')}**",
                str(c.get("mana_cost") or "").strip(),
                str(c.get("type_line") or "").strip(),
            )
            if part
        )
        for c in candidates
    ]


def candidates_message(typed: str, candidates: Sequence[dict], total: int) -> dict:
    """2-10 hits, or the ten most played of more. **A list, not a card.**

    Deliberately a different render rather than one card with a warning attached:
    showing a card and mentioning the others in small text invites the reader to
    accept the one on screen, which is exactly the mistake this is here to prevent.
    """
    shown = list(candidates)[:MAX_CANDIDATES_SHOWN]
    if total > MAX_CANDIDATES_SHOWN:
        headline = (
            f'{total} cards match "{typed}" — showing the {len(shown)} most played. '
            "Be more specific."
        )
    else:
        headline = f'{total} cards match "{typed}". Re-run with a fuller name.'

    return {
        "content": _truncate(headline, MAX_CONTENT),
        "embeds": [
            {
                "title": _truncate(f'candidates for "{typed}"', MAX_TITLE),
                "color": COLOR_AMBIGUOUS,
                "description": _truncate("\n".join(candidate_lines(shown)), MAX_DESCRIPTION),
            }
        ],
    }


# ---------------------------------------------------------------------------- messages


def card_message(payload: dict) -> dict:
    """`GET /card`'s body -> `{"content": str, "embeds": [dict, …]}`.

    The three shapes the endpoint can return — one card, several candidates, or
    nothing — are three different renders, and the caller does not have to know
    which it is getting.
    """
    typed = str(payload.get("input") or "")
    service = payload.get("service") or {}
    refreshed_at = service.get("refreshed_at")

    if payload.get("resolved") and payload.get("card"):
        return {
            "content": "",
            "embeds": [
                card_embed(
                    payload["card"],
                    layer=payload.get("layer"),
                    typed=typed,
                    distance=payload.get("distance"),
                    refreshed_at=refreshed_at,
                )
            ],
        }

    candidates = payload.get("candidates") or []
    if candidates:
        return candidates_message(typed, candidates, int(payload.get("total") or len(candidates)))

    return {"content": no_card_message(typed), "embeds": []}


def no_card_message(typed: str) -> str:
    """The honest miss. Names what was searched, and does not offer a guess."""
    return _truncate(
        f'No card matches "{typed}" in the local oracle corpus — not by name, '
        "prefix, word order or spelling. Check the spelling, or try /oracle to "
        "search by what a card does.",
        MAX_CONTENT,
    )


def corpus_missing_message() -> str:
    return (
        "The oracle corpus hasn't been built on the host yet. Someone needs to run "
        "`python -m cts oracle-ingest`."
    )


# ===========================================================================
# /oracle — ranked mechanical results
#
# Same rules as the rest of this module: pure functions, plain dicts, stdlib
# only. Two decisions worth restating here rather than leaving implicit:
#
# * **No image, no thumbnail, ever.** A picture would invite the reader to
#   judge a ranked MECHANICAL result on its art, and the whole feature exists
#   to separate those two questions. `oracle_result_embed`'s own test asserts
#   the embed dict never grows an "image" or "thumbnail" key.
# * **The full oracle text goes first, verbatim, above the rationale** — a
#   deliberate inversion of `/scry`'s embed, where the model-written rationale
#   leads. Here the evidence is authoritative (Wizards' own text) and the
#   rationale is the only model-written thing on screen, so the authoritative
#   text goes first and the claim about it goes second. A loot-for-draw error
#   is then checkable in the two seconds it takes to read the card.
# ===========================================================================

COLOR_ORACLE_PASS = 0x2ECC71     # clears the 0.5 fit bar
COLOR_ORACLE_STRETCH = 0x95A5A6  # below it — same grey as /scry's stretch colour

# Deliberately much shorter than /search's MAX_DESCRIPTION: a ranked result is
# one of up to five in a message, not the sole point of the reply, and the
# design doc specifies 700 as the truncation point with a pointer to the full
# text on Scryfall (the embed's own `url`) rather than a 4,000-character wall.
MAX_RESULT_DESCRIPTION = 700

MAX_EMBEDS = 5  # k is capped at 5 by both the API and the Discord command

ORACLE_LEGALITY_FORMATS = ("standard", "pioneer", "modern", "legacy", "vintage", "commander")

# A note is a warning when it says a stage of the pipeline did not run, same
# convention `serve/render.py::_is_warning` already established for /scry.
_WARNING_MARKERS = ("unavailable", "unreachable", "failed", "fell back", "no embedding")


def _is_warning(note: str) -> bool:
    lowered = note.lower()
    return any(marker in lowered for marker in _WARNING_MARKERS)


def _mark_matched_line(oracle_text: str, face_index: Any, ordinal: Any) -> str:
    """Prefix the matched ability's line with "▸ ", located by `ordinal`
    (position among non-blank lines within its face) — never by string
    search, which could mark the wrong occurrence of a repeated line.

    `ordinal` of -1 (the whole-card chunk matched, not one ability) or None
    (no retrieval ran — the structural-only fast path) marks nothing.
    """
    text = str(oracle_text or "")
    try:
        target_face = int(face_index)
        target_ordinal = int(ordinal)
    except (TypeError, ValueError):
        return text
    if target_ordinal < 0:
        return text

    faces = text.split("\n//\n")
    marked_faces = []
    for face_i, face_text in enumerate(faces):
        if face_i != target_face:
            marked_faces.append(face_text)
            continue
        counter = 0
        out_lines = []
        for line in face_text.split("\n"):
            if line.strip():
                prefix = "▸ " if counter == target_ordinal else "  "
                out_lines.append(prefix + line)
                counter += 1
            else:
                out_lines.append(line)
        marked_faces.append("\n".join(out_lines))
    return "\n//\n".join(marked_faces)


def oracle_result_description(result: dict) -> str:
    """Full oracle text, verbatim, marked and truncated. See the module note
    on why full text leads and the rationale trails."""
    text = str(result.get("oracle_text") or "").strip()
    if not text:
        return "_(no rules text)_"
    marked = _mark_matched_line(text, result.get("matched_face_index"), result.get("matched_ordinal"))
    truncated = len(marked) > MAX_RESULT_DESCRIPTION
    body = "```\n" + _truncate(marked, MAX_RESULT_DESCRIPTION - 10) + "\n```"
    if truncated:
        body += "\n_(truncated — the full text is one click away, on Scryfall)_"
    return body


def oracle_legality_line(legalities: dict | None) -> str:
    """Up to six major formats where the card is legal, plus any bans among
    them named explicitly — a banned card is not "not legal" in the same way
    an unreleased one is."""
    if not isinstance(legalities, dict) or not legalities:
        return "unknown"
    legal = [f for f in ORACLE_LEGALITY_FORMATS if legalities.get(f) == "legal"][:6]
    banned = [f for f in ORACLE_LEGALITY_FORMATS if legalities.get(f) == "banned"]
    parts = [", ".join(legal) if legal else "not legal in the major formats"]
    if banned:
        parts.append("Banned in " + ", ".join(banned))
    return " · ".join(parts)


def _oracle_footer(result: dict) -> str:
    set_code = result.get("set_code")
    if not set_code:
        return ""
    released = str(result.get("released_at") or "")
    year = released[:4] if released[:4].isdigit() else ""
    label = f"first printed {str(set_code).upper()}"
    return f"{label} {year}".strip()


def oracle_result_embed(result: dict, position: int) -> dict:
    """One `/oracle` result -> one Discord embed dict."""
    stretch = bool(result.get("stretch"))
    name = str(result.get("name") or "unknown card")
    mana = str(result.get("mana_cost") or "").strip()
    title = f"{name} {mana}".strip() if mana else name
    if stretch:
        title += " · STRETCH"

    cmc = result.get("cmc")
    try:
        cmc_text = f"{float(cmc):g}"
    except (TypeError, ValueError):
        cmc_text = "?"
    colors = str(result.get("color_identity") or "").strip().upper() or "C"

    fit = result.get("fit")
    fit_text = "—" if fit is None else f"{float(fit):.2f}"

    fields: list[dict] = [
        {"name": "Type", "value": _truncate(str(result.get("type_line") or "—"), MAX_FIELD_VALUE),
         "inline": True},
        {"name": "Mana value / Colours", "value": f"{cmc_text} / {colors}", "inline": True},
        {"name": "Fit", "value": fit_text, "inline": True},
        {"name": "Legal", "value": _truncate(oracle_legality_line(result.get("legalities")),
                                              MAX_FIELD_VALUE), "inline": False},
    ]
    if result.get("rationale"):
        fields.append({"name": "Rationale",
                        "value": _truncate(str(result["rationale"]), MAX_FIELD_VALUE),
                        "inline": False})

    embed: dict = {
        "title": _truncate(title, MAX_TITLE),
        "color": COLOR_ORACLE_STRETCH if stretch else COLOR_ORACLE_PASS,
        "description": oracle_result_description(result),
        "fields": fields,
    }
    if result.get("scryfall_uri"):
        embed["url"] = str(result["scryfall_uri"])
    footer = _oracle_footer(result)
    if footer:
        embed["footer"] = {"text": _truncate(footer, MAX_FOOTER)}
    # No "image" key, no "thumbnail" key — ever. See the module note above;
    # tests/test_oracle_render.py asserts this directly rather than trusting
    # that nothing here ever adds one.
    return embed


def oracle_content_line(query: str, outcome: dict) -> str:
    """The text above the embeds: header, echoed filters, the honest counts,
    any notes, and the Scryfall refine link — in that order, every time."""
    plan = outcome.get("plan") or {}
    lines = [f'🔮 "{query}"']
    if plan.get("echo"):
        lines.append(str(plan["echo"]))
    if outcome.get("message"):
        lines.append(str(outcome["message"]))
    for note in plan.get("notes") or []:
        note = str(note)
        lines.append(f"⚠️ {note}" if _is_warning(note) else f"note: {note}")
    if plan.get("scryfall_url"):
        lines.append(f"[refine on Scryfall]({plan['scryfall_url']})")
    return _truncate("\n".join(lines), MAX_CONTENT)


def oracle_message(query: str, outcome: dict) -> dict:
    """`execute()`'s outcome dict -> `{"content": str, "embeds": [dict, ...]}`.

    A guard response (card-name or rules-question) is content-only — there is
    nothing to rank and nothing to embed."""
    if outcome.get("guard"):
        return {"content": _truncate(str(outcome.get("message") or ""), MAX_CONTENT), "embeds": []}

    results = (outcome.get("results") or [])[:MAX_EMBEDS]
    return {
        "content": oracle_content_line(query, outcome),
        "embeds": [oracle_result_embed(r, i) for i, r in enumerate(results, start=1)],
    }


# --------------------------------------------------------------- feedback buttons

# A distinct prefix from /scry's "sp:v1" so serve/render.py's decode_custom_id
# and this one can never claim each other's buttons — carries oracle_id, a
# UUID, rather than illustration_id, but the same length margin against
# Discord's 100-character limit applies.
ORACLE_CUSTOM_ID_PREFIX = "sp:o1"
MAX_ORACLE_CUSTOM_ID = 100


def encode_oracle_custom_id(query_id: int | str, oracle_id: str, accepted: bool) -> str:
    vote = "u" if accepted else "d"
    custom_id = f"{ORACLE_CUSTOM_ID_PREFIX}:{query_id}:{oracle_id}:{vote}"
    if len(custom_id) > MAX_ORACLE_CUSTOM_ID:
        raise ValueError(
            f"custom_id is {len(custom_id)} characters, over Discord's "
            f"{MAX_ORACLE_CUSTOM_ID}: {custom_id!r}"
        )
    return custom_id


def decode_oracle_custom_id(custom_id: str) -> tuple[int, str, bool] | None:
    parts = str(custom_id).split(":")
    if len(parts) != 5:
        return None
    namespace, version, raw_query_id, oracle_id, vote = parts
    if (namespace, version) != tuple(ORACLE_CUSTOM_ID_PREFIX.split(":")):
        return None
    if vote not in ("u", "d") or not oracle_id:
        return None
    try:
        query_id = int(raw_query_id)
    except ValueError:
        return None
    return query_id, oracle_id, vote == "u"


def oracle_button_specs(outcome: dict) -> list[dict]:
    """Ten button descriptors in embed order, mirroring `render.button_specs`."""
    query_id = outcome.get("query_id")
    if query_id is None:
        return []
    results = (outcome.get("results") or [])[:MAX_EMBEDS]
    specs: list[dict] = []
    for accepted, emoji in ((True, "👍"), (False, "👎")):
        for position, result in enumerate(results, start=1):
            oracle_id = result.get("oracle_id")
            if not oracle_id:
                continue
            try:
                custom_id = encode_oracle_custom_id(query_id, str(oracle_id), accepted)
            except ValueError:
                continue
            specs.append(
                {
                    "custom_id": custom_id,
                    "label": str(position),
                    "emoji": emoji,
                    "row": 0 if accepted else 1,
                    "accepted": accepted,
                    "name": str(result.get("name") or "that result"),
                }
            )
    return specs
