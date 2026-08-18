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
