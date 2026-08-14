"""One definition of every reference a result carries, shared by the CLI and JSON.

EDHREC links stay keyed on the card (EDHREC does not care which printing you own);
everything else follows the *matched artwork*, because sending someone to a printing
whose art is not the one that matched defeats the point of the whole system.

Keys are omitted entirely when unavailable — never emitted as None, never guessed.
"""

from __future__ import annotations

import json
import re

EDHREC_BASE = "https://edhrec.com/commanders"

# Verified live against json.edhrec.com: EDHREC's own theme slugs reproduce exactly
# from the display name for 1212/1212 tags sampled across 8 commanders, given these
# three rules — drop apostrophes, "+" -> "plus", a "-" in front of a digit -> "minus".
# ("+1/+1 Counters" -> "plus-1-plus-1-counters", "Dragon's Approach" -> "dragons-approach")
_APOSTROPHES = str.maketrans("", "", "'’")


def theme_slug(name: str) -> str:
    """Slugify an EDHREC theme/archetype display name the way EDHREC does."""
    s = name.lower().strip().translate(_APOSTROPHES)
    s = re.sub(r"\+", "plus ", s)
    s = re.sub(r"(?<![a-z0-9])-(?=[0-9])", "minus ", s)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def _records(value) -> list[dict]:
    """Normalize an edhrec themes/archetypes column into [{name, slug, count}].

    The column is written by the data agent and may hold a JSON string or a list,
    of plain names or of EDHREC's own {"value", "slug", "count"} objects. Prefer a
    slug EDHREC gave us; derive one only when it did not.
    """
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return []

    out: list[dict] = []
    for entry in value:
        if isinstance(entry, str):
            name = entry
            slug, count = "", 0
        elif isinstance(entry, dict):
            name = str(entry.get("value") or entry.get("name") or entry.get("theme") or "")
            slug = str(entry.get("slug") or "")
            try:
                count = int(entry.get("count") or entry.get("num_decks") or 0)
            except (TypeError, ValueError):
                count = 0
        else:
            continue
        slug = slug or theme_slug(name)
        if slug:
            out.append({"name": name or slug, "slug": slug, "count": count})
    return out


def strongest_theme(row: dict) -> dict | None:
    """Pick the theme to link: the most-played one the query actually matched.

    `matched_terms` is the router's mechanical filter (deck archetypes, oracle-text
    keywords). When one of them names a theme this commander has, that theme is the
    relevant one; otherwise fall back to the commander's most popular theme.

    Only `archetypes` is used when it is populated: cts/edhrec.py stores the full tag
    list in `themes` but keeps `archetypes` as the subset EDHREC actually publishes a
    /commanders/<slug>/<theme> page for, and a tag without a page is a 404.
    """
    records = _records(row.get("archetypes")) or _records(row.get("themes"))
    if not records:
        return None

    by_slug: dict[str, dict] = {}
    for rec in records:
        by_slug.setdefault(rec["slug"], rec)
    ordered = sorted(by_slug.values(), key=lambda r: -r["count"])

    terms = [str(t).lower().strip() for t in (row.get("matched_terms") or []) if str(t).strip()]
    for term in terms:
        slug = theme_slug(term)
        matched = [
            rec
            for rec in ordered
            if term in rec["name"].lower() or (slug and slug in rec["slug"])
        ]
        if matched:
            return matched[0]
    return ordered[0]


def links_for(row: dict) -> dict:
    """Build every reference for one result row. Absent data means an absent key."""
    links: dict[str, str] = {}

    # Only a slug that already returned 200 is stored, so this link is never broken.
    slug = row.get("edhrec_slug") or row.get("slug")
    if slug:
        links["edhrec"] = f"{EDHREC_BASE}/{slug}"
        theme = strongest_theme(row)
        if theme:
            # Path shape verified live: /commanders/<slug>/<theme-slug> -> 200.
            links["edhrec_theme"] = f"{EDHREC_BASE}/{slug}/{theme['slug']}"

    for key, column in (
        ("scryfall", "scryfall_uri"),
        ("tcgplayer", "tcgplayer_uri"),
        ("art_crop", "art_crop_url"),
    ):
        value = row.get(column)
        if value:
            links[key] = str(value)

    return links
