"""Phase 2: EDHREC enrichment.

Endpoint and JSON layout, verified live 2026-08-14 against four commander pages
(atraxa-praetors-voice, krenko-mob-boss, kroxa-titan-of-deaths-hunger,
ojer-axonil-deepest-might — all HTTP 200):

    https://json.edhrec.com/pages/commanders/<slug>.json

    container.json_dict.card.num_decks   -> num_decks      (Atraxa: 43585)
    container.json_dict.card.prices      -> avg_price      (per-vendor, see below)
    tag_counts                           -> themes         [{count, slug, value}, ...]
    panels.taglinks                      -> archetypes     same shape, linkable subset

`tag_counts` and `panels.taglinks` are near-identical, and the difference is the
useful part: on all four pages taglinks is exactly tag_counts minus the "cedh"
entry, because every taglink resolves as a theme page at
/commanders/<slug>/<tag-slug> (verified: .../atraxa-praetors-voice/infect.json
-> 200) while cEDH lives elsewhere on the site. So `themes` keeps the complete
tag list and `archetypes` keeps the subset that is safe to build a theme link
from.

cEDH signal — it exists, in two places:

  * `tag_counts` contains {"count": 57, "slug": "cedh", "value": "cEDH"} — the
    number of decks for this commander tagged cEDH. Present on all four probes
    (57 / 103 / 20 / 32). This is what power.py reads, since it rides along in
    the `themes` column and needs no extra parsing.
  * `bracket_counts` maps the official Commander bracket 1-5 to deck counts, and
    bracket 5 *is* cEDH (Atraxa: {"1":143,"2":3334,"3":3617,"4":2335,"5":154}).
    Left in the `raw` blob for later use at `$.bracket_counts`.

No average *deck* price exists anywhere in the payload — a full key scan of all
four pages plus /pages/average-decks/<slug>.json found only
`container.json_dict.card.prices`, the commander card's own market price per
vendor. See `_avg_price` for what goes into the column instead.

Unknown slugs return HTTP 403 (an S3 AccessDenied XML body), not 404. Both count
as a definitive miss.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from . import db
from .config import Config
from .ingest import USER_AGENT, data_dir

BASE_URL = "https://json.edhrec.com/pages/commanders/{slug}.json"

# SPEC.md: one request per second, sleep after every request including misses.
REQUEST_DELAY = 1.0

# A slug that does not exist is served straight off the CDN as 403; 404 and 410
# are handled too in case that ever changes. Anything else is transient.
MISS_STATUSES = frozenset({403, 404, 410})

_TIMEOUT = (15, 60)

# Ligatures and strokes that NFKD leaves alone but EDHREC spells out.
_TRANSLITERATE = {
    "æ": "ae", "œ": "oe", "ø": "o", "ß": "ss",
    "đ": "d", "ł": "l", "þ": "th", "ð": "d",
}

# Deleted outright rather than turned into a separator, so "Death's Hunger"
# becomes "deaths-hunger" and not "death-s-hunger".
_DROPPED = re.compile(r"[’'`´\"“”.,!?:;()\[\]]")
_SEPARATOR = re.compile(r"[^a-z0-9]+")

# USD vendors only, in preference order. cardmarket is EUR, cardhoarder is MTGO
# tickets and face2face/cardtrader are not USD either; mixing currencies would
# corrupt the price percentile power.py computes over the whole corpus.
_USD_VENDORS = ("tcgplayer", "cardkingdom", "manapool", "scg", "mtgstocks")


class _Transient(Exception):
    """A network problem, as opposed to "EDHREC has no page for this card"."""


# --------------------------------------------------------------------------
# slugs
# --------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Card name -> EDHREC slug.

    Lowercase, fold diacritics, delete punctuation, collapse everything else to
    single hyphens. Verified against live pages:

        Atraxa, Praetors' Voice        -> atraxa-praetors-voice
        Krenko, Mob Boss               -> krenko-mob-boss
        Kroxa, Titan of Death's Hunger -> kroxa-titan-of-deaths-hunger
        Ojer Axonil, Deepest Might     -> ojer-axonil-deepest-might
    """
    text = name.strip().lower()
    text = "".join(_TRANSLITERATE.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _DROPPED.sub("", text)
    text = _SEPARATOR.sub("-", text)
    return text.strip("-")


def slug_candidates(name: str) -> list[str]:
    """Slugs to try, in order. First one that returns 200 is the one persisted.

    Double-faced names arrive as "Front // Back"; EDHREC keys those pages on the
    front face alone (verified: "Ojer Axonil, Deepest Might // Temple of Power"
    -> ojer-axonil-deepest-might -> 200), so that goes first and the whole-name
    slug is only a fallback.
    """
    out: list[str] = []
    for candidate in (name.split(" // ")[0], name):
        slug = slugify(candidate)
        if slug and slug not in out:
            out.append(slug)
    return out


# --------------------------------------------------------------------------
# fetch and cache
# --------------------------------------------------------------------------


def _valid(payload: Any) -> bool:
    """A real commander page, not an error document served with a 200."""
    if not isinstance(payload, dict):
        return False
    card = (payload.get("container") or {}).get("json_dict", {})
    return bool(isinstance(card, dict) and card.get("card"))


def _write_cache(path: Path, body: bytes) -> None:
    """Cache the raw response before parsing, atomically. Temp file plus
    os.replace so a mid-run failure never leaves a half-written cache and never
    clobbers good data from a previous run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(body)
    os.replace(tmp, path)


def _resolve(name: str, cache_dir: Path, refresh_all: bool) -> tuple[str | None, dict | None]:
    """(slug, payload) for the first candidate that resolves, (None, None) on a
    definitive miss. Raises _Transient when the network misbehaves, so the caller
    can leave the card untouched and pick it up on the next run."""
    for slug in slug_candidates(name):
        cache = cache_dir / f"{slug}.json"

        if not refresh_all and cache.is_file() and cache.stat().st_size > 0:
            try:
                payload = json.loads(cache.read_text("utf-8"))
            except (OSError, ValueError):
                payload = None
            if _valid(payload):
                return slug, payload

        try:
            resp = requests.get(
                BASE_URL.format(slug=slug),
                timeout=_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException as exc:
            raise _Transient(str(exc)) from exc
        finally:
            time.sleep(REQUEST_DELAY)

        if resp.status_code == 200:
            try:
                payload = json.loads(resp.content.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                payload = None
            if _valid(payload):
                _write_cache(cache, resp.content)
                return slug, payload
            continue
        if resp.status_code in MISS_STATUSES:
            continue
        raise _Transient(f"HTTP {resp.status_code} for {slug}")

    return None, None


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------


def _avg_price(card: dict) -> float | None:
    """The commander's own market price in USD.

    SPEC.md asks for average *deck* price. EDHREC's JSON does not carry one:
    `container.json_dict.card.prices` is the card's price per vendor, cardviews
    carry no price at all, and `budget_counts` is only a bucket histogram with no
    dollar amounts. The card's own price is the closest available proxy and
    preserves the intent — a price percentile as a power signal, since expensive
    commanders skew toward strong staples.
    """
    prices = card.get("prices") or {}
    for vendor in _USD_VENDORS:
        value = (prices.get(vendor) or {}).get("price")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def parse(payload: dict) -> dict:
    """Typed columns out of one commander page. See the module docstring for
    where each field lives in the JSON."""
    card = (payload.get("container") or {}).get("json_dict", {}).get("card") or {}
    panels = payload.get("panels") or {}

    themes = [t for t in (payload.get("tag_counts") or []) if isinstance(t, dict)]
    archetypes = [t for t in (panels.get("taglinks") or []) if isinstance(t, dict)]
    if not archetypes:
        archetypes = themes

    num_decks = card.get("num_decks")
    return {
        "themes": themes,
        "archetypes": archetypes,
        "num_decks": int(num_decks) if isinstance(num_decks, (int, float)) else None,
        "avg_price": _avg_price(card),
    }


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------

_UPSERT = """
INSERT INTO edhrec (oracle_id, slug, themes, archetypes, num_decks, avg_price, raw, fetched_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(oracle_id) DO UPDATE SET
  slug       = excluded.slug,
  themes     = excluded.themes,
  archetypes = excluded.archetypes,
  num_decks  = excluded.num_decks,
  avg_price  = excluded.avg_price,
  raw        = excluded.raw,
  fetched_at = excluded.fetched_at
"""

_PENDING = """
SELECT c.oracle_id AS oracle_id, c.name AS name
FROM cards c LEFT JOIN edhrec e ON e.oracle_id = c.oracle_id
WHERE e.oracle_id IS NULL
ORDER BY c.name
"""

_ALL = "SELECT oracle_id, name FROM cards ORDER BY name"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cfg: Config, refresh_all: bool = False) -> dict:
    """Phase 2. Default: only cards with no edhrec row. refresh_all: every card,
    re-fetched, cache overwritten on success only."""
    conn = db.connect(cfg)
    try:
        cache_dir = data_dir(cfg) / "edhrec"
        cache_dir.mkdir(parents=True, exist_ok=True)

        rows = conn.execute(_ALL if refresh_all else _PENDING).fetchall()
        total = len(rows)
        if not total:
            print("edhrec: nothing to fetch", flush=True)
            return {"updated": 0, "misses": 0}

        mode = "refreshing all" if refresh_all else "filling gaps"
        print(f"edhrec: {mode}, {total:,} commanders (~1 req/sec)", flush=True)

        updated = 0
        misses = 0
        skipped = 0

        for index, row in enumerate(rows, start=1):
            name = row["name"]
            try:
                slug, payload = _resolve(name, cache_dir, refresh_all)
            except _Transient as exc:
                # No row written, so the next default run retries this card
                # rather than remembering a network blip as a permanent miss.
                skipped += 1
                print(f"[{index}/{total}] {name} -> SKIPPED ({exc})", flush=True)
                continue

            if payload is None:
                conn.execute(
                    _UPSERT,
                    (row["oracle_id"], None, None, None, None, None, None, _now()),
                )
                conn.commit()
                misses += 1
                print(f"[{index}/{total}] {name} -> MISS", flush=True)
                continue

            fields = parse(payload)
            conn.execute(
                _UPSERT,
                (
                    row["oracle_id"],
                    slug,
                    json.dumps(fields["themes"]),
                    json.dumps(fields["archetypes"]),
                    fields["num_decks"],
                    fields["avg_price"],
                    json.dumps(payload),
                    _now(),
                ),
            )
            conn.commit()
            updated += 1
            decks = fields["num_decks"]
            print(
                f"[{index}/{total}] {name} -> {slug} "
                f"({decks if decks is not None else '?'} decks, "
                f"{len(fields['themes'])} themes)",
                flush=True,
            )

        print(f"edhrec: {updated} updated, {misses} misses, {skipped} skipped", flush=True)
        return {"updated": updated, "misses": misses}
    finally:
        conn.close()
