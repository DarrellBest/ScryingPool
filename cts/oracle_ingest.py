"""Scryfall's `oracle_cards` bulk file -> `cards`, `card_faces`, `card_types`,
`card_legalities` in the oracle corpus.

A different bulk file, deliberately
-----------------------------------
Scryfall publishes an `oracle_cards` type: **one card object per Oracle ID**.
That is exactly the grain this corpus searches, so nothing here has to regroup
printings the way `cts/ingest.py::parse_bulk` does — that machinery exists on the
art side only because *artwork* is per-printing, and reusing it here would be
pure overhead with a bug surface. It is also a third of the size (24MB against
77MB) and it carries its own `updated_at`, so each corpus skips its own work when
its own bulk file has not moved.

Everything about *how* the download works is reused verbatim from
`cts/ingest.py`: the `USER_AGENT` CONTRACT.md requires, `_download`'s chunked
streaming through a `.part` file so an interrupt never leaves a truncated file
that later looks complete, `iter_card_objects`'s three-format reader, `data_dir`,
and the `(15, 300)` timeouts. The only change made over there was giving
`bulk_entry()` and `_pick_source()` a `bulk_type` parameter that defaults to the
existing constant.

Filtering to paper
------------------
`"paper" in card["games"]`, plus an exclusion list for objects that are printed
on cardboard but have no rules text worth searching: tokens, emblems, art series
cards and oversized memorabilia. **The expected result is near 32,726, not
exactly it** — Scryfall's web search applies its own `-is:extra` semantics which
this approximates rather than reproduces. `checkpoint()` prints the number it
actually got rather than asserting an expected one; if it drifts from 32,726 by
more than a percent or so, the exclusion list is wrong and the printed number is
how anyone finds out.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import oracle_db
from .config import Config
from .ingest import (
    WUBRG,
    _download,
    _pick_source,
    bulk_entry,
    data_dir,
    iter_card_objects,
)
from .oracle_names import fold

BULK_TYPE = "oracle_cards"

# Printed on cardboard, but not cards anyone searches by rules text. Tokens and
# emblems have no oracle text of the kind this corpus is for; art series cards
# are artwork with a blank back; memorabilia is oversized promos and gold-bordered
# reprints that are not legal in any format.
EXCLUDED_LAYOUTS = frozenset({"token", "double_faced_token", "emblem", "art_series"})
EXCLUDED_SET_TYPES = frozenset({"token", "memorabilia"})

# Platforms on which a card can exist without ever having been printed. A card
# whose representative printing runs ONLY on these is digital-only and is
# dropped; anything with an `mtgo` or `paper` printing is kept. See
# `is_paper_card` for why this is not simply `"paper" in games`.
DIGITAL_ONLY_GAMES = frozenset({"arena", "astral", "sega"})

# Magic's supertypes. Everything else left of the em dash is a card type —
# including the un-sets' inventions, which are kept rather than dropped because
# silver-bordered cards are paper Magic and the design deliberately does not
# decide what counts as Magic on the user's behalf.
SUPERTYPES = frozenset({"basic", "legendary", "snow", "world", "ongoing", "host", "elite"})

# Scryfall's own separator between the types and the subtypes.
EM_DASH = "—"


# --------------------------------------------------------------------------- fields


def oracle_id_of(card: dict) -> str | None:
    """Top-level oracle_id, or the first face's (reversible_card layout)."""
    oracle_id = card.get("oracle_id")
    if oracle_id:
        return oracle_id
    for face in card.get("card_faces") or []:
        if face.get("oracle_id"):
            return face["oracle_id"]
    return None


def is_paper_card(card: dict) -> bool:
    """Paper Magic, and not a token/emblem/art-series object.

    **`games` describes the representative printing, not the card, and the
    difference costs 1,062 real cards.** The design document specified
    `"paper" in card["games"]`. Run against the live file that keeps 32,801 rows —
    a number close enough to the document's expected ~32,726 to look correct —
    and it silently drops **Taiga, Timetwister, Library of Alexandria, Strip Mine,
    Palinchron** and about a thousand others. `oracle_cards` carries exactly one
    printing per Oracle ID, and for those cards Scryfall's representative is an
    MTGO-only Masters Edition reprint whose `games` is `["mtgo"]`. The headline
    count looked right only because ~1,060 dropped paper cards were offset by
    ~1,140 un-set and funny cards that Scryfall's own `-is:extra` excludes.

    So the test is inverted: a card is dropped only when its printing runs
    *exclusively* on platforms where cards exist without being printed —
    Arena (Alchemy rebalances and Arena-only originals), and the Astral/Sega
    curiosities. `mtgo` keeps a card, because MTGO's exclusive sets are reprint
    sets for paper cards.

    Cross-checked against the paper printings in the `default_cards` bulk file:
    this keeps 33,933 cards, misses **7** genuinely-paper ones (cards whose only
    representative printing Scryfall chose is Arena-only, e.g. Raging Goblin) and
    admits 76 non-paper ones (74 MTGO Vanguard avatars, which have real rules
    text and a paper Vanguard product behind them). That is 99.98% recall on the
    paper corpus against 96.9% for the specified filter.
    """
    games = set(card.get("games") or ())
    if not games:
        return False
    if "paper" not in games and not (games - DIGITAL_ONLY_GAMES):
        return False
    if (card.get("layout") or "") in EXCLUDED_LAYOUTS:
        return False
    if (card.get("set_type") or "") in EXCLUDED_SET_TYPES:
        return False
    return True


def color_string(colors: Any) -> str:
    """A colour list as a canonical WUBRG-ordered string, "" for colourless."""
    have = set(colors or ())
    return "".join(c for c in WUBRG if c in have)


def numeric(value: Any) -> float | None:
    """`power`/`toughness`/`loyalty` as a number, or None when it is not one.

    These are TEXT in Scryfall's data because they are frequently `*`, `X`,
    `1+*`, `*/*` or `?`. Storing them as numbers loses those cards; comparing
    them as strings gives nonsense (`"10" < "2"`). So both are stored, and the
    numeric column is NULL for anything that is not a plain integer — a `*/*`
    creature's power is a characteristic-defining ability evaluated against a
    board state this corpus does not have, so the honest answer to "is it ≥ 5" is
    that we cannot say.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(int(text))
    except ValueError:
        return None


def image_normal(card: dict) -> str | None:
    """Where `image_uris` lives depends on `layout`, and this is the trap.

    | layout                       | top-level | per-face |
    | normal    (Sol Ring)         | present   | n/a      |
    | adventure (Brazen Borrower)  | present   | absent   |
    | transform (Delver of Secrets)| ABSENT    | present  |

    So: take top-level `image_uris.normal` if present, else the first face's.
    Getting this backwards blanks the image on a whole card class, which is why
    all three layouts are in the tests.
    """
    top = (card.get("image_uris") or {}).get("normal")
    if top:
        return top
    for face in card.get("card_faces") or []:
        face_image = (face.get("image_uris") or {}).get("normal")
        if face_image:
            return face_image
    return None


def parse_types(type_line: str) -> list[tuple[str, str]]:
    """`"Legendary Creature — Angel"` -> `[(supertype, legendary), (type, creature),
    (subtype, angel)]`.

    Both halves of a `//` type line contribute, so an adventure's instant half and
    a transform card's back face are both filterable. Subtypes are split on
    whitespace, which means the handful of two-word subtypes ("Time Lord") land as
    two values; that is a known, small imprecision and the alternative is shipping
    a hand-maintained copy of Scryfall's subtype catalogue.
    """
    found: list[tuple[str, str]] = []
    for part in str(type_line or "").split("//"):
        left, _, right = part.partition(EM_DASH)
        for word in left.split():
            value = word.lower().strip()
            if not value:
                continue
            found.append(("supertype" if value in SUPERTYPES else "type", value))
        for word in right.split():
            value = word.lower().strip()
            if value:
                found.append(("subtype", value))
    return list(dict.fromkeys(found))


def card_row(card: dict, oracle_id: str) -> tuple:
    """One `cards` row. Double-faced cards need every fallback in here.

    `oracle_text` is empty at the top level on transform/MDFC cards, so the faces
    are joined with `"\\n//\\n"` — the same join `cts/ingest.py::card_row` already
    uses, so both corpora spell a two-faced card's text the same way. `mana_cost`,
    `colors`, `power` and friends are likewise absent at the top level there and
    come from the front face.
    """
    faces = card.get("card_faces") or []
    front = faces[0] if faces else {}

    name = card.get("name") or ""

    oracle_text = card.get("oracle_text")
    if not oracle_text and faces:
        oracle_text = "\n//\n".join(f.get("oracle_text") or "" for f in faces).strip()

    mana_cost = card.get("mana_cost")
    if not mana_cost and faces:
        mana_cost = front.get("mana_cost") or None

    colors = card.get("colors")
    if colors is None and faces:
        merged: set[str] = set()
        for face in faces:
            merged.update(face.get("colors") or ())
        colors = sorted(merged)

    power = card.get("power")
    toughness = card.get("toughness")
    loyalty = card.get("loyalty")
    if power is None and toughness is None and faces:
        power = front.get("power")
        toughness = front.get("toughness")
    if loyalty is None and faces:
        loyalty = front.get("loyalty")

    prices = card.get("prices") or {}

    def price(key: str) -> float | None:
        try:
            return float(prices.get(key))
        except (TypeError, ValueError):
            return None

    return (
        oracle_id,
        name,
        fold(name),
        card.get("type_line") or "",
        oracle_text or "",
        mana_cost,
        float(card.get("cmc")) if card.get("cmc") is not None else 0.0,
        color_string(colors),
        color_string(card.get("color_identity")),
        power,
        toughness,
        loyalty,
        numeric(power),
        numeric(toughness),
        numeric(loyalty),
        json.dumps(card.get("keywords") or []),
        card.get("layout") or "",
        1 if card.get("reserved") else 0,
        card.get("edhrec_rank"),
        card.get("released_at") or "",
        card.get("set") or "",
        card.get("rarity") or "",
        image_normal(card),
        price("usd"),
        price("usd_foil"),
        card.get("scryfall_uri"),
        # related_uris / purchase_uris are absent on plenty of objects. An absent
        # key means an absent link, never a guessed one.
        (card.get("related_uris") or {}).get("edhrec"),
        (card.get("purchase_uris") or {}).get("tcgplayer"),
    )


def face_rows(card: dict, oracle_id: str) -> list[tuple]:
    """One row per face, carrying the per-face name and the per-face image.

    The per-face name is the L2 resolution key — it is why `Petty Theft` and
    `Insectile Aberration` resolve at all — and the per-face image is the only
    image a transform card has.
    """
    rows: list[tuple] = []
    for index, face in enumerate(card.get("card_faces") or []):
        face_name = face.get("name") or ""
        rows.append(
            (
                oracle_id,
                index,
                face_name,
                fold(face_name),
                face.get("mana_cost"),
                face.get("type_line") or "",
                face.get("oracle_text") or "",
                (face.get("image_uris") or {}).get("normal"),
            )
        )
    return rows


# ----------------------------------------------------------------------------- write

_INSERT_CARD = """
INSERT OR REPLACE INTO cards
  (oracle_id, name, name_norm, type_line, oracle_text, mana_cost, cmc, colors,
   color_identity, power, toughness, loyalty, power_num, toughness_num, loyalty_num,
   keywords, layout, reserved, edhrec_rank, released_at, set_code, rarity,
   image_normal, price_usd, price_usd_foil, scryfall_uri, related_edhrec,
   purchase_tcgplayer)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_FACE = """
INSERT INTO card_faces
  (oracle_id, face_index, name, name_norm, mana_cost, type_line, oracle_text, image_normal)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_TYPE = "INSERT INTO card_types(oracle_id, kind, value) VALUES (?, ?, ?)"
_INSERT_LEGALITY = "INSERT INTO card_legalities(oracle_id, format, status) VALUES (?, ?, ?)"


def parse_bulk(path: Path) -> list[dict]:
    """Every paper card object in the bulk file, filtered and left whole."""
    kept: list[dict] = []
    scanned = 0
    skipped_no_oracle = 0

    for card in iter_card_objects(path):
        scanned += 1
        if scanned % 25000 == 0:
            print(f"  scanned {scanned:,} card objects, kept {len(kept):,} paper cards",
                  flush=True)
        if not is_paper_card(card):
            continue
        if not oracle_id_of(card):
            skipped_no_oracle += 1
            continue
        kept.append(card)

    print(f"  scanned {scanned:,} card objects, kept {len(kept):,} paper cards", flush=True)
    if skipped_no_oracle:
        print(f"  skipped {skipped_no_oracle} objects with no oracle_id", flush=True)
    return kept


def write(conn: sqlite3.Connection, cards: Iterable[dict]) -> dict:
    """Replace the corpus with `cards`, in one transaction.

    The child tables are rewritten wholesale rather than diffed: they are ~1M
    small rows that rebuild in seconds, and "delete everything then insert
    everything, atomically" is a much shorter thing to be confident about than a
    per-row merge. `cards` itself is INSERT OR REPLACE because prices, ranks and
    Oracle text all legitimately move week to week.

    Returns the counts, plus the oracle_ids whose `oracle_text` actually changed —
    which is precisely the set a re-chunking stage has to redo.
    """
    previous_text = {
        row[0]: row[1] for row in conn.execute("SELECT oracle_id, oracle_text FROM cards")
    }

    card_rows: list[tuple] = []
    faces: list[tuple] = []
    types: list[tuple] = []
    legalities: list[tuple] = []
    changed_text: list[str] = []
    new_cards: list[str] = []
    seen: set[str] = set()

    for card in cards:
        oracle_id = oracle_id_of(card)
        if not oracle_id or oracle_id in seen:
            continue          # oracle_cards is one row per oracle_id; belt and braces
        seen.add(oracle_id)

        row = card_row(card, oracle_id)
        card_rows.append(row)
        faces.extend(face_rows(card, oracle_id))
        types.extend((oracle_id, kind, value) for kind, value in parse_types(row[3]))
        legalities.extend(
            (oracle_id, fmt, status)
            for fmt, status in sorted((card.get("legalities") or {}).items())
        )

        if oracle_id not in previous_text:
            new_cards.append(row[1])
        elif (previous_text[oracle_id] or "") != (row[4] or ""):
            changed_text.append(oracle_id)

    removed = sorted(set(previous_text) - seen)

    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM card_faces")
        conn.execute("DELETE FROM card_types")
        conn.execute("DELETE FROM card_legalities")
        conn.executemany(_INSERT_CARD, card_rows)
        conn.executemany(_INSERT_FACE, faces)
        conn.executemany(_INSERT_TYPE, types)
        conn.executemany(_INSERT_LEGALITY, legalities)
        if removed:
            # A card that left the corpus must not leave chunks behind pointing at
            # a row that no longer exists — an orphan chunk would be retrievable
            # and unrenderable, which is the worst of both.
            marks = ",".join("?" * len(removed))
            conn.execute(
                f"DELETE FROM chunk_embeddings WHERE chunk_id IN "
                f"(SELECT id FROM chunks WHERE oracle_id IN ({marks}))",
                removed,
            )
            conn.execute(f"DELETE FROM chunks WHERE oracle_id IN ({marks})", removed)
            conn.execute(f"DELETE FROM cards WHERE oracle_id IN ({marks})", removed)
    except Exception:
        conn.rollback()
        raise
    conn.commit()

    return {
        "cards": len(card_rows),
        "faces": len(faces),
        "types": len(types),
        "legalities": len(legalities),
        "new_cards": new_cards,
        "changed_text": changed_text,
        "removed": removed,
    }


# ------------------------------------------------------------------------ checkpoint


def checkpoint(conn: sqlite3.Connection) -> dict:
    """Print what actually landed. Reports, never asserts.

    Includes any `name_norm` collision it finds, which is the reason that column
    has an index and not a UNIQUE constraint: a duplicate should surface as a line
    of output rather than as a failed weekly ingest.
    """
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("cards", "card_faces", "card_types", "card_legalities")
    }
    print(f"cards:           {counts['cards']:,}   "
          "(the design doc expects NEAR 32,726, not exactly it)")
    print(f"card_faces:      {counts['card_faces']:,}")
    print(f"card_types:      {counts['card_types']:,}")
    print(f"card_legalities: {counts['card_legalities']:,}")

    layouts = conn.execute(
        "SELECT layout, COUNT(*) AS n FROM cards GROUP BY layout ORDER BY n DESC LIMIT 8"
    ).fetchall()
    print("layouts: " + ", ".join(f"{row['layout'] or '?'} {row['n']:,}" for row in layouts))

    no_image = conn.execute(
        "SELECT COUNT(*) FROM cards WHERE image_normal IS NULL OR image_normal = ''"
    ).fetchone()[0]
    print(f"cards with no image url: {no_image:,}")

    collisions = conn.execute(
        "SELECT name_norm, COUNT(*) AS n FROM cards GROUP BY name_norm "
        "HAVING n > 1 ORDER BY n DESC, name_norm LIMIT 20"
    ).fetchall()
    if collisions:
        total = conn.execute(
            "SELECT COUNT(*) FROM (SELECT name_norm FROM cards GROUP BY name_norm "
            "HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        print(f"name_norm collisions: {total} folded name(s) held by more than one card")
        for row in collisions:
            names = conn.execute(
                "SELECT name FROM cards WHERE name_norm = ? ORDER BY name", (row["name_norm"],)
            ).fetchall()
            print(f"  {row['name_norm']!r}: " + " | ".join(r["name"] for r in names))
    else:
        print("name_norm collisions: none")
    counts["name_norm_collisions"] = len(collisions)
    return counts


# ----------------------------------------------------------------------- entry point


def run(cfg: Config, force: bool = False) -> dict:
    """Download `oracle_cards` if it moved, then rewrite the corpus. Idempotent."""
    started = time.monotonic()
    conn = oracle_db.connect(cfg)
    try:
        entry = bulk_entry(BULK_TYPE)
        remote_stamp = str(entry.get("updated_at") or "")
        url, path = _pick_source(entry, data_dir(cfg) / "bulk", BULK_TYPE)

        local_stamp = oracle_db.meta_get(conn, oracle_db.UPDATED_AT_KEY)
        have_file = path.is_file() and path.stat().st_size > 0
        populated = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] > 0

        downloaded = False
        if force or local_stamp != remote_stamp or not have_file:
            reason = (
                "forced"
                if force
                else "bulk file missing"
                if not have_file
                else f"bulk data changed ({local_stamp or 'never ingested'} -> {remote_stamp})"
            )
            print(f"oracle: {reason}; downloading {url}", flush=True)
            _download(url, path)
            downloaded = True
        else:
            print(f"oracle: bulk data unchanged ({remote_stamp}); reusing {path}", flush=True)
            if populated:
                print("oracle: cards already ingested, nothing to parse", flush=True)
                checkpoint(conn)
                return {
                    "cards": 0, "new_cards": [], "changed_text": [], "removed": [],
                    "downloaded": False, "seconds": time.monotonic() - started,
                }

        print(f"parsing {path}", flush=True)
        cards = parse_bulk(path)
        result = write(conn, cards)
        oracle_db.meta_set(conn, oracle_db.UPDATED_AT_KEY, remote_stamp)
        # The first field of the oracle index fingerprint. Stamped here, not only
        # in refresh.py, so a manual `python -m cts oracle-ingest` in a terminal
        # is picked up by the running API within a minute too.
        oracle_db.meta_set(
            conn,
            oracle_db.LAST_REFRESH_KEY,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        elapsed = time.monotonic() - started
        print(
            f"oracle-ingest: {result['cards']:,} cards written "
            f"({len(result['new_cards'])} new, {len(result['changed_text'])} with changed "
            f"oracle text, {len(result['removed'])} removed) in {elapsed:.1f}s",
            flush=True,
        )
        checkpoint(conn)
        result["downloaded"] = downloaded
        result["seconds"] = elapsed
        return result
    finally:
        conn.close()
