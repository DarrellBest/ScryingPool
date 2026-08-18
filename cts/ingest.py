"""Phase 1: Scryfall bulk ingest.

Two tables come out of one pass over the bulk file:

* `cards` — one row per `oracle_id`, the gameplay identity every printing shares.
* `arts`  — one row per `illustration_id`, the only correct key for "a distinct
  piece of artwork". Reprints reuse art (deduping on printing would describe the
  same image a dozen times) and alternate arts are exactly what this project
  exists to search (deduping on card would throw them away).

Bulk-data source, verified live 2026-08-14
------------------------------------------
`GET https://api.scryfall.com/bulk-data` returns entries shaped like:

    {"object": "bulk_data", "type": "default_cards",
     "updated_at": "2026-08-14T09:05:34.747+00:00",
     "jsonl_download_uri": "https://data.scryfall.io/default-cards/"
                           "default-cards-20260814090534.jsonl.gz",
     "compressed_size": 77513777}

Scryfall no longer publishes `download_uri` (the plain JSON array) — that key is
absent from every entry, and the legacy `.json` object path 404s. Only the
gzipped JSONL file is served. `_pick_source` therefore prefers `download_uri`
when it exists (so the spec's stream-to-`default_cards.json` + `json.load` path
is used verbatim the moment Scryfall restores it) and falls back to
`jsonl_download_uri`, which is streamed to disk compressed and read a line at a
time. Either way the download is chunked and the parse yields card objects.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import requests

from . import db
from .config import Config

# CONTRACT.md network etiquette: a real User-Agent on every request. Defined
# once here; cts/edhrec.py and cts/art.py import it.
USER_AGENT = "ScryingPool/0.1 (github.com/DarrellBest/ScryingPool)"

BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
BULK_TYPE = "default_cards"
META_KEY = "scryfall_updated_at"

# Canonical Magic colour order. color_identity is stored as the subset of these
# letters in this order, so Simic is "UG", Golgari "BG", five-colour "WUBRG",
# and colourless "".
WUBRG = "WUBRG"

_CHUNK = 1 << 20  # 1 MiB per streamed write
_TIMEOUT = (15, 300)  # (connect, read)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def data_dir(cfg: Config) -> Path:
    """The `data/` root, derived from db_path so moving the db moves everything."""
    parent = Path(cfg.db_path).parent
    return Path("data") if str(parent) in ("", ".") else parent


# --------------------------------------------------------------------------
# bulk file: locate, download, iterate
# --------------------------------------------------------------------------


def bulk_entry(bulk_type: str = BULK_TYPE) -> dict:
    """One entry from Scryfall's bulk-data index, `default_cards` by default.

    The parameter exists for `cts/oracle_ingest.py`, which wants the
    `oracle_cards` file — one object per Oracle ID, a third of the size, and
    exactly the grain that corpus searches. Defaulting to the existing constant
    means no existing caller changes behaviour.
    """
    resp = requests.get(
        BULK_DATA_URL, timeout=_TIMEOUT, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    for entry in resp.json().get("data", []):
        if entry.get("type") == bulk_type:
            return entry
    raise RuntimeError(f"no {bulk_type!r} entry in {BULK_DATA_URL}")


def _pick_source(
    entry: dict, dest_dir: Path, bulk_type: str = BULK_TYPE
) -> tuple[str, Path]:
    """(download url, local path). See the module docstring for why both exist.

    The filename follows `bulk_type`, so the two corpora's bulk files sit side by
    side in `data/bulk/` and neither can overwrite the other.
    """
    url = entry.get("download_uri")
    if url:
        return url, dest_dir / f"{bulk_type}.json"
    url = entry.get("jsonl_download_uri")
    if url:
        suffix = ".jsonl.gz" if url.endswith(".gz") else ".jsonl"
        return url, dest_dir / f"{bulk_type}{suffix}"
    raise RuntimeError(
        f"{bulk_type} entry exposes no download URL; keys were {sorted(entry)}"
    )


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` in chunks, via a .part file so an interrupted
    download never leaves a truncated file that later looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    written = 0
    next_report = 25 << 20
    with requests.get(
        url, stream=True, timeout=_TIMEOUT, headers={"User-Agent": USER_AGENT}
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    pct = f" ({100 * written // total}%)" if total else ""
                    print(f"  {written >> 20} MiB{pct}", flush=True)
                    next_report += 25 << 20
    os.replace(tmp, dest)
    print(f"  saved {written >> 20} MiB to {dest}", flush=True)


def iter_card_objects(path: Path) -> Iterator[dict]:
    """Yield every card object in the bulk file, whichever format it is in."""
    name = path.name
    if name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif name.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        # SPEC.md's path: one big JSON array, loaded whole (plenty of RAM).
        with path.open("rb") as fh:
            for card in json.load(fh):
                yield card


# --------------------------------------------------------------------------
# filter and field extraction
# --------------------------------------------------------------------------


def is_commander(card: dict) -> bool:
    """SPEC.md Phase 1 filter, exactly and only:

        legalities.commander == "legal"
        AND ("Legendary Creature" in type_line
             OR "can be your commander" in oracle_text)

    Double-faced cards keep `type_line` and `oracle_text` on `card_faces`, so
    both are checked at the top level and on every face. The top-level
    `type_line` of a DFC is already the combined "Front — Type // Back — Type",
    so the face scan is insurance for layouts (reversible_card) that leave the
    top level empty.

    Nothing else is filtered — not layout, not set type, not digital-only.

    Known consequence, verified live: Grist, the Hunger Tide is *excluded*. Its
    type line is "Legendary Planeswalker — Grist" and its rules text says "As
    long as Grist isn't on the battlefield, it's a 1/1 Insect creature" rather
    than "can be your commander", so it matches neither clause. Planeswalker
    commanders that carry the explicit sentence (Freyalise, Llanowar's Fury and
    the rest of the Commander 2014 cycle) are included normally.
    """
    if (card.get("legalities") or {}).get("commander") != "legal":
        return False

    faces = card.get("card_faces") or []

    type_lines = [card.get("type_line") or ""]
    type_lines += [f.get("type_line") or "" for f in faces]
    if any("Legendary Creature" in t for t in type_lines):
        return True

    texts = [card.get("oracle_text") or ""]
    texts += [f.get("oracle_text") or "" for f in faces]
    return any("can be your commander" in t for t in texts)


def oracle_id_of(card: dict) -> str | None:
    """Top-level oracle_id, or the first face's (reversible_card layout)."""
    oracle_id = card.get("oracle_id")
    if oracle_id:
        return oracle_id
    for face in card.get("card_faces") or []:
        if face.get("oracle_id"):
            return face["oracle_id"]
    return None


def color_identity(colors: Any) -> str:
    """Colour identity as a canonical WUBRG-ordered string, "" for colourless.

    Scryfall returns an unordered list (Atraxa is ["G","W","U","B"]); ordering
    by position in "WUBRG" makes it a stable key, so Atraxa is always "WUBG".
    """
    have = set(colors or ())
    return "".join(c for c in WUBRG if c in have)


def trim(card: dict) -> dict:
    """Keep only the fields the two tables need.

    The bulk file holds ~110k card objects; the commander subset is ~25k
    printings and holding those whole would be an order of magnitude more
    memory than this.
    """
    faces = [
        {
            "illustration_id": f.get("illustration_id"),
            "art_crop": (f.get("image_uris") or {}).get("art_crop"),
            "artist": f.get("artist"),
            "type_line": f.get("type_line"),
            "oracle_text": f.get("oracle_text"),
            "mana_cost": f.get("mana_cost"),
        }
        for f in card.get("card_faces") or []
    ]
    return {
        "id": card.get("id"),
        "name": card.get("name") or "",
        "type_line": card.get("type_line") or "",
        "oracle_text": card.get("oracle_text") or "",
        "mana_cost": card.get("mana_cost"),
        "cmc": card.get("cmc"),
        "color_identity": card.get("color_identity") or [],
        "edhrec_rank": card.get("edhrec_rank"),
        "set": card.get("set"),
        "artist": card.get("artist"),
        "released_at": card.get("released_at") or "",
        "digital": bool(card.get("digital")),
        "illustration_id": card.get("illustration_id"),
        "art_crop": (card.get("image_uris") or {}).get("art_crop"),
        "scryfall_uri": card.get("scryfall_uri"),
        # purchase_uris is absent on plenty of printings (tokens, online-only,
        # some promos) — NULL is the right answer there.
        "tcgplayer_uri": (card.get("purchase_uris") or {}).get("tcgplayer"),
        "faces": faces,
    }


def art_entries(printing: dict) -> list[tuple[int, str, str, str | None]]:
    """(face_index, illustration_id, art_crop_url, artist) for one printing.

    Transform / modal-DFC printings carry art per face, so both faces become
    separate `arts` rows with face_index 0 and 1 — back faces are legitimate
    theme material. Faces genuinely without art are skipped: split, adventure,
    flip and class layouts all have `card_faces` whose `image_uris` is empty
    because the printing has a single illustration at the top level, and some
    DFC faces have an illustration_id but no images. When no face yields art,
    fall back to the top-level illustration.

    Only `art_crop` is stored — never `normal` or `png`.
    """
    out: list[tuple[int, str, str, str | None]] = []
    for index, face in enumerate(printing["faces"]):
        if face["illustration_id"] and face["art_crop"]:
            out.append(
                (
                    index,
                    face["illustration_id"],
                    face["art_crop"],
                    face["artist"] or printing["artist"],
                )
            )
    if out:
        return out
    if printing["illustration_id"] and printing["art_crop"]:
        return [
            (0, printing["illustration_id"], printing["art_crop"], printing["artist"])
        ]
    return []


def pick_default(printings: list[dict]) -> dict:
    """The printing whose art counts as the commander's primary one.

    Rule (deterministic, and the best signal the bulk file still carries): among
    a card's printings prefer the ones that are not digital-only, and inside
    that pool take the most recent `released_at`; ties break on the Scryfall id
    so the choice never depends on file order. If every printing is digital,
    fall back to the most recent of those.

    This is a proxy, not Scryfall's own default-printing algorithm — the bulk
    export exposes no "this is the default" flag, and `highres_image` is true
    for nearly every paper printing so it separates nothing. What the rule
    guarantees is what Phase 5 actually needs: exactly one printing per
    commander marked is_default=1, chosen the same way on every run, biased
    toward the newest paper art.
    """
    pool = [p for p in printings if not p["digital"]] or printings
    return max(pool, key=lambda p: (p["released_at"], p["id"] or ""))


def card_row(oracle_id: str, ordered: list[dict]) -> tuple:
    """The single `cards` row for an oracle_id, taken from its default printing.

    Mechanical fields are shared by every printing, so which one supplies them
    barely matters — except on double-faced cards, where the top level is
    partly empty and the faces have to be joined:

    * type_line — top level already reads "Front — Type // Back — Type".
    * oracle_text — empty at the top level on transform/MDFC cards; both faces
      are joined with "\\n//\\n" so downstream text search sees the whole card.
    * mana_cost — null at the top level on transform cards; the front face's
      cost is the castable one.
    * edhrec_rank — missing on some printings, so take the first that has one.
    """
    default = ordered[0]
    faces = default["faces"]

    type_line = default["type_line"]
    if not type_line and faces:
        type_line = " // ".join(f["type_line"] or "" for f in faces).strip(" /")

    oracle_text = default["oracle_text"]
    if not oracle_text and faces:
        oracle_text = "\n//\n".join(f["oracle_text"] or "" for f in faces).strip()

    mana_cost = default["mana_cost"]
    if not mana_cost and faces:
        mana_cost = faces[0]["mana_cost"] or None

    rank = next((p["edhrec_rank"] for p in ordered if p["edhrec_rank"] is not None), None)

    return (
        oracle_id,
        default["name"],
        type_line,
        oracle_text,
        mana_cost,
        float(default["cmc"]) if default["cmc"] is not None else 0.0,
        color_identity(default["color_identity"]),
        rank,
    )


# --------------------------------------------------------------------------
# parse and write
# --------------------------------------------------------------------------


def parse_bulk(path: Path) -> dict[str, list[dict]]:
    """Every commander-legal printing in the bulk file, grouped by oracle_id."""
    by_oracle: dict[str, list[dict]] = {}
    scanned = 0
    kept = 0
    skipped_no_oracle = 0

    for card in iter_card_objects(path):
        scanned += 1
        if scanned % 25000 == 0:
            print(
                f"  scanned {scanned:,} card objects, kept {kept:,} commander printings",
                flush=True,
            )
        if not is_commander(card):
            continue
        oracle_id = oracle_id_of(card)
        if not oracle_id:
            skipped_no_oracle += 1
            continue
        by_oracle.setdefault(oracle_id, []).append(trim(card))
        kept += 1

    print(
        f"  scanned {scanned:,} card objects, kept {kept:,} commander printings "
        f"across {len(by_oracle):,} oracle ids",
        flush=True,
    )
    if skipped_no_oracle:
        print(f"  skipped {skipped_no_oracle} printings with no oracle_id", flush=True)
    return by_oracle


_INSERT_CARD = """
INSERT OR IGNORE INTO cards
  (oracle_id, name, type_line, oracle_text, mana_cost, cmc, color_identity, edhrec_rank)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_ART = """
INSERT OR IGNORE INTO arts
  (illustration_id, oracle_id, face_index, scryfall_id, set_code, artist,
   is_default, art_crop_url, art_path, scryfall_uri, tcgplayer_uri)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
"""

# Any commander whose default printing's artwork was claimed by a different
# oracle_id (shared illustration) would otherwise have no default at all.
_PROMOTE_ORPHANS = """
UPDATE arts SET is_default = 1
WHERE illustration_id IN (
  SELECT MIN(a.illustration_id) FROM arts a
  WHERE NOT EXISTS (
    SELECT 1 FROM arts b WHERE b.oracle_id = a.oracle_id AND b.is_default = 1
  )
  GROUP BY a.oracle_id
)
"""


def write(conn: sqlite3.Connection, by_oracle: dict[str, list[dict]]) -> tuple[list[str], int, int]:
    """Insert cards and arts. Returns (new card names, new arts, promoted)."""
    existing_cards = {r[0] for r in conn.execute("SELECT oracle_id FROM cards")}
    existing_arts = {r[0] for r in conn.execute("SELECT illustration_id FROM arts")}

    new_cards: list[str] = []
    new_arts = 0
    total = len(by_oracle)

    # Sorted so illustration_ids shared between two commanders are always
    # claimed by the same one, whatever order the bulk file happened to use.
    for done, oracle_id in enumerate(sorted(by_oracle), start=1):
        printings = by_oracle[oracle_id]
        default = pick_default(printings)
        rest = sorted(
            (p for p in printings if p is not default),
            key=lambda p: (p["released_at"], p["id"] or ""),
            reverse=True,
        )
        ordered = [default] + rest

        row = card_row(oracle_id, ordered)
        if oracle_id not in existing_cards:
            new_cards.append(row[1])
            existing_cards.add(oracle_id)
        conn.execute(_INSERT_CARD, row)

        default_iids: list[str] = []
        for printing in ordered:
            is_default = 1 if printing is default else 0
            for face_index, illustration_id, art_crop, artist in art_entries(printing):
                if is_default:
                    default_iids.append(illustration_id)
                if illustration_id not in existing_arts:
                    existing_arts.add(illustration_id)
                    new_arts += 1
                conn.execute(
                    _INSERT_ART,
                    (
                        illustration_id,
                        oracle_id,
                        face_index,
                        printing["id"],
                        printing["set"],
                        artist,
                        is_default,
                        art_crop,
                        printing["scryfall_uri"],
                        printing["tcgplayer_uri"],
                    ),
                )

        # Arts rows are insert-or-ignore (their art_path, set and urls belong to
        # the printing that first carried the illustration and must not churn),
        # but is_default is a property of the *card* and moves when a newer
        # printing arrives, so it is re-asserted every run. Both statements are
        # no-ops when the flags are already right.
        if default_iids:
            marks = ",".join("?" * len(default_iids))
            conn.execute(
                f"UPDATE arts SET is_default = 0 WHERE oracle_id = ? AND is_default = 1 "
                f"AND illustration_id NOT IN ({marks})",
                [oracle_id, *default_iids],
            )
            conn.execute(
                f"UPDATE arts SET is_default = 1 WHERE oracle_id = ? AND is_default = 0 "
                f"AND illustration_id IN ({marks})",
                [oracle_id, *default_iids],
            )

        if done % 250 == 0:
            conn.commit()
            print(f"  wrote {done:,}/{total:,} commanders", flush=True)

    conn.commit()
    promoted = conn.execute(_PROMOTE_ORPHANS).rowcount
    conn.commit()
    return new_cards, new_arts, max(promoted, 0)


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------


def checkpoint(conn: sqlite3.Connection) -> tuple[int, int]:
    """SPEC.md's sanity check: row counts, plus the ten commanders with the most
    distinct arts. If dedup were keying on printing instead of illustration_id
    this list would be reprint-heavy staples rather than alternate-art magnets.
    """
    cards = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    arts = conn.execute("SELECT COUNT(*) FROM arts").fetchone()[0]
    defaults = conn.execute("SELECT COUNT(*) FROM arts WHERE is_default = 1").fetchone()[0]

    print(f"cards: {cards:,}   (SPEC.md expects roughly 2,500)")
    print(f"arts:  {arts:,}   (SPEC.md expects roughly 4,000-5,000; {defaults:,} default)")
    print("ten commanders with the most distinct arts:")
    rows = conn.execute(
        """
        SELECT c.name AS name, COUNT(*) AS n
        FROM arts a JOIN cards c ON c.oracle_id = a.oracle_id
        GROUP BY a.oracle_id
        ORDER BY n DESC, c.name ASC
        LIMIT 10
        """
    ).fetchall()
    for row in rows:
        print(f"  {row['n']:>3}  {row['name']}")
    return cards, arts


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run(cfg: Config, force: bool = False) -> dict:
    """Phase 1. Idempotent: a rerun with unchanged bulk data does no work."""
    conn = db.connect(cfg)
    try:
        entry = bulk_entry()
        remote_stamp = str(entry.get("updated_at") or "")
        url, path = _pick_source(entry, data_dir(cfg) / "bulk")

        local_stamp = db.meta_get(conn, META_KEY)
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
            print(f"scryfall: {reason}; downloading {url}", flush=True)
            _download(url, path)
            downloaded = True
        else:
            print(f"scryfall: bulk data unchanged ({remote_stamp}); reusing {path}", flush=True)
            if populated:
                print("scryfall: cards already ingested, nothing to parse", flush=True)
                checkpoint(conn)
                return {"new_cards": [], "new_arts": 0, "downloaded": False}

        print(f"parsing {path}", flush=True)
        by_oracle = parse_bulk(path)
        new_cards, new_arts, promoted = write(conn, by_oracle)
        db.meta_set(conn, META_KEY, remote_stamp)

        print(f"ingest: {len(new_cards)} new commanders, {new_arts} new artworks", flush=True)
        if promoted:
            print(f"ingest: promoted {promoted} arts to default (shared illustration)", flush=True)
        checkpoint(conn)
        return {"new_cards": new_cards, "new_arts": new_arts, "downloaded": downloaded}
    finally:
        conn.close()
