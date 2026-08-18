"""The oracle corpus's own database: schema, connection, meta helpers.

A **second SQLite file**, `data/oracle.db`, holding every paper card's rules
text. It is deliberately disjoint from `data/commanders.db`, and the separation
is structural rather than tidiness:

* different grain — one row per `oracle_id` here, one per `illustration_id` there;
* different scope — ~32,700 paper cards here, 3,202 commander-legal ones there;
* different rebuild cost — this file rebuilds from a 24MB download in minutes,
  that one carries ~16 hours of vision descriptions that cannot be regenerated
  cheaply.

Sharing one file would put the expensive corpus at risk on every schema
migration, every `VACUUM`, every restore-from-backup and every mistyped
`DELETE`, to serve a corpus that is cheap to rebuild. **The separation is the
backup strategy**, and `tests/test_oracle_db.py` asserts that writing here opens
no connection to the art database at all.

The two are never joined and never `ATTACH`ed, even though both key on
`oracle_id`. They have different scopes (an inner join silently drops ~90% of
this corpus) and independent refresh cadences (a cross-file read can see two
different weeks). If a combined search is ever wanted it is two independent
searches intersected in the API layer, where both sides' freshness is explicit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import Config

# `cards.name_norm` gets an index and deliberately NOT a UNIQUE constraint.
# Names are believed unique per oracle_id, but "believed" is not "verified across
# 32,726 rows including un-sets", and a UNIQUE that turns out to be wrong fails
# the weekly ingest rather than degrading. The index gives the lookup speed; the
# ingest checkpoint prints any collisions it finds, so a real duplicate surfaces
# as a line of output instead of an outage.
SCHEMA = """
-- one row per gameplay identity, straight from Scryfall's oracle_cards bulk
CREATE TABLE IF NOT EXISTS cards (
  oracle_id      TEXT PRIMARY KEY,
  name           TEXT,           -- verbatim, incl. "//" for multi-face cards
  name_norm      TEXT,           -- fold(name); the L1 resolution key
  type_line      TEXT,
  oracle_text    TEXT,           -- verbatim; faces joined with "\\n//\\n" as ingest.py does
  mana_cost      TEXT,
  cmc            REAL,
  colors         TEXT,           -- WUBRG-ordered, "" for colourless
  color_identity TEXT,           -- WUBRG-ordered
  power          TEXT,           -- verbatim: may be "*", "X", "1+*"
  toughness      TEXT,
  loyalty        TEXT,
  power_num      REAL,           -- NULL unless power parses as a plain integer
  toughness_num  REAL,
  loyalty_num    REAL,
  keywords       TEXT,           -- JSON array, Scryfall's own
  layout         TEXT,           -- decides where image_uris lives; see /search
  reserved       INTEGER,
  edhrec_rank    INTEGER,        -- popularity signal; also the ambiguity tie-break
  released_at    TEXT,
  set_code       TEXT,           -- the representative printing's, not "the" set
  rarity         TEXT,           -- likewise: one printing's rarity
  image_normal   TEXT,           -- *.scryfall.io, hot-linked, never downloaded
  price_usd      REAL,           -- weekly snapshot; see /search on staleness
  price_usd_foil REAL,
  scryfall_uri   TEXT,
  related_edhrec TEXT,           -- Scryfall's related_uris.edhrec, stored not derived
  purchase_tcgplayer TEXT        -- Scryfall's purchase_uris.tcgplayer
);

-- one row per face. Carries the per-face name (so "Petty Theft" resolves) and
-- the per-face image (transform cards have no top-level image_uris at all).
CREATE TABLE IF NOT EXISTS card_faces (
  oracle_id    TEXT,
  face_index   INTEGER,          -- 0 front, 1 back
  name         TEXT,
  name_norm    TEXT,             -- the L2 resolution key
  mana_cost    TEXT,
  type_line    TEXT,
  oracle_text  TEXT,
  image_normal TEXT
);

-- normalized so "type: planeswalker, artifact" is one indexed IN (...)
CREATE TABLE IF NOT EXISTS card_types (
  oracle_id TEXT,
  kind      TEXT,                -- supertype | type | subtype
  value     TEXT                 -- lowercased
);

CREATE TABLE IF NOT EXISTS card_legalities (
  oracle_id TEXT,
  format    TEXT,
  status    TEXT                 -- legal | not_legal | banned | restricted
);

-- one row per ability, plus one whole-card row. See the design doc's "Chunking".
CREATE TABLE IF NOT EXISTS chunks (
  id            INTEGER PRIMARY KEY,
  oracle_id     TEXT,
  face_index    INTEGER,         -- 0 front, 1 back
  ordinal       INTEGER,         -- position within the face; used to mark the match
  kind          TEXT,            -- ability | whole
  text          TEXT,            -- VERBATIM, for display
  text_embedded TEXT             -- name-substituted; what actually got embedded
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id INTEGER PRIMARY KEY,
  vec      BLOB                  -- float32 numpy tobytes(), same as embeddings.vec
);

-- the same Phase 12 bookkeeping, keyed on oracle_id instead of illustration_id
CREATE TABLE IF NOT EXISTS queries    (id INTEGER PRIMARY KEY, text TEXT, kind TEXT,
                                       params TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS retrievals (query_id INTEGER, oracle_id TEXT, rank INTEGER,
                                       score REAL, method TEXT, chunk_id INTEGER);
CREATE TABLE IF NOT EXISTS judgments  (query_id INTEGER, oracle_id TEXT, fit REAL,
                                       rationale TEXT, chunk_ids TEXT, model TEXT,
                                       source TEXT);
CREATE TABLE IF NOT EXISTS meta       (key TEXT PRIMARY KEY, value TEXT);

CREATE INDEX IF NOT EXISTS idx_chunks_oracle_id   ON chunks(oracle_id);
CREATE INDEX IF NOT EXISTS idx_cards_name_norm    ON cards(name_norm);
CREATE INDEX IF NOT EXISTS idx_faces_name_norm    ON card_faces(name_norm);
CREATE INDEX IF NOT EXISTS idx_faces_oracle_id    ON card_faces(oracle_id);
CREATE INDEX IF NOT EXISTS idx_card_types_value   ON card_types(value, kind);
CREATE INDEX IF NOT EXISTS idx_card_types_oracle  ON card_types(oracle_id);
CREATE INDEX IF NOT EXISTS idx_legalities_format  ON card_legalities(format, status);
-- Not in the design document's schema, and measured into it. /search reads one
-- card's legalities on every hit, and (format, status) does not serve that
-- lookup: SQLite answered it with a full scan of 780,459 rows, which put a
-- ~590ms floor under an endpoint whose whole budget is 20ms. With this index the
-- same lookup is a seek. tests/test_oracle_db.py pins the query plan so a future
-- schema edit cannot quietly restore the scan.
CREATE INDEX IF NOT EXISTS idx_legalities_oracle  ON card_legalities(oracle_id);
CREATE INDEX IF NOT EXISTS idx_cards_cmc          ON cards(cmc);
CREATE INDEX IF NOT EXISTS idx_judgments_query_id ON judgments(query_id);
"""

# `meta` keys this corpus owns. `last_oracle_refresh_at` is the first field of the
# oracle index fingerprint, so every stage that changes data the index cannot
# otherwise see moves it.
UPDATED_AT_KEY = "scryfall_oracle_updated_at"
LAST_REFRESH_KEY = "last_oracle_refresh_at"


def init_schema(conn: sqlite3.Connection) -> None:
    """Create every table and index if absent. Safe to call on every connect."""
    conn.executescript(SCHEMA)
    conn.commit()


def connect(cfg: Config) -> sqlite3.Connection:
    """Open `cfg.oracle_db_path` with the same three pragmas `db.connect()` sets.

    `busy_timeout` is not optional here for exactly the reason it is not optional
    there: the weekly refresh writes this file while the API reads it, WAL lets
    them coexist but writers still serialise, and sqlite3's default timeout is
    *zero* — the loser of a race raises `database is locked` on the spot.
    """
    path = Path(cfg.oracle_db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    init_schema(conn)
    return conn


def meta_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return default if row is None else row[0]


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()
