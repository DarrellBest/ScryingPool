"""SQLite connection and schema. The schema is SPEC.md verbatim, plus `meta`."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import Config

# Everything visual hangs off illustration_id; everything mechanical hangs off
# oracle_id. JSON columns are TEXT holding json.dumps() output.
SCHEMA = """
-- one row per commander, keyed on gameplay identity
CREATE TABLE IF NOT EXISTS cards (
  oracle_id      TEXT PRIMARY KEY,
  name           TEXT,
  type_line      TEXT,
  oracle_text    TEXT,
  mana_cost      TEXT,
  cmc            REAL,
  color_identity TEXT,          -- sorted WUBRG string, "" for colorless
  edhrec_rank    INTEGER
);

-- one row per distinct artwork, many per card
CREATE TABLE IF NOT EXISTS arts (
  illustration_id TEXT PRIMARY KEY,
  oracle_id       TEXT,
  face_index      INTEGER,      -- 0 front, 1 back
  scryfall_id     TEXT,         -- the printing this art was taken from
  set_code        TEXT,
  artist          TEXT,
  is_default      INTEGER,      -- 1 for the printing Scryfall considers primary
  art_crop_url    TEXT,
  art_path        TEXT,         -- null until downloaded
  scryfall_uri    TEXT,
  tcgplayer_uri   TEXT
);

CREATE TABLE IF NOT EXISTS edhrec (
  oracle_id  TEXT PRIMARY KEY,
  slug       TEXT,              -- only ever a slug that returned 200; null on a miss
  themes     TEXT,              -- JSON
  archetypes TEXT,              -- JSON
  num_decks  INTEGER,
  avg_price  REAL,
  raw        TEXT,              -- JSON
  fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS power (
  oracle_id  TEXT PRIMARY KEY,
  score      REAL,
  components TEXT               -- JSON
);

CREATE TABLE IF NOT EXISTS descriptions (
  illustration_id TEXT PRIMARY KEY,
  literal         TEXT,         -- dense factual paragraph
  interpretive    TEXT,         -- mood, narrative, style, register
  slots           TEXT,         -- JSON
  model           TEXT,
  prompt_version  INTEGER,
  created_at      TEXT
);

-- layer is 'literal' or 'interpretive'; they are embedded together but
-- weighted separately at query time
CREATE TABLE IF NOT EXISTS props (
  id              INTEGER PRIMARY KEY,
  illustration_id TEXT,
  layer           TEXT,
  text            TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
  prop_id INTEGER PRIMARY KEY,
  vec     BLOB                  -- float32 numpy tobytes()
);

-- everything below exists to produce training data, see SPEC.md Phase 12
CREATE TABLE IF NOT EXISTS queries (
  id         INTEGER PRIMARY KEY,
  text       TEXT,
  kind       TEXT,              -- user | synth | eval
  params     TEXT,              -- JSON
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS retrievals (
  query_id        INTEGER,
  illustration_id TEXT,
  rank            INTEGER,
  score           REAL,
  method          TEXT,
  layer           TEXT
);

CREATE TABLE IF NOT EXISTS judgments (
  query_id        INTEGER,
  illustration_id TEXT,
  fit             REAL,
  rationale       TEXT,
  prop_ids        TEXT,         -- JSON
  model           TEXT,
  source          TEXT          -- judge | distill | human
);

CREATE TABLE IF NOT EXISTS preferences (
  query_id INTEGER,
  art_a    TEXT,
  art_b    TEXT,
  winner   TEXT,
  source   TEXT
);

-- pipeline bookkeeping: scryfall_updated_at, index build stamps, etc.
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE INDEX IF NOT EXISTS idx_arts_oracle_id        ON arts(oracle_id);
CREATE INDEX IF NOT EXISTS idx_props_illustration_id ON props(illustration_id);
CREATE INDEX IF NOT EXISTS idx_props_layer           ON props(layer);
CREATE INDEX IF NOT EXISTS idx_retrievals_query_id   ON retrievals(query_id);
CREATE INDEX IF NOT EXISTS idx_judgments_query_id    ON judgments(query_id);
CREATE INDEX IF NOT EXISTS idx_judgments_illustration_id ON judgments(illustration_id);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Create every table and index if absent. Safe to call on every connect."""
    conn.executescript(SCHEMA)
    conn.commit()


def connect(cfg: Config) -> sqlite3.Connection:
    path = Path(cfg.db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
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
