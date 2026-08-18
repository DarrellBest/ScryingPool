"""The oracle corpus's schema and connection, and the guarantee under everything.

The load-bearing test in this file is the last one: **opening and writing
`oracle.db` opens no connection to `commanders.db` at all.** The separation of
the two databases is the backup strategy for ~16 hours of vision work, and a
guarantee nobody checks is a guarantee that quietly stops holding.
"""

from __future__ import annotations

import hashlib
import sqlite3

from cts import db, oracle_db
from cts.config import Config


def _cfg(tmp_path, **overrides) -> Config:
    base = dict(
        ollama_url="http://localhost:11434",
        vision_model="v",
        verify_model="v",
        embed_model="e",
        judge_model="j",
        db_path=str(tmp_path / "commanders.db"),
        art_dir=str(tmp_path / "art"),
        power_weights={},
        oracle_db_path=str(tmp_path / "oracle.db"),
    )
    base.update(overrides)
    return Config(**base)


# ------------------------------------------------------------------------- pragmas


def test_connect_sets_the_same_three_pragmas_db_connect_does(tmp_path):
    """WAL, foreign_keys and — the one that is not optional — busy_timeout.

    The whole argument for why 30s belongs in `db.connect()` rather than at one
    call site applies identically to a second database file: the weekly refresh
    writes this file while the API reads it, and sqlite3's default timeout is
    *zero*, so the loser of a race raises `database is locked` on the spot.
    """
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    finally:
        conn.close()


# -------------------------------------------------------------------------- schema


def test_init_schema_is_idempotent(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "twice.db"))
    try:
        oracle_db.init_schema(conn)
        oracle_db.init_schema(conn)
        oracle_db.init_schema(conn)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {
            "cards", "card_faces", "card_types", "card_legalities",
            "chunks", "chunk_embeddings", "queries", "retrievals", "judgments", "meta",
        } <= tables
    finally:
        conn.close()


def test_schema_carries_every_column_search_and_the_filters_need(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(cards)")}
        assert {
            # /search's embed
            "name", "name_norm", "type_line", "oracle_text", "mana_cost", "layout",
            "image_normal", "price_usd", "price_usd_foil", "scryfall_uri",
            "related_edhrec", "purchase_tcgplayer", "set_code", "rarity",
            # /oracle's filters, which the next phase compiles against this schema
            "cmc", "colors", "color_identity", "power_num", "toughness_num",
            "loyalty_num", "keywords", "reserved", "edhrec_rank", "released_at",
        } <= columns

        face_columns = {row[1] for row in conn.execute("PRAGMA table_info(card_faces)")}
        assert {"oracle_id", "face_index", "name", "name_norm", "image_normal"} <= face_columns
    finally:
        conn.close()


def test_name_norm_is_indexed_but_deliberately_not_unique(tmp_path):
    """A UNIQUE that turns out to be wrong fails the weekly ingest rather than
    degrading, and the un-sets alone hold six cards called Everythingamajig.
    The index gives the lookup speed; the ingest checkpoint prints collisions."""
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(cards)")}
        assert "idx_cards_name_norm" in indexes
        unique = {
            row[1] for row in conn.execute("PRAGMA index_list(cards)") if row[2]
        }
        assert "idx_cards_name_norm" not in unique

        conn.execute("INSERT INTO cards(oracle_id, name, name_norm) VALUES ('a', 'X', 'x')")
        conn.execute("INSERT INTO cards(oracle_id, name, name_norm) VALUES ('b', 'X.', 'x')")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM cards WHERE name_norm = 'x'").fetchone()[0] == 2
    finally:
        conn.close()


def test_every_lookup_a_card_render_makes_is_a_seek_not_a_scan(tmp_path):
    """`/search` reads one card, its faces and its legalities on every hit.

    The design document's schema indexed `card_legalities(format, status)` and
    nothing on `oracle_id`, so SQLite answered the per-card legality lookup with a
    full scan of 780,459 rows — a ~590ms floor under a 20ms endpoint, invisible in
    every test because a test fixture holds seven rows and scans them instantly.
    Asserting the plan rather than the timing is what makes this catchable at all.
    """
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        for query in (
            "SELECT * FROM cards WHERE oracle_id = 'x'",
            "SELECT face_index FROM card_faces WHERE oracle_id = 'x'",
            "SELECT format, status FROM card_legalities WHERE oracle_id = 'x' ORDER BY format",
            "SELECT kind, value FROM card_types WHERE oracle_id = 'x'",
            "SELECT oracle_id FROM cards WHERE name_norm = 'sol ring'",
        ):
            plan = " ".join(
                str(row[3]) for row in conn.execute("EXPLAIN QUERY PLAN " + query)
            )
            assert "SCAN" not in plan, f"{query} -> {plan}"
            assert "SEARCH" in plan, f"{query} -> {plan}"
    finally:
        conn.close()


def test_meta_round_trips(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        assert oracle_db.meta_get(conn, "nothing") is None
        assert oracle_db.meta_get(conn, "nothing", "fallback") == "fallback"
        oracle_db.meta_set(conn, oracle_db.LAST_REFRESH_KEY, "2026-08-17T03:43:02Z")
        oracle_db.meta_set(conn, oracle_db.LAST_REFRESH_KEY, "2026-08-18T03:43:02Z")
        assert oracle_db.meta_get(conn, oracle_db.LAST_REFRESH_KEY) == "2026-08-18T03:43:02Z"
    finally:
        conn.close()


# ----------------------------------------------------------- the separation itself


def test_writing_the_oracle_corpus_never_touches_the_art_database(tmp_path):
    """The guarantee everything else rests on, asserted at the byte level.

    `data/commanders.db` holds 5,530 descriptions and 170,487 embeddings — ~16
    hours of vision calls that cannot be regenerated cheaply. The oracle corpus
    rebuilds from a 24MB download in minutes. They share a file over nobody's
    dead body, and the way that stays true is this test.
    """
    cfg = _cfg(tmp_path)

    art = db.connect(cfg)
    try:
        art.execute("INSERT INTO cards(oracle_id, name) VALUES ('keep-me', 'Avacyn')")
        art.commit()
    finally:
        art.close()

    art_path = tmp_path / "commanders.db"
    before = hashlib.sha256(art_path.read_bytes()).hexdigest()
    before_mtime = art_path.stat().st_mtime_ns

    conn = oracle_db.connect(cfg)
    try:
        conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm, oracle_text) "
            "VALUES ('o1', 'Sol Ring', 'sol ring', '{T}: Add {C}{C}.')"
        )
        conn.execute("INSERT INTO card_faces(oracle_id, face_index, name) VALUES ('o1', 0, 'Sol Ring')")
        conn.execute("INSERT INTO card_types(oracle_id, kind, value) VALUES ('o1', 'type', 'artifact')")
        conn.execute("INSERT INTO chunks(oracle_id, face_index, ordinal, kind, text) "
                     "VALUES ('o1', 0, 0, 'ability', '{T}: Add {C}{C}.')")
        oracle_db.meta_set(conn, oracle_db.LAST_REFRESH_KEY, "now")
        conn.commit()
        # The oracle file exists and is populated...
        assert conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 1
    finally:
        conn.close()

    # ...and the art file is byte-for-byte what it was, with no stray WAL or
    # journal files to suggest anything opened it.
    assert hashlib.sha256(art_path.read_bytes()).hexdigest() == before
    assert art_path.stat().st_mtime_ns == before_mtime

    art = sqlite3.connect(str(art_path))
    try:
        assert art.execute("SELECT name FROM cards").fetchall() == [("Avacyn",)]
        # The oracle schema's tables must not have appeared over here.
        tables = {row[0] for row in art.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "card_faces" not in tables
        assert "chunk_embeddings" not in tables
    finally:
        art.close()


def test_the_two_databases_are_different_files_by_default():
    """Not a coincidence of one config: the defaults themselves are disjoint."""
    from cts.config import DEFAULT_DB_PATH, DEFAULT_ORACLE_DB_PATH

    assert DEFAULT_DB_PATH != DEFAULT_ORACLE_DB_PATH
