"""The pragmas db.connect() sets. One of them is load-bearing for the API.

WAL and foreign_keys were always here. busy_timeout was not, and its absence is
invisible until two processes write at once: sqlite3's default is a *zero*
timeout, so the loser of a race raises `database is locked` immediately rather
than waiting. For a search that is a catastrophic way to fail — the exception
comes out of _log_query, after ~80s of GPU work has already been spent.
"""

from __future__ import annotations

from cts import db
from cts.config import Config


def _cfg(tmp_path) -> Config:
    return Config(
        ollama_url="http://localhost:11434",
        vision_model="v",
        verify_model="v",
        embed_model="e",
        judge_model="j",
        db_path=str(tmp_path / "pragmas.db"),
        art_dir=str(tmp_path / "art"),
        power_weights={},
    )


def test_connect_sets_a_thirty_second_busy_timeout(tmp_path):
    conn = db.connect(_cfg(tmp_path))
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    finally:
        conn.close()


def test_connect_still_sets_wal_and_foreign_keys(tmp_path):
    conn = db.connect(_cfg(tmp_path))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_busy_timeout_makes_a_contended_write_wait_instead_of_raising(tmp_path):
    """Two connections, one holding the write lock. The second must not raise.

    The pragma is 30s and this test must not take 30s, so the holder releases
    after a beat; what is asserted is that the second writer *waited* for it
    rather than giving up. Without the pragma the write raises OperationalError
    on the spot — which is the bug this exists to catch if anyone removes it.
    """
    import sqlite3
    import threading
    import time

    cfg = _cfg(tmp_path)
    hold_for = 0.5
    holding = threading.Event()
    failure: list[BaseException] = []

    def hold_the_write_lock() -> None:
        # Its own connection, created and used entirely inside this thread:
        # sqlite3 connections are not shareable across threads by default.
        conn = db.connect(cfg)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO meta(key, value) VALUES ('holder', '1')")
            holding.set()
            time.sleep(hold_for)
            conn.commit()
        except BaseException as exc:             # pragma: no cover - diagnostic only
            failure.append(exc)
            holding.set()
        finally:
            conn.close()

    holder = threading.Thread(target=hold_the_write_lock)
    holder.start()
    assert holding.wait(5), "the holder thread never took the write lock"

    other = db.connect(cfg)
    try:
        started = time.monotonic()
        try:
            other.execute("INSERT INTO meta(key, value) VALUES ('other', '2')")
            other.commit()
        except sqlite3.OperationalError as exc:  # pragma: no cover - the bug we fixed
            raise AssertionError(f"contended write raised instead of waiting: {exc}") from None
        waited = time.monotonic() - started

        holder.join(10)
        assert not failure, failure
        assert waited >= hold_for / 2, (
            f"the second write returned in {waited:.3f}s, so it never actually "
            "contended — the test is not exercising busy_timeout"
        )
        assert other.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 2
    finally:
        other.close()
