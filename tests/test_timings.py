"""cts.timings: the rolling-window median behind the bot's placeholder.

Runs against real sqlite3 connections (`:memory:`, both schemas) — no
network, no Ollama, matching every other test in this suite.
"""

from __future__ import annotations

import sqlite3

from cts import db, oracle_db, timings


def _conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    db.init_schema(connection)
    return connection


def _oracle_conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    oracle_db.init_schema(connection)
    return connection


# --------------------------------------------------------------------- record/recent


def test_record_persists_a_row_recent_can_read_back():
    conn = _conn()
    timings.record(conn, query_id=1, elapsed_seconds=12.5)
    assert timings.recent(conn) == [12.5]


def test_recent_is_newest_first():
    conn = _conn()
    for i, seconds in enumerate((10.0, 20.0, 30.0)):
        timings.record(conn, query_id=i, elapsed_seconds=seconds)
    assert timings.recent(conn) == [30.0, 20.0, 10.0]


def test_the_oracle_schema_has_the_same_table():
    """Both databases grow `search_timings` identically — see the module
    docstring on why the SQL is shared but the connections never are."""
    conn = _oracle_conn()
    timings.record(conn, query_id=1, elapsed_seconds=84.5)
    assert timings.recent(conn) == [84.5]


# -------------------------------------------------------------------- rolling window


def test_recent_respects_the_window_limit():
    conn = _conn()
    for i in range(timings.WINDOW + 5):
        timings.record(conn, query_id=i, elapsed_seconds=float(i))
    values = timings.recent(conn)
    assert len(values) == timings.WINDOW
    # The 5 oldest (0..4) fell out of the window; only the newest WINDOW remain.
    assert min(values) == 5.0


def test_older_rows_are_never_deleted_only_excluded_from_the_window():
    """History is kept (append-only, like queries/retrievals/judgments) —
    only the read side enforces the rolling window."""
    conn = _conn()
    for i in range(timings.WINDOW + 5):
        timings.record(conn, query_id=i, elapsed_seconds=float(i))
    total_rows = conn.execute("SELECT COUNT(*) FROM search_timings").fetchone()[0]
    assert total_rows == timings.WINDOW + 5


def test_a_change_in_typical_duration_shows_up_within_the_window():
    """The scenario the ~20-sample window exists for: a model swap should be
    visible in the median well before it's diluted by stale history."""
    conn = _conn()
    for _ in range(50):
        timings.record(conn, query_id=1, elapsed_seconds=80.0)
    for _ in range(timings.WINDOW):
        timings.record(conn, query_id=1, elapsed_seconds=155.0)
    median_seconds, samples = timings.median(conn)
    assert median_seconds == 155.0
    assert samples == timings.WINDOW


# -------------------------------------------------------------------------- median


def test_median_of_an_odd_sample_count():
    conn = _conn()
    for seconds in (10.0, 30.0, 20.0):
        timings.record(conn, query_id=1, elapsed_seconds=seconds)
    median_seconds, samples = timings.median(conn)
    assert median_seconds == 20.0
    assert samples == 3


def test_median_of_an_even_sample_count_averages_the_middle_two():
    conn = _conn()
    for seconds in (10.0, 20.0, 30.0, 40.0):
        timings.record(conn, query_id=1, elapsed_seconds=seconds)
    median_seconds, samples = timings.median(conn)
    assert median_seconds == 25.0
    assert samples == 4


def test_median_is_not_a_mean_outliers_do_not_drag_it():
    """The whole reason this is a median: a cold load or GPU-contention spike
    must not move the number users are told nearly as much as a mean would."""
    conn = _conn()
    for seconds in (150.0, 152.0, 155.0, 158.0, 160.0):
        timings.record(conn, query_id=1, elapsed_seconds=seconds)
    timings.record(conn, query_id=1, elapsed_seconds=600.0)  # one bad outlier
    median_seconds, _ = timings.median(conn)
    mean_seconds = sum((150.0, 152.0, 155.0, 158.0, 160.0, 600.0)) / 6
    assert median_seconds < mean_seconds
    assert median_seconds in (155.0, 156.5)  # middle of the sorted 6 values


# ------------------------------------------------------------------- empty fallback


def test_median_of_an_empty_window_is_none_with_zero_samples():
    conn = _conn()
    assert timings.median(conn) == (None, 0)


# ------------------------------------------------------------------- per-command


def test_scry_and_oracle_databases_are_independent():
    """The structural half of "per command": /scry and /oracle write to two
    separate SQLite files, so a search on one never moves the other's median —
    proven here directly against real connections, not mocked apart."""
    scry_conn = _conn()
    oracle_conn = _oracle_conn()

    for _ in range(timings.WINDOW):
        timings.record(scry_conn, query_id=1, elapsed_seconds=155.0)
    for _ in range(timings.WINDOW):
        timings.record(oracle_conn, query_id=1, elapsed_seconds=84.5)

    assert timings.median(scry_conn) == (155.0, timings.WINDOW)
    assert timings.median(oracle_conn) == (84.5, timings.WINDOW)
