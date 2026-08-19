"""Search duration tracking: how long a completed `/scry` or `/oracle`
search actually took, persisted so the Discord placeholder's "~2m30s"
estimate survives an API restart (in-memory alone would forget it, and this
API restarts often).

Lives in its own module rather than being folded into `cts/db.py` or
`cts/oracle_db.py` because `/scry` and `/oracle` write to two entirely
separate SQLite files — see `oracle_db.py`'s own module docstring on why
that split is deliberate and never crossed. The SQL below is identical
either way; it just runs against whichever connection the caller already
has open. Both schemas grow the same `search_timings` table (see each
module's own `SCHEMA` string); this module never opens a connection of its
own, only ever receives one, so importing it does not blur the separation.

Design: a rolling window of the last `WINDOW` completed searches, read back
as a **median**, not a mean — a handful of cold starts or GPU-contention
outliers must not drag the number users are told away from what a normal
wait actually feels like. Nothing is ever deleted from `search_timings`;
the window is enforced purely by the `ORDER BY id DESC LIMIT` on read, so
full history stays around for later analysis, consistent with how
`queries` / `retrievals` / `judgments` are append-only everywhere else in
this schema.
"""

from __future__ import annotations

import sqlite3
import statistics
from datetime import datetime, timezone

# ~20 completed searches is small enough that a model swap (like the
# scryingpool-qwen3.8 change this module was built for) shows up in the
# median within a day of normal use, rather than being diluted for weeks by
# stale numbers from a previous model.
WINDOW = 20


def record(conn: sqlite3.Connection, query_id: int, elapsed_seconds: float) -> None:
    """One row per completed search.

    Called only after the search itself has already produced a result the
    user will see — callers treat this as best-effort and must not let a
    failure here turn a delivered answer into a reported error.
    """
    conn.execute(
        "INSERT INTO search_timings(query_id, elapsed_seconds, created_at) "
        "VALUES (?, ?, ?)",
        (int(query_id), float(elapsed_seconds), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def recent(conn: sqlite3.Connection, limit: int = WINDOW) -> list[float]:
    """The most recent `limit` durations. Order does not matter to the caller
    beyond "these are the newest ones" — only the set feeds the median."""
    rows = conn.execute(
        "SELECT elapsed_seconds FROM search_timings ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [float(r[0]) for r in rows if r[0] is not None]


def median(conn: sqlite3.Connection, limit: int = WINDOW) -> tuple[float | None, int]:
    """`(median_seconds, sample_count)`. `(None, 0)` when the window is empty —
    a fresh install or a database nobody has searched against yet."""
    values = recent(conn, limit)
    if not values:
        return None, 0
    return statistics.median(values), len(values)
