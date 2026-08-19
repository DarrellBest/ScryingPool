"""The local search API: FastAPI over `cts.search.execute`, bound to loopback.

One process, one connection, one in-memory index, one search at a time. The
design document is docs/superpowers/specs/2026-08-17-discord-service-design.md;
the four things worth knowing before reading the code are:

1. **`execute`'s dict is passed through.** `/search` adds a `service` key and
   nothing else. There is no DTO in front of the result contract, because a
   second definition of that contract is a second place to forget a field.
2. **The index can go stale and nothing tells us.** The weekly refresh writes
   new props and embeddings straight into the database this process is already
   holding open. So before every search we re-read a three-value corpus
   fingerprint and rebuild the index when it moved. A 60s background poll does
   the same thing early so the synchronous check usually finds nothing to do.
3. **One `asyncio.Lock`, and everything hangs off it.** The GPU serialises the
   work regardless, so a second concurrent search would only make both slower.
   The lock is also what makes a `check_same_thread=False` connection safe.
4. **The search runs on a worker thread.** `execute` is blocking synchronous
   code that takes on the order of a minute or more (the actual figure is
   model-dependent and tracked as a measured median — see `cts/timings.py`);
   on the event loop it would make `/health` unanswerable for the whole of
   it, which is the one moment anyone actually wants `/health`.

Run it with `python -m serve.api`.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from cts import db, ollama as ollama_mod, oracle_db, oracle_names, search as search_mod
from cts import timings as timing_mod
from cts import oracle_index as oracle_index_mod, oracle_search as oracle_search_mod
from cts.config import Config, load_config
from cts.index import SearchIndex, load_index
from cts.oracle_filters import Filters
from cts.oracle_index import OracleIndex

# --------------------------------------------------------------------------- constants

DEFAULT_ADDR = "127.0.0.1:8077"

# The only two hosts this service will ever bind. Not a default that can be
# widened by configuration: see check_bind_host().
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})

MAX_QUEUED = 4              # in-flight plus waiting; the 5th request is refused
POLL_SECONDS = 60.0         # background staleness check interval
POLL_DEBOUNCE_SECONDS = 300.0   # at most one poll-driven rebuild per 5 minutes
PROBE_CACHE_SECONDS = 10.0  # systemctl and Ollama probes, so /health cannot hot-loop
REFRESH_UNIT = "cts-refresh.service"

WUBRG = "WUBRG"

# meta.last_refresh_at, MAX(props.id), MAX(embeddings.prop_id). Three index
# seeks against three tables; no COUNT(*), which would scan 170k rows.
FINGERPRINT_SQL = (
    "SELECT (SELECT value        FROM meta       WHERE key = 'last_refresh_at'), "
    "       (SELECT MAX(id)      FROM props), "
    "       (SELECT MAX(prop_id) FROM embeddings)"
)

# The oracle corpus's own fingerprint over its own database file, with the same
# three-field rationale: a "something happened" marker for runs that changed data
# the index cannot see, MAX(id) as a single seek to the end of an INTEGER PRIMARY
# KEY (never COUNT(*), which scans), and MAX(chunk_id) to catch embed lagging
# behind chunk. Same blind spots, same escape hatch (POST /admin/reload).
ORACLE_FINGERPRINT_SQL = (
    "SELECT (SELECT value         FROM meta             WHERE key = 'last_oracle_refresh_at'), "
    "       (SELECT MAX(id)       FROM chunks), "
    "       (SELECT MAX(chunk_id) FROM chunk_embeddings)"
)

Fingerprint = tuple[Any, Any, Any]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------- bind guard


def check_bind_host(host: str) -> str:
    """Return `host` if it is loopback, else raise ValueError naming why.

    Deliberately not a default that configuration can widen. Ollama on this
    machine is already bound 0.0.0.0 with no auth; a second unauthenticated
    service that proxies straight into it is not something this project adds by
    accident. Remote access, if it is ever wanted, is Tailscale's job.
    """
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to bind {host!r}: this API has no authentication and proxies "
            f"straight into Ollama, so it binds loopback only "
            f"({' or '.join(sorted(LOOPBACK_HOSTS))}). "
            "For access from another machine use Tailscale, not a wider bind."
        )
    return host


def parse_addr(raw: str | None) -> tuple[str, int]:
    """Parse `host:port` (or `[host]:port` for IPv6) and enforce the bind guard."""
    value = (raw or DEFAULT_ADDR).strip()
    if not value:
        value = DEFAULT_ADDR

    if value.startswith("["):
        host, sep, rest = value[1:].partition("]")
        if not sep:
            raise ValueError(f"malformed address {raw!r}: unclosed '['")
        port_text = rest[1:] if rest.startswith(":") else rest
    elif value.count(":") == 1:
        host, _, port_text = value.partition(":")
    else:
        # No colon at all, or a bare unbracketed IPv6 literal such as "::1".
        host, port_text = value, ""

    host = host.strip()
    port_text = port_text.strip()
    if not port_text:
        port = int(DEFAULT_ADDR.rsplit(":", 1)[1])
    else:
        try:
            port = int(port_text)
        except ValueError:
            raise ValueError(f"malformed address {raw!r}: {port_text!r} is not a port") from None
        if not 1 <= port <= 65535:
            raise ValueError(f"port {port} out of range in {raw!r}")

    check_bind_host(host)
    return host, port


# ---------------------------------------------------------------------------- database


def corpus_fingerprint(conn: sqlite3.Connection) -> Fingerprint:
    """One row, three seeks. Sub-millisecond, and it is read before every search.

    Blind to an in-place edit that changes no ids and to a full re-embed that
    lands on the same MAX(prop_id). Both are things a human does deliberately,
    and `POST /admin/reload` is the escape hatch for them.
    """
    row = conn.execute(FINGERPRINT_SQL).fetchone()
    return (row[0], row[1], row[2])


def oracle_fingerprint(conn: sqlite3.Connection) -> Fingerprint:
    """The oracle corpus's three seeks. Read on the same 60s tick as the art one.

    One poller checks both fingerprints: a second background task would be a
    second thing to reason about for a check that costs three index seeks.
    """
    row = conn.execute(ORACLE_FINGERPRINT_SQL).fetchone()
    return (row[0], row[1], row[2])


def open_oracle_connection(cfg: Config) -> sqlite3.Connection:
    """The resident handle on `oracle_db_path`, used only by `GET /card`.

    `check_same_thread=False` for a narrower reason than the art connection's:
    this one is *created* on the startup worker thread (`build_engine` runs under
    `asyncio.to_thread`) and thereafter **used only from the event loop**. The
    stdlib guard would reject that first cross-thread use even though nothing
    concurrent ever happens.

    The invariant that makes it safe is that nothing else touches this handle at
    all: name resolution runs on the event loop, the fingerprint check runs
    inline on the event loop (three index seeks — cheaper than the thread hop),
    and the index rebuild — the one genuinely slow thing — is given its **own**
    throwaway connection in `build_name_index`. Two objects that never meet is a
    shorter thing to be confident about than a lock over one that does.
    """
    path = Path(cfg.oracle_db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    oracle_db.init_schema(conn)
    return conn


def build_name_index(cfg: Config) -> oracle_names.NameIndex:
    """Build the `/search` resolver's structures on a connection of its own.

    Runs on a worker thread, so it must not touch the connection `GET /card` is
    reading from on the event loop. Read-only: a rebuild is never a reason to
    write to the corpus.
    """
    conn = sqlite3.connect(f"file:{Path(cfg.oracle_db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return oracle_names.build_index(conn)
    finally:
        conn.close()


def build_oracle_search_index(cfg: Config) -> OracleIndex:
    """Build `/oracle`'s chunk index (vectors + BM25) on a connection of its
    own, same discipline as `build_name_index` right above: a rebuild is read
    only and must never touch the resident `oracle_conn` handle, which only
    the event loop is allowed to use.

    Rebuilt on the **same** fingerprint as the name index — both depend on
    exactly `chunks` / `chunk_embeddings` / `last_oracle_refresh_at` moving —
    so `ensure_oracle_current` rebuilds the two together rather than adding a
    second poller for a check that would cost the same three index seeks
    twice.
    """
    conn = sqlite3.connect(f"file:{Path(cfg.oracle_db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return oracle_index_mod.load_index(cfg, conn)
    finally:
        conn.close()


def read_oracle_stats(cfg: Config) -> dict:
    """Card count and last oracle refresh stamp, on a throwaway read-only handle.

    Read-only and per call for the same reason `read_corpus_stats` is: the bot
    polls `/health` before every search and `oracle_db.connect()` would run
    `init_schema`, i.e. a write transaction, on each one.
    """
    stats: dict = {"cards": None, "chunks": None, "last_oracle_refresh_at": None}
    try:
        conn = sqlite3.connect(f"file:{Path(cfg.oracle_db_path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return stats
    try:
        stats["cards"] = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        stats["chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'last_oracle_refresh_at'"
        ).fetchone()
        stats["last_oracle_refresh_at"] = row[0] if row else None
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return stats


def open_serving_connection(cfg: Config) -> sqlite3.Connection:
    """`cts.db.connect()`'s pragmas, plus `check_same_thread=False`.

    The search and the index build run on worker threads via `asyncio.to_thread`,
    and the threadpool is free to hand each call to a different thread, so the
    stdlib's same-thread guard has to come off this connection.

    That is safe **only** because `Engine.lock` guarantees exactly one user of
    this connection at any moment: the search, the synchronous fingerprint check
    and the background poll all take the same lock, and nothing else touches it.
    `/feedback` and `/health` deliberately open their own short-lived connections
    rather than borrowing this one — not for speed, but so that this invariant
    stays a one-line claim instead of a thing to re-derive per endpoint.
    """
    path = Path(cfg.db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    db.init_schema(conn)
    return conn


def read_corpus_stats(cfg: Config) -> dict:
    """Commander count and last refresh stamp, on a throwaway read-only handle.

    Read-only and opened per call on purpose: `db.connect()` would run
    `init_schema`, i.e. a write transaction, on every `/health` — and the bot
    polls `/health` before every search.
    """
    stats: dict = {"commanders": None, "last_refresh_at": None}
    try:
        conn = sqlite3.connect(f"file:{Path(cfg.db_path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return stats
    try:
        stats["commanders"] = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        row = conn.execute("SELECT value FROM meta WHERE key = 'last_refresh_at'").fetchone()
        stats["last_refresh_at"] = row[0] if row else None
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return stats


def write_feedback(
    cfg: Config,
    *,
    query_id: int,
    illustration_id: str,
    accepted: bool,
    discord_user_id: str | None,
) -> dict:
    """One `judgments` row, `source='discord'`. Mirrors `evaluate.py::_write_mark`.

    `'discord'` is a member of `cts.db.HUMAN_SOURCES`, so downstream this row is
    a human mark in every sense: it wins dedupe against the judge's own row for
    the same (theme, artwork), it carries `export_training.HUMAN_WEIGHT`, and it
    counts toward the abstract P@5 in `cts eval`. The separate value only records
    where the person was sitting.

    Two deliberate divergences from `_write_mark`:

    - **Idempotent.** `judgments` has no unique constraint, so a double-tapped 👍
      would insert twice and a 👍-then-👎 would leave two contradictory training
      rows. Any existing `(query_id, illustration_id, source='discord')` row is
      deleted first, so the latest vote wins.
    - **`query_id` is validated.** The buttons are persistent across bot
      restarts, so a tap on a three-week-old message is normal; an unknown id is
      a 404 rather than an orphan row.

    `prop_ids` is not in the request — the bot would have to carry it through a
    100-character `custom_id`. It is read back off the judge's own row for the
    same (query, artwork), which is where `execute` put the identical list.

    The Discord user id has nowhere structured to go (the schema is SPEC.md
    verbatim and this design adds no column), so it is folded into `rationale`.
    Good enough to audit, not pretending to be structured.
    """
    conn = db.connect(cfg)   # busy_timeout=30000 comes from db.connect()
    try:
        if conn.execute("SELECT 1 FROM queries WHERE id = ?", (query_id,)).fetchone() is None:
            return {"found": False, "replaced": False}

        judged = conn.execute(
            "SELECT prop_ids FROM judgments WHERE query_id = ? AND illustration_id = ? "
            "AND source = 'judge' ORDER BY rowid DESC LIMIT 1",
            (query_id, illustration_id),
        ).fetchone()
        prop_ids = (judged[0] if judged and judged[0] else None) or json.dumps([])

        cursor = conn.execute(
            "DELETE FROM judgments WHERE query_id = ? AND illustration_id = ? "
            "AND source = 'discord'",
            (query_id, illustration_id),
        )
        replaced = cursor.rowcount > 0

        who = f"discord user {discord_user_id}" if discord_user_id else "a discord user"
        verdict = "acceptable" if accepted else "not acceptable"
        conn.execute(
            "INSERT INTO judgments(query_id, illustration_id, fit, rationale, prop_ids, "
            "model, source) VALUES (?, ?, ?, ?, ?, ?, 'discord')",
            (
                query_id,
                illustration_id,
                1.0 if accepted else 0.0,
                f"{who} marked this result {verdict} from the /scry results",
                prop_ids,
                "",  # a human is not a model
            ),
        )
        conn.commit()
        return {"found": True, "replaced": replaced}
    finally:
        conn.close()


def write_oracle_feedback(
    cfg: Config, *, query_id: int, oracle_id: str, accepted: bool, discord_user_id: str | None,
) -> dict:
    """`/oracle/feedback`'s write side. Mirrors `write_feedback` exactly, one
    level down: same idempotent delete-then-insert so a double-tapped 👍 never
    inserts twice, same `query_id` validation against a real logged query, and
    the same literal `'discord'` source string the art side's `db.HUMAN_SOURCES`
    already recognises — a future export reading both databases sees one
    spelling of "a person clicked this," not two."""
    conn = oracle_db.connect(cfg)
    try:
        if conn.execute("SELECT 1 FROM queries WHERE id = ?", (query_id,)).fetchone() is None:
            return {"found": False, "replaced": False}

        judged = conn.execute(
            "SELECT chunk_ids FROM judgments WHERE query_id = ? AND oracle_id = ? "
            "AND source = 'judge' ORDER BY rowid DESC LIMIT 1",
            (query_id, oracle_id),
        ).fetchone()
        chunk_ids = (judged[0] if judged and judged[0] else None) or json.dumps([])

        cursor = conn.execute(
            "DELETE FROM judgments WHERE query_id = ? AND oracle_id = ? AND source = 'discord'",
            (query_id, oracle_id),
        )
        replaced = cursor.rowcount > 0

        who = f"discord user {discord_user_id}" if discord_user_id else "a discord user"
        verdict = "acceptable" if accepted else "not acceptable"
        conn.execute(
            "INSERT INTO judgments(query_id, oracle_id, fit, rationale, chunk_ids, model, source) "
            "VALUES (?, ?, ?, ?, ?, ?, 'discord')",
            (
                query_id, oracle_id, 1.0 if accepted else 0.0,
                f"{who} marked this result {verdict} from the /oracle results",
                chunk_ids, "",
            ),
        )
        conn.commit()
        return {"found": True, "replaced": replaced}
    finally:
        conn.close()


# ------------------------------------------------------------------------------ probes


def probe_refresh_running(unit: str = REFRESH_UNIT) -> bool | None:
    """`systemctl --user is-active <unit>`. None when systemctl cannot answer.

    Unknown and known-not-running are different answers, and the caller renders
    them differently, so this never collapses a failure into False.
    """
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    state = proc.stdout.strip()
    if state in ("active", "activating", "reloading", "refreshing"):
        return True
    if state in ("inactive", "failed", "deactivating", "unknown"):
        return False
    return None


def probe_ollama(cfg: Config) -> dict:
    """Reuse `cts.ollama.preflight`, plus whatever /api/ps says is resident."""
    info: dict = {
        "url": cfg.ollama_url,
        "reachable": False,
        "missing_models": [],
        "loaded": [],
        "error": None,
    }
    try:
        info["missing_models"] = ollama_mod.preflight(cfg)
        info["reachable"] = True
    except Exception as exc:                     # noqa: BLE001 - any failure is "down"
        info["error"] = str(exc)
        return info

    # Which models hold VRAM right now. Best-effort: an Ollama too old to have
    # /api/ps still gets a truthful reachable=True and an empty list.
    try:
        resp = requests.get(
            f"{cfg.ollama_url.rstrip('/')}/api/ps",
            timeout=5,
            headers={"User-Agent": ollama_mod.USER_AGENT},
        )
        if resp.status_code == 200:
            info["loaded"] = [
                entry.get("model") or entry.get("name")
                for entry in resp.json().get("models", [])
                if entry.get("model") or entry.get("name")
            ]
    except Exception:                            # noqa: BLE001 - decoration only
        pass
    return info


class _Cached:
    """A zero-argument callable memoised for `ttl` seconds, awaited off-loop.

    `/health` is polled before every single search, so both probes it makes —
    one subprocess spawn and one HTTP round trip — need a floor on their rate.
    """

    def __init__(self, fn: Callable[[], Any], ttl: float = PROBE_CACHE_SECONDS) -> None:
        self._fn = fn
        self._ttl = ttl
        self._value: Any = None
        self._expires = 0.0

    async def get(self) -> Any:
        if time.monotonic() >= self._expires:
            self._value = await asyncio.to_thread(self._fn)
            self._expires = time.monotonic() + self._ttl
        return self._value

    def invalidate(self) -> None:
        self._expires = 0.0


# ------------------------------------------------------------------------------ engine


class Busy(Exception):
    """Queue cap reached. Rendered as a 503 with the depth that caused it."""

    def __init__(self, queued: int, max_queued: int) -> None:
        super().__init__(f"{queued} searches queued (max {max_queued})")
        self.queued = queued
        self.max_queued = max_queued


class OracleCorpusUnavailable(Exception):
    """`GET /card` was asked something the oracle corpus cannot answer yet.

    Distinct from "no such card" on purpose: an empty corpus and a misspelled
    name are different problems with different fixes, and collapsing them would
    tell a user to check their spelling when the real answer is that nobody has
    run `python -m cts oracle-ingest` on the host.
    """


class Engine:
    """Everything the process holds between requests, and the lock over it.

    Constructed by the lifespan in production and directly by the tests, which
    is why every expensive collaborator — the search callable, the index
    builder, both probes — is an injected argument with a real default rather
    than a module-level import the tests would have to monkeypatch.
    """

    def __init__(
        self,
        cfg: Config,
        conn: sqlite3.Connection,
        index: SearchIndex,
        fingerprint: Fingerprint,
        *,
        search_fn: Callable[..., dict] | None = None,
        index_builder: Callable[[Config, sqlite3.Connection], SearchIndex] | None = None,
        refresh_probe: Callable[[], bool | None] | None = None,
        ollama_probe: Callable[[], dict] | None = None,
        corpus_stats: Callable[[], dict] | None = None,
        oracle_conn: sqlite3.Connection | None = None,
        name_index: Any = None,
        oracle_fingerprint_value: Fingerprint | None = None,
        oracle_index_builder: Callable[[Config], Any] | None = None,
        oracle_stats: Callable[[], dict] | None = None,
        oracle_search_fn: Callable[..., dict] | None = None,
        oracle_search_index: OracleIndex | None = None,
        oracle_search_index_builder: Callable[[Config], OracleIndex] | None = None,
        max_queued: int = MAX_QUEUED,
        poll_seconds: float = POLL_SECONDS,
        poll_debounce_seconds: float = POLL_DEBOUNCE_SECONDS,
    ) -> None:
        self.cfg = cfg
        self.conn = conn
        self.index = index
        self.index_fingerprint = fingerprint
        self.index_built_at = _utcnow()
        self.index_stale = False

        self.search_fn = search_fn or search_mod.execute
        self.index_builder = index_builder or load_index
        self.refresh = _Cached(refresh_probe or probe_refresh_running)
        self.ollama = _Cached(ollama_probe or (lambda: probe_ollama(cfg)))
        self.corpus = _Cached(corpus_stats or (lambda: read_corpus_stats(cfg)))

        # --- the oracle corpus, in its own file, on its own fingerprint --------
        # All optional: an engine constructed without them (every existing test)
        # serves /scry exactly as before and answers /card with an honest 503.
        self.oracle_conn = oracle_conn
        self.name_index = name_index
        self.oracle_index_fingerprint = oracle_fingerprint_value
        self.oracle_index_builder = oracle_index_builder or build_name_index
        self.oracle = _Cached(oracle_stats or (lambda: read_oracle_stats(cfg)))
        self.oracle_built_at = _utcnow()
        self.oracle_stale = False

        # --- /oracle: the chunk search index, rebuilt on the SAME fingerprint
        # as the name index above (see build_oracle_search_index's docstring).
        # `oracle_search_fn` is `cts.oracle_search.execute` by default, and
        # takes the shared search lock below — /oracle and /scry contend for
        # the same Ollama instance and the same judge_model weights, so one
        # lock, not two.
        self.oracle_search_fn = oracle_search_fn or oracle_search_mod.execute
        self.oracle_search_index = oracle_search_index
        self.oracle_search_index_builder = oracle_search_index_builder or build_oracle_search_index
        # Guards the name-index rebuild only. It is NOT a second search queue:
        # /card takes no lock at all on its fast path, because a name lookup
        # queued behind an 80-second /scry would be absurd and the shared search
        # lock exists solely to serialise contention for one Ollama instance that
        # /search never touches.
        self.oracle_rebuild_lock = asyncio.Lock()
        self.lookups_since_start = 0
        self._oracle_stats_cache: dict = {}

        self.max_queued = max_queued
        self.poll_seconds = poll_seconds
        self.poll_debounce_seconds = poll_debounce_seconds

        self.lock = asyncio.Lock()
        self.started_at = _utcnow()
        self._pending = 0          # in-flight plus waiting; what the cap counts
        self._active = False       # a search is holding the lock right now
        self.searches_since_start = 0
        self.last_search_seconds: float | None = None
        self.last_poll_rebuild = 0.0
        self.last_oracle_poll_rebuild = 0.0

    # ---------------------------------------------------------------- index freshness

    def _build(self, fingerprint: Fingerprint) -> tuple[SearchIndex, Fingerprint]:
        """Blocking. Runs on a worker thread; the caller holds the lock."""
        return self.index_builder(self.cfg, self.conn), fingerprint

    async def ensure_current(self, *, force: bool = False) -> bool:
        """Rebuild if the fingerprint moved. Returns whether it rebuilt.

        **The caller must hold `self.lock`.** The fingerprint is read *before*
        the build so that anything committed during those 4-7 seconds is caught
        by the next check rather than silently attributed to this index.

        A failed rebuild keeps the old index and marks it stale: serving last
        week's corpus is bad, serving nothing is worse, and the next search
        tries again.
        """
        try:
            fingerprint = await asyncio.to_thread(corpus_fingerprint, self.conn)
        except sqlite3.Error:
            traceback.print_exc()
            self.index_stale = True
            return False

        if not force and fingerprint == self.index_fingerprint:
            return False

        try:
            # The new index is fully built before the old one is dropped, so peak
            # memory is two matrices for the length of the build. That is the
            # right trade, and it is why the unit sets no MemoryMax=.
            index, built_fingerprint = await asyncio.to_thread(self._build, fingerprint)
        except Exception:                        # noqa: BLE001 - keep serving
            traceback.print_exc()
            self.index_stale = True
            return False

        self.index = index
        self.index_fingerprint = built_fingerprint
        self.index_built_at = _utcnow()
        self.index_stale = False
        return True

    # ------------------------------------------------------------- oracle freshness

    async def ensure_oracle_current(self, *, force: bool = False) -> bool:
        """Rebuild the name index if the oracle fingerprint moved. Returns whether it did.

        Takes `oracle_rebuild_lock`, never the search lock: this must stay
        answerable while an 80-second `/scry` holds the GPU. The builder opens its
        own connection, so nothing here touches `self.oracle_conn`, and the new
        index is fully built before the attribute is rebound — a concurrent
        `/card` therefore sees either the old index or the new one, never a
        half-built one.

        A failed rebuild keeps the old index and marks it stale, for the same
        reason the art one does: serving last week's names is bad, serving nothing
        is worse, and the next miss tries again.
        """
        if self.oracle_conn is None:
            return False
        async with self.oracle_rebuild_lock:
            try:
                # Inline, not on a worker thread: three index seeks against a
                # connection whose whole safety argument is that only the event
                # loop touches it.
                fingerprint = oracle_fingerprint(self.oracle_conn)
            except sqlite3.Error:
                traceback.print_exc()
                self.oracle_stale = True
                return False

            if not force and fingerprint == self.oracle_index_fingerprint:
                return False

            # Both indexes key off exactly this fingerprint (chunks /
            # chunk_embeddings / last_oracle_refresh_at), so one rebuild pass
            # triggers both rather than adding a second poll for a check that
            # would cost the same three index seeks twice. They are rebuilt
            # independently, not as one all-or-nothing step: `/card` depends
            # only on the name index, `/oracle` only on the chunk index, and a
            # failure building one must not stop the other from picking up
            # fresh data it is perfectly able to serve.
            try:
                name_index = await asyncio.to_thread(self.oracle_index_builder, self.cfg)
            except Exception:                    # noqa: BLE001 - keep serving
                traceback.print_exc()
                self.oracle_stale = True
                return False

            self.name_index = name_index
            self.oracle_index_fingerprint = fingerprint
            self.oracle_built_at = _utcnow()
            self.oracle_stale = False

            try:
                self.oracle_search_index = await asyncio.to_thread(
                    self.oracle_search_index_builder, self.cfg
                )
            except Exception:                    # noqa: BLE001 - the name index still rebuilt
                traceback.print_exc()
                # Not `self.oracle_stale = True`: that flag means "the name
                # index — which /search and /card depend on — is stale", and
                # it just successfully rebuilt. `cts.oracle_search.execute`
                # falls back to building its own index inline when handed
                # `index=None`, so a search still works; it is just slower
                # until the next successful poll.

            return True

    def oracle_refreshed_at(self) -> str | None:
        """`meta.last_oracle_refresh_at`, for the "prices as of" label.

        One indexed row off the resident connection, on the event loop: cheaper
        than the cached probe's thread hop, and it is read once per lookup.
        """
        if self.oracle_conn is None:
            return None
        try:
            row = self.oracle_conn.execute(
                "SELECT value FROM meta WHERE key = 'last_oracle_refresh_at'"
            ).fetchone()
        except sqlite3.Error:
            return None
        return row[0] if row else None

    async def resolve_card(self, name: str) -> dict:
        """`GET /card`. No lock, no queue, no `to_thread`, no model call.

        Fast enough to run on the event loop — dict lookups over resident
        structures plus one indexed SQLite row — which is the whole reason
        `/search` is the one command that does not `defer()`.

        The one place staleness is paid for is **failure**: a name that resolves
        to nothing might be a card from Sunday's set release, so the fingerprint
        is checked and the index rebuilt once before "no such card" is reported.
        Doing that on the success path would trade the property that makes this
        command pleasant for nothing.
        """
        self.lookups_since_start += 1
        if self.oracle_conn is None or self.name_index is None:
            raise OracleCorpusUnavailable("the oracle corpus is not configured")

        resolution = oracle_names.resolve(self.name_index, name)
        if resolution.layer is None:
            rebuilt = await self.ensure_oracle_current()
            if rebuilt:
                resolution = oracle_names.resolve(self.name_index, name)

        if len(self.name_index) == 0:
            raise OracleCorpusUnavailable(
                "the oracle corpus is empty — run 'python -m cts oracle-ingest'"
            )

        body: dict = {
            "resolved": resolution.resolved,
            "layer": resolution.layer,
            "input": resolution.query,
            "distance": resolution.distance,
            "total": resolution.total,
            "card": None,
            "candidates": [],
        }
        if resolution.resolved and resolution.oracle_id:
            body["card"] = oracle_names.card_payload(self.oracle_conn, resolution.oracle_id)
        elif resolution.oracle_ids:
            body["candidates"] = oracle_names.candidate_briefs(
                self.oracle_conn, resolution.oracle_ids
            )
        return body

    async def poll_once(self) -> str:
        """One background staleness tick. Returns what it decided, for the tests.

        Latency hiding, not the guarantee: the per-search check in `search()` is
        never skipped, so correctness does not depend on this ever running.

        **One poller, two fingerprints.** The oracle half is checked first and
        outside the search lock, because the name index is rebuilt on its own
        connection and genuinely does not contend with a search in flight —
        skipping it for 80 seconds because a `/scry` is running would delay the
        one index whose rebuild costs a fraction of a second. It keeps the other
        two guards: suppressed while `cts-refresh.service` is active, and its own
        five-minute debounce.
        """
        if self.oracle_conn is not None and await self.refresh.get() is not True:
            if time.monotonic() - self.last_oracle_poll_rebuild >= self.poll_debounce_seconds:
                if await self.ensure_oracle_current():
                    self.last_oracle_poll_rebuild = time.monotonic()

        if self.lock.locked():
            # A search holds it. Do not queue behind a minute-plus of GPU work; try again
            # in a minute.
            return "skipped"
        if await self.refresh.get() is True:
            # An embed stage committing batches would move MAX(embeddings.prop_id)
            # every tick and trigger twenty consecutive rebuilds of an index
            # nobody is querying, against a corpus that is still changing.
            return "suppressed"
        if time.monotonic() - self.last_poll_rebuild < self.poll_debounce_seconds:
            return "debounced"

        async with self.lock:
            rebuilt = await self.ensure_current()
        if rebuilt:
            self.last_poll_rebuild = time.monotonic()
        return "rebuilt" if rebuilt else "current"

    async def poll_forever(self) -> None:
        while True:
            await asyncio.sleep(self.poll_seconds)
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:                    # noqa: BLE001 - a tick may never kill the loop
                traceback.print_exc()

    # ------------------------------------------------------------------------- search

    @property
    def in_flight(self) -> int:
        return 1 if self._active else 0

    @property
    def queued(self) -> int:
        return max(0, self._pending - self.in_flight)

    async def search(
        self, *, theme: str, k: int, band: int | None, colors: str | None
    ) -> dict:
        """One search, serialised, on a worker thread. Raises Busy when full."""
        if self._pending >= self.max_queued:
            raise Busy(self._pending, self.max_queued)

        self._pending += 1
        waiting_since = time.monotonic()
        try:
            async with self.lock:
                queued_seconds = time.monotonic() - waiting_since
                self._active = True
                try:
                    rebuilt = await self.ensure_current()
                    started = time.monotonic()
                    outcome = await asyncio.to_thread(
                        self.search_fn,
                        self.cfg,
                        theme,
                        band=band,
                        colors=colors,
                        k=k,
                        # 'user', not 'discord': export_training.py filters on
                        # kind IN ('user','eval'), so real searches belong in the
                        # training exports. The Discord-ness is recorded on the
                        # feedback row instead.
                        kind="user",
                        conn=self.conn,
                        index=self.index,
                    )
                    elapsed = time.monotonic() - started
                finally:
                    self._active = False
        finally:
            self._pending -= 1

        self.searches_since_start += 1
        self.last_search_seconds = round(elapsed, 3)

        if isinstance(outcome, dict) and outcome.get("query_id") is not None:
            self._record_timing(self.conn, outcome["query_id"], elapsed)

        if isinstance(outcome, dict):
            outcome = dict(outcome)
            # Before _degraded(), which reads plan.notes: an index that is
            # missing this week's vectors is exactly the kind of thing the bot's
            # warning banner exists to say out loud.
            self._note_missing_embeddings(outcome)
            outcome["service"] = {
                "index_rebuilt": rebuilt,
                "index_built_at": _iso(self.index_built_at),
                "queued_seconds": round(queued_seconds, 3),
                "refresh_running": await self.refresh.get(),
                "degraded": _degraded(outcome),
            }
        return outcome

    def _record_timing(self, conn: sqlite3.Connection, query_id: int, elapsed: float) -> None:
        """Persist one completed search's duration for the median `/health`
        reports and the bot renders. Best-effort and deliberately swallows its
        own failures: by this point the user's actual result is already
        computed (and, for /scry, already logged to `queries`), so a timing
        write going wrong must never turn a delivered answer into an error.
        """
        try:
            timing_mod.record(conn, query_id, elapsed)
        except Exception:                              # noqa: BLE001 - see docstring
            traceback.print_exc()

    def _note_missing_embeddings(self, outcome: dict) -> None:
        """Say so when the embed stage is running behind the vision stage."""
        missing = getattr(self.index, "missing_embeddings", 0) or 0
        plan = outcome.get("plan")
        if missing and isinstance(plan, dict) and isinstance(plan.get("notes"), list):
            plan["notes"].append(
                f"{missing:,} propositions have no embedding yet and are absent from "
                "the index — run 'python -m cts embed'"
            )

    # ------------------------------------------------------------------ oracle search

    async def oracle_search(
        self, *, query: str, k: int, types: tuple[str, ...], colors: str | None,
        mv_min: int | None, mv_max: int | None, legal: tuple[str, ...],
    ) -> dict:
        """One `/oracle` search, on the SAME shared lock and queue `/scry`
        uses — they contend for the same Ollama instance and the same
        `judge_model` weights, so a second queue would only move the
        contention somewhere it cannot be reported, per the design doc."""
        if self.oracle_conn is None:
            raise OracleCorpusUnavailable("the oracle corpus is not configured")
        if self._pending >= self.max_queued:
            raise Busy(self._pending, self.max_queued)

        self._pending += 1
        waiting_since = time.monotonic()
        try:
            async with self.lock:
                queued_seconds = time.monotonic() - waiting_since
                self._active = True
                try:
                    rebuilt = await self.ensure_oracle_current()
                    started = time.monotonic()
                    outcome = await asyncio.to_thread(
                        self.oracle_search_fn,
                        self.cfg,
                        query,
                        types=types,
                        colors=colors,
                        mv_min=mv_min,
                        mv_max=mv_max,
                        legal=legal,
                        k=k,
                        kind="user",
                        conn=self.oracle_conn,
                        index=self.oracle_search_index,
                        name_index=self.name_index,
                    )
                    elapsed = time.monotonic() - started
                finally:
                    self._active = False
        finally:
            self._pending -= 1

        self.searches_since_start += 1
        self.last_search_seconds = round(elapsed, 3)

        if isinstance(outcome, dict) and outcome.get("query_id") is not None:
            # self.oracle_conn, not self.conn: /oracle's queries table (and now
            # its search_timings) lives in the separate oracle database, so
            # /scry and /oracle medians never mix.
            self._record_timing(self.oracle_conn, outcome["query_id"], elapsed)

        if isinstance(outcome, dict):
            outcome = dict(outcome)
            outcome["service"] = {
                "oracle_index_rebuilt": rebuilt,
                "oracle_index_built_at": _iso(self.oracle_built_at),
                "queued_seconds": round(queued_seconds, 3),
                "refresh_running": await self.refresh.get(),
                "degraded": _oracle_degraded(outcome),
            }
        return outcome

    # ------------------------------------------------------------------------- health

    def _oracle_health(self, now: datetime) -> dict | None:
        """The `oracle` block, or None when this process has no oracle corpus.

        Reported separately from `index` because "the art index is stale" and
        "the oracle index is stale" are different problems with different causes,
        and one number covering both would name neither.
        """
        if self.oracle_conn is None or self.name_index is None:
            return None
        stats = self._oracle_stats_cache or {}
        search_index = self.oracle_search_index
        # `chunks`/`dim`/`missing_embeddings` describe the CHUNK index that
        # `/oracle` searches, not the name index `/search` and `/card` use —
        # reported from the resident index itself when it exists (accurate at
        # this instant) and falling back to the throwaway-connection COUNT(*)
        # only before the first successful build.
        build_seconds = getattr(self.name_index, "build_seconds", 0.0)
        if search_index is not None:
            build_seconds += getattr(search_index, "build_seconds", 0.0)
        return {
            "cards": getattr(self.name_index, "card_count", 0),
            "names": getattr(self.name_index, "name_count", 0),
            "chunks": len(search_index) if search_index is not None else stats.get("chunks"),
            "dim": getattr(search_index, "dim", None),
            "missing_embeddings": getattr(search_index, "missing_embeddings", None),
            "build_seconds": round(build_seconds, 3),
            "built_at": _iso(self.oracle_built_at),
            "age_seconds": round((now - self.oracle_built_at).total_seconds(), 1),
            "stale": self.oracle_stale,
            "last_oracle_refresh_at": stats.get("last_oracle_refresh_at"),
            "lookups_since_start": self.lookups_since_start,
        }

    async def health(self) -> dict:
        ollama = await self.ollama.get()
        refresh_running = await self.refresh.get()
        corpus = await self.corpus.get()
        self._oracle_stats_cache = (
            await self.oracle.get() if self.oracle_conn is not None else {}
        )

        degraded = not ollama.get("reachable") or bool(ollama.get("missing_models"))
        if degraded:
            status = "degraded"
        elif refresh_running:
            status = "refreshing"
        else:
            status = "ok"

        now = _utcnow()
        oracle_block = self._oracle_health(now)

        # Per-command rolling-window medians (cts/timings.py) — what the bot's
        # placeholder renders instead of a hardcoded "~80s". /scry and /oracle
        # write to different databases, so this reads self.conn for one and
        # self.oracle_conn for the other rather than one shared query. "scry"
        # is always reported; "oracle" only when this process has an oracle
        # corpus at all, mirroring `oracle_block` just above.
        scry_median, scry_samples = timing_mod.median(self.conn)
        timings_block: dict[str, Any] = {
            "scry": {
                "median_seconds": None if scry_median is None else round(scry_median, 3),
                "samples": scry_samples,
            },
        }
        if self.oracle_conn is not None:
            oracle_median, oracle_samples = timing_mod.median(self.oracle_conn)
            timings_block["oracle"] = {
                "median_seconds": None if oracle_median is None else round(oracle_median, 3),
                "samples": oracle_samples,
            }

        body = {
            "status": status,
            "uptime_seconds": round((now - self.started_at).total_seconds(), 1),
            "index": {
                "props": len(self.index),
                "artworks": self.index.artwork_count,
                "dim": self.index.dim,
                "build_seconds": round(self.index.build_seconds, 3),
                "built_at": _iso(self.index_built_at),
                "age_seconds": round((now - self.index_built_at).total_seconds(), 1),
                "missing_embeddings": self.index.missing_embeddings,
                "stale": self.index_stale,
            },
            "corpus": {
                "commanders": corpus.get("commanders"),
                "last_refresh_at": corpus.get("last_refresh_at"),
            },
            "refresh": {"running": refresh_running, "unit": REFRESH_UNIT},
            "ollama": {
                "url": ollama.get("url"),
                "reachable": ollama.get("reachable"),
                "missing_models": ollama.get("missing_models"),
                "loaded": ollama.get("loaded"),
                "error": ollama.get("error"),
            },
            "search": {
                "in_flight": self.in_flight,
                "queued": self.queued,
                "max_queued": self.max_queued,
                "last_search_seconds": self.last_search_seconds,
                "searches_since_start": self.searches_since_start,
            },
        }
        if oracle_block is not None:
            body["oracle"] = oracle_block
        body["timings"] = timings_block
        return body


def _degraded(outcome: dict) -> bool:
    """True when the bot should print a warning banner over the results."""
    plan = outcome.get("plan")
    if not isinstance(plan, dict):
        return False
    notes = plan.get("notes") or []
    return bool(notes) or plan.get("vision_verified") is False


def _oracle_degraded(outcome: dict) -> bool:
    """Same idea as `_degraded`, minus the vision-verification flag this
    pipeline does not have — there is no verify stage for `/oracle` at all."""
    plan = outcome.get("plan")
    if not isinstance(plan, dict):
        return False
    return bool(plan.get("notes"))


# --------------------------------------------------------------------- request models


class SearchRequest(BaseModel):
    theme: str = Field(min_length=1, max_length=300)
    k: int = Field(default=5, ge=1, le=5)      # 5 keeps the reply inside one Discord message
    band: int | None = Field(default=None, ge=1, le=5)
    colors: str | None = None

    @field_validator("theme")
    @classmethod
    def _theme_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("theme must not be blank")
        return stripped

    @field_validator("colors")
    @classmethod
    def _colors_subset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().upper()
        if not cleaned:
            return None
        unknown = sorted(set(cleaned) - set(WUBRG))
        if unknown:
            raise ValueError(
                f"colors must be a subset of WUBRG; {''.join(unknown)!r} is not"
            )
        return "".join(sorted(set(cleaned), key=WUBRG.index))


class FeedbackRequest(BaseModel):
    query_id: int
    illustration_id: str = Field(min_length=1)
    accepted: bool
    discord_user_id: str | None = None


def _validate_colors_subset(value: str | None) -> str | None:
    """Shared by `/search` and `/oracle` — the same letters mean the same
    thing on both commands, and a duplicated validator is how they would
    quietly stop agreeing."""
    if value is None:
        return None
    cleaned = value.strip().upper()
    if not cleaned:
        return None
    unknown = sorted(set(cleaned) - set(WUBRG))
    if unknown:
        raise ValueError(f"colors must be a subset of WUBRG; {''.join(unknown)!r} is not")
    return "".join(sorted(set(cleaned), key=WUBRG.index))


class OracleSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    k: int = Field(default=5, ge=1, le=5)
    types: list[str] = Field(default_factory=list)
    colors: str | None = None
    mv_min: int | None = Field(default=None, ge=0, le=30)
    mv_max: int | None = Field(default=None, ge=0, le=30)
    legal: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped

    @field_validator("colors")
    @classmethod
    def _colors_subset(cls, value: str | None) -> str | None:
        return _validate_colors_subset(value)

    @model_validator(mode="after")
    def _mv_min_not_above_mv_max(self) -> "OracleSearchRequest":
        if self.mv_min is not None and self.mv_max is not None and self.mv_min > self.mv_max:
            raise ValueError(f"mv_min ({self.mv_min}) must not be greater than mv_max ({self.mv_max})")
        return self


class OracleFeedbackRequest(BaseModel):
    query_id: int
    oracle_id: str = Field(min_length=1)
    accepted: bool
    discord_user_id: str | None = None


# --------------------------------------------------------------------------------- app


def build_engine(config_path: str = "config.toml") -> Engine:
    """Startup: config, both connections, both indexes. ~4-7s, paid once per process."""
    cfg = load_config(config_path)
    conn = open_serving_connection(cfg)
    fingerprint = corpus_fingerprint(conn)
    index = load_index(cfg, conn)

    # The oracle corpus is optional at startup: a checkout that has not run
    # `python -m cts oracle-ingest` still serves /scry, and /card and /oracle
    # say exactly what is missing rather than 500ing.
    oracle_conn = open_oracle_connection(cfg)
    return Engine(
        cfg,
        conn,
        index,
        fingerprint,
        oracle_conn=oracle_conn,
        name_index=build_name_index(cfg),
        oracle_fingerprint_value=oracle_fingerprint(oracle_conn),
        oracle_search_index=build_oracle_search_index(cfg),
    )


def create_app(engine: Engine | None = None, *, config_path: str = "config.toml") -> FastAPI:
    """`engine=None` builds the real one at startup; the tests pass their own."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own = engine is None
        app.state.engine = engine or await asyncio.to_thread(build_engine, config_path)
        poller = None
        if app.state.engine.poll_seconds > 0:
            poller = asyncio.create_task(app.state.engine.poll_forever())
        try:
            yield
        finally:
            if poller is not None:
                poller.cancel()
                try:
                    await poller
                except asyncio.CancelledError:
                    pass
            if own:
                app.state.engine.conn.close()
                if app.state.engine.oracle_conn is not None:
                    app.state.engine.oracle_conn.close()

    app = FastAPI(
        title="Scrying Pool",
        summary="Local search API over the commander art corpus. Loopback only.",
        lifespan=lifespan,
    )

    @app.post("/search")
    async def search(request: SearchRequest) -> JSONResponse:
        engine_: Engine = app.state.engine
        try:
            outcome = await engine_.search(
                theme=request.theme, k=request.k, band=request.band, colors=request.colors
            )
        except Busy as busy:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "busy",
                    "queued": busy.queued,
                    "max_queued": busy.max_queued,
                },
            )
        except Exception as exc:                 # noqa: BLE001 - the bot needs a body, not a 502
            traceback.print_exc()
            return JSONResponse(
                status_code=500, content={"status": "error", "detail": str(exc)}
            )
        # default=str because plan carries whatever route() returned, and a
        # judge that emitted something exotic must not 500 the whole search.
        return JSONResponse(content=json.loads(json.dumps(outcome, default=str)))

    @app.post("/feedback")
    async def feedback(request: FeedbackRequest) -> JSONResponse:
        engine_: Engine = app.state.engine
        try:
            result = await asyncio.to_thread(
                write_feedback,
                engine_.cfg,
                query_id=request.query_id,
                illustration_id=request.illustration_id,
                accepted=request.accepted,
                discord_user_id=request.discord_user_id,
            )
        except Exception as exc:                 # noqa: BLE001
            traceback.print_exc()
            return JSONResponse(
                status_code=500, content={"status": "error", "detail": str(exc)}
            )
        if not result["found"]:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "detail": f"no query {request.query_id}: nothing was recorded",
                },
            )
        return JSONResponse(content={"ok": True, "replaced": result["replaced"]})

    @app.post("/oracle/search")
    async def oracle_search_endpoint(request: OracleSearchRequest) -> JSONResponse:
        """`execute()`'s dict passed through unchanged, plus a `service` block
        — same contract discipline as `/search`. Shares `/search`'s queue and
        lock (see `Engine.oracle_search`), so a `503 busy` here counts an
        `/oracle` search queued behind a `/scry` and vice versa."""
        engine_: Engine = app.state.engine
        try:
            outcome = await engine_.oracle_search(
                query=request.query, k=request.k, types=tuple(request.types),
                colors=request.colors, mv_min=request.mv_min, mv_max=request.mv_max,
                legal=tuple(request.legal),
            )
        except OracleCorpusUnavailable as exc:
            return JSONResponse(status_code=503, content={"status": "unavailable", "detail": str(exc)})
        except Busy as busy:
            return JSONResponse(
                status_code=503,
                content={"status": "busy", "queued": busy.queued, "max_queued": busy.max_queued},
            )
        except Exception as exc:                 # noqa: BLE001 - the bot needs a body, not a 502
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})
        return JSONResponse(content=json.loads(json.dumps(outcome, default=str)))

    @app.post("/oracle/feedback")
    async def oracle_feedback(request: OracleFeedbackRequest) -> JSONResponse:
        engine_: Engine = app.state.engine
        try:
            result = await asyncio.to_thread(
                write_oracle_feedback,
                engine_.cfg,
                query_id=request.query_id,
                oracle_id=request.oracle_id,
                accepted=request.accepted,
                discord_user_id=request.discord_user_id,
            )
        except Exception as exc:                 # noqa: BLE001
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})
        if not result["found"]:
            return JSONResponse(
                status_code=404,
                content={"status": "error",
                         "detail": f"no query {request.query_id}: nothing was recorded"},
            )
        return JSONResponse(content={"ok": True, "replaced": result["replaced"]})

    @app.get("/card")
    async def card(name: str = "") -> JSONResponse:
        """Resolve one card name against the local oracle corpus.

        `GET /card`, not `POST /search` — `/search` is already the art search in
        this app, and reusing the path would be a genuine collision. The Discord
        command name and the HTTP route deliberately differ.

        Ambiguity is a **200 with `resolved: false` and a populated `candidates`
        array**, not a 404: the query was well-formed and the answer is "several".
        A genuine miss is `resolved: false` with `candidates: []`. `layer` is on
        every response, which is what makes the resolver debuggable from `curl`
        without reading logs.
        """
        engine_: Engine = app.state.engine
        typed = (name or "").strip()
        if not typed:
            return JSONResponse(
                status_code=422,
                content={"status": "error", "detail": "name must not be blank"},
            )

        started = time.monotonic()
        try:
            body = await engine_.resolve_card(typed)
        except OracleCorpusUnavailable as exc:
            return JSONResponse(
                status_code=503, content={"status": "unavailable", "detail": str(exc)}
            )
        except Exception as exc:                 # noqa: BLE001 - the bot needs a body
            traceback.print_exc()
            return JSONResponse(
                status_code=500, content={"status": "error", "detail": str(exc)}
            )

        body["service"] = {
            "oracle_index_built_at": _iso(engine_.oracle_built_at),
            "oracle_index_stale": engine_.oracle_stale,
            "refreshed_at": engine_.oracle_refreshed_at(),
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }
        return JSONResponse(content=json.loads(json.dumps(body, default=str)))

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(content=await app.state.engine.health())

    @app.post("/admin/reload")
    async def reload(index: str = "all") -> JSONResponse:
        """Force a rebuild, for the blind spots the fingerprints cannot see.

        `?index=art|oracle|all`, defaulting to `all`. Naming one rebuilds only
        that one — an oracle-only reload must not pay the art index's 4-7 seconds,
        and an art-only reload must not drop a name index that is fine.
        """
        engine_: Engine = app.state.engine
        wanted = (index or "all").strip().lower()
        if wanted not in ("art", "oracle", "all"):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "detail": f"index must be one of art, oracle, all; got {index!r}",
                },
            )

        rebuilt = False
        if wanted in ("art", "all"):
            async with engine_.lock:
                rebuilt = await engine_.ensure_current(force=True)

        oracle_rebuilt = False
        if wanted in ("oracle", "all") and engine_.oracle_conn is not None:
            oracle_rebuilt = await engine_.ensure_oracle_current(force=True)

        body = {
            "ok": rebuilt or oracle_rebuilt,
            "index": wanted,
            "index_rebuilt": rebuilt,
            "index_built_at": _iso(engine_.index_built_at),
            "props": len(engine_.index),
            "artworks": engine_.index.artwork_count,
            "missing_embeddings": engine_.index.missing_embeddings,
            "stale": engine_.index_stale,
            "oracle_index_rebuilt": oracle_rebuilt,
        }
        if engine_.name_index is not None:
            body["oracle_index_built_at"] = _iso(engine_.oracle_built_at)
            body["cards"] = getattr(engine_.name_index, "card_count", 0)
            body["oracle_stale"] = engine_.oracle_stale
            body["oracle_chunks"] = (
                len(engine_.oracle_search_index) if engine_.oracle_search_index is not None else None
            )
        return JSONResponse(content=body)

    return app


app = create_app()


def main() -> int:
    import uvicorn

    try:
        host, port = parse_addr(os.environ.get("SCRYING_API_ADDR"))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"scrying-api: binding http://{host}:{port}", file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
