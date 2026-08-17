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
   code that takes ~80s; on the event loop it would make `/health` unanswerable
   for the whole of it, which is the one moment anyone actually wants `/health`.

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
from pydantic import BaseModel, Field, field_validator

from cts import db, ollama as ollama_mod, search as search_mod
from cts.config import Config, load_config
from cts.index import SearchIndex, load_index

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

    async def poll_once(self) -> str:
        """One background staleness tick. Returns what it decided, for the tests.

        Latency hiding, not the guarantee: the per-search check in `search()` is
        never skipped, so correctness does not depend on this ever running.
        """
        if self.lock.locked():
            # A search holds it. Do not queue behind ~80s of GPU work; try again
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

    def _note_missing_embeddings(self, outcome: dict) -> None:
        """Say so when the embed stage is running behind the vision stage."""
        missing = getattr(self.index, "missing_embeddings", 0) or 0
        plan = outcome.get("plan")
        if missing and isinstance(plan, dict) and isinstance(plan.get("notes"), list):
            plan["notes"].append(
                f"{missing:,} propositions have no embedding yet and are absent from "
                "the index — run 'python -m cts embed'"
            )

    # ------------------------------------------------------------------------- health

    async def health(self) -> dict:
        ollama = await self.ollama.get()
        refresh_running = await self.refresh.get()
        corpus = await self.corpus.get()

        degraded = not ollama.get("reachable") or bool(ollama.get("missing_models"))
        if degraded:
            status = "degraded"
        elif refresh_running:
            status = "refreshing"
        else:
            status = "ok"

        now = _utcnow()
        return {
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


def _degraded(outcome: dict) -> bool:
    """True when the bot should print a warning banner over the results."""
    plan = outcome.get("plan")
    if not isinstance(plan, dict):
        return False
    notes = plan.get("notes") or []
    return bool(notes) or plan.get("vision_verified") is False


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


# --------------------------------------------------------------------------------- app


def build_engine(config_path: str = "config.toml") -> Engine:
    """Startup: config, connection, index. ~4-7s, paid once per process."""
    cfg = load_config(config_path)
    conn = open_serving_connection(cfg)
    fingerprint = corpus_fingerprint(conn)
    index = load_index(cfg, conn)
    return Engine(cfg, conn, index, fingerprint)


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

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(content=await app.state.engine.health())

    @app.post("/admin/reload")
    async def reload() -> JSONResponse:
        """Force a rebuild, for the blind spots the fingerprint cannot see."""
        engine_: Engine = app.state.engine
        async with engine_.lock:
            rebuilt = await engine_.ensure_current(force=True)
        return JSONResponse(
            content={
                "ok": rebuilt,
                "index_rebuilt": rebuilt,
                "index_built_at": _iso(engine_.index_built_at),
                "props": len(engine_.index),
                "artworks": engine_.index.artwork_count,
                "missing_embeddings": engine_.index.missing_embeddings,
                "stale": engine_.index_stale,
            }
        )

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
