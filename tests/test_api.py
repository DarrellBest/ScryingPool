"""The HTTP surface: shapes, validation, the queue cap, and the threadpool.

Nothing here starts Ollama, opens a socket to anything, or reads the real
corpus. The engine is constructed by the test with a stub search callable and a
stub index builder and handed to `create_app`, which is the reason
`serve.api.Engine` takes those as constructor arguments in the first place.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient                          # noqa: E402

import serve_support as support                                    # noqa: E402
from serve.api import Engine, corpus_fingerprint, create_app       # noqa: E402


def make_engine(
    conn,
    *,
    search_fn=None,
    builder=None,
    refresh_running=False,
    ollama=None,
    max_queued=4,
    index=None,
) -> Engine:
    return Engine(
        support.config(),
        conn,
        index or support.StubIndex(),
        corpus_fingerprint(conn),
        search_fn=search_fn or support.StubSearch(),
        index_builder=builder or support.StubBuilder(),
        refresh_probe=lambda: refresh_running,
        ollama_probe=ollama or support.ollama_ok,
        corpus_stats=lambda: {
            "commanders": 3202,
            "last_refresh_at": "2026-08-17T03:39:55Z",
        },
        max_queued=max_queued,
        poll_seconds=0,          # no background task inside a TestClient
    )


@pytest.fixture
def conn():
    connection = support.memory_conn()
    yield connection
    connection.close()


@pytest.fixture
def searcher():
    return support.StubSearch()


@pytest.fixture
def client(conn, searcher):
    with TestClient(create_app(make_engine(conn, search_fn=searcher))) as test_client:
        yield test_client


# ------------------------------------------------------------------ POST /search


def test_search_passes_executes_dict_through_verbatim(client):
    resp = client.post("/search", json={"theme": "commanders that look lonely"})
    assert resp.status_code == 200
    body = resp.json()

    # Everything execute() returned is still here, unreshaped.
    assert set(body) == {"query_id", "plan", "relaxed", "results", "pool", "service"}
    assert body["query_id"] == 4242
    result = body["results"][0]
    for key in (
        "oracle_id", "name", "mana_cost", "type_line", "color_identity", "band", "fit",
        "rationale", "verified", "illustration_id", "set_code", "artist", "prop_ids",
        "links", "stretch", "vision_rejected", "verify_note", "art_count",
    ):
        assert key in result, f"{key} was dropped on the way out"
    assert result["links"]["edhrec"].startswith("https://edhrec.com/")
    # pool is kept even though the bot ignores it: it is what makes
    # `curl … | jq '.pool'` worth typing.
    assert body["pool"]


def test_search_adds_only_the_service_block(client):
    body = client.post("/search", json={"theme": "lonely"}).json()
    service = body["service"]
    assert service["index_rebuilt"] is False
    assert service["refresh_running"] is False
    assert service["degraded"] is False
    assert service["queued_seconds"] >= 0.0
    assert service["index_built_at"].endswith("Z")


def test_search_forwards_its_arguments_and_records_kind_user(client, searcher):
    client.post("/search", json={"theme": "  a hooded figure  ", "k": 3, "band": 2,
                                 "colors": "gu"})
    call = searcher.calls[0]
    assert call["query"] == "a hooded figure"       # trimmed
    assert call["k"] == 3
    assert call["band"] == 2
    assert call["colors"] == "UG"                   # upper-cased, WUBRG order
    # 'user', not 'discord': export_training.py filters on kind IN ('user','eval'),
    # so real searches belong in the training exports.
    assert call["kind"] == "user"


def test_search_reuses_the_long_lived_connection_and_index(client, searcher, conn):
    client.post("/search", json={"theme": "lonely"})
    call = searcher.calls[0]
    assert call["conn"] is conn
    assert call["index"] is not None


def test_degraded_is_true_when_the_plan_carries_notes(conn):
    searcher = support.StubSearch(
        result=support.outcome(notes=["ollama unreachable; keyword ranking only"])
    )
    with TestClient(create_app(make_engine(conn, search_fn=searcher))) as client:
        body = client.post("/search", json={"theme": "lonely"}).json()
    assert body["service"]["degraded"] is True


def test_degraded_is_true_when_vision_verification_failed(conn):
    searcher = support.StubSearch(result=support.outcome(vision_verified=False))
    with TestClient(create_app(make_engine(conn, search_fn=searcher))) as client:
        body = client.post("/search", json={"theme": "lonely"}).json()
    assert body["service"]["degraded"] is True


def test_missing_embeddings_are_reported_in_the_plan_notes(conn):
    engine = make_engine(conn, index=support.StubIndex(missing_embeddings=1234))
    with TestClient(create_app(engine)) as client:
        body = client.post("/search", json={"theme": "lonely"}).json()
    assert any("1,234" in note for note in body["plan"]["notes"])
    assert body["service"]["degraded"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {},                                             # no theme
        {"theme": ""},
        {"theme": "   "},
        {"theme": "x" * 301},
        {"theme": "ok", "k": 0},
        {"theme": "ok", "k": 6},                        # capped so results fit one message
        {"theme": "ok", "band": 0},
        {"theme": "ok", "band": 6},
        {"theme": "ok", "colors": "WUBRGX"},
        {"theme": "ok", "colors": "purple"},
    ],
)
def test_bad_requests_are_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_colors_are_accepted_case_insensitively(client, searcher):
    client.post("/search", json={"theme": "ok", "colors": "wubrg"})
    assert searcher.calls[0]["colors"] == "WUBRG"


def test_empty_colors_become_null(client, searcher):
    client.post("/search", json={"theme": "ok", "colors": "  "})
    assert searcher.calls[0]["colors"] is None


def test_a_raising_search_is_a_500_with_a_body(conn):
    searcher = support.StubSearch(raises=RuntimeError("ollama exploded"))
    with TestClient(create_app(make_engine(conn, search_fn=searcher))) as client:
        resp = client.post("/search", json={"theme": "lonely"})
    assert resp.status_code == 500
    assert resp.json() == {"status": "error", "detail": "ollama exploded"}


def test_a_raising_search_does_not_wedge_the_queue(conn):
    """The 500 path has to release the lock and the slot, or one bad search
    turns into a permanently busy service."""
    searcher = support.StubSearch(raises=RuntimeError("boom"))
    engine = make_engine(conn, search_fn=searcher)
    with TestClient(create_app(engine)) as client:
        client.post("/search", json={"theme": "lonely"})
        assert engine._pending == 0
        assert not engine.lock.locked()
        searcher.raises = None
        assert client.post("/search", json={"theme": "lonely"}).status_code == 200


# ------------------------------------------------------- the lock and the queue cap


def test_two_searches_serialise(conn):
    """One at a time. The GPU serialises the work regardless, so a second
    concurrent search would only make both of them finish later."""

    async def scenario():
        searcher = support.StubSearch(delay=0.15)
        engine = make_engine(conn, search_fn=searcher)
        await asyncio.gather(
            engine.search(theme="one", k=5, band=None, colors=None),
            engine.search(theme="two", k=5, band=None, colors=None),
        )
        assert searcher.max_concurrent == 1
        assert len(searcher.calls) == 2
        # The second one waited for the first, and says so.
        assert engine.searches_since_start == 2

    asyncio.run(scenario())


def test_the_queued_seconds_reported_are_the_real_wait(conn):
    async def scenario():
        searcher = support.StubSearch(delay=0.2)
        engine = make_engine(conn, search_fn=searcher)
        first, second = await asyncio.gather(
            engine.search(theme="one", k=5, band=None, colors=None),
            engine.search(theme="two", k=5, band=None, colors=None),
        )
        waits = sorted([first["service"]["queued_seconds"],
                        second["service"]["queued_seconds"]])
        assert waits[0] < 0.1
        assert waits[1] >= 0.15

    asyncio.run(scenario())


def test_the_fifth_request_is_a_503_busy(conn):
    async def scenario():
        searcher = support.StubSearch(delay=0.3)
        engine = make_engine(conn, search_fn=searcher, max_queued=4)
        running = [
            asyncio.create_task(engine.search(theme=str(i), k=5, band=None, colors=None))
            for i in range(4)
        ]
        await asyncio.sleep(0.05)               # let all four claim their slots
        from serve.api import Busy

        with pytest.raises(Busy) as excinfo:
            await engine.search(theme="fifth", k=5, band=None, colors=None)
        assert excinfo.value.queued == 4
        assert excinfo.value.max_queued == 4
        await asyncio.gather(*running)

    asyncio.run(scenario())


def test_the_busy_response_body_over_http(conn):
    searcher = support.StubSearch(delay=1.0)
    engine = make_engine(conn, search_fn=searcher, max_queued=1)
    with TestClient(create_app(engine)) as client:
        pool = threading.Thread(
            target=lambda: client.post("/search", json={"theme": "first"})
        )
        pool.start()
        try:
            assert searcher.started.wait(5), "the first search never started"
            resp = client.post("/search", json={"theme": "second"})
            assert resp.status_code == 503
            assert resp.json() == {"status": "busy", "queued": 1, "max_queued": 1}
        finally:
            pool.join(10)


# --------------------------------------------------------------------- GET /health


def test_health_answers_while_a_search_is_running(conn):
    """The whole reason `execute` runs on a worker thread rather than on the
    event loop. If this regresses, `/health` is unanswerable for ~80s at exactly
    the moment someone wants to know what the service is doing."""
    searcher = support.StubSearch(delay=1.0)
    engine = make_engine(conn, search_fn=searcher)
    with TestClient(create_app(engine)) as client:
        searching = threading.Thread(
            target=lambda: client.post("/search", json={"theme": "slow"})
        )
        searching.start()
        try:
            assert searcher.started.wait(5), "the search never started"
            started = time.monotonic()
            resp = client.get("/health")
            elapsed = time.monotonic() - started

            assert resp.status_code == 200
            assert elapsed < 0.9, f"/health blocked for {elapsed:.2f}s behind the search"
            assert resp.json()["search"]["in_flight"] == 1
        finally:
            searching.join(15)

    assert engine.in_flight == 0


def test_health_reports_the_index_the_models_and_the_corpus(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index"]["props"] == 170_487
    assert body["index"]["artworks"] == 5_530
    assert body["index"]["dim"] == 768
    assert body["index"]["age_seconds"] >= 0
    assert body["index"]["stale"] is False
    assert body["corpus"]["commanders"] == 3202
    assert body["corpus"]["last_refresh_at"] == "2026-08-17T03:39:55Z"
    assert body["ollama"]["reachable"] is True
    assert body["ollama"]["loaded"] == ["judge-model"]
    assert body["refresh"] == {"running": False, "unit": "cts-refresh.service"}
    assert body["search"]["max_queued"] == 4
    assert body["uptime_seconds"] >= 0


def test_health_counts_completed_searches(client):
    assert client.get("/health").json()["search"]["searches_since_start"] == 0
    client.post("/search", json={"theme": "lonely"})
    health = client.get("/health").json()
    assert health["search"]["searches_since_start"] == 1
    assert health["search"]["last_search_seconds"] is not None


def test_health_is_degraded_when_ollama_is_unreachable(conn):
    with TestClient(create_app(make_engine(conn, ollama=support.ollama_down))) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False
    assert body["ollama"]["error"]


def test_health_is_degraded_when_a_model_is_missing(conn):
    def missing():
        info = support.ollama_ok()
        info["missing_models"] = ["judge-model"]
        return info

    with TestClient(create_app(make_engine(conn, ollama=missing))) as client:
        assert client.get("/health").json()["status"] == "degraded"


def test_health_says_refreshing_while_the_refresh_unit_is_active(conn):
    with TestClient(create_app(make_engine(conn, refresh_running=True))) as client:
        body = client.get("/health").json()
    assert body["status"] == "refreshing"
    assert body["refresh"]["running"] is True


def test_the_probes_are_cached_so_polling_health_is_not_a_hot_loop(conn):
    calls = []

    def counting_probe():
        calls.append(1)
        return False

    engine = make_engine(conn)
    engine.refresh = type(engine.refresh)(counting_probe)
    with TestClient(create_app(engine)) as client:
        for _ in range(5):
            client.get("/health")
    assert len(calls) == 1


# --------------------------------------------------------------- POST /admin/reload


def test_admin_reload_forces_a_rebuild(conn):
    builder = support.StubBuilder(indexes=[support.StubIndex(label="forced", props=9)])
    engine = make_engine(conn, builder=builder)
    with TestClient(create_app(engine)) as client:
        body = client.post("/admin/reload").json()
    assert builder.calls == 1
    assert body["index_rebuilt"] is True
    assert body["props"] == 9
    assert engine.index.label == "forced"


def test_admin_reload_reports_a_failed_rebuild_without_dropping_the_index(conn):
    builder = support.StubBuilder(raises=RuntimeError("no"))
    engine = make_engine(conn, builder=builder)
    with TestClient(create_app(engine)) as client:
        body = client.post("/admin/reload").json()
    assert body["index_rebuilt"] is False
    assert body["stale"] is True
    assert engine.index is not None
