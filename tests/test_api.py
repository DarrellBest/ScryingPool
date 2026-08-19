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


def test_health_reports_an_empty_scry_window_before_any_search(client):
    """Fresh install / empty `search_timings`: the fallback case render.py's
    `median_seconds_for` reacts to."""
    body = client.get("/health").json()
    assert body["timings"]["scry"] == {"median_seconds": None, "samples": 0}


def test_health_has_no_oracle_timings_when_this_process_has_no_oracle_corpus(client):
    assert "oracle" not in client.get("/health").json()["timings"]


def test_health_reports_the_scry_median_after_a_search(client):
    client.post("/search", json={"theme": "lonely"})
    timings = client.get("/health").json()["timings"]["scry"]
    assert timings["samples"] == 1
    assert timings["median_seconds"] is not None
    assert timings["median_seconds"] >= 0.0


def test_health_scry_median_reflects_several_completed_searches(client):
    for _ in range(3):
        client.post("/search", json={"theme": "lonely"})
    timings = client.get("/health").json()["timings"]["scry"]
    assert timings["samples"] == 3


def test_health_oracle_median_is_tracked_separately_from_scry(oracle_client):
    """/scry and /oracle write to different databases, so a search on one
    must never move the other's median."""
    oracle_client.post("/oracle/search", json={"query": "cards that draw"})
    body = oracle_client.get("/health").json()
    assert body["timings"]["oracle"]["samples"] == 1
    assert body["timings"]["scry"]["samples"] == 0


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


# ------------------------------------------------------------------------ GET /card
#
# The name-lookup surface. It shares the app, the process and the port with the
# art search and shares nothing else: no lock, no queue slot, no model call, no
# worker thread. These tests assert that separation as much as the shapes.


@pytest.fixture
def oracle_conn():
    connection = support.memory_oracle_conn()
    yield connection
    connection.close()


def make_oracle_engine(
    conn, oracle_conn, *, oracle_builder=None,
    oracle_search_fn=None, oracle_search_index=None, oracle_search_index_builder=None,
    **kwargs,
) -> Engine:
    from cts import oracle_names
    from serve.api import oracle_fingerprint

    engine = make_engine(conn, **kwargs)
    engine.oracle_conn = oracle_conn
    engine.name_index = oracle_names.build_index(oracle_conn)
    engine.oracle_index_fingerprint = oracle_fingerprint(oracle_conn)
    engine.oracle_index_builder = oracle_builder or support.StubOracleBuilder(conn=oracle_conn)
    engine.oracle = type(engine.oracle)(
        lambda: {"cards": 7, "chunks": 0,
                 "last_oracle_refresh_at": "2026-08-17T03:43:02+00:00"}
    )
    engine.oracle_search_fn = oracle_search_fn or support.StubOracleSearch()
    engine.oracle_search_index = oracle_search_index if oracle_search_index is not None else (
        support.StubOracleIndex()
    )
    engine.oracle_search_index_builder = (
        oracle_search_index_builder or support.StubOracleSearchIndexBuilder()
    )
    return engine


@pytest.fixture
def card_client(conn, oracle_conn, searcher):
    engine = make_oracle_engine(conn, oracle_conn, search_fn=searcher)
    with TestClient(create_app(engine)) as test_client:
        test_client.engine = engine
        yield test_client


def test_card_resolves_an_exact_name_and_reports_the_layer(card_client):
    body = card_client.get("/card", params={"name": "Sol Ring"}).json()
    assert body["resolved"] is True
    assert body["layer"] == "L0"
    assert body["input"] == "Sol Ring"
    assert body["card"]["name"] == "Sol Ring"
    assert body["card"]["oracle_text"] == "{T}: Add {C}{C}."
    assert body["candidates"] == []


def test_card_reports_the_layer_that_fired_so_curl_can_debug_the_resolver(card_client):
    for name, layer in (
        ("Sol Ring", "L0"),
        ("gaeas cradle", "L1"),
        ("Petty Theft", "L2"),
        ("ancestral rec", "L3"),
        ("recall ancestral", "L4"),
        ("ancestrl recall", "L5"),
    ):
        body = card_client.get("/card", params={"name": name}).json()
        assert body["layer"] == layer, (name, body["layer"])


def test_card_carries_faces_legalities_and_links(card_client):
    body = card_client.get("/card", params={"name": "Brazen Borrower // Petty Theft"}).json()
    card = body["card"]
    assert [face["name"] for face in card["faces"]] == ["Brazen Borrower", "Petty Theft"]
    assert card["legalities"] == {"commander": "legal"}
    assert card["links"]["scryfall"].startswith("https://scryfall.com/")
    assert card["links"]["edhrec"].startswith("https://edhrec.com/")
    # No TCGplayer URL was stored for this fixture, so there is no key for it.
    assert "tcgplayer" not in card["links"]


def test_card_returns_200_with_candidates_for_an_ambiguous_name(card_client):
    """The query was well-formed and the answer is "several" — that is a 200 with
    `resolved: false`, not a 404. A 404 would say the request was wrong."""
    response = card_client.get("/card", params={"name": "path"})
    assert response.status_code == 200
    body = response.json()
    assert body["resolved"] is False
    assert body["total"] == 2
    assert [c["name"] for c in body["candidates"]] == ["Path of Ancestry", "Path to Exile"]
    assert body["card"] is None


def test_card_returns_200_with_no_candidates_for_a_genuine_miss(card_client):
    response = card_client.get("/card", params={"name": "qwertyuiop asdfghjkl"})
    assert response.status_code == 200
    body = response.json()
    assert body["resolved"] is False
    assert body["candidates"] == []
    assert body["layer"] is None


def test_card_never_corrects_an_exact_name_into_a_more_popular_neighbour(card_client):
    """The property the whole ladder exists for, asserted through the HTTP surface
    as well as the resolver: Ancestral Vision is three times more played and two
    edits away, and Recall still resolves to Recall."""
    body = card_client.get("/card", params={"name": "Ancestral Recall"}).json()
    assert body["layer"] == "L0"
    assert body["card"]["name"] == "Ancestral Recall"


def test_card_rejects_a_blank_name_with_a_422(card_client):
    assert card_client.get("/card", params={"name": "   "}).status_code == 422
    assert card_client.get("/card").status_code == 422


def test_card_reports_how_it_was_resolved_and_when_the_corpus_was_refreshed(card_client):
    service = card_client.get("/card", params={"name": "Sol Ring"}).json()["service"]
    assert service["refreshed_at"] == "2026-08-17T03:43:02+00:00"
    assert service["oracle_index_stale"] is False
    assert service["elapsed_ms"] >= 0


def test_card_answers_while_a_search_holds_the_lock(conn, oracle_conn):
    """`/search` takes no search lock and no queue slot: it makes zero model calls
    and contends for nothing a `/scry` is using. A name lookup queued behind an
    80-second art search would be absurd."""
    searcher = support.StubSearch(delay=0.4)
    engine = make_oracle_engine(conn, oracle_conn, search_fn=searcher)
    with TestClient(create_app(engine)) as client:
        started = threading.Event()
        done = threading.Event()

        def run_search():
            client.post("/search", json={"theme": "lonely"})
            done.set()

        worker = threading.Thread(target=run_search)
        worker.start()
        assert searcher.started.wait(5)
        started.set()

        # The search is in flight and holding the lock right now.
        begin = time.monotonic()
        response = client.get("/card", params={"name": "Sol Ring"})
        elapsed = time.monotonic() - begin

        worker.join(10)
        assert done.is_set()

    assert response.status_code == 200
    assert response.json()["card"]["name"] == "Sol Ring"
    assert elapsed < 0.3, f"the lookup waited {elapsed:.3f}s — it queued behind the search"


def test_card_does_not_count_against_the_search_queue_cap(conn, oracle_conn):
    engine = make_oracle_engine(conn, oracle_conn)
    with TestClient(create_app(engine)) as client:
        for _ in range(10):
            assert client.get("/card", params={"name": "Sol Ring"}).status_code == 200
        assert engine.queued == 0
        assert engine.in_flight == 0
        assert client.get("/health").json()["search"]["searches_since_start"] == 0


def test_card_rebuilds_the_index_once_on_a_miss_and_never_on_a_hit(conn, oracle_conn):
    """Staleness is paid for only where it could be the cause. A name that
    resolves to nothing might be a card from Sunday's set release; a name that
    resolved is not, and blocking that path for a rebuild would destroy the one
    property that makes this command pleasant."""
    builder = support.StubOracleBuilder(conn=oracle_conn)
    engine = make_oracle_engine(conn, oracle_conn, oracle_builder=builder)
    with TestClient(create_app(engine)) as client:
        client.get("/card", params={"name": "Sol Ring"})
        assert builder.calls == 0, "a successful lookup must not rebuild anything"

        client.get("/card", params={"name": "no such card at all"})
        assert builder.calls == 0, "the fingerprint had not moved, so nothing to rebuild"

        # Now something lands in the corpus that the resident index has never seen.
        oracle_conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm) VALUES ('o-new', 'Nadu, Winged Wisdom', "
            "'nadu winged wisdom')"
        )
        oracle_conn.execute(
            "INSERT INTO chunks(id, oracle_id, face_index, ordinal, kind, text) "
            "VALUES (1, 'o-new', 0, 0, 'whole', 'x')"
        )
        oracle_conn.commit()

        body = client.get("/card", params={"name": "Nadu, Winged Wisdom"}).json()
        assert builder.calls == 1
        assert body["resolved"] is True
        assert body["card"]["name"] == "Nadu, Winged Wisdom"


def test_card_survives_a_failed_rebuild_by_serving_the_index_it_has(conn, oracle_conn):
    builder = support.StubOracleBuilder(raises=RuntimeError("disk on fire"))
    engine = make_oracle_engine(conn, oracle_conn, oracle_builder=builder)
    with TestClient(create_app(engine)) as client:
        oracle_conn.execute(
            "INSERT INTO chunks(id, oracle_id, face_index, ordinal, kind, text) "
            "VALUES (1, 'o-sol', 0, 0, 'whole', 'x')"
        )
        oracle_conn.commit()
        miss = client.get("/card", params={"name": "not a card"}).json()
        assert miss["resolved"] is False
        assert engine.oracle_stale is True
        # The old index is still serving.
        assert client.get("/card", params={"name": "Sol Ring"}).json()["resolved"] is True


def test_card_works_when_its_connection_was_opened_on_another_thread(conn, tmp_path):
    """`build_engine` runs under `asyncio.to_thread`, so the resident oracle
    connection is **created on a worker thread and used from the event loop**.

    A connection opened with sqlite3's default same-thread guard raises
    ProgrammingError on the first `/card` in production while every in-process
    test passes, because the tests construct their engines on the main thread.
    That is exactly what happened; this is the test that would have caught it.
    """
    from cts import oracle_names
    from serve.api import open_oracle_connection, oracle_fingerprint

    cfg = support.config()
    cfg = type(cfg)(**{**cfg.__dict__, "oracle_db_path": str(tmp_path / "oracle.db")})

    opened: list = []

    def open_it():
        connection = open_oracle_connection(cfg)
        connection.execute(
            "INSERT INTO cards(oracle_id, name, name_norm) VALUES "
            "('o-sol', 'Sol Ring', 'sol ring')"
        )
        connection.commit()
        opened.append(connection)

    worker = threading.Thread(target=open_it)
    worker.start()
    worker.join(10)
    oracle_connection = opened[0]

    try:
        engine = make_engine(conn)
        engine.cfg = cfg
        engine.oracle_conn = oracle_connection
        engine.name_index = oracle_names.build_index(oracle_connection)
        engine.oracle_index_fingerprint = oracle_fingerprint(oracle_connection)
        with TestClient(create_app(engine)) as client:
            body = client.get("/card", params={"name": "Sol Ring"}).json()
        assert body["resolved"] is True
        assert body["card"]["name"] == "Sol Ring"
    finally:
        oracle_connection.close()


def test_card_says_the_corpus_is_missing_rather_than_no_such_card(conn):
    """An empty corpus and a misspelled name are different problems with
    different fixes, and collapsing them would tell a user to check their
    spelling when the real answer is that nobody ran the ingest."""
    engine = make_engine(conn)          # no oracle corpus at all
    with TestClient(create_app(engine)) as client:
        response = client.get("/card", params={"name": "Sol Ring"})
    assert response.status_code == 503
    assert "oracle corpus" in response.json()["detail"]


def test_health_gains_an_oracle_block_reported_separately_from_the_art_index(card_client):
    """"The art index is stale" and "the oracle index is stale" are different
    problems with different causes, so they are two blocks and not one number."""
    body = card_client.get("/health").json()
    assert body["index"]["props"] == 170_487           # unchanged
    oracle = body["oracle"]
    assert oracle["cards"] == len(support.ORACLE_CARDS)
    assert oracle["names"] >= oracle["cards"]
    assert oracle["stale"] is False
    assert oracle["last_oracle_refresh_at"] == "2026-08-17T03:43:02+00:00"
    assert oracle["age_seconds"] >= 0


def test_health_has_no_oracle_block_when_this_process_has_no_oracle_corpus(client):
    assert "oracle" not in client.get("/health").json()


def test_admin_reload_can_name_which_index_to_rebuild(conn, oracle_conn):
    art = support.StubBuilder()
    oracle = support.StubOracleBuilder(conn=oracle_conn)
    engine = make_oracle_engine(conn, oracle_conn, builder=art, oracle_builder=oracle)
    with TestClient(create_app(engine)) as client:
        client.post("/admin/reload", params={"index": "art"})
        assert (art.calls, oracle.calls) == (1, 0)

        client.post("/admin/reload", params={"index": "oracle"})
        assert (art.calls, oracle.calls) == (1, 1)

        body = client.post("/admin/reload", params={"index": "all"}).json()
        assert (art.calls, oracle.calls) == (2, 2)
        assert body["index_rebuilt"] is True
        assert body["oracle_index_rebuilt"] is True

        assert client.post("/admin/reload", params={"index": "sideways"}).status_code == 422


def test_admin_reload_defaults_to_all(conn, oracle_conn):
    art = support.StubBuilder()
    oracle = support.StubOracleBuilder(conn=oracle_conn)
    engine = make_oracle_engine(conn, oracle_conn, builder=art, oracle_builder=oracle)
    with TestClient(create_app(engine)) as client:
        body = client.post("/admin/reload").json()
    assert (art.calls, oracle.calls) == (1, 1)
    assert body["index"] == "all"


def test_one_poll_tick_checks_both_fingerprints(conn, oracle_conn):
    art = support.StubBuilder()
    oracle = support.StubOracleBuilder(conn=oracle_conn)
    engine = make_oracle_engine(conn, oracle_conn, builder=art, oracle_builder=oracle)

    async def scenario():
        assert await engine.poll_once() == "current"
        assert (art.calls, oracle.calls) == (0, 0)

        # Only the oracle fingerprint moves.
        oracle_conn.execute(
            "INSERT INTO chunks(id, oracle_id, face_index, ordinal, kind, text) "
            "VALUES (1, 'o-sol', 0, 0, 'whole', 'x')"
        )
        oracle_conn.commit()
        assert await engine.poll_once() == "current"     # the ART index is current
        assert (art.calls, oracle.calls) == (0, 1)       # ...and the oracle one rebuilt

    asyncio.run(scenario())


# ============================================================================
# POST /oracle/search, POST /oracle/feedback
# ============================================================================


@pytest.fixture
def oracle_searcher():
    return support.StubOracleSearch()


@pytest.fixture
def oracle_client(conn, oracle_conn, oracle_searcher):
    engine = make_oracle_engine(conn, oracle_conn, oracle_search_fn=oracle_searcher)
    with TestClient(create_app(engine)) as test_client:
        test_client.engine = engine
        yield test_client


def test_oracle_search_passes_executes_dict_through_verbatim(oracle_client):
    resp = oracle_client.post("/oracle/search", json={"query": "cards that draw"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"query_id", "plan", "results", "pool", "message", "service"}
    assert body["query_id"] == support.ORACLE_OUTCOME["query_id"]


def test_oracle_search_forwards_filters_to_the_search_function(oracle_client, oracle_searcher):
    oracle_client.post(
        "/oracle/search",
        json={"query": "cards that draw", "types": ["enchantment"], "colors": "g",
              "mv_max": 5, "k": 3},
    )
    call = oracle_searcher.calls[0]
    assert call["types"] == ("enchantment",)
    assert call["colors"] == "G"
    assert call["mv_max"] == 5
    assert call["k"] == 3
    assert call["kind"] == "user"


def test_oracle_search_400s_when_mv_min_exceeds_mv_max(oracle_client):
    resp = oracle_client.post(
        "/oracle/search", json={"query": "x", "mv_min": 6, "mv_max": 3}
    )
    assert resp.status_code == 422


def test_oracle_search_rejects_mv_outside_zero_to_thirty(oracle_client):
    assert oracle_client.post("/oracle/search", json={"query": "x", "mv_max": 999}).status_code == 422


def test_oracle_search_colors_are_validated_the_same_way_search_is(oracle_client):
    resp = oracle_client.post("/oracle/search", json={"query": "x", "colors": "z"})
    assert resp.status_code == 422


def test_oracle_search_503s_when_no_oracle_corpus_is_configured(conn):
    engine = make_engine(conn)   # no oracle corpus at all
    with TestClient(create_app(engine)) as client:
        resp = client.post("/oracle/search", json={"query": "cards that draw"})
    assert resp.status_code == 503


def test_oracle_search_and_scry_share_the_same_lock_and_queue(conn, oracle_conn):
    """The design's explicit decision: one shared lock, not two — an /oracle
    search queued behind a /scry (and vice versa) is the correct behaviour,
    not a bug, because both contend for the same Ollama instance."""
    art_searcher = support.StubSearch(delay=0.3)
    oracle_searcher = support.StubOracleSearch()
    engine = make_oracle_engine(
        conn, oracle_conn, search_fn=art_searcher, oracle_search_fn=oracle_searcher
    )
    with TestClient(create_app(engine)) as client:
        started = threading.Event()

        def run_scry():
            client.post("/search", json={"theme": "lonely"})

        worker = threading.Thread(target=run_scry)
        worker.start()
        assert art_searcher.started.wait(5)
        started.set()

        begin = time.monotonic()
        resp = client.post("/oracle/search", json={"query": "cards that draw"})
        elapsed = time.monotonic() - begin
        worker.join(10)

    assert resp.status_code == 200
    assert elapsed >= 0.2, "the /oracle search must have waited behind the /scry search"


def test_oracle_search_is_a_503_busy_when_the_shared_queue_is_full(conn, oracle_conn):
    slow = support.StubOracleSearch(delay=0.3)
    engine = make_oracle_engine(conn, oracle_conn, oracle_search_fn=slow, max_queued=1)
    with TestClient(create_app(engine)) as client:
        worker = threading.Thread(
            target=lambda: client.post("/oracle/search", json={"query": "x"})
        )
        worker.start()
        assert slow.started.wait(5)
        resp = client.post("/oracle/search", json={"query": "y"})
        worker.join(10)
    assert resp.status_code == 503
    assert resp.json()["status"] == "busy"


def test_oracle_search_rebuilds_the_index_when_the_fingerprint_moved(conn, oracle_conn):
    builder = support.StubOracleBuilder(conn=oracle_conn)
    search_builder = support.StubOracleSearchIndexBuilder(
        indexes=[support.StubOracleIndex(label="rebuilt")]
    )
    engine = make_oracle_engine(
        conn, oracle_conn, oracle_builder=builder, oracle_search_index_builder=search_builder,
    )
    with TestClient(create_app(engine)) as client:
        oracle_conn.execute(
            "INSERT INTO chunks(id, oracle_id, face_index, ordinal, kind, text) "
            "VALUES (1, 'o-sol', 0, 0, 'whole', 'x')"
        )
        oracle_conn.commit()
        resp = client.post("/oracle/search", json={"query": "cards that draw"})
    assert resp.status_code == 200
    assert resp.json()["service"]["oracle_index_rebuilt"] is True
    assert search_builder.calls == 1
    assert engine.oracle_search_index.label == "rebuilt"


def test_a_failed_search_index_rebuild_does_not_block_the_name_index(conn, oracle_conn):
    """The two rebuilds are independent: a broken chunk index must not stop
    /card's name resolution from picking up fresh data."""
    search_builder = support.StubOracleSearchIndexBuilder(raises=RuntimeError("disk on fire"))
    engine = make_oracle_engine(conn, oracle_conn, oracle_search_index_builder=search_builder)
    with TestClient(create_app(engine)) as client:
        oracle_conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm) VALUES ('o-new', 'Nadu', 'nadu')"
        )
        oracle_conn.execute(
            "INSERT INTO chunks(id, oracle_id, face_index, ordinal, kind, text) "
            "VALUES (1, 'o-new', 0, 0, 'whole', 'x')"
        )
        oracle_conn.commit()
        body = client.get("/card", params={"name": "Nadu"}).json()
    assert body["resolved"] is True
    assert engine.oracle_stale is False    # the name index rebuilt fine
    assert search_builder.calls == 1


@pytest.fixture
def feedback_client(tmp_path):
    """`write_oracle_feedback` opens its own connection on `cfg.oracle_db_path`
    (mirroring `write_feedback`), so this needs a real file on disk — an
    in-memory `oracle_conn` shared only with the resolver would not be visible
    to it. Same convention `tests/test_feedback.py` already uses for `/scry`."""
    from cts import oracle_db as odb

    cfg = support.config()
    cfg = type(cfg)(**{**cfg.__dict__, "oracle_db_path": str(tmp_path / "oracle.db")})
    oconn = odb.connect(cfg)
    oconn.execute(
        "INSERT INTO cards(oracle_id, name, name_norm) VALUES ('o-sol', 'Sol Ring', 'sol ring')"
    )
    oconn.commit()

    conn = support.memory_conn()
    engine = make_oracle_engine(conn, oconn, oracle_search_fn=support.StubOracleSearch())
    engine.cfg = cfg
    with TestClient(create_app(engine)) as test_client:
        test_client.oconn = oconn
        yield test_client
    conn.close()
    oconn.close()


def test_oracle_feedback_records_a_discord_vote(feedback_client):
    feedback_client.oconn.execute("INSERT INTO queries(id, text, kind) VALUES (55, 'x', 'user')")
    feedback_client.oconn.commit()
    resp = feedback_client.post(
        "/oracle/feedback",
        json={"query_id": 55, "oracle_id": "o-sol", "accepted": True, "discord_user_id": "123"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    row = feedback_client.oconn.execute(
        "SELECT fit, source FROM judgments WHERE query_id = 55 AND oracle_id = 'o-sol'"
    ).fetchone()
    assert row["fit"] == 1.0
    assert row["source"] == "discord"


def test_oracle_feedback_is_idempotent(feedback_client):
    feedback_client.oconn.execute("INSERT INTO queries(id, text, kind) VALUES (56, 'x', 'user')")
    feedback_client.oconn.commit()
    payload = {"query_id": 56, "oracle_id": "o-sol", "accepted": True}
    feedback_client.post("/oracle/feedback", json=payload)
    resp = feedback_client.post("/oracle/feedback", json={**payload, "accepted": False})
    assert resp.json()["replaced"] is True
    rows = feedback_client.oconn.execute(
        "SELECT fit FROM judgments WHERE query_id = 56 AND oracle_id = 'o-sol' AND source = 'discord'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["fit"] == 0.0


def test_oracle_feedback_404s_for_an_unknown_query(feedback_client):
    resp = feedback_client.post(
        "/oracle/feedback", json={"query_id": 999999, "oracle_id": "o-sol", "accepted": True}
    )
    assert resp.status_code == 404


def test_health_oracle_block_gains_chunk_index_stats(oracle_client):
    body = oracle_client.get("/health").json()
    oracle = body["oracle"]
    assert oracle["chunks"] == support.StubOracleIndex().chunks
    assert oracle["dim"] == support.StubOracleIndex().dim
    assert oracle["missing_embeddings"] == 0


def test_admin_reload_oracle_rebuilds_the_search_index_too(conn, oracle_conn):
    name_builder = support.StubOracleBuilder(conn=oracle_conn)
    search_builder = support.StubOracleSearchIndexBuilder()
    engine = make_oracle_engine(
        conn, oracle_conn, oracle_builder=name_builder, oracle_search_index_builder=search_builder,
    )
    with TestClient(create_app(engine)) as client:
        body = client.post("/admin/reload", params={"index": "oracle"}).json()
    assert name_builder.calls == 1
    assert search_builder.calls == 1
    assert body["oracle_chunks"] == support.StubOracleIndex(label="build-1").chunks
