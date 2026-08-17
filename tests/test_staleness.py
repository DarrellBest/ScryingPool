"""The corpus fingerprint and the rebuild it drives.

This is the heaviest of the serving tests because it is where the risk is. The
failure it prevents is silent: the API holds an index built at process start,
the weekly refresh writes new props and embeddings underneath it, and nothing
errors — last week's corpus is served indefinitely, with plausible-looking
results, forever.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

import serve_support as support                                    # noqa: E402
from serve.api import Engine, corpus_fingerprint                   # noqa: E402


def _engine(conn, *, builder=None, refresh_running=False, index=None, **kwargs) -> Engine:
    builder = builder or support.StubBuilder()
    return Engine(
        support.config(),
        conn,
        index or support.StubIndex(label="boot"),
        corpus_fingerprint(conn),
        search_fn=kwargs.pop("search_fn", support.StubSearch()),
        index_builder=builder,
        refresh_probe=lambda: refresh_running,
        ollama_probe=support.ollama_ok,
        corpus_stats=lambda: {"commanders": 3202, "last_refresh_at": None},
        poll_seconds=0,          # the tests drive poll_once() themselves
        **kwargs,
    )


def _add_prop(conn, prop_id: int) -> None:
    conn.execute(
        "INSERT INTO props(id, illustration_id, layer, text) VALUES (?, 'ill-1', 'literal', 'x')",
        (prop_id,),
    )
    conn.commit()


# ------------------------------------------------------------------ the pure function


def test_fingerprint_of_an_empty_corpus_is_all_nulls():
    conn = support.memory_conn()
    try:
        assert corpus_fingerprint(conn) == (None, None, None)
    finally:
        conn.close()


def test_fingerprint_moves_when_props_arrive():
    conn = support.memory_conn()
    try:
        before = corpus_fingerprint(conn)
        _add_prop(conn, 1)
        after = corpus_fingerprint(conn)
        assert after != before
        assert after[1] == 1
    finally:
        conn.close()


def test_fingerprint_moves_again_when_the_embed_stage_catches_up():
    """Props first, vectors later. An index built between the two is incomplete."""
    conn = support.memory_conn()
    try:
        _add_prop(conn, 1)
        props_only = corpus_fingerprint(conn)
        conn.execute("INSERT INTO embeddings(prop_id, vec) VALUES (1, X'00')")
        conn.commit()
        embedded = corpus_fingerprint(conn)
        assert embedded != props_only
        assert embedded[2] == 1
    finally:
        conn.close()


def test_fingerprint_does_not_move_when_nothing_changes():
    conn = support.memory_conn()
    try:
        _add_prop(conn, 1)
        assert corpus_fingerprint(conn) == corpus_fingerprint(conn)
        # An unrelated write is not a corpus change either.
        conn.execute("INSERT INTO meta(key, value) VALUES ('scryfall_updated_at', 'z')")
        conn.commit()
        first = corpus_fingerprint(conn)
        conn.execute("INSERT INTO cards(oracle_id, name) VALUES ('o1', 'Card')")
        conn.commit()
        assert corpus_fingerprint(conn) == first
    finally:
        conn.close()


def test_fingerprint_moves_when_refresh_stamps_meta():
    """A refresh that added no props still moved EDHREC data and power scores."""
    conn = support.memory_conn()
    try:
        before = corpus_fingerprint(conn)
        conn.execute("INSERT INTO meta(key, value) VALUES ('last_refresh_at', '2026-08-17T03:39:55Z')")
        conn.commit()
        after = corpus_fingerprint(conn)
        assert after != before
        assert after[0] == "2026-08-17T03:39:55Z"
    finally:
        conn.close()


# ------------------------------------------------------------------ the search path


def test_a_moved_fingerprint_rebuilds_before_the_search_runs():
    async def scenario():
        conn = support.memory_conn()
        try:
            builder = support.StubBuilder(indexes=[support.StubIndex(label="rebuilt")])
            searcher = support.StubSearch()
            engine = _engine(conn, builder=builder, search_fn=searcher)

            _add_prop(conn, 1)
            outcome = await engine.search(theme="lonely", k=5, band=None, colors=None)

            assert builder.calls == 1
            assert outcome["service"]["index_rebuilt"] is True
            # The search must have run against the *new* index, not the boot one.
            assert searcher.calls[0]["index"].label == "rebuilt"
            assert engine.index.label == "rebuilt"
            assert engine.index_stale is False
        finally:
            conn.close()

    asyncio.run(scenario())


def test_an_unchanged_fingerprint_does_not_rebuild():
    async def scenario():
        conn = support.memory_conn()
        try:
            builder = support.StubBuilder()
            engine = _engine(conn, builder=builder)
            outcome = await engine.search(theme="lonely", k=5, band=None, colors=None)
            assert builder.calls == 0
            assert outcome["service"]["index_rebuilt"] is False
            assert engine.index.label == "boot"
        finally:
            conn.close()

    asyncio.run(scenario())


def test_a_failed_rebuild_keeps_the_old_index_marks_stale_and_still_serves():
    async def scenario():
        conn = support.memory_conn()
        try:
            builder = support.StubBuilder(raises=RuntimeError("disk on fire"))
            searcher = support.StubSearch()
            engine = _engine(conn, builder=builder, search_fn=searcher)

            _add_prop(conn, 1)
            outcome = await engine.search(theme="lonely", k=5, band=None, colors=None)

            assert builder.calls == 1
            assert engine.index.label == "boot"        # never left without an index
            assert engine.index_stale is True
            assert outcome["service"]["index_rebuilt"] is False
            assert outcome["results"], "the search still has to return results"

            # ...and the next search tries again.
            builder.raises = None
            builder.indexes = [support.StubIndex(label="recovered")]
            await engine.search(theme="lonely", k=5, band=None, colors=None)
            assert engine.index.label == "recovered"
            assert engine.index_stale is False
        finally:
            conn.close()

    asyncio.run(scenario())


def test_health_reports_the_stale_flag():
    async def scenario():
        conn = support.memory_conn()
        try:
            engine = _engine(conn, builder=support.StubBuilder(raises=RuntimeError("nope")))
            _add_prop(conn, 1)
            await engine.search(theme="lonely", k=5, band=None, colors=None)
            health = await engine.health()
            assert health["index"]["stale"] is True
        finally:
            conn.close()

    asyncio.run(scenario())


# -------------------------------------------------------------- the background poll


def test_the_poll_rebuilds_when_the_fingerprint_moved():
    async def scenario():
        conn = support.memory_conn()
        try:
            builder = support.StubBuilder(indexes=[support.StubIndex(label="polled")])
            engine = _engine(conn, builder=builder)
            assert await engine.poll_once() == "current"
            _add_prop(conn, 1)
            assert await engine.poll_once() == "rebuilt"
            assert engine.index.label == "polled"
        finally:
            conn.close()

    asyncio.run(scenario())


def test_the_poll_skips_its_tick_while_a_search_holds_the_lock():
    async def scenario():
        conn = support.memory_conn()
        try:
            builder = support.StubBuilder()
            engine = _engine(conn, builder=builder)
            _add_prop(conn, 1)
            async with engine.lock:
                assert await engine.poll_once() == "skipped"
            assert builder.calls == 0
            # The synchronous per-search check is the guarantee; the poll is only
            # latency hiding, so skipping costs correctness nothing.
            assert await engine.poll_once() == "rebuilt"
        finally:
            conn.close()

    asyncio.run(scenario())


def test_the_poll_debounces_to_one_rebuild_per_five_minutes():
    async def scenario():
        conn = support.memory_conn()
        try:
            engine = _engine(conn, builder=support.StubBuilder())
            _add_prop(conn, 1)
            assert await engine.poll_once() == "rebuilt"

            # An embed stage committing batches moves the fingerprint again.
            _add_prop(conn, 2)
            assert await engine.poll_once() == "debounced"

            # The debounce is a wall-clock window, not a one-shot latch.
            engine.last_poll_rebuild -= engine.poll_debounce_seconds + 1
            assert await engine.poll_once() == "rebuilt"
        finally:
            conn.close()

    asyncio.run(scenario())


def test_the_poll_is_suppressed_while_the_refresh_unit_is_active():
    async def scenario():
        conn = support.memory_conn()
        try:
            builder = support.StubBuilder()
            engine = _engine(conn, builder=builder, refresh_running=True)
            _add_prop(conn, 1)
            assert await engine.poll_once() == "suppressed"
            assert builder.calls == 0

            # But a search issued mid-refresh is never suppressed: it still gets
            # a current index, which is what makes the guarantee unconditional.
            outcome = await engine.search(theme="lonely", k=5, band=None, colors=None)
            assert builder.calls == 1
            assert outcome["service"]["index_rebuilt"] is True
            assert outcome["service"]["refresh_running"] is True
        finally:
            conn.close()

    asyncio.run(scenario())


def test_an_unknown_systemctl_answer_is_null_not_false():
    async def scenario():
        conn = support.memory_conn()
        try:
            engine = _engine(conn)
            engine.refresh = type(engine.refresh)(lambda: None)
            health = await engine.health()
            assert health["refresh"]["running"] is None
        finally:
            conn.close()

    asyncio.run(scenario())


# ------------------------------------------------------------------- forced reload


def test_admin_reload_rebuilds_even_when_the_fingerprint_has_not_moved():
    """The escape hatch for what the fingerprint is blind to: a hand-edited
    props table, a restored database, a cleared-and-re-embedded corpus."""

    async def scenario():
        conn = support.memory_conn()
        try:
            builder = support.StubBuilder(indexes=[support.StubIndex(label="forced")])
            engine = _engine(conn, builder=builder)
            async with engine.lock:
                assert await engine.ensure_current(force=True) is True
            assert builder.calls == 1
            assert engine.index.label == "forced"
        finally:
            conn.close()

    asyncio.run(scenario())
