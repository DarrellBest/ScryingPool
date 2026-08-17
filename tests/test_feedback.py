"""`POST /feedback` writes the one row the eval harness almost never gets.

`judgments` with `source='discord'` is human-marked training data arriving as a
side effect of people using the thing, and `export_training.py` already reads
that table. The row therefore has to match what `cts/evaluate.py::_write_mark`
writes for `source='human'` in every column but `source` itself.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient                          # noqa: E402

import serve_support as support                                    # noqa: E402
from cts import db                                                 # noqa: E402
from serve.api import Engine, corpus_fingerprint, create_app, write_feedback  # noqa: E402

JUDGE_PROPS = [101, 102, 103]


@pytest.fixture
def cfg(tmp_path):
    """A real on-disk database with one query and one judged result in it."""
    config = support.config(db_path=str(tmp_path / "feedback.db"))
    conn = db.connect(config)
    conn.execute(
        "INSERT INTO queries(id, text, kind, params, created_at) "
        "VALUES (7, 'a hooded figure alone at dusk', 'user', '{}', '2026-08-17T04:00:00Z')"
    )
    conn.execute(
        "INSERT INTO judgments(query_id, illustration_id, fit, rationale, prop_ids, "
        "model, source) VALUES (7, 'ill-avacyn', 0.82, 'the judge said so', ?, "
        "'judge-model', 'judge')",
        (json.dumps(JUDGE_PROPS),),
    )
    conn.commit()
    conn.close()
    return config


def _rows(config, source="discord"):
    conn = db.connect(config)
    try:
        return conn.execute(
            "SELECT * FROM judgments WHERE source = ? ORDER BY rowid", (source,)
        ).fetchall()
    finally:
        conn.close()


def test_an_accepted_vote_writes_one_row_shaped_like_write_mark(cfg):
    result = write_feedback(
        cfg,
        query_id=7,
        illustration_id="ill-avacyn",
        accepted=True,
        discord_user_id="1234567890",
    )
    assert result == {"found": True, "replaced": False}

    rows = _rows(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row["query_id"] == 7
    assert row["illustration_id"] == "ill-avacyn"
    assert row["fit"] == 1.0
    assert row["model"] == ""                # a human is not a model
    assert row["source"] == "discord"        # the one field that differs from eval's marks
    # prop_ids is not in the request; it is read back off the judge's own row,
    # which is where execute() put the identical list.
    assert json.loads(row["prop_ids"]) == JUDGE_PROPS
    assert "1234567890" in row["rationale"]
    assert "acceptable" in row["rationale"]


def test_a_rejecting_vote_writes_fit_zero_and_says_so(cfg):
    write_feedback(cfg, query_id=7, illustration_id="ill-avacyn", accepted=False,
                   discord_user_id="9")
    row = _rows(cfg)[0]
    assert row["fit"] == 0.0
    assert "not acceptable" in row["rationale"]


def test_changing_your_mind_replaces_rather_than_duplicating(cfg):
    """`judgments` has no unique constraint, so without the delete a 👍 then a 👎
    would leave two contradictory training rows for the same result."""
    first = write_feedback(cfg, query_id=7, illustration_id="ill-avacyn",
                           accepted=True, discord_user_id="9")
    second = write_feedback(cfg, query_id=7, illustration_id="ill-avacyn",
                            accepted=False, discord_user_id="9")
    assert first["replaced"] is False
    assert second["replaced"] is True

    rows = _rows(cfg)
    assert len(rows) == 1
    assert rows[0]["fit"] == 0.0             # the latest vote wins


def test_replacing_a_vote_never_touches_the_judge_rows(cfg):
    write_feedback(cfg, query_id=7, illustration_id="ill-avacyn", accepted=True,
                   discord_user_id="9")
    write_feedback(cfg, query_id=7, illustration_id="ill-avacyn", accepted=False,
                   discord_user_id="9")
    judge_rows = _rows(cfg, source="judge")
    assert len(judge_rows) == 1
    assert judge_rows[0]["fit"] == 0.82


def test_votes_on_different_results_of_one_query_coexist(cfg):
    write_feedback(cfg, query_id=7, illustration_id="ill-avacyn", accepted=True,
                   discord_user_id="9")
    write_feedback(cfg, query_id=7, illustration_id="ill-other", accepted=False,
                   discord_user_id="9")
    assert len(_rows(cfg)) == 2


def test_an_unknown_query_id_writes_nothing(cfg):
    """Buttons are persistent across bot restarts, so a stale tap is normal."""
    result = write_feedback(cfg, query_id=99999, illustration_id="ill-avacyn",
                            accepted=True, discord_user_id="9")
    assert result == {"found": False, "replaced": False}
    assert _rows(cfg) == []


def test_a_result_with_no_judge_row_still_records_with_empty_prop_ids(cfg):
    write_feedback(cfg, query_id=7, illustration_id="ill-never-judged", accepted=True,
                   discord_user_id=None)
    row = _rows(cfg)[0]
    assert json.loads(row["prop_ids"]) == []
    assert "a discord user" in row["rationale"]


# ------------------------------------------------------------------- over HTTP


def _client(cfg):
    conn = support.memory_conn()
    engine = Engine(
        cfg,
        conn,
        support.StubIndex(),
        corpus_fingerprint(conn),
        search_fn=support.StubSearch(),
        index_builder=support.StubBuilder(),
        refresh_probe=lambda: False,
        ollama_probe=support.ollama_ok,
        corpus_stats=lambda: {"commanders": 3202, "last_refresh_at": None},
        poll_seconds=0,
    )
    return TestClient(create_app(engine)), conn


def test_the_endpoint_returns_ok_and_the_replaced_flag(cfg):
    client, conn = _client(cfg)
    try:
        with client:
            body = {"query_id": 7, "illustration_id": "ill-avacyn", "accepted": True,
                    "discord_user_id": "1234567890"}
            first = client.post("/feedback", json=body)
            assert first.status_code == 200
            assert first.json() == {"ok": True, "replaced": False}

            second = client.post("/feedback", json=body)
            assert second.json() == {"ok": True, "replaced": True}
        assert len(_rows(cfg)) == 1
    finally:
        conn.close()


def test_the_endpoint_404s_an_unknown_query_id(cfg):
    client, conn = _client(cfg)
    try:
        with client:
            resp = client.post(
                "/feedback",
                json={"query_id": 12345, "illustration_id": "ill-avacyn", "accepted": True},
            )
        assert resp.status_code == 404
        assert resp.json()["status"] == "error"
        assert _rows(cfg) == []
    finally:
        conn.close()


def test_the_endpoint_validates_its_body(cfg):
    client, conn = _client(cfg)
    try:
        with client:
            assert client.post("/feedback", json={"query_id": 7}).status_code == 422
            assert client.post(
                "/feedback",
                json={"query_id": 7, "illustration_id": "", "accepted": True},
            ).status_code == 422
    finally:
        conn.close()
