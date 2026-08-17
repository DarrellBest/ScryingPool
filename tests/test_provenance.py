"""Discord votes are human votes, everywhere that asks.

The bot writes `judgments` with `source='discord'`; `cts eval` writes the same
row with `source='human'`. The two values record where the person was sitting
and nothing else, so every consumer that asks "did a person say this?" has to
answer yes for both. It did not: `'discord'` was absent from the dedupe table
(so the judge's own verdict OUTRANKED the user's correction, exactly backwards),
absent from the training weight, and filtered out of the eval metrics entirely.

These tests pin the shared answer — `db.HUMAN_SOURCES` — at each of those three
sites, so a fourth human surface cannot be added and silently discarded again.
"""

from __future__ import annotations

import json
import sqlite3

from cts import db, evaluate, export_training

THEME = "quietly menacing rather than overtly evil"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    return conn


def _query(conn: sqlite3.Connection, query_id: int, text: str = THEME) -> None:
    conn.execute(
        "INSERT INTO queries(id, text, kind, params, created_at) "
        "VALUES (?, ?, 'user', '{}', '2026-08-17T04:00:00Z')",
        (query_id, text),
    )


def _judgment(conn, query_id, iid, fit, source, rationale="because"):
    conn.execute(
        "INSERT INTO judgments(query_id, illustration_id, fit, rationale, prop_ids, "
        "model, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (query_id, iid, fit, rationale, json.dumps([1, 2]), "", source),
    )


# --------------------------------------------------------------- the shared definition


def test_both_human_surfaces_are_human():
    assert db.is_human_source("human")
    assert db.is_human_source("discord")


def test_model_sources_are_not_human():
    assert not db.is_human_source("judge")
    assert not db.is_human_source("distill")
    assert not db.is_human_source("")
    assert not db.is_human_source(None)


def test_the_sql_fragment_lists_exactly_the_python_set():
    """The clause is spliced into a WHERE, so it must not drift from the set."""
    conn = _conn()
    listed = {
        row[0]
        for row in conn.execute(
            f"SELECT value FROM (SELECT 'human' AS value UNION SELECT 'discord' "
            f"UNION SELECT 'judge' UNION SELECT 'distill') "
            f"WHERE value IN {db.HUMAN_SOURCES_SQL}"
        )
    }
    assert listed == set(db.HUMAN_SOURCES)


# ------------------------------------------------------------ export_training: dedupe


def test_a_discord_vote_outranks_the_judge_on_the_same_artwork():
    """The bug this file exists for: the judge used to win, discarding the vote."""
    conn = _conn()
    _query(conn, 1)
    _judgment(conn, 1, "ill-a", 0.91, "judge", "the judge liked it")
    _judgment(conn, 1, "ill-a", 0.0, "discord", "discord user 210657742501838848 said no")
    conn.commit()

    grouped, _raw = export_training._collect_labels(conn)
    (label,) = grouped[export_training._norm_text(THEME)]
    assert label["source"] == "discord"
    assert label["fit"] == 0.0


def test_a_discord_vote_outranks_a_judge_row_written_after_it():
    """Priority, not recency: a later re-judge must not overwrite a human mark."""
    conn = _conn()
    _query(conn, 1)
    _judgment(conn, 1, "ill-a", 1.0, "discord")
    _judgment(conn, 1, "ill-a", 0.12, "judge")     # rowid is higher
    conn.commit()

    grouped, _raw = export_training._collect_labels(conn)
    (label,) = grouped[export_training._norm_text(THEME)]
    assert label["source"] == "discord"
    assert label["fit"] == 1.0


def test_the_later_human_vote_wins_against_the_earlier_one():
    """Humans tie with each other, so a changed mind is recorded, not ignored."""
    conn = _conn()
    _query(conn, 1)
    _judgment(conn, 1, "ill-a", 1.0, "discord")
    _judgment(conn, 1, "ill-a", 0.0, "human")
    conn.commit()

    grouped, _raw = export_training._collect_labels(conn)
    (label,) = grouped[export_training._norm_text(THEME)]
    assert label["fit"] == 0.0


def test_dedupe_still_keys_on_query_text_across_rows():
    """Same theme run twice is one `queries` row per run; the label is per artwork."""
    conn = _conn()
    _query(conn, 1)
    _query(conn, 2)
    _judgment(conn, 1, "ill-a", 0.88, "judge")
    _judgment(conn, 2, "ill-a", 0.0, "discord")
    conn.commit()

    grouped, _raw = export_training._collect_labels(conn)
    (label,) = grouped[export_training._norm_text(THEME)]
    assert label["source"] == "discord"


def test_judge_still_beats_distill():
    conn = _conn()
    _query(conn, 1)
    _judgment(conn, 1, "ill-a", 0.5, "distill")
    _judgment(conn, 1, "ill-a", 0.9, "judge")
    conn.commit()

    grouped, _raw = export_training._collect_labels(conn)
    (label,) = grouped[export_training._norm_text(THEME)]
    assert label["source"] == "judge"


# ------------------------------------------------------------ export_training: weight


def test_a_discord_row_carries_the_human_weight():
    assert export_training._weight("discord") == export_training.HUMAN_WEIGHT
    assert export_training._weight("human") == export_training.HUMAN_WEIGHT
    assert export_training._weight("judge") == 1
    assert export_training._weight("distill") == 1
    assert export_training._weight("") == 1


def test_every_human_source_weighs_the_same():
    weights = {export_training._weight(s) for s in db.HUMAN_SOURCES}
    assert weights == {export_training.HUMAN_WEIGHT}


# ------------------------------------------------------------------ evaluate: metrics


def test_stored_marks_counts_discord_votes():
    conn = _conn()
    _query(conn, 1)
    _judgment(conn, 1, "ill-a", 1.0, "discord")
    _judgment(conn, 1, "ill-b", 0.0, "human")
    _judgment(conn, 1, "ill-c", 0.97, "judge")     # a model's opinion is not a mark
    conn.commit()

    marks = evaluate._stored_marks(conn, THEME)
    assert marks == {"ill-a": 1.0, "ill-b": 0.0}


def test_stored_marks_is_keyed_on_text_not_query_id():
    """Deliberate: every run inserts a fresh `queries` row for the same theme."""
    conn = _conn()
    _query(conn, 1)
    _query(conn, 2, "a different theme entirely")
    _judgment(conn, 1, "ill-a", 1.0, "discord")
    _judgment(conn, 2, "ill-z", 1.0, "discord")
    conn.commit()

    assert evaluate._stored_marks(conn, THEME) == {"ill-a": 1.0}
    assert evaluate._stored_marks(conn, "a different theme entirely") == {"ill-z": 1.0}


def test_the_last_mark_on_an_artwork_wins():
    conn = _conn()
    _query(conn, 1)
    _query(conn, 2)
    _judgment(conn, 1, "ill-a", 1.0, "human")
    _judgment(conn, 2, "ill-a", 0.0, "discord")
    conn.commit()

    assert evaluate._stored_marks(conn, THEME) == {"ill-a": 0.0}
