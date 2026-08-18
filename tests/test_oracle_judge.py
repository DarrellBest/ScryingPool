"""Batch-local chunk renumbering and citation validation, mirroring
`tests/test_judge_props.py` one level down — plus the prompt-content test that
guards the mechanics rubric. No Ollama, no network.
"""

from __future__ import annotations

import re

from cts import oracle_judge


def _candidate(oid: str, name: str) -> dict:
    return {"oracle_id": oid, "name": name}


def _evidence(oid: str, name: str, first_id: int, count: int) -> dict:
    return {
        "name": name,
        "type_line": "Instant",
        "oracle_text": f"oracle text for {name}",
        "chunks": [
            {"id": first_id + i, "text": f"ability {i} of {name}"} for i in range(count)
        ],
    }


BATCH = [_candidate("o1", "Alpha"), _candidate("o2", "Beta"), _candidate("o3", "Gamma")]
EVIDENCE = {
    "o1": _evidence("o1", "Alpha", 148_201, 4),
    "o2": _evidence("o2", "Beta", 93_004, 3),
    "o3": _evidence("o3", "Gamma", 170_455, 2),
}


def test_display_ids_are_short_and_sequential_across_the_batch():
    views = oracle_judge.number_batch(BATCH, EVIDENCE)
    shown = [display for view in views for display, _ in view["numbered"]]
    assert shown == list(range(1, 10))


def test_display_ids_stay_unique_across_candidates():
    views = oracle_judge.number_batch(BATCH, EVIDENCE)
    ranges = [(view["numbered"][0][0], view["numbered"][-1][0]) for view in views]
    assert ranges == [(1, 4), (5, 7), (8, 9)]


def test_prompt_shows_short_ids_states_each_candidates_range_and_the_full_text():
    views = oracle_judge.number_batch(BATCH, EVIDENCE)
    prompt = oracle_judge.build_judge_prompt("draw a card", views, EVIDENCE)
    assert "may cite ONLY ids 1-4" in prompt
    assert "may cite ONLY ids 5-7" in prompt
    assert "148201" not in prompt
    assert not re.search(r"\[\d{5,}\]", prompt)
    # The full oracle text is shown, verbatim, ahead of the numbered abilities.
    assert "oracle text for Alpha" in prompt


def test_cited_display_ids_are_mapped_back_to_real_chunk_ids():
    views = oracle_judge.number_batch(BATCH, EVIDENCE)
    entries = [
        {"candidate": 1, "fit": 0.8, "rationale": "r", "chunk_ids": [1, 3]},
        {"candidate": 2, "fit": 0.4, "rationale": "r", "chunk_ids": [5]},
        {"candidate": 3, "fit": 0.1, "rationale": "r", "chunk_ids": []},
    ]
    results = oracle_judge._apply_entries(views, entries)
    assert results[0]["chunk_ids"] == [148_201, 148_203]
    assert results[1]["chunk_ids"] == [93_004]
    assert results[2]["chunk_ids"] == []
    assert all(r["invented_chunk_ids"] == 0 for r in results)


def test_a_neighbouring_candidates_id_is_dropped_and_counted_as_misattributed():
    views = oracle_judge.number_batch(BATCH, EVIDENCE)
    entries = [{"candidate": 1, "fit": 0.9, "rationale": "r", "chunk_ids": [1, 6]}]
    results = oracle_judge._apply_entries(views, entries)
    assert results[0]["chunk_ids"] == [148_201]
    assert results[0]["misattributed_chunk_ids"] == 1
    assert results[0]["invented_chunk_ids"] == 1


def test_an_id_that_was_never_shown_is_dropped_and_counted_as_invented():
    views = oracle_judge.number_batch(BATCH, EVIDENCE)
    entries = [{"candidate": 1, "fit": 0.9, "rationale": "r", "chunk_ids": [1, 999, "x"]}]
    results = oracle_judge._apply_entries(views, entries)
    assert results[0]["chunk_ids"] == [148_201]
    assert results[0]["misattributed_chunk_ids"] == 0
    assert results[0]["invented_chunk_ids"] == 2


def test_a_candidate_missing_from_the_response_degrades_instead_of_shifting():
    views = oracle_judge.number_batch(BATCH, EVIDENCE)
    entries = [{"candidate": 2, "fit": 0.5, "rationale": "r", "chunk_ids": [5]}]
    results = oracle_judge._apply_entries(views, entries)
    assert results[0]["fit"] is None and results[0]["judged"] is False
    assert results[1]["fit"] == 0.5


def test_a_candidate_with_no_ability_chunks_is_told_it_may_cite_none():
    evidence = {"o9": {"name": "Delta", "type_line": "Land", "oracle_text": "", "chunks": []}}
    views = oracle_judge.number_batch([_candidate("o9", "Delta")], evidence)
    prompt = oracle_judge.build_judge_prompt("theme", views, evidence)
    assert "may cite NO chunk_ids" in prompt
    results = oracle_judge._apply_entries(
        views, [{"candidate": 1, "fit": 0.2, "rationale": "r", "chunk_ids": [1]}]
    )
    assert results[0]["chunk_ids"] == []
    assert results[0]["invented_chunk_ids"] == 1


# ------------------------------------------------------------- the mechanics rubric


def test_the_mechanics_rubric_names_every_discriminating_family():
    """A prompt edit that quietly deletes two of these lines is invisible in
    review and catastrophic in output — this is the guard against exactly
    that regression."""
    rubric = oracle_judge.MECHANICS_RUBRIC.upper()
    for mechanic in ("DRAW", "LOOT", "RUMMAGE", "IMPULSE", "SURVEIL", "SCRY", "REVEAL", "TUTOR"):
        assert mechanic in rubric, mechanic


def test_the_rubric_is_actually_in_the_judge_prompt():
    views = oracle_judge.number_batch(BATCH, EVIDENCE)
    prompt = oracle_judge.build_judge_prompt("that let me draw", views, EVIDENCE)
    for mechanic in ("DRAW", "LOOT", "RUMMAGE", "IMPULSE", "SURVEIL", "SCRY", "REVEAL", "TUTOR"):
        assert mechanic in prompt


def test_the_rubric_states_impulse_is_not_a_draw():
    rubric = oracle_judge.MECHANICS_RUBRIC
    assert "NOT drawing" in rubric or "NOT be" in rubric or "This is NOT drawing" in rubric


def test_no_verification_stage_exists_on_this_module():
    """Unlike judge.py, there is deliberately no verify_finalists analogue: the
    corpus is already ground truth."""
    assert not hasattr(oracle_judge, "verify_finalists")
