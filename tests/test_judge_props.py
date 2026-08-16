"""Defect 2: the judge citing proposition ids outside its own candidate.

The prompt used to show each candidate's global props.id — six digits, ten candidates
per call. These tests pin the batch-local renumbering that replaced it, and the
validation that still catches a bad citation either way. No Ollama, no network.
"""

from __future__ import annotations

import re

from cts import judge


def _candidate(ill: str, name: str) -> dict:
    return {"illustration_id": ill, "oracle_id": f"o-{ill}", "name": name}


def _evidence(ill: str, first_id: int, count: int) -> dict:
    return {
        "literal": "a literal paragraph",
        "interpretive": "an interpretive paragraph",
        "props": [
            {"id": first_id + i, "layer": "literal", "text": f"statement {i} about {ill}"}
            for i in range(count)
        ],
    }


BATCH = [_candidate("i1", "Alpha"), _candidate("i2", "Beta"), _candidate("i3", "Gamma")]
EVIDENCE = {
    "i1": _evidence("i1", 148_201, 4),
    "i2": _evidence("i2", 93_004, 3),
    "i3": _evidence("i3", 170_455, 2),
}


def test_display_ids_are_short_and_sequential_across_the_batch():
    views = judge.number_batch(BATCH, EVIDENCE)
    shown = [display for view in views for display, _ in view["numbered"]]
    assert shown == list(range(1, 10))


def test_display_ids_stay_unique_across_candidates_so_strays_remain_detectable():
    views = judge.number_batch(BATCH, EVIDENCE)
    ranges = [(view["numbered"][0][0], view["numbered"][-1][0]) for view in views]
    assert ranges == [(1, 4), (5, 7), (8, 9)]


def test_prompt_shows_short_ids_and_states_each_candidate_s_range():
    views = judge.number_batch(BATCH, EVIDENCE)
    prompt = judge.build_judge_prompt("commanders that look lonely", views, EVIDENCE)
    assert "may cite ONLY ids 1-4" in prompt
    assert "may cite ONLY ids 5-7" in prompt
    assert "148201" not in prompt  # no six-digit ids anywhere in the prompt
    assert not re.search(r"\[\d{5,}\]", prompt)


def test_cited_display_ids_are_mapped_back_to_real_prop_ids():
    views = judge.number_batch(BATCH, EVIDENCE)
    entries = [
        {"candidate": 1, "fit": 0.8, "rationale": "r", "prop_ids": [1, 3]},
        {"candidate": 2, "fit": 0.4, "rationale": "r", "prop_ids": [5]},
        {"candidate": 3, "fit": 0.1, "rationale": "r", "prop_ids": []},
    ]
    results = judge._apply_entries(views, entries)
    assert results[0]["prop_ids"] == [148_201, 148_203]
    assert results[1]["prop_ids"] == [93_004]
    assert results[2]["prop_ids"] == []
    assert all(r["invented_prop_ids"] == 0 for r in results)


def test_a_neighbouring_candidate_s_id_is_dropped_and_counted_as_misattributed():
    views = judge.number_batch(BATCH, EVIDENCE)
    entries = [{"candidate": 1, "fit": 0.9, "rationale": "r", "prop_ids": [1, 6]}]
    results = judge._apply_entries(views, entries)
    assert results[0]["prop_ids"] == [148_201]
    assert results[0]["misattributed_prop_ids"] == 1
    assert results[0]["invented_prop_ids"] == 1


def test_an_id_that_was_never_shown_is_dropped_and_counted_as_invented():
    views = judge.number_batch(BATCH, EVIDENCE)
    entries = [{"candidate": 1, "fit": 0.9, "rationale": "r", "prop_ids": [1, 999, "x"]}]
    results = judge._apply_entries(views, entries)
    assert results[0]["prop_ids"] == [148_201]
    assert results[0]["misattributed_prop_ids"] == 0
    assert results[0]["invented_prop_ids"] == 2
    assert results[0]["cited_prop_ids"] == 3


def test_a_candidate_missing_from_the_response_degrades_instead_of_shifting():
    views = judge.number_batch(BATCH, EVIDENCE)
    entries = [{"candidate": 2, "fit": 0.5, "rationale": "r", "prop_ids": [5]}]
    results = judge._apply_entries(views, entries)
    assert results[0]["fit"] is None and results[0]["judged"] is False
    assert results[1]["fit"] == 0.5
    assert results[1]["prop_ids"] == [93_004]


def test_a_candidate_with_no_propositions_is_told_it_may_cite_none():
    views = judge.number_batch([_candidate("i9", "Delta")], {"i9": {"props": []}})
    prompt = judge.build_judge_prompt("theme", views, {"i9": {"props": []}})
    assert "may cite NO prop_ids" in prompt
    results = judge._apply_entries(views, [{"candidate": 1, "fit": 0.2, "rationale": "r", "prop_ids": [1]}])
    assert results[0]["prop_ids"] == []
    assert results[0]["invented_prop_ids"] == 1
