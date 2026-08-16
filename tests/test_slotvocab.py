"""Defect 1: slot filters that matched nothing, ever.

Every value asserted on here is a real stored slot string or a real router-emitted
filter, taken from data/commanders.db and from the logged plans of the 40-query eval.
No Ollama, no network.
"""

from __future__ import annotations

import pytest

from cts import slotvocab
from cts.search import allowed_illustrations


# ------------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    "text, expected",
    [
        ("green-skinned humanoid", "green skinned humanoid"),
        ("low_angle_shot", "low angle shot"),
        ("Treefolk", "treefolk"),
        ('"species":"dog"', "species dog"),
    ],
)
def test_normalize_flattens_case_punctuation_and_underscores(text, expected):
    assert slotvocab.normalize(text) == expected


def test_tokens_singularize_but_leave_real_us_and_ss_words_alone():
    assert slotvocab.tokens("feathered wings") == {"feathered", "wing"}
    assert "fungus" in slotvocab.tokens("fungus creature")
    assert "grass" in slotvocab.tokens("grass")


# ----------------------------------------------------------------- phrase matching


def test_equals_is_containment_not_string_identity():
    # "commanders that are human" must reach "human female", which the old
    # lower(slot) = lower(value) comparison could not.
    assert slotvocab._phrase_match("equals", "human", "human female")
    assert slotvocab._phrase_match("equals", "human", "human")


def test_equals_still_distinguishes_human_from_humanoid():
    # Tokens are compared whole, so a prefix is not a match.
    assert not slotvocab._phrase_match("equals", "human", "humanoid")
    assert not slotvocab._phrase_match("equals", "human", "elf-like humanoid")


def test_contains_keeps_substring_matching_for_serialized_lists():
    held = '[{"object":"lantern","is_weapon":false}]'
    assert slotvocab._phrase_match("contains", '"is_weapon":false', held)
    assert not slotvocab._phrase_match("contains", '"is_weapon":true', held)


def test_token_subset_does_not_apply_to_whole_sentences():
    # Three ordinary words co-occurring somewhere in a long descriptive sentence is
    # not evidence; a multi-word term has to appear contiguously.
    sentence = (
        "wide shot with the figure low in the frame, camera at eye level, a bright "
        "angle of light from the upper left"
    )
    assert not slotvocab._phrase_match("contains", "low angle", sentence)
    assert slotvocab._phrase_match(
        "contains", "low angle", "full figure shot, low angle looking up at the subject"
    )


# --------------------------------------------------------------------- alias mining


def test_mining_recovers_goblin_vocabulary_from_the_corpus(corpus_conn):
    vocab = slotvocab.build(corpus_conn)
    assert "green skinned humanoid with pointed ears" in vocab.aliases["goblin"]
    assert "green skinned humanoid" in vocab.aliases["goblin"]


def test_mining_rejects_the_uninformative_catch_all_phrase(corpus_conn):
    # "humanoid" is what the vision pass writes when it cannot tell; it is 20% of the
    # corpus and must never become vocabulary for anything.
    vocab = slotvocab.build(corpus_conn)
    for term, phrases in vocab.aliases.items():
        assert "humanoid" not in phrases, f"{term} mined the catch-all phrase"


def test_mining_stays_empty_where_the_corpus_has_no_evidence(corpus_conn):
    # Dwarves are recorded as plain "humanoid": there is nothing to learn, and
    # inventing something would be worse than matching nothing.
    vocab = slotvocab.build(corpus_conn)
    assert not vocab.aliases.get("dwarf")


# ------------------------------------------------------------------------- matching


def test_goblin_filter_reaches_goblins_without_dragging_in_every_humanoid(
    corpus_conn, names_for
):
    vocab = slotvocab.build(corpus_conn)
    matched = names_for(
        vocab.match({"path": "primary_subject.species", "op": "equals", "value": "goblin"})
    )
    assert "Krenko, Mob Boss" in matched
    assert "Rulik Mons, Warren Chief" in matched
    assert "Vial Smasher the Fierce" in matched
    assert "Gwendlyn Di Corci" not in matched
    assert "Torbran, Thane of Red Fell" not in matched


def test_angel_filter_reaches_winged_humanoids(corpus_conn, names_for):
    vocab = slotvocab.build(corpus_conn)
    matched = names_for(
        vocab.match({"path": "primary_subject.species", "op": "equals", "value": "angel"})
    )
    assert {"Radiant, Serra Archangel", "Razia, Boros Archangel"} <= matched


def test_creature_type_alias_matches_the_whole_phrase_only(corpus_conn, names_for):
    # A mined phrase is matched as a whole. If it were matched by loose token
    # containment, "green skinned humanoid" would also pull in every other
    # green-skinned thing the corpus happens to describe with more words.
    vocab = slotvocab.build(corpus_conn)
    vocab.aliases["goblin"] = ["humanoid creature"]
    matched = vocab.match(
        {"path": "primary_subject.species", "op": "equals", "value": "goblin"}
    )
    assert names_for(matched) == {"Vial Smasher the Fierce"}  # the lexical hit only


def test_snake_case_composition_value_resolves_to_real_prose(corpus_conn, names_for):
    vocab = slotvocab.build(corpus_conn)
    matched = names_for(
        vocab.match({"path": "composition", "op": "equals", "value": "low_angle_shot"})
    )
    assert "Radiant, Serra Archangel" in matched
    assert "Jazal Goldmane" in matched
    assert "Gwendlyn Di Corci" not in matched


def test_pose_phrasing_difference_is_bridged(corpus_conn, names_for):
    vocab = slotvocab.build(corpus_conn)
    matched = names_for(
        vocab.match(
            {"path": "primary_subject.pose", "op": "contains", "value": "seen from behind"}
        )
    )
    assert "Avacyn, Angel of Hope" in matched


def test_other_figures_filter_works_written_either_way(corpus_conn, names_for):
    vocab = slotvocab.build(corpus_conn)
    as_json = vocab.match(
        {"path": "other_figures", "op": "contains", "value": '"species":"dog"'}
    )
    as_word = vocab.match({"path": "other_figures", "op": "contains", "value": "dog"})
    assert "Rin and Seri, Inseparable" in names_for(as_json)
    assert "Rin and Seri, Inseparable" in names_for(as_word)


def test_negation_is_never_expanded(corpus_conn, names_for):
    # not_contains 'goblin' must not delete every green-skinned humanoid: an
    # over-broad synonym set on a negation removes correct answers outright.
    vocab = slotvocab.build(corpus_conn)
    kept = names_for(
        vocab.match(
            {"path": "primary_subject.species", "op": "not_contains", "value": "goblin"}
        )
    )
    assert "Krenko, Mob Boss" in kept
    assert "Vial Smasher the Fierce" not in kept  # its species literally says goblin


def test_figure_count_comparisons_still_work(corpus_conn, names_for):
    vocab = slotvocab.build(corpus_conn)
    two = vocab.match({"path": "figure_count", "op": "gte", "value": "2"})
    assert "Rin and Seri, Inseparable" in names_for(two)
    assert "Gwendlyn Di Corci" not in names_for(two)


# ------------------------------------------------------------- conjunction and drops


def test_a_filter_that_matches_nothing_is_dropped_and_named(monkeypatch, corpus_conn):
    # The fixture corpus is ~100 artworks, so the real pool floor would swallow
    # everything; these tests are about which filter gets blamed, not about the floor.
    monkeypatch.setattr("cts.search.MIN_FILTERED_POOL", 2)
    notes: list[str] = []
    allowed = allowed_illustrations(
        corpus_conn,
        [
            {"path": "primary_subject.species", "op": "equals", "value": "angel"},
            {"path": "primary_subject.clothing", "op": "contains", "value": "scuba gear"},
        ],
        notes,
    )
    assert allowed  # the angel filter survives on its own
    assert any("scuba gear" in n and "matched nothing" in n for n in notes)


def test_the_blame_falls_on_the_empty_filter_not_the_last_one(monkeypatch, corpus_conn):
    """The bug behind the eval's most confusing note.

    The old code intersected every filter and then popped the *last* one whenever the
    result was empty, so `not_contains 'human'` — which matches nearly the whole
    corpus — was reported as "matched nothing".
    """
    monkeypatch.setattr("cts.search.MIN_FILTERED_POOL", 2)
    notes: list[str] = []
    allowed_illustrations(
        corpus_conn,
        [
            {"path": "primary_subject.clothing", "op": "contains", "value": "scuba gear"},
            {"path": "primary_subject.species", "op": "not_contains", "value": "human"},
        ],
        notes,
    )
    empty = [n for n in notes if "matched nothing" in n]
    assert len(empty) == 1
    assert "scuba gear" in empty[0]


def test_filters_that_conflict_are_reported_as_conflicting(monkeypatch, corpus_conn):
    monkeypatch.setattr("cts.search.MIN_FILTERED_POOL", 2)
    notes: list[str] = []
    allowed = allowed_illustrations(
        corpus_conn,
        [
            {"path": "primary_subject.species", "op": "equals", "value": "angel"},
            {"path": "primary_subject.species", "op": "equals", "value": "treefolk"},
        ],
        notes,
    )
    assert allowed
    assert any("would leave only" in n for n in notes)


def test_a_conjunction_is_not_allowed_to_collapse_the_pool(monkeypatch, corpus_conn, names_for):
    """The regression that showed up end to end on "angels with feathered wings".

    The wings are recorded in `species` for some artworks and in `clothing` for others,
    so ANDing the two describes nobody: on the real corpus it cut 94 matches to 2.
    """
    monkeypatch.setattr("cts.search.MIN_FILTERED_POOL", 4)
    notes: list[str] = []
    allowed = allowed_illustrations(
        corpus_conn,
        [
            {"path": "primary_subject.species", "op": "equals", "value": "angel"},
            {"path": "primary_subject.clothing", "op": "contains", "value": "white feathered wings"},
        ],
        notes,
    )
    matched = names_for(allowed)
    assert "Razia, Boros Archangel" in matched  # gold wings, but not "white" ones
    assert any("would leave only" in n for n in notes)


def test_a_filter_too_narrow_to_fill_a_pool_is_handed_to_the_retriever(
    monkeypatch, corpus_conn
):
    """Measured on the eval gold sets: a hard mask down to 13 artworks costs recall.

    The retriever searches the same words without deleting anything, so below the floor
    the constraint goes to it instead of masking the corpus.
    """
    monkeypatch.setattr("cts.search.MIN_FILTERED_POOL", 25)
    notes: list[str] = []
    allowed = allowed_illustrations(
        corpus_conn,
        [{"path": "primary_subject.species", "op": "equals", "value": "treefolk"}],
        notes,
    )
    assert allowed is None
    assert any("the retriever ranks on it instead" in n for n in notes)


def test_a_filter_that_leaves_a_workable_pool_is_applied(monkeypatch, corpus_conn, names_for):
    monkeypatch.setattr("cts.search.MIN_FILTERED_POOL", 4)
    notes: list[str] = []
    allowed = allowed_illustrations(
        corpus_conn,
        [{"path": "primary_subject.species", "op": "equals", "value": "treefolk"}],
        notes,
    )
    assert "Kurbis, Harvest Celebrant" in names_for(allowed)
    assert not any("dropped" in n for n in notes)


def test_no_filters_means_no_restriction(corpus_conn):
    assert allowed_illustrations(corpus_conn, [], []) is None


# ------------------------------------------------------- the real corpus, if present


REVIVED = [
    # (path, op, value) — filters the real eval emitted that matched zero rows
    ("primary_subject.species", "equals", "goblin"),
    ("primary_subject.species", "equals", "angel"),
    ("primary_subject.species", "contains", "undead"),
    ("primary_subject.species", "contains", "Treefolk"),
    ("primary_subject.species", "equals", "wolf"),
    ("composition", "equals", "low_angle_shot"),
    ("primary_subject.pose", "contains", "seen from behind"),
    ("primary_subject.pose", "contains", "walking away"),
    ("primary_subject.held_objects", "contains", '"object":"banner"'),
]


@pytest.mark.parametrize("path, op, value", REVIVED)
def test_real_corpus_filters_that_used_to_be_dead_now_match(real_conn, path, op, value):
    vocab = slotvocab.load(real_conn)
    assert vocab.match({"path": path, "op": op, "value": value})


@pytest.mark.parametrize(
    "path, op, value, floor",
    [
        # filters that already worked must not regress
        ("primary_subject.facial_hair", "contains", "beard", 600),
        ("primary_subject.held_objects", "contains", '"is_weapon":false', 4000),
        ("art_style", "contains", "painterly", 4000),
        ("figure_count", "equals", "1", 3000),
        ("setting", "contains", "snow", 80),
    ],
)
def test_real_corpus_filters_that_worked_still_work(real_conn, path, op, value, floor):
    vocab = slotvocab.load(real_conn)
    assert len(vocab.match({"path": path, "op": op, "value": value})) >= floor


def test_real_corpus_still_reports_genuinely_unanswerable_filters(real_conn):
    # Honest degradation: nothing in the corpus records a "scientist" outfit or a
    # "grieving" pose, so these must keep matching nothing rather than matching a guess.
    vocab = slotvocab.load(real_conn)
    assert not vocab.match(
        {"path": "primary_subject.clothing", "op": "contains", "value": "scientist"}
    )
    assert not vocab.match(
        {"path": "primary_subject.pose", "op": "contains", "value": "grieving"}
    )


def test_real_corpus_router_hint_names_what_resolves(real_conn):
    hint = slotvocab.router_hint(slotvocab.load(real_conn))
    assert "goblin" in hint and "angel" in hint
    assert "humanoid" in hint  # the descriptive vocabulary is shown too
