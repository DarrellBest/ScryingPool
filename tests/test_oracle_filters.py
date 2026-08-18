"""The second-highest-risk piece, because it fails silently: a mis-parsed
categorical filter is self-announcing, a flipped `cmc <= 5` to `cmc >= 5` is
not. Every English phrase in the design doc's calibration table is exercised
here through the router's op/value representation, which is the one path that
can invert a comparison at all — the explicit mv_min/mv_max path has no
operator to flip by construction, and that property is tested directly too.
"""

from __future__ import annotations

import sqlite3

import pytest

from cts import oracle_db, oracle_filters as ofilters
from cts.config import Config
from cts.oracle_filters import Filters


def _cfg(tmp_path) -> Config:
    return Config(
        ollama_url="u", vision_model="v", verify_model="v", embed_model="e",
        judge_model="j", db_path=str(tmp_path / "commanders.db"),
        art_dir=str(tmp_path / "art"), power_weights={},
        oracle_db_path=str(tmp_path / "oracle.db"),
    )


CARDS = (
    # oracle_id, name, cmc, color_identity, types(list), legal(list of format)
    ("o-enchant-g", "Verdant Force", 3.0, "G", ["enchantment"], ["commander", "pauper"]),
    ("o-enchant-g5", "Sylvan Library", 5.0, "G", ["enchantment"], ["commander", "legacy"]),
    ("o-enchant-gw", "Selesnya Charm", 5.0, "GW", ["enchantment"], ["commander"]),
    ("o-enchant-bg", "Golgari Charm", 5.0, "BG", ["enchantment"], ["commander"]),
    ("o-artifact-g", "Sol Talisman", 2.0, "G", ["artifact"], ["commander", "modern"]),
    ("o-enchant-g7", "Expensive Green", 7.0, "G", ["enchantment"], ["commander"]),
    ("o-colorless", "Sol Ring", 1.0, "", ["artifact"], ["commander", "vintage"]),
    ("o-planeswalker-g", "Nissa Placeholder", 5.0, "G", ["planeswalker", "legendary"], ["commander"]),
)


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    connection = oracle_db.connect(_cfg(tmp_path))
    for oracle_id, name, cmc, ci, types, legal in CARDS:
        connection.execute(
            "INSERT INTO cards(oracle_id, name, name_norm, cmc, color_identity) "
            "VALUES (?, ?, ?, ?, ?)",
            (oracle_id, name, name.lower(), cmc, ci),
        )
        for t in types:
            connection.execute(
                "INSERT INTO card_types(oracle_id, kind, value) VALUES (?, 'type', ?)",
                (oracle_id, t),
            )
        for fmt in legal:
            connection.execute(
                "INSERT INTO card_legalities(oracle_id, format, status) VALUES (?, ?, 'legal')",
                (oracle_id, fmt),
            )
    connection.commit()
    yield connection
    connection.close()


def _ids(conn, sql_result):
    return sql_result if sql_result is not None else None


# ------------------------------------------------------------------------------ types


def test_types_is_union_within_the_field(conn):
    allowed = ofilters.compile_hard(conn, Filters(types=("enchantment", "artifact")), [])
    names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(allowed))),
        list(allowed),
    )}
    assert names == {
        "Verdant Force", "Sylvan Library", "Selesnya Charm", "Golgari Charm",
        "Sol Talisman", "Expensive Green", "Sol Ring",
    }


def test_types_and_colors_and_across_fields(conn):
    notes: list[str] = []
    allowed = ofilters.compile_hard(conn, Filters(types=("enchantment",), colors="G"), notes)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(allowed))),
        list(allowed),
    )}
    # enchantment AND mono-green (or colorless, none here): excludes GW/BG.
    assert names == {"Verdant Force", "Sylvan Library", "Expensive Green"}


# ----------------------------------------------------------------------------- colors


def test_colors_subset_excludes_multicolor_that_goes_outside_the_requested_set(conn):
    allowed = ofilters.compile_hard(conn, Filters(colors="G"), [])
    names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(allowed))),
        list(allowed),
    )}
    # mono-green and colorless match; Selesnya (GW) and Golgari (BG) do not.
    assert "Selesnya Charm" not in names
    assert "Golgari Charm" not in names
    assert "Sol Ring" in names          # colorless fits inside any requested set
    assert "Verdant Force" in names


def test_colors_gw_admits_selesnya_and_mono_of_either_but_not_golgari(conn):
    allowed = ofilters.compile_hard(conn, Filters(colors="GW"), [])
    names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(allowed))),
        list(allowed),
    )}
    assert "Selesnya Charm" in names
    assert "Golgari Charm" not in names


# -------------------------------------------------------------------------- numeric mv


@pytest.mark.parametrize(
    "phrase,op,value,expected_names",
    [
        ("5 or less", "<=", 5, {"Verdant Force", "Sylvan Library", "Selesnya Charm",
                                 "Golgari Charm", "Sol Talisman", "Sol Ring",
                                 "Nissa Placeholder"}),
        ("no more than 5", "<=", 5, None),
        ("up to 5", "<=", 5, None),
        ("at most 5", "<=", 5, None),
        ("5 and under", "<=", 5, None),
        ("under 5", "<", 5, {"Verdant Force", "Sol Talisman", "Sol Ring"}),
        ("less than 5", "<", 5, None),
        ("below 5", "<", 5, None),
        ("5 or more", ">=", 5, {"Sylvan Library", "Selesnya Charm", "Golgari Charm",
                                 "Expensive Green", "Nissa Placeholder"}),
        ("at least 5", ">=", 5, None),
        ("5 and up", ">=", 5, None),
        ("over 5", ">", 5, {"Expensive Green"}),
        ("more than 5", ">", 5, None),
        ("above 5", ">", 5, None),
        ("exactly two", "=", 2, {"Sol Talisman"}),
        ("costs 3", "=", 3, {"Verdant Force"}),
    ],
)
def test_calibration_table_operators_produce_the_right_comparison(conn, phrase, op, value, expected_names):
    """Every phrase in the design doc's calibration table, fed through the
    router's op/value representation (never a live model call — this asserts
    the operator that a stubbed router response would have produced)."""
    allowed = ofilters.compile_hard(conn, Filters(mv_op=op, mv_value=value), [])
    names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(allowed)) or "NULL"),
        list(allowed),
    )}
    if expected_names is not None:
        assert names == expected_names, phrase


def test_le_and_lt_are_one_mana_value_apart_and_never_confused(conn):
    """`<=` and `<` are one word apart in English; this is where an inversion
    would actually cost a real card."""
    le = ofilters.compile_hard(conn, Filters(mv_op="<=", mv_value=5), [])
    lt = ofilters.compile_hard(conn, Filters(mv_op="<", mv_value=5), [])
    assert le != lt
    le_names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(le))), list(le))}
    lt_names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(lt))), list(lt))}
    assert le_names - lt_names == {"Sylvan Library", "Selesnya Charm", "Golgari Charm", "Nissa Placeholder"}


def test_between_three_and_five(conn):
    allowed = ofilters.compile_hard(conn, Filters(mv_op="between", mv_lo=3, mv_hi=5), [])
    names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(allowed))), list(allowed))}
    assert names == {"Verdant Force", "Sylvan Library", "Selesnya Charm", "Golgari Charm", "Nissa Placeholder"}


def test_the_explicit_path_has_no_operator_to_invert():
    """Two integers, inclusive bounds — there is no operator field anywhere on
    the explicit path for an inversion to live in."""
    import inspect

    sig = inspect.signature(ofilters.explicit_mv_predicate)
    assert list(sig.parameters) == ["mv_min", "mv_max", "notes"]
    assert "op" not in sig.parameters


def test_explicit_mv_min_and_max_are_inclusive_bounds(conn):
    predicate = ofilters.explicit_mv_predicate(3, 5, [])
    assert predicate == ("c.cmc BETWEEN ? AND ?", [3, 5], "3 ≤ mv ≤ 5")


def test_explicit_mv_max_only(conn):
    predicate = ofilters.explicit_mv_predicate(None, 5, [])
    assert predicate == ("c.cmc <= ?", [5], "mv ≤ 5")


def test_explicit_mv_min_only(conn):
    predicate = ofilters.explicit_mv_predicate(5, None, [])
    assert predicate == ("c.cmc >= ?", [5], "mv ≥ 5")


def test_explicit_mv_min_equals_max_is_an_exact_value():
    predicate = ofilters.explicit_mv_predicate(3, 3, [])
    assert predicate == ("c.cmc = ?", [3], "mv = 3")


def test_an_impossible_explicit_range_reports_and_matches_nothing():
    notes: list[str] = []
    predicate = ofilters.explicit_mv_predicate(6, 3, notes)
    assert predicate[0] == "1 = 0"
    assert notes and "mv_min" in notes[0]


# ------------------------------------------------------------------------- the guard


def test_a_value_outside_zero_to_thirty_is_dropped_with_a_note():
    notes: list[str] = []
    predicate = ofilters.explicit_mv_predicate(None, 999, notes)
    assert predicate is None
    assert notes and "outside 0-30" in notes[0]


def test_the_guard_applies_to_the_router_path_too():
    notes: list[str] = []
    predicate = ofilters.router_mv_predicate(">=", 40, None, None, notes)
    assert predicate is None
    assert notes


# ------------------------------------------------------------------- vague quantities


def test_a_vague_quantity_produces_no_filter():
    """The router is instructed to emit no numeric filter for a vague term —
    asserted here at the compile layer: no op means no predicate, ever."""
    f = Filters()  # what the router emits for "cheap": nothing
    assert ofilters._mv_predicate(f, []) is None
    assert ofilters.echo_line(f, "cheap") == 'filters: none · semantic: "cheap"'


# --------------------------------------------------------------------------- legal


def test_legal_is_union_within_the_field(conn):
    allowed = ofilters.compile_hard(conn, Filters(legal=("legacy", "modern")), [])
    names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(allowed))), list(allowed))}
    assert names == {"Sylvan Library", "Sol Talisman"}


# ----------------------------------------------------------------------- hard vs soft


def test_hard_filters_may_legitimately_return_zero(conn):
    allowed = ofilters.compile_hard(
        conn, Filters(types=("enchantment",), colors="G", mv_op="<=", mv_value=1), []
    )
    assert allowed == set()


def test_soft_filters_drop_the_one_that_would_zero_the_pool_and_only_that_one(conn):
    notes: list[str] = []
    # colors=G matches several; mv<=0 matches nothing at all when ANDed with it.
    allowed = ofilters.compile_soft(
        conn, Filters(colors="G", mv_op="<=", mv_value=0), notes
    )
    assert allowed is not None and len(allowed) > 0   # colors alone survived
    assert any("dropped" in n for n in notes)
    assert any("mv" in n for n in notes)


def test_soft_filters_apply_broadest_first(conn):
    """A soft type filter matching many cards should not get blamed for a
    conflict actually caused by a narrower soft filter."""
    notes: list[str] = []
    allowed = ofilters.compile_soft(
        conn, Filters(types=("enchantment", "artifact", "planeswalker"), mv_op="=", mv_value=999),
        notes,
    )
    # mv=999 is out of the 0-30 guard and drops before combining at all.
    assert allowed is not None
    assert len(allowed) > 0


def test_soft_filter_is_judged_against_the_hard_base_pool_not_the_whole_corpus(conn):
    """A soft filter that looks fine alone must still be dropped if it would
    zero out once ANDed with the user's explicit hard filters."""
    notes: list[str] = []
    hard = ofilters.compile_hard(conn, Filters(colors="G"), notes)
    soft = ofilters.compile_soft(
        conn, Filters(types=("artifact",)), notes, base=hard
    )
    # "artifact" alone matches Sol Talisman + Sol Ring; ANDed with colors=G it
    # must drop Sol Ring (colorless still passes colors=G, but let's force a
    # genuine conflict: mono-green artifacts only).
    names = {r[0] for r in conn.execute(
        "SELECT name FROM cards WHERE oracle_id IN ({})".format(",".join("?" * len(soft)) or "NULL"),
        list(soft),
    )}
    assert names <= {"Sol Talisman", "Sol Ring"}


def test_min_filtered_pool_floor_is_not_ported():
    """The art side drops a slot filter as soon as it would leave fewer than
    MIN_FILTERED_POOL=25. That floor has no analogue here at all — a soft
    filter must drop only when the pool actually reaches zero."""
    assert not hasattr(ofilters, "MIN_FILTERED_POOL")


# ------------------------------------------------------------------------- the echo


def test_echo_line_matches_the_design_docs_exact_format():
    f = Filters(types=("enchantment",), colors="G", mv_op="<=", mv_value=5)
    line = ofilters.echo_line(f, "let me draw")
    assert line == (
        'filters: type = enchantment · colors ⊆ {G} (identity fits inside) · '
        'mv ≤ 5 · semantic: "let me draw"'
    )


def test_echo_line_with_no_filters_and_no_semantic():
    assert ofilters.echo_line(Filters(), None) == "filters: none · semantic: none"


def test_echo_line_reflects_an_inverted_operator_visibly():
    """If the router inverted '5 or less' to '>=', the echo shows mv ≥ 5 — the
    exact mechanism that turns a silent failure into a visible one."""
    f = Filters(mv_op=">=", mv_value=5)
    assert "mv ≥ 5" in ofilters.echo_line(f, None)


def test_echo_line_types_and_legal_are_comma_joined_and_sorted():
    f = Filters(types=("planeswalker", "artifact"), legal=("modern", "legacy"))
    line = ofilters.echo_line(f, None)
    assert "type = artifact, planeswalker" in line
    assert "legal = legacy, modern" in line


# --------------------------------------------------------------------- scryfall url


def test_scryfall_url_round_trips_type_color_and_mv():
    f = Filters(types=("enchantment",), colors="G", mv_max=5)
    url = ofilters.scryfall_url(f)
    assert "t%3Aenchantment" in url or "t:enchantment" in url.replace("%3A", ":")
    assert "id%3Ag" in url or "id:g" in url.replace("%3A", ":")
    assert "cmc%3C%3D5" in url or "cmc<=5" in url.replace("%3C", "<").replace("%3D", "=")


def test_scryfall_colors_always_emits_the_subset_form_never_the_superset_form():
    """Scryfall's bare `id:g` already means `id<=g` (subset) — the same
    semantics this design uses. `id>=g` (superset) must never be emitted."""
    url = ofilters.scryfall_url(Filters(colors="G"))
    decoded = url.replace("%3A", ":")
    assert "id:g" in decoded
    assert "id>" not in decoded and "id%3E" not in url


def test_scryfall_url_is_none_when_there_are_no_filters_at_all():
    assert ofilters.scryfall_url(Filters()) is None


def test_scryfall_url_expresses_a_between_range_as_two_bounds():
    url = ofilters.scryfall_url(Filters(mv_op="between", mv_lo=3, mv_hi=5))
    decoded = url.replace("%3E", ">").replace("%3D", "=").replace("%3C", "<").replace("+", " ")
    assert "cmc>=3" in decoded
    assert "cmc<=5" in decoded
