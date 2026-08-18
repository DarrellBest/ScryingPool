"""`fold()` and the six-layer ladder. No network, no Ollama, no `data/oracle.db`.

The corpus below is small and every name in it is a real Magic card name chosen
because it is *hostile* in a specific way — an apostrophe, a circumflex, a `//`,
a ligature the user might type, a name that is a strict prefix of another, a pair
one edit apart. A fixture of invented names would test the code and none of the
problem.

The property this file exists to protect is in
`test_a_layer_only_fires_when_every_layer_above_it_missed` and the three tests
under it: **strict short-circuiting**. An eager fuzzy layer that overrides an
exact match is the failure mode that turns a name lookup into a liar, and under
strict ordering it is structurally impossible rather than merely unlikely.
"""

from __future__ import annotations

import sqlite3
import unicodedata

import pytest

from cts import oracle_db, oracle_names
from cts.oracle_names import fold, resolve

# (name, edhrec_rank, [face names])
CORPUS: tuple[tuple[str, int | None, tuple[str, ...]], ...] = (
    ("Sol Ring", 1, ()),
    ("Gaea's Cradle", 449, ()),
    ("Lim-Dûl's Vault", 2363, ()),
    ("Atraxa, Praetors' Voice", 12, ()),
    ("Aerathi Berserker", None, ()),          # Oracle dropped the Æ; users still type it
    ("Aether Vial", 5000, ()),
    ("Jötun Grunt", 9000, ()),
    ("Fire // Ice", 12829, ("Fire", "Ice")),
    ("Fireball", 800, ()),
    ("Fire Covenant", 4100, ()),
    ("Brazen Borrower // Petty Theft", 300, ("Brazen Borrower", "Petty Theft")),
    ("Delver of Secrets // Insectile Aberration", 15955,
     ("Delver of Secrets", "Insectile Aberration")),
    ("Ancestral Recall", 20000, ()),
    ("Ancestral Vision", 6000, ()),
    ("Lightning Bolt", 900, ()),
    ("Lightning Greaves", 30, ()),
    ("Path to Exile", 250, ()),
    ("Path of Ancestry", 100, ()),
    ("Pathbreaker Ibex", 3000, ()),
    ("Counterspell", 400, ()),
    ("Bolt Hound", None, ()),
    ("Boltwing Marauder", 7000, ()),
    ("Taiga", 1500, ()),
)


@pytest.fixture(scope="module")
def index():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    oracle_db.init_schema(conn)
    for position, (name, rank, faces) in enumerate(CORPUS):
        oracle_id = f"o{position:03d}"
        conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm, edhrec_rank) VALUES (?, ?, ?, ?)",
            (oracle_id, name, fold(name), rank),
        )
        for face_index, face_name in enumerate(faces):
            conn.execute(
                "INSERT INTO card_faces(oracle_id, face_index, name, name_norm) "
                "VALUES (?, ?, ?, ?)",
                (oracle_id, face_index, face_name, fold(face_name)),
            )
    conn.commit()
    built = oracle_names.build_index(conn)
    yield built
    conn.close()


def name_of(index, resolution) -> str | None:
    return index.display[resolution.oracle_id] if resolution.resolved else None


def names_of(index, resolution) -> list[str]:
    return [index.display[oid] for oid in resolution.oracle_ids]


# ------------------------------------------------------------------------------ fold


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sol Ring", "sol ring"),
        ("SOL RING", "sol ring"),
        ("  Sol   Ring  ", "sol ring"),
        # Apostrophes are DELETED, not spaced, so "Gaeas Cradle" matches. Same
        # convention cts/links.py already uses for EDHREC slugs.
        ("Gaea's Cradle", "gaeas cradle"),
        ("Gaea’s Cradle", "gaeas cradle"),
        ("Atraxa, Praetors' Voice", "atraxa praetors voice"),
        # NFKD strips the combining marks.
        ("Juzám Djinn", "juzam djinn"),
        ("Lim-Dûl's Vault", "lim duls vault"),
        ("Jötun Grunt", "jotun grunt"),
        ("Márton Stromgald", "marton stromgald"),
        # The // face separator becomes a space like any other punctuation.
        ("Fire // Ice", "fire ice"),
        # Ligatures, which NFKD does not touch.
        ("Ærathi Berserker", "aerathi berserker"),
        ("Æther Vial", "aether vial"),
        ("Œuvre", "oeuvre"),
        ("Ætherling", "aetherling"),
    ],
)
def test_fold_produces_the_matching_key(raw, expected):
    assert fold(raw) == expected


def test_nfkd_alone_does_not_decompose_the_ae_ligature():
    """The reason `LIGATURES` exists, asserted rather than assumed.

    `Æ` is an atomic letter in Unicode, not a composed one, so NFKD leaves it
    intact and a fold built on NFKD alone would drop it entirely — making
    `Æther Vial` unreachable for anyone who types the ligature. If someone ever
    deletes the map as redundant, this test says why it is not.
    """
    assert "Æ" in unicodedata.normalize("NFKD", "Ærathi")
    assert "Æ" not in fold("Ærathi")
    assert fold("Ærathi") == "aerathi"


def test_fold_is_idempotent():
    for name, _, _ in CORPUS:
        assert fold(fold(name)) == fold(name)


# ----------------------------------------------------------------- layer by layer


def test_l0_matches_raw_bytes_including_the_face_separator(index):
    resolution = resolve(index, "Fire // Ice")
    assert resolution.layer == "L0"
    assert name_of(index, resolution) == "Fire // Ice"


def test_l1_folds_case_punctuation_and_accents(index):
    for typed in ("gaeas cradle", "GAEA'S CRADLE", "Gaea’s  Cradle"):
        resolution = resolve(index, typed)
        assert resolution.layer == "L1", typed
        assert name_of(index, resolution) == "Gaea's Cradle"

    resolution = resolve(index, "lim duls vault")
    assert resolution.layer == "L1"
    assert name_of(index, resolution) == "Lim-Dûl's Vault"


def test_l1_reaches_a_card_whose_oracle_name_dropped_the_ligature(index):
    """Wizards' current Oracle names spell these "Ae", so the ligature has to be
    folded on the *input* side for a user typing the printed name to arrive."""
    resolution = resolve(index, "Æther Vial")
    assert resolution.layer == "L1"
    assert name_of(index, resolution) == "Aether Vial"

    resolution = resolve(index, "Ærathi Berserker")
    assert resolution.layer == "L1"
    assert name_of(index, resolution) == "Aerathi Berserker"


def test_l2_resolves_a_face_name(index):
    for typed, expected in (
        ("Petty Theft", "Brazen Borrower // Petty Theft"),
        ("insectile aberration", "Delver of Secrets // Insectile Aberration"),
        ("Brazen Borrower", "Brazen Borrower // Petty Theft"),
    ):
        resolution = resolve(index, typed)
        assert resolution.layer == "L2", typed
        assert name_of(index, resolution) == expected


def test_l3_matches_a_folded_prefix(index):
    resolution = resolve(index, "atraxa praetors")
    assert resolution.layer == "L3"
    assert name_of(index, resolution) == "Atraxa, Praetors' Voice"


def test_l4_matches_tokens_in_the_wrong_order(index):
    for typed in ("voice atraxa", "praetors atraxa voice"):
        resolution = resolve(index, typed)
        assert resolution.layer == "L4", typed
        assert name_of(index, resolution) == "Atraxa, Praetors' Voice"

    resolution = resolve(index, "cradle gaeas")
    assert resolution.layer == "L4"
    assert name_of(index, resolution) == "Gaea's Cradle"


def test_l5_absorbs_a_genuine_typo(index):
    resolution = resolve(index, "lightnig bolt")
    assert resolution.layer == "L5"
    assert resolution.distance == 1
    assert name_of(index, resolution) == "Lightning Bolt"

    resolution = resolve(index, "ancestrl vsion")
    assert resolution.layer == "L5"
    assert resolution.distance == 2
    assert name_of(index, resolution) == "Ancestral Vision"


def test_a_name_that_matches_nothing_resolves_to_nothing(index):
    resolution = resolve(index, "qwertyuiop asdfghjkl")
    assert resolution.layer is None
    assert resolution.oracle_ids == ()
    assert resolution.total == 0
    assert not resolution.resolved


# ----------------------------------------------- the strict ladder, which is the point


def test_a_layer_only_fires_when_every_layer_above_it_missed(index):
    """Every card's own exact name resolves at L0 and never below it."""
    for name, _, _ in CORPUS:
        resolution = resolve(index, name)
        assert resolution.layer == "L0", f"{name} fell through to {resolution.layer}"
        assert name_of(index, resolution) == name


def test_ancestral_recall_is_never_corrected_to_ancestral_vision(index):
    """The failure mode that matters, named in the design document.

    The two names are two edit operations apart and Vision is far more played.
    An eager fuzzy layer would "helpfully" resolve a correctly-typed Recall to
    it. Strict ordering makes that impossible: Recall has an L0 hit, so L5 is
    never reached, so popularity is never consulted.
    """
    resolution = resolve(index, "Ancestral Recall")
    assert resolution.layer == "L0"
    assert name_of(index, resolution) == "Ancestral Recall"

    lowered = resolve(index, "ancestral recall")
    assert lowered.layer == "L1"
    assert name_of(index, lowered) == "Ancestral Recall"

    # And the reverse: Vision is not swallowed by Recall either.
    assert name_of(index, resolve(index, "Ancestral Vision")) == "Ancestral Vision"


def test_fire_resolves_at_l2_rather_than_prefix_matching_into_fireball(index):
    """`Fire` is a face of `Fire // Ice` and a strict prefix of `Fireball` and
    `Fire Covenant`. L2 fires first, so the prefix layer never sees it."""
    resolution = resolve(index, "Fire")
    assert resolution.layer == "L2"
    assert "Fire // Ice" in names_of(index, resolution)
    assert "Fireball" not in names_of(index, resolution)
    assert "Fire Covenant" not in names_of(index, resolution)


def test_an_exact_name_that_is_also_a_prefix_stops_at_the_exact_layer(index):
    """`Taiga` is exact; `Path to Exile` is exact and shares a prefix with two
    others. Neither may be widened into a list."""
    for name in ("Taiga", "Path to Exile"):
        resolution = resolve(index, name)
        assert resolution.layer == "L0"
        assert resolution.resolved
        assert name_of(index, resolution) == name


def test_a_query_two_edits_from_a_card_name_never_reaches_l5_when_a_prefix_matches(index):
    """"counter target spell" is close to Counterspell, but the guard that keeps
    /oracle honest is the same short-circuit: layers above L5 win outright."""
    resolution = resolve(index, "counterspel")     # prefix of "counterspell"
    assert resolution.layer == "L3"
    assert name_of(index, resolution) == "Counterspell"


def test_max_layer_stops_the_ladder_where_the_oracle_guard_needs_it(index):
    """/oracle's card-name guard runs L0-L2 only. Letting it reach the fuzzy
    layers would make it fire on genuine mechanical queries that merely sit near
    a card name, and refuse to search at all."""
    assert resolve(index, "lightnig bolt", max_layer=2).layer is None
    assert resolve(index, "atraxa praetors", max_layer=2).layer is None
    assert resolve(index, "Sol Ring", max_layer=2).layer == "L0"
    assert resolve(index, "petty theft", max_layer=2).layer == "L2"


# ------------------------------------------------------------------------ ambiguity


def test_two_to_ten_hits_return_candidates_rather_than_a_card(index):
    resolution = resolve(index, "path")
    assert resolution.layer == "L3"
    assert not resolution.resolved
    assert resolution.total == 3
    assert set(names_of(index, resolution)) == {
        "Path to Exile", "Path of Ancestry", "Pathbreaker Ibex"
    }


def test_candidates_are_ordered_by_popularity_within_the_layer_that_fired(index):
    """`edhrec_rank`, lower is more played. Only ever a tie-break *inside* one
    layer — never a reason to prefer one layer's hit over another's."""
    resolution = resolve(index, "path")
    assert names_of(index, resolution) == [
        "Path of Ancestry",     # rank 100
        "Path to Exile",        # rank 250
        "Pathbreaker Ibex",     # rank 3000
    ]


def test_unranked_cards_sort_last(index):
    resolution = resolve(index, "bolt")
    assert resolution.layer == "L3"
    # Bolt Hound has no rank at all; an unranked card is not more played than a
    # card ranked 7,000th, so it sorts behind one.
    assert names_of(index, resolution) == ["Boltwing Marauder", "Bolt Hound"]


def test_more_than_ten_hits_truncate_but_report_the_true_total():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    oracle_db.init_schema(conn)
    for i in range(41):
        conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm, edhrec_rank) VALUES (?, ?, ?, ?)",
            (f"b{i}", f"Bolt {i:02d}", fold(f"Bolt {i:02d}"), i),
        )
    conn.commit()
    try:
        index = oracle_names.build_index(conn)
        resolution = resolve(index, "bolt")
        assert resolution.total == 41
        assert len(resolution.oracle_ids) == oracle_names.MAX_CANDIDATES == 10
        assert not resolution.resolved
        # The ten most played, in order.
        assert [index.display[o] for o in resolution.oracle_ids] == [
            f"Bolt {i:02d}" for i in range(10)
        ]
    finally:
        conn.close()


# ------------------------------------------------------------------ distance bounds


@pytest.mark.parametrize(
    "query,expected",
    [("bolt", 1), ("sol", 1), ("aether", 1), ("solring", 2), ("lightning", 2),
     ("lightning bolt", 3), ("atraxa praetors voice", 3)],
)
def test_max_distance_scales_with_input_length(query, expected):
    """1 for ≤4 characters, 2 for 5-8, 3 for 9+, never more than 30% of the input.
    Without the cap, `Bolt` sits within distance 4 of a large slice of the corpus
    and L5 answers confidently with noise."""
    assert oracle_names.max_distance_for(query) == expected


def test_bolt_does_not_fuzzy_match_half_the_corpus(index):
    """A four-character input gets a budget of one edit, so `Bolt` cannot reach
    `Colt`-shaped neighbours in bulk. Here it prefix-matches instead, which is
    the right answer and a layer above L5 anyway."""
    resolution = resolve(index, "bolt")
    assert resolution.layer == "L3"
    assert all("Bolt" in name or "bolt" in name.lower()
               for name in names_of(index, resolution))


def test_bounded_distance_agrees_with_a_reference_implementation():
    def reference(a: str, b: str) -> int:
        previous = list(range(len(b) + 1))
        for i, ch_a in enumerate(a, 1):
            current = [i]
            for j, ch_b in enumerate(b, 1):
                current.append(min(previous[j] + 1, current[j - 1] + 1,
                                   previous[j - 1] + (ch_a != ch_b)))
            previous = current
        return previous[-1]

    pairs = [
        ("lightning bolt", "lightnig bolt"),
        ("ancestral vision", "ancestrl vsion"),
        ("sol ring", "sol ring"),
        ("sol ring", "solring"),
        ("", "abc"),
        ("abc", ""),
        ("kitten", "sitting"),
        ("atraxa praetors voice", "atraxa praetor voice"),
    ]
    for a, b in pairs:
        truth = reference(a, b)
        for budget in range(0, 6):
            got = oracle_names.bounded_distance(a, b, budget)
            if truth <= budget:
                assert got == truth, (a, b, budget)
            else:
                assert got is None, (a, b, budget)


# ------------------------------------------------------------------------ disclosure


def test_exactness_is_reported_so_the_renderer_can_disclose_a_reinterpretation(index):
    """L0-L2 matched what was typed. L3-L5 changed the interpretation, and
    silently correcting input is how a reader ends up looking at the wrong card's
    text and never noticing."""
    assert resolve(index, "Sol Ring").exact
    assert resolve(index, "sol ring").exact
    assert resolve(index, "petty theft").exact
    assert not resolve(index, "atraxa praetors").exact
    assert not resolve(index, "voice atraxa").exact
    assert not resolve(index, "lightnig bolt").exact


def test_l5_reports_the_edit_distance_it_used(index):
    assert resolve(index, "lightnig bolt").distance == 1
    assert resolve(index, "ancestrl vsion").distance == 2
    assert resolve(index, "Sol Ring").distance is None


# --------------------------------------------------------------------- reading a card


def test_card_payload_carries_faces_legalities_and_only_the_links_that_exist():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    oracle_db.init_schema(conn)
    conn.execute(
        "INSERT INTO cards(oracle_id, name, name_norm, oracle_text, scryfall_uri, "
        "related_edhrec, purchase_tcgplayer) VALUES "
        "('o1', 'Fire // Ice', 'fire ice', 'Fire deals 2 damage.\n//\nTap two target permanents.', "
        "'https://scryfall.com/card/x', NULL, NULL)"
    )
    conn.execute("INSERT INTO card_faces(oracle_id, face_index, name) VALUES ('o1', 0, 'Fire')")
    conn.execute("INSERT INTO card_faces(oracle_id, face_index, name) VALUES ('o1', 1, 'Ice')")
    conn.execute("INSERT INTO card_legalities(oracle_id, format, status) "
                 "VALUES ('o1', 'commander', 'legal')")
    conn.commit()
    try:
        payload = oracle_names.card_payload(conn, "o1")
        assert payload["name"] == "Fire // Ice"
        assert [f["name"] for f in payload["faces"]] == ["Fire", "Ice"]
        assert payload["legalities"] == {"commander": "legal"}
        # An absent reference is an absent key, never a guessed URL.
        assert payload["links"] == {"scryfall": "https://scryfall.com/card/x"}
        assert oracle_names.card_payload(conn, "nope") is None
    finally:
        conn.close()


def test_candidate_briefs_keep_the_order_they_were_given(index):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    oracle_db.init_schema(conn)
    for oid, name in (("a", "Path of Ancestry"), ("b", "Path to Exile")):
        conn.execute("INSERT INTO cards(oracle_id, name, name_norm, mana_cost, type_line) "
                     "VALUES (?, ?, ?, '{W}', 'Instant')", (oid, name, fold(name)))
    conn.commit()
    try:
        briefs = oracle_names.candidate_briefs(conn, ["b", "a"])
        assert [b["name"] for b in briefs] == ["Path to Exile", "Path of Ancestry"]
        assert briefs[0]["mana_cost"] == "{W}"
        assert oracle_names.candidate_briefs(conn, []) == []
    finally:
        conn.close()


def test_an_empty_or_punctuation_only_query_resolves_to_nothing(index):
    for typed in ("", "   ", "///", "!!!"):
        resolution = resolve(index, typed)
        assert resolution.layer is None, typed
        assert resolution.total == 0
