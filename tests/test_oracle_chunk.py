"""The heaviest new file, matching where the risk is: chunking is a pure
function of a card dict, so it is table-driven over real card text — the same
convention `conftest.py`'s hand-picked CORPUS already uses.
"""

from __future__ import annotations

from cts import oracle_chunk, oracle_db
from cts.config import Config


def _cfg(tmp_path) -> Config:
    return Config(
        ollama_url="u", vision_model="v", verify_model="v", embed_model="e",
        judge_model="j", db_path=str(tmp_path / "commanders.db"),
        art_dir=str(tmp_path / "art"), power_weights={},
        oracle_db_path=str(tmp_path / "oracle.db"),
    )


# ------------------------------------------------------------------------ chunk_card


def test_a_vanilla_creature_yields_only_the_whole_card_chunk():
    chunks = oracle_chunk.chunk_card("o1", "Grizzly Bears", "Creature — Bear", "")
    assert len(chunks) == 1
    assert chunks[0]["kind"] == "whole"
    assert chunks[0]["text"] == "Creature — Bear"


def test_a_two_ability_enchantment_yields_two_ability_chunks_plus_one_whole():
    text = (
        "Whenever a creature you control dies, draw a card.\n"
        "At the beginning of your end step, you lose 1 life."
    )
    chunks = oracle_chunk.chunk_card("o1", "Grim Accounting", "Enchantment", text)
    abilities = [c for c in chunks if c["kind"] == "ability"]
    whole = [c for c in chunks if c["kind"] == "whole"]
    assert len(abilities) == 2
    assert len(whole) == 1
    assert abilities[0]["text"] == "Whenever a creature you control dies, draw a card."
    assert abilities[1]["text"] == "At the beginning of your end step, you lose 1 life."
    assert whole[0]["text"] == f"Enchantment\n{text}"


def test_a_saga_splits_every_chapter_as_its_own_ability():
    text = (
        "(As this Saga enters and after your draw step, add a lore counter.)\n"
        "I — Create a 2/2 black Zombie creature token.\n"
        "II — Each opponent loses 1 life and you gain 1 life.\n"
        "III — Return target creature card from your graveyard to your hand."
    )
    chunks = oracle_chunk.chunk_card("o1", "A Saga", "Enchantment — Saga", text)
    abilities = [c for c in chunks if c["kind"] == "ability"]
    assert len(abilities) == 4
    assert [c["ordinal"] for c in abilities] == [0, 1, 2, 3]


def test_a_dfc_splits_by_face_index_on_the_ingest_join():
    text = "At the beginning of your upkeep, look at the top card of your library.\n//\nFlying"
    chunks = oracle_chunk.chunk_card(
        "o1", "Delver of Secrets // Insectile Aberration",
        "Creature — Human Wizard // Creature — Human Insect", text,
    )
    abilities = [c for c in chunks if c["kind"] == "ability"]
    assert [(c["face_index"], c["text"]) for c in abilities] == [
        (0, "At the beginning of your upkeep, look at the top card of your library."),
        (1, "Flying"),
    ]
    # ordinals are contiguous from 0 WITHIN each face
    assert [c["ordinal"] for c in abilities] == [0, 0]


def test_the_cards_own_name_is_substituted_in_text_embedded_but_never_in_text():
    text = "Sylvan Library draws you extra cards, at a cost."
    chunks = oracle_chunk.chunk_card("o1", "Sylvan Library", "Enchantment", text)
    ability = [c for c in chunks if c["kind"] == "ability"][0]
    assert ability["text"] == text  # verbatim, untouched
    assert ability["text_embedded"] == "this card draws you extra cards, at a cost."


def test_the_short_name_before_a_comma_is_also_substituted():
    text = "Whenever Atraxa attacks, proliferate."
    chunks = oracle_chunk.chunk_card(
        "o1", "Atraxa, Praetors' Voice", "Legendary Creature — Phyrexian Angel Horror", text
    )
    ability = [c for c in chunks if c["kind"] == "ability"][0]
    assert ability["text_embedded"] == "Whenever this card attacks, proliferate."
    assert ability["text"] == text


def test_reminder_text_is_kept_not_stripped():
    """The single most load-bearing chunking choice after the newline split:
    reminder text is the only English gloss a keyword ability has."""
    text = "Cycling {2} ({2}, Discard this card: Draw a card.)"
    chunks = oracle_chunk.chunk_card("o1", "Some Card", "Instant", text)
    ability = [c for c in chunks if c["kind"] == "ability"][0]
    assert "Discard this card: Draw a card." in ability["text"]
    assert "Discard this card: Draw a card." in ability["text_embedded"]


def test_no_sentence_splitting_happens_on_costs_or_pt():
    text = "{T}: Add {C}{C}. This creature gets +1/+1 until end of turn."
    chunks = oracle_chunk.chunk_card("o1", "Weird Rock", "Artifact", text)
    abilities = [c for c in chunks if c["kind"] == "ability"]
    # One newline-delimited line stays fused, periods and all.
    assert len(abilities) == 1
    assert abilities[0]["text"] == text


def test_the_whole_chunk_carries_the_type_line_the_ability_chunks_do_not():
    text = "Flying"
    chunks = oracle_chunk.chunk_card("o1", "Some Flyer", "Creature — Bird", text)
    ability = [c for c in chunks if c["kind"] == "ability"][0]
    whole = [c for c in chunks if c["kind"] == "whole"][0]
    assert "Creature" not in ability["text"]
    assert whole["text"].startswith("Creature — Bird")


def test_ordinals_are_contiguous_from_zero_within_a_face():
    text = "A.\nB.\nC.\nD."
    chunks = oracle_chunk.chunk_card("o1", "Modal Thing", "Instant", text)
    abilities = [c for c in chunks if c["kind"] == "ability"]
    assert [c["ordinal"] for c in abilities] == [0, 1, 2, 3]


# ------------------------------------------------------------------------- name_variants


def test_name_variants_includes_the_full_name_and_the_short_form():
    assert oracle_chunk.name_variants("Atraxa, Praetors' Voice") == (
        "Atraxa, Praetors' Voice", "Atraxa",
    )


def test_name_variants_is_just_the_full_name_with_no_comma():
    assert oracle_chunk.name_variants("Sol Ring") == ("Sol Ring",)


def test_name_variants_of_blank_is_empty():
    assert oracle_chunk.name_variants("") == ()
    assert oracle_chunk.name_variants(None) == ()


def test_substitute_name_is_whole_word_only():
    """A name that is a substring of another word must not be mangled."""
    text = "Ashnod's Altar has nothing to do with Ash the trainer."
    out = oracle_chunk.substitute_name(text, "Ash")
    assert out == "Ashnod's Altar has nothing to do with this card the trainer."


# --------------------------------------------------------------------------- rechunk


SOL_RING = ("o-sol", "Sol Ring", "Artifact", "{T}: Add {C}{C}.")
GRIZZLY = ("o-bear", "Grizzly Bears", "Creature — Bear", "")


def _seed_cards(conn, rows):
    for oracle_id, name, type_line, oracle_text in rows:
        conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm, type_line, oracle_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (oracle_id, name, name.lower(), type_line, oracle_text),
        )
    conn.commit()


def test_rechunk_builds_chunks_for_every_card_with_none_yet(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        _seed_cards(conn, [SOL_RING, GRIZZLY])
        result = oracle_chunk.rechunk(conn)
        assert result["cards"] == 2
        # Sol Ring: 1 ability + 1 whole; Grizzly Bears: 0 ability + 1 whole.
        assert result["chunks"] == 3
        rows = conn.execute("SELECT oracle_id, kind FROM chunks ORDER BY oracle_id, kind").fetchall()
        assert len(rows) == 3
    finally:
        conn.close()


def test_rechunk_is_a_noop_the_second_time_with_nothing_changed(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        _seed_cards(conn, [SOL_RING])
        oracle_chunk.rechunk(conn)
        result = oracle_chunk.rechunk(conn)
        assert result == {"cards": 0, "chunks": 0}
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    finally:
        conn.close()


def test_rechunk_replaces_a_changed_cards_chunks_and_its_embeddings(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        _seed_cards(conn, [SOL_RING])
        oracle_chunk.rechunk(conn)
        chunk_id = conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]
        conn.execute("INSERT INTO chunk_embeddings(chunk_id, vec) VALUES (?, X'00')", (chunk_id,))
        conn.commit()

        conn.execute(
            "UPDATE cards SET oracle_text = ? WHERE oracle_id = 'o-sol'",
            ("{T}: Add {C}{C}. (errata)",),
        )
        conn.commit()

        result = oracle_chunk.rechunk(conn, changed_ids=["o-sol"])
        assert result["cards"] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == 0
        texts = {r[0] for r in conn.execute("SELECT text FROM chunks WHERE kind = 'ability'")}
        assert "{T}: Add {C}{C}. (errata)" in texts
    finally:
        conn.close()


def test_rechunk_with_nothing_to_do_returns_zeroes(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        result = oracle_chunk.rechunk(conn)
        assert result == {"cards": 0, "chunks": 0}
    finally:
        conn.close()


def test_run_opens_and_closes_its_own_connection(tmp_path):
    cfg = _cfg(tmp_path)
    conn = oracle_db.connect(cfg)
    try:
        _seed_cards(conn, [SOL_RING])
    finally:
        conn.close()
    result = oracle_chunk.run(cfg)
    assert result["cards"] == 1
