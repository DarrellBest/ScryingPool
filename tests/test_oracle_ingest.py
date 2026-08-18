"""Field extraction from `oracle_cards` objects. No network, no downloads.

Every card dict below is trimmed from the real bulk file, so the shapes are
Scryfall's rather than invented — the same convention `conftest.py`'s hand-picked
CORPUS uses, and for the same reason: a fixture with made-up structure would test
nothing about the structure that actually arrives.

Two things in here are the reason the file exists:

* **the `image_uris` layout trap**, which is only visible against three different
  real layouts and blanks a whole card class when it is got backwards; and
* **the paper filter**, which was specified as `"paper" in games` and drops Taiga,
  Timetwister and ~1,060 other real cards when run against the live file, because
  `games` describes the representative printing rather than the card.
"""

from __future__ import annotations

import sqlite3

from cts import oracle_db, oracle_ingest
from cts.config import Config

# --------------------------------------------------------------------------- fixtures

# layout "normal": image_uris at the top level, no faces at all.
SOL_RING = {
    "oracle_id": "16de1b58-3e75-4d24-8b17-d7dcae1e4b4e",
    "name": "Sol Ring",
    "layout": "normal",
    "games": ["paper", "mtgo"],
    "set": "msc",
    "set_type": "masters",
    "type_line": "Artifact",
    "oracle_text": "{T}: Add {C}{C}.",
    "mana_cost": "{1}",
    "cmc": 1.0,
    "colors": [],
    "color_identity": [],
    "keywords": [],
    "reserved": False,
    "edhrec_rank": 1,
    "released_at": "2026-01-01",
    "rarity": "uncommon",
    "image_uris": {"normal": "https://cards.scryfall.io/normal/front/sol.jpg"},
    "prices": {"usd": "1.60", "usd_foil": None},
    "scryfall_uri": "https://scryfall.com/card/msc/1/sol-ring",
    "related_uris": {"edhrec": "https://edhrec.com/route/?cc=Sol+Ring"},
    "purchase_uris": {"tcgplayer": "https://www.tcgplayer.com/product/1"},
    "legalities": {"commander": "legal", "legacy": "banned", "vintage": "restricted"},
}

# layout "adventure": image_uris at the top level, and faces that have NONE.
BRAZEN_BORROWER = {
    "oracle_id": "c7b044c3-3cfa-407e-bf20-2875e8e04b7b",
    "name": "Brazen Borrower // Petty Theft",
    "layout": "adventure",
    "games": ["paper", "arena", "mtgo"],
    "set": "eld",
    "set_type": "expansion",
    "type_line": "Creature — Faerie Rogue // Instant — Adventure",
    "oracle_text": "",
    "mana_cost": "{1}{U}{U} // {1}{U}",
    "cmc": 3.0,
    "colors": ["U"],
    "color_identity": ["U"],
    "keywords": ["Flash", "Flying"],
    "edhrec_rank": 300,
    "rarity": "mythic",
    "image_uris": {"normal": "https://cards.scryfall.io/normal/front/brazen.jpg"},
    "card_faces": [
        {
            "name": "Brazen Borrower",
            "mana_cost": "{1}{U}{U}",
            "type_line": "Creature — Faerie Rogue",
            "oracle_text": "Flash\nFlying\nThis creature can block only creatures with flying.",
            "power": "3",
            "toughness": "1",
        },
        {
            "name": "Petty Theft",
            "mana_cost": "{1}{U}",
            "type_line": "Instant — Adventure",
            "oracle_text": "Return target nonland permanent an opponent controls to its owner's hand.",
        },
    ],
    "legalities": {"commander": "legal", "modern": "legal"},
}

# layout "transform": NO top-level image_uris at all; both faces carry their own.
DELVER = {
    "oracle_id": "edd531b9-f615-4399-8c8c-1c5e18c4acbf",
    "name": "Delver of Secrets // Insectile Aberration",
    "layout": "transform",
    "games": ["paper", "mtgo"],
    "set": "inr",
    "set_type": "expansion",
    "type_line": "Creature — Human Wizard // Creature — Human Insect",
    "cmc": 1.0,
    "color_identity": ["U"],
    "edhrec_rank": 15955,
    "rarity": "common",
    "card_faces": [
        {
            "name": "Delver of Secrets",
            "mana_cost": "{U}",
            "type_line": "Creature — Human Wizard",
            "oracle_text": "At the beginning of your upkeep, look at the top card of your library.",
            "power": "1",
            "toughness": "1",
            "colors": ["U"],
            "image_uris": {"normal": "https://cards.scryfall.io/normal/front/delver.jpg"},
        },
        {
            "name": "Insectile Aberration",
            "type_line": "Creature — Human Insect",
            "oracle_text": "Flying",
            "power": "3",
            "toughness": "2",
            "colors": ["U"],
            "image_uris": {"normal": "https://cards.scryfall.io/normal/back/aberration.jpg"},
        },
    ],
    "legalities": {"commander": "legal"},
}

# The representative printing Scryfall picked for Taiga is an MTGO-only Masters
# Edition reprint. The card is a Revised dual land.
TAIGA = {
    "oracle_id": "e2e1c1b0-0000-4000-8000-000000000001",
    "name": "Taiga",
    "layout": "normal",
    "games": ["mtgo"],
    "set": "me4",
    "set_type": "masters",
    "type_line": "Land — Mountain Forest",
    "oracle_text": "({T}: Add {R} or {G}.)",
    "cmc": 0.0,
    "color_identity": ["R", "G"],
    "image_uris": {"normal": "https://cards.scryfall.io/normal/front/taiga.jpg"},
    "legalities": {"commander": "legal", "legacy": "legal"},
}

ALCHEMY_REBALANCE = {
    "oracle_id": "e2e1c1b0-0000-4000-8000-000000000002",
    "name": "A-Sorin, Imperious Bloodlord",
    "layout": "normal",
    "games": ["arena"],
    "set": "ym20",
    "set_type": "alchemy",
    "type_line": "Legendary Planeswalker — Sorin",
    "loyalty": "4",
    "legalities": {"alchemy": "legal", "commander": "not_legal"},
}

STAR_POWER = {   # power that is not a number at all
    "oracle_id": "e2e1c1b0-0000-4000-8000-000000000003",
    "name": "Tarmogoyf",
    "layout": "normal",
    "games": ["paper"],
    "set": "fut",
    "set_type": "expansion",
    "type_line": "Creature — Lhurgoyf",
    "oracle_text": "This creature's power is equal to the number of card types among cards "
                   "in all graveyards and its toughness is equal to that number plus 1.",
    "power": "*",
    "toughness": "1+*",
    "cmc": 2.0,
    "color_identity": ["G"],
    "legalities": {"modern": "legal"},
}


def _cfg(tmp_path) -> Config:
    return Config(
        ollama_url="u", vision_model="v", verify_model="v", embed_model="e",
        judge_model="j", db_path=str(tmp_path / "commanders.db"),
        art_dir=str(tmp_path / "art"), power_weights={},
        oracle_db_path=str(tmp_path / "oracle.db"),
    )


def _written(tmp_path, cards) -> sqlite3.Connection:
    conn = oracle_db.connect(_cfg(tmp_path))
    oracle_ingest.write(conn, cards)
    return conn


# ----------------------------------------------------------------- the layout trap


def test_image_uris_normal_layout_comes_from_the_top_level():
    assert oracle_ingest.image_normal(SOL_RING) == (
        "https://cards.scryfall.io/normal/front/sol.jpg"
    )


def test_image_uris_adventure_layout_comes_from_the_top_level_despite_faces():
    """Adventure cards have faces, and those faces have no images. Reading the
    face first would blank every adventure card in the corpus."""
    assert BRAZEN_BORROWER["card_faces"][0].get("image_uris") is None
    assert oracle_ingest.image_normal(BRAZEN_BORROWER) == (
        "https://cards.scryfall.io/normal/front/brazen.jpg"
    )


def test_image_uris_transform_layout_has_none_at_the_top_and_falls_back_to_the_face():
    """The branch that only transform cards exercise, and the one that blanks a
    whole card class when it is missing."""
    assert "image_uris" not in DELVER
    assert oracle_ingest.image_normal(DELVER) == (
        "https://cards.scryfall.io/normal/front/delver.jpg"
    )


def test_all_three_layouts_land_a_non_null_image_in_the_row(tmp_path):
    conn = _written(tmp_path, [SOL_RING, BRAZEN_BORROWER, DELVER])
    try:
        rows = dict(conn.execute("SELECT name, image_normal FROM cards"))
        assert all(rows.values()), rows
        # And the back face's image is stored too, so a future "flip" costs nothing.
        backs = conn.execute(
            "SELECT image_normal FROM card_faces WHERE oracle_id = ? AND face_index = 1",
            (DELVER["oracle_id"],),
        ).fetchone()
        assert backs[0].endswith("aberration.jpg")
    finally:
        conn.close()


# -------------------------------------------------------------------- paper filter


def test_paper_cards_are_kept():
    assert oracle_ingest.is_paper_card(SOL_RING)
    assert oracle_ingest.is_paper_card(STAR_POWER)


def test_an_mtgo_only_reprint_of_a_paper_card_is_kept():
    """The regression that matters. `oracle_cards` carries one printing per Oracle
    ID, and for Taiga, Timetwister, Library of Alexandria, Strip Mine and ~1,060
    others that printing is an MTGO-only Masters Edition reprint. Filtering on
    `"paper" in games` drops all of them while leaving a card count that still
    looks about right."""
    assert TAIGA["games"] == ["mtgo"]
    assert oracle_ingest.is_paper_card(TAIGA)


def test_arena_only_cards_are_dropped():
    """Alchemy rebalances and Arena originals are not paper Magic. Keeping them
    would double some card names with different text and make results ambiguous."""
    assert not oracle_ingest.is_paper_card(ALCHEMY_REBALANCE)
    assert not oracle_ingest.is_paper_card(dict(SOL_RING, games=["arena"]))


def test_astral_and_sega_curiosities_are_dropped():
    assert not oracle_ingest.is_paper_card(dict(SOL_RING, games=["astral"]))
    assert not oracle_ingest.is_paper_card(dict(SOL_RING, games=["sega"]))


def test_tokens_emblems_and_art_series_are_dropped_by_layout_and_set_type():
    for layout in ("token", "double_faced_token", "emblem", "art_series"):
        assert not oracle_ingest.is_paper_card(dict(SOL_RING, layout=layout)), layout
    for set_type in ("token", "memorabilia"):
        assert not oracle_ingest.is_paper_card(dict(SOL_RING, set_type=set_type)), set_type


# ------------------------------------------------------------------ field extraction


def test_numeric_columns_are_null_for_star_and_plus_powers():
    """`power >= 5` matches on power_num and therefore excludes every `*`/`X`
    creature. A `*/*` power is evaluated against a board state this corpus does
    not have, so the honest answer is NULL rather than a guess."""
    assert oracle_ingest.numeric("3") == 3.0
    assert oracle_ingest.numeric("0") == 0.0
    assert oracle_ingest.numeric("*") is None
    assert oracle_ingest.numeric("1+*") is None
    assert oracle_ingest.numeric("X") is None
    assert oracle_ingest.numeric(None) is None
    assert oracle_ingest.numeric("") is None


def test_star_power_keeps_the_verbatim_text_and_nulls_the_number(tmp_path):
    conn = _written(tmp_path, [STAR_POWER])
    try:
        row = conn.execute("SELECT power, toughness, power_num, toughness_num FROM cards").fetchone()
        assert (row["power"], row["toughness"]) == ("*", "1+*")
        assert row["power_num"] is None and row["toughness_num"] is None
    finally:
        conn.close()


def test_transform_cards_join_their_faces_the_way_ingest_py_does(tmp_path):
    conn = _written(tmp_path, [DELVER])
    try:
        row = conn.execute("SELECT oracle_text, mana_cost, colors, power FROM cards").fetchone()
        assert "\n//\n" in row["oracle_text"]
        assert row["oracle_text"].endswith("Flying")
        assert row["mana_cost"] == "{U}"        # the castable, front-face cost
        assert row["colors"] == "U"             # merged from the faces
        assert row["power"] == "1"              # the front face's
    finally:
        conn.close()


def test_faces_carry_their_own_names_folded_for_l2(tmp_path):
    conn = _written(tmp_path, [BRAZEN_BORROWER])
    try:
        rows = conn.execute(
            "SELECT face_index, name, name_norm FROM card_faces ORDER BY face_index"
        ).fetchall()
        assert [(r["face_index"], r["name"], r["name_norm"]) for r in rows] == [
            (0, "Brazen Borrower", "brazen borrower"),
            (1, "Petty Theft", "petty theft"),
        ]
    finally:
        conn.close()


def test_types_are_split_by_kind_across_both_halves_of_a_type_line():
    assert oracle_ingest.parse_types("Legendary Creature — Angel") == [
        ("supertype", "legendary"), ("type", "creature"), ("subtype", "angel"),
    ]
    assert oracle_ingest.parse_types(
        "Creature — Faerie Rogue // Instant — Adventure"
    ) == [
        ("type", "creature"), ("subtype", "faerie"), ("subtype", "rogue"),
        ("type", "instant"), ("subtype", "adventure"),
    ]
    assert oracle_ingest.parse_types("Basic Snow Land — Forest") == [
        ("supertype", "basic"), ("supertype", "snow"), ("type", "land"),
        ("subtype", "forest"),
    ]


def test_links_are_stored_from_scryfalls_own_keys_and_absent_when_absent(tmp_path):
    conn = _written(tmp_path, [SOL_RING, DELVER])
    try:
        sol = conn.execute(
            "SELECT related_edhrec, purchase_tcgplayer FROM cards WHERE name = 'Sol Ring'"
        ).fetchone()
        assert sol["related_edhrec"] == "https://edhrec.com/route/?cc=Sol+Ring"
        assert sol["purchase_tcgplayer"] == "https://www.tcgplayer.com/product/1"
        # Delver's fixture has neither key: an absent link is absent, never guessed
        # from the card name.
        delver = conn.execute(
            "SELECT related_edhrec, purchase_tcgplayer FROM cards WHERE name LIKE 'Delver%'"
        ).fetchone()
        assert delver["related_edhrec"] is None
        assert delver["purchase_tcgplayer"] is None
    finally:
        conn.close()


def test_prices_land_as_numbers_and_missing_prices_as_null(tmp_path):
    conn = _written(tmp_path, [SOL_RING])
    try:
        row = conn.execute("SELECT price_usd, price_usd_foil FROM cards").fetchone()
        assert row["price_usd"] == 1.60
        assert row["price_usd_foil"] is None
    finally:
        conn.close()


def test_legalities_land_one_row_per_format(tmp_path):
    conn = _written(tmp_path, [SOL_RING])
    try:
        rows = dict(conn.execute("SELECT format, status FROM card_legalities"))
        assert rows == {"commander": "legal", "legacy": "banned", "vintage": "restricted"}
    finally:
        conn.close()


# ------------------------------------------------------------------------ idempotence


def test_a_second_write_replaces_rather_than_duplicates(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        first = oracle_ingest.write(conn, [SOL_RING, DELVER])
        second = oracle_ingest.write(conn, [SOL_RING, DELVER])
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("cards", "card_faces", "card_types", "card_legalities")
        }
        assert counts["cards"] == 2
        assert counts["card_faces"] == 2          # Delver's two; Sol Ring has none
        assert first["cards"] == second["cards"] == 2
        assert len(second["new_cards"]) == 0      # nothing is new the second time
        assert second["changed_text"] == []
    finally:
        conn.close()


def test_changed_oracle_text_is_reported_so_a_re_chunk_knows_what_moved(tmp_path):
    """Oracle text is not immutable the way artwork is — Wizards issues errata and
    templating normalisations — so the set of cards whose text moved is exactly
    what a re-chunking stage has to redo."""
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        oracle_ingest.write(conn, [SOL_RING])
        errata = dict(SOL_RING, oracle_text="{T}: Add {C}{C}. (errata)")
        result = oracle_ingest.write(conn, [errata])
        assert result["changed_text"] == [SOL_RING["oracle_id"]]
        assert result["new_cards"] == []
    finally:
        conn.close()


def test_a_card_that_leaves_the_corpus_takes_its_chunks_with_it(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        oracle_ingest.write(conn, [SOL_RING, DELVER])
        conn.execute(
            "INSERT INTO chunks(id, oracle_id, face_index, ordinal, kind, text) "
            "VALUES (1, ?, 0, 0, 'whole', 'x')", (DELVER["oracle_id"],)
        )
        conn.execute("INSERT INTO chunk_embeddings(chunk_id, vec) VALUES (1, X'00')")
        conn.commit()

        result = oracle_ingest.write(conn, [SOL_RING])
        assert result["removed"] == [DELVER["oracle_id"]]
        assert conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == 0
    finally:
        conn.close()


def test_bulk_entry_defaults_to_default_cards_and_takes_a_type():
    """The one edit made to `cts/ingest.py`: a parameter with the old constant as
    its default, so no existing caller changes behaviour."""
    import inspect

    from cts import ingest

    signature = inspect.signature(ingest.bulk_entry)
    assert signature.parameters["bulk_type"].default == ingest.BULK_TYPE == "default_cards"

    entry = {"jsonl_download_uri": "https://data.scryfall.io/x/y.jsonl.gz"}
    from pathlib import Path

    _, path = ingest._pick_source(entry, Path("data/bulk"), "oracle_cards")
    assert path.name == "oracle_cards.jsonl.gz"
    _, path = ingest._pick_source(entry, Path("data/bulk"))
    assert path.name == "default_cards.jsonl.gz"   # unchanged for the art pipeline


# ------------------------------------------------------------------ refresh wiring


def test_the_weekly_refresh_runs_oracle_ingest_and_runs_it_last(monkeypatch, tmp_path):
    """Appended at the end of the stage list, and the position is the point.

    `refresh.run` returns 1 on the first stage that raises, so newer and
    less-proven code placed anywhere but last would be able to block the art
    pipeline that has been working. Last, it cannot: every art stage is committed
    and done before this starts.
    """
    from cts import art, describe, edhrec, embed, ingest, oracle_ingest as oi, power, refresh

    order: list[str] = []

    def stub(name, result=None):
        def run(*args, **kwargs):
            order.append(name)
            return result or {}
        return run

    monkeypatch.setattr(refresh, "_preflight", lambda cfg: None)
    monkeypatch.setattr(ingest, "run", stub("ingest", {"new_cards": [], "new_arts": 0}))
    monkeypatch.setattr(edhrec, "run", stub("edhrec"))
    monkeypatch.setattr(power, "run", stub("power"))
    monkeypatch.setattr(art, "run", stub("art"))
    monkeypatch.setattr(describe, "run", stub("describe"))
    monkeypatch.setattr(embed, "run", stub("embed"))
    monkeypatch.setattr(oi, "run", stub("oracle-ingest", {"cards": 33933, "new_cards": ["X"]}))

    cfg = _cfg(tmp_path)
    assert refresh.run(cfg) == 0
    assert order[-1] == "oracle-ingest"
    assert order == ["ingest", "edhrec", "power", "art", "describe", "embed", "oracle-ingest"]


def test_a_failing_oracle_stage_does_not_undo_the_art_stages(monkeypatch, tmp_path):
    from cts import art, describe, edhrec, embed, ingest, oracle_ingest as oi, power, refresh

    order: list[str] = []

    def stub(name):
        def run(*args, **kwargs):
            order.append(name)
            return {}
        return run

    monkeypatch.setattr(refresh, "_preflight", lambda cfg: None)
    for module, name in ((ingest, "ingest"), (edhrec, "edhrec"), (power, "power"),
                         (art, "art"), (describe, "describe"), (embed, "embed")):
        monkeypatch.setattr(module, "run", stub(name))

    def explode(*args, **kwargs):
        raise RuntimeError("scryfall is down")

    monkeypatch.setattr(oi, "run", explode)

    assert refresh.run(_cfg(tmp_path)) == 1
    assert order == ["ingest", "edhrec", "power", "art", "describe", "embed"]
