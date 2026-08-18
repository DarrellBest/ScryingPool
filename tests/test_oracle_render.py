"""Pure functions over canned `GET /card` bodies. Nothing imported beyond stdlib.

`serve/oracle_render.py` renders to plain dicts matching Discord's embed JSON
rather than to `discord.Embed` objects, exactly as `serve/render.py` does, which
is what lets this file run in a checkout with nothing installed but pytest.

The load-bearing cases here are the ones where being wrong is invisible in
review: the **image-URI layout trap** (three layouts, three code paths, one of
which blanks a whole card class), the **price as-of label** (a stale number
presented as current is worse than no number), and **absent links being absent**
rather than guessed.
"""

from __future__ import annotations

import pytest

from serve import oracle_render

# --------------------------------------------------------------------------- fixtures


def _card(**overrides) -> dict:
    base = {
        "oracle_id": "16de1b58-3e75-4d24-8b17-d7dcae1e4b4e",
        "name": "Sol Ring",
        "type_line": "Artifact",
        "oracle_text": "{T}: Add {C}{C}.",
        "mana_cost": "{1}",
        "cmc": 1.0,
        "color_identity": "",
        "layout": "normal",
        "set_code": "msc",
        "rarity": "uncommon",
        "image_normal": "https://cards.scryfall.io/normal/front/sol.jpg",
        "price_usd": 1.6,
        "price_usd_foil": None,
        "scryfall_uri": "https://scryfall.com/card/msc/1/sol-ring",
        "faces": [],
        "legalities": {"commander": "legal", "legacy": "banned", "vintage": "restricted"},
        "links": {
            "scryfall": "https://scryfall.com/card/msc/1/sol-ring",
            "edhrec": "https://edhrec.com/route/?cc=Sol+Ring",
            "tcgplayer": "https://www.tcgplayer.com/product/1",
        },
    }
    base.update(overrides)
    return base


def _payload(**overrides) -> dict:
    base = {
        "resolved": True,
        "layer": "L1",
        "input": "sol ring",
        "distance": None,
        "total": 1,
        "card": _card(),
        "candidates": [],
        "service": {"refreshed_at": "2026-08-17T03:43:02+00:00"},
    }
    base.update(overrides)
    return base


def _fields(embed: dict) -> dict:
    return {field["name"]: field["value"] for field in embed["fields"]}


# ------------------------------------------------------------------------- the embed


def test_the_card_embed_leads_with_the_verbatim_oracle_text():
    """The rules text is the answer to the question, so it is the description,
    verbatim and in a code block — Discord's markdown would otherwise eat `{T}`,
    `{1}{W}` and `+1/+1`."""
    embed = oracle_render.card_embed(_card())
    assert embed["title"] == "Sol Ring"
    assert "{T}: Add {C}{C}." in embed["description"]
    assert embed["description"].startswith("```")
    assert embed["url"] == "https://scryfall.com/card/msc/1/sol-ring"


def test_a_card_with_no_rules_text_says_so_rather_than_showing_an_empty_box():
    embed = oracle_render.card_embed(_card(oracle_text="", name="Grizzly Bears"))
    assert "no rules text" in embed["description"]


def test_both_faces_of_a_multi_face_card_are_shown_as_scryfall_separates_them():
    text = ("At the beginning of your upkeep, look at the top card of your library."
            "\n//\nFlying")
    embed = oracle_render.card_embed(
        _card(name="Delver of Secrets // Insectile Aberration", oracle_text=text)
    )
    assert "//" in embed["description"]
    assert "Flying" in embed["description"]
    assert embed["title"] == "Delver of Secrets // Insectile Aberration"


def test_the_type_line_and_cost_are_verbatim():
    fields = _fields(oracle_render.card_embed(_card()))
    assert fields["Type"] == "Artifact"
    assert fields["Cost"] == "{1} · MV 1"


def test_a_card_with_no_mana_cost_still_reports_its_mana_value():
    fields = _fields(oracle_render.card_embed(_card(mana_cost="", cmc=0.0)))
    assert fields["Cost"] == "MV 0"


def test_set_and_rarity_are_labelled_as_one_printing_not_the_card():
    """`oracle_cards` carries one printing per card, so Sol Ring reports whatever
    set Scryfall picked. The label says so rather than implying it is the only one."""
    fields = _fields(oracle_render.card_embed(_card()))
    assert "Set (one printing)" in fields
    assert fields["Set (one printing)"] == "MSC · uncommon"


# ------------------------------------------------------------------- the layout trap


@pytest.mark.parametrize(
    "layout,image",
    [
        # normal: top-level image_uris
        ("normal", "https://cards.scryfall.io/normal/front/sol.jpg"),
        # adventure: top-level image_uris present, faces have none
        ("adventure", "https://cards.scryfall.io/normal/front/brazen.jpg"),
        # transform: NO top-level image_uris at all, per-face only
        ("transform", "https://cards.scryfall.io/normal/front/delver.jpg"),
    ],
)
def test_every_layout_renders_an_image(layout, image):
    """All three land in `image_normal` at ingest — top-level when it exists, the
    front face's when it does not. Getting that rule backwards blanks every
    transform card in the corpus, so all three paths are exercised here as well
    as at the extraction site."""
    embed = oracle_render.card_embed(_card(layout=layout, image_normal=image))
    assert embed["image"] == {"url": image}


def test_a_card_with_no_image_url_renders_no_image_key_rather_than_an_empty_one():
    embed = oracle_render.card_embed(_card(image_normal=None))
    assert "image" not in embed


# ------------------------------------------------------------------------ legalities


def test_legal_formats_are_listed_and_bans_and_restrictions_named_explicitly():
    """A banned card is not "not legal" the way an un-set card is, and collapsing
    the two hides the one a deckbuilder needs."""
    line = oracle_render.legality_line(
        {"commander": "legal", "modern": "legal", "legacy": "banned",
         "vintage": "restricted", "standard": "not_legal"}
    )
    assert "Commander" in line and "Modern" in line
    assert "Banned in Legacy" in line
    assert "Restricted in Vintage" in line
    assert "Standard" not in line.split("·")[0]


def test_a_card_legal_nowhere_major_says_so():
    assert "not legal in the major formats" in oracle_render.legality_line(
        {"commander": "not_legal", "modern": "not_legal"}
    )


def test_unknown_legalities_are_reported_as_unknown_not_as_legal():
    assert oracle_render.legality_line(None) == "unknown"
    assert oracle_render.legality_line({}) == "unknown"


# ---------------------------------------------------------------------------- prices


def test_the_price_carries_its_as_of_date_and_never_claims_to_be_current():
    """Prices are a weekly snapshot, so they are up to seven days old, and
    Scryfall updates them once a day and disclaims their accuracy. This embed
    does not make a stronger claim than its source does."""
    line = oracle_render.price_line(_card(), "2026-08-17T03:43:02+00:00")
    assert line == "$1.60 (as of 2026-08-17)"
    with_foil = oracle_render.price_line(
        _card(price_usd=1.6, price_usd_foil=12.0), "2026-08-17T03:43:02+00:00"
    )
    assert with_foil == "$1.60 · foil $12.00 (as of 2026-08-17)"


def test_a_card_with_no_price_shows_no_price_field():
    embed = oracle_render.card_embed(_card(price_usd=None, price_usd_foil=None))
    assert "USD" not in _fields(embed)


def test_a_price_with_no_refresh_stamp_is_labelled_a_snapshot_not_a_date():
    assert oracle_render.price_line(_card(), None).endswith("(weekly snapshot)")


# ----------------------------------------------------------------------------- links


def test_every_link_that_exists_is_rendered():
    line = oracle_render.link_line(_card()["links"])
    assert "[Scryfall](https://scryfall.com/card/msc/1/sol-ring)" in line
    assert "[EDHREC](https://edhrec.com/route/?cc=Sol+Ring)" in line
    assert "[TCGplayer](https://www.tcgplayer.com/product/1)" in line


def test_a_link_that_cannot_be_built_is_omitted_never_guessed():
    """`cts/links.py`'s standing rule, carried over: EDHREC and TCGplayer URLs
    come from Scryfall's own `related_uris` / `purchase_uris`, so an absent key
    means an absent link rather than a slugified guess at a URL."""
    line = oracle_render.link_line({"scryfall": "https://scryfall.com/card/x"})
    assert line == "[Scryfall](https://scryfall.com/card/x)"
    assert "EDHREC" not in line and "TCGplayer" not in line
    assert oracle_render.link_line({}) == ""
    assert oracle_render.link_line(None) == ""

    embed = oracle_render.card_embed(_card(links={}))
    assert "Links" not in _fields(embed)


# ------------------------------------------------------------------------ disclosure


@pytest.mark.parametrize("layer", ["L0", "L1", "L2"])
def test_an_exact_resolution_makes_no_claim_about_reinterpreting_the_input(layer):
    embed = oracle_render.card_embed(
        _card(), layer=layer, typed="sol ring", refreshed_at="2026-08-17T03:43:02+00:00"
    )
    assert "matched" not in embed["footer"]["text"]
    assert "refreshed 2026-08-17" in embed["footer"]["text"]


def test_a_fuzzy_resolution_discloses_that_it_changed_the_input():
    """Silently correcting input is how a reader ends up looking at the wrong
    card's text and never noticing."""
    embed = oracle_render.card_embed(
        _card(name="Ancestral Vision"), layer="L5", typed="ancestrl vsion", distance=2
    )
    assert embed["footer"]["text"] == (
        'matched "Ancestral Vision" from your input "ancestrl vsion" (edit distance 2)'
    )


@pytest.mark.parametrize("layer", ["L3", "L4"])
def test_prefix_and_token_resolutions_disclose_too_without_a_distance(layer):
    note = oracle_render.resolution_note(layer, "voice atraxa", "Atraxa, Praetors' Voice")
    assert note == 'matched "Atraxa, Praetors\' Voice" from your input "voice atraxa"'
    assert "edit distance" not in note


# ------------------------------------------------------------------- disambiguation


def test_two_to_ten_hits_render_a_candidate_list_and_not_a_card():
    """A different render on purpose: showing one card and mentioning the others
    in small text invites the reader to accept the one on screen."""
    candidates = [
        {"name": "Path of Ancestry", "mana_cost": "", "type_line": "Land"},
        {"name": "Path to Exile", "mana_cost": "{W}", "type_line": "Instant"},
        {"name": "Pathbreaker Ibex", "mana_cost": "{4}{G}", "type_line": "Creature — Goat"},
    ]
    message = oracle_render.candidates_message("path", candidates, 3)
    assert message["content"] == '3 cards match "path". Re-run with a fuller name.'
    assert len(message["embeds"]) == 1
    body = message["embeds"][0]["description"]
    assert "Path of Ancestry" in body and "Path to Exile" in body
    assert "{4}{G}" in body and "Creature — Goat" in body
    # Not a card embed: no image, no oracle text, no Scryfall url.
    assert "image" not in message["embeds"][0]
    assert "url" not in message["embeds"][0]


def test_more_than_ten_hits_truncate_and_report_the_true_total():
    candidates = [{"name": f"Bolt {i}", "mana_cost": "{R}", "type_line": "Instant"}
                  for i in range(10)]
    message = oracle_render.candidates_message("bolt", candidates, 41)
    assert message["content"] == (
        '41 cards match "bolt" — showing the 10 most played. Be more specific.'
    )
    assert message["embeds"][0]["description"].count("\n") == 9


# ------------------------------------------------------------------- whole messages


def test_card_message_renders_one_embed_for_a_resolved_card():
    message = oracle_render.card_message(_payload())
    assert message["content"] == ""
    assert len(message["embeds"]) == 1
    assert message["embeds"][0]["title"] == "Sol Ring"


def test_card_message_renders_the_list_for_an_ambiguous_body():
    message = oracle_render.card_message(
        _payload(
            resolved=False, layer="L3", input="path", card=None, total=3,
            candidates=[{"name": "Path to Exile", "mana_cost": "{W}", "type_line": "Instant"},
                        {"name": "Path of Ancestry", "mana_cost": "", "type_line": "Land"},
                        {"name": "Pathbreaker Ibex", "mana_cost": "", "type_line": "Creature"}],
        )
    )
    assert "3 cards match" in message["content"]
    assert message["embeds"][0]["title"] == 'candidates for "path"'


def test_card_message_renders_an_honest_miss_with_no_embed_and_no_guess():
    message = oracle_render.card_message(
        _payload(resolved=False, layer=None, input="qwertyuiop", card=None, total=0)
    )
    assert message["embeds"] == []
    assert "qwertyuiop" in message["content"]
    assert "/oracle" in message["content"]


def test_the_price_label_survives_the_whole_message_path():
    message = oracle_render.card_message(_payload())
    fields = _fields(message["embeds"][0])
    assert fields["USD"] == "$1.60 (as of 2026-08-17)"


# ------------------------------------------------------------------------ hard limits


def test_a_four_thousand_character_oracle_text_truncates_rather_than_400ing_discord():
    embed = oracle_render.card_embed(_card(oracle_text="Draw a card. " * 400))
    assert len(embed["description"]) <= oracle_render.MAX_DESCRIPTION
    assert embed["description"].rstrip().endswith("```")


def test_a_very_long_name_truncates_to_discords_title_limit():
    embed = oracle_render.card_embed(_card(name="X" * 400))
    assert len(embed["title"]) <= oracle_render.MAX_TITLE


def test_every_field_value_fits_discords_limit():
    embed = oracle_render.card_embed(
        _card(type_line="Legendary " * 200, links={"scryfall": "https://x/" + "y" * 2000})
    )
    for field in embed["fields"]:
        assert len(field["value"]) <= oracle_render.MAX_FIELD_VALUE, field["name"]


def test_the_module_imports_nothing_but_the_standard_library():
    """The property `tests/test_render.py` calls "the reason the suite is worth
    running", not being lost for a second surface."""
    import ast
    import pathlib

    source = pathlib.Path(oracle_render.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "typing"}, imported
