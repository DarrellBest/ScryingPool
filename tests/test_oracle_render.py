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


# ===========================================================================
# /oracle — ranked mechanical results
# ===========================================================================


def _result(**overrides) -> dict:
    base = {
        "oracle_id": "o-verdant",
        "name": "Verdant Genesis",
        "mana_cost": "{2}{G}",
        "type_line": "Enchantment",
        "oracle_text": "Whenever a creature enters, draw a card.\nAt the beginning of your "
                        "end step, you lose 1 life.",
        "cmc": 3.0,
        "color_identity": "G",
        "set_code": "abc",
        "rarity": "rare",
        "released_at": "2019-07-12",
        "scryfall_uri": "https://scryfall.com/card/abc/1",
        "legalities": {"commander": "legal", "modern": "legal", "legacy": "banned"},
        "fit": 0.87,
        "rationale": "It draws a card whenever a creature enters, unconditionally.",
        "chunk_ids": [7],
        "matched_face_index": 0,
        "matched_ordinal": 0,
        "score": 0.5,
        "stretch": False,
        "judged": True,
    }
    base.update(overrides)
    return base


def _oracle_fields(embed: dict) -> dict:
    return {field["name"]: field["value"] for field in embed["fields"]}


def _outcome(**overrides) -> dict:
    base = {
        "query_id": 99,
        "plan": {
            "echo": 'filters: type = enchantment · colors ⊆ {G} (identity fits inside) · '
                    'mv ≤ 5 · semantic: "let me draw"',
            "notes": [],
            "scryfall_url": "https://scryfall.com/search?q=t%3Aenchantment",
        },
        "message": "214 cards passed the filters · 40 judged · 4 of 5 clear the 0.5 fit bar",
        "results": [_result()],
        "pool": [_result()],
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ oracle text + marker


def test_the_full_oracle_text_is_shown_verbatim_above_the_rationale():
    """A loot-for-draw error has to be checkable in the two seconds it takes
    to read the card — the authoritative text leads, the model's rationale
    trails below it as a field."""
    embed = oracle_render.oracle_result_embed(_result(), 1)
    assert "draw a card" in embed["description"]
    assert "lose 1 life" in embed["description"]
    rationale_field = _oracle_fields(embed)["Rationale"]
    assert rationale_field == _result()["rationale"]
    # position check: the description (oracle text) is a top-level key that
    # Discord renders above every field, including Rationale.
    assert "description" in embed and embed["fields"][-1]["name"] == "Rationale"


def test_the_matched_line_is_marked_by_ordinal_never_by_string_search():
    result = _result(
        oracle_text="Flying\nWhenever this creature attacks, draw a card.\nFlying",
        matched_face_index=0, matched_ordinal=1,
    )
    embed = oracle_render.oracle_result_embed(result, 1)
    lines = embed["description"].split("\n")
    marked = [l for l in lines if l.startswith("▸ ")]
    assert marked == ["▸ Whenever this creature attacks, draw a card."]
    # Both "Flying" lines are identical text — only the one at ordinal 1 (by
    # position, not by matching the string) is marked, and there is only one.
    assert sum(1 for l in lines if "Flying" in l and l.startswith("▸")) == 0


def test_a_whole_chunk_match_ordinal_minus_one_marks_nothing():
    result = _result(matched_face_index=0, matched_ordinal=-1)
    embed = oracle_render.oracle_result_embed(result, 1)
    assert "▸" not in embed["description"]


def test_no_oracle_text_says_so_rather_than_an_empty_box():
    embed = oracle_render.oracle_result_embed(_result(oracle_text=""), 1)
    assert embed["description"] == "_(no rules text)_"


def test_a_long_oracle_text_truncates_at_700_characters_with_a_pointer():
    long_text = "Draw a card.\n" * 100
    embed = oracle_render.oracle_result_embed(_result(oracle_text=long_text), 1)
    assert len(embed["description"]) < len(long_text)
    assert "Scryfall" in embed["description"]


# ------------------------------------------------------------------------------ title


def test_title_carries_name_and_mana_cost_and_stretch_label():
    passing = oracle_render.oracle_result_embed(_result(), 1)
    assert passing["title"] == "Verdant Genesis {2}{G}"
    stretch = oracle_render.oracle_result_embed(_result(stretch=True, fit=0.2), 1)
    assert stretch["title"].endswith("· STRETCH")


def test_color_is_green_when_passing_and_grey_when_a_stretch():
    passing = oracle_render.oracle_result_embed(_result(), 1)
    stretch = oracle_render.oracle_result_embed(_result(stretch=True), 1)
    assert passing["color"] == oracle_render.COLOR_ORACLE_PASS
    assert stretch["color"] == oracle_render.COLOR_ORACLE_STRETCH


def test_only_two_colours_exist_not_three_there_is_no_verification_stage():
    """Unlike /scry, there is no vision-verified state here at all."""
    assert not hasattr(oracle_render, "COLOR_ORACLE_VERIFIED")


# ------------------------------------------------------------------------------ fields


def test_fit_is_always_shown_to_two_decimals():
    embed = oracle_render.oracle_result_embed(_result(fit=0.873), 1)
    assert _oracle_fields(embed)["Fit"] == "0.87"


def test_fit_of_none_shows_a_dash_never_a_fabricated_number():
    embed = oracle_render.oracle_result_embed(_result(fit=None), 1)
    assert _oracle_fields(embed)["Fit"] == "—"


def test_mana_value_and_colours_field_reports_c_for_colourless():
    embed = oracle_render.oracle_result_embed(_result(color_identity=""), 1)
    assert "/ C" in _oracle_fields(embed)["Mana value / Colours"]


def test_legal_lists_formats_and_names_a_ban_explicitly():
    line = oracle_render.oracle_legality_line(
        {"commander": "legal", "modern": "legal", "legacy": "banned"}
    )
    assert "commander" in line and "modern" in line
    assert "Banned in legacy" in line


def test_legal_caps_at_six_formats():
    legalities = {f: "legal" for f in oracle_render.ORACLE_LEGALITY_FORMATS}
    line = oracle_render.oracle_legality_line(legalities)
    assert line.count(",") <= 5  # at most six names, five separators


def test_unknown_legalities_say_so():
    assert oracle_render.oracle_legality_line(None) == "unknown"
    assert oracle_render.oracle_legality_line({}) == "unknown"


def test_footer_names_the_set_and_first_printed_year():
    embed = oracle_render.oracle_result_embed(_result(set_code="abc", released_at="2019-07-12"), 1)
    assert embed["footer"]["text"] == "first printed ABC 2019"


# ------------------------------------------------------------------- the image ban


def test_oracle_result_embed_never_carries_an_image_or_thumbnail_key():
    """The decision, not an omission: a ranked mechanical result must never
    carry an image, ever — enforced here and tested directly rather than
    trusted to stay true by absence of code."""
    embed = oracle_render.oracle_result_embed(_result(), 1)
    assert "image" not in embed
    assert "thumbnail" not in embed


# --------------------------------------------------------------------- content line


def test_content_line_echoes_filters_and_the_honest_counts():
    line = oracle_render.oracle_content_line(
        "enchantments in green that let me draw and cost 5 or less", _outcome()
    )
    assert 'filters: type = enchantment' in line
    assert '214 cards passed the filters' in line
    assert 'refine on Scryfall' in line


def test_content_line_shows_a_warning_marker_for_a_failed_stage():
    outcome = _outcome(plan={
        "echo": "filters: none · semantic: none", "notes": ["Ollama is unreachable"],
        "scryfall_url": None,
    })
    line = oracle_render.oracle_content_line("q", outcome)
    assert "⚠️" in line


def test_content_line_uses_plain_note_for_non_warning_narration():
    outcome = _outcome(plan={
        "echo": "filters: none · semantic: none",
        "notes": ['"cheap" has no defined mana value, so no cost filter was applied.'],
        "scryfall_url": None,
    })
    line = oracle_render.oracle_content_line("q", outcome)
    assert "⚠️" not in line
    assert "note:" in line


def test_content_line_names_an_ignored_set_level_constraint():
    """Defect 3: a set-level constraint the router recognised but the pipeline
    cannot enforce must show up as an honest `ignored:` line, not vanish."""
    outcome = _outcome(plan={
        "echo": 'filters: none · semantic: none',
        "notes": [],
        "ignored": ["no overlapping color identity"],
        "scryfall_url": None,
    })
    line = oracle_render.oracle_content_line("q", outcome)
    assert "ignored: no overlapping color identity" in line


def test_content_line_has_no_ignored_line_when_nothing_was_dropped():
    outcome = _outcome(plan={
        "echo": 'filters: none · semantic: none', "notes": [], "scryfall_url": None,
    })
    line = oracle_render.oracle_content_line("q", outcome)
    assert "ignored:" not in line


# ------------------------------------------------------------------------- the message


def test_oracle_message_renders_one_embed_per_result():
    message = oracle_render.oracle_message("q", _outcome())
    assert len(message["embeds"]) == 1
    assert message["content"].startswith('🔮 "q"')


def test_oracle_message_caps_at_five_embeds():
    outcome = _outcome(results=[_result(oracle_id=f"o{i}") for i in range(8)])
    message = oracle_render.oracle_message("q", outcome)
    assert len(message["embeds"]) == 5


def test_a_guard_response_is_content_only_no_embeds():
    outcome = {"guard": "rules_question", "message": "That reads as a rules question."}
    message = oracle_render.oracle_message("can I respond", outcome)
    assert message["embeds"] == []
    assert message["content"] == "That reads as a rules question."


def test_an_empty_result_set_still_carries_the_honest_counts():
    outcome = _outcome(results=[], message="no cards match type = enchantment · colors ⊆ {G}. "
                                            "That combination has 0 cards in ~32,700 paper cards.")
    message = oracle_render.oracle_message("q", outcome)
    assert message["embeds"] == []
    assert "0 cards in ~32,700" in message["content"]


# --------------------------------------------------------------------- feedback buttons


def test_oracle_custom_id_round_trips():
    encoded = oracle_render.encode_oracle_custom_id(42, "o-verdant", True)
    assert oracle_render.decode_oracle_custom_id(encoded) == (42, "o-verdant", True)


def test_oracle_custom_id_prefix_never_decodes_as_the_scry_prefix():
    """A real collision risk with two button families in one channel: this
    module's buttons and /scry's must never claim each other's."""
    from serve import render

    oracle_id = oracle_render.encode_oracle_custom_id(1, "o-verdant", True)
    assert render.decode_custom_id(oracle_id) is None

    scry_id = render.encode_custom_id(1, "ill-avacyn", True)
    assert oracle_render.decode_oracle_custom_id(scry_id) is None


def test_oracle_button_specs_carries_oracle_id_not_illustration_id():
    specs = oracle_render.oracle_button_specs(_outcome())
    assert len(specs) == 2  # one 👍 one 👎, one result
    query_id, oracle_id, accepted = oracle_render.decode_oracle_custom_id(specs[0]["custom_id"])
    assert oracle_id == "o-verdant"


def test_oracle_button_specs_is_empty_without_a_query_id():
    assert oracle_render.oracle_button_specs(_outcome(query_id=None)) == []
