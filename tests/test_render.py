"""Pure functions over canned result dicts. Nothing imported beyond the stdlib.

`serve/render.py` deliberately renders to plain dicts matching Discord's embed
JSON rather than to `discord.Embed` objects, which is what lets this file run in a
checkout with nothing installed but pytest. That property is the reason the suite
is worth running and it must not be lost.
"""

from __future__ import annotations

import pytest

from serve import render

# --------------------------------------------------------------------------- fixtures


def _result(**overrides) -> dict:
    """A passing, verified result with every link present."""
    base = {
        "oracle_id": "0f1e2d3c-4b5a-6789-abcd-ef0123456789",
        "name": "Avacyn, Angel of Hope",
        "mana_cost": "{5}{W}{W}",
        "type_line": "Legendary Creature — Angel",
        "color_identity": "W",
        "band": 3,
        "fit": 0.82,
        "rationale": "A lone armoured figure haloed against a darkening sky.",
        "verified": True,
        "illustration_id": "5f4e3d2c-1b0a-4988-9876-543210fedcba",
        "set_code": "avr",
        "artist": "Jason Chan",
        "prop_ids": [11, 12],
        "links": {
            "edhrec": "https://edhrec.com/commanders/avacyn-angel-of-hope",
            "edhrec_theme": "https://edhrec.com/commanders/avacyn-angel-of-hope/angels",
            "scryfall": "https://scryfall.com/card/avr/6",
            "tcgplayer": "https://www.tcgplayer.com/product/57330",
            "art_crop": "https://cards.scryfall.io/art_crop/front/a/b/abc.jpg",
        },
        "stretch": False,
        "vision_rejected": False,
        "verify_note": None,
        "score": 0.91,
        "art_count": 1,
    }
    base.update(overrides)
    return base


def _outcome(results=None, **overrides) -> dict:
    base = {
        "query_id": 4210,
        "plan": {
            "literal_weight": 0.3,
            "interpretive_weight": 0.7,
            "band": 3,
            "colors": None,
            "notes": [],
            "vision_verified": True,
            "counts": {"commanders": 412, "candidates": 0},
        },
        "relaxed": None,
        "results": [_result()] if results is None else results,
        "pool": [],
        "service": {"degraded": False, "index_rebuilt": False, "refresh_running": False},
    }
    base.update(overrides)
    return base


def _text(embed: dict) -> str:
    """Everything rendered in one embed, flattened, for substring assertions."""
    parts = [str(embed.get("title", "")), str(embed.get("description", ""))]
    for field in embed.get("fields", []):
        parts.append(f"{field['name']} {field['value']}")
    parts.append(str((embed.get("footer") or {}).get("text", "")))
    return "\n".join(parts)


# ------------------------------------------------------------- the band label decision


def test_band_field_says_popularity_not_power():
    """This string is a decision, not a wording preference, so it gets a test.

    The composite behind the number is 0.4 x deck count + 0.25 x price + 0.2 x cmc
    + 0.15 x a saturating cEDH flag. config.toml's own comments call deck count
    "popularity, not power". The interface says what the number measures.
    """
    embed = render.result_embed(_result(band=3), 1)
    names = [f["name"] for f in embed["fields"]]
    assert "Popularity band" in names
    assert next(f for f in embed["fields"] if f["name"] == "Popularity band")["value"] == "3/5"
    assert "power" not in _text(embed).lower()


def test_band_missing_is_unknown_not_zero():
    embed = render.result_embed(_result(band=None), 1)
    value = next(f for f in embed["fields"] if f["name"] == "Popularity band")["value"]
    assert value == "unknown"
    assert "power" not in _text(embed).lower()


# ------------------------------------------------------------------------- the embeds


def test_passing_verified_result():
    embed = render.result_embed(_result(), 1)
    assert embed["title"] == "1. Avacyn, Angel of Hope {5}{W}{W}"
    assert "STRETCH" not in embed["title"]
    assert embed["color"] == render.COLOR_VERIFIED
    assert embed["description"].startswith("A lone armoured figure")
    assert embed["thumbnail"]["url"].endswith("abc.jpg")
    assert "AVR" in embed["footer"]["text"]
    assert "Jason Chan" in embed["footer"]["text"]

    links = next(f for f in embed["fields"] if f["name"] == "Links")["value"]
    for label in ("EDHREC", "theme", "Scryfall", "TCGplayer"):
        assert f"[{label}](" in links


def test_passing_but_unverified_is_a_different_colour():
    embed = render.result_embed(_result(verified=False), 1)
    assert embed["color"] == render.COLOR_PASSING
    assert "vision verified" not in embed["footer"]["text"]


def test_stretch_is_labelled_never_silently_mixed_in():
    embed = render.result_embed(_result(stretch=True, fit=0.31, verified=False), 4)
    assert embed["title"].endswith("· STRETCH")
    assert embed["color"] == render.COLOR_STRETCH
    fit = next(f for f in embed["fields"] if f["name"] == "Fit")["value"]
    assert fit.startswith("0.31")
    assert "below the 0.5 bar" in fit


def test_result_with_no_optional_links_omits_the_field_entirely():
    embed = render.result_embed(_result(links={}), 1)
    assert [f["name"] for f in embed["fields"]] == ["Colours", "Popularity band", "Fit"]
    assert "thumbnail" not in embed


def test_partial_links_render_only_what_is_present():
    embed = render.result_embed(
        _result(links={"scryfall": "https://scryfall.com/card/avr/6"}), 1
    )
    links = next(f for f in embed["fields"] if f["name"] == "Links")["value"]
    assert links == "[Scryfall](https://scryfall.com/card/avr/6)"


def test_multiple_arts_are_disclosed_in_the_footer():
    embed = render.result_embed(_result(art_count=4), 2)
    assert "1 of 4 arts" in embed["footer"]["text"]


def test_colourless_identity_renders_as_C():
    embed = render.result_embed(_result(color_identity=""), 1)
    assert next(f for f in embed["fields"] if f["name"] == "Colours")["value"] == "C"


def test_verify_note_is_surfaced_not_swallowed():
    embed = render.result_embed(
        _result(verify_note="no local art crop to verify against"), 1
    )
    assert "no local art crop to verify against" in embed["description"]


def test_pathological_rationale_is_truncated_rather_than_400ing_the_message():
    embed = render.result_embed(_result(rationale="x" * 4000), 1)
    assert len(embed["description"]) <= render.MAX_DESCRIPTION
    assert embed["description"].endswith("…")


def test_missing_fit_renders_a_dash_not_a_crash():
    embed = render.result_embed(_result(fit=None, stretch=True), 1)
    assert next(f for f in embed["fields"] if f["name"] == "Fit")["value"].startswith("—")


# --------------------------------------------------------------------- the content line


def test_content_line_mirrors_the_cli_header():
    line = render.content_line("commanders that look lonely", _outcome())
    assert '🔮 "commanders that look lonely"' in line
    assert "30% literal / 70% interpretive" in line
    assert "band 3" in line


def test_content_line_counts_the_stretches():
    results = [_result(), _result(stretch=True), _result(stretch=True)]
    line = render.content_line("x", _outcome(results=results))
    assert "1 of 3 results clear the 0.5 fit bar; the rest are stretches." in line


def test_content_line_omits_the_stretch_count_when_all_pass():
    line = render.content_line("x", _outcome(results=[_result(), _result()]))
    assert "stretch" not in line.lower()


def test_empty_results_say_so_with_the_counts():
    outcome = _outcome(results=[])
    outcome["plan"]["counts"] = {"commanders": 412, "candidates": 0}
    line = render.content_line("x", outcome)
    assert "no matches. 412 commanders retrieved, 0 survived the filters." in line


def test_relaxed_band_is_reported():
    line = render.content_line("x", _outcome(relaxed="power band widened from 3 to 2-4"))
    assert "power band widened from 3 to 2-4" in line


# ------------------------------------------------------------------------ the whole message


def test_stretches_are_ordered_last():
    stretch = _result(name="Stretchy", stretch=True)
    passing = _result(name="Passing")
    ordered = render.order_results([stretch, passing])
    assert [r["name"] for r in ordered] == ["Passing", "Stretchy"]


def test_render_message_caps_at_five_embeds():
    message = render.render_message("x", _outcome(results=[_result() for _ in range(8)]))
    assert len(message["embeds"]) == render.MAX_EMBEDS


def test_render_message_of_an_empty_result_set_has_no_embeds():
    message = render.render_message("x", _outcome(results=[]))
    assert message["embeds"] == []
    assert "no matches" in message["content"]


def test_degraded_banner_carries_the_notes_verbatim():
    outcome = _outcome()
    outcome["service"]["degraded"] = True
    outcome["plan"]["notes"] = [
        "dense retrieval unavailable (connection refused); ranked on BM25 alone"
    ]
    message = render.render_message("x", outcome)
    assert "⚠️ dense retrieval unavailable (connection refused); ranked on BM25 alone" \
        in message["content"]


def test_routine_notes_are_not_dressed_up_as_warnings():
    """Nearly every search emits a slot-filter note. If those carried ⚠️, the one
    banner that matters — Ollama down, nothing judged — would be invisible."""
    outcome = _outcome()
    outcome["service"]["degraded"] = True
    outcome["plan"]["notes"] = [
        "slot filter figure_count equals 'a single figure' matched nothing; dropped",
        "slot filters kept 356 artworks (time_of_day contains 'dusk' -> 356)",
    ]
    banner = render.degraded_banner(outcome)
    assert "⚠️" not in banner
    assert banner.count("note: ") == 2
    # Still carried verbatim, which is what the spec requires.
    assert "matched nothing; dropped" in banner


def test_a_real_warning_still_stands_out_among_routine_notes():
    outcome = _outcome()
    outcome["service"]["degraded"] = True
    outcome["plan"]["notes"] = [
        "slot filters kept 356 artworks (time_of_day contains 'dusk' -> 356)",
        "vision verification unavailable; results are judge-ordered only",
    ]
    banner = render.degraded_banner(outcome)
    lines = banner.splitlines()
    assert lines[0].startswith("note: ")
    assert lines[1].startswith("⚠️ ")


def test_degraded_banner_absent_when_not_degraded():
    assert render.degraded_banner(_outcome()) == ""


def test_unverified_vision_produces_the_cli_s_own_sentence():
    outcome = _outcome()
    outcome["service"]["degraded"] = True
    outcome["plan"]["vision_verified"] = False
    outcome["plan"]["notes"] = []
    banner = render.degraded_banner(outcome)
    assert "vision verification unavailable" in banner
    assert "judge-ordered and unverified" in banner


def test_content_never_exceeds_discords_limit():
    outcome = _outcome()
    outcome["service"]["degraded"] = True
    outcome["plan"]["notes"] = ["a note that is far too long " * 200]
    message = render.render_message("y" * 300, outcome)
    assert len(message["content"]) <= render.MAX_CONTENT


# ------------------------------------------------------------------------ the custom_id


def test_custom_id_round_trips():
    custom_id = render.encode_custom_id(4210, "5f4e3d2c-1b0a-4988-9876-543210fedcba", True)
    assert custom_id == "sp:v1:4210:5f4e3d2c-1b0a-4988-9876-543210fedcba:u"
    assert render.decode_custom_id(custom_id) == (
        4210,
        "5f4e3d2c-1b0a-4988-9876-543210fedcba",
        True,
    )


def test_custom_id_round_trips_a_downvote():
    custom_id = render.encode_custom_id(1, "abc", False)
    assert render.decode_custom_id(custom_id) == (1, "abc", False)


def test_custom_id_stays_inside_discords_hundred_characters():
    custom_id = render.encode_custom_id(
        999999999, "5f4e3d2c-1b0a-4988-9876-543210fedcba", False
    )
    assert len(custom_id) <= render.MAX_CUSTOM_ID


def test_custom_id_refuses_to_emit_an_oversized_id():
    with pytest.raises(ValueError):
        render.encode_custom_id(1, "x" * 200, True)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "sp:v1:4210:abc",                 # too few parts
        "sp:v2:4210:abc:u",               # wrong version
        "other:v1:4210:abc:u",            # not ours
        "sp:v1:notanint:abc:u",           # unparseable query_id
        "sp:v1:4210:abc:x",               # not a vote
        "sp:v1:4210::u",                  # no illustration id
    ],
)
def test_decode_ignores_anything_that_is_not_ours(bad):
    assert render.decode_custom_id(bad) is None


# ------------------------------------------------------------------------- the buttons


def test_button_specs_are_ten_in_two_rows():
    outcome = _outcome(results=[_result(illustration_id=f"ill-{i}") for i in range(5)])
    specs = render.button_specs(outcome)
    assert len(specs) == 10
    assert sorted({s["row"] for s in specs}) == [0, 1]
    assert [s["label"] for s in specs if s["row"] == 0] == ["1", "2", "3", "4", "5"]
    assert all(s["emoji"] == "👍" for s in specs if s["row"] == 0)
    assert all(s["emoji"] == "👎" for s in specs if s["row"] == 1)


def test_button_specs_follow_the_displayed_order_not_the_raw_order():
    """Button 1 must vote on embed 1, which is the first *passing* result."""
    results = [
        _result(name="Stretchy", illustration_id="ill-stretch", stretch=True),
        _result(name="Passing", illustration_id="ill-pass"),
    ]
    specs = render.button_specs(_outcome(results=results))
    first_up = next(s for s in specs if s["row"] == 0 and s["label"] == "1")
    assert first_up["name"] == "Passing"
    assert "ill-pass" in first_up["custom_id"]


def test_button_specs_are_empty_without_a_query_id():
    assert render.button_specs(_outcome(query_id=None)) == []


def test_button_specs_are_empty_for_an_empty_result_set():
    assert render.button_specs(_outcome(results=[])) == []


# ---------------------------------------------------------------------- the placeholder


def test_placeholder_normal():
    health = {"search": {"queued": 0, "in_flight": 0}, "refresh": {"running": False}}
    assert render.placeholder(health) == "🔮 scrying… ~80s"


def test_placeholder_when_queued_behind_someone():
    health = {"search": {"queued": 0, "in_flight": 1}, "refresh": {"running": False}}
    line = render.placeholder(health)
    assert "queued behind 1 search," in line
    assert "min" in line


def test_placeholder_pluralises_the_queue():
    health = {"search": {"queued": 2, "in_flight": 1}, "refresh": {"running": False}}
    assert "queued behind 3 searches," in render.placeholder(health)


def test_placeholder_explains_a_running_refresh():
    """The entire user-facing consequence of GPU contention. Information, not a refusal."""
    health = {"search": {"queued": 0, "in_flight": 0}, "refresh": {"running": True}}
    line = render.placeholder(health)
    assert "weekly corpus refresh is running" in line
    assert "several minutes rather than ~80s" in line


def test_placeholder_warns_when_ollama_is_unreachable():
    health = {
        "search": {"queued": 0, "in_flight": 0},
        "refresh": {"running": False},
        "ollama": {"reachable": False},
    }
    assert "Ollama is unreachable" in render.placeholder(health)


def test_placeholder_without_health_still_says_something():
    assert render.placeholder(None) == "🔮 scrying… ~80s"


def test_placeholder_treats_unknown_refresh_state_as_not_running():
    """`None` is 'systemctl could not answer', which is not 'a refresh is running'."""
    health = {"search": {"queued": 0, "in_flight": 0}, "refresh": {"running": None}}
    assert "refresh is running" not in render.placeholder(health)


# ---------------------------------------------------------------------------- error prose


def test_every_error_message_names_something_actionable():
    assert "systemctl --user status scrying-api" in render.api_down_message()
    assert "Try again shortly" in render.busy_message({"queued": 4})
    assert "journalctl --user -u scrying-api" in render.search_failed_message("boom")
    assert "5 minutes" in render.timeout_message(300.0)
    assert "13 minutes" in render.timeout_message(780.0)


def test_search_failure_detail_is_truncated_not_unbounded():
    message = render.search_failed_message("z" * 5000)
    assert len(message) < 700
