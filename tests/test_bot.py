"""The bot's pure logic. No network, no gateway, no Discord connection.

discord.py's gateway behaviour is explicitly not tested — that is verified by hand
once, with `systemctl --user start` and one `/scry`. What *is* tested here is
everything that can be wrong without ever connecting: the colour validator, and
the coupling between `render.encode_custom_id` and the regex the bot matches
incoming component ids against, which is the one seam where a silent mismatch
would make every feedback button in every channel a no-op.
"""

from __future__ import annotations

import asyncio
import re

import pytest

pytest.importorskip("discord", reason="serving layer: pip install -r serve-requirements.txt")
pytest.importorskip("httpx", reason="serving layer: pip install -r serve-requirements.txt")

from serve import bot, render  # noqa: E402


# ------------------------------------------------------------------ the custom_id seam


def test_bot_template_matches_every_id_render_can_emit():
    """The one coupling that would silently kill every button if it drifted.

    `render` writes the ids; `bot.FeedbackButton`'s template is what Discord's
    dispatcher matches an incoming tap against. If these two ever disagree, no
    exception is raised anywhere — the taps simply do nothing.
    """
    pattern = re.compile(bot.CUSTOM_ID_TEMPLATE)
    for query_id in (1, 4210, 999999999):
        for accepted in (True, False):
            custom_id = render.encode_custom_id(
                query_id, "5f4e3d2c-1b0a-4988-9876-543210fedcba", accepted
            )
            match = pattern.fullmatch(custom_id)
            assert match is not None, custom_id
            assert int(match["query_id"]) == query_id
            assert match["illustration_id"] == "5f4e3d2c-1b0a-4988-9876-543210fedcba"
            assert (match["vote"] == "u") is accepted


def test_bot_template_agrees_with_the_decoder():
    pattern = re.compile(bot.CUSTOM_ID_TEMPLATE)
    custom_id = render.encode_custom_id(77, "ill-77", False)
    match = pattern.fullmatch(custom_id)
    decoded = render.decode_custom_id(custom_id)
    assert decoded == (int(match["query_id"]), match["illustration_id"], match["vote"] == "u")


@pytest.mark.parametrize("foreign", ["some:other:button", "sp:v0:1:x:u", "persistent-view:1"])
def test_bot_template_ignores_foreign_component_ids(foreign):
    assert re.compile(bot.CUSTOM_ID_TEMPLATE).fullmatch(foreign) is None


# ------------------------------------------------------------------- colour validation


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("wub", "WUB"),
        ("BUW", "WUB"),          # canonicalised into WUBRG order
        ("WW", "W"),             # deduplicated
        ("  rg  ", "RG"),
        ("WUBRG", "WUBRG"),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_normalize_colors_accepts_and_canonicalises(raw, expected):
    assert bot.normalize_colors(raw) == expected


@pytest.mark.parametrize("bad", ["X", "wubx", "123", "purple"])
def test_normalize_colors_rejects_non_wubrg(bad):
    with pytest.raises(ValueError) as excinfo:
        bot.normalize_colors(bad)
    # The user reads this in a chat client, so it has to explain itself.
    assert "WUBRG" in str(excinfo.value)


# ------------------------------------------------------------------------ the API base


def test_api_base_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv(bot.API_URL_ENV, raising=False)
    assert bot.api_base() == "http://127.0.0.1:8077"


def test_api_base_strips_a_trailing_slash(monkeypatch):
    monkeypatch.setenv(bot.API_URL_ENV, "http://127.0.0.1:9000/")
    assert bot.api_base() == "http://127.0.0.1:9000"


# ---------------------------------------------------------------------------- the view


def _outcome(n: int = 5) -> dict:
    return {
        "query_id": 4210,
        "results": [
            {
                "name": f"Card {i}",
                "illustration_id": f"5f4e3d2c-1b0a-4988-9876-54321000000{i}",
                "stretch": False,
                "links": {},
                "band": 3,
                "fit": 0.8,
                "color_identity": "W",
            }
            for i in range(n)
        ],
    }


def test_build_view_makes_ten_persistent_buttons():
    view = bot.build_view(_outcome())
    assert view is not None
    assert view.timeout is None          # persistent: it must not expire with the process
    assert len(view.children) == 10
    ids = [child.custom_id for child in view.children]
    assert len(set(ids)) == 10
    assert all(id_.startswith("sp:v1:4210:") for id_ in ids)
    assert sum(1 for id_ in ids if id_.endswith(":u")) == 5
    assert sum(1 for id_ in ids if id_.endswith(":d")) == 5


def test_build_view_handles_fewer_than_five_results():
    view = bot.build_view(_outcome(n=2))
    assert view is not None
    assert len(view.children) == 4


def test_build_view_is_none_when_there_is_nothing_to_vote_on():
    assert bot.build_view({"query_id": 4210, "results": []}) is None


def test_build_view_buttons_carry_the_ids_render_chose():
    outcome = _outcome(n=3)
    expected = {spec["custom_id"] for spec in render.button_specs(outcome)}
    view = bot.build_view(outcome)
    assert {child.custom_id for child in view.children} == expected


# --------------------------------------------------------------- response body handling


class _Response:
    def __init__(self, payload, *, raises=False):
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("not json")
        return self._payload


def test_json_or_none_returns_dicts():
    assert bot._json_or_none(_Response({"ok": True})) == {"ok": True}


def test_json_or_none_swallows_non_json():
    assert bot._json_or_none(_Response(None, raises=True)) is None


def test_json_or_none_rejects_a_non_dict_body():
    """A bare list would sail through `.get()`-free code and AttributeError later."""
    assert bot._json_or_none(_Response([1, 2, 3])) is None


# ------------------------------------------------------------------------ the timeouts


def test_the_refresh_timeout_fits_inside_discords_deferred_token():
    """15 minutes is the hard bound; 780s leaves two minutes to render and edit."""
    assert bot.SEARCH_TIMEOUT_REFRESHING < 15 * 60
    assert bot.SEARCH_TIMEOUT_REFRESHING > bot.SEARCH_TIMEOUT
    # Worst-case search is 106.7s; the normal timeout must clear it with room.
    assert bot.SEARCH_TIMEOUT > 107 * 2


# ----------------------------------------------------------------------- health probing


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Just enough httpx.AsyncClient for fetch_health."""

    def __init__(self, outcome):
        self._outcome = outcome

    async def get(self, url, timeout=None):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _run(coro):
    """Drive one coroutine to completion. No pytest-asyncio in dev-requirements."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_fetch_health_returns_the_body():
    health = {"status": "ok"}
    got, down = _run(bot.fetch_health(_FakeClient(_FakeResponse(200, health))))
    assert got == health
    assert down is False


def test_fetch_health_reports_the_api_being_absent_distinctly():
    """A ConnectError means nothing is on 8077; anything else is just a worse placeholder."""
    import httpx

    got, down = _run(bot.fetch_health(_FakeClient(httpx.ConnectError("refused"))))
    assert got is None
    assert down is True


def test_fetch_health_tolerates_a_slow_or_broken_health():
    import httpx

    got, down = _run(bot.fetch_health(_FakeClient(httpx.ReadTimeout("slow"))))
    assert (got, down) == (None, False)
    got, down = _run(bot.fetch_health(_FakeClient(_FakeResponse(500))))
    assert (got, down) == (None, False)
