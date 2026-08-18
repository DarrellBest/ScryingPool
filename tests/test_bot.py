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


# ------------------------------------------------------------- the global-sync guard
#
# `tree.sync()` with no guild does not add a command — it REPLACES the
# application's entire global command set. A misconfigured process therefore has
# to be incapable of reaching that call: a missing guild id must mean "register
# nothing", never "register globally". This is the one bit of gateway-adjacent
# behaviour worth testing offline, because the way it fails is destroying an
# application's commands with no error anywhere.


def test_sync_plan_refuses_when_nothing_is_configured():
    assert bot.sync_plan(None, False) == "none"


def test_sync_plan_is_guild_scoped_when_a_guild_id_is_set():
    assert bot.sync_plan(659588470448062464, False) == "guild"


def test_sync_plan_goes_global_only_on_the_explicit_opt_in():
    assert bot.sync_plan(None, True) == "global"


def test_a_guild_id_wins_over_the_global_opt_in():
    """Both set is not ambiguous: the scoped sync is the one that cannot destroy."""
    assert bot.sync_plan(659588470448062464, True) == "guild"


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_the_opt_in_accepts_the_obvious_affirmatives(raw):
    assert bot.env_allows_global_sync({bot.ALLOW_GLOBAL_SYNC_ENV: raw}) is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "maybe", "  "])
def test_anything_else_is_not_an_opt_in(raw):
    assert bot.env_allows_global_sync({bot.ALLOW_GLOBAL_SYNC_ENV: raw}) is False


def test_an_unset_opt_in_is_not_an_opt_in():
    assert bot.env_allows_global_sync({}) is False


class _RecordingTree:
    """Records what setup_hook asked Discord to do, without asking Discord."""

    def __init__(self):
        self.syncs = []
        self.copied = []

    def copy_global_to(self, *, guild):
        self.copied.append(guild.id)

    async def sync(self, *, guild=None):
        self.syncs.append(None if guild is None else guild.id)
        return []


def _hooked(guild_id, allow_global_sync):
    client = bot.ScryingBot(guild_id=guild_id, allow_global_sync=allow_global_sync)
    tree = _RecordingTree()
    client.tree = tree
    _run(client.setup_hook())
    return tree


def test_setup_hook_syncs_nothing_when_nothing_is_configured(caplog):
    with caplog.at_level("ERROR", logger="scrying.bot"):
        tree = _hooked(None, False)
    assert tree.syncs == []          # the whole point: no call reaches Discord
    assert tree.copied == []
    message = caplog.text
    assert bot.GUILD_ENV in message and bot.ALLOW_GLOBAL_SYNC_ENV in message
    assert "REPLACES" in message     # it says why, not just that it refused


def test_setup_hook_syncs_to_the_guild_when_one_is_set():
    tree = _hooked(659588470448062464, False)
    assert tree.syncs == [659588470448062464]
    assert tree.copied == [659588470448062464]


def test_setup_hook_syncs_globally_only_with_the_opt_in():
    tree = _hooked(None, True)
    assert tree.syncs == [None]      # None means global, and only here


# --------------------------------------------------------------------------- /search
#
# The one command that does not defer. Everything about it — its own short
# timeout, its pure reply builder, its registration — exists to keep the whole
# round trip inside Discord's 3-second acknowledgement window.


def _card_body(**overrides) -> dict:
    body = {
        "resolved": True,
        "layer": "L1",
        "input": "sol ring",
        "distance": None,
        "total": 1,
        "card": {
            "name": "Sol Ring",
            "type_line": "Artifact",
            "oracle_text": "{T}: Add {C}{C}.",
            "mana_cost": "{1}",
            "cmc": 1.0,
            "set_code": "msc",
            "rarity": "uncommon",
            "image_normal": "https://cards.scryfall.io/normal/front/sol.jpg",
            "price_usd": 1.6,
            "scryfall_uri": "https://scryfall.com/card/msc/1",
            "legalities": {"commander": "legal"},
            "links": {"scryfall": "https://scryfall.com/card/msc/1"},
        },
        "candidates": [],
        "service": {"refreshed_at": "2026-08-17T03:43:02+00:00"},
    }
    body.update(overrides)
    return body


def test_the_card_timeout_fits_inside_discords_three_second_window():
    """`/search` never defers, so the ENTIRE round trip has to land inside 3s.
    The timeout is a tripwire for a wedged API, not a budget to spend."""
    assert bot.CARD_TIMEOUT < 3.0


def test_card_reply_renders_a_card():
    message = bot.card_reply("sol ring", 200, _card_body())
    assert message["embeds"][0]["title"] == "Sol Ring"
    assert message["content"] == ""


def test_card_reply_renders_candidates_for_an_ambiguous_answer():
    body = _card_body(
        resolved=False, card=None, total=2, layer="L3", input="path",
        candidates=[{"name": "Path to Exile", "mana_cost": "{W}", "type_line": "Instant"},
                    {"name": "Path of Ancestry", "mana_cost": "", "type_line": "Land"}],
    )
    message = bot.card_reply("path", 200, body)
    assert "2 cards match" in message["content"]
    assert message["embeds"][0]["title"] == 'candidates for "path"'


def test_card_reply_says_the_corpus_is_missing_on_a_503():
    message = bot.card_reply("sol ring", 503, {"detail": "empty"})
    assert "oracle-ingest" in message["content"]
    assert message["embeds"] == []


def test_card_reply_says_the_api_is_down_when_nothing_answered():
    message = bot.card_reply("sol ring", 0, None)
    assert "scrying-api" in message["content"]


def test_card_reply_never_returns_a_dead_reply():
    """Every branch produces a specific sentence: a spinner that never resolves is
    the worst outcome available, and /search cannot even defer to buy time."""
    for status, body in ((200, _card_body()), (0, None), (-1, None), (503, {}),
                         (500, {"detail": "boom"}), (422, {"detail": "blank"}),
                         (200, None)):
        message = bot.card_reply("x", status, body)
        assert message["content"] or message["embeds"], (status, body)


# ------------------------------------------------------------------ command registration


class _FakeResponseChannel:
    def __init__(self):
        self.sent = []
        self.deferred = False

    def is_done(self):
        return bool(self.sent) or self.deferred

    async def defer(self, **kwargs):
        self.deferred = True

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)


class _FakeInteraction:
    def __init__(self):
        self.response = _FakeResponseChannel()


class _ClientStub:
    """The minimum `app_commands.CommandTree` needs to be constructed.

    Not a real client: nothing here connects, logs in or touches the gateway.
    """

    class _Connection:
        """`CommandTree` writes itself onto `client._connection`, so each stub
        needs its own rather than sharing one at class scope."""

        _command_tree = None

        def _remove_application_command(self, *args, **kwargs):
            pass

    def __init__(self):
        self.application_id = 1
        self.http = None
        self._connection = self._Connection()


def _build_tree():
    from discord import app_commands

    tree = app_commands.CommandTree(_ClientStub())
    bot.register_commands(tree)
    return tree


def _registered(tree, name):
    return next(command for command in tree.get_commands() if command.name == name)


def test_both_commands_are_registered_and_named_for_the_question_they_answer():
    """Discord shows each description in the picker as the user types, so the
    disambiguation lands at the moment of choosing. The distinguishing word is
    capitalised because it is the word that decides which one you wanted."""
    tree = _build_tree()
    names = {command.name for command in tree.get_commands()}
    assert {"scry", "search"} <= names
    assert "NAME" in _registered(tree, "search").description
    assert "ARTWORK" in _registered(tree, "scry").description


def test_search_answers_without_deferring(monkeypatch):
    """The property that makes the fast command LOOK fast: no "thinking…" state
    at all. Deferring here would be free to write and would cost the one visible
    difference between this command and the 80-second one."""

    async def fake_fetch(name):
        assert name == "Sol Ring"
        return 200, _card_body()

    monkeypatch.setattr(bot, "fetch_card", fake_fetch)

    command = _registered(_build_tree(), "search")

    interaction = _FakeInteraction()
    asyncio.run(command.callback(interaction, name="  Sol Ring  "))

    assert interaction.response.deferred is False, "/search must not defer"
    assert len(interaction.response.sent) == 1
    sent = interaction.response.sent[0]
    assert sent["embeds"][0].title == "Sol Ring"


def test_search_still_answers_when_the_lookup_explodes(monkeypatch):

    async def boom(name):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(bot, "fetch_card", boom)

    interaction = _FakeInteraction()
    asyncio.run(_registered(_build_tree(), "search").callback(interaction, name="Sol Ring"))

    assert len(interaction.response.sent) == 1
    assert "kaboom" in interaction.response.sent[0]["content"]


