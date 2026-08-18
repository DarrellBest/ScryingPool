"""The Discord bot: `/scry`, over loopback HTTP to `serve.api`.

The bot holds no corpus state at all — no index, no database handle, no model.
Everything it knows it asked the API for a moment ago, which is what makes
`systemctl --user restart scrying-bot` free and is the entire reason the serving
layer is two processes instead of one.

Three things shape every line in here:

1. **Defer first, before anything else.** Discord kills an interaction that is
   not acknowledged within 3 seconds. The search takes 76.8s on average and
   106.7s at worst, and `/health` runs before it. So the very first await in the
   command is `defer(thinking=True)`, which buys a 15-minute token.
2. **The buttons must outlive the process.** They are `DynamicItem`s whose whole
   identity is encoded in `custom_id`, not a `View` held in memory. A restart
   with an ordinary View leaves every 👍 in the channel silently dead, and this
   bot is designed to be restarted constantly.
3. **The user always gets a specific sentence.** A spinner that never resolves is
   the worst outcome available; every branch in `_run_search` exists to avoid it.

Run it with `python -m serve.bot`, with `~/.config/scrying-pool/bot.env` in the
environment. The token is read from the environment and is never logged, never
echoed, and never written anywhere.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from typing import Any

import discord
import httpx
from discord import app_commands

from serve import oracle_render, render

log = logging.getLogger("scrying.bot")

# --------------------------------------------------------------------------- constants

TOKEN_ENV = "SCRYING_DISCORD_TOKEN"
API_URL_ENV = "SCRYING_API_URL"
GUILD_ENV = "SCRYING_DISCORD_GUILD_ID"

# `tree.sync()` with no guild REPLACES the application's entire global command
# set — every global command this application has, whether or not this process
# knows about it. Point the wrong token at this bot for one run and the other
# application's commands are gone. So a global sync is never the fallback for a
# missing guild id; it has to be asked for by name.
ALLOW_GLOBAL_SYNC_ENV = "SCRYING_DISCORD_ALLOW_GLOBAL_SYNC"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_NO_SYNC_MESSAGE = (
    "refusing to register commands: neither %s nor %s is set. A global sync "
    "REPLACES this application's entire global command set, so it is never done "
    "by default — pointing the wrong token at this bot would wipe the other "
    "application's commands. The bot is running and will answer existing "
    "commands; to register /scry, set %s=<server id> (instant, recommended), or "
    "set %s=1 if a global sync is genuinely what you want (up to an hour to "
    "appear in clients)."
)

DEFAULT_API_URL = "http://127.0.0.1:8077"

HEALTH_TIMEOUT = 10.0       # cheap and cached server-side; never worth waiting on
FEEDBACK_TIMEOUT = 30.0     # its own short-lived connection, never behind the search lock

# 300s normally; 780s when /health said a refresh is running. 13 minutes sits
# inside the 15-minute deferred token with margin, and it is the real bound on how
# long a thrashing search is waited on.
SEARCH_TIMEOUT = 300.0
SEARCH_TIMEOUT_REFRESHING = 780.0

# `/search` does not defer, so the ENTIRE round trip has to fit inside Discord's
# 3-second acknowledgement window. The server side is ~20ms of dict lookups over
# loopback; 2.5s is not a budget, it is a tripwire for "the API is wedged", and
# it leaves room to still send a real sentence before the interaction expires.
CARD_TIMEOUT = 2.5

WUBRG = "WUBRG"

# Mirrors render.encode_custom_id's grammar. The illustration_id is a UUID, so
# `[^:]+` cannot swallow the trailing vote character.
CUSTOM_ID_TEMPLATE = r"sp:v1:(?P<query_id>\d+):(?P<illustration_id>[^:]+):(?P<vote>[ud])"


def api_base() -> str:
    return os.environ.get(API_URL_ENV, DEFAULT_API_URL).rstrip("/")


# ---------------------------------------------------------------------------- buttons


class FeedbackButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=CUSTOM_ID_TEMPLATE,
):
    """A 👍/👎 that survives a restart because it carries its own identity.

    `DynamicItem` is discord.py's mechanism for exactly this: the client matches
    an incoming component's `custom_id` against the class template and
    reconstructs the item, so no `View` has to have been alive since the message
    was sent. A tap three weeks and forty restarts later still resolves, and the
    API's `query_id` validation covers the case where it should not.
    """

    def __init__(
        self,
        query_id: int,
        illustration_id: str,
        accepted: bool,
        *,
        label: str = "",
        emoji: str | None = None,
        row: int = 0,
        name: str = "that result",
    ) -> None:
        self.query_id = query_id
        self.illustration_id = illustration_id
        self.accepted = accepted
        self.name = name
        super().__init__(
            discord.ui.Button(
                label=label or None,
                emoji=emoji,
                style=discord.ButtonStyle.secondary,
                custom_id=render.encode_custom_id(query_id, illustration_id, accepted),
                row=row,
            )
        )

    @classmethod
    async def from_custom_id(          # type: ignore[override]
        cls,
        interaction: discord.Interaction,
        item: discord.ui.Button,
        match: re.Match[str],
        /,
    ) -> "FeedbackButton":
        return cls(
            int(match["query_id"]),
            match["illustration_id"],
            match["vote"] == "u",
            label=item.label or "",
            emoji=str(item.emoji) if item.emoji else None,
            row=item.row or 0,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        # Ephemeral so the channel does not fill with acknowledgements.
        await interaction.response.defer(ephemeral=True, thinking=True)
        verdict = "👍" if self.accepted else "👎"
        try:
            async with httpx.AsyncClient(timeout=FEEDBACK_TIMEOUT) as client:
                response = await client.post(
                    f"{api_base()}/feedback",
                    json={
                        "query_id": self.query_id,
                        "illustration_id": self.illustration_id,
                        "accepted": self.accepted,
                        "discord_user_id": str(interaction.user.id),
                    },
                )
        except httpx.HTTPError as exc:
            log.warning(
                "feedback failed for query_id=%s illustration_id=%s: %s",
                self.query_id, self.illustration_id, exc,
            )
            await interaction.followup.send(
                "Couldn't record that — the search service isn't answering.", ephemeral=True
            )
            return

        if response.status_code == 404:
            await interaction.followup.send(
                "Couldn't record that: this result is older than the current database.",
                ephemeral=True,
            )
            return
        if response.status_code >= 400:
            log.warning(
                "feedback %s for query_id=%s: %s",
                response.status_code, self.query_id, response.text[:200],
            )
            await interaction.followup.send("Couldn't record that.", ephemeral=True)
            return

        # The name is not in the custom_id (100 characters do not stretch to it),
        # so recover it from the embed the button is attached to when we can.
        name = self.name
        message = interaction.message
        if message and message.embeds:
            position = None
            try:
                position = int(str(self.item.label))
            except (TypeError, ValueError):
                position = None
            if position and 1 <= position <= len(message.embeds):
                title = message.embeds[position - 1].title or ""
                # "3. Avacyn, Angel of Hope {5}{W}{W}" -> "Avacyn, Angel of Hope"
                stripped = title.split(". ", 1)[-1].split(" · STRETCH")[0]
                name = stripped.split("{")[0].strip() or name

        await interaction.followup.send(f"recorded {verdict} for {name}", ephemeral=True)


def build_view(outcome: dict) -> discord.ui.View | None:
    """Two action rows of persistent buttons, matched to the embed positions."""
    specs = render.button_specs(outcome)
    if not specs:
        return None
    view = discord.ui.View(timeout=None)
    for spec in specs:
        view.add_item(
            FeedbackButton(
                int(outcome["query_id"]),
                spec["custom_id"].split(":")[3],
                spec["accepted"],
                label=spec["label"],
                emoji=spec["emoji"],
                row=spec["row"],
                name=spec["name"],
            )
        )
    return view


# ------------------------------------------------------------------------- the search


def normalize_colors(raw: str | None) -> str | None:
    """Validate a WUBRG subset bot-side so the user gets prose, not a 422 dump."""
    if raw is None:
        return None
    cleaned = raw.strip().upper()
    if not cleaned:
        return None
    unknown = sorted(set(cleaned) - set(WUBRG))
    if unknown:
        raise ValueError(
            f"`colors` must be made of the letters WUBRG — {''.join(unknown)} "
            "isn't one of them. (W=white, U=blue, B=black, R=red, G=green.)"
        )
    return "".join(sorted(set(cleaned), key=WUBRG.index))


async def fetch_health(client: httpx.AsyncClient) -> tuple[dict | None, bool]:
    """`(health, api_is_down)`. Best effort — a bad `/health` must not stop a search.

    The two failures are genuinely different and the caller renders them
    differently: a `ConnectError` means there is nothing listening on 8077 and the
    search is not going to work either, which is worth saying immediately. Any
    other failure just means the placeholder is less informative than it could be.
    """
    try:
        response = await client.get(f"{api_base()}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            return response.json(), False
        log.info("health returned %s", response.status_code)
    except httpx.ConnectError as exc:
        log.warning("API is not listening: %s", exc)
        return None, True
    except (httpx.HTTPError, ValueError) as exc:
        log.info("health check failed: %s", exc)
    return None, False


async def _edit(interaction: discord.Interaction, **kwargs: Any) -> bool:
    """Edit the deferred reply, retry once, then fall back to a channel message.

    A long search plus an expired token, a deleted message or a 5xx from Discord
    all land here. The results are never lost either way — they are already
    durably in `queries`, `retrievals` and `judgments`, recoverable by
    `query_id`, which is why every failure below logs it at WARNING.
    """
    for attempt in (1, 2):
        try:
            await interaction.edit_original_response(**kwargs)
            return True
        except discord.HTTPException as exc:
            log.warning("edit attempt %d failed: %s", attempt, exc)
            if attempt == 1:
                await asyncio.sleep(1.0)

    channel = interaction.channel
    if channel is not None and hasattr(channel, "send"):
        content = kwargs.pop("content", "") or ""
        try:
            await channel.send(f"{interaction.user.mention} {content}".strip(), **kwargs)
            return True
        except discord.HTTPException as exc:
            log.warning("channel fallback failed: %s", exc)
    return False


async def run_scry(
    interaction: discord.Interaction,
    theme: str,
    k: int,
    band: int | None,
    colors: str | None,
) -> None:
    """Defer, placeholder, search, edit in place. One message, edited twice."""
    async with httpx.AsyncClient() as client:
        health, api_down = await fetch_health(client)
        if api_down:
            await _edit(interaction, content=render.api_down_message())
            return

        await _edit(interaction, content=render.placeholder(health))

        # `refresh` may be present-but-null when systemctl could not answer, so this
        # cannot be a two-step .get() with a {} default.
        refreshing = ((health or {}).get("refresh") or {}).get("running") is True
        timeout = SEARCH_TIMEOUT_REFRESHING if refreshing else SEARCH_TIMEOUT

        payload: dict[str, Any] = {"theme": theme, "k": k}
        if band is not None:
            payload["band"] = band
        if colors:
            payload["colors"] = colors

        try:
            response = await client.post(
                f"{api_base()}/search", json=payload, timeout=timeout
            )
        except httpx.ConnectError:
            await _edit(interaction, content=render.api_down_message())
            return
        except httpx.ReadTimeout:
            log.warning("search timed out after %.0fs for theme=%r", timeout, theme)
            await _edit(interaction, content=render.timeout_message(timeout))
            return
        except httpx.HTTPError as exc:
            log.warning("search transport error: %s", exc)
            await _edit(
                interaction, content=render.search_failed_message(f"transport error: {exc}")
            )
            return

    if response.status_code == 503:
        await _edit(interaction, content=render.busy_message(_json_or_none(response)))
        return
    if response.status_code == 422:
        await _edit(
            interaction,
            content="That search wasn't valid: check `k` is 1-5, `band` is 1-5, "
            "and `colors` is a subset of WUBRG.",
        )
        return
    if response.status_code >= 400:
        detail = (_json_or_none(response) or {}).get("detail") or response.text[:300]
        await _edit(interaction, content=render.search_failed_message(str(detail)))
        return

    outcome = _json_or_none(response)
    if outcome is None:
        await _edit(
            interaction,
            content=render.search_failed_message("the API returned something that wasn't JSON"),
        )
        return

    message = render.render_message(theme, outcome)
    embeds = [discord.Embed.from_dict(e) for e in message["embeds"]]
    view = build_view(outcome)

    kwargs: dict[str, Any] = {"content": message["content"], "embeds": embeds}
    if view is not None:
        kwargs["view"] = view

    if not await _edit(interaction, **kwargs):
        log.warning(
            "could not deliver results for query_id=%s — they are durable in the "
            "database and recoverable by that id",
            outcome.get("query_id"),
        )


async def fetch_card(name: str) -> tuple[int, dict | None]:
    """`GET /card?name=…`. Returns (status, body) and never raises.

    Its own client and its own short timeout: this call must not inherit the
    search path's 300-second patience, because a `/search` that takes longer than
    Discord's 3-second window is a dead interaction no matter what comes back.
    """
    try:
        async with httpx.AsyncClient(timeout=CARD_TIMEOUT) as client:
            response = await client.get(f"{api_base()}/card", params={"name": name})
    except httpx.ConnectError:
        return 0, None
    except httpx.HTTPError as exc:
        log.warning("card lookup transport error for %r: %s", name, exc)
        return -1, None
    return response.status_code, _json_or_none(response)


def card_reply(name: str, status: int, body: dict | None) -> dict:
    """Everything `/search` can say, as `{"content", "embeds"}`. Pure.

    Kept out of the command body so it is testable without a gateway: the command
    itself is then three lines and one `send_message`.
    """
    if status == 0:
        return {"content": render.api_down_message(), "embeds": []}
    if status < 0:
        return {
            "content": render.search_failed_message(
                "the lookup did not complete inside its 2.5s budget"
            ),
            "embeds": [],
        }
    if status == 503:
        return {"content": oracle_render.corpus_missing_message(), "embeds": []}
    if status != 200 or body is None:
        detail = (body or {}).get("detail") or f"the API answered {status}"
        return {"content": render.search_failed_message(str(detail)), "embeds": []}
    return oracle_render.card_message(body)


async def run_search(interaction: discord.Interaction, name: str) -> None:
    """One lookup, one reply, inside the 3-second window. **No `defer()`.**

    `/search` makes zero LLM calls — no router, no expansion, no embedding, no
    judge — so it touches neither Ollama nor the GPU, takes no search lock and is
    not counted against the queue cap. A name lookup queued behind an 80-second
    `/scry` would be absurd; the lock exists solely to serialise contention for
    one Ollama instance that this command never uses.

    The work is ~20ms, so the reply goes out with `response.send_message` and
    there is no "thinking…" state at all. That visible difference is itself
    useful: the fast command looks fast.
    """
    status, body = await fetch_card(name)
    message = card_reply(name, status, body)
    embeds = [discord.Embed.from_dict(e) for e in message["embeds"]]
    await interaction.response.send_message(
        content=message["content"] or None, embeds=embeds
    )


def _json_or_none(response: httpx.Response) -> dict | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


# ------------------------------------------------------------------------------ client


def env_allows_global_sync(env: dict[str, str] | None = None) -> bool:
    """Was a global sync explicitly opted into? Anything unset or odd means no."""
    source = os.environ if env is None else env
    return source.get(ALLOW_GLOBAL_SYNC_ENV, "").strip().lower() in _TRUTHY


def sync_plan(guild_id: int | None, allow_global_sync: bool) -> str:
    """Which command registration this configuration permits.

    "guild"  — scoped to one server, instant, and cannot touch global commands.
    "global" — destructive, and therefore only ever from an explicit opt-in.
    "none"   — neither was configured; register nothing and say so loudly.

    A guild id wins over the opt-in: the scoped sync is the safe one, so there is
    no configuration where both are set and the destructive path runs.
    """
    if guild_id is not None:
        return "guild"
    return "global" if allow_global_sync else "none"


class ScryingBot(discord.Client):
    """Default intents only. Slash commands do not need Message Content.

    Worth stating because a privileged-intent toggle nobody flipped is the single
    most common setup snag, and this bot deliberately does not need one.
    """

    def __init__(self, guild_id: int | None = None, allow_global_sync: bool = False) -> None:
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)
        self.guild_id = guild_id
        self.allow_global_sync = allow_global_sync
        register_commands(self.tree)

    async def setup_hook(self) -> None:
        # Registered before login completes, so a button tapped in the first
        # second after a restart already resolves.
        self.add_dynamic_items(FeedbackButton)

        plan = sync_plan(self.guild_id, self.allow_global_sync)
        if plan == "guild":
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info(
                "synced %d guild command(s) to guild %s: %s",
                len(synced), self.guild_id, ", ".join(c.name for c in synced),
            )
        elif plan == "global":
            log.warning(
                "%s is set: syncing GLOBALLY, which replaces this application's entire "
                "global command set. Anything else registered globally under application "
                "id %s is about to stop existing.",
                ALLOW_GLOBAL_SYNC_ENV, getattr(self.user, "id", "?"),
            )
            synced = await self.tree.sync()
            log.info(
                "synced %d global command(s): %s — global commands can take up to an "
                "hour to appear in clients",
                len(synced), ", ".join(c.name for c in synced),
            )
        else:
            # Connect and serve, but touch nothing: an unconfigured process must
            # not be able to destroy an application's commands.
            log.error(_NO_SYNC_MESSAGE, GUILD_ENV, ALLOW_GLOBAL_SYNC_ENV,
                      GUILD_ENV, ALLOW_GLOBAL_SYNC_ENV)

    async def on_ready(self) -> None:
        user = self.user
        log.info(
            "connected to Discord as %s (id %s), in %d guild(s)",
            user, getattr(user, "id", "?"), len(self.guilds),
        )
        for guild in self.guilds:
            log.info("  guild: %s (id %s)", guild.name, guild.id)
        if not self.guilds:
            log.warning(
                "this bot is not in any guild yet — invite it with the OAuth2 URL "
                "for application id %s, scopes bot + applications.commands",
                getattr(user, "id", "?"),
            )


def register_commands(tree: app_commands.CommandTree) -> None:
    """`/scry theme:<text> …` and `/search name:<text>`.

    Two commands, two different questions, and Discord shows each description
    string in the picker as the user types — so the disambiguation lands at the
    moment of choosing. The distinguishing word is capitalised in both, because
    that is the word that decides which one you wanted.
    """

    @tree.command(
        name="search",
        description="Look up one card by NAME — instant, from the local oracle corpus.",
    )
    @app_commands.describe(name="The card name. Spelling, case and accents are forgiving.")
    async def search(
        interaction: discord.Interaction,
        name: app_commands.Range[str, 1, 200],
    ) -> None:
        # Deliberately NO defer(): this answers inside the 3-second window.
        try:
            await run_search(interaction, name.strip())
        except Exception as exc:                     # noqa: BLE001 - never a dead spinner
            log.exception("unhandled error in /search")
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    content=render.search_failed_message(f"{type(exc).__name__}: {exc}")
                )

    @tree.command(
        name="scry",
        description="Find commanders by what their ARTWORK depicts, means or evokes.",
    )
    @app_commands.describe(
        theme="What the artwork should look or feel like, in your own words.",
        k="How many results (1-5, default 5).",
        band="Popularity band, 1 least played to 5 most played.",
        colors="Colour identity filter, e.g. WUB. Letters from WUBRG.",
    )
    async def scry(
        interaction: discord.Interaction,
        theme: app_commands.Range[str, 1, 300],
        k: app_commands.Range[int, 1, 5] = 5,
        band: app_commands.Range[int, 1, 5] | None = None,
        colors: str | None = None,
    ) -> None:
        # FIRST, before anything else: 3 seconds is the whole budget, and the
        # /health call below is already outside it.
        await interaction.response.defer(thinking=True)

        try:
            normalized = normalize_colors(colors)
        except ValueError as exc:
            await _edit(interaction, content=str(exc))
            return

        try:
            await run_scry(interaction, theme.strip(), k, band, normalized)
        except Exception as exc:                     # noqa: BLE001 - never a dead spinner
            log.exception("unhandled error in /scry")
            await _edit(
                interaction,
                content=render.search_failed_message(f"{type(exc).__name__}: {exc}"),
            )


# -------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(
            f"error: {TOKEN_ENV} is not set. It lives in "
            "~/.config/scrying-pool/bot.env (mode 600, outside the repo), which "
            "scrying-bot.service loads via EnvironmentFile=.",
            file=sys.stderr,
        )
        return 2

    raw_guild = os.environ.get(GUILD_ENV, "").strip()
    guild_id: int | None = None
    if raw_guild:
        try:
            guild_id = int(raw_guild)
        except ValueError:
            print(
                f"error: {GUILD_ENV}={raw_guild!r} is not a numeric guild id.",
                file=sys.stderr,
            )
            return 2

    allow_global_sync = env_allows_global_sync()

    # The URL is safe to print; the token is not, and is never logged anywhere.
    log.info("scrying-bot starting, API at %s", api_base())
    plan = sync_plan(guild_id, allow_global_sync)
    if plan == "none":
        # Logged here as well as in setup_hook so it is visible immediately in
        # `journalctl`, before the gateway connection is even attempted.
        log.error(_NO_SYNC_MESSAGE, GUILD_ENV, ALLOW_GLOBAL_SYNC_ENV,
                  GUILD_ENV, ALLOW_GLOBAL_SYNC_ENV)

    client = ScryingBot(guild_id=guild_id, allow_global_sync=allow_global_sync)
    try:
        client.run(token, log_handler=None)
    except discord.LoginFailure:
        # Never echo the token, not even a prefix of it.
        print(
            f"error: Discord rejected the token in {TOKEN_ENV}. Regenerate it in "
            "the developer portal and update ~/.config/scrying-pool/bot.env.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
