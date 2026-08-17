"""Result dicts -> Discord message JSON. Pure functions, no `discord` import.

Everything user-facing that the bot says is built here, and nothing here knows
that discord.py exists: embeds come out as **plain dicts matching Discord's embed
JSON**, which `bot.py` hands to `discord.Embed.from_dict`. That is deliberate and
it is the reason `tests/test_render.py` needs nothing installed but pytest — the
property the existing suite has and must not lose.

Two decisions in here are decisions rather than wording, and both have tests:

* **"Popularity band", never "Power level."** The composite is 0.4 x deck count +
  0.25 x price + 0.2 x cmc + 0.15 x a cEDH flag that saturates across nearly the
  whole corpus. `config.toml` says in its own comments that deck count is
  "popularity, not power". The label in the interface says what the number
  measures.
* **Stretches are labelled, never silently mixed in.** `select()` fills out `k`
  with below-the-bar results; a user who asked for five and got two real matches
  has to see that at a glance, so stretches sort last, are coloured differently,
  are titled `· STRETCH`, and are counted in the content line.
"""

from __future__ import annotations

from typing import Any, Iterable

# ------------------------------------------------------------------------- constants

# Embed accent. Green when the vision model confirmed the art, blue when the
# result passed the judge but was not verified, grey when it is a stretch.
COLOR_VERIFIED = 0x2ECC71
COLOR_PASSING = 0x3498DB
COLOR_STRETCH = 0x95A5A6

# Discord's own limits, applied here rather than discovered as a 400 at send time.
MAX_CONTENT = 2000
MAX_DESCRIPTION = 1000      # Discord allows 4096; a rationale is one sentence
MAX_TITLE = 256
MAX_FIELD_VALUE = 1024
MAX_FOOTER = 2048
MAX_EMBEDS = 5              # k is capped at 5, so this is a floor not a squeeze

PASS_FIT = 0.5              # mirrors cts.judge.PASS_FIT; only ever used in prose

# custom_id grammar: sp:v1:<query_id>:<illustration_id>:<u|d>, ~52 chars against
# Discord's 100-character limit with a UUID in the middle.
CUSTOM_ID_PREFIX = "sp:v1"
MAX_CUSTOM_ID = 100

LINK_LABELS = (
    ("edhrec", "EDHREC"),
    ("edhrec_theme", "theme"),
    ("scryfall", "Scryfall"),
    ("tcgplayer", "TCGplayer"),
)

# The one banner that is a refusal-shaped fact rather than an error: Ollama down
# means a full, fast, entirely unjudged result set. Degraded output plus an honest
# label beats a refusal, so it is rendered as a warning over real results.
DEGRADED_PREFIX = "⚠️"


def _truncate(text: str, limit: int) -> str:
    """Cut to `limit` characters, ellipsis included in the budget."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


# --------------------------------------------------------------------- custom_id codec


def encode_custom_id(query_id: int | str, illustration_id: str, accepted: bool) -> str:
    """`sp:v1:<query_id>:<illustration_id>:<u|d>`.

    The vote's whole identity lives in this string because the buttons are
    **persistent**: the entire rationale for splitting the bot from the API is
    that the bot restarts constantly, and a conventional `View` dies with its
    process, leaving every button in the channel silently dead. A tap three weeks
    and forty restarts later still resolves from the id alone.
    """
    vote = "u" if accepted else "d"
    custom_id = f"{CUSTOM_ID_PREFIX}:{query_id}:{illustration_id}:{vote}"
    if len(custom_id) > MAX_CUSTOM_ID:
        raise ValueError(
            f"custom_id is {len(custom_id)} characters, over Discord's "
            f"{MAX_CUSTOM_ID}: {custom_id!r}"
        )
    return custom_id


def decode_custom_id(custom_id: str) -> tuple[int, str, bool] | None:
    """Inverse of `encode_custom_id`. None for anything that is not ours.

    Returns None rather than raising: this parses attacker-adjacent input (any
    component id Discord hands the bot) and the caller's only sane response to
    garbage is to ignore it.
    """
    parts = str(custom_id).split(":")
    if len(parts) != 5:
        return None
    namespace, version, raw_query_id, illustration_id, vote = parts
    if (namespace, version) != tuple(CUSTOM_ID_PREFIX.split(":")):
        return None
    if vote not in ("u", "d") or not illustration_id:
        return None
    try:
        query_id = int(raw_query_id)
    except ValueError:
        return None
    return query_id, illustration_id, vote == "u"


# ------------------------------------------------------------------------ placeholders


def placeholder(health: dict | None) -> str:
    """The line posted into the deferred response before the search starts.

    Built from `/health` so that a four-minute search at 03:20 on a Sunday is
    explained rather than mysterious. Information, not enforcement — a refresh
    makes searches slow, and this design refuses to invent an outage out of that.
    """
    if not isinstance(health, dict):
        return "🔮 scrying… ~80s"

    search = health.get("search") or {}
    queued = search.get("queued") or 0
    in_flight = search.get("in_flight") or 0
    ahead = int(queued) + int(in_flight)

    line = "🔮 scrying… "
    if ahead > 0:
        # ~107s worst case each; round to whole minutes because the estimate does
        # not deserve more precision than that.
        minutes = max(1, round((ahead + 1) * 90 / 60))
        plural = "search" if ahead == 1 else "searches"
        line += f"queued behind {ahead} {plural}, ~{minutes} min"
    else:
        line += "~80s"

    if (health.get("refresh") or {}).get("running") is True:
        line += (
            "\n⚠️ the weekly corpus refresh is running, so this will be slow — "
            "several minutes rather than ~80s."
        )
    elif (health.get("ollama") or {}).get("reachable") is False:
        line += (
            "\n⚠️ Ollama is unreachable — results will be keyword-ranked only, "
            "nothing judged or verified."
        )
    return _truncate(line, MAX_CONTENT)


# ----------------------------------------------------------------------------- embeds


def _link_line(links: dict | None) -> str:
    """`[EDHREC](…) · [Scryfall](…)`, each key omitted entirely when absent."""
    if not isinstance(links, dict):
        return ""
    parts = [
        f"[{label}]({links[key]})" for key, label in LINK_LABELS if links.get(key)
    ]
    return " · ".join(parts)


def _colors_field(color_identity: Any) -> str:
    value = str(color_identity or "").strip().upper()
    return value or "C"


def result_embed(result: dict, position: int) -> dict:
    """One result -> one Discord embed dict."""
    stretch = bool(result.get("stretch"))
    verified = bool(result.get("verified"))

    title_bits = [str(result.get("name") or "unknown card")]
    if result.get("mana_cost"):
        title_bits.append(str(result["mana_cost"]))
    title = " ".join(title_bits)
    if stretch:
        title += " · STRETCH"

    if stretch:
        color = COLOR_STRETCH
    elif verified:
        color = COLOR_VERIFIED
    else:
        color = COLOR_PASSING

    description_parts: list[str] = []
    if result.get("rationale"):
        description_parts.append(str(result["rationale"]))
    if result.get("verify_note"):
        # Surfaced rather than swallowed: "no local art crop to verify against"
        # is a fact about this result, and hiding it makes the colour a lie.
        description_parts.append(f"_{result['verify_note']}_")
    description = _truncate("\n\n".join(description_parts), MAX_DESCRIPTION)

    fit = result.get("fit")
    fit_text = "—" if fit is None else f"{float(fit):.2f}"
    if stretch:
        fit_text += f" (below the {PASS_FIT} bar)"

    band = result.get("band")
    # "Popularity band", never "Power level". See the module docstring.
    band_text = "unknown" if band is None else f"{band}/5"

    fields = [
        {"name": "Colours", "value": _colors_field(result.get("color_identity")), "inline": True},
        {"name": "Popularity band", "value": band_text, "inline": True},
        {"name": "Fit", "value": fit_text, "inline": True},
    ]

    link_line = _link_line(result.get("links"))
    if link_line:
        fields.append(
            {"name": "Links", "value": _truncate(link_line, MAX_FIELD_VALUE), "inline": False}
        )

    footer_bits = [
        str(result.get("set_code") or "?").upper(),
        str(result.get("artist") or "unknown artist"),
    ]
    try:
        art_count = int(result.get("art_count") or 1)
    except (TypeError, ValueError):
        art_count = 1
    if art_count > 1:
        footer_bits.append(f"1 of {art_count} arts")
    if verified:
        footer_bits.append("vision verified")

    embed: dict = {
        "title": _truncate(f"{position}. {title}", MAX_TITLE),
        "color": color,
        "fields": fields,
        "footer": {"text": _truncate(" · ".join(footer_bits), MAX_FOOTER)},
    }
    if description:
        embed["description"] = description

    art_crop = (result.get("links") or {}).get("art_crop")
    if art_crop:
        embed["thumbnail"] = {"url": str(art_crop)}
    return embed


def order_results(results: Iterable[dict]) -> list[dict]:
    """Passing results first, stretches last, order otherwise preserved.

    `select()` already appends stretches to fill out `k`, but it is not contractually
    sorted that way, and the content line's "3 of 5 clear the bar" only reads
    correctly against a message where the stretches are the last two embeds.
    """
    ordered = list(results)
    return [r for r in ordered if not r.get("stretch")] + [
        r for r in ordered if r.get("stretch")
    ]


# --------------------------------------------------------------------- the whole message


def content_line(theme: str, outcome: dict) -> str:
    """The text above the embeds. Mirrors the CLI's own honesty, which got this right."""
    plan = outcome.get("plan") or {}
    results = outcome.get("results") or []

    header = f'🔮 "{theme}"'
    bits: list[str] = []

    literal = plan.get("literal_weight")
    interpretive = plan.get("interpretive_weight")
    if isinstance(literal, (int, float)) and isinstance(interpretive, (int, float)):
        bits.append(f"{literal:.0%} literal / {interpretive:.0%} interpretive")
    if plan.get("band") is not None:
        bits.append(f"band {plan['band']}")
    if plan.get("colors"):
        bits.append(f"colors {str(plan['colors']).upper()}")
    if bits:
        header += " · " + " · ".join(bits)

    lines = [header]

    if outcome.get("relaxed"):
        lines.append(f"note: {outcome['relaxed']}")

    if not results:
        counts = plan.get("counts") or {}
        lines.append(
            f"no matches. {counts.get('commanders', 0)} commanders retrieved, "
            f"{counts.get('candidates', 0)} survived the filters."
        )
        return _truncate("\n".join(lines), MAX_CONTENT)

    passing = sum(1 for r in results if not r.get("stretch"))
    if passing < len(results):
        lines.append(
            f"{passing} of {len(results)} results clear the {PASS_FIT} fit bar; "
            "the rest are stretches."
        )
    return _truncate("\n".join(lines), MAX_CONTENT)


# A note is a *warning* when it says a stage of the pipeline did not run. Anything
# else — a slot filter that matched nothing, renormalised route weights — is the
# search narrating its own reasoning, which the CLI prints as a plain "note:".
_WARNING_MARKERS = (
    "unavailable",
    "failed",
    "fell back",
    "no embedding",
    "index is empty",
)


def _is_warning(note: str) -> bool:
    lowered = note.lower()
    return any(marker in lowered for marker in _WARNING_MARKERS)


def degraded_banner(outcome: dict) -> str:
    """The lines above the embeds, built from `plan.notes` — one source, two renderings.

    `service.degraded` is true whenever `plan.notes` is non-empty, and in practice
    almost every search produces a note: a slot filter that matched nothing, route
    weights that were floored. Marking all of those ⚠️ would put a warning on
    essentially every result set, which is the fastest way to make the one banner
    that matters — "Ollama is unreachable, nothing was judged" — invisible.

    So the notes are all carried, verbatim, exactly as the spec requires, and the
    CLI's own distinction is kept: `⚠️` for a stage that did not run, `note:` for
    the search explaining itself. Two renderings of one fact, one source.
    """
    service = outcome.get("service") or {}
    plan = outcome.get("plan") or {}
    if not service.get("degraded"):
        return ""

    notes = [str(n) for n in (plan.get("notes") or []) if str(n).strip()]
    if plan.get("vision_verified") is False and not any("vision" in n for n in notes):
        notes.insert(
            0, "vision verification unavailable — results are judge-ordered and unverified"
        )
    if not notes:
        return ""
    return "\n".join(
        f"{DEGRADED_PREFIX} {note}" if _is_warning(note) else f"note: {note}"
        for note in notes
    )


def render_message(theme: str, outcome: dict) -> dict:
    """`{"content": str, "embeds": [dict, …]}` — everything the edited reply carries."""
    results = order_results(outcome.get("results") or [])[:MAX_EMBEDS]

    pieces = [content_line(theme, outcome)]
    banner = degraded_banner(outcome)
    if banner:
        pieces.append(banner)

    return {
        "content": _truncate("\n".join(p for p in pieces if p), MAX_CONTENT),
        "embeds": [result_embed(r, i) for i, r in enumerate(results, start=1)],
    }


def button_specs(outcome: dict) -> list[dict]:
    """Ten button descriptors in embed order: 👍1-👍5 then 👎1-👎5.

    Returned as dicts rather than components so this stays importable without
    discord.py. `bot.py` turns each into a `discord.ui.Button`.
    """
    query_id = outcome.get("query_id")
    if query_id is None:
        return []

    results = order_results(outcome.get("results") or [])[:MAX_EMBEDS]
    specs: list[dict] = []
    for accepted, emoji in ((True, "👍"), (False, "👎")):
        for position, result in enumerate(results, start=1):
            illustration_id = result.get("illustration_id")
            if not illustration_id:
                continue
            try:
                custom_id = encode_custom_id(query_id, str(illustration_id), accepted)
            except ValueError:
                continue
            specs.append(
                {
                    "custom_id": custom_id,
                    "label": str(position),
                    "emoji": emoji,
                    "row": 0 if accepted else 1,
                    "accepted": accepted,
                    "name": str(result.get("name") or "that result"),
                }
            )
    return specs


# ------------------------------------------------------------------------ error prose


def api_down_message() -> str:
    return (
        "The search service isn't running on the host. Someone needs to check "
        "`systemctl --user status scrying-api`."
    )


def busy_message(body: dict | None = None) -> str:
    queued = (body or {}).get("queued")
    depth = f"{queued} searches are" if isinstance(queued, int) else "Four searches are"
    return f"{depth} already queued (~7 min of work). Try again shortly."


def search_failed_message(detail: str) -> str:
    return (
        f"The search failed: {_truncate(detail, 500)}. "
        "This is logged — check `journalctl --user -u scrying-api`."
    )


def timeout_message(seconds: float) -> str:
    return (
        f"Gave up waiting after {round(seconds / 60)} minutes. The search may still be "
        "running — check /health."
    )
