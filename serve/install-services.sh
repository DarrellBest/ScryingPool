#!/usr/bin/env bash
# Install the Scrying Pool serving layer as two systemd *user* units.
#
#   scrying-api.service   uvicorn over cts.search, bound 127.0.0.1:8077 and nothing
#                         else. Holds the connection, the index and the warm models
#                         between requests, which is why it is expensive to start
#                         (~4-7s index build, ~17s cold model load) and why the bot
#                         is a separate process.
#   scrying-bot.service   discord.py. Holds no corpus state, so restarting it to
#                         change an embed colour costs nothing.
#
# User units, not system units, for the same reason install-timer.sh uses one: the
# corpus, the venv and the Ollama models all belong to this user, and a system unit
# would need every one of those paths spelled out and permissioned.
#
# Usage:
#   serve/install-services.sh              install, enable and start both
#   serve/install-services.sh --dry-run    print the unit files, touch nothing

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "-n" ]; then
    DRY_RUN=1
elif [ $# -gt 0 ]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

# --- Locate the repo -------------------------------------------------------
# Same trick as install-timer.sh: this script lives in serve/, so the repo root is
# its parent. Anyone cloning this puts it wherever they like.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -d "$REPO_DIR/cts" ] || [ ! -d "$REPO_DIR/serve" ]; then
    echo "error: $REPO_DIR does not look like the Scrying Pool repo (no cts/ + serve/)." >&2
    echo "       Run this script from inside your clone: serve/install-services.sh" >&2
    exit 1
fi

# --- Locate the interpreter ------------------------------------------------
# Absolute path baked into ExecStart, because a user unit inherits almost no
# environment and "python3" alone may not resolve at boot.
if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON="$REPO_DIR/.venv/bin/python"
elif PYTHON="$(command -v python3)"; then
    :
else
    echo "error: no $REPO_DIR/.venv/bin/python and no python3 on PATH." >&2
    echo "       Create a venv and install the serving layer:" >&2
    echo "         python3 -m venv .venv" >&2
    echo "         .venv/bin/pip install -r requirements.txt -r serve-requirements.txt" >&2
    exit 1
fi

if [ ! -f "$REPO_DIR/config.toml" ]; then
    echo "warning: no config.toml in $REPO_DIR — scrying-api will fail until you create one." >&2
fi

if ! "$PYTHON" -c 'import fastapi, uvicorn, discord, httpx' >/dev/null 2>&1; then
    echo "warning: $PYTHON is missing the serving dependencies." >&2
    echo "         $PYTHON -m pip install -r $REPO_DIR/serve-requirements.txt" >&2
fi

# --- The secret ------------------------------------------------------------
# Outside the repo, mode 600, loaded by EnvironmentFile=. Never in a unit file,
# never in this script, never a default in Python. This script only checks that it
# exists; it never reads it and never prints its contents.
ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/scrying-pool/bot.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "warning: $ENV_FILE does not exist — scrying-bot will not start." >&2
    echo "         Create it, mode 600, with:" >&2
    echo "           SCRYING_DISCORD_TOKEN=..." >&2
    echo "           SCRYING_API_URL=http://127.0.0.1:8077" >&2
    echo "           SCRYING_DISCORD_GUILD_ID=...   # guild-scoped = instant; without it" >&2
    echo "                                          # the bot registers NO commands" >&2
fi

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
API_PATH="$UNIT_DIR/scrying-api.service"
BOT_PATH="$UNIT_DIR/scrying-bot.service"

# No dependency on Ollama: it is a *system* unit and these are *user* units, so the
# ordering is not expressible — and more importantly the API must tolerate Ollama
# being down at boot. It builds its index and serves /health with status=degraded
# rather than exiting, because preflight-and-die would crash-loop through every
# reboot where Ollama is slow to come up.
#
# No MemoryMax=, deliberately: an index rebuild holds two 523MB matrices at once,
# and an OOM kill mid-rebuild is far worse than the spike it would prevent.
#
# Minimal hardening. PrivateTmp is safe; ProtectSystem=strict is not, because the
# service legitimately writes data/commanders.db and the WAL beside it.
# StartLimit* is not in the design document, and it is here because of something
# the first deployment actually did: with port 8077 already held by a stray
# process, the unit restarted 32 times, paying a full ~5s index build over 170,487
# propositions on every one of them. systemd's defaults (5 starts per 10s) never
# fire for a service that takes 11 seconds to fail, so "on-failure" meant "forever"
# for exactly the failure that cannot fix itself. Five tries in five minutes, then
# stop and stay diagnosable — same reasoning the bot's limit already had.
API_TEXT="[Unit]
Description=Scrying Pool search API (loopback only)
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=$PYTHON -m serve.api
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5
PrivateTmp=true

[Install]
WantedBy=default.target
"

# Wants=, not Requires=: if the API fails, the bot must stay up to *say* the API is
# down. Requires= would stop the bot along with it and turn a legible error message
# into silence.
#
# StartLimit* because a revoked token would otherwise crash-loop against Discord's
# login endpoint forever. Five tries in five minutes, then stop and stay stopped,
# which is both polite and diagnosable.
BOT_TEXT="[Unit]
Description=Scrying Pool Discord bot
Wants=scrying-api.service
After=scrying-api.service network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
EnvironmentFile=%h/.config/scrying-pool/bot.env
ExecStart=$PYTHON -m serve.bot
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=10
PrivateTmp=true

[Install]
WantedBy=default.target
"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "# would write $API_PATH"
    printf '%s\n' "$API_TEXT"
    echo "# would write $BOT_PATH"
    printf '%s\n' "$BOT_TEXT"
    echo "# would then: systemctl --user daemon-reload && systemctl --user enable --now scrying-api.service scrying-bot.service"
    exit 0
fi

mkdir -p "$UNIT_DIR"
printf '%s' "$API_TEXT" >"$API_PATH"
printf '%s' "$BOT_TEXT" >"$BOT_PATH"
echo "wrote $API_PATH"
echo "wrote $BOT_PATH"
echo "  WorkingDirectory=$REPO_DIR"
echo "  ExecStart=$PYTHON -m serve.api"
echo "  ExecStart=$PYTHON -m serve.bot"

# --- Enable ----------------------------------------------------------------
if ! command -v systemctl >/dev/null 2>&1; then
    echo >&2
    echo "error: systemctl not found — this machine does not run systemd." >&2
    echo "       The unit files above are written and valid; copy them to a systemd host," >&2
    echo "       or run '$PYTHON -m serve.api' and '$PYTHON -m serve.bot' from $REPO_DIR" >&2
    echo "       under some other supervisor." >&2
    exit 1
fi

if ! systemctl --user show-environment >/dev/null 2>&1; then
    echo >&2
    echo "error: 'systemctl --user' is unavailable — there is no user service manager" >&2
    echo "       on this session's D-Bus (common in containers, in CI, and over ssh" >&2
    echo "       when no login session exists)." >&2
    echo "       The unit files above are written. Once you have a real user session:" >&2
    echo "         loginctl enable-linger \"\$USER\"" >&2
    echo "         systemctl --user daemon-reload" >&2
    echo "         systemctl --user enable --now scrying-api.service scrying-bot.service" >&2
    exit 1
fi

systemctl --user daemon-reload
systemctl --user enable --now scrying-api.service scrying-bot.service

echo
systemctl --user --no-pager --lines=0 status scrying-api.service scrying-bot.service || true

cat <<EOF

Next steps
  1. Let both units survive logout:
         loginctl enable-linger "$USER"
     Without linger, the user manager stops when your last session ends and the
     bot disconnects from Discord the moment you close your ssh session.

  2. Watch the API finish its index build (~4-7s) and the bot sync its commands:
         journalctl --user -u scrying-api -f
         journalctl --user -u scrying-bot -f

  3. Check the API without Discord in the loop:
         curl -s localhost:8077/health | jq
         curl -s localhost:8077/search -H 'content-type: application/json' \\
              -d '{"theme":"a hooded figure alone at dusk","k":3}' | jq '.results[].name'

  4. If the bot logs "not in any guild yet", invite it: the OAuth2 URL is
     https://discord.com/api/oauth2/authorize?client_id=<APPLICATION_ID>&permissions=0&scope=bot%20applications.commands
     with the application id from the developer portal (the bot logs its own id
     on connect).
EOF
