#!/usr/bin/env bash
# Install the weekly Scrying Pool refresh as a systemd *user* timer.
#
# systemd rather than cron, for three reasons that matter here:
#   * Persistent=true runs a missed job if the machine was off on Sunday.
#   * systemd refuses to start a Type=oneshot service that is already running,
#     so a long refresh cannot overlap itself and needs no lockfile.
#   * output goes to journald, so a failed run is recoverable with
#     `journalctl --user -u cts-refresh` instead of a redirect you forgot.
#
# Usage:
#   ./install-timer.sh              install and enable
#   ./install-timer.sh --dry-run    print the unit files, touch nothing

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "-n" ]; then
    DRY_RUN=1
elif [ $# -gt 0 ]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

# --- Locate the repo -------------------------------------------------------
# The spec hardcodes %h/commander-theme-search. This repo is named ScryingPool
# and anyone cloning it puts it wherever they like, so derive the real path
# from where this script actually lives.
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$REPO_DIR/cts" ]; then
    echo "error: $REPO_DIR does not look like the Scrying Pool repo (no cts/ directory)." >&2
    echo "       Run this script from inside your clone: ./install-timer.sh" >&2
    exit 1
fi

# --- Locate the interpreter ------------------------------------------------
# Prefer the repo's virtualenv; fall back to python3 on PATH. Either way the
# absolute path is baked into ExecStart, because a systemd unit inherits almost
# no environment and "python3" alone may not resolve at boot.
if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON="$REPO_DIR/.venv/bin/python"
elif PYTHON="$(command -v python3)"; then
    :
else
    echo "error: no $REPO_DIR/.venv/bin/python and no python3 on PATH." >&2
    echo "       Create a venv (python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)" >&2
    echo "       or install python3, then re-run." >&2
    exit 1
fi

if [ ! -f "$REPO_DIR/config.toml" ]; then
    echo "warning: no config.toml in $REPO_DIR — the timer will fail until you create one." >&2
fi

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_PATH="$UNIT_DIR/cts-refresh.service"
TIMER_PATH="$UNIT_DIR/cts-refresh.timer"

SERVICE_TEXT="[Unit]
Description=Commander theme search weekly refresh
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$REPO_DIR
ExecStart=$PYTHON -m cts refresh
"

# RandomizedDelaySec spreads the EDHREC hits off the hour: basic courtesy,
# given every other scraper on earth is also scheduled at 03:00.
TIMER_TEXT="[Unit]
Description=Weekly commander refresh

[Timer]
OnCalendar=Sun 03:00
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "# would write $SERVICE_PATH"
    printf '%s\n' "$SERVICE_TEXT"
    echo "# would write $TIMER_PATH"
    printf '%s\n' "$TIMER_TEXT"
    echo "# would then: systemctl --user daemon-reload && systemctl --user enable --now cts-refresh.timer"
    exit 0
fi

mkdir -p "$UNIT_DIR"
printf '%s' "$SERVICE_TEXT" >"$SERVICE_PATH"
printf '%s' "$TIMER_TEXT" >"$TIMER_PATH"
echo "wrote $SERVICE_PATH"
echo "wrote $TIMER_PATH"
echo "  WorkingDirectory=$REPO_DIR"
echo "  ExecStart=$PYTHON -m cts refresh"

# --- Enable ----------------------------------------------------------------
if ! command -v systemctl >/dev/null 2>&1; then
    echo >&2
    echo "error: systemctl not found — this machine does not run systemd." >&2
    echo "       The unit files above are written and valid; copy them to a systemd host," >&2
    echo "       or schedule '$PYTHON -m cts refresh' (run from $REPO_DIR) some other way." >&2
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
    echo "         systemctl --user enable --now cts-refresh.timer" >&2
    exit 1
fi

systemctl --user daemon-reload
systemctl --user enable --now cts-refresh.timer

echo
systemctl --user list-timers cts-refresh.timer --all || true

cat <<EOF

Next steps
  1. Let the timer fire without you logged in:
         loginctl enable-linger "$USER"
     Without linger, the user manager stops when your last session ends and a
     Sunday-morning run on a headless box never happens.

  2. Verify the job itself once, now, before trusting the schedule:
         systemctl --user start cts-refresh.service
         journalctl --user -u cts-refresh -f
     A first run with Ollama down exits 1 immediately and changes nothing, which
     is exactly what you want to see confirmed by hand.

  3. Later: systemctl --user list-timers cts-refresh.timer
            journalctl --user -u cts-refresh --since "last week"
EOF
