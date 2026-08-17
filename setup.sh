#!/usr/bin/env bash
#
# Scrying Pool — one-command setup from the prebuilt corpus.
#
# Creates .venv, installs requirements.txt, pulls the three Ollama models named
# in config.toml, then downloads and extracts the prebuilt data archives into
# data/ with sha256 verification.
#
# Idempotent: every step checks its own output first and re-running is cheap.
# Never overwrites existing files in data/ that it did not put there — pass
# --force if you really mean to replace them.
#
# Usage:
#   ./setup.sh                      # everything, including the EDHREC cache
#   ./setup.sh --no-edhrec-cache    # skip the 74 MB EDHREC cache (adds ~45 min to ingest)
#   ./setup.sh --skip-models        # data only; pull the models yourself
#   ./setup.sh --keep-archives      # leave the .tar.gz files in data/ after extracting
#   ./setup.sh --force              # replace existing data/ artifacts

set -euo pipefail

# --- Locations -----------------------------------------------------------
# Resolve the repo root from the script's own path so this works from any cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
DATA_DIR="$REPO_DIR/data"
VENV_DIR="$REPO_DIR/.venv"
CONFIG_FILE="$REPO_DIR/config.toml"

# --- Prebuilt corpus archives -------------------------------------------
# Hosted on pCloud. The publink codes below are permanent; the actual download
# URL is minted per-request through pCloud's public API (direct links expire),
# which is what resolve_pcloud_url() does.
#
# Built 2026-08-16 from a complete corpus: 3,202 commanders, 5,530 artworks
# described, 170,487 propositions embedded.
DB_CODE="XZQi4VJZoWRa3bLxI3b2yrOtIfymBFUGmPJX"
DB_SHA="e27653ba0d99ea6c41f24441d9610b98e16feb438ad2f7697da5273c28e88389"
DB_BYTES="584080873"

ART_CODE="XZDq4VJZIPnO7mynBjRjOc3zsINTIhDr4x8X"
ART_SHA="6bd0d1994e7770bdb002fad613a0bbb0d4c37f8ab9d9e0ec0f0861a558fb1dff"
ART_BYTES="382187995"

EDHREC_CODE="XZ1q4VJZ4dEerExbkn4uh3lCxmKNf5wnGNFX"
EDHREC_SHA="014961e1e83750f4caffe636b9ba630b6b630cde11a612ddf0d655567d2300ec"
EDHREC_BYTES="77320794"

PCLOUD_API="https://api.pcloud.com/getpublinkdownload"

# --- Options -------------------------------------------------------------
WITH_EDHREC=1
SKIP_MODELS=0
KEEP_ARCHIVES=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-edhrec-cache) WITH_EDHREC=1 ;;
    --no-edhrec-cache)   WITH_EDHREC=0 ;;
    --skip-models)       SKIP_MODELS=1 ;;
    --keep-archives)     KEEP_ARCHIVES=1 ;;
    --force)             FORCE=1 ;;
    -h|--help)
      cat <<'USAGE'
Scrying Pool — one-command setup from the prebuilt corpus.

Creates .venv, installs requirements.txt, pulls the three Ollama models named
in config.toml, then downloads and extracts the prebuilt data archives into
data/ with sha256 verification.

Idempotent: every step checks its own output first and re-running is cheap.
Never overwrites existing files in data/ that it did not put there — pass
--force if you really mean to replace them.

Usage:
  ./setup.sh                      everything, including the EDHREC cache
  ./setup.sh --no-edhrec-cache    skip the 74 MB EDHREC cache (adds ~45 min to ingest)
  ./setup.sh --skip-models        data only; pull the Ollama models yourself
  ./setup.sh --keep-archives      leave the .tar.gz files in data/ after extracting
  ./setup.sh --force              replace existing data/ artifacts
  ./setup.sh --help               this message
USAGE
      exit 0
      ;;
    *)
      echo "setup.sh: unknown option '$1' (try --help)" >&2
      exit 2
      ;;
  esac
  shift
done

# --- Output helpers ------------------------------------------------------
if [[ -t 1 ]]; then
  C_STEP=$'\033[1;36m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
else
  C_STEP=""; C_OK=""; C_WARN=""; C_ERR=""; C_OFF=""
fi

STEP_N=0
step() { STEP_N=$((STEP_N + 1)); printf '\n%s==> [%d/6] %s%s\n' "$C_STEP" "$STEP_N" "$1" "$C_OFF"; }
info() { printf '    %s\n' "$1"; }
ok()   { printf '    %s✓%s %s\n' "$C_OK" "$C_OFF" "$1"; }
warn() { printf '    %s!%s %s\n' "$C_WARN" "$C_OFF" "$1"; }
die()  { printf '\n%serror:%s %s\n' "$C_ERR" "$C_OFF" "$1" >&2; exit 1; }

# =========================================================================
step "Checking prerequisites"
# =========================================================================

command -v python3 >/dev/null 2>&1 \
  || die "python3 not found on PATH. Scrying Pool needs Python 3.11+ (it uses tomllib). Install it and re-run."

PY_VER="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || die "python3 is $PY_VER, but Scrying Pool needs 3.11 or newer (tomllib is 3.11+). Install a newer Python and re-run."
ok "python3 $PY_VER"

command -v tar >/dev/null 2>&1 || die "tar not found on PATH. Install tar (and gzip) and re-run."
ok "tar"

DOWNLOADER=""
if command -v curl >/dev/null 2>&1; then
  DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget"
else
  die "neither curl nor wget found on PATH. Install one of them and re-run."
fi
ok "$DOWNLOADER"

SHA_CMD=""
if command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  SHA_CMD="shasum -a 256"
fi
if [[ -n "$SHA_CMD" ]]; then
  ok "sha256 via ${SHA_CMD%% *}"
else
  warn "no sha256sum/shasum; falling back to python3 hashlib (slower, still verified)"
fi

[[ -f "$CONFIG_FILE" ]] || die "no config.toml at '$CONFIG_FILE'. Run setup.sh from a checkout of the repo."

# Read the values out of config.toml rather than hardcoding them, so a user who
# has already swapped models or paths gets their own choices honoured.
read_config() {
  python3 - "$CONFIG_FILE" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    raw = tomllib.load(fh)
print(str(raw.get("ollama_url", "http://localhost:11434")).rstrip("/"))
print(str(raw.get("vision_model", "")))
print(str(raw.get("embed_model", "")))
print(str(raw.get("judge_model", "")))
print(str(raw.get("db_path", "data/commanders.db")))
print(str(raw.get("art_dir", "data/art")))
PY
}

{
  read -r OLLAMA_URL
  read -r VISION_MODEL
  read -r EMBED_MODEL
  read -r JUDGE_MODEL
  read -r DB_PATH
  read -r ART_DIR
} < <(read_config) || die "could not parse '$CONFIG_FILE' as TOML. Fix the syntax and re-run."

# Config paths are relative to the repo root.
[[ "$DB_PATH"  = /* ]] || DB_PATH="$REPO_DIR/$DB_PATH"
[[ "$ART_DIR"  = /* ]] || ART_DIR="$REPO_DIR/$ART_DIR"
EDHREC_DIR="$DATA_DIR/edhrec"

ok "config.toml → $OLLAMA_URL"
info "vision_model = $VISION_MODEL"
info "embed_model  = $EMBED_MODEL"
info "judge_model  = $JUDGE_MODEL"

# --- HTTP helpers (defined here, used from step 2 on) --------------------
http_get() {  # http_get URL -> body on stdout
  if [[ "$DOWNLOADER" == "curl" ]]; then
    curl -fsSL --max-time 60 "$1"
  else
    wget -qO- --timeout=60 "$1"
  fi
}

http_reachable() {  # http_reachable URL
  if [[ "$DOWNLOADER" == "curl" ]]; then
    curl -fsS --max-time 15 -o /dev/null "$1" 2>/dev/null
  else
    wget -q --timeout=15 -O /dev/null "$1" 2>/dev/null
  fi
}

# Ollama: the binary, then the server the config actually points at.
if [[ "$SKIP_MODELS" -eq 1 ]]; then
  warn "--skip-models: not checking Ollama"
else
  command -v ollama >/dev/null 2>&1 \
    || die "ollama not found on PATH. Install it from https://ollama.com/download, start it (\`ollama serve\`), then re-run. Or pass --skip-models to set up the data only."
  ok "ollama binary"

  http_reachable "$OLLAMA_URL/api/tags" \
    || die "no Ollama server reachable at '$OLLAMA_URL/api/tags' (ollama_url in config.toml). Start it with \`ollama serve\`, or point ollama_url at wherever yours is listening, then re-run."
  ok "ollama server reachable at $OLLAMA_URL"
fi

# =========================================================================
step "Creating the virtualenv and installing dependencies"
# =========================================================================

if [[ -x "$VENV_DIR/bin/python" ]]; then
  ok "reusing existing venv at '$VENV_DIR'"
else
  info "creating '$VENV_DIR'"
  python3 -m venv "$VENV_DIR" \
    || die "python3 -m venv failed. On Debian/Ubuntu this usually means the venv module is missing: apt install python3-venv"
  ok "venv created"
fi

VENV_PY="$VENV_DIR/bin/python"
info "installing requirements.txt"
"$VENV_PY" -m pip install --upgrade pip --quiet \
  || warn "could not upgrade pip; continuing with the bundled version"
"$VENV_PY" -m pip install -r "$REPO_DIR/requirements.txt" --quiet \
  || die "pip install -r requirements.txt failed. Scroll up for pip's error."
ok "requests, numpy, rank_bm25 installed"

# =========================================================================
step "Pulling Ollama models"
# =========================================================================

# Matches cts.ollama._normalize: "llama3" and "llama3:latest" are the same model.
normalize_model() { local n="$1"; printf '%s\n' "${n%:latest}"; }

model_present() {  # model_present NAME
  local want present
  want="$(normalize_model "$1")"
  present="$(http_get "$OLLAMA_URL/api/tags" 2>/dev/null | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for entry in data.get("models", []):
    for key in ("name", "model"):
        value = entry.get(key)
        if value:
            print(value[:-len(":latest")] if value.endswith(":latest") else value)
' || true)"
  printf '%s\n' "$present" | grep -Fxq -- "$want"
}

if [[ "$SKIP_MODELS" -eq 1 ]]; then
  warn "skipped (--skip-models). Pull these yourself before searching:"
  info "ollama pull $VISION_MODEL"
  info "ollama pull $EMBED_MODEL"
  info "ollama pull $JUDGE_MODEL"
else
  # `ollama pull` talks to OLLAMA_HOST, which may differ from config.toml's
  # ollama_url. Point it at the server we just checked.
  export OLLAMA_HOST="$OLLAMA_URL"
  for model in "$VISION_MODEL" "$EMBED_MODEL" "$JUDGE_MODEL"; do
    if [[ -z "${model// }" ]]; then
      warn "a model name is empty in config.toml — skipping"
      continue
    fi
    if model_present "$model"; then
      ok "$model already pulled"
    else
      info "pulling $model (this can take a while — the vision model is large)"
      ollama pull "$model" \
        || die "ollama pull '$model' failed. Check the name against \`ollama list\` / the Ollama registry, or edit config.toml."
      ok "$model pulled"
    fi
  done
fi

# =========================================================================
step "Downloading the prebuilt corpus"
# =========================================================================

mkdir -p "$DATA_DIR"

sha256_of() {  # sha256_of PATH
  if [[ -n "$SHA_CMD" ]]; then
    $SHA_CMD "$1" | awk '{print $1}'
  else
    python3 -c '
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
' "$1"
  fi
}

resolve_pcloud_url() {  # resolve_pcloud_url CODE -> direct https URL
  http_get "$PCLOUD_API?code=$1" | python3 -c '
import json, sys
data = json.load(sys.stdin)
if data.get("result") != 0:
    sys.exit("pCloud API said: %s" % data.get("error", data))
hosts = data.get("hosts") or []
if not hosts:
    sys.exit("pCloud API returned no download host")
print("https://" + hosts[0] + data["path"])
'
}

download_to() {  # download_to URL DEST
  if [[ "$DOWNLOADER" == "curl" ]]; then
    curl -fL --progress-bar -o "$2" "$1"
  else
    wget --show-progress -q -O "$2" "$1"
  fi
}

# fetch_archive LABEL CODE SHA BYTES ARCHIVE_NAME TARGET_PATH
#
# Skips entirely when the target is already in place and the stamp file records
# the same archive hash. Skips the download (but still extracts) when a verified
# copy of the archive is already sitting in data/.
fetch_archive() {
  local label="$1" code="$2" want_sha="$3" want_bytes="$4" name="$5" target="$6"
  local archive="$DATA_DIR/$name"
  local stamp="$DATA_DIR/.${name}.sha256"

  if [[ -e "$target" ]]; then
    if [[ -f "$stamp" ]] && [[ "$(cat "$stamp")" == "$want_sha" ]]; then
      ok "$label already installed at '$target' (hash matches) — skipping"
      return 0
    fi
    if [[ "$FORCE" -eq 0 ]]; then
      warn "$label: '$target' already exists but was not installed by this script."
      warn "Leaving it alone. Re-run with --force to replace it with the prebuilt copy."
      return 0
    fi
    warn "$label: --force given, replacing '$target'"
  fi

  if [[ -f "$archive" ]]; then
    info "$label: found '$archive', verifying"
    if [[ "$(sha256_of "$archive")" == "$want_sha" ]]; then
      ok "$label: existing archive verified — skipping download"
    else
      warn "$label: existing archive failed verification, re-downloading"
      rm -f "$archive"
    fi
  fi

  if [[ ! -f "$archive" ]]; then
    local mib=$(( want_bytes / 1048576 ))
    info "$label: resolving download URL"
    local url
    url="$(resolve_pcloud_url "$code")" \
      || die "could not resolve the pCloud link for $label. Check your network, or download it by hand from the URL in README.md and drop it at '$archive', then re-run."
    info "$label: downloading ${mib} MiB"
    download_to "$url" "$archive" \
      || die "download of $label failed. Re-run to resume, or fetch it by hand into '$archive'."
  fi

  local got_bytes got_sha
  got_bytes="$(python3 -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' "$archive")"
  [[ "$got_bytes" == "$want_bytes" ]] \
    || die "$label: size mismatch — expected $want_bytes bytes, got $got_bytes. Delete '$archive' and re-run."
  info "$label: verifying sha256"
  got_sha="$(sha256_of "$archive")"
  [[ "$got_sha" == "$want_sha" ]] \
    || die "$label: sha256 mismatch.
  expected $want_sha
  got      $got_sha
The download is corrupt or the file has changed. Delete '$archive' and re-run."
  ok "$label: sha256 verified"

  info "$label: extracting into '$DATA_DIR'"
  tar -xzf "$archive" -C "$DATA_DIR" \
    || die "$label: tar extraction failed. Delete '$archive' and re-run."
  printf '%s\n' "$want_sha" > "$stamp"
  ok "$label: extracted"

  if [[ "$KEEP_ARCHIVES" -eq 1 ]]; then
    info "$label: keeping '$archive' (--keep-archives)"
  else
    rm -f "$archive"
  fi
}

fetch_archive "database"  "$DB_CODE"  "$DB_SHA"  "$DB_BYTES"  "scryingpool-db.tar.gz"  "$DB_PATH"
fetch_archive "art crops" "$ART_CODE" "$ART_SHA" "$ART_BYTES" "scryingpool-art.tar.gz" "$ART_DIR"

if [[ "$WITH_EDHREC" -eq 1 ]]; then
  fetch_archive "EDHREC cache" "$EDHREC_CODE" "$EDHREC_SHA" "$EDHREC_BYTES" \
    "scryingpool-edhrec-cache.tar.gz" "$EDHREC_DIR"
else
  warn "EDHREC cache skipped (--no-edhrec-cache). The first \`cts ingest\` will re-scrape it at 1 req/sec (~45 min)."
fi

info "data/bulk/ is not shipped — it is a re-downloadable Scryfall dump and \`cts ingest\` fetches it."

# =========================================================================
step "Verifying the install"
# =========================================================================

cd "$REPO_DIR"
"$VENV_PY" - <<'PY' || die "verification failed — see the error above."
import sqlite3
import sys
from pathlib import Path

from cts.config import load_config
from cts.ollama import preflight

cfg = load_config("config.toml")
print(f"    config      : ollama_url={cfg.ollama_url}")

try:
    missing = preflight(cfg)
except Exception as exc:  # connection refused, HTTP error, ...
    print(f"    ollama      : UNREACHABLE ({exc.__class__.__name__}) — start it before searching")
else:
    if missing:
        print("    ollama      : missing models -> " + ", ".join(missing))
    else:
        print("    ollama      : all three configured models present")

db = Path(cfg.db_path)
if not db.is_file():
    sys.exit(f"no database at {db}")

con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
try:
    rows = {
        name: con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in ("cards", "arts", "descriptions", "props", "embeddings")
    }
finally:
    con.close()

print(f"    database    : {db}")
print(f"    commanders  : {rows['cards']:,}")
print(f"    artworks    : {rows['arts']:,}  ({rows['descriptions']:,} described)")
print(f"    propositions: {rows['props']:,}  ({rows['embeddings']:,} embedded)")

art_dir = Path(cfg.art_dir)
n_art = len(list(art_dir.glob("*.jpg"))) if art_dir.is_dir() else 0
print(f"    art crops   : {n_art:,} jpgs in {art_dir}")

if rows["descriptions"] < rows["arts"]:
    print("    note        : some artworks are undescribed — run `python -m cts describe`")
if rows["embeddings"] < rows["props"]:
    print("    note        : some propositions are unembedded — run `python -m cts embed`")
PY

# =========================================================================
step "Done"
# =========================================================================

printf '\n%sScrying Pool is ready.%s\n\n' "$C_OK" "$C_OFF"
printf '  source "%s/bin/activate"\n' "$VENV_DIR"
printf '  python -m cts search "commanders that look lonely" --band 3\n\n'
printf 'The corpus is already built — no ingest, describe, or embed pass is needed.\n'
printf 'Run `python -m cts refresh` when you want to pull in newly released cards.\n'
