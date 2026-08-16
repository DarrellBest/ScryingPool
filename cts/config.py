"""Load config.toml into a frozen Config dataclass, once, at startup."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Applied when the [power_weights] table is absent, and per-key when it is
# present but partial. See SPEC.md Phase 3.
DEFAULT_POWER_WEIGHTS: dict[str, float] = {
    "deck_count": 0.4,
    "price": 0.25,
    "cmc": 0.2,
    "cedh": 0.15,
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_DB_PATH = "data/commanders.db"
DEFAULT_ART_DIR = "data/art"


@dataclass(frozen=True)
class Config:
    ollama_url: str
    vision_model: str
    verify_model: str
    embed_model: str
    judge_model: str
    db_path: str
    art_dir: str
    power_weights: dict


def _weights(raw: dict) -> dict[str, float]:
    weights = dict(DEFAULT_POWER_WEIGHTS)
    table = raw.get("power_weights")
    if isinstance(table, dict):
        for key in DEFAULT_POWER_WEIGHTS:
            if key in table:
                weights[key] = float(table[key])
    return weights


def load_config(path: str = "config.toml") -> Config:
    """Read `path` and return a Config. Unknown keys are ignored.

    Missing model names default to "" — cts.ollama.preflight reports them
    rather than failing here, so read-only commands still work.

    `verify_model` is the one exception: when it is absent or blank it falls back
    to `vision_model`, which is what every config written before the key existed
    gets. See config.toml for why the two are worth separating.
    """
    p = Path(path)
    if not p.is_file():
        print(f"error: no config file at {p}", file=sys.stderr)
        print(
            "Copy config.toml from the repo root, fill in your Ollama URL and "
            "model names, then re-run (or pass --config PATH).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    with p.open("rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            print(f"error: {p} is not valid TOML: {exc}", file=sys.stderr)
            print("Fix the syntax, or copy a fresh config.toml and re-edit it.", file=sys.stderr)
            raise SystemExit(1) from None

    vision_model = str(raw.get("vision_model", ""))
    return Config(
        ollama_url=str(raw.get("ollama_url", DEFAULT_OLLAMA_URL)).rstrip("/"),
        vision_model=vision_model,
        verify_model=str(raw.get("verify_model", "")).strip() or vision_model,
        embed_model=str(raw.get("embed_model", "")),
        judge_model=str(raw.get("judge_model", "")),
        db_path=str(raw.get("db_path", DEFAULT_DB_PATH)),
        art_dir=str(raw.get("art_dir", DEFAULT_ART_DIR)),
        power_weights=_weights(raw),
    )
