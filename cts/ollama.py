"""HTTP client for Ollama. This module is the entire model layer.

Three call functions plus a preflight check. No provider abstraction, no
retries, no client object: callers own their retry policy, and model names
always come from config.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import requests

from .config import Config

USER_AGENT = "ScryingPool/0.1 (github.com/DarrellBest/ScryingPool)"

_MAX_ERROR_CHARS = 2000


def _url(cfg: Config, endpoint: str) -> str:
    return f"{cfg.ollama_url.rstrip('/')}{endpoint}"


def _fail(endpoint: str, resp: requests.Response) -> RuntimeError:
    body = resp.text or "<empty body>"
    if len(body) > _MAX_ERROR_CHARS:
        body = body[:_MAX_ERROR_CHARS] + "... (truncated)"
    return RuntimeError(f"ollama {endpoint} returned HTTP {resp.status_code}: {body}")


def _post(cfg: Config, endpoint: str, body: dict, timeout: int) -> dict:
    resp = requests.post(
        _url(cfg, endpoint),
        json=body,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    if resp.status_code != 200:
        raise _fail(endpoint, resp)
    return resp.json()


def generate(
    cfg: Config,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    format: dict | str | None = None,
    options: dict | None = None,
    timeout: int = 300,
) -> str:
    """Single-shot completion. `format` is a JSON schema dict or the string "json".

    Sends think=False: hybrid-thinking models (every vision/text model pulled
    for this project so far) otherwise put the whole structured answer in the
    `thinking` field and leave `response` empty.
    """
    body: dict = {"model": model, "prompt": prompt, "stream": False, "think": False}
    if system is not None:
        body["system"] = system
    if format is not None:
        body["format"] = format
    if options is not None:
        body["options"] = options
    return _post(cfg, "/api/generate", body, timeout).get("response", "")


def vision(
    cfg: Config,
    model: str,
    prompt: str,
    image_path: str,
    *,
    format: dict | str | None = None,
    options: dict | None = None,
    timeout: int = 600,
) -> str:
    """Same as generate(), with one base64-encoded image attached. See generate()
    for why think=False is sent."""
    try:
        raw = Path(image_path).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read image {image_path}: {exc}") from exc

    body: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "images": [base64.b64encode(raw).decode("ascii")],
    }
    if format is not None:
        body["format"] = format
    if options is not None:
        body["options"] = options
    return _post(cfg, "/api/generate", body, timeout).get("response", "")


def embed(
    cfg: Config,
    texts: list[str],
    *,
    model: str | None = None,
    timeout: int = 300,
) -> np.ndarray:
    """Embed a batch of strings. Returns a (len(texts), dim) float32 array."""
    items = list(texts)
    if not items:
        raise ValueError("embed() requires at least one text")

    name = model or cfg.embed_model
    if not name:
        raise RuntimeError("no embedding model configured (set embed_model in config.toml)")

    data = _post(cfg, "/api/embed", {"model": name, "input": items}, timeout)
    vectors = data.get("embeddings")
    if not vectors:
        raise RuntimeError(
            f"ollama /api/embed returned no embeddings for model {name!r} "
            f"(is it an embedding model?); response keys: {sorted(data)}"
        )

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(items):
        raise RuntimeError(
            f"ollama /api/embed returned shape {arr.shape} for {len(items)} inputs"
        )
    return arr


def _normalize(name: str) -> str:
    """Drop an explicit ":latest" so "llama3" and "llama3:latest" compare equal."""
    return name[: -len(":latest")] if name.endswith(":latest") else name


def preflight(cfg: Config) -> list[str]:
    """Return the configured model names that Ollama does not have pulled.

    Empty list means everything is present. Raises requests.ConnectionError
    (uncaught, on purpose) when Ollama is not running.
    """
    resp = requests.get(_url(cfg, "/api/tags"), timeout=30, headers={"User-Agent": USER_AGENT})
    if resp.status_code != 200:
        raise _fail("/api/tags", resp)

    present = set()
    for entry in resp.json().get("models", []):
        for key in ("name", "model"):
            value = entry.get(key)
            if value:
                present.add(_normalize(value))

    missing: list[str] = []
    configured = [
        ("vision_model", cfg.vision_model),
        ("embed_model", cfg.embed_model),
        ("judge_model", cfg.judge_model),
    ]
    # verify_model defaults to vision_model, and the overwhelmingly common case is
    # that they are the same string. Only a genuinely distinct value earns its own
    # check — otherwise an unset vision_model would be reported twice, once under
    # each name, which reads like two separate problems.
    if _normalize(cfg.verify_model) != _normalize(cfg.vision_model):
        configured.append(("verify_model", cfg.verify_model))
    for key, name in configured:
        label = name if name.strip() else f"<{key} not set in config.toml>"
        if (not name.strip() or _normalize(name) not in present) and label not in missing:
            missing.append(label)
    return missing
