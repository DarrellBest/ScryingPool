"""verify_model: the search-time vision model, separable from the Phase 5 one.

`vision_model` is the model that wrote the corpus; `verify_model` is the model the
search path puts in front of an art crop. They default to the same name so every
config written before the key existed keeps working, but they can be set apart so
the judge and the verifier fit in VRAM together. These tests pin the defaulting,
the preflight behaviour in both directions, and the fact that verify_finalists
actually calls the one it should. No Ollama, no network.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from cts import judge, ollama
from cts.config import Config, load_config

BASE = """\
ollama_url = "http://localhost:11434"
vision_model = "big-vision:122b"
embed_model = "nomic-embed-text"
judge_model = "small-judge:30b"
"""


def _write(tmp_path: Path, body: str) -> str:
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(body))
    return str(path)


# ------------------------------------------------------------------ defaulting


def test_verify_model_defaults_to_vision_model_when_the_key_is_absent(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg.verify_model == "big-vision:122b"
    assert cfg.vision_model == "big-vision:122b"


def test_a_blank_verify_model_also_falls_back_rather_than_becoming_empty(tmp_path):
    cfg = load_config(_write(tmp_path, BASE + 'verify_model = "   "\n'))
    assert cfg.verify_model == "big-vision:122b"


def test_an_explicit_verify_model_wins_and_leaves_vision_model_alone(tmp_path):
    cfg = load_config(_write(tmp_path, BASE + 'verify_model = "small-judge:30b"\n'))
    assert cfg.verify_model == "small-judge:30b"
    assert cfg.vision_model == "big-vision:122b"


def test_verify_model_falls_back_to_empty_when_vision_model_is_also_unset(tmp_path):
    cfg = load_config(_write(tmp_path, 'judge_model = "small-judge:30b"\n'))
    assert cfg.verify_model == ""


# ------------------------------------------------------------------- preflight


def _cfg(**over) -> Config:
    base = dict(
        ollama_url="http://localhost:11434",
        vision_model="big-vision:122b",
        verify_model="big-vision:122b",
        embed_model="nomic-embed-text",
        judge_model="small-judge:30b",
        db_path=":memory:",
        art_dir="art",
        power_weights={},
    )
    base.update(over)
    return Config(**base)


class _Resp:
    status_code = 200

    def __init__(self, names):
        self._names = names

    def json(self):
        return {"models": [{"name": n, "model": n} for n in self._names]}


@pytest.fixture
def tags(monkeypatch):
    """Stub /api/tags with whatever set of pulled models a test wants."""

    def install(*names):
        monkeypatch.setattr(ollama.requests, "get", lambda *a, **k: _Resp(list(names)))

    return install


def test_preflight_passes_when_verify_model_is_pulled(tags):
    tags("big-vision:122b", "nomic-embed-text", "small-judge:30b")
    assert ollama.preflight(_cfg(verify_model="small-judge:30b")) == []


def test_preflight_reports_a_distinct_verify_model_that_is_not_pulled(tags):
    tags("big-vision:122b", "nomic-embed-text", "small-judge:30b")
    assert ollama.preflight(_cfg(verify_model="never-pulled:8b")) == ["never-pulled:8b"]


def test_preflight_does_not_report_verify_model_twice_when_it_equals_vision_model(tags):
    """The default case: one name, one complaint, not the same name under two keys."""
    tags("nomic-embed-text", "small-judge:30b")
    assert ollama.preflight(_cfg()) == ["big-vision:122b"]


def test_an_unset_vision_model_is_still_reported_only_once(tags):
    """verify_model defaults to "" alongside it; that must not double the message."""
    tags("nomic-embed-text", "small-judge:30b")
    missing = ollama.preflight(_cfg(vision_model="", verify_model=""))
    assert missing == ["<vision_model not set in config.toml>"]


def test_the_latest_suffix_does_not_make_the_two_names_look_distinct(tags):
    """judge/verify pointed at the same model, written with and without ":latest"."""
    tags("big-vision:122b", "nomic-embed-text", "small-judge:30b")
    cfg = _cfg(vision_model="big-vision:122b", verify_model="big-vision:122b:latest")
    assert "big-vision:122b:latest" not in ollama.preflight(cfg)


# -------------------------------------------------------------- call site wiring


def test_verify_finalists_calls_verify_model_not_vision_model(monkeypatch, tmp_path):
    """The whole point of the key: this stage must not reach for the Phase 5 model."""
    crop = tmp_path / "art.jpg"
    crop.write_bytes(b"\xff\xd8\xff")
    used: list[str] = []

    def fake_vision(cfg, model, prompt, image_path, **kw):
        used.append(model)
        return '{"holds": true, "why": "a lone figure"}'

    monkeypatch.setattr(judge.ollama, "vision", fake_vision)
    cfg = _cfg(verify_model="small-judge:30b")
    judged = [{"illustration_id": "i1", "fit": 0.9, "art_path": str(crop)}]

    results, available = judge.verify_finalists(cfg, judged, "commanders that look lonely")

    assert used == ["small-judge:30b"]
    assert available is True
    assert results[0]["verified"] is True
    assert results[0]["verify_note"] == "a lone figure"


def test_verify_finalists_still_works_when_the_two_models_are_the_same(monkeypatch, tmp_path):
    """Default configs keep the old behaviour exactly: the vision model gets called."""
    crop = tmp_path / "art.jpg"
    crop.write_bytes(b"\xff\xd8\xff")
    used: list[str] = []

    monkeypatch.setattr(
        judge.ollama,
        "vision",
        lambda cfg, model, *a, **k: (used.append(model), '{"holds": false, "why": "no"}')[1],
    )
    judged = [{"illustration_id": "i1", "fit": 0.9, "art_path": str(crop)}]

    results, available = judge.verify_finalists(_cfg(), judged, "theme")

    assert used == ["big-vision:122b"]
    assert results[0]["vision_rejected"] is True
