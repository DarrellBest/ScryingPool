"""The `/oracle` pipeline. No Ollama, no network: every model call is
monkeypatched at `cts.ollama.generate` / `cts.ollama.embed`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("rank_bm25")

from cts import oracle_chunk, oracle_db, oracle_index, oracle_names, oracle_search  # noqa: E402
from cts.config import Config                                                        # noqa: E402
from cts.oracle_filters import Filters                                               # noqa: E402


def _cfg(tmp_path) -> Config:
    return Config(
        ollama_url="u", vision_model="v", verify_model="v", embed_model="e",
        judge_model="j", db_path=str(tmp_path / "commanders.db"),
        art_dir=str(tmp_path / "art"), power_weights={},
        oracle_db_path=str(tmp_path / "oracle.db"),
    )


# ------------------------------------------------------------------------- guards


def test_rules_questions_are_detected():
    for q in ["Can I respond to a triggered ability", "does this trigger twice",
              "How does first strike work", "what happens if I sacrifice it",
              "When do I lose the game"]:
        assert oracle_search.looks_like_rules_question(q), q


def test_mechanical_queries_are_not_flagged_as_rules_questions():
    for q in ["cards that draw", "green enchantments under 5 mana",
              "counter target spell", "cheap removal"]:
        assert not oracle_search.looks_like_rules_question(q), q


def test_card_name_guard_fires_on_an_exact_name_but_not_a_near_miss(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm) VALUES "
            "('o-cs', 'Counterspell', 'counterspell')"
        )
        conn.commit()
        index = oracle_names.build_index(conn)
        hit = oracle_search.card_name_guard(index, "Counterspell")
        assert hit is not None and hit.oracle_id == "o-cs"
        # "counter target spell" is two edits from "Counterspell" but must not
        # fire the guard: only L0-L2 (exact-ish) run here.
        miss = oracle_search.card_name_guard(index, "counter target spell")
        assert miss is None
    finally:
        conn.close()


def test_card_name_guard_is_none_without_a_name_index():
    assert oracle_search.card_name_guard(None, "Sol Ring") is None


# --------------------------------------------------------------------------- route


def test_route_parses_a_well_formed_router_response(tmp_path, monkeypatch):
    payload = {
        "types": ["Enchantment"], "colors": "g", "legal": [],
        "mv_op": "<=", "mv_value": 5, "mv_lo": 0, "mv_hi": 0,
        "semantic_intent": "let me draw", "vague_quantity_note": "", "reasoning": "x",
    }
    monkeypatch.setattr(oracle_search.ollama, "generate", lambda *a, **k: json.dumps(payload))
    out = oracle_search.route(_cfg(tmp_path), "enchantments in green that draw, 5 or less")
    assert out["router_ok"] is True
    assert out["filters"].types == ("enchantment",)
    assert out["filters"].colors == "G"
    assert out["filters"].mv_op == "<="
    assert out["filters"].mv_value == 5
    assert out["semantic_intent"] == "let me draw"


def test_route_carries_a_vague_quantity_note(tmp_path, monkeypatch):
    payload = {
        "types": [], "colors": "", "legal": [], "mv_op": "", "mv_value": 0, "mv_lo": 0, "mv_hi": 0,
        "semantic_intent": "removal", "vague_quantity_note":
            '"cheap" has no defined mana value, so no cost filter was applied.',
        "reasoning": "x",
    }
    monkeypatch.setattr(oracle_search.ollama, "generate", lambda *a, **k: json.dumps(payload))
    out = oracle_search.route(_cfg(tmp_path), "cheap removal")
    assert out["filters"].mv_op is None
    assert any("cheap" in n for n in out["notes"])


def test_router_schema_requires_the_numeric_fields_not_just_mv_op():
    """Regression: measured live against qwen3.6:latest, when mv_value was
    merely OPTIONAL in the schema, the model correctly chose mv_op="<=" for
    "cost 5 or less" and then silently omitted mv_value — a 7-mana card then
    passed a "5 or less" filter with no note anywhere explaining why. Requiring
    the numeric fields forces the model to write an actual number (0 when
    genuinely unused, per the prompt's own instruction) instead of a missing
    key that `_to_float` turns into a silently-ignored None."""
    required = set(oracle_search.ROUTER_SCHEMA["required"])
    assert {"mv_op", "mv_value", "mv_lo", "mv_hi"} <= required


def test_route_degrades_to_no_filters_when_ollama_is_unreachable(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(oracle_search.ollama, "generate", explode)
    out = oracle_search.route(_cfg(tmp_path), "green enchantments that draw")
    assert out["router_ok"] is False
    assert out["filters"] == Filters()
    assert out["semantic_intent"] == "green enchantments that draw"
    assert any("routing failed" in n for n in out["notes"])


# -------------------------------------------------------------------------- expand


def test_expand_always_includes_the_intent_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(
        oracle_search.ollama, "generate",
        lambda *a, **k: json.dumps({"expansions": ["draw a card", "draws two cards"]}),
    )
    out = oracle_search.expand(_cfg(tmp_path), "let me draw", [])
    assert "let me draw" in out
    assert "draw a card" in out


def test_expand_of_empty_intent_is_empty(tmp_path):
    assert oracle_search.expand(_cfg(tmp_path), "", []) == []


def test_expand_degrades_to_just_the_intent_on_failure(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(oracle_search.ollama, "generate", explode)
    notes: list[str] = []
    out = oracle_search.expand(_cfg(tmp_path), "let me draw", notes)
    assert out == ["let me draw"]
    assert notes


# ------------------------------------------------------------------------- retrieve


def _fake_index() -> oracle_index.OracleIndex:
    vecs = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return oracle_index.OracleIndex(
        vecs=vecs,
        chunk_ids=[1, 2, 3],
        oracle_ids=["o-draw", "o-scry", "o-draw2"],
        kinds=["ability", "ability", "whole"],
        face_indices=[0, 0, 0],
        ordinals=[0, 0, -1],
        texts=["draw a card", "scry 2", "Enchantment\ndraw a card"],
        bm25=None,
        dim=2,
        build_seconds=0.0,
        missing_embeddings=0,
        by_oracle_id={"o-draw": [0], "o-scry": [1], "o-draw2": [2]},
    )


def test_retrieve_masks_to_the_allowed_set():
    index = _fake_index()
    query_vecs = np.array([[1.0, 0.0]], dtype=np.float32)
    fused, best = oracle_search.retrieve(index, ["draw a card"], query_vecs, {"o-draw"})
    assert set(fused) == {"o-draw"}
    assert best["o-draw"] == 0


def test_retrieve_never_sums_across_a_cards_own_chunks():
    """Two chunks belong to the same card in one list; only the best-ranked
    one may contribute, never both summed."""
    index = _fake_index()
    # both chunk 0 (o-draw) and chunk 2 (o-draw2) score high on this vector,
    # but they belong to different cards, so this only tests within-list dedup
    # for a single card via a synthetic duplicate-oracle-id index.
    dup = oracle_index.OracleIndex(
        vecs=np.array([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32),
        chunk_ids=[10, 11], oracle_ids=["o-x", "o-x"], kinds=["ability", "whole"],
        face_indices=[0, 0], ordinals=[0, -1], texts=["draw a card", "Type\ndraw a card"],
        bm25=None, dim=2, build_seconds=0.0, missing_embeddings=0,
        by_oracle_id={"o-x": [0, 1]},
    )
    query_vecs = np.array([[1.0, 0.0]], dtype=np.float32)
    fused, best = oracle_search.retrieve(dup, ["draw a card"], query_vecs, None)
    # Only one contribution (1/(60+1)) landed for o-x, not two summed.
    assert fused["o-x"] == pytest.approx(1.0 / 61)


# ------------------------------------------------------------------------- pipeline


CARDS = (
    # oracle_id, name, type_line, oracle_text, cmc, color_identity
    ("o-draw", "Verdant Genesis", "Enchantment", "Whenever a creature enters, draw a card.", 3.0, "G"),
    ("o-scry", "Green Scry Thing", "Enchantment", "At the beginning of your turn, scry 2.", 2.0, "G"),
    ("o-loot", "Green Loot Box", "Enchantment", "{1}, T: Draw a card, then discard a card.", 2.0, "G"),
    ("o-red", "Red Fireball", "Sorcery", "Deal 5 damage to any target.", 3.0, "R"),
)


def _seed(conn):
    for oid, name, type_line, text, cmc, ci in CARDS:
        conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm, type_line, oracle_text, cmc, "
            "color_identity, edhrec_rank) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (oid, name, name.lower(), type_line, text, cmc, ci, hash(oid) % 1000),
        )
        for t in type_line.lower().split():
            conn.execute(
                "INSERT INTO card_types(oracle_id, kind, value) VALUES (?, 'type', ?)", (oid, t)
            )
    conn.commit()
    result = oracle_chunk.rechunk(conn)
    assert result["cards"] == len(CARDS)


def _fake_embed(cfg, texts, **kwargs):
    """A tiny deterministic embedding: vectors close for texts sharing words."""
    vocab = ["draw", "card", "scry", "discard", "damage", "green", "enchant"]
    out = []
    for t in texts:
        words = t.lower().split()
        vec = [sum(1.0 for w in words if v in w) for v in vocab]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        out.append([x / norm for x in vec])
    return np.array(out, dtype=np.float32)


@pytest.fixture
def conn(tmp_path):
    connection = oracle_db.connect(_cfg(tmp_path))
    _seed(connection)
    yield connection
    connection.close()


@pytest.fixture
def index(tmp_path, conn):
    # embed every chunk with the fake embedder so oracle_index has vectors.
    rows = conn.execute("SELECT id, text_embedded FROM chunks").fetchall()
    vecs = _fake_embed(None, [r["text_embedded"] for r in rows])
    conn.executemany(
        "INSERT INTO chunk_embeddings(chunk_id, vec) VALUES (?, ?)",
        [(r["id"], np.asarray(v, dtype=np.float32).tobytes()) for r, v in zip(rows, vecs)],
    )
    conn.commit()
    return oracle_index.load_index(_cfg(tmp_path), conn)


def _stub_router(payload):
    return lambda *a, **k: json.dumps(payload)


def _stub_expand(phrases):
    return lambda *a, **k: json.dumps({"expansions": phrases})


def _stub_judge(entries):
    return lambda *a, **k: json.dumps({"judgments": entries})


def test_hard_filter_matching_zero_returns_an_honest_message_with_no_judge_call(tmp_path, conn, index, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("must not call the router when hard filters are already zero")

    monkeypatch.setattr(oracle_search.ollama, "generate", explode)
    out = oracle_search.execute(
        _cfg(tmp_path), "anything", types=("enchantment",), colors="R", conn=conn, index=index,
    )
    assert out["results"] == []
    assert "0 cards" in out["message"]


def test_structural_only_fast_path_when_the_router_finds_no_mechanical_intent(tmp_path, conn, index, monkeypatch):
    payload = {
        "types": [], "colors": "G", "legal": [], "mv_op": "", "mv_value": 0, "mv_lo": 0, "mv_hi": 0,
        "semantic_intent": "", "vague_quantity_note": "", "reasoning": "purely structural",
    }
    monkeypatch.setattr(oracle_search.ollama, "generate", _stub_router(payload))
    out = oracle_search.execute(_cfg(tmp_path), "green enchantments", conn=conn, index=index)
    assert out["plan"]["structural_only"] is True
    assert "Nothing was judged" in out["message"]
    names = {r["name"] for r in out["results"]}
    assert names <= {"Verdant Genesis", "Green Scry Thing", "Green Loot Box"}
    assert all(r["fit"] is None for r in out["results"])


def test_the_full_semantic_pipeline_end_to_end(tmp_path, conn, index, monkeypatch):
    router_payload = {
        "types": ["enchantment"], "colors": "G", "legal": [], "mv_op": "<=", "mv_value": 5,
        "mv_lo": 0, "mv_hi": 0, "semantic_intent": "let me draw", "vague_quantity_note": "",
        "reasoning": "x",
    }
    calls = {"n": 0}

    def fake_generate(cfg, model, prompt, **kwargs):
        calls["n"] += 1
        fmt = kwargs.get("format")
        if fmt is oracle_search.ROUTER_SCHEMA:
            return json.dumps(router_payload)
        if fmt is oracle_search.EXPANSION_SCHEMA:
            return json.dumps({"expansions": ["draw a card", "draws a card"]})
        # judge call
        import re

        numbers = sorted(int(n) for n in re.findall(r"CANDIDATE (\d+)", prompt))
        judgments = []
        for n in numbers:
            fit = 0.9 if f"CANDIDATE {n} — Verdant Genesis" in prompt else 0.1
            judgments.append({"candidate": n, "fit": fit, "rationale": "r", "chunk_ids": []})
        return json.dumps({"judgments": judgments})

    monkeypatch.setattr(oracle_search.ollama, "generate", fake_generate)
    monkeypatch.setattr(oracle_search.ollama, "embed", _fake_embed)

    out = oracle_search.execute(_cfg(tmp_path), "enchantments in green that draw, 5 or less",
                                 conn=conn, index=index)
    assert out["plan"]["echo"].startswith("filters: type = enchantment")
    assert "colors ⊆ {G}" in out["plan"]["echo"]
    assert "mv ≤ 5" in out["plan"]["echo"]
    assert out["results"]
    assert out["results"][0]["name"] == "Verdant Genesis"
    assert out["results"][0]["fit"] == 0.9
    # every result must be a green enchantment mv<=5, i.e. never the red card
    assert all(r["name"] != "Red Fireball" for r in out["results"])
