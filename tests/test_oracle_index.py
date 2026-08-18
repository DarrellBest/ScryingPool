"""The oracle chunk index: matrix + BM25 + by_oracle_id groupings.

No Ollama and no network: vectors are written directly as float32 blobs, the
same shape `oracle_embed.run` would have produced.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rank_bm25")

from cts import oracle_db, oracle_index                            # noqa: E402
from cts.config import Config                                      # noqa: E402


def _cfg(tmp_path) -> Config:
    return Config(
        ollama_url="u", vision_model="v", verify_model="v", embed_model="e",
        judge_model="j", db_path=str(tmp_path / "commanders.db"),
        art_dir=str(tmp_path / "art"), power_weights={},
        oracle_db_path=str(tmp_path / "oracle.db"),
    )


def _vec(seed: int, dim: int = 4) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float32).astype(np.float32).tobytes()


def _seed(conn, dim: int = 4) -> None:
    conn.execute(
        "INSERT INTO cards(oracle_id, name, name_norm, type_line, oracle_text) "
        "VALUES ('o-sol', 'Sol Ring', 'sol ring', 'Artifact', '{T}: Add {C}{C}.')"
    )
    conn.execute(
        "INSERT INTO cards(oracle_id, name, name_norm, type_line, oracle_text) "
        "VALUES ('o-bear', 'Grizzly Bears', 'grizzly bears', 'Creature — Bear', '')"
    )
    rows = [
        ("o-sol", 0, 0, "ability", "{T}: Add {C}{C}.", "this card: Add {C}{C}."),
        ("o-sol", 0, -1, "whole", "Artifact\n{T}: Add {C}{C}.", "Artifact\nthis card: Add {C}{C}."),
        ("o-bear", 0, -1, "whole", "Creature — Bear", "Creature — Bear"),
    ]
    for oracle_id, face_index, ordinal, kind, text, embedded in rows:
        cur = conn.execute(
            "INSERT INTO chunks(oracle_id, face_index, ordinal, kind, text, text_embedded) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (oracle_id, face_index, ordinal, kind, text, embedded),
        )
        conn.execute(
            "INSERT INTO chunk_embeddings(chunk_id, vec) VALUES (?, ?)",
            (cur.lastrowid, _vec(cur.lastrowid, dim)),
        )
    conn.commit()


def test_load_index_stacks_every_embedded_chunk(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        _seed(conn)
        index = oracle_index.load_index(_cfg(tmp_path), conn)
        assert len(index) == 3
        assert index.dim == 4
        assert index.vecs.shape == (3, 4)
        assert index.missing_embeddings == 0
        assert index.card_count == 2
        assert sorted(index.by_oracle_id) == ["o-bear", "o-sol"]
        assert index.by_oracle_id["o-sol"] == [0, 1]
    finally:
        conn.close()


def test_vectors_are_l2_normalized(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        _seed(conn)
        index = oracle_index.load_index(_cfg(tmp_path), conn)
        norms = np.linalg.norm(index.vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    finally:
        conn.close()


def test_bm25_is_built_over_text_embedded_not_the_verbatim_text(tmp_path):
    """The corpus is padded with filler rows so a term appearing in one row out
    of many has a sane (positive) BM25 IDF — with only 2-3 rows total, a term
    shared by most of them gets a negative IDF, which is BM25's own documented
    behaviour for tiny corpora and not something this test should be fighting.
    """
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        _seed(conn)
        for i in range(10):
            conn.execute(
                "INSERT INTO chunks(oracle_id, face_index, ordinal, kind, text, text_embedded) "
                "VALUES ('o-bear', 0, ?, 'ability', 'filler filler', 'filler filler')",
                (i + 1,),
            )
        conn.commit()
        for row in conn.execute("SELECT id FROM chunks WHERE text = 'filler filler'"):
            conn.execute(
                "INSERT INTO chunk_embeddings(chunk_id, vec) VALUES (?, ?)",
                (row[0], _vec(row[0])),
            )
        conn.commit()

        index = oracle_index.load_index(_cfg(tmp_path), conn)
        # Row 0's text_embedded is "this card: Add {C}{C}."; its verbatim text
        # is "{T}: Add {C}{C}." — "card" is present only in the embedded form.
        scores = index.bm25.get_scores(oracle_index.tokenize("card"))
        assert scores[0] == max(scores)
    finally:
        conn.close()


def test_missing_embeddings_are_reported(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        _seed(conn)
        conn.execute(
            "INSERT INTO chunks(oracle_id, face_index, ordinal, kind, text, text_embedded) "
            "VALUES ('o-bear', 0, 0, 'ability', 'Flying', 'Flying')"
        )
        conn.commit()
        index = oracle_index.load_index(_cfg(tmp_path), conn)
        assert index.missing_embeddings == 1
        assert len(index) == 3  # the unembedded chunk is absent from the matrix
    finally:
        conn.close()


def test_an_empty_corpus_loads_a_zero_row_index(tmp_path):
    conn = oracle_db.connect(_cfg(tmp_path))
    try:
        index = oracle_index.load_index(_cfg(tmp_path), conn)
        assert len(index) == 0
        assert index.bm25 is None
        assert index.card_count == 0
    finally:
        conn.close()
