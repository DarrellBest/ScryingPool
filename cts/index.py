"""Phase 6, read side: the whole search index, held in memory.

One float32 matrix over every embedded proposition, L2-normalized so cosine is a
dot product, plus one BM25 index over the *same rows in the same order*. Layer is
a mask applied at query time, never a second matrix: that is what lets Phase 7
weight literal and interpretive evidence continuously instead of branching.

At ~25 propositions per artwork over ~4,500 artworks this is ~110k vectors, which
brute-force scans in about a millisecond. There is no ANN index and there should
not be one.
"""

from __future__ import annotations

import re
import sqlite3
import sys
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Config

# The one tokenizer. Used to build the BM25 corpus here and to tokenize every
# query string in search.py. If these ever diverge, BM25 silently stops working.
_WORD = re.compile(r"\w+")

LAYERS = ("literal", "interpretive")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer, shared by index build and query time."""
    return _WORD.findall(text.lower())


@dataclass
class SearchIndex:
    """Parallel arrays: row i of `vecs` is prop_ids[i] / illustration_ids[i] / ..."""

    vecs: np.ndarray                      # (n, dim) float32, rows L2-normalized
    prop_ids: list[int]
    illustration_ids: list[str]
    layers: list[str]
    texts: list[str]
    bm25: BM25Okapi | None                # None only when the corpus is empty
    dim: int
    build_seconds: float                  # pinned to eval results, see Phase 11
    missing_embeddings: int               # props with no vector yet (embed lagging)
    layer_index: dict[str, np.ndarray]    # layer -> row indices, for masking
    by_illustration: dict[str, list[int]]  # illustration_id -> row indices

    def __len__(self) -> int:
        return len(self.prop_ids)

    @property
    def artwork_count(self) -> int:
        return len(self.by_illustration)

    def mean_vector(self, illustration_id: str) -> np.ndarray:
        """Mean of an artwork's proposition vectors, renormalized. Used by MMR."""
        rows = self.by_illustration.get(illustration_id)
        if not rows:
            return np.zeros(self.dim, dtype=np.float32)
        vec = self.vecs[rows].mean(axis=0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


_JOINED = (
    "FROM props p JOIN embeddings e ON e.prop_id = p.id "
    "WHERE p.text IS NOT NULL AND p.text != ''"
)


def load_index(cfg: Config, conn: sqlite3.Connection) -> SearchIndex:
    """Load every embedded proposition into one matrix plus one BM25 index."""
    start = time.perf_counter()

    total = conn.execute(f"SELECT COUNT(*) {_JOINED}").fetchone()[0]
    # The embed stage runs behind the vision stage; say so rather than pretending
    # the index is complete.
    missing = conn.execute(
        "SELECT COUNT(*) FROM props p LEFT JOIN embeddings e ON e.prop_id = p.id "
        "WHERE e.prop_id IS NULL"
    ).fetchone()[0]

    prop_ids: list[int] = []
    illustration_ids: list[str] = []
    layers: list[str] = []
    texts: list[str] = []
    dim = 0
    vecs = np.zeros((0, 0), dtype=np.float32)
    bad_blobs = 0

    if total:
        first = conn.execute(f"SELECT e.vec {_JOINED} ORDER BY p.id LIMIT 1").fetchone()
        dim = len(first["vec"]) // 4
        vecs = np.zeros((total, dim), dtype=np.float32)
        rows = conn.execute(
            f"SELECT p.id, p.illustration_id, p.layer, p.text, e.vec {_JOINED} ORDER BY p.id"
        )
        kept = 0
        for row in rows:
            blob = row["vec"]
            if blob is None or len(blob) != dim * 4:
                bad_blobs += 1  # dimension changed under us, or a truncated write
                continue
            vecs[kept] = np.frombuffer(blob, dtype=np.float32)
            prop_ids.append(int(row["id"]))
            illustration_ids.append(row["illustration_id"])
            layers.append(row["layer"])
            texts.append(row["text"])
            kept += 1
        vecs = vecs[:kept]

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # a zero vector stays zero instead of becoming NaN
        vecs /= norms

    # Row groupings computed once: layer masks for weighted retrieval, artwork
    # groups for the MMR diversity pass.
    layer_index: dict[str, np.ndarray] = {}
    for layer in LAYERS:
        layer_index[layer] = np.array(
            [i for i, lay in enumerate(layers) if lay == layer], dtype=np.int64
        )
    for layer in set(layers) - set(LAYERS):
        layer_index[layer] = np.array(
            [i for i, lay in enumerate(layers) if lay == layer], dtype=np.int64
        )

    by_illustration: dict[str, list[int]] = {}
    for i, ill in enumerate(illustration_ids):
        by_illustration.setdefault(ill, []).append(i)

    # BM25Okapi divides by corpus size and by the vocabulary size, so an empty
    # corpus (or one of only empty documents) raises. Skip it instead.
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized) if any(tokenized) else None

    build_seconds = time.perf_counter() - start

    counts = " / ".join(
        f"{len(layer_index.get(layer, ())):,} {layer}" for layer in LAYERS
    )
    print(
        f"index: {len(prop_ids):,} props ({counts}) over {len(by_illustration):,} "
        f"artworks, dim {dim}, {build_seconds:.2f}s",
        file=sys.stderr,
    )
    if missing:
        print(
            f"warning: {missing:,} props have no embedding yet — run 'python -m cts embed'",
            file=sys.stderr,
        )
    if bad_blobs:
        print(
            f"warning: skipped {bad_blobs:,} embeddings whose blob is not {dim} float32 "
            "values (embed model changed? clear the embeddings table and re-run)",
            file=sys.stderr,
        )
    if bm25 is None and total:
        print("warning: BM25 index empty — proposition texts are blank", file=sys.stderr)

    return SearchIndex(
        vecs=vecs,
        prop_ids=prop_ids,
        illustration_ids=illustration_ids,
        layers=layers,
        texts=texts,
        bm25=bm25,
        dim=dim,
        build_seconds=build_seconds,
        missing_embeddings=int(missing),
        layer_index=layer_index,
        by_illustration=by_illustration,
    )
