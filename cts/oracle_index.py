"""Stage 8, read side: the whole oracle chunk index, held in memory.

The exact analogue of `cts/index.py`, one layer down:

    art side:     props  ->  artwork  ->  card       (SearchIndex)
    oracle side:  chunks ->  card                     (OracleIndex)

One float32 matrix over every embedded chunk, L2-normalized so cosine is a dot
product, plus one BM25 index over the *same rows in the same order*. There is
no `layer` concept here — oracle text has one register, Wizards' templating,
so there is nothing to mask or blend at retrieval time the way literal and
interpretive art descriptions are.

BM25 is built over `text_embedded`, the same string that was embedded, not the
verbatim `text` column: that keeps "the thing retrieval scored" a single
source of truth. The only difference between the two columns is the card's own
name, which structured filters already own (`types`, `colors`) — retrieval was
never supposed to be finding cards by their own name in the first place.

At ~95,000 chunks this is a fraction of the art side's 170,487-row matrix and
brute-force scans in well under a millisecond. No ANN index, for the same
reason `cts/index.py` has none.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Config
from .index import tokenize  # the one tokenizer, shared everywhere in the repo


@dataclass
class OracleIndex:
    """Parallel arrays: row i of `vecs` is chunk_ids[i] / oracle_ids[i] / ..."""

    vecs: np.ndarray                      # (n, dim) float32, rows L2-normalized
    chunk_ids: list[int]
    oracle_ids: list[str]
    kinds: list[str]                      # "ability" | "whole"
    face_indices: list[int]
    ordinals: list[int]
    texts: list[str]                      # verbatim, for display/citation
    bm25: BM25Okapi | None                # None only when the corpus is empty
    dim: int
    build_seconds: float
    missing_embeddings: int               # chunks with no vector yet
    by_oracle_id: dict[str, list[int]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @property
    def card_count(self) -> int:
        return len(self.by_oracle_id)


_JOINED = (
    "FROM chunks c JOIN chunk_embeddings e ON e.chunk_id = c.id "
    "WHERE c.text_embedded IS NOT NULL AND c.text_embedded != ''"
)


def load_index(cfg: Config, conn: sqlite3.Connection) -> OracleIndex:
    """Load every embedded chunk into one matrix plus one BM25 index."""
    start = time.perf_counter()

    total = conn.execute(f"SELECT COUNT(*) {_JOINED}").fetchone()[0]
    missing = conn.execute(
        "SELECT COUNT(*) FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id "
        "WHERE e.chunk_id IS NULL"
    ).fetchone()[0]

    chunk_ids: list[int] = []
    oracle_ids: list[str] = []
    kinds: list[str] = []
    face_indices: list[int] = []
    ordinals: list[int] = []
    texts: list[str] = []
    embedded_texts: list[str] = []
    dim = 0
    vecs = np.zeros((0, 0), dtype=np.float32)
    bad_blobs = 0

    if total:
        first = conn.execute(f"SELECT e.vec {_JOINED} ORDER BY c.id LIMIT 1").fetchone()
        dim = len(first["vec"]) // 4
        vecs = np.zeros((total, dim), dtype=np.float32)
        rows = conn.execute(
            f"SELECT c.id, c.oracle_id, c.kind, c.face_index, c.ordinal, "
            f"c.text, c.text_embedded, e.vec {_JOINED} ORDER BY c.id"
        )
        kept = 0
        for row in rows:
            blob = row["vec"]
            if blob is None or len(blob) != dim * 4:
                bad_blobs += 1
                continue
            vecs[kept] = np.frombuffer(blob, dtype=np.float32)
            chunk_ids.append(int(row["id"]))
            oracle_ids.append(row["oracle_id"])
            kinds.append(row["kind"])
            face_indices.append(int(row["face_index"] or 0))
            ordinals.append(int(row["ordinal"] if row["ordinal"] is not None else -1))
            texts.append(row["text"] or "")
            embedded_texts.append(row["text_embedded"] or "")
            kept += 1
        vecs = vecs[:kept]

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms

    by_oracle_id: dict[str, list[int]] = {}
    for i, oid in enumerate(oracle_ids):
        by_oracle_id.setdefault(oid, []).append(i)

    tokenized = [tokenize(t) for t in embedded_texts]
    bm25 = BM25Okapi(tokenized) if any(tokenized) else None

    build_seconds = time.perf_counter() - start

    print(
        f"oracle-index: {len(chunk_ids):,} chunks over {len(by_oracle_id):,} cards, "
        f"dim {dim}, {build_seconds:.2f}s",
        file=sys.stderr,
    )
    if missing:
        print(
            f"warning: {missing:,} chunks have no embedding yet — run "
            "'python -m cts oracle-embed'",
            file=sys.stderr,
        )
    if bad_blobs:
        print(
            f"warning: skipped {bad_blobs:,} embeddings whose blob is not {dim} float32 "
            "values (embed model changed? clear chunk_embeddings and re-run)",
            file=sys.stderr,
        )
    if bm25 is None and total:
        print("warning: BM25 index empty — chunk texts are blank", file=sys.stderr)

    return OracleIndex(
        vecs=vecs,
        chunk_ids=chunk_ids,
        oracle_ids=oracle_ids,
        kinds=kinds,
        face_indices=face_indices,
        ordinals=ordinals,
        texts=texts,
        bm25=bm25,
        dim=dim,
        build_seconds=build_seconds,
        missing_embeddings=int(missing),
        by_oracle_id=by_oracle_id,
    )
