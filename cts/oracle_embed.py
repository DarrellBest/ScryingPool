"""Stage 8, write side: embed every chunk exactly once.

A near-copy of `cts/embed.py`, deliberately not a generalisation of it: the two
differ in table names, id column, and the `layer` concept that only the art
side has, and parameterising three things to save forty lines would make the
art pipeline's embed stage harder to read for no gain — see the design doc's
"New modules" section.

Chunks are embedded on `text_embedded` (the name-substituted column), never on
`text` (the verbatim display column) — see `cts/oracle_chunk.py`'s module
docstring for why that substitution exists. Selection is "chunks with no
`chunk_embeddings` row", so the pass is idempotent and resumable: interrupt it,
re-run it, it picks up the remainder. `cts/oracle_chunk.py::rechunk` deletes a
re-chunked card's embeddings along with its chunks, so a card whose text
changed is automatically re-embedded here.
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np

from . import oracle_db, ollama
from .config import Config

BATCH_SIZE = 32
PROGRESS_EVERY_BATCHES = 10  # ~320 chunks per progress line


def _prune_orphans(conn: sqlite3.Connection) -> int:
    """Drop vectors whose chunk no longer exists.

    `rechunk` already deletes them alongside the chunks it rewrites; this is a
    cheap safety net so a stray vector can never be stacked into the matrix and
    mapped to the wrong card.
    """
    cur = conn.execute(
        "DELETE FROM chunk_embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    )
    conn.commit()
    return cur.rowcount or 0


def _stored_dim(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT length(vec) AS n FROM chunk_embeddings LIMIT 1").fetchone()
    return None if row is None or not row["n"] else int(row["n"]) // 4


def run(cfg: Config) -> dict:
    if not cfg.embed_model.strip():
        print("error: no embed_model set in config.toml", file=sys.stderr)
        raise SystemExit(1)

    conn = oracle_db.connect(cfg)
    embedded = 0
    try:
        orphans = _prune_orphans(conn)
        if orphans:
            print(f"oracle-embed: pruned {orphans} vector(s) whose chunk is gone", flush=True)

        rows = conn.execute(
            """
            SELECT c.id, c.text_embedded
              FROM chunks c
              LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
             WHERE e.chunk_id IS NULL
               AND c.text_embedded IS NOT NULL AND c.text_embedded != ''
             ORDER BY c.id
            """
        ).fetchall()
        if not rows:
            print("oracle-embed: nothing to do", flush=True)
            return {"embedded": 0}

        existing_dim = _stored_dim(conn)
        print(
            f"oracle-embed: {len(rows)} chunks with {cfg.embed_model}, "
            f"batches of {BATCH_SIZE}",
            flush=True,
        )

        for batch_no, start in enumerate(range(0, len(rows), BATCH_SIZE), start=1):
            batch = rows[start : start + BATCH_SIZE]
            try:
                vecs = ollama.embed(cfg, [row["text_embedded"] for row in batch])
            except Exception as exc:
                print(
                    f"oracle-embed: stopped after {embedded} embeddings: {exc}\n"
                    "re-run `python -m cts oracle-embed` to resume.",
                    file=sys.stderr,
                    flush=True,
                )
                break

            dim = int(vecs.shape[1])
            if existing_dim is None:
                existing_dim = dim
            elif dim != existing_dim:
                print(
                    f"error: {cfg.embed_model} returns {dim}-dimensional vectors but the "
                    f"database already holds {existing_dim}-dimensional ones. Mixed "
                    "dimensions cannot be stacked into one matrix. Run "
                    "`DELETE FROM chunk_embeddings;` and re-run this command to rebuild.",
                    file=sys.stderr,
                    flush=True,
                )
                break

            conn.executemany(
                "INSERT OR REPLACE INTO chunk_embeddings (chunk_id, vec) VALUES (?, ?)",
                [
                    (row["id"], np.asarray(vec).astype(np.float32).tobytes())
                    for row, vec in zip(batch, vecs)
                ],
            )
            conn.commit()
            embedded += len(batch)
            if batch_no % PROGRESS_EVERY_BATCHES == 0:
                print(f"oracle-embed: {embedded}/{len(rows)}", flush=True)
    finally:
        conn.close()

    print(f"oracle-embed: done. embedded={embedded}", flush=True)
    return {"embedded": embedded}
