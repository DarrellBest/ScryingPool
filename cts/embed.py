"""Phase 6, write side: embed every proposition exactly once.

Propositions are embedded individually and stored as float32 blobs. Selection is
"props with no embeddings row", so the pass is idempotent and resumable: interrupt it,
re-run it, and it picks up the remainder. Phase 5 deletes a rewritten artwork's
embeddings along with its props, so a re-described artwork is automatically re-embedded
here.

The read side — stacking these blobs into one (n_props, dim) matrix and building the
BM25 index over the same texts — lives in cts/index.py.
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np

from . import db, ollama
from .config import Config

BATCH_SIZE = 32
PROGRESS_EVERY_BATCHES = 10  # ~320 propositions per progress line


def _prune_orphans(conn: sqlite3.Connection) -> int:
    """Drop vectors whose proposition no longer exists.

    Phase 5 already deletes them alongside the props it rewrites; this is a cheap
    safety net so a stray vector can never be stacked into the matrix and mapped to
    the wrong artwork.
    """
    cur = conn.execute(
        "DELETE FROM embeddings WHERE prop_id NOT IN (SELECT id FROM props)"
    )
    conn.commit()
    return cur.rowcount or 0


def _stored_dim(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT length(vec) AS n FROM embeddings LIMIT 1").fetchone()
    return None if row is None or not row["n"] else int(row["n"]) // 4


def run(cfg: Config) -> dict:
    if not cfg.embed_model.strip():
        print("error: no embed_model set in config.toml", file=sys.stderr)
        raise SystemExit(1)

    conn = db.connect(cfg)
    embedded = 0
    try:
        orphans = _prune_orphans(conn)
        if orphans:
            print(f"embed: pruned {orphans} vector(s) whose proposition is gone", flush=True)

        rows = conn.execute(
            """
            SELECT p.id, p.text
              FROM props p
              LEFT JOIN embeddings e ON e.prop_id = p.id
             WHERE e.prop_id IS NULL AND p.text IS NOT NULL AND p.text != ''
             ORDER BY p.id
            """
        ).fetchall()
        if not rows:
            print("embed: nothing to do", flush=True)
            return {"embedded": 0}

        existing_dim = _stored_dim(conn)
        print(
            f"embed: {len(rows)} propositions with {cfg.embed_model}, "
            f"batches of {BATCH_SIZE}",
            flush=True,
        )

        for batch_no, start in enumerate(range(0, len(rows), BATCH_SIZE), start=1):
            batch = rows[start : start + BATCH_SIZE]
            try:
                vecs = ollama.embed(cfg, [row["text"] for row in batch])
            except Exception as exc:
                print(
                    f"embed: stopped after {embedded} embeddings: {exc}\n"
                    "re-run `python -m cts embed` to resume.",
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
                    "`DELETE FROM embeddings;` and re-run this command to rebuild.",
                    file=sys.stderr,
                    flush=True,
                )
                break

            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (prop_id, vec) VALUES (?, ?)",
                [
                    (row["id"], np.asarray(vec).astype(np.float32).tobytes())
                    for row, vec in zip(batch, vecs)
                ],
            )
            conn.commit()
            embedded += len(batch)
            if batch_no % PROGRESS_EVERY_BATCHES == 0:
                print(f"embed: {embedded}/{len(rows)}", flush=True)
    finally:
        conn.close()

    print(f"embed: done. embedded={embedded}", flush=True)
    return {"embedded": embedded}
