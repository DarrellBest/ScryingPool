"""Phase 4: art crop download.

One JPEG per `illustration_id` at `{cfg.art_dir}/{illustration_id}.jpg`, 100ms
apart, skipping anything already on disk. `arts.art_path` is set to the relative
path exactly as config spells it ("data/art/<id>.jpg"), which is what the vision
pass opens.

Downloads stream to a .part file and are renamed into place only once complete,
so an interrupted run never leaves a truncated JPEG that every later run would
happily skip.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from . import db
from .config import Config
from .ingest import USER_AGENT

# SPEC.md: 100ms between downloads.
REQUEST_DELAY = 0.1

_TIMEOUT = (15, 120)
_CHUNK = 1 << 16

# is_default first so a partial run still covers every commander once before it
# starts on alternate printings — the same ordering the vision pass uses.
_SELECT = """
SELECT illustration_id, art_crop_url, art_path
FROM arts
WHERE art_crop_url IS NOT NULL AND art_crop_url != ''
ORDER BY is_default DESC, illustration_id
"""


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(
            url, stream=True, timeout=_TIMEOUT, headers={"User-Agent": USER_AGENT}
        ) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        fh.write(chunk)
        if tmp.stat().st_size == 0:
            raise RuntimeError("empty response body")
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def run(cfg: Config) -> dict:
    """Phase 4. Idempotent: an art crop already on disk is never re-fetched."""
    conn = db.connect(cfg)
    try:
        art_dir = Path(cfg.art_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        prefix = cfg.art_dir.rstrip("/")

        rows = conn.execute(_SELECT).fetchall()
        total = len(rows)
        if not total:
            print("art: nothing to download", flush=True)
            return {"downloaded": 0, "skipped": 0}

        downloaded = 0
        skipped = 0
        failed = 0

        for index, row in enumerate(rows, start=1):
            illustration_id = row["illustration_id"]
            dest = art_dir / f"{illustration_id}.jpg"
            rel = f"{prefix}/{illustration_id}.jpg"

            if dest.is_file() and dest.stat().st_size > 0:
                # Already on disk. Still claim it, in case art_path is null
                # (first run after an interrupted one) or points somewhere else.
                if row["art_path"] != rel:
                    conn.execute(
                        "UPDATE arts SET art_path = ? WHERE illustration_id = ?",
                        (rel, illustration_id),
                    )
                    conn.commit()
                skipped += 1
                continue

            try:
                _download(row["art_crop_url"], dest)
            except (requests.RequestException, RuntimeError, OSError) as exc:
                failed += 1
                print(f"[{index}/{total}] {illustration_id} -> FAILED ({exc})", flush=True)
                continue

            conn.execute(
                "UPDATE arts SET art_path = ? WHERE illustration_id = ?",
                (rel, illustration_id),
            )
            conn.commit()
            downloaded += 1
            if downloaded % 50 == 0 or downloaded == 1:
                print(f"[{index}/{total}] downloaded {downloaded} art crops", flush=True)
            time.sleep(REQUEST_DELAY)

        print(
            f"art: {downloaded} downloaded, {skipped} already present"
            + (f", {failed} failed" if failed else ""),
            flush=True,
        )
        return {"downloaded": downloaded, "skipped": skipped}
    finally:
        conn.close()
