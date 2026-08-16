"""Phase 10: the weekly refresh.

One idempotent entry point behind `python -m cts refresh`, scheduled by the
systemd units that `install-timer.sh` writes.

The refresh is not a rebuild. Each stage below already selects only the rows
that lack its output, so a quiet week costs a few minutes of EDHREC polling and
zero model calls, while a set release picks up exactly the new cards and the new
artwork.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone

import requests

from .config import Config


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def _preflight(cfg: Config) -> str | None:
    """Return None when Ollama is up with every configured model, else why not.

    This runs before any other work on purpose. Steps 1-3 are pure HTTP and SQL
    and would happily complete with Ollama dead, leaving power scores recomputed
    over cards that can never get descriptions this run — a half-updated
    database whose failure is invisible until someone searches.
    """
    from . import ollama

    try:
        missing = ollama.preflight(cfg)
    except requests.ConnectionError:
        return (
            f"cannot reach Ollama at {cfg.ollama_url}\n"
            "  start it (`ollama serve`) or fix ollama_url in config.toml, then re-run.\n"
            "  nothing was changed."
        )
    except requests.Timeout:
        return (
            f"Ollama at {cfg.ollama_url} did not answer /api/tags within 30s\n"
            "  it may still be loading; re-run once it responds. nothing was changed."
        )
    except (requests.RequestException, RuntimeError) as exc:
        return f"Ollama preflight failed: {exc}\n  nothing was changed."

    if missing:
        lines = "\n".join(f"    - {name}" for name in missing)
        return (
            "Ollama is up but these configured models are not pulled:\n"
            f"{lines}\n"
            "  run `ollama pull <model>` for each, or point config.toml at models you "
            "have.\n  nothing was changed."
        )
    return None


def run(cfg: Config) -> int:
    """Refresh everything that moves. Returns a process exit code."""
    started = time.monotonic()
    print(f"refresh: starting {datetime.now(timezone.utc).isoformat(timespec='seconds')}")

    # ------------------------------------------------------------------
    # Step 0. Preflight, before touching anything.
    # ------------------------------------------------------------------
    problem = _preflight(cfg)
    if problem is not None:
        print(f"refresh: {problem}")
        return 1
    names = [cfg.vision_model, cfg.embed_model, cfg.judge_model]
    if cfg.verify_model and cfg.verify_model not in names:
        names.append(cfg.verify_model)
    print(f"refresh: preflight ok — {', '.join(names)} all present")

    # Stage modules are imported here rather than at module scope so that a
    # module which is missing or fails to import reports itself as a named
    # stage failure instead of breaking `python -m cts` entirely.
    from . import art, describe, edhrec, embed, ingest, power

    stages: list[tuple[str, str, object]] = [
        # 1. Bulk data. ingest.run compares the bulk-data updated_at against the
        #    meta table and skips the 200MB download when it has not moved; new
        #    cards are inserted, existing ones left alone.
        ("ingest", "scryfall bulk + cards/arts", lambda: ingest.run(cfg)),
        # 2. EDHREC for the WHOLE corpus, not just new cards: deck counts and
        #    archetype tags drift for everything. ~45 min at 1 req/s, weekly.
        ("edhrec", "edhrec (all cards)", lambda: edhrec.run(cfg, refresh_all=True)),
        # 3. Power for every card. The score is relative to the corpus
        #    distribution, so new cards move everyone. Pure SQL + numpy, seconds.
        ("power", "power scores", lambda: power.run(cfg)),
        # 4a. Art downloads are keyed on arts rows with no art_path — i.e. on
        #     illustration_id, not on card id. This is the distinction that makes
        #     the refresh actually work: Secret Lairs, precon alt-arts and
        #     reprint sets attach BRAND NEW illustration_ids to commanders that
        #     have been in `cards` for years. A "new cards only" check would find
        #     zero new cards that week and silently skip every one of those
        #     artworks.
        ("art", "art downloads", lambda: art.run(cfg)),
        # 4b. Same orientation for the vision pass: describe.run selects arts
        #     with art_path set and no `descriptions` row, so it picks up exactly
        #     the new illustration_ids. Artwork is immutable, so an
        #     illustration_id already in `descriptions` is never re-described.
        #
        #     backfill_stale is deliberately False. A prompt_version bump must
        #     NOT re-describe the corpus from inside the weekly job: that turns a
        #     five-minute refresh into an overnight one with no warning. Backfills
        #     are an explicit, separate command:
        #         python -m cts describe --backfill-stale
        ("describe", "vision pass (new artwork)", lambda: describe.run(cfg, backfill_stale=False)),
        # 4c. Embeddings for props that have no vector yet.
        ("embed", "embeddings", lambda: embed.run(cfg)),
        # 5. Indexes. There is no step here on purpose: cts.index builds the BM25
        #    index and the (n_props, dim) embedding matrix from scratch at load
        #    time, every time a search runs. At ~125k vectors that is cheap, and
        #    cheaper than maintaining incremental state between runs, so "rebuild
        #    the indexes" is satisfied by the next process that loads them.
    ]

    results: dict[str, dict] = {}
    for name, label, call in stages:
        stage_started = time.monotonic()
        print(f"refresh: [{name}] {label} ...", flush=True)
        try:
            out = call()
        except Exception:  # noqa: BLE001 - the summary must name the failed stage
            traceback.print_exc()
            print(
                f"refresh: FAILED in stage '{name}' ({label}) after "
                f"{_fmt_duration(time.monotonic() - started)}."
            )
            print(
                "refresh: earlier stages are committed and idempotent — fix the cause "
                "and re-run `python -m cts refresh`; completed work is not repeated."
            )
            return 1
        results[name] = out if isinstance(out, dict) else {}
        print(f"refresh: [{name}] done in {_fmt_duration(time.monotonic() - stage_started)}")

    elapsed = time.monotonic() - started

    # ------------------------------------------------------------------
    # Summary. This is the block that actually gets read on Monday morning,
    # so new commanders are listed by name, not counted.
    # ------------------------------------------------------------------
    new_cards = list(results.get("ingest", {}).get("new_cards") or [])
    new_arts = int(results.get("ingest", {}).get("new_arts") or 0)
    downloaded_bulk = bool(results.get("ingest", {}).get("downloaded"))
    edhrec_updated = int(results.get("edhrec", {}).get("updated") or 0)
    edhrec_misses = int(results.get("edhrec", {}).get("misses") or 0)
    scored = int(results.get("power", {}).get("scored") or 0)
    art_downloaded = int(results.get("art", {}).get("downloaded") or 0)
    described = int(results.get("describe", {}).get("described") or 0)
    describe_failed = int(results.get("describe", {}).get("failed") or 0)
    embedded = int(results.get("embed", {}).get("embedded") or 0)

    rule = "=" * 66
    print()
    print(rule)
    print("refresh summary")
    print(rule)
    print(f"  bulk data          {'re-downloaded' if downloaded_bulk else 'unchanged, skipped'}")
    if new_cards:
        print(f"  new commanders     {len(new_cards)}")
        for name in sorted(new_cards):
            print(f"                     - {name}")
    else:
        print("  new commanders     none")
    print(f"  new artworks       {new_arts} ({art_downloaded} art crops downloaded)")
    print(f"  edhrec rows        {edhrec_updated} updated, {edhrec_misses} misses")
    print(f"  power scores       {scored} recomputed")
    print(
        f"  vision calls       {described} artworks described"
        + (f", {describe_failed} failed" if describe_failed else "")
    )
    print(f"  embeddings         {embedded} propositions embedded")
    print("  indexes            rebuilt on next load (BM25 + matrix are built at load time)")
    print(f"  total runtime      {_fmt_duration(elapsed)}")
    print(rule)

    # Stamp the run so `sqlite3 data/commanders.db "select * from meta"` answers
    # "did the timer actually fire last Sunday?" without digging in journald.
    try:
        from . import db

        conn = db.connect(cfg)
        db.meta_set(conn, "last_refresh_at", datetime.now(timezone.utc).isoformat())
        db.meta_set(conn, "last_refresh_seconds", f"{elapsed:.1f}")
        db.meta_set(conn, "last_refresh_new_cards", str(len(new_cards)))
        db.meta_set(conn, "last_refresh_described", str(described))
        conn.close()
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail the run
        print(f"refresh: warning: could not stamp meta ({exc})")

    return 0
