"""Phase 11: run the held-out query set and score it.

The output of this system is subjective, so a broken pipeline and a working one
produce results that look equally plausible until someone opens the images.
This module is what makes a regression visible.

Three metrics, three query kinds, scored differently on purpose:

  literal      recall of a hand-built gold set, measured twice: inside the
               retrieval pool (did the retriever surface it at all) and inside
               the returned top 5 (did the judge keep it). Those two numbers
               fail for completely different reasons and must not be merged.
  abstract     precision at 5, from an operator opening the art and marking each
               result acceptable. Marks persist in `judgments` with
               source='human', keyed on the artwork, so the second run of the
               week is not interactive at all.
  adversarial  no score to optimise; run them, print what came back, and record
               the shape of the failure.

Plus pairwise agreement against stored `preferences`, which is the only reliable
way to measure something with no ground truth, and mean latency.

Every report pins prompt_version, the three model names and the index build
time, so a regression can be traced to what changed.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Config

QUERIES_PATH = "eval/queries.jsonl"
RESULTS_DIR = "eval/results"

# "within the returned pool" per SPEC.md Phase 7 step 6: retrieval returns the
# top 100 commanders and optimises recall; precision is the judge's job.
POOL_SIZE = 100

# Pairs offered per abstract query under --collect-prefs. Deliberately few and
# spread across the ranking: preferences are cheap to give but tedious in bulk,
# and comparisons at different depths are worth more than ten adjacent ones.
PREF_PAIRS = ((0, 1), (2, 3), (1, 4))


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def load_queries(path: str = QUERIES_PATH) -> list[dict]:
    """Read eval/queries.jsonl. Blank lines skipped, bad lines reported loudly."""
    p = Path(path)
    if not p.is_file():
        print(f"error: no eval query set at {p}", file=sys.stderr)
        print("Expected 40 held-out queries, one JSON object per line.", file=sys.stderr)
        raise SystemExit(1)

    rows: list[dict] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"error: {p}:{lineno} is not valid JSON: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        row.setdefault("id", f"?{lineno}")
        row.setdefault("kind", "abstract")
        row.setdefault("gold", [])
        row.setdefault("notes", "")
        if not row.get("text"):
            print(f"error: {p}:{lineno} has no 'text'", file=sys.stderr)
            raise SystemExit(1)
        rows.append(row)
    return rows


def _name_keys(name: str) -> set[str]:
    """Comparison keys for a card name.

    Case- and whitespace-insensitive, and a double-faced card stored as
    "Front // Back" also matches on its front face, since that is how people
    write gold sets by hand.
    """
    base = " ".join(str(name).split()).casefold()
    keys = {base}
    if "//" in base:
        keys.add(base.split("//", 1)[0].strip())
    return keys


def _matches(name: str, gold_keys: set[str]) -> bool:
    return bool(_name_keys(name) & gold_keys)


# ---------------------------------------------------------------------------
# database reads
# ---------------------------------------------------------------------------


def _pool_names(conn: sqlite3.Connection, query_id: int | None, limit: int = POOL_SIZE) -> list[str]:
    """Commander names in the retrieval pool for this query, best rank first.

    Read from `retrievals` rather than from the returned results because the
    pool is retrieval's product: search.execute returns the judged top k, and
    recall of the pool is the number that tells you whether the retriever ever
    had a chance.
    """
    if query_id is None:
        return []
    rows = conn.execute(
        """
        SELECT c.name AS name, MIN(r.rank) AS best
          FROM retrievals r
          JOIN arts  a ON a.illustration_id = r.illustration_id
          JOIN cards c ON c.oracle_id = a.oracle_id
         WHERE r.query_id = ?
         GROUP BY c.oracle_id
         ORDER BY best ASC
         LIMIT ?
        """,
        (query_id, limit),
    ).fetchall()
    return [r["name"] for r in rows if r["name"]]


def _pool_rank_map(conn: sqlite3.Connection, query_id: int | None) -> dict[str, int]:
    """illustration_id -> rank across the whole logged pool."""
    if query_id is None:
        return {}
    rows = conn.execute(
        """
        SELECT illustration_id, MIN(rank) AS best
          FROM retrievals
         WHERE query_id = ?
         GROUP BY illustration_id
         ORDER BY best ASC
        """,
        (query_id,),
    ).fetchall()
    return {r["illustration_id"]: i for i, r in enumerate(rows)}


def _stored_marks(conn: sqlite3.Connection, text: str) -> dict[str, float]:
    """Operator marks for this query TEXT, keyed on artwork.

    Keyed on text, not query_id: every run inserts a fresh `queries` row, but
    the judgment "this artwork does/doesn't fit this theme" is about the
    artwork and survives across runs. Later marks win.
    """
    rows = conn.execute(
        """
        SELECT j.illustration_id AS iid, j.fit AS fit
          FROM judgments j
          JOIN queries  q ON q.id = j.query_id
         WHERE j.source = 'human' AND q.text = ?
         ORDER BY j.rowid ASC
        """,
        (text,),
    ).fetchall()
    return {r["iid"]: float(r["fit"]) for r in rows if r["iid"] is not None}


def _stored_prefs(conn: sqlite3.Connection, text: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT p.art_a AS a, p.art_b AS b, p.winner AS winner
          FROM preferences p
          JOIN queries q ON q.id = p.query_id
         WHERE q.text = ?
        """,
        (text,),
    ).fetchall()
    return [{"a": r["a"], "b": r["b"], "winner": r["winner"]} for r in rows]


def _write_mark(
    conn: sqlite3.Connection, query_id: int | None, result: dict, accepted: bool
) -> None:
    if query_id is None:
        return
    conn.execute(
        "INSERT INTO judgments(query_id, illustration_id, fit, rationale, prop_ids, "
        "model, source) VALUES (?, ?, ?, ?, ?, ?, 'human')",
        (
            query_id,
            result.get("illustration_id"),
            1.0 if accepted else 0.0,
            "operator marked this result "
            + ("acceptable" if accepted else "not acceptable")
            + " during `python -m cts eval`",
            json.dumps(result.get("prop_ids") or []),
            "",  # a human is not a model
        ),
    )
    conn.commit()


def _write_pref(
    conn: sqlite3.Connection, query_id: int | None, art_a: str, art_b: str, winner: str
) -> None:
    if query_id is None:
        return
    conn.execute(
        "INSERT INTO preferences(query_id, art_a, art_b, winner, source) "
        "VALUES (?, ?, ?, ?, 'human')",
        (query_id, art_a, art_b, winner),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# operator interaction (always optional; a non-TTY run never blocks)
# ---------------------------------------------------------------------------


def _ask(session: dict, prompt: str, allowed: str) -> str | None:
    """Ask for one keystroke-ish answer. None means 'stop asking, ever'."""
    if not session["interactive"] or session["stopped"]:
        return None
    while True:
        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n(input closed — continuing without further prompts)")
            session["stopped"] = True
            return None
        if answer == "q":
            session["stopped"] = True
            return None
        if answer in allowed:
            return answer
        print(f"  please answer one of: {', '.join(allowed)} (or q to stop asking)")


def _art_link(result: dict) -> str:
    links = result.get("links") or {}
    return links.get("art_crop") or result.get("art_crop_url") or "(no art link)"


def _print_result_line(i: int, result: dict, mark: float | None) -> None:
    flag = "  " if mark is None else ("OK" if mark >= 0.5 else "no")
    fit = result.get("fit")
    fit_s = f"{float(fit):.2f}" if isinstance(fit, (int, float)) else " -- "
    name = str(result.get("name") or result.get("illustration_id") or "?")
    setc = str(result.get("set_code") or "?").upper()
    print(f"    {flag} {i}. {name}  [{setc}]  fit {fit_s}")
    print(f"          {_art_link(result)}")


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run(cfg: Config, collect_prefs: bool = False) -> dict:
    """Score eval/queries.jsonl. Returns the report dict that is also written to disk."""
    from . import db

    started_at = datetime.now(timezone.utc)
    queries = load_queries()
    conn = db.connect(cfg)

    session = {"interactive": sys.stdin.isatty() and sys.stdout.isatty(), "stopped": False}
    if collect_prefs and not session["interactive"]:
        print("eval: --collect-prefs needs a terminal; running without preference collection.")
        collect_prefs = False

    # --- pinned provenance -------------------------------------------------
    prompt_version = _prompt_version(conn)
    index_started = time.monotonic()
    index = _load_index(cfg, conn)
    # cts.index times its own build and exposes it for exactly this purpose; use
    # that number so the eval report and the plan logged by search.execute pin
    # the same value. Wall clock here is the fallback.
    index_build_seconds = round(
        float(getattr(index, "build_seconds", None) or (time.monotonic() - index_started)), 3
    )
    index_size = {
        "props": int(len(getattr(index, "prop_ids", []) or [])),
        "artworks": int(
            getattr(index, "artwork_count", None)
            or len(set(getattr(index, "illustration_ids", []) or []))
        ),
        "dim": int(getattr(index, "dim", 0) or 0),
        "missing_embeddings": int(getattr(index, "missing_embeddings", 0) or 0),
    }

    header = {
        "run_at": started_at.isoformat(),
        "prompt_version": prompt_version,
        "models": {
            "vision": cfg.vision_model,
            "embed": cfg.embed_model,
            "judge": cfg.judge_model,
        },
        "index_build_seconds": index_build_seconds,
        "index_size": index_size,
        "queries_file": QUERIES_PATH,
        "n_queries": len(queries),
        "interactive": session["interactive"],
        "collect_prefs": collect_prefs,
    }
    _print_header(header)

    try:
        from .search import execute as search_execute
    except ImportError as exc:
        print(f"error: cannot import cts.search ({exc}) — nothing to evaluate.", file=sys.stderr)
        raise SystemExit(1) from None

    per_query: list[dict] = []
    errors: list[dict] = []
    latencies: list[float] = []

    for row in queries:
        qid, kind, text = row["id"], row["kind"], row["text"]
        print(f"\n[{qid}] {kind}: {text}", flush=True)

        t0 = time.monotonic()
        try:
            out = search_execute(cfg, text, kind="eval", conn=conn, index=index, k=5)
        except Exception as exc:  # noqa: BLE001 - one bad query must not kill the run
            elapsed = round(time.monotonic() - t0, 3)
            print(f"    ERROR after {elapsed}s: {type(exc).__name__}: {exc}")
            errors.append({"id": qid, "text": text, "error": f"{type(exc).__name__}: {exc}"})
            per_query.append({"id": qid, "kind": kind, "text": text, "error": str(exc)})
            continue
        elapsed = round(time.monotonic() - t0, 3)
        latencies.append(elapsed)

        out = out if isinstance(out, dict) else {}
        query_id = out.get("query_id")
        results = list(out.get("results") or [])
        entry: dict = {
            "id": qid,
            "kind": kind,
            "text": text,
            "query_id": query_id,
            "latency_s": elapsed,
            "plan": out.get("plan"),
            "relaxed": out.get("relaxed"),
            "results": [
                {
                    "rank": i + 1,
                    "name": r.get("name"),
                    "illustration_id": r.get("illustration_id"),
                    "set_code": r.get("set_code"),
                    "artist": r.get("artist"),
                    "fit": r.get("fit"),
                    "verified": r.get("verified"),
                    "art_crop": _art_link(r),
                }
                for i, r in enumerate(results)
            ],
        }

        if kind == "literal":
            _score_literal(conn, row, out, results, entry)
        else:
            _score_marked(conn, session, row, out, results, entry, query_id)
            if collect_prefs and kind == "abstract":
                _collect_prefs(conn, session, text, results, query_id, entry)

        entry["pairwise"] = _score_pairwise(conn, text, results, query_id)
        per_query.append(entry)

    report = _summarize(header, per_query, errors, latencies)
    _print_summary(report)
    path = _write_report(report)
    report["report_path"] = path
    print(f"  report                         {path}")

    _persist_meta(conn, db, report, path)
    conn.close()
    return report


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _score_literal(
    conn: sqlite3.Connection, row: dict, out: dict, results: list[dict], entry: dict
) -> None:
    gold = [g for g in (row.get("gold") or []) if g]
    gold_keys: set[str] = set()
    for g in gold:
        gold_keys |= _name_keys(g)

    pool = _pool_names(conn, out.get("query_id"))
    if not pool:
        # No retrievals logged (a stubbed or partial search): fall back to the
        # returned results so the run still produces a number, flagged below.
        pool = [str(r.get("name") or "") for r in results]
        entry["pool_source"] = "results"
    else:
        entry["pool_source"] = "retrievals"

    top5 = [str(r.get("name") or "") for r in results[:5]]

    def _hits(names: list[str]) -> list[str]:
        found: list[str] = []
        for g in gold:
            keys = _name_keys(g)
            if any(keys & _name_keys(n) for n in names):
                found.append(g)
        return found

    hits_pool = _hits(pool)
    hits_top5 = _hits(top5)
    denom = len(gold) or 1

    entry["gold_size"] = len(gold)
    entry["gold_verified"] = bool(row.get("gold_verified"))
    entry["pool_size"] = len(pool)
    entry["recall_pool"] = round(len(hits_pool) / denom, 4)
    entry["recall_at_5"] = round(len(hits_top5) / denom, 4)
    entry["hits_pool"] = hits_pool
    entry["hits_top5"] = hits_top5
    entry["missed"] = [g for g in gold if g not in hits_pool]

    print(
        f"    recall pool {entry['recall_pool']:.2f} ({len(hits_pool)}/{len(gold)})"
        f"   top5 {entry['recall_at_5']:.2f} ({len(hits_top5)}/{len(gold)})"
        f"   [{_elapsed_str(entry)}]"
    )
    for i, r in enumerate(results[:5], start=1):
        _print_result_line(i, r, 1.0 if _matches(str(r.get("name") or ""), gold_keys) else 0.0)


def _elapsed_str(entry: dict) -> str:
    return f"{entry.get('latency_s', 0.0):.1f}s"


def _score_marked(
    conn: sqlite3.Connection,
    session: dict,
    row: dict,
    out: dict,
    results: list[dict],
    entry: dict,
    query_id: int | None,
) -> None:
    """Precision at 5 from operator marks, for abstract and adversarial queries."""
    text = row["text"]
    stored = _stored_marks(conn, text)
    marks: list[float | None] = []

    if row.get("notes"):
        print(f"    note: {row['notes'][:160]}")

    for i, result in enumerate(results[:5], start=1):
        iid = result.get("illustration_id")
        mark = stored.get(iid)
        _print_result_line(i, result, mark)
        if mark is None:
            answer = _ask(
                session,
                "          acceptable for this theme? [y/n/s=skip, q=stop asking] ",
                "yns",
            )
            if answer in ("y", "n"):
                mark = 1.0 if answer == "y" else 0.0
                _write_mark(conn, query_id, result, answer == "y")
        marks.append(mark)

    marked = [m for m in marks if m is not None]
    accepted = [m for m in marked if m >= 0.5]
    entry["marks"] = marks
    entry["n_marked"] = len(marked)
    entry["n_pending"] = len(marks) - len(marked)
    entry["n_accepted"] = len(accepted)
    # Denominator is marks actually given, not 5: an unopened result is unknown,
    # not a miss. n_pending is reported alongside so a thin number is obvious.
    entry["p_at_5"] = round(len(accepted) / len(marked), 4) if marked else None
    label = "P@5" if row["kind"] == "abstract" else "accept rate"
    shown = f"{entry['p_at_5']:.2f}" if entry["p_at_5"] is not None else "--"
    print(
        f"    {label} {shown}  ({len(accepted)} accepted of {len(marked)} marked, "
        f"{entry['n_pending']} pending)   [{_elapsed_str(entry)}]"
    )


def _score_pairwise(
    conn: sqlite3.Connection, text: str, results: list[dict], query_id: int | None
) -> dict:
    """Does this run's ranking agree with previously stored preferences?

    The ranking used is the system's end-to-end order: the returned results
    first, then the rest of the retrieval pool by rank. Pairs where one side is
    absent from both are not comparable and are counted separately rather than
    scored as a disagreement.
    """
    prefs = _stored_prefs(conn, text)
    if not prefs:
        return {"stored": 0, "comparable": 0, "agree": 0}

    order: dict[str, int] = {}
    for i, r in enumerate(results):
        iid = r.get("illustration_id")
        if iid and iid not in order:
            order[iid] = i
    offset = len(order)
    for iid, rank in _pool_rank_map(conn, query_id).items():
        if iid not in order:
            order[iid] = offset + rank

    comparable = 0
    agree = 0
    for pref in prefs:
        a, b, winner = pref["a"], pref["b"], pref["winner"]
        if a not in order or b not in order or winner not in (a, b):
            continue
        comparable += 1
        loser = b if winner == a else a
        if order[winner] < order[loser]:
            agree += 1
    return {"stored": len(prefs), "comparable": comparable, "agree": agree}


def _collect_prefs(
    conn: sqlite3.Connection,
    session: dict,
    text: str,
    results: list[dict],
    query_id: int | None,
    entry: dict,
) -> None:
    """Show pairs of matched artworks and record which fits the theme better.

    Pairwise judgments are far more consistent than absolute scoring and they
    double as Phase 12 training data, which is why they are collected here
    rather than in a separate tool.
    """
    existing = {
        frozenset((p["a"], p["b"])) for p in _stored_prefs(conn, text) if p["a"] and p["b"]
    }
    collected = 0
    for i, j in PREF_PAIRS:
        if session["stopped"] or i >= len(results) or j >= len(results):
            continue
        a, b = results[i], results[j]
        iid_a, iid_b = a.get("illustration_id"), b.get("illustration_id")
        if not iid_a or not iid_b or iid_a == iid_b:
            continue
        if frozenset((iid_a, iid_b)) in existing:
            continue
        print(f"\n    which fits better — \"{text}\"?")
        print(f"      [a] {a.get('name')}  {_art_link(a)}")
        print(f"      [b] {b.get('name')}  {_art_link(b)}")
        answer = _ask(session, "      better fit? [a/b/s=skip, q=stop asking] ", "abs")
        if answer in ("a", "b"):
            _write_pref(conn, query_id, iid_a, iid_b, iid_a if answer == "a" else iid_b)
            existing.add(frozenset((iid_a, iid_b)))
            collected += 1
    entry["prefs_collected"] = collected


# ---------------------------------------------------------------------------
# provenance, reporting, output
# ---------------------------------------------------------------------------


def _prompt_version(conn: sqlite3.Connection) -> int | None:
    """Current vision prompt version, from cts.prompts, else from the data."""
    try:
        from .prompts import PROMPT_VERSION

        return int(PROMPT_VERSION)
    except Exception:  # noqa: BLE001 - vision module may not be importable
        row = conn.execute("SELECT MAX(prompt_version) AS v FROM descriptions").fetchone()
        return int(row["v"]) if row and row["v"] is not None else None


def _load_index(cfg: Config, conn: sqlite3.Connection):
    """Load the BM25 index and embedding matrix once, for every query to share."""
    try:
        from .index import load_index

        return load_index(cfg, conn)
    except Exception as exc:  # noqa: BLE001 - degrade rather than abort the eval
        print(f"eval: warning: could not load the index ({exc}); each query will build its own.")
        return None


def _print_header(header: dict) -> None:
    models = header["models"]
    print("=" * 78)
    print(f"Scrying Pool eval — {header['run_at']}")
    print(
        f"  prompt_version {header['prompt_version']} | vision {models['vision']} | "
        f"embed {models['embed']} | judge {models['judge']}"
    )
    size = header["index_size"]
    missing = f", {size['missing_embeddings']} props unembedded" if size["missing_embeddings"] else ""
    print(
        f"  index build {header['index_build_seconds']}s "
        f"({size['props']} props / {size['artworks']} artworks, dim {size['dim']}{missing})"
        f" | {header['n_queries']} queries"
        f" | {'interactive' if header['interactive'] else 'non-interactive'}"
    )
    print("=" * 78)


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _summarize(header: dict, per_query: list[dict], errors: list[dict], latencies: list[float]) -> dict:
    lit = [e for e in per_query if e["kind"] == "literal" and "recall_pool" in e]
    absx = [e for e in per_query if e["kind"] == "abstract" and "p_at_5" in e]
    adv = [e for e in per_query if e["kind"] == "adversarial" and "p_at_5" in e]

    def marks_block(entries: list[dict]) -> dict:
        accepted = sum(e["n_accepted"] for e in entries)
        marked = sum(e["n_marked"] for e in entries)
        return {
            "n_queries": len(entries),
            "p_at_5": round(accepted / marked, 4) if marked else None,
            "p_at_5_macro": _mean([e["p_at_5"] for e in entries if e["p_at_5"] is not None]),
            "accepted": accepted,
            "marked": marked,
            "pending": sum(e["n_pending"] for e in entries),
        }

    comparable = sum(e.get("pairwise", {}).get("comparable", 0) for e in per_query)
    agree = sum(e.get("pairwise", {}).get("agree", 0) for e in per_query)
    stored = sum(e.get("pairwise", {}).get("stored", 0) for e in per_query)

    report = dict(header)
    report.update(
        {
            "literal": {
                "n_queries": len(lit),
                "recall_pool": _mean([e["recall_pool"] for e in lit]),
                "recall_at_5": _mean([e["recall_at_5"] for e in lit]),
                "gold_names": sum(e["gold_size"] for e in lit),
                "gold_verified": all(e.get("gold_verified") for e in lit) if lit else False,
            },
            "abstract": marks_block(absx),
            "adversarial": marks_block(adv),
            "pairwise": {
                "stored": stored,
                "comparable": comparable,
                "agree": agree,
                "agreement": round(agree / comparable, 4) if comparable else None,
            },
            "latency": {
                "n": len(latencies),
                "mean_s": _mean(latencies),
                "median_s": round(statistics.median(latencies), 4) if latencies else None,
                "max_s": round(max(latencies), 4) if latencies else None,
            },
            "errors": errors,
            "per_query": per_query,
        }
    )
    return report


def _print_summary(report: dict) -> None:
    lit, absx, adv = report["literal"], report["abstract"], report["adversarial"]
    pair, lat = report["pairwise"], report["latency"]

    def num(value, fmt="{:.3f}") -> str:
        return fmt.format(value) if isinstance(value, (int, float)) else "--"

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"  literal recall     pool {num(lit['recall_pool'])}   top5 "
        f"{num(lit['recall_at_5'])}   ({lit['n_queries']} queries, "
        f"{lit['gold_names']} gold names, "
        f"{'verified' if lit['gold_verified'] else 'UNVERIFIED gold'})"
    )
    print(
        f"  abstract P@5       {num(absx['p_at_5'])}   ({absx['accepted']} accepted of "
        f"{absx['marked']} marked, {absx['pending']} pending)"
    )
    print(
        f"  adversarial        {num(adv['p_at_5'])} accept rate   ({adv['accepted']} accepted "
        f"of {adv['marked']} marked, {adv['pending']} pending) — expected to be low"
    )
    print(
        f"  pairwise agreement {num(pair['agreement'])}   ({pair['agree']}/{pair['comparable']} "
        f"comparable of {pair['stored']} stored)"
    )
    print(
        f"  latency            mean {num(lat['mean_s'], '{:.2f}')}s   "
        f"median {num(lat['median_s'], '{:.2f}')}s   max {num(lat['max_s'], '{:.2f}')}s"
    )
    if report["errors"]:
        print(f"  errors             {len(report['errors'])} queries failed:")
        for err in report["errors"][:10]:
            print(f"                       {err['id']}: {err['error'][:90]}")


def _write_report(report: dict) -> str:
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _persist_meta(conn: sqlite3.Connection, db, report: dict, path: str) -> None:
    """Pin the run's provenance in `meta` so the last eval is queryable from SQL."""
    try:
        db.meta_set(conn, "eval_last_run_at", str(report["run_at"]))
        db.meta_set(conn, "eval_last_report", path)
        db.meta_set(conn, "eval_prompt_version", str(report["prompt_version"]))
        db.meta_set(conn, "eval_vision_model", report["models"]["vision"])
        db.meta_set(conn, "eval_embed_model", report["models"]["embed"])
        db.meta_set(conn, "eval_judge_model", report["models"]["judge"])
        db.meta_set(conn, "eval_index_build_seconds", str(report["index_build_seconds"]))
        db.meta_set(conn, "eval_literal_recall_pool", str(report["literal"]["recall_pool"]))
        db.meta_set(conn, "eval_abstract_p_at_5", str(report["abstract"]["p_at_5"]))
        db.meta_set(conn, "eval_pairwise_agreement", str(report["pairwise"]["agreement"]))
        db.meta_set(conn, "eval_mean_latency_s", str(report["latency"]["mean_s"]))
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail the eval
        print(f"eval: warning: could not stamp meta ({exc})")
