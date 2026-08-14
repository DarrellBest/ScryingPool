"""Phase 12: write the training sets.

`python -m cts export-training --target embed|judge`

The exports are the real output of Phase 12. The adapter is disposable and will
be obsoleted by the next base model; a few tens of thousands of labeled
(theme, artwork, fit) judgments spanning literal through analogical themes is
not.

Two targets:

  embed   (theme, positive artwork props, hard negatives) triples for
          MultipleNegativesRankingLoss. The negatives carry almost all the
          value: an artwork that was retrieved and then rejected was
          semantically close and wrong, which is exactly the distinction the
          base embedding model cannot make.
  judge   task-tagged SFT records for one multi-task adapter covering routing,
          decomposition and judging. They share the underlying skill of knowing
          what an abstract art theme means in this domain, and one adapter is
          one thing to serve.

Both split ~95/5 train/val BY QUERY TEXT, never by artwork. The skill that has
to generalize is understanding a theme it has never seen, so an artwork in both
splits is harmless while a theme in both makes the eval meaningless.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .config import Config

DEFAULT_OUT_DIR = "exports"

POSITIVE_FIT = 0.7  # SPEC.md Phase 8: fit is continuous; these are the export cuts
NEGATIVE_FIT = 0.3
ACCEPT_FIT = 0.5  # the accept/reject boundary for the judge SFT balance

VAL_FRACTION = 0.05
MAX_NEGATIVES = 8
MAX_PROPS = 4  # per artwork, per record
MAX_PROP_CHARS = 700
MAX_JUDGE_INPUT_CHARS = 4000

# Human rows are emitted three times. Most trainers — sentence-transformers'
# MultipleNegativesRankingLoss included — have no per-example weight argument, so
# duplication is how a row gets weighted. Three is chosen to keep a few hundred
# human corrections influential against tens of thousands of synthetic pairs
# without letting them dominate: they encode what *you* meant by a theme rather
# than what a model guessed you meant, and they are the only rows in the corpus
# that do.
HUMAN_WEIGHT = 3

_SOURCE_PRIORITY = {"distill": 1, "judge": 2, "human": 3}

# Registers whose evidence lives in the literal layer. Everything else reads
# more naturally off the interpretive props.
_LITERAL_REGISTERS = frozenset({"literal", "compositional"})

_PLAN_KEYS = (
    "literal_weight",
    "interpretive_weight",
    "weights",
    "weight",
    "slot_filter",
    "slot_filters",
    "slots",
    "mechanical_filter",
    "mechanical",
    "filters",
    "expansions",
    "direct",
    "decomposed",
)
_PLAN_NOISE_KEYS = frozenset(
    {"illustration_id", "polarity", "register", "band", "colors", "k", "kind", "as_json"}
)
_DIRECT_KEYS = ("direct", "direct_expansions", "interpretive_expansions", "interpretive")
_DECOMPOSED_KEYS = ("decomposed", "decomposed_expansions", "literal_expansions", "literal")


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def split_for(text: str, val_fraction: float = VAL_FRACTION) -> str:
    """Deterministic train/val assignment from the query text alone.

    Hashing the text (rather than shuffling rows) guarantees every record with
    the same theme lands in the same split, in this run and in every future one,
    including the duplicated human rows.
    """
    digest = hashlib.sha1(_norm_text(text).encode("utf-8")).hexdigest()
    return "val" if (int(digest[:8], 16) % 10_000) < val_fraction * 10_000 else "train"


# ---------------------------------------------------------------------------
# proposition lookup
# ---------------------------------------------------------------------------


class _Props:
    """Lazy per-artwork proposition cache. Real state, so a real object."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: dict[str, dict] = {}

    def get(self, illustration_id: str) -> dict:
        cached = self._cache.get(illustration_id)
        if cached is None:
            rows = self._conn.execute(
                "SELECT id, layer, text FROM props WHERE illustration_id = ? ORDER BY id",
                (illustration_id,),
            ).fetchall()
            cached = {
                "by_id": {int(r["id"]): str(r["text"] or "") for r in rows},
                "literal": [str(r["text"] or "") for r in rows if r["layer"] == "literal"],
                "interpretive": [
                    str(r["text"] or "") for r in rows if r["layer"] == "interpretive"
                ],
                "all": [str(r["text"] or "") for r in rows],
                "numbered": "\n".join(
                    f"[{r['id']}] ({r['layer']}) {r['text']}" for r in rows
                ),
            }
            self._cache[illustration_id] = cached
        return cached

    def text_for(
        self, illustration_id: str, cited: list[int] | None = None, register: str | None = None
    ) -> str:
        """The artwork side of a training pair: a few propositions, not the record.

        Deliberately a small selection rather than the whole description. At
        inference each proposition is embedded on its own and the query is
        matched against individual propositions, so training on one concatenated
        blob per artwork would teach the model a granularity it never sees.
        Cited propositions win when the judgment named them, since those are the
        evidence the label actually rests on.
        """
        props = self.get(illustration_id)
        chosen: list[str] = []
        for pid in cited or []:
            text = props["by_id"].get(int(pid))
            if text:
                chosen.append(text)
        if not chosen:
            layer = "literal" if (register or "") in _LITERAL_REGISTERS else "interpretive"
            chosen = props[layer] or props["all"]
        chosen = [c for c in chosen if c][:MAX_PROPS]
        return " ".join(chosen)[:MAX_PROP_CHARS].strip()


# ---------------------------------------------------------------------------
# label collection (shared by both targets)
# ---------------------------------------------------------------------------


def _collect_labels(conn: sqlite3.Connection) -> tuple[dict, dict]:
    """Fold `judgments` into one label per (query text, artwork).

    Keyed on text rather than query_id because the same theme is logged as a new
    `queries` row on every run; the label is about the artwork. A human mark
    overrides a model's on the same pair, which is the entire point of recording
    corrections.
    """
    rows = conn.execute(
        """
        SELECT q.text AS text, q.kind AS kind, q.params AS params,
               j.illustration_id AS iid, j.fit AS fit, j.rationale AS rationale,
               j.prop_ids AS prop_ids, j.source AS source, j.model AS model
          FROM judgments j
          JOIN queries q ON q.id = j.query_id
         WHERE j.illustration_id IS NOT NULL AND j.fit IS NOT NULL
         ORDER BY j.rowid ASC
        """
    ).fetchall()

    labels: dict[tuple[str, str], dict] = {}
    raw_text: dict[str, str] = {}
    for row in rows:
        text = str(row["text"] or "")
        if not text.strip():
            continue
        key_text = _norm_text(text)
        raw_text.setdefault(key_text, text.strip())
        params = _load_json(row["params"]) or {}
        record = {
            "text": raw_text[key_text],
            "kind": row["kind"],
            "register": params.get("register"),
            "polarity": params.get("polarity"),
            "iid": row["iid"],
            "fit": float(row["fit"]),
            "rationale": str(row["rationale"] or ""),
            "prop_ids": _load_json(row["prop_ids"]) or [],
            "source": str(row["source"] or "judge"),
        }
        key = (key_text, str(row["iid"]))
        previous = labels.get(key)
        if previous is None or _SOURCE_PRIORITY.get(record["source"], 0) >= _SOURCE_PRIORITY.get(
            previous["source"], 0
        ):
            labels[key] = record

    grouped: dict[str, list[dict]] = {}
    for (key_text, _iid), record in labels.items():
        grouped.setdefault(key_text, []).append(record)
    return grouped, raw_text


def _load_json(raw) -> object:
    if not raw:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _weight(source: str) -> int:
    return HUMAN_WEIGHT if source == "human" else 1


# ---------------------------------------------------------------------------
# target: embed
# ---------------------------------------------------------------------------


def _build_embed(conn: sqlite3.Connection, props: _Props) -> tuple[list[dict], dict]:
    grouped, _raw = _collect_labels(conn)
    records: list[dict] = []
    stats = {
        "query_texts": len(grouped),
        "positive_pairs": 0,
        "negative_pairs": 0,
        "records_from_judgments": 0,
        "records_from_preferences": 0,
        "texts_with_negatives": 0,
        "orphan_negative_texts": 0,
        "human_records": 0,
    }

    for key_text, rows in sorted(grouped.items()):
        positives = [r for r in rows if r["fit"] >= POSITIVE_FIT]
        negatives = [r for r in rows if r["fit"] <= NEGATIVE_FIT]
        stats["positive_pairs"] += len(positives)
        stats["negative_pairs"] += len(negatives)
        if negatives:
            stats["texts_with_negatives"] += 1
        if negatives and not positives:
            # A near-miss theme whose text never appears as a positive anywhere.
            # It stays unusable for MultipleNegativesRankingLoss, which needs an
            # anchor; counted so the ratio of wasted hard negatives is visible.
            stats["orphan_negative_texts"] += 1
            continue
        if not positives:
            continue

        negative_texts: list[str] = []
        seen_negatives: set[str] = set()
        for neg in sorted(negatives, key=lambda r: (-r["fit"], str(r["iid"]))):
            text = props.text_for(neg["iid"], neg["prop_ids"], neg["register"])
            if text and text not in seen_negatives:
                seen_negatives.add(text)
                negative_texts.append(text)
            if len(negative_texts) >= MAX_NEGATIVES:
                break

        for pos in sorted(positives, key=lambda r: str(r["iid"])):
            positive_text = props.text_for(pos["iid"], pos["prop_ids"], pos["register"])
            if not positive_text:
                continue
            weight = _weight(pos["source"])
            if weight > 1:
                stats["human_records"] += 1
            records.append(
                {
                    "record": {
                        "query": pos["text"],
                        "positive": positive_text,
                        "negatives": [n for n in negative_texts if n != positive_text],
                        "meta": {
                            "illustration_id": pos["iid"],
                            "source": pos["source"],
                            "register": pos["register"],
                            "fit": pos["fit"],
                        },
                    },
                    "text": pos["text"],
                    "weight": weight,
                }
            )
            stats["records_from_judgments"] += 1

    records.extend(_preference_records(conn, props, stats))
    return records, stats


def _preference_records(conn: sqlite3.Connection, props: _Props, stats: dict) -> list[dict]:
    """Pairwise preferences are already a triple: winner positive, loser negative.

    These come from Phase 11 and from real use, so they are dataset 3 and get the
    human weight when they were recorded by a person.
    """
    rows = conn.execute(
        """
        SELECT q.text AS text, p.art_a AS a, p.art_b AS b, p.winner AS winner,
               p.source AS source
          FROM preferences p
          JOIN queries q ON q.id = p.query_id
         WHERE p.winner IS NOT NULL
        """
    ).fetchall()

    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        text = str(row["text"] or "").strip()
        winner, a, b = row["winner"], row["a"], row["b"]
        if not text or winner not in (a, b):
            continue
        loser = b if winner == a else a
        key = (_norm_text(text), str(winner), str(loser))
        if key in seen:
            continue
        seen.add(key)
        positive_text = props.text_for(winner)
        negative_text = props.text_for(loser)
        if not positive_text or not negative_text:
            continue
        weight = _weight(str(row["source"] or ""))
        if weight > 1:
            stats["human_records"] += 1
        out.append(
            {
                "record": {
                    "query": text,
                    "positive": positive_text,
                    "negatives": [negative_text],
                    "meta": {
                        "illustration_id": winner,
                        "source": str(row["source"] or ""),
                        "register": None,
                        "from": "preference",
                    },
                },
                "text": text,
                "weight": weight,
            }
        )
        stats["records_from_preferences"] += 1
    return out


# ---------------------------------------------------------------------------
# target: judge (multi-task: route | decompose | judge)
# ---------------------------------------------------------------------------

ROUTE_INSTRUCTION = (
    "task: route\n"
    "Read a free-text art search theme and return the retrieval plan as JSON: any "
    "structured slot filter it maps to, any mechanical filter over card text or deck "
    "archetypes, and a literal/interpretive weight pair summing to 1. This is a blend, "
    "not a branch — a theme that is genuinely both must not be forced to pick."
)
DECOMPOSE_INSTRUCTION = (
    "task: decompose\n"
    "Expand a free-text art search theme two ways and return JSON. \"direct\": 5-8 "
    "restatements in interpretive register, as an image description would phrase the "
    "feeling. \"decomposed\": the concrete physical evidence a matching image would "
    "actually contain — subjects, poses, palette, composition — as a literal description "
    "would phrase it."
)
JUDGE_INSTRUCTION = (
    "task: judge\n"
    "Decide how well one artwork fits a search theme. Return JSON with \"fit\" (0 to 1, "
    "continuous — literal themes cluster near 0 and 1 on their own, abstract themes "
    "genuinely live in the middle), \"rationale\" (one sentence), and \"prop_ids\" (the "
    "numbered propositions the rationale relies on)."
)


def _extract_plan(params: object) -> dict | None:
    if not isinstance(params, dict):
        return None
    plan = params.get("plan")
    if isinstance(plan, dict) and plan:
        return plan
    if any(key in params for key in _PLAN_KEYS):
        return {k: v for k, v in params.items() if k not in _PLAN_NOISE_KEYS}
    return None


def _first_list(plan: dict, keys: tuple[str, ...]) -> list:
    for key in keys:
        value = plan.get(key)
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return value
    expansions = plan.get("expansions")
    if isinstance(expansions, dict):
        for key in keys:
            value = expansions.get(key)
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                return value
    return []


def _build_judge(conn: sqlite3.Connection, props: _Props) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {
        "route": 0,
        "decompose": 0,
        "judge": 0,
        "judge_accept_raw": 0,
        "judge_reject_raw": 0,
        "judge_dropped_for_balance": 0,
        "human_records": 0,
        "registers": {},
    }

    # --- route and decompose, from the plans logged by every real search -----
    for row in conn.execute(
        "SELECT text, params FROM queries WHERE kind IN ('user', 'eval') ORDER BY id ASC"
    ):
        text = str(row["text"] or "").strip()
        plan = _extract_plan(_load_json(row["params"]))
        if not text or not plan:
            continue
        direct = _first_list(plan, _DIRECT_KEYS)
        decomposed = _first_list(plan, _DECOMPOSED_KEYS)
        route_plan = {
            k: v for k, v in plan.items() if k not in {"direct", "decomposed", "expansions"}
        }
        if route_plan:
            records.append(
                {
                    "record": {
                        "task": "route",
                        "instruction": ROUTE_INSTRUCTION,
                        "input": text,
                        "output": json.dumps(route_plan, ensure_ascii=False, sort_keys=True),
                        "meta": {"source": "log"},
                    },
                    "text": text,
                    "weight": 1,
                }
            )
            stats["route"] += 1
        if direct or decomposed:
            records.append(
                {
                    "record": {
                        "task": "decompose",
                        "instruction": DECOMPOSE_INSTRUCTION,
                        "input": text,
                        "output": json.dumps(
                            {"direct": direct, "decomposed": decomposed}, ensure_ascii=False
                        ),
                        "meta": {"source": "log"},
                    },
                    "text": text,
                    "weight": 1,
                }
            )
            stats["decompose"] += 1

    # --- judge, from every stored judgment -----------------------------------
    grouped, _raw = _collect_labels(conn)
    accepts: list[dict] = []
    rejects: list[dict] = []
    for _key_text, rows in sorted(grouped.items()):
        for label in sorted(rows, key=lambda r: str(r["iid"])):
            art = props.get(label["iid"])
            if not art["numbered"]:
                continue
            description = conn.execute(
                "SELECT literal, interpretive FROM descriptions WHERE illustration_id = ?",
                (label["iid"],),
            ).fetchone()
            literal = str(description["literal"] or "") if description else ""
            interpretive = str(description["interpretive"] or "") if description else ""
            payload = (
                f"THEME: {label['text']}\n\n"
                f"=== LITERAL ===\n{literal}\n\n"
                f"=== INTERPRETIVE ===\n{interpretive}\n\n"
                f"=== PROPOSITIONS ===\n{art['numbered']}"
            )[:MAX_JUDGE_INPUT_CHARS]
            item = {
                "record": {
                    "task": "judge",
                    "instruction": JUDGE_INSTRUCTION,
                    "input": payload,
                    "output": json.dumps(
                        {
                            "fit": round(float(label["fit"]), 3),
                            "rationale": label["rationale"],
                            "prop_ids": label["prop_ids"],
                        },
                        ensure_ascii=False,
                    ),
                    "meta": {
                        "source": label["source"],
                        "register": label["register"],
                        "illustration_id": label["iid"],
                    },
                },
                "text": label["text"],
                "weight": _weight(label["source"]),
            }
            register = label["register"] or ("literal" if label["kind"] == "literal" else "unknown")
            stats["registers"][register] = stats["registers"].get(register, 0) + 1
            if label["fit"] >= ACCEPT_FIT:
                accepts.append(item)
            else:
                rejects.append(item)

    stats["judge_accept_raw"] = sum(i["weight"] for i in accepts)
    stats["judge_reject_raw"] = sum(i["weight"] for i in rejects)

    kept_accepts, kept_rejects = _balance(accepts, rejects)
    stats["judge_dropped_for_balance"] = (len(accepts) - len(kept_accepts)) + (
        len(rejects) - len(kept_rejects)
    )
    stats["judge"] = len(kept_accepts) + len(kept_rejects)
    stats["judge_accept"] = sum(i["weight"] for i in kept_accepts)
    stats["judge_reject"] = sum(i["weight"] for i in kept_rejects)
    stats["human_records"] = sum(
        1 for i in records + kept_accepts + kept_rejects if i["weight"] > 1
    )
    records.extend(kept_accepts)
    records.extend(kept_rejects)
    return records, stats


def _balance(accepts: list[dict], rejects: list[dict]) -> tuple[list[dict], list[dict]]:
    """Downsample the majority class so the emitted mix is near 50/50.

    A judge trained mostly on positives becomes a yes-machine and quietly
    destroys precision, which is the single most common way this kind of
    fine-tune fails. Balance is enforced by construction here rather than hoped
    for. Weighted counts are used, since a human row is emitted three times and
    the trainer sees all three. Human rows are kept first and never dropped
    while any model-labeled row of the same class remains.
    """
    if not accepts or not rejects:
        return accepts, rejects

    def order(items: list[dict]) -> list[dict]:
        return sorted(
            items,
            key=lambda i: (
                -i["weight"],
                hashlib.sha1(
                    (str(i["record"]["meta"].get("illustration_id")) + i["text"]).encode("utf-8")
                ).hexdigest(),
            ),
        )

    target = min(sum(i["weight"] for i in accepts), sum(i["weight"] for i in rejects))

    def take(items: list[dict]) -> list[dict]:
        kept: list[dict] = []
        total = 0
        for item in order(items):
            if total >= target:
                break
            kept.append(item)
            total += item["weight"]
        return kept

    return take(accepts), take(rejects)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _write_split(records: list[dict], out_dir: Path, target: str) -> dict:
    """Expand weights into duplicates and write the two JSONL files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / f"{target}_train.jsonl",
        "val": out_dir / f"{target}_val.jsonl",
    }
    counts = {"train": 0, "val": 0}
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        for item in records:
            # Duplication happens after the split assignment, and the split is a
            # function of the query text, so every copy of a row lands in the
            # same file by construction.
            split = split_for(item["text"])
            line = json.dumps(item["record"], ensure_ascii=False)
            for _ in range(max(1, int(item["weight"]))):
                handles[split].write(line + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "train_path": str(paths["train"]),
        "val_path": str(paths["val"]),
        "train": counts["train"],
        "val": counts["val"],
    }


def run(cfg: Config, target: str, out: str | None = None) -> dict:
    """Export the `embed` or `judge` training set. `out` is a directory."""
    from . import db

    if target not in ("embed", "judge"):
        print(f"error: unknown --target {target!r} (expected 'embed' or 'judge')")
        raise SystemExit(2)

    out_dir = Path(out or DEFAULT_OUT_DIR)
    if out_dir.exists() and not out_dir.is_dir():
        print(f"error: --out {out_dir} exists and is not a directory")
        print(f"       this target writes two files, {target}_train.jsonl and {target}_val.jsonl")
        raise SystemExit(2)

    conn = db.connect(cfg)
    props = _Props(conn)
    records, stats = (_build_embed if target == "embed" else _build_judge)(conn, props)
    conn.close()

    if not records:
        print(f"export-training: nothing to export for --target {target}.")
        print(
            "  run `python -m cts synth` for the synthetic corpus, and/or some searches "
            "to accumulate logged plans and judgments."
        )
        return {"target": target, "records": 0, "train": 0, "val": 0, "stats": stats}

    written = _write_split(records, out_dir, target)
    report = {
        "target": target,
        "out_dir": str(out_dir),
        "records": len(records),
        "lines_after_weighting": written["train"] + written["val"],
        "stats": stats,
        **written,
    }
    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    stats = report["stats"]
    target = report["target"]
    total = report["train"] + report["val"]
    print()
    print(f"export-training --target {target}")
    print(f"  {report['train_path']}   {report['train']} lines")
    print(f"  {report['val_path']}   {report['val']} lines")
    print(
        f"  {report['records']} records -> {total} lines after weighting "
        f"(human rows x{HUMAN_WEIGHT}); split by query text, "
        f"{report['val'] / total * 100:.1f}% val"
    )

    if target == "embed":
        print(
            f"  {stats['query_texts']} distinct themes | {stats['positive_pairs']} positive "
            f"pairs | {stats['negative_pairs']} negative pairs"
        )
        print(
            f"  {stats['records_from_judgments']} triples from judgments, "
            f"{stats['records_from_preferences']} from preferences, "
            f"{stats['human_records']} human-weighted"
        )
        if stats["orphan_negative_texts"]:
            print(
                f"  note: {stats['orphan_negative_texts']} near-miss themes had no positive "
                "artwork anywhere and produced no triple — MultipleNegativesRankingLoss "
                "needs an anchor. They become usable once a search judges some artwork "
                "as fitting that theme."
            )
        if not stats["texts_with_negatives"]:
            print(
                "  WARNING: no hard negatives in this export. The negatives carry most of "
                "the value here; run `python -m cts synth` and some real searches first."
            )
    else:
        accept = stats.get("judge_accept", 0)
        reject = stats.get("judge_reject", 0)
        judged = accept + reject
        ratio = accept / judged if judged else 0.0
        print(
            f"  tasks: route {stats['route']} | decompose {stats['decompose']} | "
            f"judge {stats['judge']}"
        )
        print(
            f"  judge accept/reject: {accept}/{reject} = {ratio:.2f} accept "
            f"(raw {stats['judge_accept_raw']}/{stats['judge_reject_raw']}, "
            f"{stats['judge_dropped_for_balance']} rows dropped to balance)"
        )
        if judged and not 0.4 <= ratio <= 0.6:
            print(
                "  WARNING: accept ratio is outside 0.40-0.60. A judge trained mostly on "
                "one class becomes a yes- or no-machine; get more of the missing class "
                "before training."
            )
        registers = stats.get("registers") or {}
        if registers:
            spread = " | ".join(f"{k} {v}" for k, v in sorted(registers.items()))
            print(f"  judge registers: {spread}")
            if len(registers) < 3:
                print(
                    "  WARNING: fewer than three registers represented. The adapter will "
                    "overfit to one and get worse than the base model at the others."
                )
        if not stats["route"] and not stats["decompose"]:
            print(
                "  note: no routing plans logged yet, so this export is judge-only. Run "
                "some searches — search.execute logs its plan into queries.params."
            )
