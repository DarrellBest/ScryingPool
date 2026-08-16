"""Phase 8: judge the retrieved pool, verify the finalists with eyes, diversify.

Retrieval optimizes recall; everything here is precision. Three stages:

1. `judge_batches` — the judge model scores each candidate 0-1 against the *matched
   artwork's* two description layers and its propositions, citing the prop ids it
   relied on. A rationale that cannot point at a proposition is one the model invented,
   so invented ids are dropped and counted.
2. `verify_finalists` — the top eight go back to a multimodal model (`verify_model`)
   against the actual art crop. Everything upstream reasons over a description written
   by another model at another time; this is the only stage that can catch a detail the
   vision pass compressed away.
3. `diversify` — colour cap plus MMR, so a theme that attracts one visual convention
   does not return five of it.

Every stage degrades instead of crashing: a failed batch falls back to retrieval
order with a null fit, and an unreachable verify model leaves results unverified and
says so.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import requests

from . import ollama
from .config import Config
from .index import SearchIndex

BATCH_SIZE = 10
VERIFY_TOP_N = 8
PASS_FIT = 0.5          # at or above this a result is a real match, below it a stretch
MMR_LAMBDA = 0.7
COLOR_CAP = 2           # at most two results per colour identity
DIVERSITY_POOL = 15     # MMR picks the final k out of this many survivors
MAX_PROPS_SHOWN = 40    # an artwork has ~25; the cap only bounds pathological rows

JUDGE_SYSTEM = (
    "You are the precision stage of an art search engine for Magic: The Gathering "
    "commanders. You score how well each candidate's ARTWORK fits a theme. You never "
    "score the card's gameplay, rules text, price or reputation — only the artwork "
    "described to you."
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate": {"type": "integer"},
                    "fit": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                    "prop_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["candidate", "fit", "rationale", "prop_ids"],
            },
        }
    },
    "required": ["judgments"],
}

JUDGE_RULES = """Rules:
- fit is CONTINUOUS from 0.0 to 1.0, never a yes/no. Literal themes settle near 0 or 1
  by themselves; abstract themes genuinely live in the middle, and rounding them off
  throws away the only information that makes a pool rankable.
  1.0 unmistakable, 0.7 strong, 0.5 real but partial, 0.3 a stretch, 0.0 unrelated.
- rationale is ONE sentence about this specific image, concrete, no hedging.
- prop_ids are the bracketed numbers you actually relied on, copied exactly. Every
  candidate header states the only range of numbers that candidate may cite — a number
  outside its own range belongs to a different image and is always wrong. Check each
  number against that range before you write it. Copy digits, do not reconstruct them.
  If nothing in the propositions supports the theme, return an empty list and a low fit.
- When a candidate is marked with more than one art in the index, name the printing
  (set code and artist) in your rationale: "Atraxa fits" is misleading when only one of
  six printings does.
- Judge only what the descriptions say. Do not credit the theme to lore you happen to
  know about the card, and do not reward a card for being powerful or popular.
- Return exactly one entry per candidate, reusing the candidate numbers as given."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "holds": {"type": "boolean"},
        "why": {"type": "string"},
    },
    "required": ["holds", "why"],
}


def parse_json(text: str) -> dict:
    """Parse a model response that should be a JSON object, tolerating fences."""
    body = (text or "").strip()
    if body.startswith("```"):
        body = body.strip("`")
        if "\n" in body:
            body = body.split("\n", 1)[1]
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model response: {text[:200]!r}")
    parsed = json.loads(body[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response is not a JSON object")
    return parsed


# --------------------------------------------------------------------------- evidence


def load_evidence(conn: sqlite3.Connection, illustration_ids: list[str]) -> dict[str, dict]:
    """Both description layers and every proposition, per matched artwork.

    Only the matched artwork: not the whole card record, and never a merge across
    printings, which is exactly the confusion this system exists to avoid.
    """
    evidence: dict[str, dict] = {
        ill: {"literal": "", "interpretive": "", "props": []} for ill in illustration_ids
    }
    for chunk in chunks(illustration_ids, 400):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT illustration_id, literal, interpretive FROM descriptions "
            f"WHERE illustration_id IN ({marks})",
            chunk,
        ):
            entry = evidence[row["illustration_id"]]
            entry["literal"] = row["literal"] or ""
            entry["interpretive"] = row["interpretive"] or ""
        for row in conn.execute(
            f"SELECT id, illustration_id, layer, text FROM props "
            f"WHERE illustration_id IN ({marks}) ORDER BY id",
            chunk,
        ):
            evidence[row["illustration_id"]]["props"].append(
                {"id": int(row["id"]), "layer": row["layer"], "text": row["text"]}
            )
    return evidence


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------- judging


def number_batch(batch: list[dict], evidence: dict[str, dict]) -> list[dict]:
    """Give every proposition in one batch a short id, numbered from 1.

    Propositions carry their global `props.id`, which is a six-digit number. Asking a
    model to copy ten candidates' worth of six-digit ids exactly, and to keep straight
    which candidate each belongs to, is the source of Defect 2: the ids it returned
    were frequently near-misses or a neighbouring candidate's. Renumbering per batch
    replaces them with one- to three-digit ids that are still globally unique *within
    the prompt*, so a stray citation is still detectable as belonging to another
    candidate rather than silently accepted, and the mapping back to the real prop id
    is exact.
    """
    views: list[dict] = []
    next_id = 1
    for cand in batch:
        props = (evidence.get(cand["illustration_id"]) or {}).get("props") or []
        numbered = []
        for prop in props[:MAX_PROPS_SHOWN]:
            numbered.append((next_id, prop))
            next_id += 1
        views.append({"candidate": cand, "numbered": numbered})
    return views


def render_candidate(number: int, view: dict, evidence: dict) -> str:
    """One candidate block: name, printing, both layers, numbered propositions."""
    cand = view["candidate"]
    lines = [f"CANDIDATE {number} — {cand.get('name') or cand['oracle_id']}"]
    printing = f"  printing: {cand.get('set_code') or '?'} · art by {cand.get('artist') or 'unknown'}"
    art_count = int(cand.get("art_count") or 1)
    if art_count > 1:
        printing += f"  ({art_count} arts in the index — name this printing in your rationale)"
    lines.append(printing)
    lines.append(f"  literal: {evidence.get('literal') or '(none recorded)'}")
    lines.append(f"  interpretive: {evidence.get('interpretive') or '(none recorded)'}")

    numbered = view["numbered"]
    if not numbered:
        lines.append("  propositions: (none recorded)")
        lines.append(f"  CANDIDATE {number} may cite NO prop_ids at all.")
        return "\n".join(lines)

    low, high = numbered[0][0], numbered[-1][0]
    span = str(low) if low == high else f"{low}-{high}"
    lines.append(f"  propositions (CANDIDATE {number} may cite ONLY ids {span}):")
    for display_id, prop in numbered:
        lines.append(f"    [{display_id}] ({prop['layer']}) {prop['text']}")
    return "\n".join(lines)


def build_judge_prompt(query: str, views: list[dict], evidence: dict[str, dict]) -> str:
    blocks = [
        render_candidate(i + 1, view, evidence.get(view["candidate"]["illustration_id"], {}))
        for i, view in enumerate(views)
    ]
    return (
        f'THEME: "{query}"\n\n'
        "Score every candidate below on how well its artwork fits that theme.\n\n"
        f"{JUDGE_RULES}\n\n" + "\n\n".join(blocks)
    )


def _fallback_entry(cand: dict, reason: str) -> dict:
    """Unjudged: keep retrieval order, mark the fit null rather than faking a number."""
    return {
        **cand,
        "fit": None,
        "rationale": reason,
        "prop_ids": [],
        "judged": False,
    }


def judge_batch(cfg: Config, query: str, batch: list[dict], evidence: dict[str, dict]) -> list[dict]:
    """Judge up to BATCH_SIZE candidates in one call. Retries once, then degrades."""
    views = number_batch(batch, evidence)
    prompt = build_judge_prompt(query, views, evidence)
    last_error = ""
    for attempt in (1, 2):
        try:
            raw = ollama.generate(
                cfg,
                cfg.judge_model,
                prompt,
                system=JUDGE_SYSTEM,
                format=JUDGE_SCHEMA,
                options={"temperature": 0},
            )
            entries = parse_json(raw).get("judgments")
            if not isinstance(entries, list) or not entries:
                raise ValueError("no judgments array in response")
            return _apply_entries(views, entries)
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            last_error = str(exc)
            if attempt == 1:
                print(f"judge: batch failed ({exc}); retrying once", file=sys.stderr)
    print(
        f"judge: batch failed twice ({last_error}); keeping retrieval order for "
        f"{len(batch)} candidates",
        file=sys.stderr,
    )
    return [_fallback_entry(c, "not judged: judge model unavailable for this batch") for c in batch]


def _apply_entries(views: list[dict], entries: list) -> list[dict]:
    """Attach one model entry per candidate, validating fits and cited prop ids.

    Citations are resolved against the batch-local numbering, and the two ways of
    getting one wrong are counted separately: a number belonging to another candidate
    in the same prompt (misattributed) and a number that was never shown at all
    (invented). Both are dropped; only the counts differ, and they say different things
    about the model.
    """
    by_number: dict[int, dict] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        try:
            number = int(entry.get("candidate", position + 1))
        except (TypeError, ValueError):
            number = position + 1
        by_number.setdefault(number, entry)

    shown: dict[int, int] = {}
    for view in views:
        for display_id, prop in view["numbered"]:
            shown[display_id] = prop["id"]

    results: list[dict] = []
    for i, view in enumerate(views):
        cand = view["candidate"]
        entry = by_number.get(i + 1)
        if entry is None:
            results.append(_fallback_entry(cand, "not judged: missing from judge response"))
            continue

        try:
            fit = float(entry.get("fit"))
        except (TypeError, ValueError):
            fit = 0.0
        fit = min(1.0, max(0.0, fit))

        own = {display_id: prop["id"] for display_id, prop in view["numbered"]}
        cited: list[int] = []
        invented = misattributed = 0
        for value in entry.get("prop_ids") or []:
            try:
                display_id = int(value)
            except (TypeError, ValueError):
                invented += 1
                continue
            if display_id in own:
                cited.append(own[display_id])
            elif display_id in shown:
                misattributed += 1
            else:
                invented += 1

        rationale = " ".join(str(entry.get("rationale") or "").split())
        results.append(
            {
                **cand,
                "fit": fit,
                "rationale": rationale or "(no rationale returned)",
                "prop_ids": cited,
                "judged": True,
                "invented_prop_ids": invented + misattributed,
                "misattributed_prop_ids": misattributed,
                "cited_prop_ids": len(cited) + invented + misattributed,
            }
        )
    return results


def judge_batches(
    cfg: Config,
    conn: sqlite3.Connection,
    query: str,
    candidates: list[dict],
    *,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Judge the whole candidate pool in batches, preserving retrieval order."""
    if not candidates:
        return []
    evidence = load_evidence(conn, [c["illustration_id"] for c in candidates])
    judged: list[dict] = []
    for batch in chunks(candidates, batch_size):
        judged.extend(judge_batch(cfg, query, batch, evidence))
    dropped = sum(int(r.get("invented_prop_ids") or 0) for r in judged)
    misattributed = sum(int(r.get("misattributed_prop_ids") or 0) for r in judged)
    total = sum(int(r.get("cited_prop_ids") or 0) for r in judged)
    if dropped:
        rate = f" ({dropped / total:.1%} of {total} citations)" if total else ""
        print(
            f"judge: dropped {dropped} cited prop ids that did not belong{rate}"
            f" — {misattributed} belonged to another candidate, {dropped - misattributed} "
            "were never shown",
            file=sys.stderr,
        )
    return judged


def log_judgments(
    conn: sqlite3.Connection, query_id: int, model: str, judged: list[dict]
) -> None:
    """Every judged candidate, source='judge'. Phase 12 trains on these."""
    conn.executemany(
        "INSERT INTO judgments(query_id, illustration_id, fit, rationale, prop_ids, "
        "model, source) VALUES (?, ?, ?, ?, ?, ?, 'judge')",
        [
            (
                query_id,
                r["illustration_id"],
                r.get("fit"),
                r.get("rationale"),
                json.dumps(r.get("prop_ids") or []),
                model,
            )
            for r in judged
        ],
    )
    conn.commit()


# ------------------------------------------------------------------------ verification


def build_verify_prompt(query: str) -> str:
    return (
        f'THEME: "{query}"\n\n'
        "Look at this artwork and answer one question: does this theme genuinely hold "
        "for this image?\n"
        "Judge the image itself. Ignore any card, character, set or lore you recognize.\n"
        'Set "holds" to true only if someone who asked for this theme would accept this '
        "image as an answer.\n"
        '"why" is one short sentence pointing at what you actually see.'
    )


def verify_finalists(
    cfg: Config, judged: list[dict], query: str, *, top_n: int = VERIFY_TOP_N
) -> tuple[list[dict], bool]:
    """Re-check the best candidates against the real image. Returns (results, available).

    Uses `cfg.verify_model`, not `cfg.vision_model`. The two default to the same
    name, but they are different jobs: `vision_model` writes the exhaustive
    descriptions in Phase 5 and wants to be as large as the card allows, whereas
    this stage asks one binary "does this claim hold for this image" question per
    finalist. Pointing both at an 81 GB model means it cannot be co-resident with
    the judge, so every search swaps models in and out of VRAM ~17 times. See
    config.toml.

    A rejection removes the candidate from selection but stays in the pool, marked, so
    `--json` still shows what happened. If the verify model is unreachable the whole
    stage is skipped after the first failure — results stay unverified and the caller
    reports that rather than pretending they were checked.
    """
    prompt = build_verify_prompt(query)
    ordered = sorted(judged, key=fit_key, reverse=True)[:top_n]
    available = True

    for cand in ordered:
        art_path = cand.get("art_path")
        if not art_path or not Path(art_path).is_file():
            cand["verify_note"] = "no local art crop to verify against"
            continue
        try:
            raw = ollama.vision(
                cfg,
                cfg.verify_model,
                prompt,
                str(art_path),
                format=VERIFY_SCHEMA,
                options={"temperature": 0},
            )
        except (RuntimeError, requests.RequestException) as exc:
            # Infrastructure failure, not a verdict: stop calling and degrade.
            available = False
            print(
                f"verify: verify model unavailable ({exc}); keeping judge ordering "
                "and marking results unverified",
                file=sys.stderr,
            )
            break
        try:
            parsed = parse_json(raw)
        except ValueError as exc:
            cand["verify_note"] = f"unreadable verification response ({exc})"
            continue

        why = " ".join(str(parsed.get("why") or "").split())
        if bool(parsed.get("holds")):
            cand["verified"] = True
            cand["verify_note"] = why
        else:
            cand["vision_rejected"] = True
            cand["verify_note"] = why or "vision check rejected this image"

    return judged, available


def fit_key(result: dict) -> float:
    """Sort key that puts null fits (unjudged) below everything scored."""
    fit = result.get("fit")
    return -1.0 if fit is None else float(fit)


# --------------------------------------------------------------------------- diversity


def color_cap(results: list[dict], cap: int = COLOR_CAP) -> list[dict]:
    """Keep at most `cap` results per colour identity, best first."""
    seen: dict[str, int] = {}
    kept = []
    for r in results:
        ci = r.get("color_identity") or ""
        if seen.get(ci, 0) >= cap:
            continue
        seen[ci] = seen.get(ci, 0) + 1
        kept.append(r)
    return kept


def mmr(index: SearchIndex, results: list[dict], k: int, lam: float = MMR_LAMBDA) -> list[dict]:
    """Maximal marginal relevance over the matched artworks' mean prop embeddings.

    score = lam * fit - (1 - lam) * (highest similarity to anything already picked).
    Relevance still dominates at lam=0.7; the penalty only breaks up near-duplicates.
    """
    if len(results) <= k:
        return list(results)

    vectors = [index.mean_vector(r["illustration_id"]) for r in results]
    relevance = [0.0 if r.get("fit") is None else float(r["fit"]) for r in results]

    selected: list[int] = []
    remaining = list(range(len(results)))
    while remaining and len(selected) < k:
        best_i, best_score = remaining[0], -1e9
        for i in remaining:
            if selected:
                similarity = max(float(np.dot(vectors[i], vectors[j])) for j in selected)
            else:
                similarity = 0.0
            score = lam * relevance[i] - (1.0 - lam) * similarity
            if score > best_score:
                best_i, best_score = i, score
        selected.append(best_i)
        remaining.remove(best_i)
    return [results[i] for i in selected]


def diversify(
    index: SearchIndex, results: list[dict], k: int, *, lam: float = MMR_LAMBDA
) -> list[dict]:
    """Colour cap first, then MMR over the survivors, then take k."""
    capped = color_cap(results)
    return mmr(index, capped[:DIVERSITY_POOL], k, lam)
