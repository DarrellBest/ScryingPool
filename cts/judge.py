"""Phase 8: judge the retrieved pool, verify the finalists with eyes, diversify.

Retrieval optimizes recall; everything here is precision. Three stages:

1. `judge_batches` — the judge model scores each candidate 0-1 against the *matched
   artwork's* two description layers and its propositions, citing the prop ids it
   relied on. A rationale that cannot point at a proposition is one the model invented,
   so invented ids are dropped and counted.
2. `verify_finalists` — the top eight go back to the vision model against the actual
   art crop. Everything upstream reasons over a description written by another model
   at another time; this is the only stage that can catch a detail the vision pass
   compressed away.
3. `diversify` — colour cap plus MMR, so a theme that attracts one visual convention
   does not return five of it.

Every stage degrades instead of crashing: a failed batch falls back to retrieval
order with a null fit, and an unreachable vision model leaves results unverified and
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
- prop_ids are the bracketed numbers from THIS candidate's propositions, and only the
  ones you actually relied on. Never invent an id, never cite another candidate's. If
  nothing in the propositions supports the theme, return an empty list and a low fit.
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


def render_candidate(number: int, cand: dict, evidence: dict) -> str:
    """One candidate block: name, printing, both layers, numbered propositions."""
    lines = [f"CANDIDATE {number} — {cand.get('name') or cand['oracle_id']}"]
    printing = f"  printing: {cand.get('set_code') or '?'} · art by {cand.get('artist') or 'unknown'}"
    art_count = int(cand.get("art_count") or 1)
    if art_count > 1:
        printing += f"  ({art_count} arts in the index — name this printing in your rationale)"
    lines.append(printing)
    lines.append(f"  literal: {evidence.get('literal') or '(none recorded)'}")
    lines.append(f"  interpretive: {evidence.get('interpretive') or '(none recorded)'}")
    lines.append("  propositions:")
    props = evidence.get("props") or []
    if not props:
        lines.append("    (none recorded)")
    for prop in props[:MAX_PROPS_SHOWN]:
        lines.append(f"    [{prop['id']}] ({prop['layer']}) {prop['text']}")
    return "\n".join(lines)


def build_judge_prompt(query: str, batch: list[dict], evidence: dict[str, dict]) -> str:
    blocks = [
        render_candidate(i + 1, cand, evidence.get(cand["illustration_id"], {}))
        for i, cand in enumerate(batch)
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
    prompt = build_judge_prompt(query, batch, evidence)
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
            return _apply_entries(batch, evidence, entries)
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


def _apply_entries(batch: list[dict], evidence: dict[str, dict], entries: list) -> list[dict]:
    """Attach one model entry per candidate, validating fits and cited prop ids."""
    by_number: dict[int, dict] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        try:
            number = int(entry.get("candidate", position + 1))
        except (TypeError, ValueError):
            number = position + 1
        by_number.setdefault(number, entry)

    results: list[dict] = []
    for i, cand in enumerate(batch):
        entry = by_number.get(i + 1)
        if entry is None:
            results.append(_fallback_entry(cand, "not judged: missing from judge response"))
            continue

        try:
            fit = float(entry.get("fit"))
        except (TypeError, ValueError):
            fit = 0.0
        fit = min(1.0, max(0.0, fit))

        # A cited id that does not belong to this candidate is a confabulation; drop it.
        own = {p["id"] for p in evidence.get(cand["illustration_id"], {}).get("props", [])}
        cited, invented = [], 0
        for value in entry.get("prop_ids") or []:
            try:
                pid = int(value)
            except (TypeError, ValueError):
                invented += 1
                continue
            if pid in own:
                cited.append(pid)
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
                "invented_prop_ids": invented,
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
    invented = sum(int(r.get("invented_prop_ids") or 0) for r in judged)
    if invented:
        print(f"judge: dropped {invented} cited prop ids that did not belong", file=sys.stderr)
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

    A rejection removes the candidate from selection but stays in the pool, marked, so
    `--json` still shows what happened. If the vision model is unreachable the whole
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
                cfg.vision_model,
                prompt,
                str(art_path),
                format=VERIFY_SCHEMA,
                options={"temperature": 0},
            )
        except (RuntimeError, requests.RequestException) as exc:
            # Infrastructure failure, not a verdict: stop calling and degrade.
            available = False
            print(
                f"verify: vision model unavailable ({exc}); keeping judge ordering "
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
