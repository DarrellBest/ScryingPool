"""The precision stage of `/oracle`: score the retrieved pool against the
card's own rules text and cite the chunk(s) relied on.

Unlike `cts/judge.py`, there is **no verification stage** here and there
should not be one — the design doc's argument, restated because it is the
reason this module is smaller than its art-side counterpart: the art judge
reasons over propositions a vision model wrote about a picture, a lossy
intermediate that can lose a detail permanently, which is why
`verify_finalists` exists. This judge reasons over Wizards' own Oracle text,
verbatim, byte for byte as Scryfall publishes it. There is no intermediate to
be wrong, so a second model pass re-reading text the first pass already read
is cost with no new information.

Batch-local citation renumbering is reused **exactly** as `judge.number_batch`
established it (Defect 2, `tests/test_judge_props.py`): global `chunks.id`
values are large and models copy them wrong, so each batch gets its own
one-to-N numbering and a citation outside a candidate's own declared range is
dropped and counted, same mechanism, same reason, same test shape here.

Mechanical precision — the make-or-break risk
-----------------------------------------------
Retrieval's job is recall; precision is entirely this module's job, and the
danger is specific: draw / loot / rummage / impulse / surveil / scry / reveal
/ tutor use the same words in the same templating register about the same
zones ("draw", "discard", "top card", "library", "hand"), so cosine similarity
alone cannot and does not separate them. `JUDGE_RULES` below is the mechanism
that is supposed to. It is a prompt, so it is the softest of every mechanism
this design has and the easiest to silently break — which is exactly why
`test_oracle_judge.py` pins that every named mechanic is still spelled out in
it, verbatim, as a guard against a prompt edit that quietly drops two lines.
"""

from __future__ import annotations

import sqlite3
import sys

import requests

from . import judge as judge_mod
from . import ollama
from .config import Config

BATCH_SIZE = 10
MAX_CHUNKS_SHOWN = 20  # a card has a handful of abilities; this only bounds pathology

# num_ctx is not decoration — cts/prompts.py:189 already established that the
# hard way for the vision pass, and this module needs its own value for the
# same reason: leaving it unset does not mean "no limit," it means Ollama
# allocates a KV cache sized for the MODEL's own advertised context window
# (262,144 tokens for qwen3.6:latest), and that oversized allocation is what
# turned a batch that should take a few seconds into one that blew past a
# 300s client timeout — measured directly against this exact query via the
# live ollama journal: `n_ctx_slot = 262144` for a `task.n_tokens = 2153`
# prompt. A real 10-candidate batch (full oracle text + up to
# MAX_CHUNKS_SHOWN numbered ability lines per candidate + the mechanics
# rubric) measured 2,153 tokens; 8192 is ~4x headroom over that for the
# pathological case (ten simultaneously verbose Sagas/Classes all near the
# MAX_CHUNKS_SHOWN cap) without paying for anywhere near the full window.
JUDGE_NUM_CTX = 8192

PASS_FIT = judge_mod.PASS_FIT
fit_key = judge_mod.fit_key
chunks_of = judge_mod.chunks  # generic batching iterator, reused verbatim
parse_json = judge_mod.parse_json

JUDGE_SYSTEM = (
    "You are the precision stage of a rules-text search engine for Magic: The "
    "Gathering. You score how well each candidate CARD's actual Oracle text "
    "satisfies a mechanical query. You reason only over the rules text shown to "
    "you — never over the card's reputation, price, art, or anything you "
    "otherwise know about it."
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
                    "chunk_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["candidate", "fit", "rationale", "chunk_ids"],
            },
        }
    },
    "required": ["judgments"],
}

# The mechanics rubric. Every named family below is asserted present, verbatim,
# by test_oracle_judge.py's prompt-content test — see the module docstring.
MECHANICS_RUBRIC = """Distinguish these. They use the same words and mean different things:
  DRAW      cards move library -> hand. "draw a card", "draws N cards". This is a draw.
  LOOT      a draw with a discard attached in the same ability ("draw a card, then
            discard a card"). Say so in the rationale; it is a draw AND a discard,
            not pure card advantage.
  RUMMAGE   discard first, then draw ("discard a card, then draw a card"). The cost
            is paid before the card arrives — order matters, and it is still not
            pure card advantage.
  IMPULSE   "exile the top card ... you may play it this turn/until end of turn."
            The card NEVER enters the hand. This is NOT drawing. A query asking to
            draw is not satisfied by an impulse-draw effect, even though it moves a
            card out of the library.
  SURVEIL / SCRY   reorder or bin cards from the top of the library ("surveil N",
            "scry N"). No card is drawn, none moves to hand. Not a draw.
  REVEAL    "reveal the top card of your library and put it into your hand" IS a
            draw in all but name — score it as a draw. "...and put it on the
            bottom/into the graveyard" is NOT a draw.
  TUTOR     "search your library for a card ... and put it into your hand" is not
            drawing, though it answers a related question (card selection, not card
            advantage). Score it low for a draw query and say why in the rationale.
  WHEEL     "each player discards their hand, then draws seven" — draws, symmetrically;
            note in the rationale that it is symmetric when relevant to the query.
If an ability only draws (or otherwise satisfies the query) under a condition the
card's own text never actually establishes, say so and score it below 0.5."""

JUDGE_RULES = f"""Rules:
- fit is CONTINUOUS from 0.0 to 1.0, never a yes/no. 1.0 unmistakable, 0.7 strong,
  0.5 real but partial, 0.3 a stretch, 0.0 unrelated. A clear-cut mechanical query
  usually settles near 0 or 1; do not round off a genuine partial match.
- rationale is ONE sentence about this specific card's text, concrete, no hedging,
  and MUST name the mechanic (draw, loot, impulse, etc.) when the rubric below
  distinguishes it from a near-miss — that is what makes a wrong call checkable at
  a glance against the oracle text shown above it.
- chunk_ids are the bracketed numbers you actually relied on, copied exactly. Every
  candidate header states the only range of numbers that candidate may cite — a
  number outside its own range belongs to a different card and is always wrong.
  Copy digits, do not reconstruct them. If nothing in the text supports the query,
  return an empty list and a low fit.
- Judge ONLY the Oracle text shown. Do not credit a card for reputation, price,
  or lore you happen to know about it, and do not penalize a card for being
  unfamiliar.
- Return exactly one entry per candidate, reusing the candidate numbers as given.

{MECHANICS_RUBRIC}"""


# --------------------------------------------------------------------------- evidence


def load_evidence(conn: sqlite3.Connection, oracle_ids: list[str]) -> dict[str, dict]:
    """Full oracle text plus every ability chunk, per candidate card.

    The whole card's text is shown once, verbatim, ahead of the numbered
    chunks — the judge should be reading the real text, not reconstructing it
    from fragments, and the numbered lines exist only so it can cite exactly
    which part it relied on.
    """
    evidence: dict[str, dict] = {
        oid: {"name": "", "type_line": "", "oracle_text": "", "chunks": []} for oid in oracle_ids
    }
    for chunk in chunks_of(oracle_ids, 400):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT oracle_id, name, type_line, oracle_text FROM cards "
            f"WHERE oracle_id IN ({marks})",
            chunk,
        ):
            entry = evidence[row["oracle_id"]]
            entry["name"] = row["name"] or ""
            entry["type_line"] = row["type_line"] or ""
            entry["oracle_text"] = row["oracle_text"] or ""
        for row in conn.execute(
            f"SELECT id, oracle_id, face_index, ordinal, text FROM chunks "
            f"WHERE oracle_id IN ({marks}) AND kind = 'ability' "
            f"ORDER BY oracle_id, face_index, ordinal",
            chunk,
        ):
            evidence[row["oracle_id"]]["chunks"].append(
                {"id": int(row["id"]), "face_index": row["face_index"],
                 "ordinal": row["ordinal"], "text": row["text"]}
            )
    return evidence


# ---------------------------------------------------------------------------- judging


def number_batch(batch: list[dict], evidence: dict[str, dict]) -> list[dict]:
    """Give every ability chunk in one batch a short id, numbered from 1 —
    exactly `judge.number_batch`'s renumbering trick, one level down."""
    views: list[dict] = []
    next_id = 1
    for cand in batch:
        ability_chunks = (evidence.get(cand["oracle_id"]) or {}).get("chunks") or []
        numbered = []
        for chunk in ability_chunks[:MAX_CHUNKS_SHOWN]:
            numbered.append((next_id, chunk))
            next_id += 1
        views.append({"candidate": cand, "numbered": numbered})
    return views


def render_candidate(number: int, view: dict, evidence: dict) -> str:
    cand = view["candidate"]
    name = evidence.get("name") or cand.get("oracle_id", "?")
    lines = [f"CANDIDATE {number} — {name}"]
    lines.append(f"  type: {evidence.get('type_line') or '(unknown)'}")
    lines.append(f"  oracle text:\n    {evidence.get('oracle_text') or '(no rules text)'}")

    numbered = view["numbered"]
    if not numbered:
        lines.append(f"  CANDIDATE {number} may cite NO chunk_ids at all.")
        return "\n".join(lines)

    low, high = numbered[0][0], numbered[-1][0]
    span = str(low) if low == high else f"{low}-{high}"
    lines.append(f"  abilities (CANDIDATE {number} may cite ONLY ids {span}):")
    for display_id, chunk in numbered:
        lines.append(f"    [{display_id}] {chunk['text']}")
    return "\n".join(lines)


def build_judge_prompt(query: str, views: list[dict], evidence: dict[str, dict]) -> str:
    blocks = [
        render_candidate(i + 1, view, evidence.get(view["candidate"]["oracle_id"], {}))
        for i, view in enumerate(views)
    ]
    return (
        f'QUERY: "{query}"\n\n'
        "Score every candidate card below on how well its Oracle text satisfies "
        "the mechanical part of that query.\n\n"
        f"{JUDGE_RULES}\n\n" + "\n\n".join(blocks)
    )


def _fallback_entry(cand: dict, reason: str) -> dict:
    return {**cand, "fit": None, "rationale": reason, "chunk_ids": [], "judged": False}


def judge_batch(cfg: Config, query: str, batch: list[dict], evidence: dict[str, dict]) -> list[dict]:
    views = number_batch(batch, evidence)
    prompt = build_judge_prompt(query, views, evidence)
    last_error = ""
    for attempt in (1, 2):
        try:
            raw = ollama.generate(
                cfg, cfg.judge_model, prompt, system=JUDGE_SYSTEM, format=JUDGE_SCHEMA,
                options={"temperature": 0, "num_ctx": JUDGE_NUM_CTX},
            )
            entries = parse_json(raw).get("judgments")
            if not isinstance(entries, list) or not entries:
                raise ValueError("no judgments array in response")
            return _apply_entries(views, entries)
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            last_error = str(exc)
            if attempt == 1:
                print(f"oracle-judge: batch failed ({exc}); retrying once", file=sys.stderr)
    print(
        f"oracle-judge: batch failed twice ({last_error}); keeping retrieval order for "
        f"{len(batch)} candidates",
        file=sys.stderr,
    )
    return [_fallback_entry(c, "not judged: judge model unavailable for this batch") for c in batch]


def _apply_entries(views: list[dict], entries: list) -> list[dict]:
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
        for display_id, chunk in view["numbered"]:
            shown[display_id] = chunk["id"]

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

        own = {display_id: chunk["id"] for display_id, chunk in view["numbered"]}
        cited: list[int] = []
        invented = misattributed = 0
        for value in entry.get("chunk_ids") or []:
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
                "chunk_ids": cited,
                "judged": True,
                "invented_chunk_ids": invented + misattributed,
                "misattributed_chunk_ids": misattributed,
                "cited_chunk_ids": len(cited) + invented + misattributed,
            }
        )
    return results


def judge_batches(
    cfg: Config, conn: sqlite3.Connection, query: str, candidates: list[dict],
    *, batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Judge the whole candidate pool in batches, preserving retrieval order."""
    if not candidates:
        return []
    evidence = load_evidence(conn, [c["oracle_id"] for c in candidates])
    judged: list[dict] = []
    for batch in chunks_of(candidates, batch_size):
        judged.extend(judge_batch(cfg, query, batch, evidence))
    dropped = sum(int(r.get("invented_chunk_ids") or 0) for r in judged)
    misattributed = sum(int(r.get("misattributed_chunk_ids") or 0) for r in judged)
    total = sum(int(r.get("cited_chunk_ids") or 0) for r in judged)
    if dropped:
        rate = f" ({dropped / total:.1%} of {total} citations)" if total else ""
        print(
            f"oracle-judge: dropped {dropped} cited chunk ids that did not belong{rate}"
            f" — {misattributed} belonged to another candidate, {dropped - misattributed} "
            "were never shown",
            file=sys.stderr,
        )
    return judged
