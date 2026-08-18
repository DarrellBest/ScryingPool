"""The `/oracle` pipeline: route -> expand -> sql filter -> retrieve -> collapse
-> judge -> select. `python -m cts oracle "QUERY"` is the CLI entry point.

Deliberately smaller than `cts/search.py` in exactly the ways the design doc
argues for:

* **No layer blend.** Oracle text has one register (Wizards' templating), not
  a literal/interpretive split, so there is one expansion call, not two, and
  no `literal_weight`/`interpretive_weight` to combine.
* **No verification stage.** The corpus is already ground truth — see
  `cts/oracle_judge.py`'s module docstring.
* **No diversity beyond one-per-card**, which retrieval already guarantees by
  construction (a card contributes its single best-ranked chunk, never more).
  No MMR, no colour cap: a mechanical query has no visual-convention pull to
  break up, and "enchantments in green that draw" should return five green
  cards, not be capped away from them.
* **A structural-only fast path.** When the router finds no mechanical intent
  at all, there is nothing to retrieve or judge — the SQL answer *is* the
  answer, ordered by `edhrec_rank`, and it completes in well under a second.

Two guards run before anything is spent, matching the design doc: a query
that is itself a card name is pointed at `/search`; a query that reads as a
rules question is refused honestly rather than answered badly.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

from . import judge as judge_mod
from . import oracle_db, oracle_filters as ofilters, oracle_judge, oracle_names, ollama
from .config import Config
from .index import tokenize  # the one tokenizer, shared everywhere in the repo
from .oracle_filters import Filters
from .oracle_index import OracleIndex, load_index

RRF_K = 60
CHUNK_DEPTH = 200       # chunks inspected per (expansion, method) list
POOL_SIZE = 40          # candidates handed to the judge — 4 batches of 10
PASS_FIT = oracle_judge.PASS_FIT

# num_ctx, explicit for the same reason cts/oracle_judge.py::JUDGE_NUM_CTX is:
# an unset value does not mean "unlimited," it means Ollama allocates a KV
# cache sized for the model's own full advertised context window (262,144
# tokens for qwen3.6:latest), which is what turned one judge batch into a
# 300s+ timeout — see JUDGE_NUM_CTX's comment for the live measurement.
# ROUTER_PROMPT is the static template (~450 words, including the numeric
# calibration table) plus one query; EXPANSION_PROMPT is shorter still (~150
# words plus one short intent phrase). Both get their own small budget rather
# than reusing the judge's 8192, since a smaller context is also a smaller,
# cheaper allocation, and neither prompt is remotely close to needing more.
ROUTER_NUM_CTX = 4096
EXPANSION_NUM_CTX = 2048

RULES_QUESTION_PREFIXES = (
    "can i", "can you", "does", "do i", "do you", "how does", "how do",
    "what happens if", "what happens when", "when do i", "when does",
    "am i allowed to", "is it legal to", "am i able to",
)


# ------------------------------------------------------------------------- guards


def looks_like_rules_question(query: str) -> bool:
    """A query opening with "can I", "does", "how does", etc. is a rules
    question, not a card search — running the search anyway would produce
    five cards containing vaguely related words and present them as an
    answer. Pure function over a string, no model call, no I/O."""
    q = " ".join(str(query).strip().lower().split())
    return any(q.startswith(prefix) for prefix in RULES_QUESTION_PREFIXES)


def card_name_guard(name_index, query: str) -> oracle_names.Resolution | None:
    """L0-L2 only — the exact-ish layers. Restricted deliberately: the fuzzy
    layers would fire on genuine mechanical queries that merely sit near some
    card's name ("counter target spell" is two edits from *Counterspell*),
    and a guard that hijacks real queries is worse than no guard."""
    if name_index is None:
        return None
    resolution = oracle_names.resolve(name_index, query, max_layer=2)
    return resolution if resolution.resolved else None


# --------------------------------------------------------------------------- router

ROUTER_SYSTEM = (
    "You route natural-language card searches for a Magic: The Gathering rules-text "
    "search engine. A query usually has a structured half (card type, colour identity, "
    "mana value, format legality) and a mechanical half (what the card actually does). "
    "Your job is to pull out every structured constraint you can find and leave "
    "everything else as a clean mechanical-intent phrase."
)

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "types": {"type": "array", "items": {"type": "string"}},
        "colors": {"type": "string"},
        "legal": {"type": "array", "items": {"type": "string"}},
        "mv_op": {"type": "string", "enum": ["", "<", "<=", "=", ">=", ">", "between"]},
        "mv_value": {"type": "number"},
        "mv_lo": {"type": "number"},
        "mv_hi": {"type": "number"},
        "semantic_intent": {"type": "string"},
        "vague_quantity_note": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    # mv_value/mv_lo/mv_hi are REQUIRED, not merely present in the schema:
    # measured live against this exact router (qwen3.6:latest) — when they
    # were optional, the model correctly chose mv_op="<=" for "cost 5 or
    # less" and then silently omitted mv_value entirely, even though the
    # prompt already tells it to "leave the numbers 0" when unused. An
    # optional numeric field is exactly the silent-failure shape the whole
    # numeric-filter section of the design doc exists to prevent: `mv_op`
    # alone parsed as non-empty, `router_mv_predicate` then received
    # `value=None`, the 0-30 guard's `is None` check returned None with no
    # note, and a 7-mana card passed a "5 or less" filter with nothing on
    # screen to say why. Requiring the field forces the model to write an
    # actual number (0 when genuinely unused, per the prompt's own
    # instruction), which is a value `_guard` can evaluate instead of a
    # missing key `_to_float` silently turns into None.
    "required": ["types", "colors", "legal", "mv_op", "mv_value", "mv_lo", "mv_hi",
                 "semantic_intent"],
}

ROUTER_PROMPT = """QUERY: "{query}"

Split the query into its structured half and its mechanical half.

types — zero or more Magic card types/supertypes/subtypes the query explicitly
names (e.g. "enchantment", "artifact", "creature", "planeswalker", "instant",
"legendary"). Lowercase. Empty list if the query names none.

colors — a string of letters from WUBRG if the query names a colour-identity
constraint ("green" -> "G", "Selesnya" -> "GW"), else "". This is a SUBSET test:
"in green" means the card's colour identity fits inside {{G}}, so mono-green and
colourless both match — never invert this into "contains every one of these
colours".

legal — zero or more format names the query names ("commander", "modern",
"pauper"), lowercase, else empty list.

mv_op / mv_value / mv_lo / mv_hi — a numeric mana-value constraint, using this
EXACT calibration:
  "5 or less" / "no more than 5" / "up to 5" / "at most 5" / "5 and under"  -> op "<=", value 5
  "under 5" / "less than 5" / "below 5"                                     -> op "<",  value 5
  "5 or more" / "at least 5" / "5 and up"                                   -> op ">=", value 5
  "over 5" / "more than 5" / "above 5"                                      -> op ">",  value 5
  "exactly two" / "two-drop" / "costs 3"                                    -> op "=",  value N
  "three to five" / "between 3 and 5"                                       -> op "between", lo 3, hi 5
"<=" and "<" are one word apart in English and mean a whole mana value's
difference — read the phrase carefully before choosing.
If the query names NO numeric cost at all, set mv_op to "" and leave the numbers 0.
If the query uses a VAGUE quantity word ("cheap", "expensive", "big", "small",
"low to the ground", "top-end") that has no defined mana value, you MUST ALSO set
mv_op to "" — do not guess a number — and put a short note about the word and
what filter to use instead in vague_quantity_note.

semantic_intent — the mechanical intent left over after removing the structured
parts above, in a few plain words ("that let me draw", "that make tokens").
Empty string "" if the query has NO mechanical intent at all — i.e. it is
entirely structural ("green enchantments costing 5 or less" has no verb and
gets "").

reasoning — one short sentence naming where the query's meaning lives."""


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def route(cfg: Config, query: str) -> dict:
    """One judge_model call -> {filters, semantic_intent, notes}. Never raises."""
    notes: list[str] = []
    last_error = ""
    for attempt in (1, 2):
        try:
            raw = ollama.generate(
                cfg, cfg.judge_model, ROUTER_PROMPT.format(query=query),
                system=ROUTER_SYSTEM, format=ROUTER_SCHEMA,
                options={"temperature": 0, "num_ctx": ROUTER_NUM_CTX},
            )
            parsed = judge_mod.parse_json(raw)
            types = tuple(
                str(t).strip().lower() for t in (parsed.get("types") or []) if str(t).strip()
            )
            colors_raw = str(parsed.get("colors") or "").strip().upper()
            colors = "".join(c for c in "WUBRG" if c in colors_raw) or None
            legal = tuple(
                str(t).strip().lower() for t in (parsed.get("legal") or []) if str(t).strip()
            )
            mv_op = str(parsed.get("mv_op") or "").strip()
            if mv_op not in ofilters.MV_OPS:
                mv_op = None
            vague_note = str(parsed.get("vague_quantity_note") or "").strip()
            if vague_note:
                notes.append(vague_note)
            semantic_intent = " ".join(str(parsed.get("semantic_intent") or "").split())
            filters = Filters(
                types=types, colors=colors, legal=legal, mv_op=mv_op,
                mv_value=_to_float(parsed.get("mv_value")),
                mv_lo=_to_float(parsed.get("mv_lo")), mv_hi=_to_float(parsed.get("mv_hi")),
            )
            return {
                "filters": filters,
                "semantic_intent": semantic_intent,
                "reasoning": str(parsed.get("reasoning") or "").strip(),
                "router_ok": True,
                "notes": notes,
            }
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            last_error = str(exc)
            if attempt == 1:
                print(f"oracle-route: router call failed ({exc}); retrying once", file=sys.stderr)

    print(f"oracle-route: router unavailable ({last_error}); no filters, full corpus", file=sys.stderr)
    notes.append(f"query routing failed ({last_error}); searched the whole corpus with no filters")
    return {
        "filters": Filters(), "semantic_intent": query, "reasoning": "",
        "router_ok": False, "notes": notes,
    }


# ------------------------------------------------------------------------- expansion

EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {"expansions": {"type": "array", "items": {"type": "string"}}},
    "required": ["expansions"],
}

EXPANSION_SYSTEM = (
    "You rewrite a mechanical Magic: The Gathering query into the exact phrasings "
    "Wizards' own Oracle-text templating uses. You never invent a mechanic the "
    "query did not ask for."
)

EXPANSION_PROMPT = """INTENT: "{intent}"

Write 6 to 8 short phrasings, in Magic's own Oracle-text templating register, that a
card satisfying this intent would actually contain. Bridge plain English to the exact
words the rules text uses.

Example, for the intent "let me draw":
  "draw a card"
  "draws two cards"
  "draw a card for each ..."
  "draw cards equal to ..."
  "you may draw a card"
  "target player draws a card"

Rules:
- Each phrasing stands alone and is independently meaningful.
- Stay faithful to the intent; do not drift to an adjacent mechanic (a "draw"
  intent must not produce scry/surveil/impulse phrasings — those are different
  mechanics, see the judge's own rubric).
- Vary the phrasing; near-duplicates of each other are wasted."""


def expand(cfg: Config, semantic_intent: str, notes: list[str]) -> list[str]:
    """One expansion call, only when there is a mechanical intent to expand.
    Always includes the intent itself, so a call failure still leaves one
    real query to retrieve on."""
    if not semantic_intent:
        return []
    cleaned: list[str] = []
    try:
        raw = ollama.generate(
            cfg, cfg.judge_model, EXPANSION_PROMPT.format(intent=semantic_intent),
            system=EXPANSION_SYSTEM, format=EXPANSION_SCHEMA,
            options={"temperature": 0.3, "num_ctx": EXPANSION_NUM_CTX},
        )
        items = judge_mod.parse_json(raw).get("expansions")
        cleaned = [" ".join(str(i).split()) for i in (items or []) if str(i).strip()][:8]
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        print(f"oracle-expand: expansion failed ({exc}); using the intent itself", file=sys.stderr)
        notes.append(f"expansion failed ({exc}); used the semantic intent verbatim")

    seen: set[str] = set()
    out: list[str] = []
    for text in [semantic_intent, *cleaned]:
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


# -------------------------------------------------------------------------- retrieval


def _query_vectors(cfg: Config, texts: list[str], notes: list[str]) -> np.ndarray | None:
    try:
        vecs = ollama.embed(cfg, texts)
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        print(f"oracle-retrieve: embedding failed ({exc}); BM25 only", file=sys.stderr)
        notes.append(f"dense retrieval unavailable ({exc}); ranked on BM25 alone")
        return None
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def retrieve(
    index: OracleIndex, expansions: list[str], query_vecs: np.ndarray | None,
    allowed: set[str] | None,
) -> tuple[dict[str, float], dict[str, int]]:
    """Rank each (expansion, method) chunk list, RRF-fuse into one score PER
    CARD. Never a sum: within one ranked list only a card's best-ranked chunk
    contributes, exactly once — collapsing to best-chunk-per-card happens
    here, at the only level oracle chunking has (there is no artwork
    intermediate the way the art side has propositions -> artworks -> cards).

    Returns `(fused, best_chunk_row)`: the RRF score per `oracle_id`, and the
    row index of the single best-ranked chunk that earned it — used to mark
    the matched line in the rendered oracle text.
    """
    fused: dict[str, float] = {}
    best_chunk_row: dict[str, int] = {}
    best_rank: dict[str, int] = {}

    if not len(index):
        return fused, best_chunk_row

    rows_all = np.arange(len(index))
    if allowed is not None:
        mask = np.fromiter((oid in allowed for oid in index.oracle_ids), dtype=bool, count=len(index))
        rows_all = rows_all[mask]
        if rows_all.size == 0:
            return fused, best_chunk_row

    for position, text in enumerate(expansions):
        lists: list[tuple[str, np.ndarray]] = []
        if query_vecs is not None:
            lists.append(("dense", (index.vecs @ query_vecs[position])[rows_all]))
        if index.bm25 is not None:
            tokens = tokenize(text)
            if tokens:
                lists.append(("bm25", index.bm25.get_scores(tokens)[rows_all]))

        for method, scores in lists:
            if method == "bm25":
                keep = scores > 0
                candidate_rows, candidate_scores = rows_all[keep], scores[keep]
            else:
                candidate_rows, candidate_scores = rows_all, scores
            if candidate_rows.size == 0:
                continue

            order = np.argsort(-candidate_scores)[:CHUNK_DEPTH]
            seen: set[str] = set()
            for rank, offset in enumerate(order, start=1):
                row = int(candidate_rows[offset])
                oid = index.oracle_ids[row]
                if oid in seen:
                    continue  # this card's best-ranked chunk in THIS list, once
                seen.add(oid)
                fused[oid] = fused.get(oid, 0.0) + 1.0 / (RRF_K + rank)
                if rank < best_rank.get(oid, 1 << 30):
                    best_rank[oid] = rank
                    best_chunk_row[oid] = row

    return fused, best_chunk_row


def candidate_rows(conn: sqlite3.Connection, oracle_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    sql = (
        "SELECT oracle_id, name, type_line, oracle_text, mana_cost, cmc, color_identity, "
        "edhrec_rank, scryfall_uri, set_code, rarity, released_at FROM cards "
        "WHERE oracle_id IN ({marks})"
    )
    for chunk in judge_mod.chunks(oracle_ids, 400):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(sql.format(marks=marks), chunk):
            out[row["oracle_id"]] = dict(row)
    return out


def legalities_for(conn: sqlite3.Connection, oracle_ids: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {oid: {} for oid in oracle_ids}
    for chunk in judge_mod.chunks(oracle_ids, 400):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT oracle_id, format, status FROM card_legalities WHERE oracle_id IN ({marks})",
            chunk,
        ):
            out[row["oracle_id"]][row["format"]] = row["status"]
    return out


def collapse(
    fused: dict[str, float], rows: dict[str, dict], index: OracleIndex,
    best_chunk_row: dict[str, int],
) -> list[dict]:
    """Score order, joined with the card row and its matched-chunk location."""
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    out: list[dict] = []
    for oid, score in ordered:
        row = rows.get(oid)
        if row is None:
            continue  # card row vanished between index build and query
        chunk_row = best_chunk_row.get(oid)
        entry = {**row, "score": score}
        if chunk_row is not None:
            entry["matched_chunk_id"] = index.chunk_ids[chunk_row]
            entry["matched_face_index"] = index.face_indices[chunk_row]
            entry["matched_ordinal"] = index.ordinals[chunk_row]
            entry["matched_kind"] = index.kinds[chunk_row]
        out.append(entry)
    return out


# ---------------------------------------------------------------------------- select


def _passes(result: dict) -> bool:
    fit = result.get("fit")
    return fit is not None and fit >= PASS_FIT


def select(judged: list[dict], k: int) -> list[dict]:
    """Sort by fit, take k. No MMR, no colour cap: one-per-card is already
    guaranteed by `retrieve`'s collapse, and a mechanical query has no visual
    convention to break up — five green cards that all draw is the correct
    answer to "green cards that draw", not a redundancy."""
    return sorted(judged, key=oracle_judge.fit_key, reverse=True)[:k]


def _result_dict(cand: dict, legalities: dict[str, str]) -> dict:
    return {
        "oracle_id": cand["oracle_id"],
        "name": cand.get("name"),
        "mana_cost": cand.get("mana_cost"),
        "type_line": cand.get("type_line"),
        "oracle_text": cand.get("oracle_text"),
        "cmc": cand.get("cmc"),
        "color_identity": cand.get("color_identity") or "",
        "set_code": cand.get("set_code"),
        "rarity": cand.get("rarity"),
        "released_at": cand.get("released_at"),
        "scryfall_uri": cand.get("scryfall_uri"),
        "legalities": legalities,
        "fit": cand.get("fit"),
        "rationale": cand.get("rationale"),
        "chunk_ids": cand.get("chunk_ids") or [],
        "matched_face_index": cand.get("matched_face_index"),
        "matched_ordinal": cand.get("matched_ordinal"),
        "score": float(cand.get("score") or 0.0),
        "stretch": not _passes(cand),
        "judged": bool(cand.get("judged")),
    }


# --------------------------------------------------------------------------- logging


def _log_query(conn: sqlite3.Connection, query: str, kind: str, plan: dict) -> int:
    cur = conn.execute(
        "INSERT INTO queries(text, kind, params, created_at) VALUES (?, ?, ?, ?)",
        (query, kind, json.dumps(plan, default=str), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return int(cur.lastrowid)


def _log_judgments(conn: sqlite3.Connection, query_id: int, model: str, judged: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO judgments(query_id, oracle_id, fit, rationale, chunk_ids, model, source) "
        "VALUES (?, ?, ?, ?, ?, ?, 'judge')",
        [
            (query_id, r["oracle_id"], r.get("fit"), r.get("rationale"),
             json.dumps(r.get("chunk_ids") or []), model)
            for r in judged
        ],
    )
    conn.commit()


# --------------------------------------------------------------------------- pipeline


def _merge_for_echo(hard: Filters, soft: Filters) -> Filters:
    """Explicit fields win over router-inferred ones for display: if the user
    typed `colors:G`, the echo shows that, not whatever the router separately
    guessed about the same field."""
    return Filters(
        types=hard.types or soft.types,
        colors=hard.colors or soft.colors,
        legal=hard.legal or soft.legal,
        mv_min=hard.mv_min,
        mv_max=hard.mv_max,
        mv_op=None if (hard.mv_min is not None or hard.mv_max is not None) else soft.mv_op,
        mv_value=soft.mv_value, mv_lo=soft.mv_lo, mv_hi=soft.mv_hi,
    )


def execute(
    cfg: Config,
    query: str,
    *,
    types: tuple[str, ...] = (),
    colors: str | None = None,
    mv_min: float | None = None,
    mv_max: float | None = None,
    legal: tuple[str, ...] = (),
    k: int = 5,
    kind: str = "user",
    conn: sqlite3.Connection | None = None,
    index: OracleIndex | None = None,
    name_index=None,
) -> dict:
    """Guards -> hard filter -> route -> soft filter -> [structural fast path
    | expand -> retrieve -> collapse -> judge] -> select."""
    started = time.perf_counter()
    own_conn = conn is None
    conn = conn or oracle_db.connect(cfg)
    if index is None:
        index = load_index(cfg, conn)

    try:
        # -------------------------------------------------------- guard 1: card name
        hit = card_name_guard(name_index, query)
        if hit is not None:
            card = oracle_names.card_payload(conn, hit.oracle_id) if hit.oracle_id else None
            name = (card or {}).get("name") or query
            return {
                "guard": "card_name",
                "query": query,
                "message": (
                    f'"{name}" is a card name. Try /search {name} for the card itself — '
                    "/oracle searches for cards by what they do."
                ),
                "results": [], "pool": [], "plan": {"notes": []},
            }

        # --------------------------------------------------- guard 2: rules question
        if looks_like_rules_question(query):
            return {
                "guard": "rules_question",
                "query": query,
                "message": (
                    "That reads as a rules question. /oracle searches card text — it "
                    "does not answer rules questions. Try the Comprehensive Rules or a judge."
                ),
                "results": [], "pool": [], "plan": {"notes": []},
            }

        total_cards = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]

        # ------------------------------------------------------------- hard filters
        hard = Filters(
            types=tuple(types or ()), colors=colors, legal=tuple(legal or ()),
            mv_min=mv_min, mv_max=mv_max,
        )
        notes: list[str] = []
        hard_ids = ofilters.compile_hard(conn, hard, notes)

        if hard_ids is not None and len(hard_ids) == 0:
            echo = ofilters.echo_line(hard, None)
            plan = {
                "filters": hard.__dict__, "semantic_intent": "", "notes": notes,
                "echo": echo, "scryfall_url": ofilters.scryfall_url(hard),
                "structural_only": True,
                "counts": {"filtered": 0, "total_cards": total_cards, "judged": 0},
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            return {
                "query_id": _log_query(conn, query, kind, plan),
                "plan": plan,
                "message": (
                    f"no cards match {echo.replace('filters: ', '').split(' · semantic', 1)[0]}. "
                    f"That combination has 0 cards in ~{total_cards:,} paper cards."
                ),
                "results": [], "pool": [],
            }

        # ---------------------------------------------------------------------- route
        routed = route(cfg, query)
        notes.extend(routed["notes"])
        soft = routed["filters"]
        semantic_intent = routed["semantic_intent"]

        allowed = ofilters.compile_soft(conn, soft, notes, base=hard_ids)
        effective = _merge_for_echo(hard, soft)
        echo = ofilters.echo_line(effective, semantic_intent or None)
        scryfall_url = ofilters.scryfall_url(effective)

        plan: dict = {
            "filters": {**hard.__dict__, **{k2: v for k2, v in soft.__dict__.items() if v}},
            "semantic_intent": semantic_intent,
            "notes": notes,
            "echo": echo,
            "scryfall_url": scryfall_url,
            "router_ok": routed["router_ok"],
            "models": {"judge": cfg.judge_model, "embed": cfg.embed_model},
            "index": {
                "chunks": len(index), "cards": index.card_count, "dim": index.dim,
                "build_seconds": round(index.build_seconds, 4),
                "missing_embeddings": index.missing_embeddings,
            },
        }

        if allowed is not None and len(allowed) == 0:
            plan["structural_only"] = False
            plan["counts"] = {"filtered": 0, "total_cards": total_cards, "judged": 0}
            plan["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            return {
                "query_id": _log_query(conn, query, kind, plan),
                "plan": plan, "results": [], "pool": [],
                "message": f"no cards passed the filters out of ~{total_cards:,} paper cards.",
            }

        pool_size = len(allowed) if allowed is not None else total_cards

        # -------------------------------------------------- structural-only fast path
        if not semantic_intent:
            plan["structural_only"] = True
            rows_sql = (
                "SELECT oracle_id, name, type_line, oracle_text, mana_cost, cmc, "
                "color_identity, edhrec_rank, scryfall_uri, set_code, rarity, released_at "
                "FROM cards"
            )
            params: list = []
            if allowed is not None:
                marks = ",".join("?" * len(allowed))
                rows_sql += f" WHERE oracle_id IN ({marks})"
                params = list(allowed)
            rows_sql += " ORDER BY (edhrec_rank IS NULL), edhrec_rank ASC LIMIT ?"
            params.append(k)
            top = [dict(r) for r in conn.execute(rows_sql, params)]
            legalities = legalities_for(conn, [r["oracle_id"] for r in top])
            results = [_result_dict({**r, "fit": None, "chunk_ids": []}, legalities[r["oracle_id"]])
                       for r in top]
            for r in results:
                r["stretch"] = False  # not judged at all; not a "below the bar" stretch
            plan["counts"] = {"filtered": pool_size, "total_cards": total_cards, "judged": 0}
            plan["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            return {
                "query_id": _log_query(conn, query, kind, plan),
                "plan": plan, "results": results, "pool": results,
                "message": (
                    f"no mechanical intent in this query — these are {pool_size:,} cards "
                    "matching the filters, most-played first. Nothing was judged."
                ),
            }

        # --------------------------------------------------------------- semantic path
        expansions = expand(cfg, semantic_intent, notes)
        query_vecs = _query_vectors(cfg, expansions, notes) if expansions else None
        fused, best_chunk_row = retrieve(index, expansions, query_vecs, allowed)

        rows = candidate_rows(conn, list(fused))
        collapsed = collapse(fused, rows, index, best_chunk_row)[:POOL_SIZE]

        plan["counts"] = {
            "filtered": pool_size, "total_cards": total_cards,
            "retrieved": len(fused), "candidates": len(collapsed),
        }

        judged = oracle_judge.judge_batches(cfg, conn, query, collapsed)
        query_id = _log_query(conn, query, kind, plan)
        _log_judgments(conn, query_id, cfg.judge_model, judged)

        legalities = legalities_for(conn, [j["oracle_id"] for j in judged])
        chosen = select(judged, k)
        results = [_result_dict(c, legalities[c["oracle_id"]]) for c in chosen]
        pool = [
            _result_dict(c, legalities[c["oracle_id"]])
            for c in sorted(judged, key=oracle_judge.fit_key, reverse=True)
        ]

        plan["counts"]["judged"] = len(judged)
        plan["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        conn.execute(
            "UPDATE queries SET params = ? WHERE id = ?",
            (json.dumps(plan, default=str), query_id),
        )
        conn.commit()

        passing = sum(1 for r in results if not r["stretch"])
        message = (
            f"{pool_size:,} cards passed the filters · {len(judged)} judged · "
            f"{passing} of {len(results)} clear the {PASS_FIT} fit bar"
        )
        return {"query_id": query_id, "plan": plan, "results": results, "pool": pool,
                "message": message}
    finally:
        if own_conn:
            conn.close()


# --------------------------------------------------------------------------------- CLI


def run(
    cfg: Config, query: str, *, types: tuple[str, ...] = (), colors: str | None = None,
    mv_min: float | None = None, mv_max: float | None = None, legal: tuple[str, ...] = (),
    k: int = 5, as_json: bool = False,
) -> None:
    outcome = execute(
        cfg, query, types=types, colors=colors, mv_min=mv_min, mv_max=mv_max, legal=legal, k=k,
    )

    if as_json:
        print(json.dumps(outcome, indent=2, default=str))
        return

    if outcome.get("guard"):
        print(outcome["message"])
        return

    plan = outcome["plan"]
    print(f'🔮 "{query}"')
    print(plan.get("echo", ""))
    if outcome.get("message"):
        print(outcome["message"])
    for note in plan.get("notes", []):
        print(f"note: {note}")
    if plan.get("scryfall_url"):
        print(f"refine on Scryfall: {plan['scryfall_url']}")

    results = outcome["results"]
    if not results:
        return

    print()
    for i, r in enumerate(results, start=1):
        bits = [f"{i}. {r['name']}"]
        if r.get("mana_cost"):
            bits.append(r["mana_cost"])
        fit = r.get("fit")
        bits.append("fit —" if fit is None else f"fit {fit:.2f}")
        if r["stretch"]:
            bits.append("STRETCH")
        print("  ".join(bits))
        print(f"   {r.get('type_line') or ''}")
        text = str(r.get("oracle_text") or "").strip()
        if text:
            print("   " + text.replace("\n", "\n   "))
        if r.get("rationale"):
            print(f"   -> {r['rationale']}")
        print()
