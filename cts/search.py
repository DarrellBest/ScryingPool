"""Phases 7 and 9: route, expand two ways, retrieve, collapse, judge, print.

The shape of this module follows from one fact stated at the top of SPEC.md: a single
literal description layer cannot serve both ends of the theme spectrum. "A hooded
figure stands alone on a cliff at dusk" contains every scrap of evidence for "lonely"
and not one word that embeds near it. So every query is routed to a *blend* of the two
layers, and anything with interpretive weight runs two independent expansions — one
that restates the theme in the interpretive register, one that asks what a matching
image would physically contain. Both always. The decomposed route is what makes a
genuinely novel abstract theme work, because it does not require the vision pass to
have anticipated the concept, only to have recorded the evidence.

Diagnostics go to stderr so `--json` stays pipeable.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

from . import db, judge as judge_mod, ollama
from .config import Config
from .index import SearchIndex, load_index, tokenize
from .links import links_for

RRF_K = 60              # SPEC.md: 1/(60 + rank)
PROP_DEPTH = 200        # propositions inspected per (expansion, method) list
POOL_SIZE = 100         # candidates handed to the judge
PER_METHOD_LOG = 50     # per-method retrieval rows logged for Phase 12
MIN_ROUTE_WEIGHT = 0.05  # a route we still run never contributes exactly nothing
WUBRG = "WUBRG"

# ----------------------------------------------------------------------------- router

SLOT_PATHS = (
    "primary_subject.species",
    "primary_subject.facial_hair",
    "primary_subject.clothing",
    "primary_subject.pose",
    "primary_subject.held_objects",
    "other_figures",
    "figure_count",
    "setting",
    "time_of_day",
    "palette",
    "art_style",
    "composition",
)

SLOT_OPS = ("equals", "contains", "not_contains", "gte", "lte")

ROUTER_SYSTEM = (
    "You route theme queries for an art search engine over Magic: The Gathering "
    "commander artwork. Every artwork is described in two separate layers: a LITERAL "
    "layer holding only what is physically visible (figures, objects, poses, palette, "
    "setting, composition) and an INTERPRETIVE layer holding mood, implied narrative, "
    "power dynamics, genre register and analogies to film, music or period art. Your "
    "job is to say how much of the query's meaning lives in each layer, and to pull out "
    "any constraint that can be checked structurally."
)

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "literal_weight": {"type": "number", "minimum": 0, "maximum": 1},
        "interpretive_weight": {"type": "number", "minimum": 0, "maximum": 1},
        "slot_filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "enum": list(SLOT_PATHS)},
                    "op": {"type": "string", "enum": list(SLOT_OPS)},
                    "value": {"type": "string"},
                },
                "required": ["path", "op", "value"],
            },
        },
        "mechanical_terms": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["literal_weight", "interpretive_weight", "slot_filters", "mechanical_terms"],
}

ROUTER_PROMPT = """QUERY: "{query}"

Return:

literal_weight and interpretive_weight — two numbers in [0,1] that SUM TO 1. This is a
blend, not a choice: never force the query into one layer. Calibration:
  "commanders with beards"                      -> 1.0 / 0.0
  "holding something that isn't a weapon"        -> 0.9 / 0.1
  "a single figure against a huge empty background" -> 0.7 / 0.3
  "painterly, muted, almost watercolor"          -> 0.6 / 0.4
  "menacing dragons with beards"                 -> 0.5 / 0.5
  "commanders that look lonely"                  -> 0.3 / 0.7
  "the moment right before a betrayal"           -> 0.25 / 0.75
  "would fit on a black metal album cover"       -> 0.2 / 0.8

slot_filters — zero or more hard constraints on the literal layer's structured slots.
Available paths: {paths}
Ops: equals, contains, not_contains, gte, lte (gte/lte only on figure_count).
held_objects, other_figures and palette are lists: use contains.
Only add a filter when the query states a requirement that is certain to be recorded in
that slot, and prefer no filter at all over a shaky one — the retriever already searches
the full text of every literal statement, while a wrong filter deletes correct answers
outright.
A slot has three possible states, not two: the attribute itself ("full grey beard"),
the literal string "none" when it is genuinely absent, and "obscured (...)" when the
image hides it. So "must have X" is `contains X`, and "must not have X" is
`not_contains X` — which keeps "none" and "obscured" rows, correctly, since an obscured
attribute is unknown rather than known-absent.

mechanical_terms — zero or more gameplay terms (oracle-text keywords, card types, EDHREC
deck archetypes such as "lifegain", "sacrifice", "landfall", "voltron") but ONLY when the
query actually names a game mechanic or deck strategy. A purely visual theme has none;
return an empty list, which is the normal case.

reasoning — one short sentence naming where the query's meaning lives."""


def _clamp01(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return min(1.0, max(0.0, out))


def _normalize_weights(literal: float, interpretive: float, notes: list[str]) -> tuple[float, float]:
    """Clamp to [0,1] and renormalize to sum 1, defensively."""
    total = literal + interpretive
    if total <= 0:
        notes.append("router returned zero weights; using 0.5/0.5")
        return 0.5, 0.5
    if abs(total - 1.0) > 1e-6:
        notes.append(f"router weights summed to {total:.2f}; renormalized")
    return literal / total, interpretive / total


def _clean_slot_filters(raw, notes: list[str]) -> list[dict]:
    filters = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        op = str(entry.get("op") or "").strip().lower()
        value = str(entry.get("value") or "").strip()
        if path not in SLOT_PATHS or op not in SLOT_OPS or not value:
            notes.append(f"dropped unusable slot filter {entry!r}")
            continue
        filters.append({"path": path, "op": op, "value": value})
    return filters


def route(cfg: Config, query: str) -> dict:
    """One judge_model call returning the retrieval plan. Never raises."""
    notes: list[str] = []
    prompt = ROUTER_PROMPT.format(query=query, paths=", ".join(SLOT_PATHS))
    last_error = ""

    for attempt in (1, 2):
        try:
            raw = ollama.generate(
                cfg,
                cfg.judge_model,
                prompt,
                system=ROUTER_SYSTEM,
                format=ROUTER_SCHEMA,
                options={"temperature": 0},
            )
            parsed = judge_mod.parse_json(raw)
            literal = _clamp01(parsed.get("literal_weight"), 0.5)
            interpretive = _clamp01(parsed.get("interpretive_weight"), 0.5)
            literal, interpretive = _normalize_weights(literal, interpretive, notes)
            terms = [str(t).strip() for t in (parsed.get("mechanical_terms") or []) if str(t).strip()]
            return {
                "literal_weight": round(literal, 4),
                "interpretive_weight": round(interpretive, 4),
                "slot_filters": _clean_slot_filters(parsed.get("slot_filters"), notes),
                "mechanical_terms": terms,
                "router_reasoning": str(parsed.get("reasoning") or "").strip(),
                "router_ok": True,
                "notes": notes,
            }
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            last_error = str(exc)
            if attempt == 1:
                print(f"route: router call failed ({exc}); retrying once", file=sys.stderr)

    print(f"route: router unavailable ({last_error}); falling back to 0.5/0.5", file=sys.stderr)
    notes.append(f"router failed ({last_error}); fell back to 0.5/0.5 with no filters")
    return {
        "literal_weight": 0.5,
        "interpretive_weight": 0.5,
        "slot_filters": [],
        "mechanical_terms": [],
        "router_reasoning": "",
        "router_ok": False,
        "notes": notes,
    }


# -------------------------------------------------------------------------- expansion

EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "expansions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["expansions"],
}

INTERPRETIVE_SYSTEM = (
    "You rewrite art themes into the register an interpretive image description is "
    "written in: mood, emotional register, implied narrative, power dynamics, genre and "
    "tonal register, analogies to film, music or period art. You never describe physical "
    "objects — another route handles those."
)

INTERPRETIVE_PROMPT = """THEME: "{query}"

Write 6 short statements an art critic might write about an image that fits this theme,
in the interpretive register: what it conveys, what it feels like, what kind of story it
looks like a frame from, what it evokes by analogy.

Examples of the register, for the theme "commanders that look lonely":
  "conveys isolation and quiet resignation"
  "melancholy and still, nothing moving in the frame"
  "reads as someone left behind after everyone else has gone"
  "the emotional register of a long unanswered wait"

Rules:
- Each statement stands alone and is independently meaningful.
- No physical inventory: not "a figure on a cliff" but what such an image conveys.
- Vary the phrasing; near-duplicates of each other are wasted.
- Stay faithful to the theme. Do not drift to an adjacent mood."""

VISUAL_SYSTEM = (
    "You decompose art themes into physical visual evidence. Every artwork in the index "
    "has a literal description listing only what is visibly present: figures, species, "
    "facial hair, held objects, clothing, poses, background, palette, lighting, "
    "composition. You write the concrete evidence statements that a matching image "
    "would contain, in exactly that vocabulary."
)

VISUAL_PROMPT = """THEME: "{query}"

Write 6 short statements describing what an image matching this theme would PHYSICALLY
contain. Only what a camera would record: figures and their species, facial hair, held
objects, clothing, posture, setting, lighting, colours, composition.

Examples, for the theme "commanders that look lonely":
  "a single figure with their back turned"
  "vast empty landscape with no other figures"
  "cold desaturated palette, blues and greys"
  "downcast posture, lowered head"
  "one small figure against a large empty background"

Rules:
- Physical evidence only. No mood words, no "lonely", no "menacing", no interpretation.
- Each statement stands alone, phrased as a factual observation about the image.
- Prefer the evidence that would most reliably distinguish a match from a non-match.
- Cover different kinds of evidence rather than restating one six times."""


def _expand_once(cfg: Config, query: str, system: str, prompt: str, notes: list[str], label: str) -> list[str]:
    """One expansion call. On failure, note it and fall back to the query itself."""
    try:
        raw = ollama.generate(
            cfg,
            cfg.judge_model,
            prompt.format(query=query),
            system=system,
            format=EXPANSION_SCHEMA,
            options={"temperature": 0.3},
        )
        items = judge_mod.parse_json(raw).get("expansions")
        cleaned = [" ".join(str(i).split()) for i in (items or []) if str(i).strip()]
        if not cleaned:
            raise ValueError("no expansion strings returned")
        return cleaned[:8]
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        print(f"expand: {label} expansion failed ({exc}); using the query itself", file=sys.stderr)
        notes.append(f"{label} expansion failed ({exc}); used the raw query")
        return []


def expand(cfg: Config, query: str, plan: dict) -> list[dict]:
    """Build every (string, layer) pair to search. Both routes whenever they apply.

    The literal route always runs: for a literal-dominant query it is the query plus
    paraphrases of the physical thing asked for; for an abstract one it is the
    decomposition into visual evidence. The interpretive route runs whenever the theme
    carries any interpretive weight at all.
    """
    notes: list[str] = plan.setdefault("notes", [])
    expansions: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(text: str, layer: str, route_name: str) -> None:
        key = (layer, text.lower())
        if text and key not in seen:
            seen.add(key)
            expansions.append({"text": text, "layer": layer, "route": route_name})

    add(query, "literal", "query")
    for text in _expand_once(cfg, query, VISUAL_SYSTEM, VISUAL_PROMPT, notes, "decomposed"):
        add(text, "literal", "decomposed")

    if plan["interpretive_weight"] > 0:
        add(query, "interpretive", "query")
        for text in _expand_once(
            cfg, query, INTERPRETIVE_SYSTEM, INTERPRETIVE_PROMPT, notes, "direct"
        ):
            add(text, "interpretive", "direct")

    plan["expansions"] = {
        "literal": [e["text"] for e in expansions if e["layer"] == "literal"],
        "interpretive": [e["text"] for e in expansions if e["layer"] == "interpretive"],
    }
    return expansions


# ------------------------------------------------------------------------ slot filters


def _slot_clause(flt: dict) -> tuple[str, list]:
    """One json_extract condition over descriptions.slots.

    List-valued slots (held_objects, other_figures, palette) come back from
    json_extract as their JSON text, so `contains` matches inside the serialized list.
    """
    path = "$." + flt["path"]
    op, value = flt["op"], flt["value"]
    field = "json_extract(slots, ?)"
    if op == "equals":
        return f"lower(CAST({field} AS TEXT)) = lower(?)", [path, value]
    if op == "contains":
        return f"lower(CAST({field} AS TEXT)) LIKE '%' || lower(?) || '%'", [path, value]
    if op == "not_contains":
        # `field` appears twice, so `path` must be bound twice.
        return (
            f"({field} IS NULL OR lower(CAST({field} AS TEXT)) NOT LIKE '%' || lower(?) || '%')",
            [path, path, value],
        )
    if op == "gte":
        return f"CAST({field} AS REAL) >= CAST(? AS REAL)", [path, value]
    return f"CAST({field} AS REAL) <= CAST(? AS REAL)", [path, value]


def allowed_illustrations(
    conn: sqlite3.Connection, filters: list[dict], notes: list[str]
) -> set[str] | None:
    """Illustration ids passing every slot filter, or None when unfiltered.

    Soft-fail: a filter that empties the pool is dropped rather than returning nothing.
    """
    active = list(filters)
    while active:
        clauses, params = [], []
        for flt in active:
            clause, values = _slot_clause(flt)
            clauses.append(clause)
            params.extend(values)
        rows = conn.execute(
            f"SELECT illustration_id FROM descriptions WHERE {' AND '.join(clauses)}", params
        ).fetchall()
        if rows:
            return {r["illustration_id"] for r in rows}
        dropped = active.pop()
        note = f"slot filter {dropped['path']} {dropped['op']} {dropped['value']!r} matched nothing; dropped"
        print(f"filter: {note}", file=sys.stderr)
        notes.append(note)
    return None


# --------------------------------------------------------------------------- retrieval


def _query_vectors(cfg: Config, texts: list[str], notes: list[str]) -> np.ndarray | None:
    """Embed every expansion string in one call. None means dense retrieval is off."""
    try:
        vecs = ollama.embed(cfg, texts)
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        print(f"retrieve: embedding failed ({exc}); BM25 only", file=sys.stderr)
        notes.append(f"dense retrieval unavailable ({exc}); ranked on BM25 alone")
        return None
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def retrieve(
    index: SearchIndex,
    expansions: list[dict],
    weights: dict[str, float],
    query_vecs: np.ndarray | None,
    allowed: set[str] | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[tuple[str, str], str]]:
    """Rank each (expansion, method) list separately and fuse the ranks with RRF.

    Per list: rank propositions, keep each artwork's best-ranked proposition, scale its
    contribution by the routed weight of that proposition's layer, and add
    weight/(60+rank) to the artwork's fused score. Raw dense and BM25 scores are never
    normalized against each other — only their ranks are ever compared.
    """
    fused: dict[str, float] = {}
    per_method: dict[str, dict[str, float]] = {"dense": {}, "bm25": {}}
    best_layer: dict[tuple[str, str], str] = {}
    best_rank: dict[tuple[str, str], int] = {}

    if not len(index):
        return fused, per_method, best_layer

    allowed_mask = None
    if allowed is not None:
        allowed_mask = np.fromiter(
            (ill in allowed for ill in index.illustration_ids), dtype=bool, count=len(index)
        )

    for position, exp in enumerate(expansions):
        rows = index.layer_index.get(exp["layer"])
        if rows is None or rows.size == 0:
            continue
        if allowed_mask is not None:
            rows = rows[allowed_mask[rows]]
            if rows.size == 0:
                continue

        lists: list[tuple[str, np.ndarray]] = []
        if query_vecs is not None:
            # One full dot product, then mask: slicing the matrix first would copy it.
            lists.append(("dense", (index.vecs @ query_vecs[position])[rows]))
        if index.bm25 is not None:
            tokens = tokenize(exp["text"])
            if tokens:
                lists.append(("bm25", index.bm25.get_scores(tokens)[rows]))

        for method, scores in lists:
            if method == "bm25":
                # A zero BM25 score means no term overlap at all; ranking zeros would
                # just fuse arbitrary rows into the pool.
                keep = scores > 0
                candidate_rows, candidate_scores = rows[keep], scores[keep]
            else:
                candidate_rows, candidate_scores = rows, scores
            if candidate_rows.size == 0:
                continue

            order = np.argsort(-candidate_scores)[:PROP_DEPTH]
            seen: set[str] = set()
            for rank, offset in enumerate(order, start=1):
                row = int(candidate_rows[offset])
                ill = index.illustration_ids[row]
                if ill in seen:
                    continue  # keep only this artwork's best-ranked proposition
                seen.add(ill)
                layer = index.layers[row]
                contribution = weights.get(layer, 0.0) / (RRF_K + rank)
                fused[ill] = fused.get(ill, 0.0) + contribution
                per_method[method][ill] = per_method[method].get(ill, 0.0) + contribution
                # Log the layer of the single best-ranked proposition this method found
                # for this artwork, across every expansion — not whichever ran first.
                key = (method, ill)
                if rank < best_rank.get(key, 1 << 30):
                    best_rank[key] = rank
                    best_layer[key] = layer

    return fused, per_method, best_layer


# ------------------------------------------------------------------ candidates and bands


def candidate_rows(conn: sqlite3.Connection, illustration_ids: list[str]) -> dict[str, dict]:
    """Everything a result needs, joined at the artwork level."""
    out: dict[str, dict] = {}
    sql = """
        SELECT a.illustration_id, a.oracle_id, a.set_code, a.artist, a.art_path,
               a.art_crop_url, a.scryfall_uri, a.tcgplayer_uri, a.face_index,
               c.name, c.mana_cost, c.type_line, c.oracle_text, c.color_identity,
               p.score AS power_score,
               e.slug AS edhrec_slug, e.themes, e.archetypes,
               (SELECT COUNT(*) FROM arts a2 WHERE a2.oracle_id = a.oracle_id) AS art_count
        FROM arts a
        JOIN cards c ON c.oracle_id = a.oracle_id
        LEFT JOIN power p ON p.oracle_id = a.oracle_id
        LEFT JOIN edhrec e ON e.oracle_id = a.oracle_id
        WHERE a.illustration_id IN ({marks})
    """
    for chunk in judge_mod.chunks(illustration_ids, 400):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(sql.format(marks=marks), chunk):
            out[row["illustration_id"]] = dict(row)
    return out


def power_bands(conn: sqlite3.Connection) -> tuple[dict[str, int], list[float]]:
    """Five quantile buckets over power.score, computed at query time, never stored."""
    rows = conn.execute("SELECT oracle_id, score FROM power WHERE score IS NOT NULL").fetchall()
    if not rows:
        return {}, []
    scores = np.array([float(r["score"]) for r in rows], dtype=np.float64)
    edges = np.quantile(scores, [0.2, 0.4, 0.6, 0.8])
    bands = {
        r["oracle_id"]: int(np.searchsorted(edges, float(r["score"]), side="right")) + 1
        for r in rows
    }
    return bands, [float(e) for e in edges]


def collapse(fused: dict[str, float], rows: dict[str, dict], bands: dict[str, int]) -> list[dict]:
    """Best-scoring artwork per commander, after ranking, never before.

    Collapsing earlier would average one matching printing together with five
    non-matching ones and bury the hit.
    """
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    best: list[dict] = []
    seen: set[str] = set()
    for ill, score in ordered:
        row = rows.get(ill)
        if row is None:
            continue  # art row vanished between index build and query
        oracle_id = row["oracle_id"]
        if oracle_id in seen:
            continue
        seen.add(oracle_id)
        best.append({**row, "score": score, "band": bands.get(oracle_id)})
    return best


# ----------------------------------------------------------------------- post-filtering


def color_set(value: str | None) -> set[str]:
    return {ch for ch in (value or "").upper() if ch in WUBRG}


def _matches_mechanical(card: dict, terms: list[str]) -> bool:
    haystack = " ".join(
        str(card.get(field) or "").lower()
        for field in ("oracle_text", "type_line", "themes", "archetypes")
    )
    return any(term.lower() in haystack for term in terms)


def post_filter(
    cards: list[dict],
    *,
    band_range: tuple[int, int] | None,
    colors: str | None,
    mechanical_terms: list[str],
    notes: list[str],
    has_bands: bool,
) -> list[dict]:
    """Card-level filters, applied after collapse. Returns the surviving order."""
    out = cards

    if mechanical_terms:
        filtered = [c for c in out if _matches_mechanical(c, mechanical_terms)]
        if filtered:
            out = filtered
        else:
            note = f"mechanical filter {mechanical_terms} matched nothing; dropped"
            print(f"filter: {note}", file=sys.stderr)
            notes.append(note)

    if band_range is not None:
        if not has_bands:
            note = "power band filter ignored: no power scores in the database"
            print(f"filter: {note}", file=sys.stderr)
            if note not in notes:
                notes.append(note)
        else:
            low, high = band_range
            out = [c for c in out if c.get("band") is not None and low <= c["band"] <= high]

    if colors:
        wanted = color_set(colors)
        out = [c for c in out if color_set(c.get("color_identity")) <= wanted]

    return out


# --------------------------------------------------------------------------- logging


def _log_query(conn: sqlite3.Connection, query: str, kind: str, plan: dict) -> int:
    cur = conn.execute(
        "INSERT INTO queries(text, kind, params, created_at) VALUES (?, ?, ?, ?)",
        (query, kind, json.dumps(plan, default=str), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return int(cur.lastrowid)


def _log_retrievals(conn: sqlite3.Connection, query_id: int, rows: list[tuple]) -> None:
    conn.executemany(
        "INSERT INTO retrievals(query_id, illustration_id, rank, score, method, layer) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def _fused_rows(pool: list[dict]) -> list[tuple]:
    """The pool the judge actually sees: rank, RRF score, method 'rrf', layer 'fused'."""
    return [
        (None, c["illustration_id"], rank, float(c.get("score") or 0.0), "rrf", "fused")
        for rank, c in enumerate(pool, start=1)
    ]


def _method_rows(
    per_method: dict[str, dict[str, float]], best_layer: dict[tuple[str, str], str]
) -> list[tuple]:
    """Per-method top-50, carrying the layer of the proposition that contributed."""
    rows: list[tuple] = []
    for method, scores in per_method.items():
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:PER_METHOD_LOG]
        for rank, (ill, score) in enumerate(ordered, start=1):
            rows.append((None, ill, rank, float(score), method, best_layer.get((method, ill), "")))
    return rows


# --------------------------------------------------------------------------- selection


def _passes(result: dict) -> bool:
    fit = result.get("fit")
    return fit is not None and fit >= judge_mod.PASS_FIT and not result.get("vision_rejected")


def _result_dict(cand: dict, terms: list[str]) -> dict:
    row = {**cand, "matched_terms": terms}
    return {
        "oracle_id": cand["oracle_id"],
        "name": cand.get("name"),
        "mana_cost": cand.get("mana_cost"),
        "type_line": cand.get("type_line"),
        "color_identity": cand.get("color_identity") or "",
        "band": cand.get("band"),
        "fit": cand.get("fit"),
        "rationale": cand.get("rationale"),
        "verified": bool(cand.get("verified")),
        "illustration_id": cand["illustration_id"],
        "set_code": cand.get("set_code"),
        "artist": cand.get("artist"),
        "prop_ids": cand.get("prop_ids") or [],
        "links": links_for(row),
        # beyond the contract's required keys, for honest labelling in both outputs
        "stretch": not _passes(cand),
        "vision_rejected": bool(cand.get("vision_rejected")),
        "verify_note": cand.get("verify_note"),
        "score": float(cand.get("score") or 0.0),
        "art_count": int(cand.get("art_count") or 1),
    }


def select(index: SearchIndex, judged: list[dict], k: int) -> list[dict]:
    """Diversity over the passing results, then stretches to fill, clearly separated."""
    ordered = sorted(judged, key=judge_mod.fit_key, reverse=True)
    survivors = [r for r in ordered if _passes(r)]
    chosen = judge_mod.diversify(index, survivors, k)

    if len(chosen) < k:
        # Below the bar. They are returned only to fill the pool, and marked as such.
        taken = {r["illustration_id"] for r in chosen}
        rest = [
            r
            for r in ordered
            if r["illustration_id"] not in taken and not r.get("vision_rejected")
        ]
        for extra in judge_mod.color_cap(rest):
            if len(chosen) >= k:
                break
            chosen.append(extra)
    return chosen


# ---------------------------------------------------------------------------- pipeline


def execute(
    cfg: Config,
    query: str,
    *,
    band: int | None = None,
    colors: str | None = None,
    k: int = 5,
    kind: str = "user",
    conn: sqlite3.Connection | None = None,
    index: SearchIndex | None = None,
) -> dict:
    """Route → expand → retrieve → collapse → filter → judge → verify → diversify.

    `conn` and `index` are accepted so callers running many queries (evaluate, synth)
    pay the index build once.
    """
    started = time.perf_counter()
    own_conn = conn is None
    conn = conn or db.connect(cfg)
    if index is None:
        index = load_index(cfg, conn)

    try:
        plan = route(cfg, query)
        notes: list[str] = plan["notes"]

        # Both routes run whenever the theme has interpretive weight, so a route we
        # will run never contributes exactly nothing.
        if plan["interpretive_weight"] > 0:
            literal = max(plan["literal_weight"], MIN_ROUTE_WEIGHT)
            interpretive = max(plan["interpretive_weight"], MIN_ROUTE_WEIGHT)
            if (literal, interpretive) != (plan["literal_weight"], plan["interpretive_weight"]):
                notes.append(
                    f"floored route weights at {MIN_ROUTE_WEIGHT} so both routes count"
                )
                literal, interpretive = _normalize_weights(literal, interpretive, [])
                plan["literal_weight"] = round(literal, 4)
                plan["interpretive_weight"] = round(interpretive, 4)

        weights = {
            "literal": plan["literal_weight"],
            "interpretive": plan["interpretive_weight"],
        }
        plan.update(
            {
                "query": query,
                "band": band,
                "colors": colors,
                "k": k,
                "kind": kind,
                "models": {
                    "judge": cfg.judge_model,
                    "embed": cfg.embed_model,
                    "vision": cfg.vision_model,
                },
                "index": {
                    "props": len(index),
                    "artworks": index.artwork_count,
                    "dim": index.dim,
                    "build_seconds": round(index.build_seconds, 4),
                    "missing_embeddings": index.missing_embeddings,
                },
            }
        )

        if not len(index):
            notes.append("index is empty: run 'python -m cts describe' then 'embed'")
            print("search: index is empty — nothing to retrieve", file=sys.stderr)
            query_id = _log_query(conn, query, kind, plan)
            return {"query_id": query_id, "plan": plan, "relaxed": None, "results": [], "pool": []}

        expansions = expand(cfg, query, plan)
        allowed = allowed_illustrations(conn, plan["slot_filters"], notes)
        query_vecs = _query_vectors(cfg, [e["text"] for e in expansions], notes)

        fused, per_method, best_layer = retrieve(index, expansions, weights, query_vecs, allowed)
        rows = candidate_rows(conn, list(fused))
        bands, edges = power_bands(conn)
        plan["band_edges"] = edges
        collapsed = collapse(fused, rows, bands)

        band_range = (band, band) if band is not None else None
        candidates = post_filter(
            collapsed,
            band_range=band_range,
            colors=colors,
            mechanical_terms=plan["mechanical_terms"],
            notes=notes,
            has_bands=bool(bands),
        )[:POOL_SIZE]

        plan["counts"] = {
            "expansions": len(expansions),
            "artworks_ranked": len(fused),
            "commanders": len(collapsed),
            "candidates": len(candidates),
        }

        query_id = _log_query(conn, query, kind, plan)
        _log_retrievals(
            conn,
            query_id,
            [(query_id, *row[1:]) for row in _fused_rows(candidates)]
            + [(query_id, *row[1:]) for row in _method_rows(per_method, best_layer)],
        )

        judged = judge_mod.judge_batches(cfg, conn, query, candidates)
        judge_mod.log_judgments(conn, query_id, cfg.judge_model, judged)
        judged, vision_ok = judge_mod.verify_finalists(cfg, judged, query)

        relaxed: str | None = None
        survivors = [r for r in judged if _passes(r)]
        if len(survivors) < k and band is not None and bands:
            # One widening, reported, never silent.
            low, high = max(1, band - 1), min(5, band + 1)
            if (low, high) != (band, band):
                wider = post_filter(
                    collapsed,
                    band_range=(low, high),
                    colors=colors,
                    mechanical_terms=plan["mechanical_terms"],
                    notes=notes,
                    has_bands=True,
                )[:POOL_SIZE]
                already = {r["illustration_id"] for r in judged}
                fresh = [c for c in wider if c["illustration_id"] not in already]
                if fresh:
                    relaxed = f"power band widened from {band} to {low}-{high}"
                    print(f"search: {relaxed} ({len(fresh)} more candidates)", file=sys.stderr)
                    _log_retrievals(
                        conn, query_id, [(query_id, *row[1:]) for row in _fused_rows(fresh)]
                    )
                    extra = judge_mod.judge_batches(cfg, conn, query, fresh)
                    judge_mod.log_judgments(conn, query_id, cfg.judge_model, extra)
                    judged = judged + extra
                    judged, vision_ok_2 = judge_mod.verify_finalists(cfg, judged, query)
                    vision_ok = vision_ok and vision_ok_2
                    plan["counts"]["candidates_after_relax"] = len(judged)

        if not vision_ok:
            notes.append("vision verification unavailable; results are judge-ordered only")
        plan["vision_verified"] = vision_ok
        plan["relaxed"] = relaxed

        terms = plan["mechanical_terms"]
        chosen = select(index, judged, k)
        results = [_result_dict(c, terms) for c in chosen]
        pool = [
            _result_dict(c, terms)
            for c in sorted(judged, key=judge_mod.fit_key, reverse=True)
        ]

        plan["counts"]["judged"] = len(judged)
        plan["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        conn.execute(
            "UPDATE queries SET params = ? WHERE id = ?",
            (json.dumps(plan, default=str), query_id),
        )
        conn.commit()

        return {
            "query_id": query_id,
            "plan": plan,
            "relaxed": relaxed,
            "results": results,
            "pool": pool,
        }
    finally:
        if own_conn:
            conn.close()


# --------------------------------------------------------------------------------- CLI

LINK_LABELS = (
    ("edhrec", "edhrec"),
    ("edhrec_theme", "theme"),
    ("scryfall", "scryfall"),
    ("tcgplayer", "tcgplayer"),
    ("art_crop", "art crop"),
)


def _format_result(position: int, result: dict, show_fit: bool) -> str:
    bits = [f"{position}. {result['name']}"]
    if result.get("mana_cost"):
        bits.append(result["mana_cost"])
    bits.append(f"[{result['color_identity'] or 'C'}]")
    if result.get("band") is not None:
        bits.append(f"band {result['band']}")
    fit = result.get("fit")
    if show_fit or result["stretch"]:
        bits.append("fit —" if fit is None else f"fit {fit:.2f}")
    if result["verified"]:
        bits.append("verified")
    if result["stretch"]:
        bits.append("STRETCH (below the 0.5 bar)")

    lines = ["  ".join(bits)]
    printing = f"   art: {result.get('set_code') or '?'} · {result.get('artist') or 'unknown artist'}"
    if result.get("art_count", 1) > 1:
        printing += f" · 1 of {result['art_count']} arts"
    lines.append(printing)
    if result.get("rationale"):
        lines.append(f"   {result['rationale']}")
    for key, label in LINK_LABELS:
        if key in result["links"]:
            lines.append(f"      {label:<10} {result['links'][key]}")
    return "\n".join(lines)


def run(
    cfg: Config,
    query: str,
    band: int | None = None,
    colors: str | None = None,
    k: int = 5,
    as_json: bool = False,
) -> None:
    """CLI entry point. Prints the pool, honestly labelled."""
    outcome = execute(cfg, query, band=band, colors=colors, k=k, kind="user")

    if as_json:
        print(json.dumps(outcome, indent=2, default=str))
        return

    plan = outcome["plan"]
    literal, interpretive = plan["literal_weight"], plan["interpretive_weight"]
    header = [f'Scrying Pool · "{query}"']
    route_bits = [f"route: {literal:.0%} literal / {interpretive:.0%} interpretive"]
    if band is not None:
        route_bits.append(f"band {band}")
    if colors:
        route_bits.append(f"colors {colors.upper()}")
    if plan.get("slot_filters"):
        route_bits.append(
            "slots " + ", ".join(f"{f['path']} {f['op']} {f['value']}" for f in plan["slot_filters"])
        )
    if plan.get("mechanical_terms"):
        route_bits.append("mechanical " + ", ".join(plan["mechanical_terms"]))
    header.append(" · ".join(route_bits))
    print("\n".join(header))

    if outcome["relaxed"]:
        print(f"note: {outcome['relaxed']} — fewer than {k} results passed at the requested band")
    if not plan.get("vision_verified", True):
        print("note: vision verification unavailable — results are judge-ordered and unverified")
    for note in plan.get("notes", []):
        print(f"note: {note}")

    results = outcome["results"]
    if not results:
        counts = plan.get("counts", {})
        print(
            f"\nno matches. {counts.get('commanders', 0)} commanders retrieved, "
            f"{counts.get('candidates', 0)} survived the filters."
        )
        return

    # Abstract themes genuinely live in the middle of the scale, so show the number.
    show_fit = interpretive >= 0.4
    print()
    for position, result in enumerate(results, start=1):
        print(_format_result(position, result, show_fit))
        print()

    passing = sum(1 for r in results if not r["stretch"])
    if passing < len(results):
        print(f"{passing} of {len(results)} results clear the 0.5 fit bar; the rest are stretches.")
