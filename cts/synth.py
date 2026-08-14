"""Phase 12, dataset 1: the synthetic theme corpus.

For every described artwork, ask the judge model which themes the artwork
genuinely satisfies and which it almost-but-doesn't. Across ~4,500 artworks that
is roughly 25k-35k positive (theme, artwork) pairs and ~10k hard negatives, in an
afternoon of local inference, with no user traffic needed.

Themes are generated FORWARD, from the art. The alternative — sample a list of
themes and search for matching artworks — inherits whatever bias the current
retriever already has, so every pair it produces is a pair the retriever already
found, and training on that just sharpens the existing failure modes. Generating
from the description guarantees the pair is genuine independent of retrieval,
which is the whole point of a cold-start dataset.

Each theme becomes its own `queries` row (kind='synth') plus a `judgments` row
(source='distill'), so the export in Phase 12 reads one uniform table shape
whether a label came from here, from a production search, or from a human.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone

from .config import Config

# Registers the model may tag a theme with. These are the spectrum from SPEC.md:
# literal attributes, compositional/stylistic, affective/narrative, analogical.
REGISTERS = ("literal", "compositional", "stylistic", "affective", "narrative", "analogical")

# Registers that count as "abstract" when checking the spread of a generation.
ABSTRACT_REGISTERS = frozenset({"compositional", "stylistic", "affective", "narrative"})

_THEME_ITEM = {
    "type": "object",
    "properties": {
        "theme": {"type": "string"},
        "register": {"type": "string", "enum": list(REGISTERS)},
        "rationale": {"type": "string"},
        "prop_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["theme", "register", "rationale", "prop_ids"],
}

SYNTH_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "themes": {"type": "array", "items": _THEME_ITEM, "minItems": 5, "maxItems": 8},
        "near_misses": {"type": "array", "items": _THEME_ITEM, "minItems": 2, "maxItems": 3},
    },
    "required": ["themes", "near_misses"],
}

SYNTH_SYSTEM = (
    "You generate search themes for an image search engine over fantasy artwork. "
    "You are given the complete record of ONE artwork and nothing else. You never "
    "name characters, cards, sets, planes, or artists, because the record does not "
    "contain them and inventing them poisons the dataset. You answer with JSON only."
)

SYNTH_PROMPT = """Everything recorded about a single piece of fantasy artwork follows: a literal \
description of what is physically visible, a structured slot table, an interpretive \
description of what the image conveys, and the numbered propositions extracted from \
both layers.

=== LITERAL (only what is physically visible) ===
{literal}

=== SLOTS ===
{slots}

=== INTERPRETIVE (what the image conveys) ===
{interpretive}

=== PROPOSITIONS ===
{props}

Write two lists of search themes for this artwork.

1. "themes" — 5 to 8 themes this artwork GENUINELY satisfies. Someone who typed the \
theme into a search box and was shown this image would say "yes, exactly". Span the \
spectrum deliberately:
   - 1 or 2 LITERAL: a verifiable attribute you could point at ("a figure with a full \
beard", "holding something that is not a weapon").
   - several ABSTRACT: mood and emotional register (affective), what just happened or \
is about to (narrative), how the frame is built (compositional), or how it is painted \
(stylistic).
   - at least 1 ANALOGICAL: what it resembles outside this world ("would fit on a black \
metal album cover", "feels like a Ghibli character", "belongs in a noir film").

2. "near_misses" — 2 or 3 themes this artwork ALMOST satisfies but does not. These are \
the most valuable lines here: a near miss is a theme a careless search would return this \
image for, and the difference must come down to ONE decidable thing that this image gets \
wrong. "a peaceful pastoral scene" for a quiet image that nonetheless holds a drawn \
blade is a good near miss. A theme about a completely different subject is a useless one.

Rules for every entry:
- Phrase it the way a person searches: a short phrase, lowercase, no card names, no \
character names, no set names, no game jargon, no "Magic" or "MTG".
- It must be true (or nearly true) of THIS image specifically. "a fantasy character" or \
"detailed digital painting" is true of everything and worth nothing.
- Do not simply copy a proposition back as a theme. A theme is what someone would search \
for; a proposition is the evidence that answers it.
- "register" is one of: literal, compositional, stylistic, affective, narrative, analogical.
- "rationale" is ONE sentence. For a theme, why the image satisfies it. For a near miss, \
the single specific thing that disqualifies it.
- "prop_ids" lists the numbered propositions above that your rationale relies on. Cite \
only ids that appear in the list. If nothing supports it, return an empty list — a \
rationale that cannot point at a proposition is one you invented.

Return JSON with exactly the keys "themes" and "near_misses"."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pretty_slots(raw: str | None) -> str:
    if not raw:
        return "(none recorded)"
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return str(raw)


def _format_props(props: list[sqlite3.Row]) -> str:
    if not props:
        return "(none recorded)"
    return "\n".join(f"[{p['id']}] ({p['layer']}) {p['text']}" for p in props)


def _parse(raw: str) -> dict:
    """Parse the model's JSON, tolerating a stray code fence or prose wrapper."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(match.group(0))


def _clean_theme(text: object) -> str:
    theme = " ".join(str(text or "").split()).strip().strip('"').strip()
    return theme[:200]


def _clean_items(items: object, valid_prop_ids: set[int]) -> list[dict]:
    """Normalize one list from the model into storable rows, dropping junk."""
    out: list[dict] = []
    seen: set[str] = set()
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        theme = _clean_theme(item.get("theme"))
        if len(theme) < 3 or theme.casefold() in seen:
            continue
        seen.add(theme.casefold())
        register = str(item.get("register") or "").strip().lower()
        if register not in REGISTERS:
            register = "abstract"
        # Keep only citations that name a proposition of THIS artwork: an id the
        # model invented is exactly the confabulation the citation requirement
        # exists to expose, and storing it would launder it into training data.
        cited = [
            int(pid)
            for pid in (item.get("prop_ids") or [])
            if isinstance(pid, (int, float)) and int(pid) in valid_prop_ids
        ]
        out.append(
            {
                "theme": theme,
                "register": register,
                "rationale": " ".join(str(item.get("rationale") or "").split())[:600],
                "prop_ids": cited,
            }
        )
    return out


def _done_illustration_ids(conn: sqlite3.Connection) -> set[str]:
    """Artworks that already have synth rows, so a re-run is a no-op for them."""
    done: set[str] = set()
    for (params,) in conn.execute("SELECT params FROM queries WHERE kind = 'synth'"):
        if not params:
            continue
        try:
            iid = json.loads(params).get("illustration_id")
        except (json.JSONDecodeError, AttributeError):
            continue
        if iid:
            done.add(str(iid))
    return done


def _store(
    conn: sqlite3.Connection,
    cfg: Config,
    illustration_id: str,
    items: list[dict],
    polarity: str,
) -> int:
    """Write one queries row + one judgments row per theme. Caller commits."""
    now = datetime.now(timezone.utc).isoformat()
    fit = 1.0 if polarity == "positive" else 0.0
    written = 0
    for item in items:
        params = {
            "illustration_id": illustration_id,
            "polarity": polarity,
            "register": item["register"],
        }
        cur = conn.execute(
            "INSERT INTO queries(text, kind, params, created_at) VALUES (?, 'synth', ?, ?)",
            (item["theme"], json.dumps(params), now),
        )
        conn.execute(
            "INSERT INTO judgments(query_id, illustration_id, fit, rationale, prop_ids, "
            "model, source) VALUES (?, ?, ?, ?, ?, ?, 'distill')",
            (
                cur.lastrowid,
                illustration_id,
                fit,
                item["rationale"],
                json.dumps(item["prop_ids"]),
                cfg.judge_model,
            ),
        )
        written += 1
    return written


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(cfg: Config, limit: int | None = None) -> dict:
    """Generate themes for every described artwork that does not have them yet."""
    from . import db, ollama

    conn = db.connect(cfg)

    # is_default first so an interrupted run still covers every commander once
    # before it goes deep on any of them; illustration_id breaks ties so the
    # order is stable across runs and the job is genuinely resumable.
    rows = conn.execute(
        """
        SELECT d.illustration_id AS iid, d.literal AS literal,
               d.interpretive AS interpretive, d.slots AS slots
          FROM descriptions d
          LEFT JOIN arts a ON a.illustration_id = d.illustration_id
         WHERE d.literal IS NOT NULL AND TRIM(d.literal) <> ''
         ORDER BY COALESCE(a.is_default, 0) DESC, d.illustration_id ASC
        """
    ).fetchall()

    done = _done_illustration_ids(conn)
    outstanding = [r for r in rows if r["iid"] not in done]
    already = len(rows) - len(outstanding)
    pending = outstanding[: max(0, int(limit))] if limit is not None else outstanding

    print(
        f"synth: {len(rows)} described artworks, {already} already have themes, "
        f"{len(pending)} to generate this run"
    )

    stats = {
        "artworks": 0,
        "skipped": already,
        "positives": 0,
        "near_misses": 0,
        "failed": 0,
        "missing_literal_register": 0,
        "missing_analogical_register": 0,
    }

    for i, row in enumerate(pending, start=1):
        iid = row["iid"]
        props = conn.execute(
            "SELECT id, layer, text FROM props WHERE illustration_id = ? ORDER BY id",
            (iid,),
        ).fetchall()
        valid_prop_ids = {int(p["id"]) for p in props}

        prompt = SYNTH_PROMPT.format(
            literal=(row["literal"] or "").strip(),
            slots=_pretty_slots(row["slots"]),
            interpretive=(row["interpretive"] or "(none recorded)").strip(),
            props=_format_props(props),
        )

        data = None
        for attempt in (1, 2):  # validate and retry once, then record and move on
            try:
                raw = ollama.generate(
                    cfg,
                    cfg.judge_model,
                    prompt if attempt == 1 else prompt + "\n\nReturn valid JSON only.",
                    system=SYNTH_SYSTEM,
                    format=SYNTH_SCHEMA,
                    options={"temperature": 0.7 if attempt == 1 else 0.3},
                )
                data = _parse(raw)
                break
            except Exception as exc:  # noqa: BLE001 - one bad artwork must not stop the run
                if attempt == 2:
                    print(f"synth: [{i}/{len(pending)}] {iid} FAILED: {exc}", flush=True)
                    stats["failed"] += 1

        if data is None:
            continue

        positives = _clean_items(data.get("themes"), valid_prop_ids)
        near_misses = _clean_items(data.get("near_misses"), valid_prop_ids)
        if not positives:
            print(f"synth: [{i}/{len(pending)}] {iid} FAILED: no usable themes", flush=True)
            stats["failed"] += 1
            continue

        registers = {p["register"] for p in positives}
        if "literal" not in registers:
            stats["missing_literal_register"] += 1
        if "analogical" not in registers:
            stats["missing_analogical_register"] += 1

        stats["positives"] += _store(conn, cfg, iid, positives, "positive")
        stats["near_misses"] += _store(conn, cfg, iid, near_misses, "near_miss")
        conn.commit()  # per artwork: an interrupted run loses at most one call
        stats["artworks"] += 1

        spread = "/".join(sorted(registers))
        print(
            f"synth: [{i}/{len(pending)}] {iid} +{len(positives)} themes "
            f"-{len(near_misses)} near misses ({spread})",
            flush=True,
        )

    conn.close()
    print(
        f"synth: done — {stats['artworks']} artworks, {stats['positives']} positive pairs, "
        f"{stats['near_misses']} hard negatives, {stats['failed']} failed"
    )
    if stats["artworks"]:
        print(
            f"synth: spectrum — {stats['missing_literal_register']} artworks produced no "
            f"literal theme, {stats['missing_analogical_register']} produced no analogical "
            "theme (both should stay near zero; if they climb, tighten SYNTH_PROMPT)"
        )
    return stats
