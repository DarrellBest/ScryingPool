"""Phase 5: the vision pass. One Ollama call per artwork, two layers out.

The unit of work is an artwork (`arts.illustration_id`), never a card. Default
printings are described before alternates so an interrupted run still covers every
commander once before it goes deep on any of them, and within each group the order is
fixed so a resumed run picks up exactly where it stopped.

The vision model is sent the image and `prompts.VISION_PROMPT`. It is never sent the
card name, oracle text, set, or artist — the SELECT below deliberately does not even
read those columns. See SPEC.md Phase 5.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import db, ollama, prompts
from .config import Config

# Stop the run after this many artworks fail back to back. One failure is a bad
# response; five in a row is Ollama down, the wrong model pulled, or a context that
# cannot hold the prompt — none of which get better by grinding through 4,000 more.
MAX_CONSECUTIVE_FAILURES = 5

_BULLET = re.compile(r"^\s*(?:[-*•]|\(?\d{1,2}[.)])\s+")
_WS = re.compile(r"\s+")
_INT = re.compile(r"-?\d+")


# --- selection -----------------------------------------------------------


def _pending(conn: sqlite3.Connection, cfg: Config, backfill_stale: bool) -> tuple[list[dict], int]:
    """Artworks needing description, default printings first, then by id.

    Returns (rows, missing_file_count). Rows carry only the illustration id and the
    image path: nothing else may reach the vision model.
    """
    sql = """
        SELECT a.illustration_id, a.art_path, COALESCE(a.is_default, 0) AS is_default
          FROM arts a
          LEFT JOIN descriptions d ON d.illustration_id = a.illustration_id
         WHERE a.art_path IS NOT NULL AND a.art_path != ''
           AND (d.illustration_id IS NULL
                OR (? = 1 AND COALESCE(d.prompt_version, -1) < ?))
         ORDER BY is_default DESC, a.illustration_id ASC
    """
    rows = conn.execute(sql, (1 if backfill_stale else 0, prompts.PROMPT_VERSION)).fetchall()

    pending: list[dict] = []
    missing = 0
    art_dir = Path(cfg.art_dir)
    for row in rows:
        path = Path(row["art_path"])
        if not path.is_file():
            # art_path is written relative to the repo root; fall back to the
            # configured art_dir so a moved database is still usable.
            fallback = art_dir / f"{row['illustration_id']}.jpg"
            if fallback.is_file():
                path = fallback
            else:
                missing += 1
                continue
        pending.append(
            {
                "illustration_id": row["illustration_id"],
                "path": str(path),
                "is_default": int(row["is_default"]),
            }
        )
    return pending, missing


# --- parsing and validation ----------------------------------------------


def _loads(raw: str) -> Any:
    """json.loads, with a fallback for a model that fences or chats around the object."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("response contains no JSON object")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"unparseable JSON ({exc.msg} at char {exc.pos})") from None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return _WS.sub(" ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_text(item) for item in value if _text(item))
    return ""


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().strip(".").casefold()


def _is_absent(value: str) -> bool:
    return _norm(value) in prompts.ABSENT_TOKENS


def _clean_props(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _text(item)
        text = _BULLET.sub("", text).strip().strip('"').strip()
        if len(text) < 3:
            continue
        key = _norm(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _coerce_int(value: Any, fallback: int) -> tuple[int, bool]:
    if isinstance(value, bool):
        return fallback, True
    if isinstance(value, int):
        return value, False
    if isinstance(value, float):
        return int(value), True
    match = _INT.search(str(value or ""))
    if match:
        return int(match.group()), True
    return fallback, True


def _slot_text(slots: dict, key: str, repairs: list[str]) -> str:
    """Read a scalar slot. Presence is the model's job; typing is ours."""
    if key not in slots:
        raise ValueError(f"slots.{key} missing")
    text = _text(slots[key])
    if not text:
        text = "none"
        repairs.append(f"slots.{key} was empty")
    return text


def validate(data: Any) -> tuple[dict, list[str]]:
    """Structurally check a parsed vision response and normalize it.

    Raises ValueError with a terse reason on a hard failure: unparseable shape, a
    missing key, or too few propositions. A missing key is always a hard failure —
    silent omission is the one thing that makes absence and inattention
    indistinguishable downstream. Wrong types on present keys are repaired instead,
    since losing twenty-five good propositions over one malformed integer is a worse
    trade than recording the repair.

    Returns (clean, repairs).
    """
    if not isinstance(data, dict):
        raise ValueError(f"top level is {type(data).__name__}, not an object")

    missing = [key for key in prompts.TOP_LEVEL_KEYS if key not in data]
    if missing:
        raise ValueError(f"missing top-level key(s): {', '.join(missing)}")

    repairs: list[str] = []

    literal = _text(data["literal"])
    interpretive = _text(data["interpretive"])
    if not literal:
        raise ValueError("literal paragraph is empty")
    if not interpretive:
        raise ValueError("interpretive paragraph is empty")

    lit_props = _clean_props(data["literal_propositions"], "literal_propositions")
    int_props = _clean_props(data["interpretive_propositions"], "interpretive_propositions")
    if len(lit_props) < prompts.LITERAL_MIN:
        raise ValueError(
            f"only {len(lit_props)} literal propositions, need at least {prompts.LITERAL_MIN}"
        )
    if len(int_props) < prompts.INTERPRETIVE_MIN:
        raise ValueError(
            f"only {len(int_props)} interpretive propositions, "
            f"need at least {prompts.INTERPRETIVE_MIN}"
        )
    if len(lit_props) > prompts.LITERAL_MAX:
        repairs.append(f"trimmed {len(lit_props)} literal propositions to {prompts.LITERAL_MAX}")
        lit_props = lit_props[: prompts.LITERAL_MAX]
    if len(int_props) > prompts.INTERPRETIVE_MAX:
        repairs.append(
            f"trimmed {len(int_props)} interpretive propositions to {prompts.INTERPRETIVE_MAX}"
        )
        int_props = int_props[: prompts.INTERPRETIVE_MAX]

    raw_slots = data["slots"]
    if not isinstance(raw_slots, dict):
        raise ValueError(f"slots is {type(raw_slots).__name__}, not an object")
    for key in prompts.SLOT_KEYS:
        if key not in raw_slots:
            raise ValueError(f"slots.{key} missing")

    raw_subject = raw_slots["primary_subject"]
    if not isinstance(raw_subject, dict):
        raise ValueError("slots.primary_subject is not an object")
    for key in prompts.PRIMARY_SUBJECT_KEYS:
        if key not in raw_subject:
            raise ValueError(f"slots.primary_subject.{key} missing")

    subject: dict[str, Any] = {}
    for key in ("species", "facial_hair", "clothing", "pose"):
        text = _text(raw_subject[key])
        if not text:
            text = "none"
            repairs.append(f"primary_subject.{key} was empty")
        subject[key] = text

    held: list[dict] = []
    raw_held = raw_subject["held_objects"]
    if not isinstance(raw_held, list):
        raw_held = [raw_held]
    for item in raw_held:
        if isinstance(item, dict):
            name = _text(item.get("object"))
            is_weapon = bool(item.get("is_weapon", False))
        else:
            name = _text(item)
            is_weapon = False
            if name:
                repairs.append("held_objects entry was not an object")
        if not name:
            continue
        held.append({"object": name, "is_weapon": is_weapon})
    if not held:
        held = [{"object": "none", "is_weapon": False}]
    subject["held_objects"] = held

    others: list[dict] = []
    raw_others = raw_slots["other_figures"]
    if not isinstance(raw_others, list):
        raw_others = [raw_others]
    for item in raw_others:
        if isinstance(item, dict):
            species = _text(item.get("species"))
            role = _text(item.get("role")) or "none"
        else:
            species = _text(item)
            role = "none"
            if species:
                repairs.append("other_figures entry was not an object")
        if not species:
            continue
        others.append({"species": species, "role": role})
    if not others:
        others = [{"species": "none", "role": "none"}]

    real_others = [f for f in others if not _is_absent(f["species"])]
    default_count = (0 if _is_absent(subject["species"]) else 1) + len(real_others)
    figure_count, repaired = _coerce_int(raw_slots["figure_count"], default_count)
    if repaired:
        repairs.append("figure_count was not an integer")

    palette_raw = raw_slots["palette"]
    if isinstance(palette_raw, str):
        palette = [part.strip() for part in palette_raw.split(",")]
        repairs.append("palette was a string")
    elif isinstance(palette_raw, list):
        palette = [_text(item) for item in palette_raw]
    else:
        palette = []
    palette = [c for c in palette if c and not _is_absent(c)]
    if not palette:
        palette = ["none"]
        repairs.append("palette was empty")

    slots = {
        "primary_subject": subject,
        "other_figures": others,
        "figure_count": figure_count,
        "setting": _slot_text(raw_slots, "setting", repairs),
        "time_of_day": _slot_text(raw_slots, "time_of_day", repairs),
        "palette": palette,
        "art_style": _slot_text(raw_slots, "art_style", repairs),
        "composition": _slot_text(raw_slots, "composition", repairs),
    }

    clean = {
        "literal": literal,
        "literal_propositions": lit_props,
        "slots": slots,
        "interpretive": interpretive,
        "interpretive_propositions": int_props,
    }
    return clean, repairs


# --- propositions --------------------------------------------------------


def _figure_phrase(count: int) -> str:
    if count <= 0:
        return "no figures"
    if count == 1:
        return "a single figure"
    if count == 2:
        return "two figures"
    if count <= 5:
        return "a small group of figures"
    return "a crowd of figures"


def fold_slots(slots: dict) -> list[str]:
    """Turn filled slots into terse literal propositions.

    Slot values otherwise reach only structured filters; written as propositions they
    become reachable by BM25 and dense retrieval too, which is where most queries
    actually land. Absent values ("none") are deliberately NOT folded in: indexing
    "primary subject facial hair: none" puts the token "facial hair" next to an image
    that has none, and a dense search for beards would happily match it. The absence
    still lives in descriptions.slots for anyone who wants to filter on it.
    """
    subject = slots["primary_subject"]
    out: list[str] = []

    def add(text: str, value: str) -> None:
        if not _is_absent(value):
            out.append(text)

    add(f"primary subject species: {subject['species']}", subject["species"])
    add(f"primary subject facial hair: {subject['facial_hair']}", subject["facial_hair"])
    add(f"primary subject clothing: {subject['clothing']}", subject["clothing"])
    add(f"primary subject pose: {subject['pose']}", subject["pose"])

    for item in subject["held_objects"][:4]:
        kind = "a weapon" if item["is_weapon"] else "a non-weapon object"
        add(f"primary subject holds {kind}: {item['object']}", item["object"])

    for figure in slots["other_figures"][:4]:
        add(
            f"other figure: {figure['species']} ({figure['role']})"
            if not _is_absent(figure["role"])
            else f"other figure: {figure['species']}",
            figure["species"],
        )

    count = slots["figure_count"]
    out.append(f"figure count: {count} ({_figure_phrase(count)})")

    add(f"setting: {slots['setting']}", slots["setting"])
    add(f"time of day: {slots['time_of_day']}", slots["time_of_day"])
    palette = ", ".join(slots["palette"])
    add(f"palette: {palette}", palette)
    add(f"art style: {slots['art_style']}", slots["art_style"])
    add(f"composition: {slots['composition']}", slots["composition"])
    return out


def build_props(clean: dict) -> list[tuple[str, str]]:
    """(layer, text) rows for `props`, deduplicated, literal evidence first."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for layer, texts in (
        ("literal", clean["literal_propositions"]),
        ("literal", fold_slots(clean["slots"])),
        ("interpretive", clean["interpretive_propositions"]),
    ):
        for text in texts:
            key = _norm(text)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append((layer, text))
    return rows


# --- writing -------------------------------------------------------------


def _write(
    conn: sqlite3.Connection,
    cfg: Config,
    illustration_id: str,
    clean: dict,
    rows: list[tuple[str, str]],
) -> None:
    """Replace this artwork's description and propositions in one transaction.

    The old props (and their embeddings, which are keyed on prop id) are deleted in
    the same transaction as the rewrite, so a re-describe never leaves stale text in
    the index and an interrupt never leaves half-written propositions.
    """
    with conn:  # BEGIN ... COMMIT, or ROLLBACK if anything raises
        conn.execute(
            "DELETE FROM embeddings WHERE prop_id IN "
            "(SELECT id FROM props WHERE illustration_id = ?)",
            (illustration_id,),
        )
        conn.execute("DELETE FROM props WHERE illustration_id = ?", (illustration_id,))
        conn.execute(
            "INSERT OR REPLACE INTO descriptions "
            "(illustration_id, literal, interpretive, slots, model, prompt_version, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                illustration_id,
                clean["literal"],
                clean["interpretive"],
                json.dumps(clean["slots"], ensure_ascii=False),
                cfg.vision_model,
                prompts.PROMPT_VERSION,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.executemany(
            "INSERT INTO props (illustration_id, layer, text) VALUES (?, ?, ?)",
            [(illustration_id, layer, text) for layer, text in rows],
        )


# --- the pass ------------------------------------------------------------


def describe_one(cfg: Config, image_path: str) -> tuple[dict, list[str]]:
    """One artwork: call, parse, validate, retry once. Raises ValueError if both fail."""
    reason: str | None = None
    for attempt in (1, 2):
        raw = ollama.vision(
            cfg,
            cfg.vision_model,
            prompts.vision_prompt(reason),
            image_path,
            format=prompts.VISION_SCHEMA,
            options=prompts.VISION_OPTIONS,
        )
        try:
            return validate(_loads(raw))
        except ValueError as exc:
            reason = str(exc)
            if attempt == 2:
                raise ValueError(reason) from None
            print(f"  retrying: {reason}", flush=True)
    raise AssertionError("unreachable")


def run(cfg: Config, limit: int | None = None, backfill_stale: bool = False) -> dict:
    if not cfg.vision_model.strip():
        print("error: no vision_model set in config.toml", file=sys.stderr)
        raise SystemExit(1)

    conn = db.connect(cfg)
    described = failed = consecutive = 0
    try:
        pending, missing = _pending(conn, cfg, backfill_stale)
        if missing:
            print(f"describe: {missing} artwork(s) skipped, image file not found", flush=True)
        if not pending:
            print("describe: nothing to do", flush=True)
            return {"described": 0, "failed": 0}

        todo = pending if limit is None else pending[:limit]
        defaults = sum(1 for row in todo if row["is_default"])
        print(
            f"describe: {len(todo)} artworks ({defaults} default, {len(todo) - defaults} "
            f"alternate) with {cfg.vision_model}, prompt_version={prompts.PROMPT_VERSION}"
            + (", backfilling stale" if backfill_stale else ""),
            flush=True,
        )

        for position, row in enumerate(todo, start=1):
            art_id = row["illustration_id"]
            tag = "default  " if row["is_default"] else "alternate"
            started = time.perf_counter()
            try:
                clean, repairs = describe_one(cfg, row["path"])
            except Exception as exc:  # bad response, HTTP error, unreadable image
                failed += 1
                consecutive += 1
                print(
                    f"[{position}/{len(todo)}] {art_id[:8]} {tag} FAILED: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    print(
                        f"describe: {consecutive} failures in a row, stopping. "
                        "Check that Ollama is up and the vision model is pulled; "
                        "re-run to resume.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                continue

            consecutive = 0
            rows = build_props(clean)
            _write(conn, cfg, art_id, clean, rows)
            described += 1
            written = len(rows)
            n_lit = sum(1 for layer, _ in rows if layer == "literal")
            print(
                f"[{position}/{len(todo)}] {art_id[:8]} {tag} props={written} "
                f"({n_lit}L/{written - n_lit}I) {time.perf_counter() - started:.1f}s"
                + (f" repaired: {'; '.join(repairs)}" if repairs else ""),
                flush=True,
            )
    finally:
        conn.close()

    print(f"describe: done. described={described} failed={failed}", flush=True)
    return {"described": described, "failed": failed}
