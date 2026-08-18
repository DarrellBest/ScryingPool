"""Oracle text -> chunks. Pure functions, no I/O, no network — the read side that
writes chunks into `oracle.db` lives at the bottom of this file (`run`), but the
decision that matters is `chunk_card`, which never touches a database.

**Decision, from the design doc: split each card's oracle text on newlines into
one chunk per ability, plus one whole-card chunk. Embed both kinds. Retrieve
over chunks, collapse to cards.**

Why the newline and nothing finer
----------------------------------
Scryfall's `oracle_text` separates abilities with `\\n` — authored by Wizards,
normalised by Scryfall, correct by definition. Splitting further (sentences,
clauses) is rejected: Magic text is hostile to sentence tokenisation (`{T}:`
has a colon, `{1}{W}` has braces, "+1/+1" has a slash, reminder text nests
whole sentences in parentheses). The newline is free and correct; anything
finer is expensive and approximate, so a compound single-line ability ("When
this enters, draw a card and lose 1 life") stays fused. Accepted cost.

Why the whole-card chunk, too
------------------------------
Per-ability chunks lose cross-ability context (an Aura's "enchanted creature
gets…" only makes sense with "Enchant creature" above it) and the card's
gestalt ("an engine that does three things"). One extra chunk per card holds
`type_line` + the full `oracle_text`, so both readings are retrievable.
Scoring is max-over-chunks per card via RRF, never a sum — summing would
reward verbosity.

Three transforms, `text_embedded` only
----------------------------------------
The chunk stored for *display* (`text`) is always verbatim. The chunk stored
for *embedding* (`text_embedded`) differs in exactly one way:

1. **Substitute the card's own name with "this card".** A proper noun embeds
   as whatever the words mean ("Bloodthirsty Blade" drags the vector toward
   blood and weapons), which is noise for a mechanical query. Both the full
   name and, when the name has a comma ("Atraxa, Praetors' Voice"), the short
   form before it are replaced — Magic's own templating convention for
   self-reference on legendary permanents uses exactly that short form.

Reminder text is **kept**, deliberately, by doing nothing to it: it is the
only English gloss a keyword ability has ("Cycling {2} (**{2}, Discard this
card: Draw a card.**)" embeds near "draw", which is correct — cycling
genuinely draws). Stripping it would make every keyword-only card invisible to
semantic search. And chunks are never prefixed with the type line: that is a
hard SQL filter, so the semantic layer never needs to carry it, and prefixing
tens of thousands of ability chunks with a mostly-constant string would dilute
every one of them. Only the whole-card chunk carries the type line, verbatim,
as part of its own text.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from typing import Iterable, Sequence

from . import oracle_db
from .config import Config

FACE_SEPARATOR = "\n//\n"

# The whole-card chunk's ordinal is never a line position — nothing in the
# design's "mark the matched line via ordinal" scheme applies to it, since it
# is not one line. -1 keeps it out of the ability numbering (which starts at
# 0) rather than overloading a value abilities could also have.
WHOLE_CHUNK_ORDINAL = -1


# --------------------------------------------------------------------------- pure


def face_texts(oracle_text: str | None) -> list[str]:
    """Split `cards.oracle_text` back into per-face texts on ingest's own join.

    `oracle_ingest.card_row` joins multi-face text with `"\\n//\\n"` (matching
    `cts/ingest.py`'s convention for the art corpus), so this is the exact
    inverse. A single-face card's text has no separator and comes back as one
    element.
    """
    text = oracle_text or ""
    return text.split(FACE_SEPARATOR)


def name_variants(name: str | None) -> tuple[str, ...]:
    """The full name, and the short form before a comma when the name has one.

    "Atraxa, Praetors' Voice" -> ("Atraxa, Praetors' Voice", "Atraxa"). Magic's
    own templating self-references a legendary permanent by the short form
    ("Atraxa deals damage equal to..."), which is the form that actually shows
    up in oracle text far more often than the full name does. Longest first,
    so a substitution pass never lets the short form eat into the full one.
    """
    name = (name or "").strip()
    if not name:
        return ()
    variants = [name]
    if "," in name:
        short = name.split(",", 1)[0].strip()
        if short and short != name:
            variants.append(short)
    return tuple(sorted(set(variants), key=len, reverse=True))


def substitute_name(text: str, name: str | None) -> str:
    """Replace the card's own name (and short form) with "this card".

    Whole-word matches only (`\\b`), so a name that happens to be a common
    word substring of something else in the text is not mangled. Deterministic
    and cannot go wrong: it is an exact string replacement on a known value.
    """
    out = text
    for variant in name_variants(name):
        out = re.sub(r"\b" + re.escape(variant) + r"\b", "this card", out)
    return out


def _split_abilities(face_text: str) -> list[str]:
    """One ability per non-blank line. A vanilla card's face contributes none."""
    return [line.strip() for line in face_text.split("\n") if line.strip()]


def chunk_card(oracle_id: str, name: str, type_line: str, oracle_text: str) -> list[dict]:
    """One card's `chunks` rows: one per ability (per face), plus one whole-card.

    Pure. No database, no network. Returns dicts with exactly the columns
    `chunks` needs: `oracle_id`, `face_index`, `ordinal`, `kind`, `text`,
    `text_embedded`. A vanilla creature (empty `oracle_text`) yields only the
    whole-card chunk — there is nothing else to say about it, and the
    whole-card chunk (type line + empty text) still lets it be found by a pure
    type/color/mv query with no mechanical intent at all.
    """
    rows: list[dict] = []

    for face_index, face_text in enumerate(face_texts(oracle_text)):
        for ordinal, line in enumerate(_split_abilities(face_text)):
            rows.append(
                {
                    "oracle_id": oracle_id,
                    "face_index": face_index,
                    "ordinal": ordinal,
                    "kind": "ability",
                    "text": line,
                    "text_embedded": substitute_name(line, name),
                }
            )

    type_line = (type_line or "").strip()
    oracle_text = oracle_text or ""
    whole = f"{type_line}\n{oracle_text}".strip("\n") if type_line else oracle_text
    rows.append(
        {
            "oracle_id": oracle_id,
            "face_index": 0,
            "ordinal": WHOLE_CHUNK_ORDINAL,
            "kind": "whole",
            "text": whole,
            "text_embedded": substitute_name(whole, name),
        }
    )
    return rows


# ----------------------------------------------------------------------------- write


_INSERT = (
    "INSERT INTO chunks(oracle_id, face_index, ordinal, kind, text, text_embedded) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


def rechunk(conn: sqlite3.Connection, changed_ids: Sequence[str] = ()) -> dict:
    """(Re)build chunks for every card with no chunks at all, plus `changed_ids`.

    Two things land in the same call because they are the same operation with
    different selectors: a card with zero chunk rows is either brand new (first
    build, or a card that just entered the corpus) or has never been chunked
    yet; a card in `changed_ids` already has chunks but its `oracle_text` moved
    since they were written (Wizards issues errata and templating updates —
    text is not immutable the way artwork is). Both need the same treatment:
    delete whatever chunks (and their embeddings) exist for the card, then
    write fresh ones, so stage 8 (embed) picks up exactly what changed.

    `oracle_ingest.write()` already detects the "text changed" set directly,
    by comparing the previous stored text to the new one inside its own
    transaction — no separate content hash column is needed here.
    """
    never_chunked = [
        row["oracle_id"]
        for row in conn.execute(
            "SELECT oracle_id FROM cards WHERE oracle_id NOT IN "
            "(SELECT DISTINCT oracle_id FROM chunks)"
        )
    ]
    target_ids = list(dict.fromkeys([*never_chunked, *(i for i in changed_ids if i)]))
    if not target_ids:
        return {"cards": 0, "chunks": 0}

    marks = ",".join("?" * len(target_ids))
    rows = conn.execute(
        f"SELECT oracle_id, name, type_line, oracle_text FROM cards "
        f"WHERE oracle_id IN ({marks})",
        target_ids,
    ).fetchall()

    conn.execute("BEGIN")
    try:
        conn.execute(
            f"DELETE FROM chunk_embeddings WHERE chunk_id IN "
            f"(SELECT id FROM chunks WHERE oracle_id IN ({marks}))",
            target_ids,
        )
        conn.execute(f"DELETE FROM chunks WHERE oracle_id IN ({marks})", target_ids)

        inserted = 0
        for row in rows:
            chunks = chunk_card(
                row["oracle_id"], row["name"] or "", row["type_line"] or "",
                row["oracle_text"] or "",
            )
            conn.executemany(
                _INSERT,
                [
                    (c["oracle_id"], c["face_index"], c["ordinal"], c["kind"],
                     c["text"], c["text_embedded"])
                    for c in chunks
                ],
            )
            inserted += len(chunks)
    except Exception:
        conn.rollback()
        raise
    conn.commit()

    return {"cards": len(rows), "chunks": inserted}


def run(cfg: Config, changed_ids: Iterable[str] = ()) -> dict:
    """CLI/refresh entry point: open the oracle db, rechunk, report, close."""
    conn = oracle_db.connect(cfg)
    try:
        result = rechunk(conn, list(changed_ids))
        print(
            f"oracle-chunk: {result['cards']:,} card(s) (re)chunked, "
            f"{result['chunks']:,} chunk row(s) written",
            flush=True,
        )
        return result
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover - convenience only
    from .config import load_config

    run(load_config())
    sys.exit(0)
