"""Structured filters -> SQL / sets, and the echo line. The highest-risk file
in this phase, because a mis-parsed filter is silent: a mis-parsed
categorical filter is self-announcing (ask for enchantments, get
planeswalkers), but a flipped `cmc <= 5` to `cmc >= 5` returns real, correct,
legal cards that all happen to cost seven mana — nothing *looks* wrong.

The filter algebra
-------------------
**UNION within a field, AND across fields.** `types=("enchantment",
"artifact")` means enchantment OR artifact; that combined with `colors="G"`
means (enchantment OR artifact) AND green. There is no NOT, no nesting, no
parentheses — see the design doc's *Not building*.

`colors` is the one field that is not UNION/AND at all: it is a single
**subset** test, `color_identity ⊆ requested` — the Commander deck-legality
rule. `cts/search.py::post_filter` already implements the identical test for
`/scry` (`color_set(...) <= wanted`), so `colors` means one thing across the
whole bot.

Hard versus soft
-----------------
`compile_hard` ANDs every provided field exactly, never drops anything, and
may legitimately return an empty set — a precise question with no answer gets
the honest answer "none", not a silently widened search. `compile_soft`
applies the same fields broadest-first and drops the *one* field that would
otherwise empty the combined pool, reporting the drop. `MIN_FILTERED_POOL`
from the art side (`cts/search.py`) is deliberately **not** ported: oracle
columns are complete and exact, so a filter that leaves three cards has left
the *correct* three, and dropping it for being small would throw away a right
answer. Soft filters here drop only at zero.

Numeric mana value
-------------------
Two representations, by construction, so the explicit path literally cannot
invert a comparison:

* **Explicit** (`mv_min`/`mv_max`, both optional, both inclusive) — the only
  representation the Discord command and the API request body expose. Two
  integers with an inclusive-bounds convention have no operator to flip.
* **Router** (`mv_op` + `mv_value`, or `mv_op="between"` + `mv_lo`/`mv_hi`) —
  the only representation that can misread "5 or less" as `>= 5`, which is
  exactly why it is soft and always echoed with its real symbol
  (`mv ≥ 5`), so an inversion is visible on screen rather than hidden in a
  plausible-looking result set.

A value outside 0-30 is dropped with a note: Magic's highest real paper mana
value is 16 (Draco, Gleemax), so anything past 30 is a parse artefact.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from urllib.parse import quote

WUBRG = "WUBRG"
MV_GUARD = (0, 30)
MV_OPS = ("<", "<=", "=", ">=", ">", "between")

_SYMBOL = {"<": "<", "<=": "≤", "=": "=", ">=": "≥", ">": ">"}


@dataclass(frozen=True)
class Filters:
    """One field per column this phase filters on. Everything is optional."""

    types: tuple[str, ...] = ()
    colors: str | None = None          # WUBRG letters; subset test
    legal: tuple[str, ...] = ()
    # Explicit path — inclusive bounds, no operator to invert.
    mv_min: float | None = None
    mv_max: float | None = None
    # Router path only — never set alongside mv_min/mv_max by any caller in
    # this codebase; kept as a separate representation on purpose (see module
    # docstring) rather than a single ambiguous pair of fields.
    mv_op: str | None = None
    mv_value: float | None = None
    mv_lo: float | None = None
    mv_hi: float | None = None

    def is_empty(self) -> bool:
        return not (
            self.types or self.colors or self.legal
            or self.mv_min is not None or self.mv_max is not None
            or self.mv_op is not None
        )


def _num(value: float) -> str:
    return f"{value:g}"


# --------------------------------------------------------------------- numeric mv


def _guard(value: float | None, notes: list[str], label: str) -> float | None:
    if value is None:
        return None
    lo, hi = MV_GUARD
    if not (lo <= value <= hi):
        notes.append(
            f"mv value {value:g} in {label} is outside 0-30 (a parse artefact, not a "
            "real Magic card) and was dropped"
        )
        return None
    return value


def explicit_mv_predicate(
    mv_min: float | None, mv_max: float | None, notes: list[str]
) -> tuple[str, list, str] | None:
    """Two inclusive bounds, no operator. Either or both may be set."""
    mv_min = _guard(mv_min, notes, "mv_min")
    mv_max = _guard(mv_max, notes, "mv_max")
    if mv_min is None and mv_max is None:
        return None
    if mv_min is not None and mv_max is not None:
        if mv_min > mv_max:
            notes.append(f"mv_min {mv_min:g} > mv_max {mv_max:g}; no card can satisfy both")
            return ("1 = 0", [], f"mv between {_num(mv_min)} and {_num(mv_max)} (impossible)")
        if mv_min == mv_max:
            return ("c.cmc = ?", [mv_min], f"mv = {_num(mv_min)}")
        return ("c.cmc BETWEEN ? AND ?", [mv_min, mv_max],
                f"{_num(mv_min)} ≤ mv ≤ {_num(mv_max)}")
    if mv_max is not None:
        return ("c.cmc <= ?", [mv_max], f"mv ≤ {_num(mv_max)}")
    return ("c.cmc >= ?", [mv_min], f"mv ≥ {_num(mv_min)}")


def router_mv_predicate(
    op: str | None, value: float | None, lo: float | None, hi: float | None,
    notes: list[str],
) -> tuple[str, list, str] | None:
    """`op` + `value` (or `lo`/`hi` for "between"). The only path that can
    invert a comparison — see the module docstring — which is exactly why its
    result is always echoed with the real operator symbol it used."""
    if op not in MV_OPS:
        return None
    if op == "between":
        lo, hi = _guard(lo, notes, "mv_lo"), _guard(hi, notes, "mv_hi")
        if lo is None or hi is None:
            return None
        if lo > hi:
            lo, hi = hi, lo
        return ("c.cmc BETWEEN ? AND ?", [lo, hi], f"{_num(lo)} ≤ mv ≤ {_num(hi)}")
    value = _guard(value, notes, "mv_value")
    if value is None:
        return None
    sql_op = "==" if op == "=" else op
    return (f"c.cmc {sql_op} ?", [value], f"mv {_SYMBOL[op]} {_num(value)}")


def _mv_predicate(f: Filters, notes: list[str]) -> tuple[str, list, str] | None:
    if f.mv_op is not None:
        return router_mv_predicate(f.mv_op, f.mv_value, f.mv_lo, f.mv_hi, notes)
    return explicit_mv_predicate(f.mv_min, f.mv_max, notes)


# ------------------------------------------------------------------- per-field sets


def _types_ids(conn: sqlite3.Connection, types: tuple[str, ...]) -> set[str] | None:
    """UNION within the field: matches supertype, type OR subtype."""
    if not types:
        return None
    values = [t.strip().lower() for t in types if t.strip()]
    if not values:
        return None
    marks = ",".join("?" * len(values))
    rows = conn.execute(
        f"SELECT DISTINCT oracle_id FROM card_types WHERE value IN ({marks})", values
    )
    return {r[0] for r in rows}


def _legal_ids(conn: sqlite3.Connection, legal: tuple[str, ...]) -> set[str] | None:
    if not legal:
        return None
    values = [f.strip().lower() for f in legal if f.strip()]
    if not values:
        return None
    marks = ",".join("?" * len(values))
    rows = conn.execute(
        f"SELECT DISTINCT oracle_id FROM card_legalities "
        f"WHERE status = 'legal' AND format IN ({marks})",
        values,
    )
    return {r[0] for r in rows}


def _colors_ids(conn: sqlite3.Connection, colors: str | None) -> set[str] | None:
    """`color_identity ⊆ requested` — the Commander deck-legality rule.

    Matches `cts/search.py::post_filter`'s `color_set(...) <= wanted` exactly,
    so `colors` has one meaning across the whole bot. Computed in Python
    because a subset-of-a-short-string test has no SQL expression cheaper
    than reading the column, and the corpus is ~33,000 rows either way.
    """
    if colors is None:
        return None
    wanted = {c for c in colors.upper() if c in WUBRG}
    if not wanted:
        return None
    return {
        row[0]
        for row in conn.execute("SELECT oracle_id, color_identity FROM cards")
        if set(row[1] or "") <= wanted
    }


def field_sets(
    conn: sqlite3.Connection, f: Filters, notes: list[str]
) -> dict[str, set[str]]:
    """Every non-empty field's own matching set, computed independently."""
    sets: dict[str, set[str]] = {}
    types = _types_ids(conn, f.types)
    if types is not None:
        sets[f"type = {', '.join(sorted(t.strip().lower() for t in f.types if t.strip()))}"] = types
    legal = _legal_ids(conn, f.legal)
    if legal is not None:
        sets[f"legal = {', '.join(sorted(t.strip().lower() for t in f.legal if t.strip()))}"] = legal
    colors = _colors_ids(conn, f.colors)
    if colors is not None:
        sets[f"colors ⊆ {{{f.colors.upper()}}} (identity fits inside)"] = colors
    predicate = _mv_predicate(f, notes)   # the ONE guard check that gets to notify
    if predicate is not None:
        sql, params, label = predicate
        rows = conn.execute(f"SELECT oracle_id FROM cards c WHERE {sql}", params)
        sets[label] = {r[0] for r in rows}
    return sets


# ------------------------------------------------------------------------- combining


def compile_hard(
    conn: sqlite3.Connection, f: Filters, notes: list[str]
) -> set[str] | None:
    """AND every provided field exactly. None means "no filters at all" — the
    caller must not restrict the pool. May return an empty set: that is a
    legitimate, correct answer to a precise question, never widened."""
    sets = field_sets(conn, f, notes)
    if not sets:
        return None
    allowed: set[str] | None = None
    for matched in sets.values():
        allowed = matched if allowed is None else allowed & matched
    return allowed


def compile_soft(
    conn: sqlite3.Connection, f: Filters, notes: list[str], *, base: set[str] | None = None
) -> set[str] | None:
    """Same fields, applied broadest-first; the one field that would zero the
    combined pool is dropped, reported, and nothing else. `MIN_FILTERED_POOL`
    is deliberately not ported from the art side — see the module docstring.

    `base` is an already-decided hard-filter pool (from `compile_hard`), so a
    soft field is judged against the *real* combined pool rather than against
    the whole corpus in isolation — a soft filter that looks fine alone but
    would zero out once ANDed with the user's explicit filters must still be
    dropped and reported, not silently applied to a pool of nothing.
    """
    sets = field_sets(conn, f, notes)
    if not sets:
        return base
    allowed: set[str] | None = base
    kept: list[str] = []
    for label, matched in sorted(sets.items(), key=lambda kv: -len(kv[1])):
        merged = matched if allowed is None else allowed & matched
        if not merged:
            where = "on its own" if allowed is None else "with the filters already applied"
            note = (
                f"inferred filter {label} matched {len(matched)} cards but would leave 0 "
                f"{where}; dropped — the retriever ranks on it instead"
            )
            notes.append(note)
            continue
        allowed = merged
        kept.append(label)
    return allowed


# --------------------------------------------------------------------------- the echo


def echo_line(f: Filters, semantic: str | None) -> str:
    """`filters: type = enchantment · colors ⊆ {G} (identity fits inside) · mv ≤ 5
    · semantic: "let me draw"`. First line of every result set, always — a
    mis-parse is only recognisable against the habit of reading a correct one.

    Takes no `notes` list: this only reflects what the guard *decided* (a
    dropped value simply does not appear), it never re-announces the decision
    — `compile_hard`/`compile_soft`, called separately, already did that.
    """
    parts: list[str] = []
    if f.types:
        parts.append("type = " + ", ".join(sorted(t.strip().lower() for t in f.types if t.strip())))
    if f.colors:
        parts.append(f"colors ⊆ {{{f.colors.upper()}}} (identity fits inside)")
    predicate = _mv_predicate(f, [])  # a throwaway list: see the docstring above
    if predicate is not None:
        parts.append(predicate[2])
    if f.legal:
        parts.append("legal = " + ", ".join(sorted(t.strip().lower() for t in f.legal if t.strip())))

    line = "filters: " + (" · ".join(parts) if parts else "none")
    line += f' · semantic: "{semantic}"' if semantic else " · semantic: none"
    return line


# ------------------------------------------------------------------------ scryfall url

_SCRYFALL_SEARCH = "https://scryfall.com/search?q="


def scryfall_url(f: Filters) -> str | None:
    """A Scryfall search reproducing the structured half, in Scryfall's own
    syntax — a second, independent rendering of the same parse. `colors`
    always emits the SUBSET form `id:g`, never `id>=g` (the superset form):
    Scryfall's bare `id:g` already means `id<=g`, so this is a genuine
    cross-check and not a second, quietly different query.
    """
    terms: list[str] = []
    if f.types:
        values = sorted(t.strip().lower() for t in f.types if t.strip())
        clause = " or ".join(f"t:{v}" for v in values)
        terms.append(f"({clause})" if len(values) > 1 else clause)
    if f.colors:
        terms.append(f"id:{f.colors.lower()}")
    if f.legal:
        values = sorted(t.strip().lower() for t in f.legal if t.strip())
        clause = " or ".join(f"legal:{v}" for v in values)
        terms.append(f"({clause})" if len(values) > 1 else clause)

    notes: list[str] = []
    predicate = _mv_predicate(f, notes)
    if predicate is not None:
        _, params, _ = predicate
        if len(params) == 2:
            lo, hi = params
            terms.append(f"cmc>={_num(lo)}")
            terms.append(f"cmc<={_num(hi)}")
        else:
            if f.mv_op is not None:
                op = f.mv_op                # "<","<=","=",">=",">"; scryfall spells them the same
            elif f.mv_max is not None:
                op = "<="
            elif f.mv_min is not None:
                op = ">="
            else:
                op = "="
            terms.append(f"cmc{op}{_num(params[0])}")

    if not terms:
        return None
    return _SCRYFALL_SEARCH + quote(" ".join(terms))
