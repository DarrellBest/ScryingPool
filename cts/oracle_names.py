"""Local card-name resolution for `/search`: fold, then a six-layer ladder.

Everything here is in-memory and stdlib. No Scryfall API call is made at query
time, ever, not even as a fallback — their rate-limits page says card-name
lookup *"must use the bulk data files"*, and a per-invocation `/cards/named` at
2 req/s is exactly the usage that prohibits.

The ladder
----------

    L0  exact, raw bytes          dict            paste from Scryfall, "//" and diacritics
    L1  exact on fold(name)       dict            case, punctuation, accents, Æ
    L2  exact on a folded FACE    dict            "Petty Theft", "Ice"
    L3  folded prefix             bisect          "atraxa praetors", truncated typing
    L4  all query tokens present  token index     "voice atraxa" — wrong order
    L5  bounded edit distance     bigram + DP     genuine typos: "lightnig bolt"

**Each layer fires only if every layer above it returned nothing, and that strict
short-circuit is the correctness property of this module.** The dangerous
behaviour in any fuzzy resolver is an eager fuzzy layer overriding an exact
match — quietly "correcting" a correctly-typed rare card into a more popular
near-neighbour. Under strict ordering that is structurally impossible: a name
with an L1 hit never reaches L5, so `Ancestral Recall` can never become
`Ancestral Vision`, and `Fire` resolves at L2 to `Fire // Ice` rather than being
prefix-matched into `Fireball`.

`edhrec_rank` (lower is more popular, NULL for many cards) breaks ties **only
within the single layer that fired**, never across layers. NULLs sort last.

Why not FTS5
------------
~32,700 names average ~20 characters — about 650KB of text, so an in-memory
index is trivially affordable and FTS5's reason to exist does not apply. FTS5 is
also a compile-time SQLite option (a missing module would be a hard deployment
failure), a second structure the weekly refresh must keep in sync, and it does
not do the hard part: it has prefix and token matching — layers 3 and 4, the
easy ones — and no edit distance at all, so L5 would still be hand-written.

Why L5 prunes before it measures
--------------------------------
Pure-Python Levenshtein over two ~20-character strings is ~400 cell operations,
call it 40-80µs; sweeping all 32,726 names is 1.3-2.6 seconds per query, paid on
exactly the queries that are already going badly. So L5 prunes first: a bigram
inverted index (Dice ≥ 0.4, top 200 by overlap), a length band (edit distance is
at least the length difference), and a banded, early-terminating DP that only
computes the 2k+1 diagonal and aborts a pair as soon as a row's minimum exceeds
the budget. `tests/test_oracle_names_bench.py` measures the result against the
real corpus rather than trusting this paragraph.
"""

from __future__ import annotations

import bisect
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- folding

# NFKD decomposes accented letters into a base plus a combining mark, which is
# how "Juzám" becomes "juzam". It does **not** decompose these: they are atomic
# letters in Unicode, not composed ones, so `Æ` survives NFKD untouched and
# `Ærathi Berserker` would be unreachable by anyone typing "Aerathi" without an
# explicit map. Verified rather than assumed — the test asserts NFKD alone is
# insufficient, so nobody deletes this table as redundant.
LIGATURES = {
    "Æ": "ae", "æ": "ae",
    "Œ": "oe", "œ": "oe",
    "ß": "ss",
    "Ø": "o", "ø": "o",
    "Þ": "th", "þ": "th",
    "Ð": "d", "ð": "d",
    "Đ": "d", "đ": "d",
    "Ł": "l", "ł": "l",
}

# Deleted, not replaced with a space, so "Gaea's" folds to "gaeas" and a user
# typing "Gaeas Cradle" matches. This is the convention cts/links.py already
# established for EDHREC slugs — one folding habit across the repo, not two.
APOSTROPHES = "'’ʼ‘`"


def fold(name: str) -> str:
    """The matching key. Consistent between the stored name and the typed one.

    `Lim-Dûl's Vault` -> `lim duls vault`, `Æther Vial` -> `aether vial`,
    `Fire // Ice` -> `fire ice`. The output is not pretty; it only has to be the
    same on both sides, which it is.
    """
    text = unicodedata.normalize("NFKD", str(name))
    out: list[str] = []
    for ch in text:
        if unicodedata.combining(ch):
            continue                      # the accent NFKD just split off
        if ch in LIGATURES:
            out.append(LIGATURES[ch])
        elif ch in APOSTROPHES:
            continue                      # deleted, never spaced
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append(" ")               # includes "//", commas, hyphens
    return " ".join("".join(out).lower().split())


def tokens(folded: str) -> tuple[str, ...]:
    return tuple(folded.split())


def bigrams(folded: str) -> set[str]:
    """2-grams of a folded name, spaces included so word boundaries count.

    A one-character name has no 2-gram, so it contributes itself; without that
    it would be invisible to the L5 prefilter entirely.
    """
    squashed = folded
    if len(squashed) < 2:
        return {squashed} if squashed else set()
    return {squashed[i : i + 2] for i in range(len(squashed) - 1)}


# ------------------------------------------------------------------- edit distance


def max_distance_for(query: str) -> int:
    """How wrong an input is allowed to be, scaled to its length.

    1 for ≤4 characters, 2 for 5-8, 3 for 9+, and never more than 30% of the
    input. Without the scaling, `Bolt` sits within distance 4 of a large slice of
    the corpus and L5 would answer confidently with noise.
    """
    length = len(query)
    if length <= 4:
        base = 1
    elif length <= 8:
        base = 2
    else:
        base = 3
    return max(1, min(base, int(length * 0.3)))


def bounded_distance(a: str, b: str, budget: int) -> int | None:
    """Levenshtein(a, b) when it is ≤ `budget`, else None.

    Banded: only the `2*budget+1` diagonal around the main one can hold a value
    within budget, so the rest of the matrix is never computed. Early
    termination: a row whose minimum already exceeds the budget cannot recover,
    because distance is non-decreasing down the rows.
    """
    if abs(len(a) - len(b)) > budget:
        return None
    if a == b:
        return 0
    if not a:
        return len(b) if len(b) <= budget else None
    if not b:
        return len(a) if len(a) <= budget else None

    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        # Only columns within the band matter; everything outside it is already
        # further than `budget` from the diagonal.
        low = max(1, i - budget)
        high = min(len(b), i + budget)
        current = [0] * (len(b) + 1)
        current[0] = i
        row_min = i
        if low > 1:
            current[low - 1] = budget + 1
        for j in range(low, high + 1):
            cost = 0 if ch_a == b[j - 1] else 1
            value = min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + cost,   # substitution
            )
            current[j] = value
            if value < row_min:
                row_min = value
        for j in range(high + 1, len(b) + 1):
            current[j] = budget + 1
        if row_min > budget:
            return None
        previous = current

    result = previous[len(b)]
    return result if result <= budget else None


# ------------------------------------------------------------------------- the index

# How many candidates a layer ever hands back. Above this the response is a
# truncated list plus the true total ("41 cards match `bolt` — showing the 10
# most played"), which is why `Resolution.total` is separate from `oracle_ids`.
MAX_CANDIDATES = 10

# L5 prefilter knobs. Dice ≥ 0.4 over bigram sets, then the 200 best by raw
# overlap. Both are ceilings on work, not on quality: anything they cut is far
# enough away that the banded DP would have rejected it anyway.
DICE_FLOOR = 0.4
PREFILTER_CAP = 200


@dataclass(frozen=True)
class Resolution:
    """What the ladder decided, and which rung decided it.

    `layer` is None only on a genuine miss. `total` is how many cards the firing
    layer matched *before* truncation to `MAX_CANDIDATES`, so the caller can say
    "41 cards match" while showing ten.
    """

    query: str
    layer: str | None
    oracle_ids: tuple[str, ...]
    total: int = 0
    distance: int | None = None
    matched_name: str | None = None

    @property
    def resolved(self) -> bool:
        """Exactly one card. Two is an ambiguity, not a weaker success."""
        return self.total == 1 and len(self.oracle_ids) == 1

    @property
    def oracle_id(self) -> str | None:
        return self.oracle_ids[0] if self.resolved else None

    @property
    def exact(self) -> bool:
        """L0-L2 matched what the user actually typed; L3-L5 reinterpreted it."""
        return self.layer in ("L0", "L1", "L2")


class NameIndex:
    """Every structure the ladder needs, built once from `cards` + `card_faces`.

    ~60-80MB against the API's ~1.8GB steady state, so it is simply resident.
    Rebuilt from the same oracle fingerprint as everything else.
    """

    def __init__(self) -> None:
        self.raw: dict[str, list[str]] = {}
        self.folded: dict[str, list[str]] = {}
        self.face_folded: dict[str, list[str]] = {}
        self.names: list[str] = []               # sorted, unique folded strings
        self.owners: list[tuple[str, ...]] = []  # parallel to names
        self.token_index: dict[str, set[int]] = {}
        # gram -> (lengths, positions), both sorted by the folded name's length so
        # the length band can be taken as a slice instead of tested per candidate.
        # That band is the difference between scanning a common bigram's whole
        # 5,000-name posting list and scanning the fifth of it that could possibly
        # be within `max_distance` edits.
        self.bigram_index: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        self.rank: dict[str, int | None] = {}
        self.display: dict[str, str] = {}
        self.build_seconds: float = 0.0

    # ------------------------------------------------------------------ construction

    def __len__(self) -> int:
        return len(self.rank)

    @property
    def card_count(self) -> int:
        return len(self.rank)

    @property
    def name_count(self) -> int:
        return len(self.names)

    def _order(self, oracle_ids: Iterable[str]) -> tuple[str, ...]:
        """Popularity tie-break, applied only within the layer that fired.

        NULL `edhrec_rank` sorts last — a card EDHREC has never seen is not more
        played than one ranked 30,000th, it is unranked.
        """
        unique = list(dict.fromkeys(oracle_ids))
        return tuple(
            sorted(
                unique,
                key=lambda oid: (
                    self.rank.get(oid) is None,
                    self.rank.get(oid) if self.rank.get(oid) is not None else 0,
                    self.display.get(oid, ""),
                ),
            )
        )


def build_index(conn: sqlite3.Connection) -> NameIndex:
    """Read every name and face name out of the oracle corpus into a NameIndex."""
    import time

    started = time.monotonic()
    index = NameIndex()

    entries: dict[str, set[str]] = {}   # folded name -> oracle_ids (cards + faces)

    for row in conn.execute(
        "SELECT oracle_id, name, name_norm, edhrec_rank FROM cards"
    ):
        oracle_id = row["oracle_id"]
        name = row["name"] or ""
        folded_name = row["name_norm"] or fold(name)
        index.rank[oracle_id] = row["edhrec_rank"]
        index.display[oracle_id] = name
        index.raw.setdefault(name, []).append(oracle_id)
        index.folded.setdefault(folded_name, []).append(oracle_id)
        entries.setdefault(folded_name, set()).add(oracle_id)

    for row in conn.execute(
        "SELECT oracle_id, name, name_norm FROM card_faces"
    ):
        oracle_id = row["oracle_id"]
        if oracle_id not in index.rank:
            continue                      # a face whose card was filtered out
        folded_face = row["name_norm"] or fold(row["name"] or "")
        if not folded_face:
            continue
        index.face_folded.setdefault(folded_face, []).append(oracle_id)
        entries.setdefault(folded_face, set()).add(oracle_id)

    index.names = sorted(entries)
    index.owners = [tuple(sorted(entries[name])) for name in index.names]

    postings: dict[str, list[int]] = {}
    for position, folded_name in enumerate(index.names):
        for token in tokens(folded_name):
            index.token_index.setdefault(token, set()).add(position)
        for gram in bigrams(folded_name):
            postings.setdefault(gram, []).append(position)

    for gram, positions in postings.items():
        positions.sort(key=lambda p: len(index.names[p]))
        index.bigram_index[gram] = (
            tuple(len(index.names[p]) for p in positions),
            tuple(positions),
        )

    index.build_seconds = time.monotonic() - started
    return index


# ---------------------------------------------------------------------- the ladder


def _finish(
    index: NameIndex,
    query: str,
    layer: str,
    oracle_ids: Sequence[str],
    *,
    distance: int | None = None,
    matched_name: str | None = None,
) -> Resolution:
    ordered = index._order(oracle_ids)
    return Resolution(
        query=query,
        layer=layer,
        oracle_ids=ordered[:MAX_CANDIDATES],
        total=len(ordered),
        distance=distance,
        matched_name=matched_name,
    )


def _prefix_hits(index: NameIndex, folded_query: str) -> list[str]:
    start = bisect.bisect_left(index.names, folded_query)
    hits: list[str] = []
    for position in range(start, len(index.names)):
        if not index.names[position].startswith(folded_query):
            break
        hits.extend(index.owners[position])
    return hits


def _token_hits(index: NameIndex, folded_query: str) -> list[str]:
    wanted = tokens(folded_query)
    if not wanted:
        return []
    postings = []
    for token in wanted:
        found = index.token_index.get(token)
        if not found:
            return []                     # one missing token means no subset match
        postings.append(found)
    postings.sort(key=len)                # intersect the rarest token first
    common = set(postings[0])
    for other in postings[1:]:
        common &= other
        if not common:
            return []
    hits: list[str] = []
    for position in common:
        hits.extend(index.owners[position])
    return hits


def _fuzzy_hits(
    index: NameIndex, folded_query: str
) -> tuple[list[str], int | None, str | None]:
    """L5: prefilter hard, then measure. Returns (ids, best distance, best name)."""
    budget = max_distance_for(folded_query)
    query_grams = bigrams(folded_query)
    if not query_grams:
        return [], None, None

    # Edit distance is at least the length difference, so anything outside this
    # band is excluded for free — and because each posting list is sorted by name
    # length, "excluded for free" means never being looked at at all.
    low, high = len(folded_query) - budget, len(folded_query) + budget

    overlap: dict[int, int] = {}
    for gram in query_grams:
        entry = index.bigram_index.get(gram)
        if entry is None:
            continue
        lengths, positions = entry
        start = bisect.bisect_left(lengths, low)
        stop = bisect.bisect_right(lengths, high)
        for position in positions[start:stop]:
            overlap[position] = overlap.get(position, 0) + 1

    scored: list[tuple[int, int]] = []
    for position, shared in overlap.items():
        size = max(1, len(index.names[position]) - 1)
        dice = 2 * shared / (len(query_grams) + size)
        if dice >= DICE_FLOOR:
            scored.append((shared, position))

    scored.sort(key=lambda pair: -pair[0])
    best: int | None = None
    hits: list[str] = []
    matched: str | None = None
    for _, position in scored[:PREFILTER_CAP]:
        distance = bounded_distance(folded_query, index.names[position], budget)
        if distance is None:
            continue
        if best is None or distance < best:
            best, hits, matched = distance, list(index.owners[position]), index.names[position]
        elif distance == best:
            hits.extend(index.owners[position])
    return hits, best, matched


LAYERS = ("L0", "L1", "L2", "L3", "L4", "L5")


def resolve(index: NameIndex, query: str, *, max_layer: int = 5) -> Resolution:
    """Walk the ladder top-down and stop at the first rung that matches.

    `max_layer` exists for `/oracle`'s card-name guard, which runs L0-L2 only.
    Letting that guard reach the fuzzy layers would make it fire on genuine
    mechanical queries that merely sit near a card name — "counter target spell"
    is two edit operations from *Counterspell* — and refuse to search at all. A
    guard that hijacks real queries is worse than no guard.
    """
    raw_query = str(query).strip()
    if not raw_query:
        return Resolution(query=raw_query, layer=None, oracle_ids=(), total=0)

    # L0 — exact bytes. Someone pasted a name out of Scryfall, "//" and all.
    if max_layer >= 0 and raw_query in index.raw:
        return _finish(index, raw_query, "L0", index.raw[raw_query])

    folded_query = fold(raw_query)
    if not folded_query:
        return Resolution(query=raw_query, layer=None, oracle_ids=(), total=0)

    # L1 — folded exact. Case, punctuation, accents, Æ. Most real queries end here.
    if max_layer >= 1 and folded_query in index.folded:
        return _finish(index, raw_query, "L1", index.folded[folded_query],
                       matched_name=folded_query)

    # L2 — a folded FACE name. "Petty Theft" is not a card, it is half of one.
    if max_layer >= 2 and folded_query in index.face_folded:
        return _finish(index, raw_query, "L2", index.face_folded[folded_query],
                       matched_name=folded_query)

    # L3 — folded prefix. Truncated typing, and "atraxa praetors".
    if max_layer >= 3:
        hits = _prefix_hits(index, folded_query)
        if hits:
            return _finish(index, raw_query, "L3", hits)

    # L4 — every query token present, in any order. "voice atraxa".
    if max_layer >= 4:
        hits = _token_hits(index, folded_query)
        if hits:
            return _finish(index, raw_query, "L4", hits)

    # L5 — bounded edit distance. Genuine typos, and nothing else by then.
    if max_layer >= 5:
        hits, distance, matched = _fuzzy_hits(index, folded_query)
        if hits:
            return _finish(index, raw_query, "L5", hits,
                           distance=distance, matched_name=matched)

    return Resolution(query=raw_query, layer=None, oracle_ids=(), total=0)


# -------------------------------------------------------------------- reading a card


def _links(row: sqlite3.Row | dict) -> dict[str, str]:
    """`scryfall` / `edhrec` / `tcgplayer`, each omitted entirely when absent.

    EDHREC and TCGplayer come from Scryfall's own `related_uris.edhrec` and
    `purchase_uris.tcgplayer`, **stored at ingest, never derived from the name**.
    The alternative would have been slugifying card names into EDHREC URLs and
    hoping, and cts/links.py's standing rule is that a reference which cannot be
    built is omitted, never guessed and never emitted as a broken link.
    """
    links: dict[str, str] = {}
    for key, column in (
        ("scryfall", "scryfall_uri"),
        ("edhrec", "related_edhrec"),
        ("tcgplayer", "purchase_tcgplayer"),
    ):
        value = row[column] if column in row.keys() else None
        if value:
            links[key] = str(value)
    return links


def card_payload(conn: sqlite3.Connection, oracle_id: str) -> dict | None:
    """One card, every column, plus its faces, legalities and links."""
    row = conn.execute("SELECT * FROM cards WHERE oracle_id = ?", (oracle_id,)).fetchone()
    if row is None:
        return None

    card = {key: row[key] for key in row.keys()}
    card["faces"] = [
        {key: face[key] for key in face.keys()}
        for face in conn.execute(
            "SELECT face_index, name, mana_cost, type_line, oracle_text, image_normal "
            "FROM card_faces WHERE oracle_id = ? ORDER BY face_index",
            (oracle_id,),
        )
    ]
    card["legalities"] = {
        legality["format"]: legality["status"]
        for legality in conn.execute(
            "SELECT format, status FROM card_legalities WHERE oracle_id = ? ORDER BY format",
            (oracle_id,),
        )
    }
    card["links"] = _links(row)
    return card


def candidate_briefs(conn: sqlite3.Connection, oracle_ids: Sequence[str]) -> list[dict]:
    """Name, mana cost and type line for a disambiguation list, in the given order."""
    if not oracle_ids:
        return []
    marks = ",".join("?" * len(oracle_ids))
    rows = {
        row["oracle_id"]: {key: row[key] for key in row.keys()}
        for row in conn.execute(
            f"SELECT oracle_id, name, mana_cost, type_line, edhrec_rank "
            f"FROM cards WHERE oracle_id IN ({marks})",
            list(oracle_ids),
        )
    }
    return [rows[oid] for oid in oracle_ids if oid in rows]
