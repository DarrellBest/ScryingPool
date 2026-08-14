"""Phase 3: composite power score.

EDHREC rank measures popularity, not power, so the score is a weighted blend of
four components stored alongside it in `power.components`. Weights live in
config, and because both the components and the raw inputs are persisted, they
can be retuned by re-running this stage without re-fetching anything.

Components, all in [0, 1]:

  log_decks_norm  log1p(EDHREC deck count), min-max normalised over the corpus.
  price_pct       percentile of the commander's price among every priced card.
  cmc_norm        the commander's own mana value, INVERTED: 1.0 is the cheapest
                  card in the corpus and 0.0 the most expensive, because cheap
                  commanders do more, sooner.
  cedh            1 when EDHREC reports at least one cEDH-tagged deck for this
                  commander, else 0.

    score = w.deck_count * log_decks_norm
          + w.price      * price_pct
          + w.cmc        * cmc_norm
          + w.cedh       * cedh

A caveat on `cedh`, worth knowing before tuning its weight. SPEC.md and
config.toml both describe the cEDH flag as "rare but highly informative". It is
not rare: EDHREC tags a handful of cEDH decks for very nearly every commander
that has a page at all (Atraxa 57, Krenko 103, Kroxa 20, Beluna 1), so a literal
presence flag is 1 for almost the whole corpus and mostly re-states deck count.
The flag is still computed literally, because redefining it silently would be
worse — but `cedh_share` and `bracket5_share` (bracket 5 is the official cEDH
bracket) go into the components next to it, so switching the scored term to a
continuous signal is a one-line change here and needs no re-fetch.
`power.run` prints what fraction of the corpus the flag fired on.

No banding here. SPEC.md wants five quantile bands computed at query time over
the whole distribution, so nothing about a band is written to disk.

Every row is recomputed on every run: the normalisations are relative to the
corpus, so one new commander shifts every other score.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from . import db
from .config import Config

_BATCH = 500


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A flat corpus (or a corpus of one) scores zero."""
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _percentile(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value among the values that exist.

    Cards with no price sit at 0.0 — the same floor a card with no EDHREC row
    gets for deck count. Ties share the midpoint of the ranks they span, so a
    corpus where half the cards cost exactly $1 does not hand them all a
    different percentile.
    """
    out = np.zeros_like(values, dtype=np.float64)
    known = values[present]
    if known.size == 0:
        return out
    ordered = np.sort(known)
    low = np.searchsorted(ordered, known, side="left")
    high = np.searchsorted(ordered, known, side="right")
    out[present] = (low + high) / (2.0 * ordered.size)
    return out


def _cedh_decks(themes_json: str | None) -> int:
    """Decks tagged cEDH, from the EDHREC tag list stored in `edhrec.themes`.

    The tag looks like {"count": 57, "slug": "cedh", "value": "cEDH"} and was
    present on every live page checked. `edhrec.raw` also carries
    `$.bracket_counts` where bracket 5 is cEDH, if a future weighting wants a
    continuous signal instead of this flag.
    """
    if not themes_json:
        return 0
    try:
        themes = json.loads(themes_json)
    except (TypeError, ValueError):
        return 0
    if not isinstance(themes, list):
        return 0
    for tag in themes:
        if isinstance(tag, dict) and tag.get("slug") == "cedh":
            count = tag.get("count")
            return int(count) if isinstance(count, (int, float)) else 0
    return 0


def _bracket5(brackets_json: str | None) -> int:
    """Decks in Commander bracket 5, which is cEDH by definition.

    Pulled out of `edhrec.raw` at `$.bracket_counts` by SQLite rather than
    loading the ~120KB blob per row into Python.
    """
    if not brackets_json:
        return 0
    try:
        brackets = json.loads(brackets_json)
    except (TypeError, ValueError):
        return 0
    if not isinstance(brackets, dict):
        return 0
    count = brackets.get("5")
    return int(count) if isinstance(count, (int, float)) else 0


_SELECT = """
SELECT c.oracle_id AS oracle_id,
       c.name      AS name,
       c.cmc       AS cmc,
       e.num_decks AS num_decks,
       e.avg_price AS avg_price,
       e.themes    AS themes,
       {brackets}  AS brackets
FROM cards c
LEFT JOIN edhrec e ON e.oracle_id = c.oracle_id
ORDER BY c.oracle_id
"""


def _fetch_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Rows for scoring, with bracket_counts if this SQLite has JSON1."""
    try:
        return conn.execute(
            _SELECT.format(brackets="json_extract(e.raw, '$.bracket_counts')")
        ).fetchall()
    except sqlite3.OperationalError:
        return conn.execute(_SELECT.format(brackets="NULL")).fetchall()

_UPSERT = """
INSERT INTO power (oracle_id, score, components)
VALUES (?, ?, ?)
ON CONFLICT(oracle_id) DO UPDATE SET
  score = excluded.score, components = excluded.components
"""


def run(cfg: Config) -> dict:
    """Phase 3. Always recomputes every row."""
    conn = db.connect(cfg)
    try:
        rows = _fetch_rows(conn)
        n = len(rows)
        if not n:
            print("power: no cards to score", flush=True)
            return {"scored": 0}

        weights = cfg.power_weights
        w_decks = float(weights.get("deck_count", 0.0))
        w_price = float(weights.get("price", 0.0))
        w_cmc = float(weights.get("cmc", 0.0))
        w_cedh = float(weights.get("cedh", 0.0))

        decks = np.array(
            [float(r["num_decks"]) if r["num_decks"] is not None else 0.0 for r in rows],
            dtype=np.float64,
        )
        prices = np.array(
            [float(r["avg_price"]) if r["avg_price"] is not None else 0.0 for r in rows],
            dtype=np.float64,
        )
        has_price = np.array([r["avg_price"] is not None for r in rows], dtype=bool)
        cmcs = np.array(
            [float(r["cmc"]) if r["cmc"] is not None else 0.0 for r in rows],
            dtype=np.float64,
        )
        cedh_decks = np.array([_cedh_decks(r["themes"]) for r in rows], dtype=np.int64)
        bracket5 = np.array([_bracket5(r["brackets"]) for r in rows], dtype=np.int64)
        denom = np.where(decks > 0, decks, 1.0)

        log_decks_norm = _minmax(np.log1p(decks))
        price_pct = _percentile(prices, has_price)
        cmc_norm = 1.0 - _minmax(cmcs)  # inverted: cheaper commander = stronger
        cedh = (cedh_decks > 0).astype(np.float64)

        score = (
            w_decks * log_decks_norm
            + w_price * price_pct
            + w_cmc * cmc_norm
            + w_cedh * cedh
        )

        payload = []
        for i, row in enumerate(rows):
            components = {
                # the four weighted components
                "log_decks_norm": round(float(log_decks_norm[i]), 6),
                "price_pct": round(float(price_pct[i]), 6),
                "cmc_norm": round(float(cmc_norm[i]), 6),
                "cedh": int(cedh[i]),
                # raw inputs, so weights can be retuned from stored data alone
                "num_decks": int(decks[i]),
                "avg_price": float(prices[i]) if has_price[i] else None,
                "cmc": float(cmcs[i]),
                # unscored, but kept so the cEDH term can be made continuous
                # without re-fetching anything. See the module docstring.
                "cedh_decks": int(cedh_decks[i]),
                "cedh_share": round(float(cedh_decks[i] / denom[i]), 6),
                "bracket5_decks": int(bracket5[i]),
                "bracket5_share": round(float(bracket5[i] / denom[i]), 6),
                "weights": {
                    "deck_count": w_decks,
                    "price": w_price,
                    "cmc": w_cmc,
                    "cedh": w_cedh,
                },
            }
            payload.append((row["oracle_id"], float(score[i]), json.dumps(components)))

        for start in range(0, len(payload), _BATCH):
            conn.executemany(_UPSERT, payload[start : start + _BATCH])
            conn.commit()
            print(f"  scored {min(start + _BATCH, len(payload)):,}/{n:,}", flush=True)

        with_edhrec = int(np.count_nonzero(decks > 0))
        flagged = int(cedh.sum())
        print(
            f"power: scored {n:,} commanders "
            f"({with_edhrec:,} with EDHREC decks, {flagged:,} cEDH-tagged, "
            f"{int(has_price.sum()):,} priced)",
            flush=True,
        )
        if n and flagged / n > 0.8:
            print(
                f"power: note - the cEDH flag fired on {100 * flagged / n:.0f}% of the "
                f"corpus, so it barely separates anything; components carry "
                f"cedh_share and bracket5_share if you want a continuous term",
                flush=True,
            )
        print(
            f"power: score min {score.min():.4f} / median {float(np.median(score)):.4f} "
            f"/ max {score.max():.4f} (bands are quantiles at query time)",
            flush=True,
        )
        return {"scored": n}
    finally:
        conn.close()
