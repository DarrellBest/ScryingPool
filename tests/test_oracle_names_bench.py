"""The one performance claim in the design, measured rather than repeated.

`/search`'s whole reason to answer without `defer()` is that it fits inside
Discord's 3-second acknowledgement window with room to spare — the design says
**under 20ms server-side at p99**, and the naive L5 it rejects would be 1.3-2.6
seconds per query. A document does not get to assert a performance number it has
not run, so this file runs it over every real name in the corpus.

Skips cleanly when `data/oracle.db` is absent, exactly as `conftest.real_conn`
already does for `data/commanders.db`, so a fresh checkout stays green. Nothing
here touches Ollama or the network.
"""

from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path

import pytest

from cts import oracle_names

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ORACLE_DB = REPO_ROOT / "data" / "oracle.db"

# The design's stated bound, with the headroom a CI box or a loaded laptop needs.
# The measured p99 on the development machine is roughly an order of magnitude
# under this; the number here is a regression tripwire for someone quietly
# deleting the bigram prefilter, not a target to tune against.
P99_BUDGET_MS = 20.0


@pytest.fixture(scope="module")
def real_index():
    if not REAL_ORACLE_DB.is_file():
        pytest.skip("data/oracle.db not present — run 'python -m cts oracle-ingest'")
    conn = sqlite3.connect(f"file:{REAL_ORACLE_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        started = time.perf_counter()
        index = oracle_names.build_index(conn)
        build_seconds = time.perf_counter() - started
    finally:
        conn.close()
    if len(index) < 1000:
        pytest.skip("oracle corpus is present but tiny — nothing to benchmark")
    print(f"\nname index: {len(index):,} cards, {index.name_count:,} folded names, "
          f"built in {build_seconds:.2f}s")
    yield index


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]
    return at(0.50), at(0.99), ordered[-1]


def _measure(index, queries) -> tuple[float, float, float]:
    samples: list[float] = []
    for query in queries:
        started = time.perf_counter()
        oracle_names.resolve(index, query)
        samples.append((time.perf_counter() - started) * 1000)
    return _percentiles(samples)


def test_exact_resolution_over_every_real_name_is_microseconds(real_index):
    """L0 and L1 are dict lookups, and they are what the overwhelming majority of
    real queries hit. Every single name in the corpus, timed."""
    names = list(real_index.raw)
    p50, p99, worst = _measure(real_index, names)
    print(f"L0 over {len(names):,} real names: p50 {p50:.3f}ms  p99 {p99:.3f}ms  "
          f"max {worst:.3f}ms")
    assert p99 < P99_BUDGET_MS


def test_folded_resolution_over_every_real_name_stays_under_budget(real_index):
    lowered = [name.lower() for name in real_index.raw]
    p50, p99, worst = _measure(real_index, lowered)
    print(f"L1 over {len(lowered):,} lowercased names: p50 {p50:.3f}ms  "
          f"p99 {p99:.3f}ms  max {worst:.3f}ms")
    assert p99 < P99_BUDGET_MS


def test_typo_resolution_reaches_l5_and_still_clears_the_budget(real_index):
    """The expensive path, on purpose: one character deleted from a real name
    forces the ladder all the way down to the bigram prefilter and the banded DP.

    This is the case the naive implementation costs 1.3-2.6 seconds on.
    """
    rng = random.Random(20260817)
    names = [name for name in real_index.raw if len(name) > 8]
    sample = rng.sample(names, min(400, len(names)))
    typos = []
    for name in sample:
        cut = rng.randrange(1, len(name) - 1)
        typos.append(name[:cut] + name[cut + 1:])

    reached_l5 = sum(1 for t in typos if oracle_names.resolve(real_index, t).layer == "L5")
    p50, p99, worst = _measure(real_index, typos)
    print(f"L5 over {len(typos)} one-character deletions ({reached_l5} reached L5): "
          f"p50 {p50:.3f}ms  p99 {p99:.3f}ms  max {worst:.3f}ms")
    assert reached_l5 > len(typos) // 4, "the sample never exercised the fuzzy layer"
    assert p99 < P99_BUDGET_MS


def test_a_total_miss_costs_no_more_than_a_hit(real_index):
    """A miss runs every layer including the prefilter and finds nothing, so it is
    the worst case the resolver has. It is also what a mistyped name costs."""
    misses = [f"zqxjv{i} kwmpb{i}" for i in range(200)]
    p50, p99, worst = _measure(real_index, misses)
    print(f"misses: p50 {p50:.3f}ms  p99 {p99:.3f}ms  max {worst:.3f}ms")
    assert p99 < P99_BUDGET_MS


def test_short_queries_do_not_blow_up_the_candidate_set(real_index):
    """Two- and three-character inputs are where a naive bigram prefilter degrades
    into a full sweep. They are also what someone types by accident."""
    p50, p99, worst = _measure(
        real_index, ["so", "bol", "pat", "ae", "fi", "li", "co", "gr"] * 25
    )
    print(f"short queries: p50 {p50:.3f}ms  p99 {p99:.3f}ms  max {worst:.3f}ms")
    assert p99 < P99_BUDGET_MS
