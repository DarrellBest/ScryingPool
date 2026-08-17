"""The serving layer: a loopback HTTP API in front of `cts.search.execute`.

Optional and separate from the search engine on purpose. `cts/` runs on three
dependencies (`requests`, `numpy`, `rank_bm25`); everything this package needs
lives in `serve-requirements.txt` and nothing under `cts/` imports it.
"""
