"""Shared fakes for the serving-layer tests.

Nothing here imports fastapi, discord or requests, and nothing here touches the
network or Ollama — the test files do the `importorskip` themselves. The point
of the file is that `serve.api.Engine` takes its search callable, index builder
and both probes as constructor arguments, so the tests supply real objects
instead of monkeypatching module globals.
"""

from __future__ import annotations

import copy
import sqlite3
import threading
import time
from dataclasses import dataclass, field

from cts import db, oracle_db, oracle_names
from cts.config import Config

# Real names, chosen for the resolution layers they exercise: an exact hit, a
# face name, a prefix cluster, and a pair one edit apart.
ORACLE_CARDS: tuple[tuple[str, str, int | None, tuple[str, ...]], ...] = (
    ("o-sol", "Sol Ring", 1, ()),
    ("o-cradle", "Gaea's Cradle", 449, ()),
    ("o-borrower", "Brazen Borrower // Petty Theft", 300, ("Brazen Borrower", "Petty Theft")),
    ("o-path-exile", "Path to Exile", 250, ()),
    ("o-path-ancestry", "Path of Ancestry", 100, ()),
    ("o-recall", "Ancestral Recall", 20000, ()),
    ("o-vision", "Ancestral Vision", 6000, ()),
)


def memory_oracle_conn() -> sqlite3.Connection:
    """An in-memory oracle corpus holding ORACLE_CARDS, wired like the real one."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    oracle_db.init_schema(conn)
    for oracle_id, name, rank, faces in ORACLE_CARDS:
        conn.execute(
            "INSERT INTO cards(oracle_id, name, name_norm, type_line, oracle_text, "
            "mana_cost, cmc, color_identity, layout, edhrec_rank, set_code, rarity, "
            "image_normal, price_usd, scryfall_uri, related_edhrec) VALUES "
            "(?, ?, ?, 'Artifact', '{T}: Add {C}{C}.', '{1}', 1.0, '', 'normal', ?, "
            "'msc', 'uncommon', ?, 1.6, ?, ?)",
            (
                oracle_id,
                name,
                oracle_names.fold(name),
                rank,
                f"https://cards.scryfall.io/normal/{oracle_id}.jpg",
                f"https://scryfall.com/card/{oracle_id}",
                f"https://edhrec.com/route/?cc={name.replace(' ', '+')}",
            ),
        )
        for face_index, face_name in enumerate(faces):
            conn.execute(
                "INSERT INTO card_faces(oracle_id, face_index, name, name_norm) "
                "VALUES (?, ?, ?, ?)",
                (oracle_id, face_index, face_name, oracle_names.fold(face_name)),
            )
        conn.execute(
            "INSERT INTO card_legalities(oracle_id, format, status) "
            "VALUES (?, 'commander', 'legal')",
            (oracle_id,),
        )
    oracle_db.meta_set(conn, oracle_db.LAST_REFRESH_KEY, "2026-08-17T03:43:02+00:00")
    conn.commit()
    return conn


@dataclass
class StubOracleBuilder:
    """Stands in for `serve.api.build_name_index`. Counts builds; can be made to fail.

    Takes only a Config, because the real builder opens its own read-only
    connection rather than sharing the one `GET /card` reads from.
    """

    conn: sqlite3.Connection | None = None
    calls: int = 0
    raises: BaseException | None = None

    def __call__(self, cfg) -> oracle_names.NameIndex:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return oracle_names.build_index(self.conn or memory_oracle_conn())


def memory_conn() -> sqlite3.Connection:
    """An empty schema, opened the way the serving connection is opened.

    `check_same_thread=False` matters: the engine reads the fingerprint and
    builds the index through `asyncio.to_thread`, so a default connection would
    raise ProgrammingError instead of testing anything.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    return conn


def config(db_path: str = ":memory:", art_dir: str = "art") -> Config:
    return Config(
        ollama_url="http://localhost:11434",
        vision_model="vision-model",
        verify_model="vision-model",
        embed_model="embed-model",
        judge_model="judge-model",
        db_path=db_path,
        art_dir=art_dir,
        power_weights={"deck_count": 0.4, "price": 0.25, "cmc": 0.2, "cedh": 0.15},
    )


@dataclass
class StubIndex:
    """Only the surface `serve.api` actually reads off a SearchIndex."""

    props: int = 170_487
    artworks: int = 5_530
    dim: int = 768
    build_seconds: float = 5.1
    missing_embeddings: int = 0
    label: str = "first"

    def __len__(self) -> int:
        return self.props

    @property
    def artwork_count(self) -> int:
        return self.artworks


# A trimmed but structurally real `execute()` return value: the keys the bot
# renders, in the shape search.py's _result_dict actually produces.
RESULT = {
    "oracle_id": "e2e0d6d1-0000-4000-8000-000000000001",
    "name": "Avacyn, Angel of Hope",
    "mana_cost": "{5}{W}{W}{W}",
    "type_line": "Legendary Creature — Angel",
    "color_identity": "W",
    "band": 3,
    "fit": 0.82,
    "rationale": "a lone winged figure against a vast empty sky",
    "verified": True,
    "illustration_id": "ill-avacyn",
    "set_code": "avr",
    "artist": "Jason Chan",
    "prop_ids": [11, 12, 13],
    "links": {
        "edhrec": "https://edhrec.com/commanders/avacyn-angel-of-hope",
        "scryfall": "https://scryfall.com/card/avr/6",
        "art_crop": "https://cards.scryfall.io/art_crop/front/x.jpg",
    },
    "stretch": False,
    "vision_rejected": False,
    "verify_note": None,
    "score": 0.91,
    "art_count": 2,
}

OUTCOME = {
    "query_id": 4242,
    "plan": {
        "notes": [],
        "vision_verified": True,
        "literal_weight": 0.3,
        "interpretive_weight": 0.7,
        "counts": {"commanders": 412, "candidates": 100, "judged": 40},
    },
    "relaxed": None,
    "results": [RESULT],
    "pool": [RESULT],
}


def outcome(**plan_overrides) -> dict:
    """A fresh deep copy, so a handler mutating `plan.notes` cannot leak."""
    fresh = copy.deepcopy(OUTCOME)
    fresh["plan"].update(copy.deepcopy(plan_overrides))
    return fresh


@dataclass
class StubSearch:
    """A stand-in for `cts.search.execute` that records how it was called.

    `delay` is a real `time.sleep`, because the thing under test is that the
    call happens on a worker thread rather than on the event loop.
    """

    delay: float = 0.0
    raises: BaseException | None = None
    result: dict | None = None
    calls: list[dict] = field(default_factory=list)
    max_concurrent: int = 0
    _live: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    started: threading.Event = field(default_factory=threading.Event)

    def __call__(self, cfg, query, *, band=None, colors=None, k=5, kind="user",
                 conn=None, index=None) -> dict:
        with self._lock:
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
        self.started.set()
        try:
            self.calls.append(
                {
                    "query": query,
                    "band": band,
                    "colors": colors,
                    "k": k,
                    "kind": kind,
                    "conn": conn,
                    "index": index,
                }
            )
            if self.delay:
                time.sleep(self.delay)
            if self.raises is not None:
                raise self.raises
            return copy.deepcopy(self.result) if self.result is not None else outcome()
        finally:
            with self._lock:
                self._live -= 1


@dataclass
class StubBuilder:
    """Stands in for `cts.index.load_index`. Counts builds; can be made to fail."""

    indexes: list = field(default_factory=list)
    calls: int = 0
    raises: BaseException | None = None

    def __call__(self, cfg, conn) -> StubIndex:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        if self.indexes:
            return self.indexes.pop(0)
        return StubIndex(label=f"build-{self.calls}")


def ollama_ok(loaded=("judge-model",)) -> dict:
    return {
        "url": "http://localhost:11434",
        "reachable": True,
        "missing_models": [],
        "loaded": list(loaded),
        "error": None,
    }


def ollama_down() -> dict:
    return {
        "url": "http://localhost:11434",
        "reachable": False,
        "missing_models": [],
        "loaded": [],
        "error": "connection refused",
    }
