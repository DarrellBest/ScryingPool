"""Fixtures for the offline tests. Nothing here touches Ollama or the network.

The corpus below is small but every string in it is a real value taken out of
data/commanders.db, paired with the real type line of the card it was written for.
That matters: the whole point of slotvocab is that the vision pass writes
"green-skinned humanoid with pointed ears" where the router writes "goblin", so a
fixture with invented values would test nothing.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from cts import db

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DB = REPO_ROOT / "data" / "commanders.db"

# (card name, type line, species, clothing, pose, composition, other figures)
CORPUS: tuple[tuple[str, str, str, str, str, str, list[dict]], ...] = (
    # --- goblins: the vision pass never wrote the word
    ("Rulik Mons, Warren Chief", "Legendary Creature — Goblin",
     "green-skinned humanoid with pointed ears", "leather harness", "standing, arms crossed",
     "medium shot, figure centered", [{"species": "none", "role": "none"}]),
    ("Green Goblin, Nemesis", "Legendary Creature — Goblin Human Villain",
     "green-skinned humanoid with pointed ears", "purple tunic", "hunched forward",
     "medium shot, low angle looking up at the subject", [{"species": "none", "role": "none"}]),
    ("Ardoz, Cobbler of War", "Legendary Creature — Goblin Shaman",
     "green-skinned humanoid with pointed ears", "ragged smock", "mid-stride",
     "full figure shot centered in frame", [{"species": "none", "role": "none"}]),
    ("Krenko, Mob Boss", "Legendary Creature — Goblin Warrior",
     "green-skinned humanoid", "spiked pauldrons", "standing, one fist raised",
     "medium shot, figure filling the frame", [{"species": "humanoid", "role": "crowd"}]),
    ("Krenko, Tin Street Kingpin", "Legendary Creature — Goblin Warrior",
     "green-skinned humanoid", "patched coat", "leaning on a rail",
     "wide shot, small figure in the lower third", [{"species": "none", "role": "none"}]),
    ("Vial Smasher the Fierce", "Legendary Creature — Goblin Berserker",
     "goblin-like humanoid", "scrap armor", "throwing motion",
     "medium shot, diagonal composition", [{"species": "none", "role": "none"}]),
    # a green-skinned humanoid that is emphatically not a goblin
    ("Captain Vargus Wrath", "Legendary Creature — Orc Pirate",
     "green-skinned humanoid", "long coat", "standing at a ship's wheel",
     "medium shot, figure right of center", [{"species": "none", "role": "none"}]),

    # --- angels: recorded as winged humanoids
    ("Radiant, Serra Archangel", "Legendary Creature — Angel",
     "winged humanoid", "white robes, large white feathered wings", "hovering, arms spread",
     "full figure, low angle looking up at the subject", [{"species": "none", "role": "none"}]),
    ("Razia, Boros Archangel", "Legendary Creature — Angel",
     "winged humanoid female", "gold armor, large feathered wings", "descending, sword raised",
     "full figure centered", [{"species": "none", "role": "none"}]),
    ("Selenia, Dark Angel", "Legendary Creature — Phyrexian Angel",
     "winged humanoid female", "dark armor, large white feathered wings", "standing still",
     "medium shot", [{"species": "none", "role": "none"}]),
    ("Avacyn, Angel of Hope", "Legendary Creature — Angel",
     "winged humanoid", "dark armor or clothing on torso and shoulders, large white feathered wings",
     "standing still, facing away from the viewer", "wide shot, vast empty sky",
     [{"species": "none", "role": "none"}]),

    # --- cats
    ("Kemba, Kha Regent", "Legendary Creature — Cat Cleric",
     "lion-headed humanoid", "bronze plate", "seated on a throne",
     "medium shot, symmetrical", [{"species": "none", "role": "none"}]),
    ("Jazal Goldmane", "Legendary Creature — Cat Warrior",
     "lion-headed humanoid", "gold armor", "charging forward",
     "full figure, low angle looking up at the subject", [{"species": "none", "role": "none"}]),
    ("Mirri, Weatherlight Duelist", "Legendary Creature — Cat Warrior",
     "feline humanoid", "leather jerkin", "lunging with a blade",
     "medium shot, diagonal", [{"species": "none", "role": "none"}]),
    ("Runadi, Behemoth Caller", "Legendary Creature — Cat Shaman",
     "feline humanoid", "beaded shawl", "standing, arms lowered",
     "wide shot", [{"species": "none", "role": "none"}]),
    ("Arahbo, Roar of the World", "Legendary Creature — Cat Avatar",
     "feline humanoid", "gold circlet", "walking away from the viewer",
     "wide shot, negative space above", [{"species": "none", "role": "none"}]),

    # --- treefolk
    ("Ferrafor, Young Yew", "Legendary Creature — Treefolk Druid",
     "treant-like humanoid", "none", "standing rooted",
     "full figure centered", [{"species": "none", "role": "none"}]),
    ("Doran, the Siege Tower", "Legendary Creature — Treefolk Shaman",
     "treant-like humanoid", "none", "leaning forward",
     "medium shot", [{"species": "none", "role": "none"}]),
    ("Kurbis, Harvest Celebrant", "Legendary Creature — Treefolk",
     "plant-like humanoid", "none", "arms outstretched",
     "wide shot", [{"species": "none", "role": "none"}]),
    ("Nemata, Primeval Warden", "Legendary Creature — Treefolk",
     "plant-like humanoid", "none", "standing among saplings",
     "wide shot", [{"species": "none", "role": "none"}]),

    # --- dwarves: the vision pass recorded nothing that distinguishes them
    ("Torbran, Thane of Red Fell", "Legendary Creature — Dwarf Noble",
     "humanoid", "red cloak", "seated", "medium shot", [{"species": "none", "role": "none"}]),
    ("Dáin Ironfoot", "Legendary Creature — Dwarf Warrior",
     "humanoid", "mail hauberk", "standing with an axe", "medium shot",
     [{"species": "none", "role": "none"}]),
    ("Depala, Pilot Exemplar", "Legendary Creature — Dwarf Pilot",
     "humanoid male", "flight jacket", "standing", "medium shot",
     [{"species": "none", "role": "none"}]),

    # --- humans, and a dog, and plain humanoids for the mining baseline
    ("Gwendlyn Di Corci", "Legendary Creature — Human Rogue",
     "human", "silk gown", "seated at a table", "medium shot",
     [{"species": "none", "role": "none"}]),
    ("Baral, Chief of Compliance", "Legendary Creature — Human Wizard",
     "human male", "blue uniform", "standing", "medium shot",
     [{"species": "none", "role": "none"}]),
    ("Mary Read and Anne Bonny", "Legendary Creature — Human Assassin Pirate",
     "human female", "coat and sash", "standing back to back",
     "medium shot", [{"species": "human", "role": "companion"}]),
    ("Lilah, Undefeated Slickshot", "Legendary Creature — Human Rogue",
     "humanoid", "duster coat", "drawing a pistol", "medium shot",
     [{"species": "none", "role": "none"}]),
    ("Smellerbee, Rebel Fighter", "Legendary Creature — Human Rebel Ally",
     "humanoid", "leather vest", "crouching", "medium shot",
     [{"species": "none", "role": "none"}]),
    ("Kaalia of the Vast", "Legendary Creature — Human Cleric",
     "winged humanoid female", "red armor, large feathered wings", "diving",
     "full figure, low angle looking up at the subject", [{"species": "none", "role": "none"}]),
    ("Rin and Seri, Inseparable", "Legendary Creature — Dog Cat",
     "humanoid", "none", "kneeling between two animals", "medium shot",
     [{"species": "dog", "role": "companion"}, {"species": "cat", "role": "companion"}]),
    ("Isoda, Kendo Master", "Legendary Creature — Dog Samurai",
     "anthropomorphic dog", "hakama", "standing still",
     "medium shot", [{"species": "none", "role": "none"}]),
)


# The mining thresholds are ratios, so the fixture needs a realistic baseline of
# ordinary artwork underneath the interesting cases — otherwise "goblin" looks like 20%
# of all Magic rather than 2%, and nothing clears the lift bar. These are the four
# commonest species strings in the real corpus, over the commonest type lines.
_FILLER_SPECIES = ("humanoid", "human", "humanoid male", "elf-like humanoid")
_FILLER_TYPES = (
    "Legendary Creature — Human Wizard",
    "Legendary Creature — Human Soldier",
    "Legendary Creature — Elf Druid",
    "Legendary Creature — Vampire Noble",
    "Legendary Creature — Spirit",
    "Legendary Creature — Merfolk Wizard",
    "Legendary Creature — Zombie Warrior",
)

FILLER = tuple(
    (
        f"Filler {i}",
        _FILLER_TYPES[i % len(_FILLER_TYPES)],
        _FILLER_SPECIES[i % len(_FILLER_SPECIES)],
        "assorted clothing",
        "standing",
        "medium shot",
        [{"species": "none", "role": "none"}],
    )
    for i in range(72)
)


def _slots(species: str, clothing: str, pose: str, composition: str, others: list[dict]) -> str:
    return json.dumps(
        {
            "primary_subject": {
                "species": species,
                "facial_hair": "none",
                "clothing": clothing,
                "pose": pose,
                "held_objects": [{"object": "none", "is_weapon": False}],
            },
            "other_figures": others,
            "figure_count": 1 + sum(1 for o in others if o.get("species") != "none"),
            "setting": "indeterminate",
            "time_of_day": "indeterminate",
            "palette": ["deep crimson", "muted gold"],
            "art_style": "digital painting, painterly brushwork",
            "composition": composition,
        }
    )


@pytest.fixture
def corpus_conn() -> sqlite3.Connection:
    """An in-memory database holding the fixture corpus, wired like the real one."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    for i, (name, type_line, species, clothing, pose, composition, others) in enumerate(
        CORPUS + FILLER
    ):
        oracle_id, ill = f"o{i:03d}", f"i{i:03d}"
        conn.execute(
            "INSERT INTO cards(oracle_id, name, type_line, color_identity) VALUES (?, ?, ?, '')",
            (oracle_id, name, type_line),
        )
        conn.execute(
            "INSERT INTO arts(illustration_id, oracle_id, face_index) VALUES (?, ?, 0)",
            (ill, oracle_id),
        )
        conn.execute(
            "INSERT INTO descriptions(illustration_id, literal, interpretive, slots) "
            "VALUES (?, '', '', ?)",
            (ill, _slots(species, clothing, pose, composition, others)),
        )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def names_for(corpus_conn):
    """Map a set of illustration ids back to card names, for readable assertions."""

    def lookup(ids) -> set[str]:
        rows = corpus_conn.execute(
            "SELECT a.illustration_id AS ill, c.name AS name FROM arts a "
            "JOIN cards c ON c.oracle_id = a.oracle_id"
        )
        by_ill = {r["ill"]: r["name"] for r in rows}
        return {by_ill[i] for i in ids}

    return lookup


@pytest.fixture(scope="session")
def real_conn():
    """The real corpus, read-only. Skipped when the prebuilt database is absent."""
    if not REAL_DB.is_file():
        pytest.skip("data/commanders.db not present")
    conn = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
