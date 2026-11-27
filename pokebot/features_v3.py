"""Observation features, version 3 (superset of v2).

`embed_battle_v3(battle)` = `features_v2.embed_battle_v2(battle)` (the 346 v2 entries,
byte-for-byte, so v2 models can be widened with `pokebot.surgery` and start out playing
identically) followed by:

  items    71  held items: our / opponent's item (one-hot, or a prior over likely items when the
               opponent's is unknown), Choice-lock state for both sides, item-aware damage and KO
               flags per move slot, opponent's item-aware threat, Scarf-aware speed, bench Boots
               (`pokebot/v3_items.py`)
  crit     22  critical hits priced in: crit chance, "KO only on a crit", expected damage per move
               slot; the opponent's crit threat; crit immunity / Focus Energy / crit-stage bonuses
               (`pokebot/v3_items.py`)
  history 196  memory: the last 6 turns in detail (our decision incl. move slot, opponent's
               decision, damage, KOs, crits, who moved first), whole-battle aggregates (switch
               tendencies, damage totals, hazards, tera) and our previous action
               (`pokebot/v3_history.py`)

float32, clipped to [-1, 1], never raises (each block is computed independently and left as zeros if
it fails).
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from pokebot.features_v2 import OBS_SIZE_V2, embed_battle_v2
from pokebot.v3_history import HISTORY_V3_SIZE, history_block
from pokebot.v3_items import CRIT_BLOCK_SIZE, ITEMS_BLOCK_SIZE, crit_block, items_block

V3_BLOCKS: Sequence[Tuple[str, int, Callable[[object], List[float]]]] = (
    ("items", ITEMS_BLOCK_SIZE, items_block),
    ("crit", CRIT_BLOCK_SIZE, crit_block),
    ("history", HISTORY_V3_SIZE, history_block),
)

OBS_SIZE_V3: int = OBS_SIZE_V2 + sum(size for _, size, _ in V3_BLOCKS)

BLOCK_SLICES_V3 = {"v2": slice(0, OBS_SIZE_V2)}
_start = OBS_SIZE_V2
for _name, _size, _ in V3_BLOCKS:
    BLOCK_SLICES_V3[_name] = slice(_start, _start + _size)
    _start += _size
del _start, _name, _size


def embed_battle_v3(battle) -> np.ndarray:
    """Return a float32 vector of shape (OBS_SIZE_V3,) in [-1, 1]. Never raises."""
    obs = np.zeros(OBS_SIZE_V3, dtype=np.float32)
    if battle is None:
        return obs
    try:
        obs[:OBS_SIZE_V2] = embed_battle_v2(battle)
    except Exception:
        pass
    for name, size, fn in V3_BLOCKS:
        sl = BLOCK_SLICES_V3[name]
        try:
            vals = fn(battle)
            if len(vals) == size:
                obs[sl] = vals
        except Exception:
            obs[sl] = 0.0
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(obs, -1.0, 1.0, out=obs)
    return obs


__all__ = ["OBS_SIZE_V3", "embed_battle_v3", "BLOCK_SLICES_V3", "V3_BLOCKS"]
