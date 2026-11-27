"""Observation features, version 4 (superset of v3).

`embed_battle_v4(battle)` = `features_v3.embed_battle_v3(battle)` (the 635 v3 entries,
byte-for-byte, so v3 models can be widened with `pokebot.surgery` and start out playing
identically) followed by:

  abilities 117  what each side's ability does (effect vectors; the opponent's as a posterior
                 over its random-battle abilities), ability-aware damage / immunity / KO per move
                 slot, the opponent's ability-aware threat, end-of-turn residual damage (sand,
                 status, seeds, Leftovers, ...), ability-aware speed, Regenerator, and per team
                 slot the effect of switching that mon in: weather it sets and whether that
                 weather finishes off their active, Intimidate value, immunities to their moves,
                 ability-aware threat and speed (`pokebot/v4_abilities.py`)

float32, clipped to [-1, 1], never raises.
"""

from __future__ import annotations

import numpy as np

from pokebot.features_v3 import OBS_SIZE_V3, embed_battle_v3
from pokebot.v4_abilities import ABILITIES_BLOCK_SIZE, abilities_block

OBS_SIZE_V4: int = OBS_SIZE_V3 + ABILITIES_BLOCK_SIZE  # 752
BLOCK_SLICES_V4 = {"v3": slice(0, OBS_SIZE_V3), "abilities": slice(OBS_SIZE_V3, OBS_SIZE_V4)}


def embed_battle_v4(battle) -> np.ndarray:
    """Return a float32 vector of shape (OBS_SIZE_V4,) in [-1, 1]. Never raises."""
    obs = np.zeros(OBS_SIZE_V4, dtype=np.float32)
    if battle is None:
        return obs
    try:
        obs[:OBS_SIZE_V3] = embed_battle_v3(battle)
    except Exception:
        pass
    try:
        vals = abilities_block(battle)
        if len(vals) == ABILITIES_BLOCK_SIZE:
            obs[OBS_SIZE_V3:] = vals
    except Exception:
        pass
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(obs, -1.0, 1.0, out=obs)
    return obs


__all__ = ["OBS_SIZE_V4", "embed_battle_v4", "BLOCK_SLICES_V4"]
