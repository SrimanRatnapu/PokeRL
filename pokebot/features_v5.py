"""Observation features, version 5 (superset of v4): v4 + the lookahead's verdict on every action.

`embed_battle_v5(battle)` = `features_v4.embed_battle_v4(battle)` (the 752 v4 entries,
byte-for-byte, so v4 models can be widened with `pokebot.surgery` and start out playing
identically) followed by a 52-entry search block, indexed like the 26-action space
(0-5 switch to team slot, 6-9 move, 10-21 always 0, 22-25 move + tera):

  score    26  the one-turn lookahead's value of the action (`LookaheadScorer.action_scores`,
               HP units of a full-health mon) x 0.5, clipped to [-1, 1]; 0 if illegal
  regret   26  (score - best legal score) x 0.5, clipped: 0 for the lookahead's pick,
               negative for worse actions, -1 if illegal

So the policy sees the search result directly and can learn when to trust it, rather than
having to rediscover one-turn tactics from a single win/loss reward per game.

float32, clipped to [-1, 1], never raises.
"""

from __future__ import annotations

import numpy as np
from poke_env.environment import SinglesEnv

from bots.lookahead_bot import LookaheadScorer
from pokebot.config import ACTION_SPACE_SIZE
from pokebot.features_v4 import OBS_SIZE_V4, embed_battle_v4

SEARCH_BLOCK_SIZE = 2 * ACTION_SPACE_SIZE  # 52
OBS_SIZE_V5: int = OBS_SIZE_V4 + SEARCH_BLOCK_SIZE  # 804
BLOCK_SLICES_V5 = {
    "v4": slice(0, OBS_SIZE_V4),
    "score": slice(OBS_SIZE_V4, OBS_SIZE_V4 + ACTION_SPACE_SIZE),
    "regret": slice(OBS_SIZE_V4 + ACTION_SPACE_SIZE, OBS_SIZE_V5),
}
SCALE = 0.5
_SCORER = LookaheadScorer()


def search_block(battle) -> np.ndarray:
    """The 52 search entries (see module docstring). Never raises."""
    out = np.zeros(SEARCH_BLOCK_SIZE, dtype=np.float32)
    out[ACTION_SPACE_SIZE:] = -1.0
    try:
        mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=bool)
        s = _SCORER.action_scores(battle)
        ok = mask & ~np.isnan(s)
        if not ok.any():
            return out
        best = s[ok].max()
        out[:ACTION_SPACE_SIZE][ok] = np.clip(s[ok] * SCALE, -1.0, 1.0)
        out[ACTION_SPACE_SIZE:][ok] = np.clip((s[ok] - best) * SCALE, -1.0, 0.0)
    except Exception:
        pass
    return out


def embed_battle_v5(battle) -> np.ndarray:
    """Return a float32 vector of shape (OBS_SIZE_V5,) in [-1, 1]. Never raises."""
    obs = np.zeros(OBS_SIZE_V5, dtype=np.float32)
    if battle is None:
        return obs
    try:
        obs[:OBS_SIZE_V4] = embed_battle_v4(battle)
    except Exception:
        pass
    obs[OBS_SIZE_V4:] = search_block(battle)
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(obs, -1.0, 1.0, out=obs)
    return obs


__all__ = ["OBS_SIZE_V5", "SEARCH_BLOCK_SIZE", "BLOCK_SLICES_V5", "embed_battle_v5", "search_block"]
