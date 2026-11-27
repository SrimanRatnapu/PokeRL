"""Registry of observation feature versions.

    v1  pokebot.features.embed_battle        226 dims  (original; v1 checkpoints)
    v2  pokebot.features_v2.embed_battle_v2  v1 + damage/threat/team/history blocks
    v3  pokebot.features_v3.embed_battle_v3  v2 + held items, critical hits, battle history / past decisions
    v4  pokebot.features_v4.embed_battle_v4  v3 + abilities (effects, immunities, weather on switch-in, residual)
    v5  pokebot.features_v5.embed_battle_v5  v4 + the one-turn lookahead's score for every action

A trained model's version is recovered from its observation size
(`version_for_obs_size(model.observation_space.shape[0])`), so checkpoints of any
version keep working with PolicyPlayer / evaluate.py / play.py / self-play pools.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from pokebot.features import OBS_SIZE as OBS_SIZE_V1, embed_battle as embed_battle_v1
from pokebot.features_v2 import OBS_SIZE_V2, embed_battle_v2
from pokebot.features_v3 import OBS_SIZE_V3, embed_battle_v3
from pokebot.features_v4 import OBS_SIZE_V4, embed_battle_v4
from pokebot.features_v5 import OBS_SIZE_V5, embed_battle_v5

EmbedFn = Callable[[object], np.ndarray]

FEATURE_SETS: Dict[str, Tuple[int, EmbedFn]] = {
    "v1": (OBS_SIZE_V1, embed_battle_v1),
    "v2": (OBS_SIZE_V2, embed_battle_v2),
    "v3": (OBS_SIZE_V3, embed_battle_v3),
    "v4": (OBS_SIZE_V4, embed_battle_v4),
    "v5": (OBS_SIZE_V5, embed_battle_v5),
}
DEFAULT_FEATURE_VERSION = "v2"
FEATURE_VERSIONS = tuple(FEATURE_SETS)


def get_feature_set(version: str) -> Tuple[int, EmbedFn]:
    """(obs_size, embed_fn) for a feature version ("v1" ... "v5")."""
    try:
        return FEATURE_SETS[version]
    except KeyError:
        raise ValueError(f"Unknown feature version {version!r}; choose from {FEATURE_VERSIONS}") from None


def version_for_obs_size(n: int) -> Optional[str]:
    """Feature version whose observation size is `n`, or None if no version matches."""
    for version, (size, _) in FEATURE_SETS.items():
        if size == int(n):
            return version
    return None


def version_for_model(model) -> Optional[str]:
    """Feature version a loaded SB3 model was trained with (from its observation space)."""
    shape = tuple(getattr(getattr(model, "observation_space", None), "shape", ()) or ())
    return version_for_obs_size(shape[0]) if len(shape) == 1 else None


__all__ = ["FEATURE_SETS", "FEATURE_VERSIONS", "DEFAULT_FEATURE_VERSION", "OBS_SIZE_V1", "OBS_SIZE_V2", "OBS_SIZE_V3", "OBS_SIZE_V4", "OBS_SIZE_V5",
           "embed_battle_v1", "embed_battle_v2", "embed_battle_v3", "embed_battle_v4", "embed_battle_v5", "get_feature_set", "version_for_obs_size", "version_for_model"]
