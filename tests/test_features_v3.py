"""Tests for the v3 feature set (v2 + items + crit + history) and the v2 -> v3 warm start.

Requires the local Showdown server on localhost:8000 for the env tests.
    .venv/bin/python -m pytest tests/test_features_v3.py -q -s
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from pokebot.env import make_env, make_opponent  # noqa: E402
from pokebot.feature_sets import FEATURE_SETS, get_feature_set, version_for_obs_size  # noqa: E402
from pokebot.features_v2 import OBS_SIZE_V2, embed_battle_v2  # noqa: E402
from pokebot.features_v3 import BLOCK_SLICES_V3, OBS_SIZE_V3, embed_battle_v3  # noqa: E402

from tests.test_features_v2 import _empty_battle, _play_episode  # noqa: E402

torch.set_num_threads(1)
CHAMPION = ROOT / "checkpoints" / "champion.zip"


def _check(obs: np.ndarray) -> None:
    assert obs.shape == (OBS_SIZE_V3,) and obs.dtype == np.float32
    assert np.isfinite(obs).all() and obs.min() >= -1.0 and obs.max() <= 1.0


def test_registry_and_layout():
    assert "v3" in FEATURE_SETS
    assert get_feature_set("v3") == (OBS_SIZE_V3, embed_battle_v3)
    assert version_for_obs_size(OBS_SIZE_V3) == "v3"
    assert OBS_SIZE_V3 == 635
    covered = sorted((sl.start, sl.stop) for sl in BLOCK_SLICES_V3.values())
    assert covered[0] == (0, OBS_SIZE_V2) and covered[-1][1] == OBS_SIZE_V3
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))


def test_empty_and_garbage():
    b = _empty_battle()
    obs = embed_battle_v3(b)
    _check(obs)
    assert np.array_equal(obs[:OBS_SIZE_V2], embed_battle_v2(b))
    for junk in (None, object(), "not a battle", 42):
        _check(embed_battle_v3(junk))  # type: ignore[arg-type]


def test_real_battles_v2_prefix_timing_and_warm_start_parity():
    """Real battle states: v3 = v2 prefix + new blocks, and a surgically widened champion
    picks exactly the same actions as the champion on every state."""
    from sb3_contrib import MaskablePPO

    from pokebot.surgery import expand_model

    champ = widened = None
    if CHAMPION.exists():
        champ = MaskablePPO.load(str(CHAMPION), device="cpu", custom_objects={"seed": None})
        widened, _, _ = expand_model(champ, OBS_SIZE_V3, verbose=False)

    env = make_env(opponent=make_opponent("heuristic"), feature_version="v3")
    times, nonzero, same, total = [], {k: 0 for k in BLOCK_SLICES_V3}, 0, 0

    def check(e, obs):
        nonlocal same, total
        t0 = time.perf_counter()
        again = embed_battle_v3(e.battle)
        times.append(time.perf_counter() - t0)
        _check(again)
        assert np.array_equal(again, obs)
        assert np.array_equal(again[:OBS_SIZE_V2], embed_battle_v2(e.battle))
        for k, sl in BLOCK_SLICES_V3.items():
            nonzero[k] += int(np.any(again[sl] != 0))
        if champ is not None:
            mask = e.action_masks()
            a_old, _ = champ.predict(obs[:OBS_SIZE_V2], action_masks=mask, deterministic=True)
            a_new, _ = widened.predict(obs, action_masks=mask, deterministic=True)
            same += int(a_old == a_new)
            total += 1

    try:
        assert env.observation_space.shape == (OBS_SIZE_V3,)
        rng = np.random.default_rng(0)
        for _ in range(3):
            _play_episode(env, rng, check)
    finally:
        env.close()
    t = np.array(times) * 1e3
    print(f"\nembed_battle_v3: {len(t)} states, mean {t.mean():.3f} ms, p95 {np.percentile(t, 95):.3f} ms; "
          f"non-zero blocks {nonzero}; warm-start agreement {same}/{total}")
    assert t.mean() < 5.0
    assert all(n > 0 for n in nonzero.values()), nonzero
    assert same == total
