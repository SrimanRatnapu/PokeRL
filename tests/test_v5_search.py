"""Tests for the v5 search features, the policy+lookahead blend and the E10 lookahead changes.

The env test needs the local Showdown server on localhost:8000.
    .venv/bin/python -m pytest tests/test_v5_search.py -q -s
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
from poke_env.battle import PokemonType  # noqa: E402

from bots.lookahead_bot import LookaheadScorer, _with_tera, our_tera_type  # noqa: E402
from pokebot.config import ACTION_SPACE_SIZE  # noqa: E402
from pokebot.env import make_env, make_opponent  # noqa: E402
from pokebot.feature_sets import FEATURE_SETS, version_for_obs_size  # noqa: E402
from pokebot.features_v2 import _Ctx  # noqa: E402
from pokebot.features_v4 import OBS_SIZE_V4, embed_battle_v4  # noqa: E402
from pokebot.features_v5 import BLOCK_SLICES_V5, OBS_SIZE_V5, embed_battle_v5, search_block  # noqa: E402
from pokebot.search import blend  # noqa: E402

from tests.test_features_v2 import _play_episode  # noqa: E402
from tests.test_v3_items import _empty_battle  # noqa: E402
from tests.test_v4_abilities import _team_battle  # noqa: E402

torch.set_num_threads(1)
V5_INIT = ROOT / "checkpoints" / "v5_init" / "init.zip"
V4_INIT = ROOT / "checkpoints" / "v4_init" / "init.zip"


def test_registry_and_layout():
    assert OBS_SIZE_V5 == OBS_SIZE_V4 + 52 == 804 and version_for_obs_size(804) == "v5" and "v5" in FEATURE_SETS
    b = _empty_battle()
    obs = embed_battle_v5(b)
    assert obs.shape == (OBS_SIZE_V5,) and np.array_equal(obs[:OBS_SIZE_V4], embed_battle_v4(b))
    for junk in (None, object(), "x"):
        blk = search_block(junk)
        assert blk.shape == (52,) and np.all(blk[:26] == 0) and np.all(blk[26:] == -1)


def test_tera_type_read_from_request():
    b = _team_battle([("Garchomp", "roughskin")], "Gyarados")
    me = b.active_pokemon
    assert me.tera_type is None and our_tera_type(me) is None
    me._last_request = {"teraType": "Steel"}
    assert our_tera_type(me) == PokemonType.STEEL
    tc = _with_tera(_Ctx(b).me_c, PokemonType.STEEL)
    assert tc.types == [PokemonType.STEEL] and tc.tera == PokemonType.STEEL


def test_blend_rules():
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[[0, 6, 7]] = True
    logp = np.full(ACTION_SPACE_SIZE, -np.inf)
    logp[[0, 6, 7]] = np.log([0.1, 0.6, 0.3])
    scores = np.full(ACTION_SPACE_SIZE, np.nan)
    scores[[0, 6, 7]] = [0.9, 0.0, 0.2]
    assert blend(logp, scores, mask, 0.0) == 6           # plain policy
    assert blend(logp, scores, mask, 100.0) == 0         # pure lookahead
    scores[25] = 50.0                                    # illegal actions never win
    assert blend(logp, scores, mask, 100.0) == 0


def test_real_battles_scores_timing_and_warm_start_parity():
    from sb3_contrib import MaskablePPO

    v5 = v4 = None
    if V5_INIT.exists() and V4_INIT.exists():
        v5 = MaskablePPO.load(str(V5_INIT), device="cpu", custom_objects={"seed": None})
        v4 = MaskablePPO.load(str(V4_INIT), device="cpu", custom_objects={"seed": None})
    env = make_env(opponent=make_opponent("heuristic"), feature_version="v5")
    scorer = LookaheadScorer()
    times, scored, same, total = [], 0, 0, 0

    def check(e, obs):
        nonlocal scored, same, total
        t0 = time.perf_counter()
        s = scorer.action_scores(e.battle)
        times.append(time.perf_counter() - t0)
        mask = e.action_masks()
        assert obs.shape == (OBS_SIZE_V5,)
        if not (e.battle.available_moves or e.battle.available_switches):
            return  # waiting turn (e.g. their mon fainted): no real decision, nothing to score
        assert np.array_equal(obs[:OBS_SIZE_V4], embed_battle_v4(e.battle))
        legal_scored = mask & ~np.isnan(s)
        if not legal_scored.any():
            return  # the env's "no legal action reported" fallback mask (slot 0 only): a stale request
        scored += int(legal_scored.sum() == mask.sum())
        regret = obs[BLOCK_SLICES_V5["regret"]]
        assert np.all(regret[~mask] == -1.0) and np.isclose(regret[mask].max(), 0.0)
        if v5 is not None:
            a_old, _ = v4.predict(obs[:OBS_SIZE_V4], action_masks=mask, deterministic=True)
            a_new, _ = v5.predict(obs, action_masks=mask, deterministic=True)
            same += int(a_old == a_new)
            total += 1

    try:
        rng = np.random.default_rng(2)
        for _ in range(3):
            _play_episode(env, rng, check)
    finally:
        env.close()
    t = np.array(times[1:]) * 1e3
    print(f"\naction_scores: {len(t)} states, mean {t.mean():.2f} ms, p95 {np.percentile(t, 95):.2f} ms; "
          f"every legal action scored in {scored}/{len(times)}; warm-start agreement {same}/{total}")
    assert t.mean() < 10.0
    assert scored >= 0.9 * total if total else scored > 0
    assert same == total
