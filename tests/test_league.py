"""Tests for the league pieces: PFSP sampling, the fixed pool and --opponent-mix parsing.

    .venv/bin/python -m pytest tests/test_league.py -q
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from pokebot import callbacks as cb  # noqa: E402
from pokebot.train import parse_args  # noqa: E402


class _Dummy(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,))
    action_space = gym.spaces.Discrete(1)


def _wrapper(pool, pfsp=True):
    return cb.OpponentMixWrapper(_Dummy(), mix={cb.POOL: 1.0}, pool=pool, seed=0, pfsp=pfsp)


def test_pfsp_weights_favour_opponents_we_lose_to():
    cb.PFSP_STATS.clear()
    for _ in range(40):
        cb.pfsp_record("policy:easy.zip", True)
        cb.pfsp_record("policy:hard.zip", False)
    assert cb.pfsp_weight("policy:hard.zip") > 0.5 > cb.pfsp_weight("policy:easy.zip")
    assert cb.pfsp_weight("policy:easy.zip") >= cb.PFSP_FLOOR
    assert abs(cb.pfsp_win_rate("policy:new.zip")[0] - 0.5) < 1e-9  # unseen = 0.5


def test_pfsp_sampling_and_recent_share():
    cb.PFSP_STATS.clear()
    pool = ["easy.zip", "hard.zip", "s1.zip", "s2.zip", "s3.zip", "s4.zip"]
    for _ in range(40):
        cb.pfsp_record("policy:easy.zip", True)
        cb.pfsp_record("policy:hard.zip", False)
    w = _wrapper(pool)
    c = Counter(w._sample_key() for _ in range(4000))
    assert c["policy:hard.zip"] > 5 * c["policy:easy.zip"]
    newest = sum(c[f"policy:s{i}.zip"] for i in (2, 3, 4))
    assert newest > 0.25 * 4000 * 0.9  # the recent share goes to the newest 3 snapshots


def test_uniform_sampling_unchanged_without_pfsp():
    cb.PFSP_STATS.clear()
    for _ in range(40):
        cb.pfsp_record("policy:a.zip", True)
    w = _wrapper(["a.zip", "b.zip"], pfsp=False)
    c = Counter(w._sample_key() for _ in range(2000))
    assert 0.4 < c["policy:a.zip"] / 2000 < 0.6


def test_fixed_pool_first_and_never_evicted():
    st = cb.TrainState(pool=["s1.zip", "s2.zip"], fixed_pool=["champ.zip"])
    assert st.full_pool == ["champ.zip", "s1.zip", "s2.zip"]


def test_stage_counts_explicit_policy_target():
    st = cb.Stage("league", {"policy:t.zip": 0.85, "simple_heuristic": 0.15}, None, 1.01)
    assert st.counts("policy:t.zip") and st.counts("simple_heuristic") and not st.counts("policy:other.zip")


def test_parse_league_args():
    a = parse_args(["--opponent-mix", "pool=0.55,lookahead=0.2,simple_heuristic=0.25", "--pfsp",
                    "--pool-extra", "checkpoints/champion.zip"])
    assert a.opponent_mix == {"pool": 0.55, "lookahead": 0.2, "simple_heuristic": 0.25}
    assert a.pfsp and a.pool_extra == ["checkpoints/champion.zip"]
