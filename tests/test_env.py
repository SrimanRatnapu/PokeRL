"""Tests for the environment layer (pokebot/features.py, pokebot/env.py).

Requires the local Showdown server on localhost:8000.
Run from the project root:
    .venv/bin/python -m pytest tests/test_env.py -q

Kept deliberately small (a handful of battles, <= 2 envs, 512 PPO steps) so it
finishes in well under a few minutes on a laptop.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from gymnasium.spaces import Box, Discrete  # noqa: E402
from poke_env.battle import Battle  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from pokebot.config import ACTION_SPACE_SIZE, RewardConfig  # noqa: E402
from pokebot.env import MaskedPokeEnv, PolicyPlayer, make_env, make_opponent  # noqa: E402
from pokebot.feature_sets import DEFAULT_FEATURE_VERSION, get_feature_set  # noqa: E402

# make_env() builds the default feature version; v1-specific checks live in tests/test_features_v2.py.
OBS_SIZE, embed_battle = get_feature_set(DEFAULT_FEATURE_VERSION)

torch.set_num_threads(1)  # be gentle with the laptop

MAX_STEPS = 1500  # safety cap per episode


def _check_obs(env: MaskedPokeEnv, obs: np.ndarray) -> None:
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert env.observation_space.contains(obs), (obs.min(), obs.max())
    mask = env.action_masks()
    assert mask.shape == (ACTION_SPACE_SIZE,) and mask.dtype == bool
    assert mask.any()


def _play_episode(env: MaskedPokeEnv, rng: np.random.Generator):
    obs, info = env.reset()
    assert isinstance(info, dict)
    _check_obs(env, obs)
    total, steps = 0.0, 0
    while True:
        legal = np.flatnonzero(env.action_masks())
        action = int(rng.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        _check_obs(env, obs)
        assert isinstance(reward, float) and np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        total += reward
        steps += 1
        if terminated or truncated:
            break
        assert steps < MAX_STEPS, "episode did not terminate"
    assert "won" in info and isinstance(info["won"], bool)
    assert env.battle is not None and env.battle.finished
    return total, steps, info


# --------------------------------------------------------------------------- #
# Pure feature tests (no server needed)
# --------------------------------------------------------------------------- #
def test_embed_empty_battle():
    battle = Battle("battle-gen9randombattle-0", "tester", logging.getLogger("test"), gen=9)
    obs = embed_battle(battle)
    assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
    assert np.isfinite(obs).all() and obs.min() >= -1 and obs.max() <= 1


def test_embed_never_raises_on_garbage():
    for junk in (None, object(), "not a battle"):
        obs = embed_battle(junk)  # type: ignore[arg-type]
        assert obs.shape == (OBS_SIZE,) and np.isfinite(obs).all()


# --------------------------------------------------------------------------- #
# Env tests (need the local server)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def env():
    e = make_env()  # default opponent: RandomPlayer
    yield e
    e.close()


def test_spaces_and_basics(env):
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (OBS_SIZE,)
    assert env.observation_space.dtype == np.float32
    assert isinstance(env.action_space, Discrete) and env.action_space.n == ACTION_SPACE_SIZE
    obs, info = env.reset(seed=0)
    _check_obs(env, obs)
    assert isinstance(info, dict)


def test_episodes_vs_random_and_heuristic(env):
    rng = np.random.default_rng(0)
    results = {}
    for kind in ("random", "heuristic"):
        env.set_opponent(make_opponent(kind))
        for _ in range(2):
            total, steps, info = _play_episode(env, rng)
            assert type(env.opponent).__name__ == {"random": "RandomPlayer", "heuristic": "HeuristicBot"}[kind]
            results.setdefault(kind, []).append((round(total, 2), steps, info["won"]))
    print("episode results:", results)


def test_unscaled_victory_reward(env):
    """With shaping_scale=0 the only reward is the terminal +/-victory."""
    env.set_opponent(make_opponent("random"))
    env.set_reward_config(RewardConfig(shaping_scale=0.0, victory=30.0))
    try:
        total, _, info = _play_episode(env, np.random.default_rng(1))
        expected = 30.0 if info["won"] else (-30.0 if info["lost"] else 0.0)
        assert total == pytest.approx(expected)
    finally:
        env.set_reward_config(RewardConfig())


def test_two_envs_coexist_with_shared_opponent():
    """Two envs in one process (DummyVecEnv), sharing one opponent object."""
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv

    shared = make_opponent("heuristic")
    vec = DummyVecEnv([lambda: make_env(opponent=shared), lambda: make_env(opponent=shared)])
    try:
        usernames = {u for e in vec.envs for u in e.poke_env.possible_agents}
        assert len(usernames) == 4
        vec.reset()
        rng = np.random.default_rng(2)
        finished = 0
        for _ in range(400):
            masks = get_action_masks(vec)
            actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            obs, rewards, dones, infos = vec.step(actions)
            assert obs.shape == (2, OBS_SIZE) and np.isfinite(rewards).all()
            for d, inf in zip(dones, infos):
                if d:
                    assert "won" in inf
                    finished += 1
            if finished >= 2:
                break
        assert finished >= 2
    finally:
        vec.close()


def test_maskable_ppo_and_policy_player(tmp_path):
    from sb3_contrib import MaskablePPO

    env = make_env()
    try:
        model = MaskablePPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=2, verbose=0, device="cpu", seed=0)
        model.learn(total_timesteps=512)
        path = tmp_path / "tiny_model.zip"
        model.save(path)

        # Self-play: PolicyPlayer as a sync opponent inside the env.
        env.set_opponent(make_opponent("policy", model=model, deterministic=False))
        _, _, info = _play_episode(env, np.random.default_rng(3))
        assert "won" in info
    finally:
        env.close()

    # Networked: loaded from disk, plays real battles via battle_against.
    player = PolicyPlayer(
        model_path=str(path),
        account_configuration=AccountConfiguration.generate("PolicyTest", rand=True),
    )
    opponent = RandomPlayer(account_configuration=AccountConfiguration.generate("RandTest", rand=True))

    async def run():
        await asyncio.wait_for(player.battle_against(opponent, n_battles=2), timeout=180)

    asyncio.run(run())
    assert player.n_finished_battles == 2
    print(f"PolicyPlayer (512 steps) vs RandomPlayer: {player.n_won_battles}/2 wins")
