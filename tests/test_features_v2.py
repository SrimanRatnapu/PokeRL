"""Tests for feature versioning (v1 frozen, v2 superset) and the v2 damage model.

Requires the local Showdown server on localhost:8000 for the env/PolicyPlayer tests.
Run from the project root:
    .venv/bin/python -m pytest tests/test_features_v2.py -q -s

Kept small (a few battles, one env at a time, 512 PPO steps).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from poke_env.battle import Battle, Pokemon  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from pokebot import features as features_v1  # noqa: E402
from pokebot.feature_sets import (  # noqa: E402
    FEATURE_SETS,
    OBS_SIZE_V1,
    OBS_SIZE_V2,
    get_feature_set,
    version_for_model,
    version_for_obs_size,
)
from pokebot.env import MaskedPokeEnv, PolicyPlayer, make_env, make_opponent  # noqa: E402
from pokebot.features_v2 import (  # noqa: E402
    BLOCK_SLICES_V2,
    Combatant,
    embed_battle_v2,
    estimate_damage,
    estimate_stats,
    move_info,
    opponent_movepool,
)

torch.set_num_threads(1)  # be gentle with the laptop

V1_CHECKPOINT = ROOT / "checkpoints" / "v1" / "final.zip"
MAX_STEPS = 1500


def _empty_battle() -> Battle:
    return Battle("battle-gen9randombattle-0", "tester", logging.getLogger("test"), gen=9)


def _check_v2(obs: np.ndarray) -> None:
    assert obs.shape == (OBS_SIZE_V2,) and obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert obs.min() >= -1.0 and obs.max() <= 1.0


# --------------------------------------------------------------------------- #
# Registry / pure feature tests (no server needed)
# --------------------------------------------------------------------------- #
def test_registry():
    assert OBS_SIZE_V1 == 226 == features_v1.OBS_SIZE
    assert {"v1", "v2"} <= set(FEATURE_SETS)
    assert get_feature_set("v1") == (226, features_v1.embed_battle)
    assert get_feature_set("v2") == (OBS_SIZE_V2, embed_battle_v2)
    assert OBS_SIZE_V2 > OBS_SIZE_V1
    assert version_for_obs_size(226) == "v1" and version_for_obs_size(OBS_SIZE_V2) == "v2"
    assert version_for_obs_size(123) is None
    with pytest.raises(ValueError):
        get_feature_set("v9")
    # Block layout covers the whole vector exactly once.
    covered = sorted((sl.start, sl.stop) for sl in BLOCK_SLICES_V2.values())
    assert covered[0][0] == 0 and covered[-1][1] == OBS_SIZE_V2
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))


def test_v2_empty_and_garbage():
    b = _empty_battle()
    obs = embed_battle_v2(b)
    _check_v2(obs)
    assert np.array_equal(obs[:OBS_SIZE_V1], features_v1.embed_battle(b))
    for junk in (None, object(), "not a battle", 42):
        _check_v2(embed_battle_v2(junk))  # type: ignore[arg-type]


def test_estimate_stats_randbats_spread():
    # base 100 at L100: hp = (200+31+21) + 100 + 10 = 362; others = 252 + 5 = 257
    hp, atk, *_ = estimate_stats((100, 100, 100, 100, 100, 100), 100)
    assert (hp, atk) == (362.0, 257.0)
    assert estimate_stats((1, 90, 45, 30, 30, 40), 90)[0] == 1.0  # Shedinja


def test_damage_model_basics():
    chomp = Pokemon(gen=9, species="garchomp")
    rotom = Pokemon(gen=9, species="rotomwash")  # Levitate is its only ability
    blissey = Pokemon(gen=9, species="blissey")
    att = Combatant(chomp, ours=False)
    eq, dclaw, sd = move_info("earthquake"), move_info("dragonclaw"), move_info("swordsdance")
    assert estimate_damage(eq, att, Combatant(rotom, ours=False)) == 0.0  # Levitate immunity
    d = estimate_damage(eq, att, Combatant(blissey, ours=False))
    assert 0.1 < d < 1.0  # sizeable but not a KO on Blissey
    assert estimate_damage(sd, att, Combatant(blissey, ours=False)) == 0.0  # status move
    assert estimate_damage(dclaw, att, Combatant(blissey, ours=False)) < d  # both STAB, lower BP
    # Seismic Toss = level damage
    toss = move_info("seismictoss")
    assert estimate_damage(toss, Combatant(blissey, ours=False), att) == pytest.approx(
        100 / Combatant(chomp, ours=False).hp_cur
    )
    assert sd.setup and move_info("recover").recovery and move_info("uturn").pivot
    assert move_info("stealthrock").hazard and move_info("willowisp").inflicts
    assert move_info("notarealmove") is None


def test_opponent_movepool_from_sets():
    mon = Pokemon(gen=9, species="garchomp")
    pool, tera, known = opponent_movepool(mon)
    ids = {m.id for m in pool}
    assert known and "earthquake" in ids and len(tera) > 0
    unknown = Pokemon(gen=9, species="pikachu")
    unknown._species = "notarealspecies"
    pool2, tera2, known2 = opponent_movepool(unknown)
    assert pool2 == () and not known2 and not tera2


# --------------------------------------------------------------------------- #
# Env / PolicyPlayer tests (need the local server)
# --------------------------------------------------------------------------- #
def _play_episode(env: MaskedPokeEnv, rng: np.random.Generator, on_state=None):
    obs, _ = env.reset()
    steps = 0
    while True:
        assert obs.shape == env.observation_space.shape and env.observation_space.contains(obs)
        if on_state is not None:
            on_state(env, obs)
        legal = np.flatnonzero(env.action_masks())
        obs, _r, terminated, truncated, info = env.step(int(rng.choice(legal)))
        steps += 1
        if terminated or truncated:
            break
        assert steps < MAX_STEPS
    assert "won" in info
    return steps, info


def test_v2_real_battle_states_and_timing():
    env = make_env(opponent=make_opponent("heuristic"), feature_version="v2")
    times: list[float] = []
    nonzero_blocks: dict[str, int] = {k: 0 for k in BLOCK_SLICES_V2}

    def check(e, obs):
        b = e.battle
        t0 = time.perf_counter()
        again = embed_battle_v2(b)
        times.append(time.perf_counter() - t0)
        _check_v2(again)
        assert np.array_equal(again, obs)  # the env really serves v2
        assert np.array_equal(again[:OBS_SIZE_V1], features_v1.embed_battle(b))  # v1 prefix untouched
        for k, sl in BLOCK_SLICES_V2.items():
            nonzero_blocks[k] += int(np.any(again[sl] != 0))

    try:
        assert env.observation_space.shape == (OBS_SIZE_V2,)
        rng = np.random.default_rng(0)
        for _ in range(3):
            _play_episode(env, rng, check)
    finally:
        env.close()
    t = np.array(times) * 1e3
    print(f"\nembed_battle_v2: {len(t)} states, mean {t.mean():.3f} ms, p95 {np.percentile(t, 95):.3f} ms, "
          f"max {t.max():.3f} ms; states with non-zero block: {nonzero_blocks}")
    assert t.mean() < 5.0  # target is < 2 ms; loose bound so a busy laptop doesn't flake
    for k, n in nonzero_blocks.items():
        assert n > 0, f"block {k} was always zero"


class _Counting:
    """Wraps a model to count predict() calls (i.e. turns not played randomly)."""

    def __init__(self, model):
        self._model = model
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._model, name)

    def predict(self, *a, **kw):
        self.calls += 1
        return self._model.predict(*a, **kw)


def _v1_model(tmp_path_factory):
    from sb3_contrib import MaskablePPO

    if V1_CHECKPOINT.exists():
        return MaskablePPO.load(str(V1_CHECKPOINT), device="cpu", custom_objects={"seed": None}), str(V1_CHECKPOINT)
    env = make_env(feature_version="v1")
    try:
        model = MaskablePPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=1, verbose=0, device="cpu")
        model.learn(total_timesteps=256)
        path = tmp_path_factory.mktemp("v1") / "tiny_v1.zip"
        model.save(path)
        return model, str(path)
    finally:
        env.close()


def _networked_battles(player: PolicyPlayer, n: int = 1) -> None:
    opponent = RandomPlayer(account_configuration=AccountConfiguration.generate("RandV2T", rand=True))

    async def run():
        await asyncio.wait_for(player.battle_against(opponent, n_battles=n), timeout=180)

    asyncio.run(run())
    assert player.n_finished_battles == n


def test_v1_checkpoint_still_plays(tmp_path_factory):
    model, path = _v1_model(tmp_path_factory)
    assert model.observation_space.shape == (226,) and version_for_model(model) == "v1"

    # 1) as a self-play opponent inside a v2 training env
    counted = _Counting(model)
    opp = make_opponent("policy", model=counted, deterministic=False)
    assert opp.feature_version == "v1"
    env = make_env(opponent=opp, feature_version="v2")
    try:
        assert env.observation_space.shape == (OBS_SIZE_V2,)
        steps, _ = _play_episode(env, np.random.default_rng(5))
    finally:
        env.close()
    assert counted.calls >= steps // 2, (counted.calls, steps)

    # 2) networked, loaded from disk (the evaluate.py / play.py path)
    player = PolicyPlayer(model_path=path, account_configuration=AccountConfiguration.generate("PolV1T", rand=True))
    assert player.feature_version == "v1"
    player.model = _Counting(player.model)
    _networked_battles(player, 1)
    assert player.model.calls > 0
    print(f"\nv1 checkpoint {Path(path).name}: {counted.calls} in-env predictions over {steps} steps; "
          f"networked {player.n_won_battles}/1 wins vs random")


def test_v2_maskable_ppo_and_policy_player(tmp_path):
    from sb3_contrib import MaskablePPO

    env = make_env(feature_version="v2")
    try:
        model = MaskablePPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=2, verbose=0, device="cpu")
        model.learn(total_timesteps=512)
        assert model.observation_space.shape == (OBS_SIZE_V2,)
        path = tmp_path / "tiny_v2.zip"
        model.save(path)
        opp = make_opponent("policy", model=model, deterministic=False)
        assert opp.feature_version == "v2"
        env.set_opponent(opp)
        _play_episode(env, np.random.default_rng(6))
    finally:
        env.close()

    player = PolicyPlayer(model_path=str(path), account_configuration=AccountConfiguration.generate("PolV2T", rand=True))
    assert player.feature_version == "v2"
    player.model = _Counting(player.model)
    _networked_battles(player, 1)
    assert player.model.calls > 0


def test_unknown_obs_size_falls_back_to_random():
    from gymnasium.spaces import Box

    class Fake:
        observation_space = Box(-1, 1, (7,), np.float32)

    p = PolicyPlayer(model=Fake(), start_listening=False)
    assert p.feature_version is None and p._embed_fn is None
