"""Tests for the v4 ability block and the v4 feature set (v3 + abilities).

The env test needs the local Showdown server on localhost:8000.
    .venv/bin/python -m pytest tests/test_v4_abilities.py -q -s
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
from poke_env.battle import Pokemon  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402

from pokebot import v4_abilities as v4  # noqa: E402
from pokebot.env import make_env, make_opponent  # noqa: E402
from pokebot.feature_sets import FEATURE_SETS, version_for_obs_size  # noqa: E402
from pokebot.features_v2 import Combatant, move_info  # noqa: E402
from pokebot.features_v3 import OBS_SIZE_V3, embed_battle_v3  # noqa: E402
from pokebot.features_v4 import BLOCK_SLICES_V4, OBS_SIZE_V4, embed_battle_v4  # noqa: E402

from tests.test_features_v2 import _play_episode  # noqa: E402
from tests.test_v3_items import _battle, _check, _empty_battle, _mv, _sw  # noqa: E402

torch.set_num_threads(1)
V3_MODEL = ROOT / "checkpoints" / "v3ft" / "final.zip"
T = v4._A_TEAM
F = v4.TEAM_FEATS


def _team_battle(my_mons, opp, opp_hp="100/100", fainted=(), extra=()):
    """We are p1; `my_mons` = [(species, ability)], the last one is active. Opponent active `opp`."""
    msgs = [_sw("p1", s) for s, _ in my_mons]
    msgs += [["", "faint", f"p1a: {s}"] for s in fainted]
    msgs += [_sw("p2", opp), ["", "-damage", f"p2a: {opp}", opp_hp]] + list(extra)
    b = _battle(msgs)
    abil = {to_id_str(s): a for s, a in my_mons}
    for m in b.team.values():
        m.ability = abil[m.species]
    return b


def _slot(out, i):
    return out[T + F * i: T + F * (i + 1)]


def test_sizes_and_never_raises():
    assert v4.ABILITIES_BLOCK_SIZE == 117 and v4.N_EFFECTS == 26
    _check(v4.abilities_block(_empty_battle()), v4.ABILITIES_BLOCK_SIZE)
    for junk in (None, object(), "x", 3):
        _check(v4.abilities_block(junk), v4.ABILITIES_BLOCK_SIZE)  # type: ignore[arg-type]


def test_the_lost_game_sand_stream_finishes_gyarados():
    """Eiscue fainted, Gyarados on 2%, no weather: Tyranitar's Sand Stream KOs it at end of turn."""
    b = _team_battle([("Tyranitar", "sandstream"), ("Espeon", "magicbounce"), ("Eiscue", "iceface")],
                     "Gyarados", "2/100", fainted=["Eiscue"])
    out = v4.abilities_block(b)
    _check(out, v4.ABILITIES_BLOCK_SIZE)
    ttar, espeon, eiscue = _slot(out, 0), _slot(out, 1), _slot(out, 2)
    assert ttar[0] == 1.0 and ttar[1] == 1.0 and ttar[6] == 1.0   # chip, KO, changes weather
    assert espeon[0] == 0.0 and espeon[1] == 0.0 and espeon[6] == 0.0
    assert eiscue == [0.0] * F                                     # fainted


def test_cloud_nine_cancels_the_sand_kill():
    b = _team_battle([("Tyranitar", "sandstream"), ("Espeon", "magicbounce"), ("Eiscue", "iceface")],
                     "Golduck", "2/100", fainted=["Eiscue"],
                     extra=[["", "-ability", "p2a: Golduck", "Cloud Nine"]])
    out = v4.abilities_block(b)
    assert _slot(out, 0)[1] == 0.0


def test_sand_already_up_counts_as_residual():
    b = _team_battle([("Espeon", "magicbounce")], "Gyarados", "4/100",
                     extra=[["", "-weather", "Sandstorm"]])
    out = v4.abilities_block(b)
    assert out[v4._A_RESID + 2] < 0 and out[v4._A_RESID + 3] == 1.0   # opponent dies to sand
    assert out[v4._A_RESID + 1] == 0.0                                 # Espeon at full HP does not


def test_posterior_from_random_battle_sets():
    golduck = Pokemon(gen=9, species="golduck")
    post = dict(v4.ability_posterior(golduck))
    assert set(post) == {"cloudnine", "swiftswim"} and abs(sum(post.values()) - 1) < 1e-9
    assert v4.ability_posterior(Pokemon(gen=9, species="tyranitar")) == (("sandstream", 1.0),)


def test_ability_damage_rules():
    eq, aj, surf = move_info("earthquake"), move_info("aquajet"), move_info("surf")
    att = Combatant(Pokemon(gen=9, species="garchomp"), ours=False)
    dfn = Combatant(Pokemon(gen=9, species="snorlax"), ours=False)
    assert v4.nullifies("levitate", eq) and not v4.nullifies("levitate", eq, "moldbreaker")
    assert v4.nullifies("waterabsorb", surf) and not v4.nullifies("waterabsorb", eq)
    d, _ = v4.ability_damage(0.3, aj, att, dfn, "hugepower", None, frozenset(), frozenset())
    assert abs(d - 0.6) < 1e-9
    d, _ = v4.ability_damage(0.8, eq, att, dfn, None, "multiscale", frozenset(), frozenset())
    assert abs(d - 0.4) < 1e-9                                          # full HP
    d, ko = v4.ability_damage(1.7, eq, att, dfn, None, "sturdy", frozenset(), frozenset())
    assert d < 1.0 and not ko
    d, ko = v4.ability_damage(1.7, eq, att, dfn, None, None, frozenset(), frozenset())
    assert ko


def test_intimidate_value_and_immunity_on_switch():
    # Garchomp's likely moves are mostly physical: Intimidate on our Gyarados is worth something.
    b = _team_battle([("Gyarados", "intimidate"), ("Rotom-Wash", "levitate"), ("Blissey", "naturalcure")],
                     "Garchomp")
    out = v4.abilities_block(b)
    # team order: Gyarados, Rotom-Wash, Blissey (active)
    assert _slot(out, 0)[2] > 0.2
    assert _slot(out, 1)[3] > 0.0                       # Levitate dodges some of Garchomp's moves
    # Kingambit (Defiant) punishes Intimidate.
    b = _team_battle([("Gyarados", "intimidate"), ("Blissey", "naturalcure")], "Kingambit",
                     extra=[["", "-ability", "p2a: Kingambit", "Defiant"]])
    assert _slot(v4.abilities_block(b), 0)[2] < 0


def test_registry_and_layout():
    assert "v4" in FEATURE_SETS and OBS_SIZE_V4 == 752 and version_for_obs_size(752) == "v4"
    assert BLOCK_SLICES_V4["v3"] == slice(0, OBS_SIZE_V3)
    b = _empty_battle()
    obs = embed_battle_v4(b)
    assert obs.shape == (OBS_SIZE_V4,) and np.array_equal(obs[:OBS_SIZE_V3], embed_battle_v3(b))


def test_real_battles_prefix_timing_and_warm_start_parity():
    from sb3_contrib import MaskablePPO

    from pokebot.surgery import expand_model

    src = widened = None
    if V3_MODEL.exists():
        src = MaskablePPO.load(str(V3_MODEL), device="cpu", custom_objects={"seed": None})
        widened, _, _ = expand_model(src, OBS_SIZE_V4, verbose=False)
    env = make_env(opponent=make_opponent("heuristic"), feature_version="v4")
    times, nonzero, same, total = [], 0, 0, 0

    def check(e, obs):
        nonlocal nonzero, same, total
        t0 = time.perf_counter()
        blk = v4.abilities_block(e.battle)
        times.append(time.perf_counter() - t0)
        _check(blk, v4.ABILITIES_BLOCK_SIZE)
        assert obs.shape == (OBS_SIZE_V4,)
        assert np.array_equal(obs[:OBS_SIZE_V3], embed_battle_v3(e.battle))
        assert np.allclose(obs[OBS_SIZE_V3:], np.clip(np.array(blk, dtype=np.float32), -1, 1))
        nonzero += int(np.any(obs[OBS_SIZE_V3:] != 0))
        if src is not None:
            mask = e.action_masks()
            a_old, _ = src.predict(obs[:OBS_SIZE_V3], action_masks=mask, deterministic=True)
            a_new, _ = widened.predict(obs, action_masks=mask, deterministic=True)
            same += int(a_old == a_new)
            total += 1

    try:
        rng = np.random.default_rng(1)
        for _ in range(3):
            _play_episode(env, rng, check)
    finally:
        env.close()
    t = np.array(times[1:]) * 1e3
    print(f"\nabilities_block: {len(t)} states, mean {t.mean():.3f} ms, p95 {np.percentile(t, 95):.3f} ms; "
          f"non-zero {nonzero}/{len(times)}; warm-start agreement {same}/{total}")
    assert t.mean() < 5.0
    assert nonzero == len(times)
    assert same == total
