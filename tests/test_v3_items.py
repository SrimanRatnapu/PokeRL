"""Tests for the v3 held-item / critical-hit blocks (pokebot/v3_items.py). No server needed.

Run from the project root:
    .venv/bin/python -m pytest tests/test_v3_items.py -q
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_env.battle import Battle, Effect, Pokemon  # noqa: E402

from pokebot import v3_items as v3  # noqa: E402
from pokebot.features_v2 import Combatant, estimate_damage, move_info  # noqa: E402
from pokebot.v3_items import (  # noqa: E402
    CRIT_BLOCK_SIZE,
    ITEM_VOCAB,
    ITEMS_BLOCK_SIZE,
    crit_block,
    crit_chance,
    crit_chance_from_stage,
    crit_stage_bonus,
    estimate_damage_items,
    items_block,
    species_item_prior,
    stint_moves,
)

I = {k: i for i, k in enumerate(ITEM_VOCAB)}


def _empty_battle() -> Battle:
    return Battle("battle-gen9randombattle-0", "tester", logging.getLogger("test"), gen=9)


def _battle(messages, my_item=None, opp_item=None) -> Battle:
    """Battle built from protocol lines (we are p1). Items are set after parsing."""
    b = _empty_battle()
    b.player_role = "p1"
    for m in [["", "player", "p1", "tester", "", ""], ["", "player", "p2", "opp", "", ""]] + messages:
        b.parse_message(m)
    if my_item is not None:
        b.active_pokemon.item = my_item
    if opp_item is not None:
        b.opponent_active_pokemon.item = opp_item
    return b


def _sw(side: str, species: str, level: int = 80):
    return ["", "switch", f"{side}a: {species}", f"{species}, L{level}", "100/100"]


def _mv(side: str, species: str, move: str, target: str = "", *extra):
    return ["", "move", f"{side}a: {species}", move, target, *extra]


def _check(vals, size):
    assert isinstance(vals, list) and len(vals) == size
    assert all(isinstance(x, float) for x in vals)
    assert all(-1.0 <= x <= 1.0 for x in vals)


def _opp(species: str) -> Combatant:
    return Combatant(Pokemon(gen=9, species=species), ours=False)


# --------------------------------------------------------------------------- #
# Shape / robustness
# --------------------------------------------------------------------------- #
def test_sizes_and_layout():
    assert ITEMS_BLOCK_SIZE == 71 and CRIT_BLOCK_SIZE == 22
    assert 60 <= ITEMS_BLOCK_SIZE + CRIT_BLOCK_SIZE <= 110
    assert len(ITEM_VOCAB) == 22 and ITEM_VOCAB[-2:] == ("other", "none")


def test_empty_and_garbage_never_raise():
    b = _empty_battle()
    for blk, size in ((items_block, ITEMS_BLOCK_SIZE), (crit_block, CRIT_BLOCK_SIZE)):
        _check(blk(b), size)
        assert not any(blk(b))
        for junk in (None, object(), "battle", 42):
            _check(blk(junk), size)  # type: ignore[arg-type]


def test_odd_protocol_never_raises():
    b = _battle([_sw("p1", "Garchomp"), _sw("p2", "Blissey")])
    b._replay_data.extend([[], ["x"], ["", "move"], ["", "move", "p2a"], ["", "switch", ""], [""] * 10])
    _check(items_block(b), ITEMS_BLOCK_SIZE)
    _check(crit_block(b), CRIT_BLOCK_SIZE)
    b.player_role = None
    _check(items_block(b), ITEMS_BLOCK_SIZE)


# --------------------------------------------------------------------------- #
# Item-aware damage
# --------------------------------------------------------------------------- #
def test_no_item_matches_v2():
    att, dfn = _opp("garchomp"), _opp("blissey")
    for mid in ("earthquake", "fireblast", "seismictoss", "dragonclaw"):
        mi = move_info(mid)
        assert estimate_damage_items(mi, att, dfn) == pytest.approx(estimate_damage(mi, att, dfn))


def test_choice_band_and_specs():
    att, dfn = _opp("garchomp"), _opp("blissey")
    eq, fb = move_info("earthquake"), move_info("fireblast")
    base_eq, base_fb = estimate_damage_items(eq, att, dfn), estimate_damage_items(fb, att, dfn)
    band_eq = estimate_damage_items(eq, att, dfn, att_item="choiceband")
    assert 1.4 * base_eq < band_eq <= 1.5 * base_eq  # Atk x1.5 (the +2 constant is not scaled)
    assert estimate_damage_items(fb, att, dfn, att_item="choiceband") == pytest.approx(base_fb)
    assert estimate_damage_items(fb, att, dfn, att_item="choicespecs") > 1.4 * base_fb
    assert estimate_damage_items(eq, att, dfn, att_item="choicespecs") == pytest.approx(base_eq)
    # Body Press uses Def, which Choice Band does not boost
    bp = move_info("bodypress")
    assert estimate_damage_items(bp, att, dfn, att_item="choiceband") == pytest.approx(
        estimate_damage_items(bp, att, dfn))


def test_life_orb_expert_belt_type_boost_knock_off():
    att = _opp("garchomp")
    blissey, heatran = _opp("blissey"), _opp("heatran")
    eq, dclaw = move_info("earthquake"), move_info("dragonclaw")
    assert estimate_damage_items(eq, att, blissey, att_item="lifeorb") == pytest.approx(
        estimate_damage_items(eq, att, blissey) * 5324 / 4096)
    # Expert Belt: only on super-effective hits (EQ is 4x on Heatran, neutral on Blissey)
    assert estimate_damage_items(eq, att, blissey, att_item="expertbelt") == pytest.approx(
        estimate_damage_items(eq, att, blissey))
    assert estimate_damage_items(eq, att, heatran, att_item="expertbelt") == pytest.approx(
        estimate_damage_items(eq, att, heatran) * 4915 / 4096)
    assert estimate_damage_items(dclaw, att, blissey, att_item="dragonfang") > estimate_damage_items(
        dclaw, att, blissey)
    assert estimate_damage_items(eq, att, blissey, att_item="dragonfang") == pytest.approx(
        estimate_damage_items(eq, att, blissey))
    ko = move_info("knockoff")
    weavile = _opp("weavile")
    assert estimate_damage_items(ko, weavile, blissey, dfn_item="leftovers") == pytest.approx(
        1.5 * estimate_damage_items(ko, weavile, blissey, dfn_item=None))


def test_defensive_items():
    att = _opp("garchomp")
    eq, fb = move_info("earthquake"), move_info("fireblast")
    blissey = _opp("blissey")
    # Assault Vest lowers special damage only
    assert estimate_damage_items(fb, att, blissey, dfn_item="assaultvest") < estimate_damage_items(fb, att, blissey)
    assert estimate_damage_items(eq, att, blissey, dfn_item="assaultvest") == pytest.approx(
        estimate_damage_items(eq, att, blissey))
    # Eviolite: Chansey (can evolve) yes, Blissey (fully evolved) no
    chansey = _opp("chansey")
    assert estimate_damage_items(eq, att, chansey, dfn_item="eviolite") < 0.75 * estimate_damage_items(eq, att, chansey)
    assert estimate_damage_items(eq, att, blissey, dfn_item="eviolite") == pytest.approx(
        estimate_damage_items(eq, att, blissey))
    # Air Balloon: Ground immunity
    heatran = _opp("heatran")
    assert estimate_damage_items(eq, att, heatran) > 1.0
    assert estimate_damage_items(eq, att, heatran, dfn_item="airballoon") == 0.0
    assert estimate_damage_items(move_info("dragonclaw"), att, heatran, dfn_item="airballoon") > 0.0


def test_crit_damage_ignores_boosts():
    att, dfn = _opp("garchomp"), _opp("blissey")
    eq = move_info("earthquake")
    plain = estimate_damage_items(eq, att, dfn)
    crit = estimate_damage_items(eq, att, dfn, crit=True)
    assert crit == pytest.approx(1.5 * plain)
    # defender +2 Def: normal hit drops, crit ignores it
    dfn.mon._boosts["def"] = 2
    assert estimate_damage_items(eq, att, dfn) < 0.6 * plain
    assert estimate_damage_items(eq, att, dfn, crit=True) == pytest.approx(crit)
    dfn.mon._boosts["def"] = 0
    # attacker -2 Atk: crit ignores it; attacker +2 Atk is kept
    att.mon._boosts["atk"] = -2
    assert estimate_damage_items(eq, att, dfn, crit=True) == pytest.approx(crit)
    att.mon._boosts["atk"] = 2
    assert estimate_damage_items(eq, att, dfn, crit=True) > 1.8 * crit
    att.mon._boosts["atk"] = 0
    # fixed damage does not crit
    toss = move_info("seismictoss")
    assert estimate_damage_items(toss, dfn, att, crit=True) == pytest.approx(estimate_damage_items(toss, dfn, att))


def test_item_speed():
    c = _opp("garchomp")
    assert v3.item_speed(c, "choicescarf") == pytest.approx(1.5 * v3.item_speed(c, None))


# --------------------------------------------------------------------------- #
# Crit math
# --------------------------------------------------------------------------- #
def test_crit_stage_math():
    assert [crit_chance_from_stage(s) for s in range(5)] == pytest.approx([1 / 24, 1 / 8, 1 / 2, 1, 1])
    assert crit_chance("tackle") == pytest.approx(1 / 24)
    assert crit_chance("stoneedge") == pytest.approx(1 / 8)  # critRatio 2
    for always in ("flowertrick", "wickedblow", "surgingstrikes", "frostbreath", "stormthrow"):
        assert crit_chance(always) == 1.0
        assert crit_chance(always, defender_immune=True) == 0.0
    assert crit_stage_bonus("scopelens", None, False) == 1
    assert crit_stage_bonus("razorclaw", "superluck", False) == 2
    assert crit_stage_bonus(None, None, True) == 2  # Focus Energy
    assert crit_chance("tackle", crit_stage_bonus(None, None, True)) == pytest.approx(1 / 2)
    assert crit_chance("stoneedge", crit_stage_bonus("scopelens", "superluck", False)) == 1.0
    assert crit_chance("stoneedge", 3, defender_immune=True) == 0.0


def test_crit_block_features():
    msgs = [["", "turn", "1"], _sw("p1", "Garchomp"), _sw("p2", "Blissey"),
            _mv("p1", "Garchomp", "Stone Edge", "p2a: Blissey"), _mv("p1", "Garchomp", "Earthquake", "p2a: Blissey")]
    b = _battle(msgs, my_item="scopelens")
    cb = crit_block(b)
    _check(cb, CRIT_BLOCK_SIZE)
    assert cb[0] == pytest.approx(1 / 2)  # Stone Edge (+1) with Scope Lens (+1) -> stage 2
    assert cb[3] == pytest.approx(1 / 8)  # Earthquake with Scope Lens
    assert cb[20] == pytest.approx(1 / 3)  # our crit-stage bonus
    b.active_pokemon.start_effect("move: Focus Energy")
    assert Effect.FOCUS_ENERGY in b.active_pokemon.effects
    cb = crit_block(b)
    assert cb[0] == 1.0 and cb[18] == 1.0 and cb[20] == 1.0
    # Shell Armor on our side blocks the opponent's crits
    b.active_pokemon.ability = "shellarmor"
    cb = crit_block(b)
    assert cb[16] == 1.0 and cb[12] == 0.0


# --------------------------------------------------------------------------- #
# Protocol / Choice lock
# --------------------------------------------------------------------------- #
def test_stint_moves():
    msgs = [["", "turn", "1"], _sw("p1", "Garchomp"), _sw("p2", "Blissey"),
            _mv("p1", "Garchomp", "Earthquake", "p2a: Blissey"),
            _mv("p2", "Blissey", "Seismic Toss", "p1a: Garchomp"),
            ["", "turn", "2"],
            _mv("p1", "Garchomp", "Outrage", "p2a: Blissey"),
            _mv("p1", "Garchomp", "Outrage", "p2a: Blissey", "[from]lockedmove"),
            _sw("p2", "Corviknight"),
            _mv("p2", "Corviknight", "U-turn", "p1a: Garchomp")]
    b = _battle(msgs)
    assert stint_moves(b) == (["earthquake", "outrage"], ["uturn"])
    assert stint_moves(_empty_battle()) == ([], [])


def test_our_choice_lock():
    base = [["", "turn", "1"], _sw("p1", "Garchomp"), _sw("p2", "Heatran")]
    b = _battle(base, my_item="choiceband")
    assert items_block(b)[45] == 0.0  # just switched in: not locked yet
    b = _battle(base + [_mv("p1", "Garchomp", "Earthquake", "p2a: Heatran")], my_item="choiceband")
    ib = items_block(b)
    assert ib[I["choiceband"]] == 1.0
    assert ib[45] == 1.0 and ib[46] == 1.0 and ib[47] > 0.5  # locked into a 4x super-effective move
    b = _battle(base + [_mv("p1", "Garchomp", "Earthquake", "p2a: Heatran")], my_item="leftovers")
    assert items_block(b)[45] == 0.0
    # switched out and back in: lock resets
    b = _battle(base + [_mv("p1", "Garchomp", "Earthquake", "p2a: Heatran"), _sw("p1", "Blissey"),
                        _sw("p1", "Garchomp")], my_item="choiceband")
    assert items_block(b)[45] == 0.0


def test_opponent_choice_inference():
    base = [["", "turn", "1"], _sw("p1", "Heatran"), _sw("p2", "Garchomp")]
    # revealed Choice Band, one move used -> locked into Earthquake, which is SE on Heatran
    b = _battle(base + [_mv("p2", "Garchomp", "Earthquake", "p1a: Heatran")], opp_item="choiceband")
    ib = items_block(b)
    assert ib[22 + I["choiceband"]] == 1.0 and ib[44] == 0.0
    assert ib[48] == 1.0 and ib[49] == 1.0 and ib[50] == 1.0 and ib[51] > 0.5 and ib[52] == 0.0
    # locked into a Ground move against a Flying type: a free turn
    b = _battle([["", "turn", "1"], _sw("p1", "Corviknight"), _sw("p2", "Garchomp"),
                 _mv("p2", "Garchomp", "Earthquake", "p1a: Corviknight")], opp_item="choicescarf")
    assert items_block(b)[52] == 1.0
    # unknown item: Choice probability comes from the prior; two different moves rule it out
    sp = "Hydreigon"
    pri = species_item_prior("hydreigon")
    p_choice = sum(pri[I[c]] for c in ("choiceband", "choicespecs", "choicescarf"))
    assert p_choice > 0.1
    b = _battle([["", "turn", "1"], _sw("p1", "Heatran"), _sw("p2", sp), _mv("p2", sp, "Draco Meteor", "p1a: Heatran")])
    ib = items_block(b)
    assert ib[44] == 1.0 and ib[48] > 0.1 and ib[49] == ib[48]
    assert sum(ib[22:44]) == pytest.approx(1.0, abs=1e-6)
    b = _battle([["", "turn", "1"], _sw("p1", "Heatran"), _sw("p2", sp), _mv("p2", sp, "Draco Meteor", "p1a: Heatran"),
                 ["", "turn", "2"], _mv("p2", sp, "Dark Pulse", "p1a: Heatran")])
    ib = items_block(b)
    assert ib[48] == 0.0 and ib[49] == 0.0
    # Knocked off / consumed -> "none"
    b = _battle(base)
    b.opponent_active_pokemon.end_item("leftovers")
    ib = items_block(b)
    assert ib[22 + I["none"]] == 1.0 and ib[44] == 0.0 and ib[48] == 0.0


def test_item_prior():
    for sp in ("chansey", "garchomp", "blissey", "notarealmon"):
        pr = species_item_prior(sp)
        assert len(pr) == len(ITEM_VOCAB) and sum(pr) == pytest.approx(1.0) and min(pr) >= 0.0
    assert max(range(len(ITEM_VOCAB)), key=species_item_prior("chansey").__getitem__) == I["eviolite"]
    assert species_item_prior("volcarona")[I["heavydutyboots"]] > 0.9  # 4x Rock weak
    # a revealed status move rules out Choice items and Assault Vest
    pr = species_item_prior("hydreigon", frozenset({"nastyplot"}))
    assert all(pr[I[c]] == 0.0 for c in ("choiceband", "choicespecs", "choicescarf", "assaultvest"))


def test_item_one_hots_and_boots():
    b = _battle([["", "turn", "1"], _sw("p1", "Garchomp"), _sw("p2", "Rotom-Wash")], my_item="heavydutyboots")
    ib = items_block(b)
    assert ib[I["heavydutyboots"]] == 1.0 and sum(ib[:22]) == 1.0
    assert ib[65] == 1.0 and sum(ib[65:71]) == 1.0
    b.active_pokemon.item = ""  # consumed / knocked off (our request item is "")
    assert items_block(b)[I["none"]] == 1.0
    b.active_pokemon.item = "mysticwater"
    assert items_block(b)[I["other"]] == 1.0


def test_scarf_speed_feature():
    # Garchomp (base 102) vs Dragapult (base 142): Dragapult outspeeds unless we hold a Scarf
    msgs = [["", "turn", "1"], _sw("p1", "Garchomp", 80), _sw("p2", "Dragapult", 80)]
    b = _battle(msgs, opp_item="lifeorb")
    assert items_block(b)[64] == 1.0
    b = _battle(msgs, my_item="choicescarf", opp_item="lifeorb")
    assert items_block(b)[64] == 0.0
    b = _battle(msgs, my_item="choicescarf", opp_item="choicescarf")
    assert items_block(b)[64] == 1.0


def test_sash_blocks_ko_flag():
    msgs = [["", "turn", "1"], _sw("p1", "Garchomp"), _sw("p2", "Heatran"),
            _mv("p1", "Garchomp", "Earthquake", "p2a: Heatran")]
    ib = items_block(_battle(msgs, opp_item="leftovers"))
    assert ib[53] == 1.0 and ib[54] == 1.0
    ib = items_block(_battle(msgs, opp_item="focussash"))
    assert ib[54] == 0.0  # full HP + Sash: single-hit move cannot KO
    ib = items_block(_battle(msgs, opp_item="airballoon"))
    assert ib[53] == 0.0 and ib[54] == 0.0
