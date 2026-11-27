"""Tests for pokebot/v3_history.py (battle-memory observation block). No server needed.

Run from the project root:
    .venv/bin/python -m pytest tests/test_v3_history.py -q
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_env.battle import Battle  # noqa: E402

from pokebot import v3_history as h  # noqa: E402
from pokebot.v3_history import (  # noqa: E402
    AGG_FIELDS,
    AGG_SLICE,
    HISTORY_V3_SIZE,
    LAST_ACTION_FIELDS,
    LAST_ACTION_SLICE,
    TURN_FEATS,
    TURN_FIELDS,
    history_block,
)

MOVES = ["swordsdance", "earthquake", "stoneedge", "firefang"]  # Earthquake is slot 1 -> action 7


def _battle(role: str, tag: str = "battle-gen9randombattle-1") -> Battle:
    b = Battle(tag, "tester", logging.getLogger("test"), gen=9)
    b.player_role = role
    return b


def _turn(vec, k):
    rec = vec[k * TURN_FEATS:(k + 1) * TURN_FEATS]
    return dict(zip(TURN_FIELDS, rec))


def _agg(vec):
    return dict(zip(AGG_FIELDS, vec[AGG_SLICE]))


def _last(vec):
    return [n for n, v in zip(LAST_ACTION_FIELDS, vec[LAST_ACTION_SLICE]) if v > 0]


def _check(vec):
    assert len(vec) == HISTORY_V3_SIZE
    assert all(isinstance(v, float) for v in vec)
    assert all(-1.0 <= v <= 1.0 for v in vec)


# A crafted log written from p1's point of view ("us" = p1, Garchomp). The opponent's HP is /100,
# ours absolute, like the real server sends it.
LEAD = [
    ["", "player", "p1", "tester", "", ""],
    ["", "start"],
    ["", "switch", "p1a: Garchomp", "Garchomp, L80, M", "300/300"],
    ["", "switch", "p2a: Heatran", "Heatran, L82, M", "100/100"],
    ["", "turn", "1"],
]
TURN1 = [  # Heatran moves first (resisted), then our Earthquake crits, is SE and KOs.
    ["", ""],
    ["", "t:", "1"],
    ["", "move", "p2a: Heatran", "Magma Storm", "p1a: Garchomp"],
    ["", "-resisted", "p1a: Garchomp"],
    ["", "-damage", "p1a: Garchomp", "240/300"],
    ["", "move", "p1a: Garchomp", "Earthquake", "p2a: Heatran"],
    ["", "-supereffective", "p2a: Heatran"],
    ["", "-crit", "p2a: Heatran"],
    ["", "-damage", "p2a: Heatran", "0 fnt"],
    ["", "faint", "p2a: Heatran"],
    ["", ""],
    ["", "upkeep"],
    ["", "switch", "p2a: Gholdengo", "Gholdengo, L78", "100/100"],  # faint replacement: not voluntary
    ["", "turn", "2"],
]
TURN2 = [  # Opponent switches out (voluntarily) into Corviknight; our Earthquake is immune.
    ["", ""],
    ["", "switch", "p2a: Corviknight", "Corviknight, L80, M", "100/100"],
    ["", "move", "p1a: Garchomp", "Earthquake", "p2a: Corviknight"],
    ["", "-immune", "p2a: Corviknight"],
    ["", ""],
    ["", "upkeep"],
    ["", "turn", "3"],
]
TURN3_PARTIAL = [  # Corviknight crits and KOs us; we're asked for a replacement mid-turn.
    ["", ""],
    ["", "move", "p2a: Corviknight", "Brave Bird", "p1a: Garchomp"],
    ["", "-crit", "p1a: Garchomp"],
    ["", "-damage", "p1a: Garchomp", "0 fnt"],
    ["", "-damage", "p2a: Corviknight", "80/100", "[from] Recoil"],
    ["", "faint", "p1a: Garchomp"],
    ["", ""],
    ["", "upkeep"],
]


def _mirror(lines):
    """Swap p1 <-> p2 everywhere (to replay the same story as p2)."""
    out = []
    for ev in lines:
        new = []
        for tok in ev:
            if tok.startswith("p1"):
                tok = "p2" + tok[2:]
            elif tok.startswith("p2"):
                tok = "p1" + tok[2:]
            new.append(tok)
        out.append(new)
    return out


def _feed(b: Battle, lines, me: str):
    for ev in lines:
        b.parse_message(ev)
        if ev[1] == "switch" and ev[2].startswith(me + "a: Garchomp"):
            mon = b.team[f"{me}: Garchomp"]
            for m in MOVES:
                mon._add_move(m)


# --------------------------------------------------------------------------- #
def test_size_and_layout():
    assert HISTORY_V3_SIZE == 6 * TURN_FEATS + len(AGG_FIELDS) + len(LAST_ACTION_FIELDS) == 196
    assert LAST_ACTION_SLICE.stop == HISTORY_V3_SIZE


def test_never_raises_on_empty_missing_and_garbage():
    zeros = [0.0] * HISTORY_V3_SIZE
    assert history_block(None) == zeros
    assert history_block(SimpleNamespace(player_role="p1")) == zeros  # no _replay_data
    assert history_block(SimpleNamespace(player_role="p1", _replay_data=None)) == zeros
    b = _battle("p1")
    assert history_block(b) == zeros  # empty log
    nb = _battle(None)
    nb._replay_data.extend(LEAD)
    assert history_block(nb) == zeros  # unknown role
    junk = SimpleNamespace(player_role="p2", battle_tag="x", team={"p2: X": 5}, active_pokemon=None,
                           available_moves=None, _replay_data=[
                               [], ["x"], None, 5, "str", ["", "move"], ["", "turn", "notanint"],
                               ["", "-damage", "p1a: X", "abc"], ["", "switch", "p2a"], ["", "turn", "4"],
                               ["", "switch", "p2a: X", "", "1/0"], ["", "move", "p2a: X", "Tackle", "", "[miss]"],
                               ["", "-sidestart", "p1"], ["", "-status", "p1a: Y"], ["", "faint", 7],
                               ["", "-crit", "p3a: Z"], ["", "-supereffective"], ["", "move", "p1a: Q", "???"]])
    vec = history_block(junk)
    _check(vec)
    assert history_block(junk, use_cache=False) == vec


@pytest.mark.parametrize("me", ["p1", "p2"])
def test_crafted_log(me):
    lines_lead, t1, t2, t3 = LEAD, TURN1, TURN2, TURN3_PARTIAL
    if me == "p2":
        lines_lead, t1, t2, t3 = map(_mirror, (LEAD, TURN1, TURN2, TURN3_PARTIAL))
    b = _battle(me, tag=f"battle-gen9randombattle-crafted-{me}")
    _feed(b, lines_lead, me)
    v = history_block(b)
    _check(v)
    assert _turn(v, 0)["present"] == 0  # nothing has happened yet at turn 1
    assert _last(v) == []

    _feed(b, t1, me)
    v = history_block(b)
    _check(v)
    r = _turn(v, 0)
    assert r["present"] == 1 and r["in_progress"] == 0
    assert r["me_move"] == 1 and r["me_switch"] == 0
    assert r["me_slot1"] == 1 and r["me_slot0"] == r["me_slot2"] == r["me_slot3"] == 0
    assert r["me_bp"] == pytest.approx(100 / 150)
    assert r["me_eff"] == 1.0 and r["me_dmg"] == pytest.approx(1.0) and r["me_ko"] == 1
    assert r["crit"] == 1.0
    assert r["opp_move"] == 1 and r["opp_eff"] == -0.5 and r["opp_dmg"] == pytest.approx(0.2)
    assert r["opp_bp"] == pytest.approx(100 / 150)
    assert r["opp_switch"] == 0  # replacement after a faint is not a decision to switch
    assert r["first"] == -1.0
    assert _turn(v, 1)["present"] == 0
    a = _agg(v)
    assert a["kos_by_me"] == pytest.approx(1 / 6) and a["crits_by_me"] == pytest.approx(0.25)
    assert a["slot1_uses"] == pytest.approx(0.1)
    assert _last(v) == ["slot1"]

    _feed(b, t2, me)
    v = history_block(b)
    r0, r1 = _turn(v, 0), _turn(v, 1)
    assert r0["opp_switch"] == 1 and r0["opp_move"] == 0
    assert r0["me_move"] == 1 and r0["me_eff"] == -1.0 and r0["me_dmg"] == 0 and r0["first"] == 0
    assert r1["me_ko"] == 1  # previous turn shifted to slot 1 of the record list
    a = _agg(v)
    assert a["opp_switch_rate"] == pytest.approx(1 / 3)  # 1 voluntary switch in 3 turns
    assert a["slot1_uses"] == pytest.approx(0.2)
    assert a["opp_moves_seen"] == pytest.approx(1 / 24)
    # Both turns began with an Earthquake-weak foe (Heatran stayed, Gholdengo switched): (1+1)/(2+2).
    assert a["opp_sw_threat"] == pytest.approx(0.5) and a["opp_sw_threat_conf"] == pytest.approx(0.2)
    assert a["opp_sw_safe"] == pytest.approx(0.5)  # no unthreatened turns yet: prior

    _feed(b, t3, me)
    v = history_block(b)
    r = _turn(v, 0)
    assert r["in_progress"] == 1 and r["opp_ko"] == 1 and r["crit"] == -1.0
    assert r["opp_dmg"] == pytest.approx(0.8)  # 240/300 -> 0; recoil on them isn't our damage
    assert r["me_move"] == 0
    assert _turn(v, 1)["opp_switch"] == 1
    a = _agg(v)
    assert a["kos_on_me"] == pytest.approx(1 / 6) and a["crits_on_me"] == pytest.approx(0.25)
    assert a["dmg_dealt"] == pytest.approx(0.1) and a["dmg_taken"] == pytest.approx(0.1)

    # Replacement + next turn: the switch becomes our last action.
    _feed(b, [["", "switch", f"{me}a: Dragonite", "Dragonite, L74, F", "280/280"], ["", "turn", "4"]], me)
    v = history_block(b)
    assert _last(v) == ["switch"]
    assert _agg(v)["my_switch_rate"] == 0.0


def test_both_perspectives_identical():
    vecs = []
    for me in ("p1", "p2"):
        b = _battle(me, tag=f"battle-gen9randombattle-persp-{me}")
        lines = LEAD + TURN1 + TURN2 + TURN3_PARTIAL
        _feed(b, lines if me == "p1" else _mirror(lines), me)
        vecs.append(history_block(b))
    assert vecs[0] == vecs[1]


def test_tera_status_hazards_protect():
    b = _battle("p1", tag="battle-gen9randombattle-misc")
    _feed(b, LEAD + [
        ["", "-terastallize", "p1a: Garchomp", "Fire"],
        ["", "move", "p2a: Heatran", "Protect", "p2a: Heatran"],
        ["", "-singleturn", "p2a: Heatran", "Protect"],
        ["", "move", "p1a: Garchomp", "Earthquake", "p2a: Heatran"],
        ["", "-activate", "p2a: Heatran", "move: Protect"],
        ["", "upkeep"], ["", "turn", "2"],
        ["", "move", "p1a: Garchomp", "Stealth Rock", "p2a: Heatran"],
        ["", "-sidestart", "p2: opponent", "move: Stealth Rock"],
        ["", "move", "p2a: Heatran", "Will-O-Wisp", "p1a: Garchomp"],
        ["", "-status", "p1a: Garchomp", "brn"],
        ["", "upkeep"], ["", "turn", "3"],
    ], "p1")
    v = history_block(b)
    r0, r1 = _turn(v, 0), _turn(v, 1)
    assert r1["me_tera"] == 1 and r1["me_fail"] == 1 and r1["opp_fail"] == 0 and r1["first"] == -1
    assert r0["status"] == -1.0 and r0["me_bp"] == 0 and r0["first"] == 1
    a = _agg(v)
    assert a["me_tera_used"] == 1 and a["me_tera_turn"] == pytest.approx(1 / 60)
    assert a["opp_tera_used"] == 0 and a["my_hazards_set"] == pytest.approx(0.25)
    assert _last(v) == ["other"]  # Stealth Rock isn't in Garchomp's 4 tracked slots here


def test_cache_matches_fresh_parse_and_no_leak():
    h.clear_cache()
    lines = LEAD + TURN1 + TURN2 + TURN3_PARTIAL
    b = _battle("p1", tag="battle-gen9randombattle-cache")
    for ev in lines:
        _feed(b, [ev], "p1")
        assert history_block(b) == history_block(b, use_cache=False)
    # A *different* battle object with the same tag (e.g. a new env episode) must not reuse the state.
    b2 = _battle("p1", tag="battle-gen9randombattle-cache")
    _feed(b2, LEAD, "p1")
    assert history_block(b2) == history_block(b2, use_cache=False)
    assert _turn(history_block(b2), 0)["present"] == 0
    # The log list being replaced on the same object also invalidates the cached state.
    b._replay_data = [list(ev) for ev in LEAD]
    assert history_block(b) == history_block(b, use_cache=False)


def test_timing_long_battle():
    h.clear_cache()
    b = _battle("p1", tag="battle-gen9randombattle-long")
    _feed(b, LEAD, "p1")
    body = [ev for ev in TURN1 + TURN2 if ev[1] != "turn"]
    t_total, calls = 0.0, 0
    for turn in range(2, 120):
        _feed(b, body + [["", "turn", str(turn)]], "p1")
        t0 = time.perf_counter()
        vec = history_block(b)
        t_total += time.perf_counter() - t0
        calls += 1
    _check(vec)
    assert len(b._replay_data) > 2000
    t0 = time.perf_counter()
    fresh = history_block(b, use_cache=False)
    t_fresh = time.perf_counter() - t0
    assert fresh == vec
    per_call = t_total / calls
    print(f"\nhistory_block cached: {per_call * 1e3:.3f} ms/call, fresh parse of "
          f"{len(b._replay_data)} msgs: {t_fresh * 1e3:.2f} ms")
    assert per_call < 0.5e-3
