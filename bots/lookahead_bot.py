"""A one-turn lookahead bot: scores every legal action with the v2-v4 damage model.

For each of our moves (with and without Terastallization) and each switch it estimates
what happens this turn: who moves first (priority, speed incl. abilities/items/weather/
Trick Room), how much each side loses, the chance of a KO on either side, and end-of-turn
residual damage (sand, status, Leftovers, ... incl. a switch-in that changes the weather).
The opponent is assumed to answer with its most damaging likely move (from the
random-battle sets, over the posterior of its ability). On top of this turn's exchange it
adds a discounted value for the matchup it leaves us in, so switching into a better
matchup can beat attacking.

E10 additions (both on by default; `LookaheadBot(tera=False, predict_switch=False)` is the
E9 bot, registered as the opponent `lookahead1`):
  tera            score "move + terastallize" too: our types become the tera type (defence
                  and STAB change), minus a cost for spending the once-per-game tera that
                  shrinks as our team shrinks.
  predict_switch  if their active is losing the matchup and they have a revealed bench mon
                  that takes much less from us, assume they switch with some probability,
                  and value each move partly by what it does to that switch-in (a move the
                  switch-in is immune to loses value; setup and hazards gain).

`action_scores(battle)` returns the score of every action index of the 26-action space
(NaN for illegal ones). The policy+search blend (`pokebot.search`) and the v5 features use it.

Run vs the scripted bots (Showdown server on localhost:8000):
    .venv/bin/python -m pokebot.evaluate --model lookahead --opponents simple_heuristic --battles 200
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np
from poke_env.battle import Battle, Pokemon, PokemonType, SideCondition
from poke_env.player import Player

from pokebot.config import ACTION_SPACE_SIZE
from pokebot.features_v2 import _TYPE_INDEX, Combatant, MoveInfo, move_info
from pokebot.v3_items import estimate_damage_items
from pokebot.v4_abilities import (_MOLD, _ACtx, _item, _known_ability, _opp_item, ability_damage,
                                  ability_posterior, nullifies, residual)

W_KO = 0.6  # value of knocking a mon out, on top of the HP it had (units: a full-HP mon = 1.0)
GAMMA = 0.6  # weight of the matchup we are left in, relative to this turn's exchange
TERA_COST_PER_MON = 0.04  # cost of spending tera, per mon we still have (saving it matters less late)
P_SWITCH_MAX = 0.5  # most we ever believe they switch
SWITCH_MARGIN = 0.25  # how far (in HP units) they must be losing the damage race before switching gets likely
SWITCH_SCALE = 0.1
SWITCH_BETTER = 0.15  # their switch-in must take at least this much less from us than their active

_STATUS_IMMUNE = {
    "willowisp": (PokemonType.FIRE,),
    "toxic": (PokemonType.POISON, PokemonType.STEEL),
    "poisonpowder": (PokemonType.POISON, PokemonType.STEEL, PokemonType.GRASS),
    "thunderwave": (PokemonType.ELECTRIC, PokemonType.GROUND),
    "glare": (PokemonType.ELECTRIC,),
    "stunspore": (PokemonType.ELECTRIC, PokemonType.GRASS),
    "spore": (PokemonType.GRASS,),
    "sleeppowder": (PokemonType.GRASS,),
}
_ROCK = move_info("stealthrock")


def _with_tera(c: Combatant, t) -> Optional[Combatant]:
    """A copy of our combatant after terastallizing into type `t` (None if `t` is not a normal type)."""
    if t not in _TYPE_INDEX:
        return None
    n = object.__new__(Combatant)
    for s in Combatant.__slots__:
        setattr(n, s, getattr(c, s))
    n.types = [t]
    n.tera = t
    n._eff = {}
    return n


def our_tera_type(mon):
    """Our mon's tera type. poke-env leaves `tera_type` empty until we tera, so read Showdown's request."""
    t = mon.tera_type
    if t is not None:
        return t
    req = getattr(mon, "_last_request", None) or {}
    name = req.get("teraType")
    if not name:
        return None
    try:
        return PokemonType.from_name(name)
    except Exception:
        return None


def move_list_for_actions(battle: Battle) -> list:
    """The move list that actions 6-9 / 22-25 index into (same rule as SinglesEnv.action_to_order)."""
    me = battle.active_pokemon
    if me is None:
        return []
    avail = list(battle.available_moves or [])
    known = list(me.moves.values())[:4]
    known_ids = [m.id for m in known]
    if len(avail) == 1 and avail[0].id not in known_ids:
        return avail
    return known


class LookaheadScorer:
    """The scoring half of the bot, usable without a Showdown connection (search blend, v5 features)."""

    def __init__(self, tera: bool = True, predict_switch: bool = True):
        self.tera = tera
        self.predict_switch = predict_switch

    def action_scores(self, battle: Battle) -> np.ndarray:
        """Score per action index (0-5 switch to team slot, 6-9 move, 22-25 move + tera); NaN = not scored."""
        out = np.full(ACTION_SPACE_SIZE, np.nan, dtype=np.float64)
        try:
            scored = self.score_actions(battle)
        except Exception:
            return out
        team = list(battle.team.values())
        mvs = move_list_for_actions(battle)
        ids = [m.id for m in mvs]
        for v, obj, tera in scored:
            if isinstance(obj, Pokemon):
                for i, m in enumerate(team):
                    if m is obj:
                        out[i] = v
            else:
                mid = getattr(obj, "id", None)
                for i, x in enumerate(ids[:4]):
                    if x == mid:
                        out[(22 if tera else 6) + i] = v
        return out

    def score_actions(self, battle: Battle) -> List[Tuple[float, object, bool]]:
        """[(value, Move | Pokemon, terastallize)] for every legal action this turn."""
        ac = _ACtx(battle)
        v2 = ac.v2
        moves = list(battle.available_moves or [])
        switches = list(battle.available_switches or [])
        if v2.opp_c is None:  # nothing to plan against yet: strongest-looking move or first switch
            if moves:
                return [(m.base_power * m.accuracy if m.accuracy is not True else m.base_power, m, False)
                        for m in moves]
            return [(0.0, s, False) for s in switches]
        forced = bool(battle.force_switch) or v2.me_c is None
        out: List[Tuple[float, object, bool]] = []
        if not forced and v2.me_c is not None and moves:
            me_ab, me_item = ac.me_ab, ac.me_item
            variants = [(v2.me_c, False, 0.0)]
            if self.tera and battle.can_tera and v2.me is not None:
                tc = _with_tera(v2.me_c, our_tera_type(v2.me))
                if tc is not None:
                    alive = sum(1 for m in battle.team.values() if not m.fainted)
                    variants.append((tc, True, TERA_COST_PER_MON * max(alive - 1, 0)))
            sw = self._switch_guess(ac, battle, v2.me_c, me_ab, me_item, moves) if self.predict_switch else None
            for me, tera, cost in variants:
                stay = self._matchup(ac, battle, me, me_ab, me_item, v2.me, ac.weather)
                for mv in moves:
                    mi = move_info(mv)
                    gain, p_survive = self._exchange(ac, battle, me, me_ab, me_item, mi, ac.weather)
                    v = gain + GAMMA * p_survive * stay
                    if sw is not None:
                        p_sw, target = sw
                        v = (1 - p_sw) * v + p_sw * (self._vs_switch_in(ac, battle, me, me_ab, me_item, mi, target)
                                                     + GAMMA * stay)
                    out.append((v - cost, mv, tera))
        for mon in switches:
            out.append((self._switch_value(ac, battle, mon, forced), mon, False))
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ours_vs(ac: _ACtx, mi: MoveInfo, me: Combatant, me_ab, me_item, tc: Combatant, tpost, titem
                 ) -> Tuple[float, float]:
        """(expected damage as a fraction of tc's current HP, P(KO)) of our move vs any of their mons."""
        base = estimate_damage_items(mi, me, tc, ac.weather, att_item=me_item, dfn_item=titem)
        if base <= 0.0:
            return 0.0, 0.0
        exp = p_ko = 0.0
        for a, p in (tpost or ((None, 1.0),)):
            if me_ab not in _MOLD and nullifies(a, mi, me_ab):
                continue
            d, ko = ability_damage(base, mi, me, tc, me_ab, a, ac.weather, ac.terrain)
            exp += p * min(d, 1.0)
            p_ko += p * float(ko)
        return exp, p_ko

    def _switch_guess(self, ac: _ACtx, battle: Battle, me: Combatant, me_ab, me_item, moves):
        """(P(they switch this turn), (combatant, posterior, item) of the likely switch-in) or None."""
        v2 = ac.v2
        opp = v2.opp_c
        infos = [mi for mi in (move_info(m) for m in moves) if mi is not None and not mi.status]
        if not infos:
            return None
        bench = [m for m in (battle.opponent_team or {}).values() if not m.fainted and not m.active]
        if not bench:
            return None

        def taken(tc, tpost, titem) -> float:
            best = 0.0
            for mi in infos:
                exp, ko = self._ours_vs(ac, mi, me, me_ab, me_item, tc, tpost, titem)
                best = max(best, mi.acc * (min(exp, 1.0) * tc.hp_frac + W_KO * ko))
            return best

        ours_now = taken(opp, ac.opp_post, ac.opp_item)
        threat, threat_ko, _ = ac.opp_vs(me, me_ab, me_item, ac.weather)
        theirs_now = min(threat, 1.0) * me.hp_frac + W_KO * threat_ko
        best_t, best_taken = None, None
        for m in bench:
            tc = v2.combatant(m, False)
            t = (tc, ability_posterior(m), _opp_item(m))
            x = taken(*t)
            if best_taken is None or x < best_taken:
                best_t, best_taken = t, x
        if best_t is None or best_taken > ours_now - SWITCH_BETTER:
            return None
        z = (ours_now - theirs_now - SWITCH_MARGIN) / SWITCH_SCALE
        p = P_SWITCH_MAX / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
        return (p, best_t) if p > 0.01 else None

    def _vs_switch_in(self, ac: _ACtx, battle: Battle, me: Combatant, me_ab, me_item, mi: Optional[MoveInfo],
                      target) -> float:
        """Value of `mi` if they switch: it hits the switch-in, and nobody attacks us this turn."""
        tc, tpost, titem = target
        r_me = residual(me, me_ab, me_item, ac.weather, ac.terrain)
        if mi is None:
            return r_me
        if mi.status:
            return self._status_value(ac, battle, me, mi, 0.0) + r_me
        exp, ko = self._ours_vs(ac, mi, me, me_ab, me_item, tc, tpost, titem)
        return mi.acc * (min(exp, 1.0) * tc.hp_frac + W_KO * ko) + r_me

    def _exchange(self, ac: _ACtx, battle: Battle, me: Combatant, me_ab, me_item, mi: Optional[MoveInfo],
                  weather) -> Tuple[float, float]:
        """(expected value of using `mi` this turn, P(our mon survives the turn))."""
        v2 = ac.v2
        opp = v2.opp_c
        threat, threat_ko, _ = ac.opp_vs(me, me_ab, me_item, weather)
        threat = min(threat, 1.0)
        hit = ko = 0.0
        bonus = 0.0
        if mi is not None and not mi.status:
            _, exp, p_ko = ac.ours_vs_opp(mi, me, me_ab, me_item)
            hit, ko = mi.acc * min(exp, 1.0), mi.acc * p_ko
        elif mi is not None:
            bonus = self._status_value(ac, battle, me, mi, threat)
        if mi is not None and mi.priority > 0:
            pf = 1.0
        elif mi is not None and mi.priority < 0:
            pf = 0.0
        else:
            pf = ac.p_faster(me, me_ab, me_item, weather, v2.my_tailwind)
        # We move first: a KO stops their attack. They move first: a KO stops ours.
        opp_loss = pf * hit + (1 - pf) * (1 - threat_ko) * hit
        p_ko_them = pf * ko + (1 - pf) * (1 - threat_ko) * ko
        our_loss = pf * (1 - ko) * threat + (1 - pf) * threat
        p_ko_us = pf * (1 - ko) * threat_ko + (1 - pf) * threat_ko
        if mi is not None and mi.status and not mi.recovery:
            bonus *= 1 - p_ko_us  # a setup / status move only counts if we live to use it (roughly)
        gain = opp_loss * opp.hp_frac - our_loss * me.hp_frac + W_KO * (p_ko_them - p_ko_us) + bonus
        # End of turn: residual damage on whoever is still in.
        r_opp = ac.opp_residual(weather)
        opp_left = opp.hp_frac * (1 - opp_loss)
        gain += (1 - p_ko_them) * (-r_opp + (W_KO if (r_opp < 0 and -r_opp >= opp_left - 1e-9) else 0.0))
        r_me = residual(me, me_ab, me_item, weather, ac.terrain)
        gain += (1 - p_ko_us) * r_me
        return gain, 1 - p_ko_us

    def _status_value(self, ac: _ACtx, battle: Battle, me: Combatant, mi: MoveInfo, threat: float) -> float:
        opp = ac.v2.opp_c
        if mi.recovery:
            return min(0.5, 1.0 - me.hp_frac) if me.hp_frac < 0.65 else -0.05
        if mi.setup:
            boosted = max((me.boosts or {}).get(k, 0) for k in ("atk", "spa", "spe", "def", "spd"))
            return 0.25 if (threat < 0.35 and me.hp_frac > 0.5 and boosted < 2) else -0.05
        if mi.hazard:
            sc = battle.opponent_side_conditions or {}
            left = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
            if mi.id == "stealthrock" and SideCondition.STEALTH_ROCK not in sc:
                return 0.05 * left
            if mi.id == "spikes" and sc.get(SideCondition.SPIKES, 0) < 3:
                return 0.03 * left
            if mi.id == "toxicspikes" and sc.get(SideCondition.TOXIC_SPIKES, 0) < 2:
                return 0.02 * left
            if mi.id == "stickyweb" and SideCondition.STICKY_WEB not in sc:
                return 0.03 * left
            return -0.05
        if mi.inflicts:
            if opp.mon.status is not None or any(t in opp.types for t in _STATUS_IMMUNE.get(mi.id, ())):
                return -0.1
            if mi.id in ("thunderwave", "glare", "stunspore", "nuzzle"):
                return 0.15
            return 0.2
        return -0.02  # protect, screens, etc.: a small cost so attacking wins ties

    def _matchup(self, ac: _ACtx, battle: Battle, c: Combatant, ab, item, mon, weather) -> float:
        """Best one-turn exchange value for `mon` against their active (0 if it has no known moves)."""
        best = None
        for mv in list(mon.moves.values())[:4]:
            mi = move_info(mv)
            if mi is None:
                continue
            g, _ = self._exchange(ac, battle, c, ab, item, mi, weather)
            best = g if best is None else max(best, g)
        return best if best is not None else 0.0

    def _switch_value(self, ac: _ACtx, battle: Battle, mon, forced: bool) -> float:
        v2 = ac.v2
        c = v2.combatant(mon, True)
        ab, item = _known_ability(mon), _item(mon)
        weather, _changed = ac.weather_after(ab)
        hazard = self._hazard_damage(battle, c, ab, item)
        if forced:
            loss, p_die = hazard, 0.0
        else:
            incoming, inc_ko, _ = ac.opp_vs(c, ab, item, weather)
            loss = min(incoming, 1.0) * c.hp_frac + hazard
            p_die = inc_ko
        opp = v2.opp_c
        r_opp = ac.opp_residual(weather)
        res = -r_opp + (W_KO if (r_opp < 0 and -r_opp >= opp.hp_frac - 1e-9) else 0.0)
        future = self._matchup(ac, battle, c, ab, item, mon, weather)
        return -loss - W_KO * p_die + res + GAMMA * (1 - p_die) * future

    @staticmethod
    def _hazard_damage(battle: Battle, c: Combatant, ab, item) -> float:
        if ab == "magicguard" or item == "heavydutyboots":
            return 0.0
        sc = battle.side_conditions or {}
        d = 0.0
        if SideCondition.STEALTH_ROCK in sc and _ROCK is not None:
            d += 0.125 * c.effectiveness(_ROCK)
        layers = sc.get(SideCondition.SPIKES, 0)
        grounded = not (PokemonType.FLYING in c.types or ab == "levitate" or item == "airballoon")
        if layers and grounded:
            d += {1: 1 / 8, 2: 1 / 6}.get(layers, 1 / 4)
        return d


class LookaheadBot(Player, LookaheadScorer):
    """Greedy one-turn search over our actions with the feature damage model."""

    def __init__(self, *args, epsilon: float = 0.0, tera: bool = True, predict_switch: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon  # chance to play the second-best action (diversity as a training opponent)
        self.tera = tera
        self.predict_switch = predict_switch

    # ------------------------------------------------------------------ #
    def choose_move(self, battle):
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        try:
            scored = self.score_actions(battle)
        except Exception:  # never crash a battle over a model gap
            scored = []
        if not scored:
            return self.choose_random_move(battle)
        scored.sort(key=lambda t: t[0], reverse=True)
        pick = scored[1] if (len(scored) > 1 and self.epsilon > 0 and random.random() < self.epsilon) else scored[0]
        return self.create_order(pick[1], terastallize=pick[2])


class LookaheadBotE9(LookaheadBot):
    """The E9 lookahead bot: no tera, no switch prediction (opponent name `lookahead1`)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("tera", False)
        kwargs.setdefault("predict_switch", False)
        super().__init__(*args, **kwargs)


class LookaheadBotTera(LookaheadBot):
    """Ablation: tera on, switch prediction off (opponent name `lookahead_tera`)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("predict_switch", False)
        super().__init__(*args, **kwargs)


class LookaheadBotSwitch(LookaheadBot):
    """Ablation: switch prediction on, tera off (opponent name `lookahead_switch`)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("tera", False)
        super().__init__(*args, **kwargs)


__all__ = ["our_tera_type", "LookaheadScorer", "LookaheadBot", "LookaheadBotE9", "LookaheadBotTera", "LookaheadBotSwitch", "move_list_for_actions"]
