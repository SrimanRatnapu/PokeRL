"""A one-turn lookahead bot: scores every legal action with the v2-v4 damage model.

For each of our moves and each switch it estimates what happens this turn:
who moves first (priority, speed incl. abilities/items/weather/Trick Room), how much
each side loses, the chance of a KO on either side, and end-of-turn residual damage
(sand, status, Leftovers, ... incl. a switch-in that changes the weather). The
opponent is assumed to answer with its most damaging likely move (from the
random-battle sets, over the posterior of its ability). On top of this turn's
exchange it adds a discounted value for the matchup it leaves us in, so switching
into a better matchup can beat attacking.

It never terastallizes and does not model the opponent switching. It is a training
opponent and benchmark, stronger at tactics than SimpleHeuristicsPlayer: in the
"Tyranitar's sand KOs a 2% Gyarados" position it sends Tyranitar.

Run vs the scripted bots (Showdown server on localhost:8000):
    .venv/bin/python -m pokebot.evaluate --model lookahead --opponents simple_heuristic --battles 200
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from poke_env.battle import Battle, PokemonType, SideCondition
from poke_env.player import Player

from pokebot.features_v2 import Combatant, MoveInfo, move_info
from pokebot.v4_abilities import _ACtx, _item, _known_ability, residual

W_KO = 0.6  # value of knocking a mon out, on top of the HP it had (units: a full-HP mon = 1.0)
GAMMA = 0.6  # weight of the matchup we are left in, relative to this turn's exchange

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


class LookaheadBot(Player):
    """Greedy one-turn search over our actions with the feature damage model."""

    def __init__(self, *args, epsilon: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon  # chance to play the second-best action (diversity as a training opponent)

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
        return self.create_order(pick[1])

    def score_actions(self, battle: Battle) -> List[Tuple[float, object]]:
        """[(value, Move | Pokemon)] for every legal action this turn."""
        ac = _ACtx(battle)
        v2 = ac.v2
        moves = list(battle.available_moves or [])
        switches = list(battle.available_switches or [])
        if v2.opp_c is None:  # nothing to plan against yet: strongest-looking move or first switch
            if moves:
                return [(m.base_power * m.accuracy if m.accuracy is not True else m.base_power, m) for m in moves]
            return [(0.0, s) for s in switches]
        forced = bool(battle.force_switch) or v2.me_c is None
        out: List[Tuple[float, object]] = []
        if not forced and v2.me_c is not None:
            me_ab, me_item = ac.me_ab, ac.me_item
            stay = self._matchup(ac, battle, v2.me_c, me_ab, me_item, v2.me, ac.weather)
            for mv in moves:
                mi = move_info(mv)
                gain, p_survive = self._exchange(ac, battle, v2.me_c, me_ab, me_item, mi, ac.weather)
                out.append((gain + GAMMA * p_survive * stay, mv))
        for mon in switches:
            out.append((self._switch_value(ac, battle, mon, forced), mon))
        return out

    # ------------------------------------------------------------------ #
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


__all__ = ["LookaheadBot"]
