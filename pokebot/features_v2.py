"""Observation features, version 2 (superset of v1).

`embed_battle_v2(battle)` = `features.embed_battle(battle)` (the 226 v1 entries,
byte-for-byte) followed by the v2 blocks below. float32, clipped to [-1, 1],
never raises (each block is computed independently and left as zeros if it fails).

v2 blocks (see `V2_BLOCKS` / `BLOCK_SLICES_V2` for exact offsets):
  dmg_moves   12  our 4 move slots (same order as actions 6-9): est. damage as a
                  fraction of the opponent's *current* HP (capped at 1), likely-KO,
                  priority-KO
  opp_threat  31  opponent active: max est. damage vs our active, likely KO, has a
                  super-effective move, has priority attack, priority KO, setup,
                  recovery, pivot, hazards, status move, outspeeds (sign), speed
                  log-ratio, movepool-known flag, + 18 possible tera types
  my_team     30  per our team slot (switch actions 0-5): level, speed sign vs opp
                  active, best damage out, opp active's threat in, net
  opp_team    42  per opponent team slot: revealed, hp, fainted, status, active,
                  its threat vs our active, our active's best damage vs it
  history      5  opp switched last turn, we switched last turn, effectiveness of
                  the opponent's last move on us, turns our active / their active
                  has been in (/10)

Damage model (`estimate_damage`): a simplified gen-9 formula
    ((2L/5 + 2) * BP * A/D / 50 + 2) * STAB * type * burn * weather * 0.925 * hits
using real stats for our mons (from the request) and random-battle spreads
(31 IV / 84 EV / neutral nature, level from the details line) for the opponent.
Items, most abilities, crits and field modifiers other than sun/rain are ignored;
a few ability immunities (Levitate, Flash Fire, Water/Volt Absorb, ...) are applied
when the ability is known or every random-battle ability of the species grants it.
Accuracy is kept separate (used for the KO flags).

Opponent movepools come from Showdown's gen9 random-battle sets
(`pokemon-showdown/data/random-battles/gen9/sets.json`): the union of the
movepools of the roles that are consistent with the moves revealed so far
(all roles if none are consistent), plus the revealed moves. If the species is
missing, only revealed moves are used. With 4 revealed moves, only those.
"""

from __future__ import annotations

import json
import math
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from poke_env.battle import Battle, Field, MoveCategory, Pokemon, PokemonType, SideCondition, Status, Weather
from poke_env.battle.move import Move
from poke_env.data import GenData, to_id_str

from pokebot.config import PROJECT_ROOT
from pokebot.features import N_MOVES, N_TEAM, OBS_SIZE as OBS_SIZE_V1, TYPES, embed_battle as embed_battle_v1, move_slots

SETS_PATH = PROJECT_ROOT / "pokemon-showdown" / "data" / "random-battles" / "gen9" / "sets.json"

_GEN = 9
_TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_STAT_IDX = {k: i for i, k in enumerate(_STAT_KEYS)}

# --------------------------------------------------------------------------- #
# Static tables (built once)
# --------------------------------------------------------------------------- #
_TYPE_CHART: Dict[PokemonType, Dict[PokemonType, float]] = {}  # [attack][defend] -> mult


def _type_chart() -> Dict[PokemonType, Dict[PokemonType, float]]:
    if not _TYPE_CHART:
        raw = GenData.from_gen(_GEN).type_chart  # raw[defender][attacker]
        for atk in TYPES:
            _TYPE_CHART[atk] = {}
            for dfn in TYPES:
                try:
                    _TYPE_CHART[atk][dfn] = float(raw[dfn.name][atk.name])
                except Exception:
                    _TYPE_CHART[atk][dfn] = 1.0
    return _TYPE_CHART


_ABILITY_IMMUNITY = {
    "levitate": PokemonType.GROUND,
    "eartheater": PokemonType.GROUND,
    "flashfire": PokemonType.FIRE,
    "wellbakedbody": PokemonType.FIRE,
    "waterabsorb": PokemonType.WATER,
    "stormdrain": PokemonType.WATER,
    "dryskin": PokemonType.WATER,
    "voltabsorb": PokemonType.ELECTRIC,
    "lightningrod": PokemonType.ELECTRIC,
    "motordrive": PokemonType.ELECTRIC,
    "sapsipper": PokemonType.GRASS,
}

_EXTRA_SETUP = {"bellydrum", "curse", "tidyup", "takeheart", "geomancy", "victorydance", "filletaway", "shellsmash"}
_EXTRA_RECOVERY = {"morningsun", "synthesis", "moonlight", "shoreup", "rest", "wish", "strengthsap",
                   "lunarblessing", "junglehealing", "lifedew", "painsplit"}
_HAZARD_IDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "ceaselessedge", "stoneaxe"}
_STATUS_VOLATILES = {"yawn", "confusion"}
_HALF_HP_MOVES = {"superfang", "ruination", "naturesmadness"}
_NO_ESTIMATE = {"finalgambit", "endeavor", "counter", "mirrorcoat", "metalburst", "comeuppance", "bide"}
_VARIABLE_BP = {"lowkick": 80, "grassknot": 80, "heavyslam": 80, "heatcrash": 80, "gyroball": 80,
                "electroball": 80}
_HP_SCALED_BP = {"eruption", "waterspout", "dragonenergy"}

_SUN = {Weather.SUNNYDAY, Weather.DESOLATELAND}
_RAIN = {Weather.RAINDANCE, Weather.PRIMORDIALSEA}


class MoveInfo:
    """Pre-extracted static move data (Move.entry lookups are too slow per call)."""

    __slots__ = ("id", "type", "status", "physical", "bp", "hits", "acc", "priority", "off_stat", "def_stat",
                 "target_off", "fixed", "half_hp", "hp_scaled", "freeze_dry",
                 "setup", "recovery", "pivot", "hazard", "inflicts")

    def __init__(self, move_id: str):
        e = Move(move_id, gen=_GEN).entry
        self.id = move_id
        self.type = PokemonType.from_name(e.get("type", "???"))
        cat = str(e.get("category", "Status")).upper()
        self.status = cat == "STATUS"
        self.physical = cat == "PHYSICAL"
        bp = float(e.get("basePower", 0) or 0)
        if not self.status and bp == 0:
            bp = float(_VARIABLE_BP.get(move_id, 60))
        self.bp = bp
        mh = e.get("multihit")
        if move_id in ("tripleaxel", "triplekick"):
            self.hits = 1 + 2 * 0.9 + 3 * 0.81  # escalating BP, per-hit accuracy (like poke-env)
        elif isinstance(mh, (list, tuple)) and len(mh) == 2:
            self.hits = (2 + 3) / 3 + (4 + 5) / 6 if tuple(mh) == (2, 5) else (mh[0] + mh[1]) / 2
        elif isinstance(mh, (int, float)):
            self.hits = float(mh)
        else:
            self.hits = 1.0
        acc = e.get("accuracy", True)
        self.acc = 1.0 if acc is True else float(acc) / 100.0
        self.priority = int(e.get("priority", 0) or 0)
        ovr_off = e.get("overrideOffensiveStat")
        self.off_stat = ovr_off if ovr_off in ("def", "spd") else ("atk" if self.physical else "spa")
        ovr_def = e.get("overrideDefensiveStat")
        self.def_stat = ovr_def if ovr_def in ("def", "spd") else ("def" if self.physical else "spd")
        self.target_off = e.get("overrideOffensivePokemon") == "target"
        dmg = e.get("damage")
        self.fixed = dmg if (dmg == "level" or isinstance(dmg, (int, float))) and not isinstance(dmg, bool) else None
        self.half_hp = move_id in _HALF_HP_MOVES
        if move_id in _NO_ESTIMATE:
            self.bp = 0.0
        self.hp_scaled = move_id in _HP_SCALED_BP
        self.freeze_dry = move_id == "freezedry"
        boosts = e.get("boosts") or {}
        self.setup = bool(self.status and (
            (e.get("target") == "self" and sum(v for v in boosts.values() if isinstance(v, (int, float))) > 0)
            or move_id in _EXTRA_SETUP))
        self.recovery = bool(self.status and (e.get("heal") or move_id in _EXTRA_RECOVERY))
        self.pivot = bool(e.get("selfSwitch"))
        self.hazard = move_id in _HAZARD_IDS or e.get("sideCondition") in _HAZARD_IDS
        self.inflicts = bool(self.status and (e.get("status") or e.get("volatileStatus") in _STATUS_VOLATILES))


_MOVE_CACHE: Dict[str, Optional[MoveInfo]] = {}


def move_info(move_or_id) -> Optional[MoveInfo]:
    """Cached MoveInfo for a Move / move id; None if unknown."""
    mid = move_or_id if isinstance(move_or_id, str) else getattr(move_or_id, "id", None)
    if not mid:
        return None
    try:
        return _MOVE_CACHE[mid]
    except KeyError:
        pass
    try:
        info: Optional[MoveInfo] = MoveInfo(mid)
    except Exception:
        info = None
    _MOVE_CACHE[mid] = info
    return info


# --------------------------------------------------------------------------- #
# Random-battle sets
# --------------------------------------------------------------------------- #
class _SpeciesSets:
    __slots__ = ("level", "roles", "tera", "abilities", "immune")

    def __init__(self, entry: dict):
        self.level = int(entry.get("level", 80) or 80)
        self.roles: List[Tuple[FrozenSet[str], FrozenSet[PokemonType]]] = []
        abilities = set()
        for s in entry.get("sets", []):
            moves = frozenset(to_id_str(m) for m in s.get("movepool", []))
            tera = set()
            for t in s.get("teraTypes", []) or []:
                try:
                    pt = PokemonType.from_name(t)
                    if pt in _TYPE_INDEX:
                        tera.add(pt)
                except Exception:
                    pass
            self.roles.append((moves, frozenset(tera)))
            abilities.update(to_id_str(a) for a in s.get("abilities", []) or [])
        self.abilities = frozenset(abilities)
        imm = {_ABILITY_IMMUNITY.get(a) for a in abilities}
        # Immune only if *every* possible ability grants the same immunity.
        self.immune = next(iter(imm)) if len(imm) == 1 and None not in imm else None


_SETS: Optional[Dict[str, _SpeciesSets]] = None


def random_battle_sets() -> Dict[str, _SpeciesSets]:
    global _SETS
    if _SETS is None:
        sets: Dict[str, _SpeciesSets] = {}
        try:
            raw = json.loads(SETS_PATH.read_text())
            for k, v in raw.items():
                try:
                    sets[to_id_str(k)] = _SpeciesSets(v)
                except Exception:
                    pass
        except Exception:
            pass
        _SETS = sets
    return _SETS


def _species_sets(mon: Pokemon) -> Optional[_SpeciesSets]:
    sets = random_battle_sets()
    try:
        s = sets.get(mon.species)
        if s is None:
            s = sets.get(mon.base_species)
        return s
    except Exception:
        return None


_POOL_CACHE: Dict[Tuple[str, FrozenSet[str]], Tuple[Tuple[MoveInfo, ...], FrozenSet[PokemonType], bool]] = {}


def opponent_movepool(mon: Pokemon) -> Tuple[Tuple[MoveInfo, ...], FrozenSet[PokemonType], bool]:
    """(possible moves, possible tera types, species-known) for an opponent mon."""
    revealed = frozenset(m.id for m in mon.moves.values())
    key = (mon.species, revealed)
    try:
        return _POOL_CACHE[key]
    except KeyError:
        pass
    sp = _species_sets(mon)
    tera: FrozenSet[PokemonType] = frozenset()
    if sp is not None and sp.roles:
        consistent = [r for r in sp.roles if revealed <= r[0]] or sp.roles
        ids = set(revealed)
        tset = set()
        for moves, t in consistent:
            ids |= moves
            tset |= t
        if len(revealed) >= 4:
            ids = set(revealed)
        tera = frozenset(tset)
    else:
        ids = set(revealed)
    infos = tuple(i for i in (move_info(m) for m in sorted(ids)) if i is not None)
    out = (infos, tera, sp is not None)
    if len(_POOL_CACHE) > 20000:
        _POOL_CACHE.clear()
    _POOL_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- #
# Stats and per-mon combat context
# --------------------------------------------------------------------------- #
_STAT_CACHE: Dict[Tuple[Tuple[int, ...], int], Tuple[float, ...]] = {}


def estimate_stats(base: Sequence[int], level: int) -> Tuple[float, ...]:
    """Random-battle spread: 31 IVs, 84 EVs, neutral nature. Returns (hp, atk, def, spa, spd, spe)."""
    key = (tuple(base), level)
    try:
        return _STAT_CACHE[key]
    except KeyError:
        pass
    out = []
    for i, b in enumerate(base):
        core = math.floor((2 * b + 31 + 21) * level / 100)
        out.append(float(core + level + 10) if i == 0 else float(core + 5))
    if base[0] == 1:  # Shedinja
        out[0] = 1.0
    res = tuple(out)
    _STAT_CACHE[key] = res
    return res


def _boost_mult(stage) -> float:
    try:
        s = max(-6, min(6, int(stage or 0)))
    except Exception:
        s = 0
    return (2 + s) / 2 if s >= 0 else 2 / (2 - s)


class Combatant:
    """Everything the damage formula needs about one mon, computed once per embed."""

    __slots__ = ("mon", "level", "stats", "hp_max", "hp_cur", "hp_frac", "types", "orig_types", "tera",
                 "burned", "paralyzed", "boosts", "immune", "fainted", "_eff")

    def __init__(self, mon: Pokemon, ours: bool):
        self.mon = mon
        level = int(mon.level or 0)
        sp = None
        if level <= 0 or level > 100:
            sp = _species_sets(mon)
            level = sp.level if sp is not None else 80
        self.level = level
        bs = mon.base_stats
        base = tuple(int(bs.get(k, 80) or 80) for k in _STAT_KEYS)
        est = estimate_stats(base, level)
        stats = list(est)
        if ours:
            known = mon.stats or {}
            for i, k in enumerate(_STAT_KEYS[1:], start=1):
                v = known.get(k)
                if v:
                    stats[i] = float(v)
            if mon.max_hp:
                stats[0] = float(mon.max_hp)
        self.stats = stats
        self.fainted = bool(mon.fainted)
        self.hp_max = max(stats[0], 1.0)
        if self.fainted:
            frac = 0.0
        elif ours and mon.max_hp:
            frac = (mon.current_hp or 0) / mon.max_hp
        elif mon.max_hp:
            frac = mon.current_hp_fraction
        else:
            frac = 1.0
        self.hp_frac = max(0.0, min(1.0, float(frac)))
        self.hp_cur = self.hp_max * self.hp_frac
        self.types = [t for t in mon.types if t in _TYPE_INDEX]
        self.orig_types = [t for t in (mon.base_types or []) if t in _TYPE_INDEX] or self.types
        self.tera = mon.tera_type if (mon.is_terastallized and mon.tera_type in _TYPE_INDEX) else None
        st = mon.status
        self.burned = st == Status.BRN
        self.paralyzed = st == Status.PAR
        self.boosts = mon.boosts or {}
        immune = set()
        ability = None
        try:
            ability = mon.ability
        except Exception:
            pass
        if ability:
            t = _ABILITY_IMMUNITY.get(ability)
            if t is not None:
                immune.add(t)
        elif not ours:
            sp = sp or _species_sets(mon)
            if sp is not None and sp.immune is not None:
                immune.add(sp.immune)
        try:
            if mon.item == "airballoon":
                immune.add(PokemonType.GROUND)
        except Exception:
            pass
        self.immune = immune
        self._eff: Dict[PokemonType, float] = {}

    def stat(self, key: str) -> float:
        return self.stats[_STAT_IDX[key]] * _boost_mult(self.boosts.get(key, 0))

    def effectiveness(self, mi: MoveInfo) -> float:
        t = mi.type
        if t in self.immune:
            return 0.0
        if mi.freeze_dry:
            m = 1.0
            chart = _type_chart()[t] if t in _TYPE_INDEX else None
            for dt in self.types:
                m *= 2.0 if dt == PokemonType.WATER else (chart[dt] if chart else 1.0)
            return m
        try:
            return self._eff[t]
        except KeyError:
            pass
        m = 1.0
        if t in _TYPE_INDEX:
            chart = _type_chart()[t]
            for dt in self.types:
                m *= chart[dt]
        self._eff[t] = m
        return m

    def stab(self, t: PokemonType) -> float:
        if self.tera is not None:
            if t == self.tera:
                return 2.0 if t in self.orig_types else 1.5
            return 1.5 if t in self.orig_types else 1.0
        return 1.5 if t in self.types else 1.0

    def speed(self, tailwind: bool = False) -> float:
        s = self.stat("spe")
        if self.paralyzed:
            s *= 0.5
        if tailwind:
            s *= 2.0
        return s


def estimate_damage(mi: Optional[MoveInfo], att: Combatant, dfn: Combatant, weather=None) -> float:
    """Expected damage of `mi` from att to dfn as a fraction of dfn's *current* HP (accuracy excluded)."""
    if mi is None or mi.status or dfn.fainted or dfn.hp_cur <= 0:
        return 0.0
    eff = dfn.effectiveness(mi)
    if eff == 0.0:
        return 0.0
    if mi.half_hp:
        return 0.5
    if mi.fixed is not None:
        dmg = float(att.level) if mi.fixed == "level" else float(mi.fixed)
        return dmg / dfn.hp_cur
    if mi.bp <= 0:
        return 0.0
    bp = mi.bp
    if mi.hp_scaled:
        bp = max(1.0, bp * att.hp_frac)
    a = dfn.stat("atk") if mi.target_off else att.stat(mi.off_stat)
    d = max(dfn.stat(mi.def_stat), 1.0)
    base = (2.0 * att.level / 5.0 + 2.0) * bp * a / d / 50.0 + 2.0
    mod = att.stab(mi.type) * eff * 0.925 * mi.hits
    if mi.physical and att.burned and mi.id != "facade":
        mod *= 0.5
    if weather:
        if mi.type == PokemonType.FIRE:
            if weather & _SUN:
                mod *= 1.5
            elif weather & _RAIN:
                mod *= 0.0 if Weather.PRIMORDIALSEA in weather else 0.5
        elif mi.type == PokemonType.WATER:
            if weather & _RAIN:
                mod *= 1.5
            elif weather & _SUN:
                mod *= 0.0 if Weather.DESOLATELAND in weather else 0.5
    return base * mod / dfn.hp_cur


# --------------------------------------------------------------------------- #
# Per-embed context (combatants and movepools computed once)
# --------------------------------------------------------------------------- #
class _Ctx:
    __slots__ = ("battle", "weather", "trick_room", "my_tailwind", "opp_tailwind", "me", "opp",
                 "me_c", "opp_c", "opp_pool", "opp_tera", "opp_known", "_cache")

    def __init__(self, battle: Battle):
        self.battle = battle
        self.weather = frozenset((battle.weather or {}).keys())
        self.trick_room = Field.TRICK_ROOM in (battle.fields or {})
        self.my_tailwind = SideCondition.TAILWIND in (battle.side_conditions or {})
        self.opp_tailwind = SideCondition.TAILWIND in (battle.opponent_side_conditions or {})
        self.me = battle.active_pokemon
        self.opp = battle.opponent_active_pokemon
        self._cache: Dict[int, Combatant] = {}
        # A fainted active (waiting on a forced switch) contributes no matchup features.
        self.me_c = self.combatant(self.me, True) if self.me is not None and not self.me.fainted else None
        self.opp_c = self.combatant(self.opp, False) if self.opp is not None and not self.opp.fainted else None
        if self.opp is not None:
            self.opp_pool, self.opp_tera, self.opp_known = opponent_movepool(self.opp)
        else:
            self.opp_pool, self.opp_tera, self.opp_known = (), frozenset(), False

    def combatant(self, mon: Pokemon, ours: bool) -> Combatant:
        k = id(mon)
        c = self._cache.get(k)
        if c is None:
            c = Combatant(mon, ours)
            self._cache[k] = c
        return c

    def dmg(self, mi, att: Combatant, dfn: Combatant) -> float:
        return estimate_damage(mi, att, dfn, self.weather)

    def best(self, moves, att: Combatant, dfn: Combatant) -> float:
        best = 0.0
        for mi in moves:
            d = estimate_damage(mi, att, dfn, self.weather)
            if d > best:
                best = d
        return best

    def speed_sign(self, mine: Combatant, theirs: Combatant) -> float:
        a, b = mine.speed(self.my_tailwind), theirs.speed(self.opp_tailwind)
        if self.trick_room:
            a, b = -a, -b
        return 1.0 if a > b else (-1.0 if a < b else 0.0)


def _infos(moves) -> List[MoveInfo]:
    return [i for i in (move_info(m) for m in moves) if i is not None]


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
DMG_MOVE_FEATS = 3
OPP_THREAT_FEATS = 13 + len(TYPES)  # 31
MY_TEAM_FEATS = 5
OPP_TEAM_FEATS = 7
HISTORY_FEATS = 5


def _dmg_moves_block(battle: Battle, ctx: _Ctx) -> List[float]:
    out = [0.0] * (N_MOVES * DMG_MOVE_FEATS)
    if ctx.me_c is None or ctx.opp_c is None:
        return out
    for i, m in enumerate(move_slots(battle)[:N_MOVES]):
        mi = move_info(m)
        if mi is None:
            continue
        d = ctx.dmg(mi, ctx.me_c, ctx.opp_c)
        ko = d * mi.acc >= 1.0
        out[3 * i] = min(d, 1.0)
        out[3 * i + 1] = 1.0 if ko else 0.0
        out[3 * i + 2] = 1.0 if (ko and mi.priority > 0) else 0.0
    return out


def _opp_threat_block(battle: Battle, ctx: _Ctx) -> List[float]:
    out = [0.0] * OPP_THREAT_FEATS
    opp_c, me_c, pool = ctx.opp_c, ctx.me_c, ctx.opp_pool
    if opp_c is None:
        return out
    if me_c is not None:
        best = ko = se = prio = prio_ko = 0.0
        for mi in pool:
            if mi.status:
                continue
            d = ctx.dmg(mi, opp_c, me_c)
            if d > best:
                best = d
            k = d * mi.acc >= 1.0
            if k:
                ko = 1.0
            if me_c.effectiveness(mi) > 1.0:
                se = 1.0
            if mi.priority > 0 and d > 0:
                prio = 1.0
                if k:
                    prio_ko = 1.0
        out[0], out[1], out[2], out[3], out[4] = min(best, 1.0), ko, se, prio, prio_ko
        out[10] = -ctx.speed_sign(me_c, opp_c)  # +1: opponent outspeeds
        a, b = me_c.speed(ctx.my_tailwind), opp_c.speed(ctx.opp_tailwind)
        ratio = math.log2(max(b, 1.0) / max(a, 1.0))
        out[11] = max(-1.0, min(1.0, (-ratio if ctx.trick_room else ratio) / 2.0))
    out[5] = 1.0 if any(mi.setup for mi in pool) else 0.0
    out[6] = 1.0 if any(mi.recovery for mi in pool) else 0.0
    out[7] = 1.0 if any(mi.pivot for mi in pool) else 0.0
    out[8] = 1.0 if any(mi.hazard for mi in pool) else 0.0
    out[9] = 1.0 if any(mi.inflicts for mi in pool) else 0.0
    out[12] = 1.0 if ctx.opp_known else 0.0
    if not ctx.opp.is_terastallized and not battle.opponent_used_tera:
        for t in ctx.opp_tera:
            idx = _TYPE_INDEX.get(t)
            if idx is not None:
                out[13 + idx] = 1.0
    return out


def _my_team_block(battle: Battle, ctx: _Ctx) -> List[float]:
    out = [0.0] * (N_TEAM * MY_TEAM_FEATS)
    opp_c = ctx.opp_c
    for i, mon in enumerate(list(battle.team.values())[:N_TEAM]):
        try:
            c = ctx.combatant(mon, True)
            j = MY_TEAM_FEATS * i
            out[j] = (c.level or 0) / 100.0
            if c.fainted or opp_c is None:
                continue
            out[j + 1] = ctx.speed_sign(c, opp_c)
            best_out = min(ctx.best(_infos(mon.moves.values()), c, opp_c), 1.0)
            threat_in = min(ctx.best(ctx.opp_pool, opp_c, c), 1.0)
            out[j + 2] = best_out
            out[j + 3] = threat_in
            out[j + 4] = best_out - threat_in
        except Exception:
            out[MY_TEAM_FEATS * i: MY_TEAM_FEATS * (i + 1)] = [0.0] * MY_TEAM_FEATS
    return out


def _opp_team_block(battle: Battle, ctx: _Ctx) -> List[float]:
    out = [0.0] * (N_TEAM * OPP_TEAM_FEATS)
    me_c = ctx.me_c
    my_moves = _infos(ctx.me.moves.values()) if ctx.me is not None else []
    for i, mon in enumerate(list(battle.opponent_team.values())[:N_TEAM]):
        j = OPP_TEAM_FEATS * i
        try:
            c = ctx.combatant(mon, False)
            out[j] = 1.0
            out[j + 1] = c.hp_frac
            out[j + 2] = 1.0 if c.fainted else 0.0
            out[j + 3] = 1.0 if (mon.status is not None and not c.fainted) else 0.0
            out[j + 4] = 1.0 if mon.active else 0.0
            if c.fainted or me_c is None:
                continue
            pool = ctx.opp_pool if mon is ctx.opp else opponent_movepool(mon)[0]
            out[j + 5] = min(ctx.best(pool, c, me_c), 1.0)
            out[j + 6] = min(ctx.best(my_moves, me_c, c), 1.0)
        except Exception:
            out[j: j + OPP_TEAM_FEATS] = [0.0] * OPP_TEAM_FEATS
    return out


_EFFECT_VALUE = {"-supereffective": 1.0, "-resisted": -0.5, "-immune": -1.0}
_HISTORY_MAX_TURNS = 10
_HISTORY_MAX_MSGS = 1500


def _history_block(battle: Battle, ctx: _Ctx) -> List[float]:
    """Backward scan of the protocol log (battle._replay_data), bounded in length."""
    out = [0.0] * HISTORY_FEATS
    data = getattr(battle, "_replay_data", None)
    me_role = battle.player_role
    if not data or me_role not in ("p1", "p2"):
        return out
    opp_role = "p2" if me_role == "p1" else "p1"
    markers = 0  # `turn` markers passed (scanning backwards)
    partial = {"me": False, "opp": False, "act": False}  # after the latest turn marker
    prev = {"me": False, "opp": False}  # between the last two turn markers
    prev_complete = False
    my_in = opp_in = None
    last_eff = None
    pending_eff = None
    stop = max(0, len(data) - _HISTORY_MAX_MSGS)
    for idx in range(len(data) - 1, stop - 1, -1):
        ev = data[idx]
        if len(ev) < 2:
            continue
        kind = ev[1]
        if kind == "turn":
            markers += 1
            if markers == 2:
                prev_complete = True
            if markers > _HISTORY_MAX_TURNS:
                break
            continue
        if len(ev) < 3:
            continue
        who = ev[2][:2]
        if kind in ("switch", "drag"):
            side = "me" if who == me_role else ("opp" if who == opp_role else None)
            if side is None:
                continue
            if markers == 0:
                partial[side] = True
                partial["act"] = True
            elif markers == 1:
                prev[side] = True
            if side == "me" and my_in is None:
                my_in = markers
            elif side == "opp" and opp_in is None:
                opp_in = markers
        elif kind in _EFFECT_VALUE:
            if who == me_role:
                pending_eff = _EFFECT_VALUE[kind]
        elif kind == "move":
            if markers == 0:
                partial["act"] = True
            if who == opp_role and last_eff is None:
                # SE / resisted / immune message (on our side) right after it; else neutral / status.
                last_eff = pending_eff if pending_eff is not None else 0.0
            pending_eff = None
        if markers >= 2 and my_in is not None and opp_in is not None and last_eff is not None:
            break
    if partial["act"]:
        out[0] = 1.0 if partial["opp"] else 0.0
        out[1] = 1.0 if partial["me"] else 0.0
    elif prev_complete:
        out[0] = 1.0 if prev["opp"] else 0.0
        out[1] = 1.0 if prev["me"] else 0.0
    out[2] = last_eff or 0.0
    cap = float(_HISTORY_MAX_TURNS)
    out[3] = min(my_in if my_in is not None else cap, cap) / cap if ctx.me is not None else 0.0
    out[4] = min(opp_in if opp_in is not None else cap, cap) / cap if ctx.opp is not None else 0.0
    return out


V2_BLOCKS: Sequence[Tuple[str, int, Callable[[Battle, _Ctx], List[float]]]] = (
    ("dmg_moves", N_MOVES * DMG_MOVE_FEATS, _dmg_moves_block),  # 12
    ("opp_threat", OPP_THREAT_FEATS, _opp_threat_block),  # 31
    ("my_team", N_TEAM * MY_TEAM_FEATS, _my_team_block),  # 30
    ("opp_team", N_TEAM * OPP_TEAM_FEATS, _opp_team_block),  # 42
    ("history", HISTORY_FEATS, _history_block),  # 5
)

OBS_SIZE_V2: int = OBS_SIZE_V1 + sum(size for _, size, _ in V2_BLOCKS)

BLOCK_SLICES_V2 = {"v1": slice(0, OBS_SIZE_V1)}
_start = OBS_SIZE_V1
for _name, _size, _ in V2_BLOCKS:
    BLOCK_SLICES_V2[_name] = slice(_start, _start + _size)
    _start += _size
del _start, _name, _size


def embed_battle_v2(battle: Battle) -> np.ndarray:
    """Return a float32 vector of shape (OBS_SIZE_V2,) in [-1, 1]. Never raises."""
    obs = np.zeros(OBS_SIZE_V2, dtype=np.float32)
    if battle is None:
        return obs
    try:
        obs[:OBS_SIZE_V1] = embed_battle_v1(battle)
    except Exception:
        pass
    try:
        ctx = _Ctx(battle)
    except Exception:
        ctx = None
    if ctx is not None:
        for name, size, fn in V2_BLOCKS:
            sl = BLOCK_SLICES_V2[name]
            try:
                vals = fn(battle, ctx)
                if len(vals) != size:
                    continue
                obs[sl] = vals
            except Exception:
                obs[sl] = 0.0
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(obs, -1.0, 1.0, out=obs)
    return obs


__all__ = ["OBS_SIZE_V2", "embed_battle_v2", "BLOCK_SLICES_V2", "V2_BLOCKS", "estimate_damage", "Combatant",
           "MoveInfo", "move_info", "opponent_movepool", "estimate_stats", "random_battle_sets"]
