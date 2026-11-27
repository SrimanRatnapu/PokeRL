"""Observation features, version 1 ("v1", 226 dims) for the gen9randombattle RL agent.

Frozen: v1 checkpoints depend on this exact layout. New features go in
`pokebot/features_v2.py`; the version registry is `pokebot/feature_sets.py`.

`embed_battle(battle)` turns a poke-env `Battle` into a flat float32 vector of
length `OBS_SIZE`, every entry clipped to [-1, 1].

The vector is built from fixed-size blocks (see `_BLOCKS`). Each block is
computed independently and, if anything inside it raises on odd/missing data,
that block is left as zeros, so `embed_battle` never raises and the layout is
always the same.

Index conventions match `SinglesEnv.get_action_mask` / `action_to_order`:
  * move slot i  <-> actions 6+i (and 22+i for tera): the first 4 entries of
    `battle.active_pokemon.moves`, except when the only available move is not a
    known move (e.g. Struggle), in which case slot 0 is that move.
  * team slot i  <-> action i (switch): `list(battle.team.values())[i]`.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import numpy as np
from poke_env.battle import Battle, Field, MoveCategory, Pokemon, PokemonType, SideCondition, Status, Weather
from poke_env.battle.move import Move

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #
TYPES: List[PokemonType] = [
    t for t in PokemonType if t not in (PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR)
]  # the 18 real types
_TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}

STATUSES: List[Status] = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
BASE_STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
CATEGORIES = [MoveCategory.PHYSICAL, MoveCategory.SPECIAL, MoveCategory.STATUS]

WEATHER_GROUPS = [
    {Weather.RAINDANCE, Weather.PRIMORDIALSEA},
    {Weather.SUNNYDAY, Weather.DESOLATELAND},
    {Weather.SANDSTORM},
    {Weather.SNOWSCAPE, Weather.HAIL},
    {Weather.DELTASTREAM},
]
FIELDS = [
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
    Field.TRICK_ROOM,
    Field.GRAVITY,
]
# (condition, max layers) -- value is layers/max for stackables, else presence.
SIDE_CONDITIONS = [
    (SideCondition.STEALTH_ROCK, 1),
    (SideCondition.SPIKES, 3),
    (SideCondition.TOXIC_SPIKES, 2),
    (SideCondition.STICKY_WEB, 1),
    (SideCondition.REFLECT, 1),
    (SideCondition.LIGHT_SCREEN, 1),
    (SideCondition.AURORA_VEIL, 1),
    (SideCondition.TAILWIND, 1),
    (SideCondition.SAFEGUARD, 1),
]
_STACKABLE = {SideCondition.SPIKES, SideCondition.TOXIC_SPIKES}

N_MOVES = 4
N_TEAM = 6
MOVE_FEATS = 15
MON_FEATS = 1 + 1 + len(STATUSES) + len(BOOST_KEYS) + len(TYPES) + 1 + len(BASE_STAT_KEYS)  # 40
TEAM_SLOT_FEATS = 7


# --------------------------------------------------------------------------- #
# Small safe helpers
# --------------------------------------------------------------------------- #
def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    return max(lo, min(hi, x))


def _hp_fraction(mon: Optional[Pokemon]) -> float:
    if mon is None:
        return 0.0
    try:
        if mon.fainted:
            return 0.0
        if mon.max_hp:
            return _clip(mon.current_hp_fraction, 0.0, 1.0)
        # Unknown max hp (e.g. not yet seen): assume full.
        return 1.0
    except Exception:
        return 0.0


def _base_stat(mon: Optional[Pokemon], key: str, default: float = 80.0) -> float:
    if mon is None:
        return default
    try:
        v = mon.base_stats.get(key)
        return float(v) if v else default
    except Exception:
        return default


def _types(mon: Optional[Pokemon]) -> List[PokemonType]:
    if mon is None:
        return []
    try:
        return [t for t in mon.types if t is not None]
    except Exception:
        return []


def _damage_multiplier(defender: Optional[Pokemon], type_or_move) -> float:
    """Type effectiveness; 1.0 when unknown."""
    if defender is None or type_or_move is None:
        return 1.0
    try:
        return float(defender.damage_multiplier(type_or_move))
    except Exception:
        return 1.0


def _boost_multiplier(stage: int) -> float:
    stage = max(-6, min(6, int(stage or 0)))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def _effective_speed(mon: Optional[Pokemon]) -> float:
    if mon is None:
        return 0.0
    spe = _base_stat(mon, "spe")
    try:
        spe *= _boost_multiplier(mon.boosts.get("spe", 0))
        if mon.status == Status.PAR:
            spe *= 0.5
    except Exception:
        pass
    return spe


def move_slots(battle: Battle) -> List[Move]:
    """The moves that actions 6..9 refer to, in order (same logic as SinglesEnv.action_to_order)."""
    me = battle.active_pokemon
    if me is None:
        return []
    known = list(me.moves.values())[:N_MOVES]
    avail = list(battle.available_moves or [])
    known_ids = [m.id for m in known]
    if len(avail) == 1 and avail[0].id not in known_ids:
        return avail
    return known


def matchup_score(mon: Optional[Pokemon], opponent: Optional[Pokemon]) -> float:
    """Same idea as HeuristicBot.matchup_score: offense - defense + 0.5*hp (unscaled)."""
    if mon is None or opponent is None:
        return 0.0
    my_types, opp_types = _types(mon), _types(opponent)
    offense = max((_damage_multiplier(opponent, t) for t in my_types), default=1.0)
    defense = max((_damage_multiplier(mon, t) for t in opp_types), default=1.0)
    return offense - defense + 0.5 * _hp_fraction(mon)


def estimated_damage(move: Move, attacker: Optional[Pokemon], defender: Optional[Pokemon]) -> float:
    """HeuristicBot.move_score: rough relative damage (0 for status moves)."""
    if move.category == MoveCategory.STATUS:
        return 0.0
    power = float(move.base_power or 0) * float(move.expected_hits or 1)
    stab = 1.5 if move.type in _types(attacker) else 1.0
    eff = _damage_multiplier(defender, move)
    acc = float(move.accuracy or 0)
    if move.category == MoveCategory.PHYSICAL:
        ratio = _base_stat(attacker, "atk") / max(_base_stat(defender, "def"), 1.0)
    else:
        ratio = _base_stat(attacker, "spa") / max(_base_stat(defender, "spd"), 1.0)
    return power * stab * eff * acc * ratio


# --------------------------------------------------------------------------- #
# Blocks: each returns exactly `size` floats
# --------------------------------------------------------------------------- #
def _move_block(battle: Battle) -> List[float]:
    out: List[float] = []
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    slots = move_slots(battle)
    avail_ids = set() if getattr(battle, "_wait", False) else {m.id for m in (battle.available_moves or [])}
    special = len(slots) == 1 and me is not None and slots[0].id not in [m.id for m in list(me.moves.values())[:N_MOVES]]
    my_types = _types(me)
    tera_type = None
    try:
        tera_type = me.tera_type if me is not None else None
    except Exception:
        pass
    for i in range(N_MOVES):
        f = [0.0] * MOVE_FEATS
        if i < len(slots):
            m = slots[i]
            try:
                f[0] = 1.0 if (m.id in avail_ids or (special and i == 0 and avail_ids)) else 0.0  # available now
                f[1] = 1.0  # slot known
                bp = float(m.base_power or 0) * float(m.expected_hits or 1)
                f[2] = _clip(bp / 150.0, 0.0, 1.0)
                mult = _damage_multiplier(opp, m) if m.category != MoveCategory.STATUS else 1.0
                f[3] = -1.0 if mult == 0 else _clip(math.log2(mult) / 2.0)  # 0.25->-1, 4->+1
                f[4] = 1.0 if mult == 0 else 0.0  # immune
                f[5] = 1.0 if m.type in my_types else 0.0  # STAB
                f[6] = 1.0 if (tera_type is not None and m.type == tera_type and m.category != MoveCategory.STATUS) else 0.0
                f[7] = _clip(m.accuracy, 0.0, 1.0)
                for j, c in enumerate(CATEGORIES):
                    f[8 + j] = 1.0 if m.category == c else 0.0
                f[11] = _clip((m.priority or 0) / 5.0)
                max_pp = m.max_pp or 0
                f[12] = _clip(m.current_pp / max_pp, 0.0, 1.0) if max_pp else 1.0
                f[13] = math.tanh(estimated_damage(m, me, opp) / 150.0)
                # Attack with 0 listed bp: fixed/variable damage (Seismic Toss, Night Shade, Super Fang, ...)
                f[14] = 1.0 if (m.category != MoveCategory.STATUS and not (m.base_power or 0)) else 0.0
            except Exception:
                f = [0.0] * MOVE_FEATS
        out.extend(f)
    return out


def _mon_features(mon: Optional[Pokemon]) -> List[float]:
    f = [0.0] * MON_FEATS
    if mon is None:
        return f
    k = 0
    f[k] = 1.0  # present
    k += 1
    f[k] = _hp_fraction(mon)
    k += 1
    try:
        st = mon.status
    except Exception:
        st = None
    for s in STATUSES:
        f[k] = 1.0 if st == s else 0.0
        k += 1
    try:
        boosts = mon.boosts or {}
    except Exception:
        boosts = {}
    for b in BOOST_KEYS:
        f[k] = _clip((boosts.get(b, 0) or 0) / 6.0)
        k += 1
    for t in _types(mon):
        idx = _TYPE_INDEX.get(t)
        if idx is not None:
            f[k + idx] = 1.0
    k += len(TYPES)
    try:
        f[k] = 1.0 if mon.is_terastallized else 0.0
    except Exception:
        pass
    k += 1
    for s in BASE_STAT_KEYS:
        f[k] = _clip(_base_stat(mon, s, 0.0) / 200.0, 0.0, 1.0)
        k += 1
    return f


def _actives_block(battle: Battle) -> List[float]:
    return _mon_features(battle.active_pokemon) + _mon_features(battle.opponent_active_pokemon)


def _speed_block(battle: Battle) -> List[float]:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    if me is None or opp is None:
        return [0.0, 0.0]
    a, b = _effective_speed(me), _effective_speed(opp)
    trick_room = Field.TRICK_ROOM in (battle.fields or {})
    diff = (a - b) / max(a + b, 1.0)
    if trick_room:
        diff = -diff
    faster = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
    return [_clip(diff * 2.0), faster]


def _team_block(battle: Battle) -> List[float]:
    out: List[float] = []
    opp = battle.opponent_active_pokemon
    team = list(battle.team.values())
    switch_species = {p.base_species for p in (battle.available_switches or [])}
    trapped = bool(battle.trapped) or bool(getattr(battle, "_wait", False))
    for i in range(N_TEAM):
        f = [0.0] * TEAM_SLOT_FEATS
        if i < len(team):
            mon = team[i]
            try:
                f[0] = 1.0  # present
                f[1] = _hp_fraction(mon)
                f[2] = 1.0 if mon.fainted else 0.0
                f[3] = 1.0 if mon.active else 0.0
                f[4] = 1.0 if (not trapped and mon.base_species in switch_species) else 0.0
                f[5] = 1.0 if (mon.status is not None and not mon.fainted) else 0.0
                f[6] = _clip(matchup_score(mon, opp) / 4.0) if not mon.fainted else -1.0
            except Exception:
                f = [0.0] * TEAM_SLOT_FEATS
        out.extend(f)
    return out


def _team_summary_block(battle: Battle) -> List[float]:
    opp_team = list(battle.opponent_team.values())
    revealed = len(opp_team)
    opp_fainted = sum(1 for m in opp_team if m.fainted)
    opp_hp = sum(_hp_fraction(m) for m in opp_team)
    unrevealed = max(0, N_TEAM - revealed)
    my_team = list(battle.team.values())
    my_fainted = sum(1 for m in my_team if m.fainted)
    my_hp = sum(_hp_fraction(m) for m in my_team)
    return [
        _clip(revealed / N_TEAM, 0.0, 1.0),
        _clip(opp_fainted / N_TEAM, 0.0, 1.0),
        _clip(opp_hp / N_TEAM, 0.0, 1.0),
        _clip((opp_hp + unrevealed) / N_TEAM, 0.0, 1.0),  # estimated remaining (unseen = full)
        _clip(my_fainted / N_TEAM, 0.0, 1.0),
        _clip(my_hp / N_TEAM, 0.0, 1.0),
    ]


def _field_block(battle: Battle) -> List[float]:
    weather = set((battle.weather or {}).keys())
    fields = set((battle.fields or {}).keys())
    out = [1.0 if weather & g else 0.0 for g in WEATHER_GROUPS]
    out += [1.0 if f in fields else 0.0 for f in FIELDS]
    return out


def _side_block(conds) -> List[float]:
    conds = conds or {}
    out = []
    for c, max_layers in SIDE_CONDITIONS:
        if c not in conds:
            out.append(0.0)
        elif c in _STACKABLE:
            out.append(_clip((conds.get(c) or 0) / max_layers, 0.0, 1.0))
        else:
            out.append(1.0)
    return out


def _sides_block(battle: Battle) -> List[float]:
    return _side_block(battle.side_conditions) + _side_block(battle.opponent_side_conditions)


def _misc_block(battle: Battle) -> List[float]:
    me = battle.active_pokemon
    can_tera = bool(battle.can_tera)
    tera_stab_gain = 0.0
    if can_tera and me is not None and me.tera_type is not None:
        # Tera type would add a new STAB type for at least one known move.
        my_types = _types(me)
        if me.tera_type not in my_types and any(
            m.type == me.tera_type and m.category != MoveCategory.STATUS for m in move_slots(battle)
        ):
            tera_stab_gain = 1.0
    return [
        1.0 if can_tera else 0.0,
        tera_stab_gain,
        1.0 if battle.opponent_used_tera else 0.0,
        1.0 if any(bool(x) for x in (battle.force_switch if isinstance(battle.force_switch, list) else [battle.force_switch])) else 0.0,
        1.0 if battle.trapped else 0.0,
        _clip((battle.turn or 0) / 100.0, 0.0, 1.0),
        1.0 if getattr(battle, "_wait", False) else 0.0,  # waiting: only the no-op action 0 is legal
    ]


_BLOCKS: Sequence[tuple[str, int, Callable[[Battle], List[float]]]] = (
    ("moves", N_MOVES * MOVE_FEATS, _move_block),  # 4 x 15 = 60
    ("actives", 2 * MON_FEATS, _actives_block),  # 80
    ("speed", 2, _speed_block),
    ("team", N_TEAM * TEAM_SLOT_FEATS, _team_block),  # 42
    ("team_summary", 6, _team_summary_block),
    ("field", len(WEATHER_GROUPS) + len(FIELDS), _field_block),  # 11
    ("sides", 2 * len(SIDE_CONDITIONS), _sides_block),  # 18
    ("misc", 7, _misc_block),
)

OBS_SIZE: int = sum(size for _, size, _ in _BLOCKS)

# Slices into the observation vector, handy for debugging / tests.
BLOCK_SLICES = {}
_start = 0
for _name, _size, _ in _BLOCKS:
    BLOCK_SLICES[_name] = slice(_start, _start + _size)
    _start += _size
del _start, _name, _size


def embed_battle(battle: Battle) -> np.ndarray:
    """Return a float32 vector of shape (OBS_SIZE,) in [-1, 1]. Never raises."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    if battle is None:
        return obs
    for name, size, fn in _BLOCKS:
        sl = BLOCK_SLICES[name]
        try:
            vals = fn(battle)
            if len(vals) != size:  # defensive: never corrupt the layout
                continue
            obs[sl] = vals
        except Exception:
            obs[sl] = 0.0
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(obs, -1.0, 1.0, out=obs)
    return obs


# Explicit version-1 names (OBS_SIZE / embed_battle stay as v1 for backward compatibility).
OBS_SIZE_V1 = OBS_SIZE
embed_battle_v1 = embed_battle

__all__ = ["OBS_SIZE", "embed_battle", "OBS_SIZE_V1", "embed_battle_v1", "BLOCK_SLICES", "move_slots",
           "matchup_score", "estimated_damage"]
