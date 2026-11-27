"""Ability feature block (appended to the v3 vector in v4).

    abilities_block(battle) -> list[float]   (ABILITIES_BLOCK_SIZE = 117)

Returns exactly its size, values in [-1, 1], never raises (each part is wrapped in try/except
and left as zeros). Pure function of the poke-env Battle.

Motivation (a real game, 2026-11-24): our Eiscue fainted with the opponent's Gyarados on ~2% HP
and sand *not* up. Sending Tyranitar (Sand Stream) would have KO'd Gyarados at the end of the
turn no matter what it did; the bot sent Espeon (56% vs 41% Tyranitar) and lost. Nothing in
v1-v3 said "this switch-in sets a weather that finishes them off". The switch-in part of this
block (per team slot, aligned with switch actions 0-5) is built for exactly that.

What poke-env tells us (poke_env/battle/pokemon.py):
  * our mons: `Pokemon.ability` comes from the request, so it is always known.
  * opponent mons: known once revealed (`-ability`, `[from] ability: X`, weather on switch-in,
    ...) or when the species has a single possible ability. Otherwise we use a posterior: the
    random-battle roles (pokemon-showdown/data/random-battles/gen9/sets.json) consistent with
    the revealed moves, uniform over roles, uniform over each role's abilities.

Layout (ABILITIES_BLOCK_SIZE = 117):
  [  0: 26] our active's ability, as an effect vector over ABILITY_EFFECTS (multi-hot)
  [ 26: 52] opponent active's ability effect vector, *expected* under its ability posterior
  [ 52]     opponent's ability is known (revealed or single-option)
  [ 53]     opponent ability uncertainty: (#possible abilities - 1) / 3
  [ 54: 66] our 4 move slots (actions 6-9) x 3: P(move nullified by their ability),
            ability-aware expected damage (fraction of their current HP, capped at 1),
            ability-aware P(KO) (Sturdy / Multiscale / Disguise / Ice Face included)
  [ 66]     opponent's ability-aware max expected damage vs our active (capped at 1)
  [ 67]     ... P(it KOs us)
  [ 68]     fraction of the opponent's likely damaging moves our active's ability nullifies
  [ 69]     our active's end-of-turn residual (heal +, damage -; x4, fraction of max HP)
  [ 70]     our active dies to end-of-turn residual (sand, poison, burn, seeds, Life Orb not incl.)
  [ 71]     opponent active's end-of-turn residual (x4)
  [ 72]     opponent active dies to end-of-turn residual
  [ 73]     ability-aware speed: 2 * P(we outspeed) - 1 (weather / terrain speed abilities, Quick
            Feet, Protosynthesis / Quark Drive, Choice Scarf when known, Trick Room)
  [ 74]     Regenerator heal we would get by switching out now (fraction of max HP x3)
  [ 75:117] our 6 team slots (switch actions 0-5) x 7, as if that mon switched in now:
            0 opponent's end-of-turn damage under the weather after the switch (fraction of its
              current HP, capped at 1). Sand Stream / Drought / Drizzle / Snow Warning change it
            1 ... that end-of-turn damage KOs the opponent's active
            2 Intimidate value: + if the opponent relies on physical moves, - if it has Defiant /
              Competitive / Guard Dog / Contrary (expected under its posterior)
            3 fraction of the opponent's likely damaging moves this mon's ability nullifies
            4 ability-aware max expected damage the opponent's active deals to this mon
            5 ability-aware speed: 2 * P(this mon outspeeds) - 1, under the post-switch weather
            6 this mon changes the weather on entry
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from poke_env.battle import Battle, Effect, Field, PokemonType, Status, Weather
from poke_env.battle.move import Move
from poke_env.data import to_id_str

from pokebot.features import N_MOVES, N_TEAM, move_slots
from pokebot.features_v2 import (
    _ABILITY_IMMUNITY,
    Combatant,
    MoveInfo,
    _boost_mult,
    _Ctx,
    move_info,
)
from pokebot.v3_items import UNKNOWN_ITEM, _raw_sets, estimate_damage_items, item_speed

_GEN = 9

# --------------------------------------------------------------------------- #
# What each ability does, as a fixed set of effect categories
# --------------------------------------------------------------------------- #
ABILITY_EFFECTS: Tuple[str, ...] = (
    "sun", "rain", "sand", "snow", "terrain", "weather_negate", "intimidate", "entry_boost",
    "atk_double", "offense", "defense", "survive_hit", "immunity", "weather_speed", "snowball",
    "priority", "drop_guard", "unaware", "magic_guard", "regenerator", "status_guard",
    "status_boost", "contact_punish", "mold_breaker", "trap", "other",
)
N_EFFECTS = len(ABILITY_EFFECTS)  # 26
_E = {k: i for i, k in enumerate(ABILITY_EFFECTS)}


def _t(*names: str) -> Tuple[str, ...]:
    return names


ABILITY_TABLE: Dict[str, Tuple[str, ...]] = {
    # weather / terrain setters and negation
    "drought": _t("sun"), "orichalcumpulse": _t("sun", "offense"), "desolateland": _t("sun"),
    "drizzle": _t("rain"), "primordialsea": _t("rain"), "sandstream": _t("sand"), "sandspit": _t("sand"),
    "snowwarning": _t("snow"),
    "electricsurge": _t("terrain"), "psychicsurge": _t("terrain"), "grassysurge": _t("terrain"),
    "mistysurge": _t("terrain"), "hadronengine": _t("terrain", "offense"), "seedsower": _t("terrain"),
    "cloudnine": _t("weather_negate"), "airlock": _t("weather_negate"),
    # entry effects
    "intimidate": _t("intimidate"), "intrepidsword": _t("entry_boost"), "dauntlessshield": _t("entry_boost"),
    "download": _t("entry_boost"), "embodyaspectcornerstone": _t("entry_boost"),
    "embodyaspecthearthflame": _t("entry_boost"), "embodyaspectwellspring": _t("entry_boost"),
    "embodyaspectteal": _t("entry_boost"), "supersweetsyrup": _t("entry_boost"),
    "frisk": _t("other"), "trace": _t("other"), "pressure": _t("other"), "neutralizinggas": _t("other"),
    # offense
    "hugepower": _t("atk_double"), "purepower": _t("atk_double"),
    "adaptability": _t("offense"), "technician": _t("offense"), "sheerforce": _t("offense"),
    "toughclaws": _t("offense"), "ironfist": _t("offense"), "strongjaw": _t("offense"),
    "sharpness": _t("offense"), "megalauncher": _t("offense"), "transistor": _t("offense"),
    "dragonsmaw": _t("offense"), "steelworker": _t("offense"), "steelyspirit": _t("offense"),
    "rockypayload": _t("offense"), "gorillatactics": _t("offense"), "hustle": _t("offense"),
    "solarpower": _t("offense"), "sandforce": _t("offense"), "supremeoverlord": _t("offense", "snowball"),
    "protosynthesis": _t("offense", "weather_speed"), "quarkdrive": _t("offense", "weather_speed"),
    "aerilate": _t("offense"), "pixilate": _t("offense"), "refrigerate": _t("offense"),
    "galvanize": _t("offense"), "tintedlens": _t("offense"), "protean": _t("offense"), "libero": _t("offense"),
    "neuroforce": _t("offense"), "analytic": _t("offense"), "stakeout": _t("offense"), "sniper": _t("offense"),
    "superluck": _t("offense"), "skilllink": _t("offense"), "compoundeyes": _t("offense"),
    "noguard": _t("offense"), "contrary": _t("snowball"), "simple": _t("snowball"),
    "punkrock": _t("offense", "defense"), "waterbubble": _t("offense", "defense"),
    "mindseye": _t("offense"), "scrappy": _t("offense", "drop_guard"), "toxicdebris": _t("other"),
    "parentalbond": _t("offense"),
    # defense
    "multiscale": _t("defense", "survive_hit"), "shadowshield": _t("defense", "survive_hit"),
    "furcoat": _t("defense"), "fluffy": _t("defense"), "icescales": _t("defense"), "thickfat": _t("defense"),
    "filter": _t("defense"), "solidrock": _t("defense"), "prismarmor": _t("defense"),
    "heatproof": _t("defense"), "purifyingsalt": _t("defense", "status_guard"), "marvelscale": _t("defense"),
    "terashell": _t("defense", "survive_hit"), "battlearmor": _t("defense"), "shellarmor": _t("defense"),
    "sturdy": _t("survive_hit"), "disguise": _t("survive_hit"), "iceface": _t("survive_hit"),
    "wonderguard": _t("immunity"), "stamina": _t("defense"),
    # immunities
    "levitate": _t("immunity"), "eartheater": _t("immunity"), "flashfire": _t("immunity"),
    "wellbakedbody": _t("immunity"), "waterabsorb": _t("immunity"), "stormdrain": _t("immunity"),
    "dryskin": _t("immunity"), "voltabsorb": _t("immunity"), "lightningrod": _t("immunity"),
    "motordrive": _t("immunity"), "sapsipper": _t("immunity"), "windrider": _t("immunity"),
    "soundproof": _t("immunity"), "bulletproof": _t("immunity"), "overcoat": _t("immunity"),
    "goodasgold": _t("immunity", "status_guard"), "magicbounce": _t("status_guard"),
    "armortail": _t("immunity"), "queenlymajesty": _t("immunity"), "dazzling": _t("immunity"),
    # speed / snowball / priority
    "chlorophyll": _t("weather_speed"), "swiftswim": _t("weather_speed"), "sandrush": _t("weather_speed"),
    "slushrush": _t("weather_speed"), "surgesurfer": _t("weather_speed"), "quickfeet": _t("weather_speed"),
    "unburden": _t("weather_speed"), "speedboost": _t("snowball"), "moxie": _t("snowball"),
    "beastboost": _t("snowball"), "chillingneigh": _t("snowball"), "grimneigh": _t("snowball"),
    "asoneglastrier": _t("snowball"), "asonespectrier": _t("snowball"), "soulheart": _t("snowball"),
    "berserk": _t("snowball"), "angershell": _t("snowball"), "weakarmor": _t("snowball"),
    "justified": _t("snowball"), "steamengine": _t("snowball"), "windpower": _t("snowball"),
    "electromorphosis": _t("snowball"), "opportunist": _t("snowball"),
    "prankster": _t("priority"), "galewings": _t("priority"), "triage": _t("priority"),
    "quickdraw": _t("priority"),
    # stat-drop guards / punishers, unaware
    "defiant": _t("drop_guard"), "competitive": _t("drop_guard"), "mirrorarmor": _t("drop_guard"),
    "clearbody": _t("drop_guard"), "whitesmoke": _t("drop_guard"), "fullmetalbody": _t("drop_guard"),
    "hypercutter": _t("drop_guard"), "guarddog": _t("drop_guard"), "innerfocus": _t("drop_guard"),
    "owntempo": _t("drop_guard", "status_guard"), "oblivious": _t("drop_guard"),
    "unaware": _t("unaware"),
    # sustain
    "magicguard": _t("magic_guard"), "regenerator": _t("regenerator"), "naturalcure": _t("status_guard"),
    "shedskin": _t("status_guard"), "poisonheal": _t("status_boost"), "raindish": _t("other"),
    "icebody": _t("other"), "harvest": _t("other"), "cheekpouch": _t("other"), "ripen": _t("other"),
    # status
    "insomnia": _t("status_guard"), "vitalspirit": _t("status_guard"), "comatose": _t("status_guard"),
    "limber": _t("status_guard"), "immunity": _t("status_guard"), "waterveil": _t("status_guard"),
    "leafguard": _t("status_guard"), "pastelveil": _t("status_guard"), "sweetveil": _t("status_guard"),
    "thermalexchange": _t("status_guard"), "magmaarmor": _t("status_guard"), "shieldsdown": _t("status_guard"),
    "guts": _t("status_boost"), "toxicboost": _t("status_boost"), "flareboost": _t("status_boost"),
    "synchronize": _t("status_boost"),
    "roughskin": _t("contact_punish"), "ironbarbs": _t("contact_punish"), "static": _t("contact_punish"),
    "flamebody": _t("contact_punish"), "poisonpoint": _t("contact_punish"), "effectspore": _t("contact_punish"),
    "cottondown": _t("contact_punish"), "gooey": _t("contact_punish"), "tanglinghair": _t("contact_punish"),
    "cursedbody": _t("contact_punish"), "perishbody": _t("contact_punish"), "mummy": _t("contact_punish"),
    "wanderingspirit": _t("contact_punish"), "lingeringaroma": _t("contact_punish"),
    "poisontouch": _t("contact_punish"), "toxicchain": _t("contact_punish"), "stench": _t("other"),
    "serenegrace": _t("offense"),
    # ability-ignoring, trapping
    "moldbreaker": _t("mold_breaker"), "teravolt": _t("mold_breaker"), "turboblaze": _t("mold_breaker"),
    "myceliummight": _t("mold_breaker"),
    "shadowtag": _t("trap"), "arenatrap": _t("trap"), "magnetpull": _t("trap"),
}

_EFFECT_VEC_CACHE: Dict[Optional[str], Tuple[float, ...]] = {}


def ability_effects(ability: Optional[str]) -> Tuple[float, ...]:
    """Multi-hot effect vector for one ability id ("other" for anything not in the table)."""
    try:
        return _EFFECT_VEC_CACHE[ability]
    except KeyError:
        pass
    v = [0.0] * N_EFFECTS
    if ability:
        for e in ABILITY_TABLE.get(ability, ("other",)):
            v[_E[e]] = 1.0
    out = tuple(v)
    _EFFECT_VEC_CACHE[ability] = out
    return out


# --------------------------------------------------------------------------- #
# Opponent ability posterior
# --------------------------------------------------------------------------- #
_POST_CACHE: Dict[Tuple[str, FrozenSet[str]], Tuple[Tuple[str, float], ...]] = {}


def _known_ability(mon) -> Optional[str]:
    try:
        a = mon.ability
    except Exception:
        return None
    return to_id_str(a) if a else None


def ability_posterior(mon) -> Tuple[Tuple[str, float], ...]:
    """((ability_id, probability), ...) for a mon. One entry with p=1 when known."""
    known = _known_ability(mon)
    if known:
        return ((known, 1.0),)
    try:
        species = mon.species
        revealed = frozenset(m.id for m in mon.moves.values())
    except Exception:
        return ()
    key = (species, revealed)
    hit = _POST_CACHE.get(key)
    if hit is not None:
        return hit
    raw = _raw_sets()
    entry = raw.get(species)
    if entry is None:
        try:
            entry = raw.get(mon.base_species)
        except Exception:
            entry = None
    probs: Dict[str, float] = {}
    if entry:
        roles = []
        for s in entry.get("sets", []) or []:
            pool = frozenset(to_id_str(m) for m in s.get("movepool", []) or [])
            abil = [to_id_str(a) for a in s.get("abilities", []) or []]
            if abil:
                roles.append((pool, abil))
        consistent = [r for r in roles if revealed <= r[0]] or roles
        for _, abil in consistent:
            for a in abil:
                probs[a] = probs.get(a, 0.0) + 1.0 / (len(consistent) * len(abil))
    if not probs:
        try:
            poss = [to_id_str(a) for a in (mon.possible_abilities or []) if a]
        except Exception:
            poss = []
        for a in poss:
            probs[a] = probs.get(a, 0.0) + 1.0 / len(poss)
    out = tuple(sorted(probs.items(), key=lambda kv: -kv[1]))
    if len(_POST_CACHE) > 20000:
        _POST_CACHE.clear()
    _POST_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- #
# Move flags (contact, sound, ...) not stored in MoveInfo
# --------------------------------------------------------------------------- #
class _Flags:
    __slots__ = ("contact", "punch", "bite", "slicing", "pulse", "sound", "bullet", "wind", "powder",
                 "secondary")

    def __init__(self, move_id: str):
        try:
            e = Move(move_id, gen=_GEN).entry
        except Exception:
            e = {}
        f = e.get("flags") or {}
        for k in ("contact", "punch", "bite", "slicing", "pulse", "sound", "bullet", "wind", "powder"):
            setattr(self, k, bool(f.get(k)))
        self.secondary = bool(e.get("secondary") or e.get("secondaries"))


_FLAG_CACHE: Dict[str, _Flags] = {}


def move_flags(move_id: str) -> _Flags:
    f = _FLAG_CACHE.get(move_id)
    if f is None:
        f = _Flags(move_id)
        _FLAG_CACHE[move_id] = f
    return f


# --------------------------------------------------------------------------- #
# Ability damage multipliers
# --------------------------------------------------------------------------- #
_MOLD = frozenset({"moldbreaker", "teravolt", "turboblaze", "myceliummight"})
_SUN = frozenset({Weather.SUNNYDAY, Weather.DESOLATELAND})
_RAIN = frozenset({Weather.RAINDANCE, Weather.PRIMORDIALSEA})
_SAND = frozenset({Weather.SANDSTORM})
_SNOW = frozenset({Weather.SNOWSCAPE, Weather.HAIL})
_ENTRY_WEATHER = {
    "drought": Weather.SUNNYDAY, "orichalcumpulse": Weather.SUNNYDAY, "drizzle": Weather.RAINDANCE,
    "sandstream": Weather.SANDSTORM, "snowwarning": Weather.SNOWSCAPE,
}
_PRIMAL = frozenset({Weather.DESOLATELAND, Weather.PRIMORDIALSEA, Weather.DELTASTREAM})


def nullifies(dfn_ability: Optional[str], mi: MoveInfo, att_ability: Optional[str] = None) -> bool:
    """Does the defender's ability make this damaging move do nothing?"""
    if not dfn_ability or mi.status or att_ability in _MOLD:
        return False
    t = _ABILITY_IMMUNITY.get(dfn_ability)
    if t is not None and mi.type == t:
        return True
    if dfn_ability in ("windrider", "soundproof", "bulletproof", "overcoat"):
        f = move_flags(mi.id)
        return ((dfn_ability == "windrider" and f.wind) or (dfn_ability == "soundproof" and f.sound)
                or (dfn_ability == "bulletproof" and f.bullet) or (dfn_ability == "overcoat" and f.powder))
    if dfn_ability in ("armortail", "queenlymajesty", "dazzling") and mi.priority > 0:
        return True
    return False


def _att_mult(a: Optional[str], mi: MoveInfo, att: Combatant, eff: float, weather: FrozenSet,
              terrain: FrozenSet) -> float:
    if not a:
        return 1.0
    phys = mi.physical
    if a in ("hugepower", "purepower"):
        return 2.0 if phys else 1.0
    if a == "adaptability":
        return 2.0 / 1.5 if att.stab(mi.type) >= 1.5 else 1.0
    if a == "technician":
        return 1.5 if mi.bp <= 60 else 1.0
    if a in ("protean", "libero"):
        return 1.5 if att.stab(mi.type) < 1.5 else 1.0
    if a == "tintedlens":
        return 2.0 if 0 < eff < 1.0 else 1.0
    if a == "neuroforce":
        return 1.25 if eff > 1.0 else 1.0
    if a in ("gorillatactics", "hustle"):
        return 1.5 if phys else 1.0
    if a == "guts":
        st = att.mon.status if att.mon is not None else None
        if phys and st is not None and st != Status.FNT:
            return 3.0 if att.burned else 1.5
        return 1.0
    if a == "solarpower":
        return 1.5 if (not phys and weather & _SUN) else 1.0
    if a == "sandforce":
        return 1.3 if (weather & _SAND and mi.type in (PokemonType.ROCK, PokemonType.GROUND,
                                                       PokemonType.STEEL)) else 1.0
    if a == "orichalcumpulse":
        return 1.33 if (phys and weather & _SUN) else 1.0
    if a == "hadronengine":
        return 1.33 if (not phys and Field.ELECTRIC_TERRAIN in terrain) else 1.0
    if a == "protosynthesis":
        return 1.3 if weather & _SUN else 1.0
    if a == "quarkdrive":
        return 1.3 if Field.ELECTRIC_TERRAIN in terrain else 1.0
    if a in ("aerilate", "pixilate", "refrigerate", "galvanize"):
        return 1.2 if mi.type == PokemonType.NORMAL else 1.0
    type_boost = {"transistor": (PokemonType.ELECTRIC, 1.3), "dragonsmaw": (PokemonType.DRAGON, 1.5),
                  "steelworker": (PokemonType.STEEL, 1.5), "steelyspirit": (PokemonType.STEEL, 1.5),
                  "rockypayload": (PokemonType.ROCK, 1.5), "waterbubble": (PokemonType.WATER, 2.0)}.get(a)
    if type_boost is not None:
        return type_boost[1] if mi.type == type_boost[0] else 1.0
    if a in ("sheerforce", "toughclaws", "ironfist", "strongjaw", "sharpness", "megalauncher", "punkrock"):
        f = move_flags(mi.id)
        cond, m = {"sheerforce": (f.secondary, 1.3), "toughclaws": (f.contact, 1.3),
                   "ironfist": (f.punch, 1.2), "strongjaw": (f.bite, 1.5), "sharpness": (f.slicing, 1.5),
                   "megalauncher": (f.pulse, 1.5), "punkrock": (f.sound, 1.3)}[a]
        return m if cond else 1.0
    return 1.0


def _def_mult(d: Optional[str], mi: MoveInfo, dfn: Combatant, eff: float, att: Combatant) -> float:
    if not d:
        return 1.0
    phys = mi.physical
    if d in ("multiscale", "shadowshield"):
        return 0.5 if dfn.hp_frac >= 0.999 else 1.0
    if d == "furcoat":
        return 0.5 if phys else 1.0
    if d == "icescales":
        return 0.5 if not phys else 1.0
    if d == "fluffy":
        m = 2.0 if mi.type == PokemonType.FIRE else 1.0
        return m * (0.5 if move_flags(mi.id).contact else 1.0)
    if d == "thickfat":
        return 0.5 if mi.type in (PokemonType.FIRE, PokemonType.ICE) else 1.0
    if d in ("heatproof", "waterbubble"):
        return 0.5 if mi.type == PokemonType.FIRE else 1.0
    if d == "purifyingsalt":
        return 0.5 if mi.type == PokemonType.GHOST else 1.0
    if d == "dryskin":
        return 1.25 if mi.type == PokemonType.FIRE else 1.0
    if d in ("filter", "solidrock", "prismarmor"):
        return 0.75 if eff > 1.0 else 1.0
    if d == "marvelscale":
        st = dfn.mon.status if dfn.mon is not None else None
        return (1 / 1.5) if (phys and st is not None and st != Status.FNT) else 1.0
    if d == "punkrock":
        return 0.5 if move_flags(mi.id).sound else 1.0
    if d == "terashell":
        return (0.5 / eff) if (dfn.hp_frac >= 0.999 and eff > 0.5) else 1.0
    if d == "wonderguard":
        return 0.0 if eff <= 1.0 else 1.0
    if d == "unaware":
        stage = att.boosts.get(mi.off_stat, 0) or 0
        return 1.0 / _boost_mult(stage) if stage > 0 else 1.0
    return 1.0


def _one_hit_shield(d: Optional[str], mi: MoveInfo, dfn: Combatant) -> bool:
    """Disguise / Ice Face (physical) absorb the next hit completely."""
    try:
        species = dfn.mon.species
    except Exception:
        return False
    if d == "disguise" and species == "mimikyu":
        return True
    if d == "iceface" and species == "eiscue" and mi.physical:
        return True
    return False


def ability_damage(base: float, mi: MoveInfo, att: Combatant, dfn: Combatant, att_ab: Optional[str],
                   dfn_ab: Optional[str], weather: FrozenSet, terrain: FrozenSet) -> Tuple[float, bool]:
    """(damage as a fraction of dfn's current HP, KO?) after both abilities are applied to `base`."""
    if base <= 0.0 or mi.status:
        return 0.0, False
    if att_ab in _MOLD:
        dfn_ab = None
    if nullifies(dfn_ab, mi, att_ab):
        return 0.0, False
    eff = dfn.effectiveness(mi)
    scale = mi.fixed is None and not mi.half_hp
    d = base
    if scale:
        d *= _att_mult(att_ab, mi, att, eff, weather, terrain) * _def_mult(dfn_ab, mi, dfn, eff, att)
    if _one_hit_shield(dfn_ab, mi, dfn):
        return 0.0, False
    if dfn_ab == "sturdy" and dfn.hp_frac >= 0.999 and mi.hits <= 1.0:
        return min(d, 0.99), False
    return d, d >= 1.0


# --------------------------------------------------------------------------- #
# Speed and end-of-turn residual
# --------------------------------------------------------------------------- #
def ability_speed(c: Combatant, ability: Optional[str], item: Optional[str], weather: FrozenSet,
                  terrain: FrozenSet, tailwind: bool) -> float:
    s = item_speed(c, item if item != UNKNOWN_ITEM else None, tailwind)
    if not ability:
        return s
    if ((ability == "chlorophyll" and weather & _SUN) or (ability == "swiftswim" and weather & _RAIN)
            or (ability == "sandrush" and weather & _SAND) or (ability == "slushrush" and weather & _SNOW)
            or (ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in terrain)):
        return s * 2.0
    if ability == "quickfeet":
        st = c.mon.status if c.mon is not None else None
        if st is not None and st != Status.FNT:
            return s * (3.0 if c.paralyzed else 1.5)
    if ((ability == "protosynthesis" and weather & _SUN)
            or (ability == "quarkdrive" and Field.ELECTRIC_TERRAIN in terrain)):
        st = c.stats
        if st[5] >= max(st[1:5]):
            return s * 1.5
    return s


def _grounded(c: Combatant, ability: Optional[str], item: Optional[str]) -> bool:
    return not (PokemonType.FLYING in c.types or ability == "levitate" or item == "airballoon")


def residual(c: Combatant, ability: Optional[str], item: Optional[str], weather: FrozenSet,
             terrain: FrozenSet) -> float:
    """Net end-of-turn HP change as a fraction of max HP (heal > 0, damage < 0)."""
    if c.fainted or c.mon is None:
        return 0.0
    mon = c.mon
    mg = ability == "magicguard"
    r = 0.0
    if weather & _SAND and not mg and ability not in ("sandveil", "sandrush", "sandforce", "overcoat") \
            and item != "safetygoggles" \
            and not any(t in (PokemonType.ROCK, PokemonType.GROUND, PokemonType.STEEL) for t in c.types):
        r -= 1 / 16
    if weather & _RAIN:
        if ability == "raindish":
            r += 1 / 16
        elif ability == "dryskin":
            r += 1 / 8
    if weather & _SUN and ability in ("dryskin", "solarpower") and not mg:
        r -= 1 / 8
    if weather & _SNOW and ability == "icebody":
        r += 1 / 16
    if Field.GRASSY_TERRAIN in terrain and _grounded(c, ability, item):
        r += 1 / 16
    if item == "leftovers":
        r += 1 / 16
    elif item == "blacksludge":
        r += 1 / 16 if PokemonType.POISON in c.types else (0.0 if mg else -1 / 8)
    st = mon.status
    if st in (Status.PSN, Status.TOX):
        if ability == "poisonheal":
            r += 1 / 8
        elif not mg:
            if st == Status.TOX:
                n = max(1, min(15, int(getattr(mon, "status_counter", 1) or 1)))
                r -= n / 16
            else:
                r -= 1 / 8
    elif st == Status.BRN and not mg:
        r -= 1 / 32 if ability == "heatproof" else 1 / 16
    effects = mon.effects or {}
    if not mg:
        if Effect.LEECH_SEED in effects:
            r -= 1 / 8
        if Effect.SALT_CURE in effects:
            r -= 1 / 4 if any(t in (PokemonType.WATER, PokemonType.STEEL) for t in c.types) else 1 / 8
    return r


# --------------------------------------------------------------------------- #
# The block
# --------------------------------------------------------------------------- #
_A_OPP = N_EFFECTS
_A_OPP_INFO = 2 * N_EFFECTS                      # 52
_A_MOVES = _A_OPP_INFO + 2                       # 54
MOVE_FEATS = 3
_A_THREAT = _A_MOVES + N_MOVES * MOVE_FEATS      # 66
_A_RESID = _A_THREAT + 3                         # 69
_A_SPEED = _A_RESID + 4                          # 73
_A_REGEN = _A_SPEED + 1                          # 74
_A_TEAM = _A_REGEN + 1                           # 75
TEAM_FEATS = 7
ABILITIES_BLOCK_SIZE = _A_TEAM + N_TEAM * TEAM_FEATS  # 117

_INTIM_PUNISH = frozenset({"defiant", "competitive", "guarddog", "contrary", "mirrorarmor"})
_INTIM_IMMUNE = frozenset({"clearbody", "whitesmoke", "fullmetalbody", "hypercutter", "innerfocus",
                           "oblivious", "owntempo", "scrappy"})


def _clip(v: float) -> float:
    if v != v:
        return 0.0
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else float(v))


def _item(mon) -> Optional[str]:
    try:
        it = mon.item
    except Exception:
        return None
    return it if it else None


def _opp_item(mon) -> Optional[str]:
    it = _item(mon)
    return None if it == UNKNOWN_ITEM else it


class _ACtx:
    """Shared per-embed state: v2 context + abilities, items, effective weather and terrain."""

    __slots__ = ("v2", "me_ab", "opp_post", "opp_known", "me_item", "opp_item", "weather_raw", "weather",
                 "terrain", "negated", "opp_moves")

    def __init__(self, battle: Battle):
        v2 = _Ctx(battle)
        self.v2 = v2
        self.me_ab = _known_ability(v2.me) if v2.me is not None else None
        self.opp_post = ability_posterior(v2.opp) if v2.opp is not None else ()
        self.opp_known = len(self.opp_post) == 1 and self.opp_post[0][1] >= 0.999
        self.me_item = _item(v2.me) if v2.me is not None else None
        self.opp_item = _opp_item(v2.opp) if v2.opp is not None else None
        self.weather_raw = frozenset((battle.weather or {}).keys())
        self.terrain = frozenset(k for k in (battle.fields or {}).keys()
                                 if k in (Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.PSYCHIC_TERRAIN,
                                          Field.MISTY_TERRAIN))
        self.negated = self._cloud_nine(self.me_ab if v2.me_c is not None else None)
        self.weather = frozenset() if self.negated else self.weather_raw
        self.opp_moves = [mi for mi in v2.opp_pool if not mi.status] if v2.opp_c is not None else []

    def _cloud_nine(self, my_active_ab: Optional[str]) -> bool:
        if my_active_ab in ("cloudnine", "airlock"):
            return True
        return any(a in ("cloudnine", "airlock") and p >= 0.999 for a, p in self.opp_post)

    def weather_after(self, entering_ab: Optional[str]) -> Tuple[FrozenSet, bool]:
        """(effective weather if a mon with `entering_ab` switches in, did it change the weather)."""
        w = self.weather_raw
        changed = False
        new = _ENTRY_WEATHER.get(entering_ab or "")
        if new is not None and new not in w and not (w & _PRIMAL):
            w = frozenset({new})
            changed = True
        opp_negates = any(a in ("cloudnine", "airlock") and p >= 0.999 for a, p in self.opp_post)
        if opp_negates or entering_ab in ("cloudnine", "airlock"):
            return frozenset(), changed
        return w, changed

    # --- damage helpers ----------------------------------------------------- #
    def ours_vs_opp(self, mi: MoveInfo, me: Combatant, me_ab: Optional[str], me_item) -> Tuple[float, float, float]:
        """(P(nullified), expected damage, P(KO)) of our move vs their active, over their posterior."""
        v2 = self.v2
        opp = v2.opp_c
        base = estimate_damage_items(mi, me, opp, self.weather, att_item=me_item, dfn_item=self.opp_item)
        if base <= 0.0:
            return 0.0, 0.0, 0.0
        post = self.opp_post or ((None, 1.0),)
        p_null = exp = p_ko = 0.0
        for a, p in post:
            if me_ab not in _MOLD and nullifies(a, mi, me_ab):
                p_null += p
                continue
            d, ko = ability_damage(base, mi, me, opp, me_ab, a, self.weather, self.terrain)
            exp += p * min(d, 1.0)
            p_ko += p * float(ko)
        return p_null, exp, p_ko

    def opp_vs(self, c: Combatant, c_ab: Optional[str], c_item, weather: FrozenSet) -> Tuple[float, float, float]:
        """(max expected damage, P(KO) of that move, fraction of their damaging moves `c_ab` nullifies)."""
        opp = self.v2.opp_c
        moves = self.opp_moves
        if opp is None or not moves:
            return 0.0, 0.0, 0.0
        post = self.opp_post or ((None, 1.0),)
        best = best_ko = 0.0
        n_null = 0.0
        for mi in moves:
            base = estimate_damage_items(mi, opp, c, weather, att_item=self.opp_item, dfn_item=c_item)
            exp = ko = nul = 0.0
            for a, p in post:
                if nullifies(c_ab, mi, a):
                    nul += p
                    continue
                d, k = ability_damage(base, mi, opp, c, a, c_ab, weather, self.terrain)
                exp += p * min(d, 1.0)
                ko += p * float(k)
            n_null += nul
            if exp > best:
                best, best_ko = exp, ko
        return best, best_ko, n_null / len(moves)

    def p_faster(self, c: Combatant, c_ab: Optional[str], c_item, weather: FrozenSet, tailwind: bool) -> float:
        v2 = self.v2
        opp = v2.opp_c
        mine = ability_speed(c, c_ab, c_item, weather, self.terrain, tailwind)
        post = self.opp_post or ((None, 1.0),)
        p = 0.0
        for a, pa in post:
            theirs = ability_speed(opp, a, self.opp_item, weather, self.terrain, v2.opp_tailwind)
            faster = (mine < theirs) if v2.trick_room else (mine > theirs)
            tie = mine == theirs
            p += pa * (1.0 if faster else (0.5 if tie else 0.0))
        return p

    def opp_residual(self, weather: FrozenSet) -> float:
        opp = self.v2.opp_c
        post = self.opp_post or ((None, 1.0),)
        return sum(p * residual(opp, a, self.opp_item, weather, self.terrain) for a, p in post)


def _part_identity(battle: Battle, ac: _ACtx, out: List[float]) -> None:
    if ac.v2.me is not None:
        out[0:N_EFFECTS] = ability_effects(ac.me_ab)
    if ac.v2.opp is not None:
        vec = [0.0] * N_EFFECTS
        for a, p in ac.opp_post:
            for i, x in enumerate(ability_effects(a)):
                vec[i] += p * x
        out[_A_OPP:_A_OPP + N_EFFECTS] = vec
        out[_A_OPP_INFO] = 1.0 if ac.opp_known else 0.0
        out[_A_OPP_INFO + 1] = min(max(len(ac.opp_post) - 1, 0) / 3.0, 1.0)


def _part_moves(battle: Battle, ac: _ACtx, out: List[float]) -> None:
    v2 = ac.v2
    if v2.me_c is None or v2.opp_c is None:
        return
    for i, mv in enumerate(move_slots(battle)):
        if mv is None:
            continue
        mi = move_info(mv)
        if mi is None or mi.status:
            continue
        p_null, exp, p_ko = ac.ours_vs_opp(mi, v2.me_c, ac.me_ab, ac.me_item)
        j = _A_MOVES + MOVE_FEATS * i
        out[j], out[j + 1], out[j + 2] = p_null, exp, p_ko


def _part_threat(battle: Battle, ac: _ACtx, out: List[float]) -> None:
    v2 = ac.v2
    if v2.me_c is None or v2.opp_c is None:
        return
    best, ko, nul = ac.opp_vs(v2.me_c, ac.me_ab, ac.me_item, ac.weather)
    out[_A_THREAT], out[_A_THREAT + 1], out[_A_THREAT + 2] = best, ko, nul


def _part_residual(battle: Battle, ac: _ACtx, out: List[float]) -> None:
    v2 = ac.v2
    if v2.me_c is not None:
        r = residual(v2.me_c, ac.me_ab, ac.me_item, ac.weather, ac.terrain)
        out[_A_RESID] = 4.0 * r
        out[_A_RESID + 1] = 1.0 if (r < 0 and -r >= v2.me_c.hp_frac) else 0.0
    if v2.opp_c is not None:
        r = ac.opp_residual(ac.weather)
        out[_A_RESID + 2] = 4.0 * r
        out[_A_RESID + 3] = 1.0 if (r < 0 and -r >= v2.opp_c.hp_frac) else 0.0


def _part_speed(battle: Battle, ac: _ACtx, out: List[float]) -> None:
    v2 = ac.v2
    if v2.me_c is not None and v2.opp_c is not None:
        out[_A_SPEED] = 2.0 * ac.p_faster(v2.me_c, ac.me_ab, ac.me_item, ac.weather, v2.my_tailwind) - 1.0
    if v2.me_c is not None and ac.me_ab == "regenerator":
        out[_A_REGEN] = 3.0 * min(1 / 3, 1.0 - v2.me_c.hp_frac)


def _part_team(battle: Battle, ac: _ACtx, out: List[float]) -> None:
    v2 = ac.v2
    opp_c = v2.opp_c
    if opp_c is None:
        return
    moves = ac.opp_moves
    phys_frac = (sum(1 for m in moves if m.physical) / len(moves)) if moves else 0.0
    punish = sum(p for a, p in ac.opp_post if a in _INTIM_PUNISH)
    immune = sum(p for a, p in ac.opp_post if a in _INTIM_IMMUNE)
    for i, mon in enumerate(list(battle.team.values())[:N_TEAM]):
        j = _A_TEAM + TEAM_FEATS * i
        try:
            if mon.fainted:
                continue
            c = v2.combatant(mon, True)
            ab = _known_ability(mon)
            item = _item(mon)
            active = mon is v2.me and v2.me_c is not None
            # The active mon "switching in" means staying in: current weather, no entry effect.
            weather, changed = (ac.weather, False) if active else ac.weather_after(ab)
            r = ac.opp_residual(weather)
            dmg = max(0.0, -r)
            hp = max(opp_c.hp_frac, 1e-6)
            out[j] = min(dmg / hp, 1.0)
            out[j + 1] = 1.0 if dmg >= opp_c.hp_frac - 1e-9 and dmg > 0 else 0.0
            if ab == "intimidate" and not active:
                out[j + 2] = phys_frac * max(0.0, 1.0 - punish - immune) - punish
            best, _, nul = ac.opp_vs(c, ab, item, weather)
            out[j + 3] = nul
            out[j + 4] = best
            out[j + 5] = 2.0 * ac.p_faster(c, ab, item, weather, v2.my_tailwind) - 1.0
            out[j + 6] = 1.0 if changed else 0.0
        except Exception:
            out[j:j + TEAM_FEATS] = [0.0] * TEAM_FEATS


_PARTS = (_part_identity, _part_moves, _part_threat, _part_residual, _part_speed, _part_team)
_PART_SLICES = (slice(0, _A_MOVES), slice(_A_MOVES, _A_THREAT), slice(_A_THREAT, _A_RESID),
                slice(_A_RESID, _A_SPEED), slice(_A_SPEED, _A_TEAM), slice(_A_TEAM, ABILITIES_BLOCK_SIZE))


def abilities_block(battle: Battle) -> List[float]:
    """ABILITIES_BLOCK_SIZE floats in [-1, 1]. Never raises."""
    out = [0.0] * ABILITIES_BLOCK_SIZE
    try:
        ac = _ACtx(battle)
    except Exception:
        return out
    for part, sl in zip(_PARTS, _PART_SLICES):
        try:
            part(battle, ac, out)
        except Exception:
            out[sl] = [0.0] * (sl.stop - sl.start)
    return [_clip(float(x)) for x in out]


__all__ = ["ABILITIES_BLOCK_SIZE", "ABILITY_EFFECTS", "ABILITY_TABLE", "abilities_block", "ability_effects",
           "ability_posterior", "ability_damage", "ability_speed", "nullifies", "residual"]
