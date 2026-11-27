"""Held-item and critical-hit feature blocks (to be appended to the v2 vector in v3).

    items_block(battle) -> list[float]   (ITEMS_BLOCK_SIZE = 71)
    crit_block(battle)  -> list[float]   (CRIT_BLOCK_SIZE  = 22)

Both return exactly their size, values in [-1, 1], and never raise (each sub-part
is wrapped in try/except and left as zeros). Pure functions of the poke-env Battle;
`features.py` / `features_v2.py` are only imported, never modified.

How poke-env 0.17 represents items (checked in poke_env/battle/pokemon.py and
abstract_battle.py):
  * our mons: `Pokemon.item` is overwritten from every request (`request["item"]`,
    a Showdown id such as "heavydutyboots", or "" once consumed / knocked off).
  * opponent mons start as "unknown_item" (GenData.UNKNOWN_ITEM). `-item` messages
    (Air Balloon on switch-in, Frisk, Trick/Switcheroo swaps, Thief, ...) and
    `[from] item: X` tags on -damage/-heal (Life Orb, Leftovers, Rocky Helmet, ...)
    set the id. `-enditem` (berry eaten, Focus Sash / Booster Energy / balloon used,
    Knock Off) sets it to None. Trick swaps the two `_item`s in place.
  * Choice items are never announced, so the opponent's Choice state is inferred.

Layout (ITEMS_BLOCK_SIZE = 71):
  [ 0:22] our active's item, one-hot over ITEM_VOCAB (20 common items, other, none)
  [22:44] opponent active's item *posterior* over ITEM_VOCAB: one-hot when revealed
          (or "none" when removed), else the random-battle prior below
  [44]    opponent item still unknown
  [45]    we are Choice-locked (hold a Choice item and used a move this stint)
  [46]    our locked move's type effectiveness vs their active, log2(eff)/2, -1 if immune
  [47]    our locked move's item-aware damage vs their active (fraction of current HP)
  [48]    P(opponent holds a Choice item) (posterior)
  [49]    P(opponent is locked) = [48] if it has used a move this stint, else 0
  [50]    the opponent's (possibly) locked move is super effective on our active
  [51]    its item-aware damage vs our active (fraction of current HP)
  [52]    it is a status move or does no damage to us (free turn if they are locked)
  [53:61] our 4 move slots (actions 6-9): item-aware damage (fraction of their
          current HP, capped at 1), item-aware likely-KO flag (incl. Sash/Sturdy)
  [61]    opponent's item-aware max damage vs our active (revealed or most likely item)
  [62]    ... it can KO us (flag)
  [63]    P(opponent can KO us) marginalised over its item posterior
  [64]    P(opponent outspeeds us), item-aware (Choice Scarf both sides, Trick Room)
  [65:71] our team slots (switch actions 0-5): holds Heavy-Duty Boots

Layout (CRIT_BLOCK_SIZE = 22):
  [ 0:12] per our move slot: crit probability, "KO only if it crits" flag,
          expected damage incl. crits = (1-p)*d + p*d_crit (capped at 1)
  [12]    opponent: highest crit probability among its likely damaging moves vs us
  [13]    opponent can KO us only with a crit (no non-crit KO in its movepool)
  [14]    ... highest crit probability among those crit-only KO moves
  [15]    opponent has an always-crit move in its likely movepool
  [16]    our active is crit-immune (Battle Armor / Shell Armor)
  [17]    opponent active is crit-immune (known ability or every randbats ability)
  [18]    our active has Focus Energy, [19] theirs has Focus Energy
  [20]    our crit-stage bonus (item + ability + Focus Energy) / 3, [21] theirs

Size rationale: one-hots for two actives dominate (2 x 22); a 22-entry vocabulary
covers ~97% of items gen9 random singles assigns (teams.ts getPriorityItem/getItem),
the long tail goes to "other". Using the *same* vocabulary for the opponent's
posterior lets one vector say both "revealed: X" and "probably X" without a second
one-hot. Damage / crit features are per move slot (4) so they line up with the
action indices; everything else is a handful of scalars. Total 93 dims.

Crit rules (Showdown sim/battle-actions.ts): crit ratio starts at move.critRatio
(default 1), +1 Scope Lens / Razor Claw, +1 Super Luck, +2 Focus Energy, clamped to
4; chance = 1/24, 1/8, 1/2, 1 for ratio 1..4. `willCrit` moves (Flower Trick,
Wicked Blow, Surging Strikes, Frost Breath, Storm Throw, ...) always crit; Merciless
crits poisoned targets. Battle Armor / Shell Armor block crits. A crit is x1.5
(x2.25 with Sniper) and ignores the attacker's negative offensive boosts and the
defender's positive defensive boosts. Fixed-damage moves ignore crits.

Simplifications (all deliberate, to stay cheap):
  * Item multipliers: Choice Band/Specs x1.5 Atk/SpA, Assault Vest x1.5 SpD,
    Eviolite x1.5 Def/SpD (species with evolutions), Life Orb x1.3, Expert Belt
    x1.2 on super-effective hits, type-boosting items / Ogerpon masks x1.2 (applied
    to final damage rather than base power: < 2% difference), Knock Off x1.5 vs a
    held item, Air Balloon = Ground immunity, Choice Scarf x1.5 Spe. Everything else
    (Booster Energy, Light Ball, Loaded Dice hit counts, ...) is ignored.
  * Multi-hit moves use one crit roll for all hits.
  * The opponent's ability is used only when poke-env knows/infers it.
  * The item prior emulates the decision list in teams.ts per random-battle role,
    replacing "the 4 chosen moves are all physical / all special / have no status
    move" by hypergeometric probabilities over the role's movepool (setup roles
    always keep their setup move). Roles consistent with the revealed moves are
    mixed uniformly; lead-only and team-dependent rules are dropped.
"""

from __future__ import annotations

import json
import math
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from poke_env.battle import Battle, Effect, Field, PokemonType, Status, Weather
from poke_env.battle.move import Move
from poke_env.data import GenData, to_id_str

from pokebot.features import N_MOVES, N_TEAM, move_slots
from pokebot.features_v2 import (
    SETS_PATH,
    Combatant,
    MoveInfo,
    _boost_mult,
    _Ctx,
    _STAT_IDX,
    _type_chart,
    estimate_damage,
    move_info,
)

_GEN = 9

# --------------------------------------------------------------------------- #
# Item vocabulary
# --------------------------------------------------------------------------- #
ITEM_VOCAB: Tuple[str, ...] = (
    "heavydutyboots", "leftovers", "lifeorb", "choiceband", "choicespecs", "choicescarf",
    "assaultvest", "focussash", "rockyhelmet", "eviolite", "boosterenergy", "sitrusberry",
    "lumberry", "weaknesspolicy", "airballoon", "lightclay", "loadeddice", "chestoberry",
    "whiteherb", "statusorb",  # Flame Orb / Toxic Orb
    "other", "none",
)
N_ITEMS = len(ITEM_VOCAB)  # 22
_ITEM_IDX = {k: i for i, k in enumerate(ITEM_VOCAB)}
_OTHER, _NONE = _ITEM_IDX["other"], _ITEM_IDX["none"]
_ITEM_ALIAS = {"flameorb": "statusorb", "toxicorb": "statusorb"}
UNKNOWN_ITEM = GenData.UNKNOWN_ITEM  # "unknown_item"

CHOICE_ITEMS = frozenset({"choiceband", "choicespecs", "choicescarf"})
_CHOICE_IDX = tuple(_ITEM_IDX[i] for i in ("choiceband", "choicespecs", "choicescarf"))
_TYPE_BOOST = {
    "charcoal": PokemonType.FIRE, "mysticwater": PokemonType.WATER, "magnet": PokemonType.ELECTRIC,
    "miracleseed": PokemonType.GRASS, "nevermeltice": PokemonType.ICE, "blackbelt": PokemonType.FIGHTING,
    "poisonbarb": PokemonType.POISON, "softsand": PokemonType.GROUND, "sharpbeak": PokemonType.FLYING,
    "twistedspoon": PokemonType.PSYCHIC, "silverpowder": PokemonType.BUG, "hardstone": PokemonType.ROCK,
    "spelltag": PokemonType.GHOST, "dragonfang": PokemonType.DRAGON, "blackglasses": PokemonType.DARK,
    "metalcoat": PokemonType.STEEL, "silkscarf": PokemonType.NORMAL, "fairyfeather": PokemonType.FAIRY,
}
_ALL_MOVE_BOOST = {"hearthflamemask", "wellspringmask", "cornerstonemask"}  # Ogerpon masks: x1.2 all moves
_CRIT_ITEMS = {"scopelens", "razorclaw"}
_CRIT_IMMUNE = {"battlearmor", "shellarmor"}
_SUN = frozenset({Weather.SUNNYDAY, Weather.DESOLATELAND})


def item_index(item: Optional[str]) -> int:
    """Vocabulary index of a poke-env item id ('' / None -> none, unlisted -> other)."""
    if not item:
        return _NONE
    item = _ITEM_ALIAS.get(item, item)
    return _ITEM_IDX.get(item, _OTHER)


# --------------------------------------------------------------------------- #
# Species data
# --------------------------------------------------------------------------- #
_DEX: Optional[dict] = None


def _dex() -> dict:
    global _DEX
    if _DEX is None:
        _DEX = GenData.from_gen(_GEN).pokedex
    return _DEX


_NFE_CACHE: Dict[str, bool] = {}


def is_nfe(species: Optional[str]) -> bool:
    """True if the species can still evolve (Eviolite applies)."""
    if not species:
        return False
    try:
        return _NFE_CACHE[species]
    except KeyError:
        pass
    try:
        ok = bool(_dex().get(species, {}).get("evos"))
    except Exception:
        ok = False
    _NFE_CACHE[species] = ok
    return ok


# --------------------------------------------------------------------------- #
# Item-aware damage
# --------------------------------------------------------------------------- #
class _View:
    """A Combatant seen through an item (stat multipliers) and/or a crit (boost clamping).

    Duck-types the parts of Combatant that features_v2.estimate_damage uses.
    """

    __slots__ = ("c", "level", "burned", "hp_frac", "fainted", "hp_cur", "mult", "clamp", "ground_immune")

    def __init__(self, c: Combatant, mult: Optional[Dict[str, float]], clamp: int, ground_immune: bool):
        self.c = c
        self.level = c.level
        self.burned = c.burned
        self.hp_frac = c.hp_frac
        self.fainted = c.fainted
        self.hp_cur = c.hp_cur
        self.mult = mult
        self.clamp = clamp  # 0: none, -1: ignore negative boosts (crit attacker), +1: ignore positive def boosts
        self.ground_immune = ground_immune

    def stat(self, key: str) -> float:
        c = self.c
        stage = c.boosts.get(key, 0) or 0
        if self.clamp < 0 and stage < 0:
            stage = 0
        elif self.clamp > 0 and stage > 0 and key in ("def", "spd"):
            stage = 0
        v = c.stats[_STAT_IDX[key]] * _boost_mult(stage)
        if self.mult:
            v *= self.mult.get(key, 1.0)
        return v

    def stab(self, t: PokemonType) -> float:
        return self.c.stab(t)

    def effectiveness(self, mi: MoveInfo) -> float:
        if self.ground_immune and mi.type == PokemonType.GROUND:
            return 0.0
        return self.c.effectiveness(mi)


_BAND = {"atk": 1.5}
_SPECS = {"spa": 1.5}
_AV = {"spd": 1.5}
_EVIO = {"def": 1.5, "spd": 1.5}


def _att_view(c: Combatant, item: Optional[str], crit: bool) -> _View:
    mult = _BAND if item == "choiceband" else (_SPECS if item == "choicespecs" else None)
    return _View(c, mult, -1 if crit else 0, False)


def _def_view(c: Combatant, item: Optional[str], crit: bool) -> _View:
    mult = None
    if item == "assaultvest":
        mult = _AV
    elif item == "eviolite" and is_nfe(getattr(c.mon, "species", None)):
        mult = _EVIO
    return _View(c, mult, 1 if crit else 0, item == "airballoon")


def _holds_item(item: Optional[str]) -> bool:
    return bool(item) and item != "none"


def _dmg(mi: MoveInfo, av: _View, dv: _View, weather, att_item: Optional[str], dfn_item: Optional[str],
         crit: bool) -> float:
    d = estimate_damage(mi, av, dv, weather)
    if d <= 0.0 or mi.fixed is not None or mi.half_hp:
        return d
    m = 1.0
    if att_item:
        if att_item == "lifeorb":
            m *= 5324 / 4096
        elif att_item == "expertbelt":
            if dv.effectiveness(mi) > 1.0:
                m *= 4915 / 4096
        elif att_item in _ALL_MOVE_BOOST or _TYPE_BOOST.get(att_item) == mi.type:
            m *= 4915 / 4096
    if mi.id == "knockoff" and _holds_item(dfn_item):
        m *= 1.5
    if crit:
        m *= 2.25 if _ability(av.c.mon) == "sniper" else 1.5
    return d * m


def estimate_damage_items(mi: Optional[MoveInfo], att: Combatant, dfn: Combatant, weather=None,
                          att_item: Optional[str] = None, dfn_item: Optional[str] = None,
                          crit: bool = False) -> float:
    """v2 `estimate_damage` plus held items (and optionally a critical hit).

    Fraction of dfn's *current* HP, accuracy excluded. Items are poke-env ids
    ("choiceband", ...); None / "" / "unknown_item" mean no modelled effect
    (except that "unknown_item" counts as a held item for Knock Off).
    """
    if mi is None:
        return 0.0
    return _dmg(mi, _att_view(att, att_item, crit), _def_view(dfn, dfn_item, crit), weather, att_item,
                dfn_item, crit)


def item_speed(c: Combatant, item: Optional[str], tailwind: bool = False) -> float:
    s = c.speed(tailwind)
    if item == "choicescarf":
        s *= 1.5
    elif item == "ironball":
        s *= 0.5
    return s


def _ability(mon) -> Optional[str]:
    try:
        return mon.ability
    except Exception:
        return None


def _ko(d: float, mi: MoveInfo, dfn: Combatant, dfn_item: Optional[str]) -> bool:
    """Likely KO (v2 convention: accuracy-weighted damage >= current HP), respecting Sash / Sturdy."""
    if d * mi.acc < 1.0:
        return False
    if dfn.hp_frac >= 0.999 and mi.hits <= 1.0 and (dfn_item == "focussash" or _ability(dfn.mon) == "sturdy"):
        return False
    return True


# --------------------------------------------------------------------------- #
# Crits
# --------------------------------------------------------------------------- #
_CRIT_TABLE = (1 / 24, 1 / 8, 1 / 2, 1.0)  # stage 0, 1, 2, 3+
_CRIT_CACHE: Dict[str, Tuple[int, bool]] = {}


def move_crit_info(move_id: str) -> Tuple[int, bool]:
    """(base crit stage, always-crits) from the move data (critRatio / willCrit)."""
    try:
        return _CRIT_CACHE[move_id]
    except KeyError:
        pass
    stage, will = 0, False
    try:
        e = Move(move_id, gen=_GEN).entry
        stage = max(0, int(e.get("critRatio", 1) or 1) - 1)
        will = bool(e.get("willCrit"))
    except Exception:
        pass
    _CRIT_CACHE[move_id] = (stage, will)
    return stage, will


def crit_chance_from_stage(stage: int) -> float:
    """Showdown gen 7+: stage 0 -> 1/24, 1 -> 1/8, 2 -> 1/2, 3+ -> always."""
    return _CRIT_TABLE[max(0, min(3, int(stage)))]


def crit_stage_bonus(item: Optional[str], ability: Optional[str], focus_energy: bool) -> int:
    return (1 if item in _CRIT_ITEMS else 0) + (1 if ability == "superluck" else 0) + (2 if focus_energy else 0)


def crit_chance(move_id: str, bonus: int = 0, defender_immune: bool = False, always: bool = False) -> float:
    """Probability that `move_id` crits given the attacker's stage bonus."""
    if defender_immune:
        return 0.0
    stage, will = move_crit_info(move_id)
    if will or always:
        return 1.0
    return crit_chance_from_stage(stage + bonus)


def _focus_energy(mon) -> bool:
    try:
        return Effect.FOCUS_ENERGY in (mon.effects or {})
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Opponent item prior (emulates gen9 teams.ts getPriorityItem/getItem per role)
# --------------------------------------------------------------------------- #
_RAW_SETS: Optional[dict] = None


def _raw_sets() -> dict:
    global _RAW_SETS
    if _RAW_SETS is None:
        try:
            _RAW_SETS = {to_id_str(k): v for k, v in json.loads(SETS_PATH.read_text()).items()}
        except Exception:
            _RAW_SETS = {}
    return _RAW_SETS


_SETUP_ROLES = {"Setup Sweeper", "Bulky Setup", "Fast Bulky Setup"}
_BULKY_ROLES = {"Bulky Attacker", "Bulky Support", "Bulky Setup"}
_PIVOTS = {"uturn", "flipturn", "voltswitch", "partingshot", "chillyreception", "shedtail", "teleport"}
_NO_BAND = {"dragontail", "fakeout", "firstimpression", "flamecharge", "rapidspin", "trailblaze"}
_NO_LO = {"flamecharge", "nuzzle", "rapidspin"}
_CHOICE_OK_STATUS = {"trick", "switcheroo", "healingwish"}  # status moves Choice users may carry
_PHYSICAL_SETUP = {"bellydrum", "bulkup", "coil", "curse", "dragondance", "honeclaws", "howl", "meditate",
                   "poweruppunch", "swordsdance", "tidyup", "victorydance"}
_SPEED_SETUP = ("agility", "autotomize", "flamecharge", "rockpolish", "snowscape", "trailblaze")


def _type_mult(attack: PokemonType, types: Sequence[PokemonType]) -> float:
    chart = _type_chart().get(attack, {})
    m = 1.0
    for t in types:
        m *= chart.get(t, 1.0)
    return m


_MULTIHIT_CACHE: Dict[str, bool] = {}


def _is_2_to_5_hit(move_id: str) -> bool:
    try:
        return _MULTIHIT_CACHE[move_id]
    except KeyError:
        pass
    try:
        mh = Move(move_id, gen=_GEN).entry.get("multihit")
        ok = isinstance(mh, (list, tuple)) and len(mh) == 2 and mh[1] == 5
    except Exception:
        ok = False
    _MULTIHIT_CACHE[move_id] = ok
    return ok


def _role_prior(sid: str, role_entry: dict, dex: dict) -> List[float]:
    """Approximate item distribution (over ITEM_VOCAB) for one species/role."""
    dist = [0.0] * N_ITEMS
    mass = [1.0]

    def give(item: str, frac: float) -> None:
        frac = max(0.0, min(1.0, frac))
        amt = mass[0] * frac
        if amt > 0:
            dist[item_index(item)] += amt
            mass[0] -= amt

    role = role_entry.get("role", "")
    pool = [to_id_str(m) for m in role_entry.get("movepool", [])]
    pset = set(pool)
    abil = [to_id_str(a) for a in role_entry.get("abilities", []) or []] or [""]
    n = len(pool)
    k = min(4, n)

    def incl(*moves: str) -> float:  # P(at least one of these pool moves is among the 4 chosen)
        c = sum(1 for m in moves if m in pset)
        if c == 0 or n == 0:
            return 0.0
        return 1.0 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0.0)

    def afrac(*names: str) -> float:  # fraction of the set's abilities in `names`
        return sum(1 for a in abil if a in names) / len(abil)

    infos = [mi for mi in (move_info(m) for m in pool) if mi is not None]
    phys = sum(1 for mi in infos if not mi.status and mi.physical)
    spec = sum(1 for mi in infos if not mi.status and not mi.physical)
    stat = sum(1 for mi in infos if mi.status)
    has_setup = any(mi.setup for mi in infos)
    has_prio = any(mi.priority > 0 and not mi.status for mi in infos)
    has_recovery = any(mi.recovery for mi in infos)
    bs = dex.get("baseStats", {}) or {}
    spe, atk, spa = bs.get("spe", 80), bs.get("atk", 80), bs.get("spa", 80)
    hp, df = bs.get("hp", 80), bs.get("def", 80)
    types = []
    for t in dex.get("types", []) or []:
        try:
            types.append(PokemonType.from_name(t))
        except Exception:
            pass
    rock = _type_mult(PokemonType.ROCK, types)
    ground = _type_mult(PokemonType.GROUND, types)
    fighting = _type_mult(PokemonType.FIGHTING, types)

    def pall(cnt: int) -> float:  # P(all k chosen moves come from a group of size cnt)
        if n == 0 or cnt < k:
            return 0.0
        return math.comb(cnt, k) / math.comb(n, k)

    # ---- getPriorityItem (singles) ----
    if role == "Fast Bulky Setup":
        give("boosterenergy", afrac("quarkdrive", "protosynthesis"))
    if sid == "lokix":
        give("other" if role == "Fast Attacker" else "lifeorb", 1.0)  # Silver Powder / Life Orb
    if dex.get("requiredItems") or dex.get("requiredItem") or sid == "pikachu":
        give("other", 1.0)
    if role == "AV Pivot":
        give("assaultvest", 1.0)
    if sid in ("regieleki", "froslass"):  # Magnet / Wide Lens
        give("other", 1.0)
    if "normal" in [t.name.lower() for t in types]:
        give("other", incl("doubleedge") * incl("fakeout"))  # Silk Scarf
    give("other", incl("populationbomb"))  # Wide Lens
    if afrac("hustle") > 0 and has_setup:
        give("other", 0.5 * afrac("hustle"))
    if sid == "smeargle":
        give("focussash", 1.0)
    give("other", incl("clangoroussoul") + (incl("shiftgear") if sid == "toxtricity" else 0.0))  # Throat Spray
    if sid.startswith("magearna") and role == "Tera Blast user":
        give("weaknesspolicy", 1.0)
    give("choicescarf", incl("dragonenergy", "lastrespects", "waterspout"))
    if afrac("imposter") > 0 or (sid == "magnezone" and role == "Fast Attacker") or (
            sid == "rampardos" and role == "Fast Attacker"):
        give("choicescarf", max(afrac("imposter"), 1.0 if sid in ("magnezone", "rampardos") else 0.0))
    give("heavydutyboots", incl("courtchange"))
    if sid == "luvdisc":
        give("heavydutyboots", 1.0)
    if sid == "terapagos":
        give("heavydutyboots", 1.0 - incl("rest"))
    give("sitrusberry", max(afrac("cheekpouch", "cudchew", "harvest", "ripen"), incl("bellydrum", "filletaway")))
    p_trick = incl("healingwish", "switcheroo", "trick")
    if p_trick > 0:
        if 60 <= spe <= 108 and role != "Wallbreaker" and not has_prio:
            give("choicescarf", p_trick)
        else:
            give("choiceband" if phys > spec else "choicespecs", p_trick)
    if sid in ("latias", "latios"):  # Soul Dew if any status move is chosen
        give("other", 1.0 - pall(phys + spec))
    if sid == "scyther":
        give("heavydutyboots", 1.0)  # Eviolite only as a lead without U-turn
    give("statusorb", afrac("poisonheal", "quickfeet"))
    if is_nfe(sid):
        give("eviolite", 1.0)
    give("statusorb", max(afrac("guts"), incl("facade")) * (1.0 - incl("sleeptalk")))
    give("lifeorb", afrac("magicguard", "sheerforce"))
    if afrac("angershell") > 0:  # sample(Expert Belt, Lum Berry, Scope Lens, Sitrus Berry)
        a = afrac("angershell")
        dist[item_index("lumberry")] += mass[0] * a / 4
        dist[item_index("sitrusberry")] += mass[0] * a / 4
        give("other", a / 2 / (1.0 - a / 2) if a < 2 else 1.0)
    if afrac("unburden") > 0:
        p_wh = incl("closecombat", "leafstorm")
        a = afrac("unburden")
        dist[item_index("whiteherb")] += mass[0] * a * p_wh
        give("sitrusberry", a * (1.0 - p_wh) / max(1e-9, 1.0 - a * p_wh))
    if afrac("skilllink") < 1.0:
        sl = [m for m in pool if _is_2_to_5_hit(m)]
        give("loadeddice", incl(*sl) * (1.0 - afrac("skilllink")) if sid != "breloom" else 0.0)
    give("whiteherb", incl("shellsmash"))
    give("other", incl("meteorbeam", "electroshot"))  # Power Herb
    give("none", incl("acrobatics"))
    give("lightclay", max(incl("auroraveil"), incl("lightscreen") * incl("reflect")))
    give("other", afrac("gluttony"))  # pinch berries
    if sid.startswith("giratina") and "sleeptalk" not in pset:
        give("leftovers", incl("rest"))
    give("chestoberry", incl("rest") * (1.0 - incl("sleeptalk")))
    if rock >= 4.0 and sid != "yanmega":
        give("heavydutyboots", 1.0)

    # ---- getItem ----
    forced_status = role in _SETUP_ROLES and has_setup
    if forced_status or n == 0:
        p_phys = p_spec = p_nostat = 0.0
    else:
        p_phys = pall(phys) * (1.0 - incl(*_NO_BAND))
        p_spec = pall(spec)
        if "uturn" in pset or "flipturn" in pset:  # 3 special + U-turn/Flip Turn also gets Specs/Scarf
            p_spec += math.comb(spec, 3) / math.comb(n, k) if n >= 4 and spec >= 3 else 0.0
        p_nostat = pall(phys + spec)
    p_mixed = max(0.0, p_nostat - p_phys - p_spec)
    m0 = mass[0]
    scarf_p = (role != "Wallbreaker" and (atk >= 100 or afrac("hugepower", "purepower") > 0)
               and 60 <= spe <= 108 and afrac("speedboost") < 1.0 and not has_prio)
    scarf_s = (role != "Wallbreaker" and spa >= 100 and 60 <= spe <= 108 and "uturn" not in pset
               and afrac("speedboost", "tintedlens") < 1.0 and not has_prio)
    for item, p, scarf in (("choiceband", p_phys, scarf_p), ("choicespecs", p_spec, scarf_s)):
        if p > 0:
            dist[item_index(item)] += m0 * p * (0.5 if scarf else 1.0)
            if scarf:
                dist[item_index("choicescarf")] += m0 * p * 0.5
    mass[0] = m0 * max(0.0, 1.0 - p_phys - p_spec)
    rest = max(1e-9, 1.0 - p_phys - p_spec)
    if role == "Bulky Setup" and not (pset & _PHYSICAL_SETUP):
        give("weaknesspolicy", incl(*_SPEED_SETUP))
    if role not in ("Fast Attacker", "Wallbreaker", "Tera Blast user"):
        give("assaultvest", p_mixed / rest)
    if sid == "golem":
        give("weaknesspolicy", incl(*_SPEED_SETUP))
        give("other", 1.0)  # Custap Berry
    give("leftovers", incl("substitute"))
    if rock >= 2.0:
        give("heavydutyboots", 1.0)
    if role == "Fast Support" and PokemonType.FLYING not in types and afrac("levitate") < 1.0:
        give("heavydutyboots", incl(*(_PIVOTS | {"defog", "mortalspin", "rapidspin"})))
    if role == "Bulky Setup":
        give("weaknesspolicy", incl("dragondance"))
    p_rh = afrac("roughskin")
    if role in ("Bulky Support", "Bulky Attacker") and hp + df >= 180:
        p_rh = max(p_rh, 0.5 * afrac("regenerator"))
    if not has_setup and has_recovery and fighting < 2.0 and hp + df > 200:
        p_rh = max(p_rh, 0.5 * (1.0 - afrac("regenerator")))
    give("rockyhelmet", p_rh)
    if has_setup:
        give("lumberry", incl("outrage"))
    give("leftovers", incl("protect") * (1.0 - afrac("speedboost")))
    if not has_setup and afrac("levitate") < 1.0 and ground >= 4.0:
        give("airballoon", 1.0)
    if role in _BULKY_ROLES:
        give("leftovers", 1.0)
    lo_ok = 1.0 - incl(*_NO_LO)
    if role in ("Fast Support", "Fast Bulky Setup"):
        give("lifeorb", lo_ok if phys + spec > stat else 0.0)
        give("leftovers", 1.0)
    if role in ("Fast Attacker", "Setup Sweeper", "Tera Blast user", "Wallbreaker"):
        give("lifeorb", lo_ok)
    give("leftovers", 1.0)
    return dist


_PRIOR_CACHE: Dict[Tuple[str, FrozenSet[str]], Tuple[float, ...]] = {}
_GLOBAL_PRIOR: Optional[Tuple[float, ...]] = None


def _normalise(d: Sequence[float]) -> Tuple[float, ...]:
    s = sum(d)
    if s <= 0:
        out = [0.0] * N_ITEMS
        out[_OTHER] = 1.0
        return tuple(out)
    return tuple(x / s for x in d)


def global_item_prior() -> Tuple[float, ...]:
    """Average role prior over every species in sets.json (used for unknown species)."""
    global _GLOBAL_PRIOR
    if _GLOBAL_PRIOR is None:
        acc = [0.0] * N_ITEMS
        dex = _dex()
        for sid, entry in _raw_sets().items():
            roles = entry.get("sets", [])
            for r in roles:
                try:
                    for i, x in enumerate(_role_prior(sid, r, dex.get(sid, {}))):
                        acc[i] += x / len(roles)
                except Exception:
                    pass
        _GLOBAL_PRIOR = _normalise(acc)
    return _GLOBAL_PRIOR


def species_item_prior(species: str, revealed: FrozenSet[str] = frozenset()) -> Tuple[float, ...]:
    """P(item) over ITEM_VOCAB for a random-battle species given its revealed move ids.

    Roles whose movepool contains every revealed move are mixed uniformly (all roles
    if none is consistent); then revealed moves rule out items: a status move (other
    than Trick/Switcheroo/Healing Wish) rules out Choice items and Assault Vest, a
    special attack rules out Choice Band, a physical attack (other than U-turn /
    Flip Turn) rules out Choice Specs, and both kinds rule out Choice Scarf.
    """
    key = (species, revealed)
    try:
        return _PRIOR_CACHE[key]
    except KeyError:
        pass
    sets = _raw_sets()
    entry = sets.get(species)
    if entry is None:  # cosmetic / battle formes: fall back to the base species
        try:
            bsp = to_id_str(_dex().get(species, {}).get("baseSpecies", ""))
            entry = sets.get(bsp)
            species_for_dex = bsp
        except Exception:
            entry = None
    else:
        species_for_dex = species
    if entry is None:
        base = list(global_item_prior())
    else:
        roles = entry.get("sets", [])
        consistent = [r for r in roles if revealed <= {to_id_str(m) for m in r.get("movepool", [])}] or roles
        dex = _dex().get(species_for_dex, {})
        base = [0.0] * N_ITEMS
        for r in consistent:
            for i, x in enumerate(_role_prior(species_for_dex, r, dex)):
                base[i] += x
        if not consistent:
            base = list(global_item_prior())
    infos = [mi for mi in (move_info(m) for m in revealed) if mi is not None]
    if not any(mi.id in _CHOICE_OK_STATUS for mi in infos):
        rv_status = any(mi.status for mi in infos)
        rv_spec = any(not mi.status and not mi.physical for mi in infos)
        rv_phys = any(not mi.status and mi.physical and mi.id not in ("uturn", "flipturn") for mi in infos)
        if rv_status:
            for i in _CHOICE_IDX + (_ITEM_IDX["assaultvest"],):
                base[i] = 0.0
        if rv_spec:
            base[_ITEM_IDX["choiceband"]] = 0.0
        if rv_phys:
            base[_ITEM_IDX["choicespecs"]] = 0.0
        if rv_spec and rv_phys:
            base[_ITEM_IDX["choicescarf"]] = 0.0
    out = _normalise(base)
    if len(_PRIOR_CACHE) > 20000:
        _PRIOR_CACHE.clear()
    _PRIOR_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- #
# Protocol log: moves used since each side's latest switch-in
# --------------------------------------------------------------------------- #
_SCAN_MAX_MSGS = 1500


def stint_moves(battle: Battle) -> Tuple[List[str], List[str]]:
    """(our moves, opponent's moves) since each active's latest switch-in, oldest first.

    Called moves (`[from] ...`, e.g. Sleep Talk's callee or Outrage's lockedmove
    continuation) are skipped: the Choice lock is on the move actually selected.
    """
    data = getattr(battle, "_replay_data", None)
    me_role = getattr(battle, "player_role", None)
    if not data or me_role not in ("p1", "p2"):
        return [], []
    opp_role = "p2" if me_role == "p1" else "p1"
    moves: Dict[str, List[str]] = {me_role: [], opp_role: []}
    done = {me_role: False, opp_role: False}
    stop = max(0, len(data) - _SCAN_MAX_MSGS)
    for idx in range(len(data) - 1, stop - 1, -1):
        ev = data[idx]
        if len(ev) < 3:
            continue
        kind = ev[1]
        if kind not in ("move", "switch", "drag"):
            continue
        who = ev[2][:2]
        if who not in done or done[who]:
            continue
        if kind == "move":
            if len(ev) >= 4 and not any("[from]" in f for f in ev[4:]):
                moves[who].append(to_id_str(ev[3]))
        else:
            done[who] = True
            if done[me_role] and done[opp_role]:
                break
    return moves[me_role][::-1], moves[opp_role][::-1]


# --------------------------------------------------------------------------- #
# Per-call context
# --------------------------------------------------------------------------- #
def _item_of(mon) -> Optional[str]:
    try:
        it = mon.item
    except Exception:
        return None
    return it if it else None


class _ItemCtx:
    __slots__ = ("v2", "me_item", "opp_item_raw", "opp_known", "post", "opp_item", "my_stint", "opp_stint",
                 "p_choice", "my_bonus", "opp_bonus", "me_immune", "opp_immune")

    def __init__(self, battle: Battle):
        v2 = _Ctx(battle)
        self.v2 = v2
        self.me_item = _item_of(v2.me) if v2.me is not None else None
        try:
            self.my_stint, self.opp_stint = stint_moves(battle)
        except Exception:
            self.my_stint, self.opp_stint = [], []
        opp = v2.opp
        self.post: Tuple[float, ...] = tuple([0.0] * N_ITEMS)
        self.opp_item_raw: Optional[str] = None
        self.opp_known = True
        self.opp_item: Optional[str] = None  # revealed item, or the posterior's most likely one
        self.p_choice = 0.0
        if opp is not None:
            raw = _item_of(opp)
            self.opp_item_raw = raw
            if raw == UNKNOWN_ITEM:
                self.opp_known = False
                revealed = frozenset(m.id for m in opp.moves.values())
                post = list(species_item_prior(opp.species, revealed))
                # Air Balloon is announced on switch-in, and Booster Energy is used up on
                # switch-in unless sun / Electric Terrain already powers the ability.
                post[_ITEM_IDX["airballoon"]] = 0.0
                if not (v2.weather & _SUN or Field.ELECTRIC_TERRAIN in (battle.fields or {})):
                    post[_ITEM_IDX["boosterenergy"]] = 0.0
                post = list(_normalise(post))
                if len(set(self.opp_stint)) >= 2:  # two different moves without switching: not Choice
                    for i in _CHOICE_IDX:
                        post[i] = 0.0
                    post = list(_normalise(post))
                self.post = tuple(post)
                best = max(range(N_ITEMS), key=post.__getitem__)
                name = ITEM_VOCAB[best]
                self.opp_item = None if name in ("other", "none") else ("flameorb" if name == "statusorb" else name)
                self.p_choice = sum(post[i] for i in _CHOICE_IDX)
            else:
                one = [0.0] * N_ITEMS
                one[item_index(raw)] = 1.0
                self.post = tuple(one)
                self.opp_item = raw
                self.p_choice = 1.0 if raw in CHOICE_ITEMS else 0.0
        me = v2.me
        my_ab = _ability(me) if me is not None else None
        opp_ab = _ability(opp) if opp is not None else None
        self.my_bonus = crit_stage_bonus(self.me_item, my_ab, _focus_energy(me)) if me is not None else 0
        self.opp_bonus = (crit_stage_bonus(self.opp_item_raw, opp_ab, _focus_energy(opp))
                          if opp is not None else 0)
        self.me_immune = my_ab in _CRIT_IMMUNE
        self.opp_immune = False
        if opp is not None:
            if opp_ab:
                self.opp_immune = opp_ab in _CRIT_IMMUNE
            else:
                ent = _raw_sets().get(opp.species)
                if ent:
                    abil = {to_id_str(a) for s in ent.get("sets", []) for a in s.get("abilities", []) or []}
                    self.opp_immune = bool(abil) and abil <= _CRIT_IMMUNE

    def opp_held_for_knockoff(self) -> Optional[str]:
        return self.opp_item if self.opp_known else (self.opp_item or UNKNOWN_ITEM)

    def my_crit_p(self, mi: MoveInfo) -> float:
        opp = self.v2.opp
        always = _ability(self.v2.me) == "merciless" and opp is not None and opp.status in (Status.PSN, Status.TOX)
        return crit_chance(mi.id, self.my_bonus, self.opp_immune, always)

    def opp_crit_p(self, mi: MoveInfo) -> float:
        me = self.v2.me
        always = _ability(self.v2.opp) == "merciless" and me is not None and me.status in (Status.PSN, Status.TOX)
        return crit_chance(mi.id, self.opp_bonus, self.me_immune, always)


def _log_eff(eff: float) -> float:
    if eff <= 0.0:
        return -1.0
    return max(-1.0, min(1.0, math.log2(eff) / 2.0))


def _clip(v: float) -> float:
    if v != v:  # NaN
        return 0.0
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else float(v))


# --------------------------------------------------------------------------- #
# Items block
# --------------------------------------------------------------------------- #
ITEMS_BLOCK_SIZE = 2 * N_ITEMS + 1 + 8 + 2 * N_MOVES + 3 + 1 + N_TEAM  # 22+22+1 + 8 + 8+3 + 1 + 6 = 71
_I_OPP = N_ITEMS
_I_UNK = 2 * N_ITEMS
_I_LOCK = _I_UNK + 1
_I_DMG = _I_LOCK + 8
_I_THREAT = _I_DMG + 2 * N_MOVES
_I_SPEED = _I_THREAT + 3
_I_BOOTS = _I_SPEED + 1
assert _I_BOOTS + N_TEAM == ITEMS_BLOCK_SIZE


def _items_onehots(battle: Battle, ic: _ItemCtx, out: List[float]) -> None:
    if ic.v2.me is not None:
        out[item_index(ic.me_item)] = 1.0
    if ic.v2.opp is not None:
        out[_I_OPP:_I_OPP + N_ITEMS] = list(ic.post)
        out[_I_UNK] = 0.0 if ic.opp_known else 1.0


def _items_choice(battle: Battle, ic: _ItemCtx, out: List[float]) -> None:
    v2 = ic.v2
    me_c, opp_c = v2.me_c, v2.opp_c
    j = _I_LOCK
    if me_c is not None and ic.me_item in CHOICE_ITEMS and ic.my_stint:
        out[j] = 1.0
        mi = move_info(ic.my_stint[0])
        if mi is not None and opp_c is not None and not mi.status:
            out[j + 1] = _log_eff(opp_c.effectiveness(mi))
            out[j + 2] = min(1.0, estimate_damage_items(mi, me_c, opp_c, v2.weather, ic.me_item,
                                                        ic.opp_held_for_knockoff()))
    if v2.opp is None:
        return
    out[j + 3] = ic.p_choice
    if ic.opp_stint and ic.p_choice > 0:
        out[j + 4] = ic.p_choice
        mi = move_info(ic.opp_stint[0])
        if mi is not None and me_c is not None and opp_c is not None:
            if mi.status:
                out[j + 7] = 1.0
            else:
                eff = me_c.effectiveness(mi)
                out[j + 5] = 1.0 if eff > 1.0 else 0.0
                att_item = ic.opp_item if ic.opp_item in CHOICE_ITEMS else (
                    "choiceband" if mi.physical else "choicespecs")
                d = estimate_damage_items(mi, opp_c, me_c, v2.weather, att_item, ic.me_item)
                out[j + 6] = min(1.0, d)
                out[j + 7] = 1.0 if d <= 0.0 else 0.0


def _items_damage(battle: Battle, ic: _ItemCtx, out: List[float]) -> None:
    v2 = ic.v2
    me_c, opp_c = v2.me_c, v2.opp_c
    if me_c is None or opp_c is None:
        return
    w = v2.weather
    opp_item = ic.opp_held_for_knockoff()
    av, dv = _att_view(me_c, ic.me_item, False), _def_view(opp_c, opp_item, False)
    avc, dvc = _att_view(me_c, ic.me_item, True), _def_view(opp_c, opp_item, True)
    for i, m in enumerate(move_slots(battle)[:N_MOVES]):
        mi = move_info(m)
        if mi is None or mi.status:
            continue
        if ic.my_crit_p(mi) >= 1.0:
            d = _dmg(mi, avc, dvc, w, ic.me_item, opp_item, True)
        else:
            d = _dmg(mi, av, dv, w, ic.me_item, opp_item, False)
        out[_I_DMG + 2 * i] = min(1.0, d)
        out[_I_DMG + 2 * i + 1] = 1.0 if _ko(d, mi, opp_c, opp_item) else 0.0

    # Opponent threat on our active: assumed item, and KO probability over the item posterior.
    pool = [mi for mi in v2.opp_pool if not mi.status]
    if not pool:
        return
    me_item = ic.me_item
    dv_me = _def_view(me_c, me_item, False)
    dvc_me = _def_view(me_c, me_item, True)
    opp_views = {}

    def view(item, crit):
        key = (item if item in ("choiceband", "choicespecs") else None, crit)
        v = opp_views.get(key)
        if v is None:
            v = opp_views[key] = _att_view(opp_c, key[0], crit)
        return v

    # item -> P(item) for the items that change the opponent's damage; rest lumped into None
    if ic.opp_known:
        dist = {ic.opp_item: 1.0}
    else:
        post = ic.post
        dist = {it: post[_ITEM_IDX[it]] for it in ("choiceband", "choicespecs", "lifeorb")}
        dist[None] = max(0.0, 1.0 - sum(dist.values()))
    best = 0.0
    ko_assumed = False
    ko_item = {it: False for it in dist}
    for mi in pool:
        crit = ic.opp_crit_p(mi) >= 1.0
        dvm = dvc_me if crit else dv_me
        d = _dmg(mi, view(ic.opp_item, crit), dvm, w, ic.opp_item, me_item, crit)
        if d > best:
            best = d
        if _ko(d, mi, me_c, me_item):
            ko_assumed = True
        for it in dist:
            if ko_item[it] or dist[it] <= 0:
                continue
            if it in ("choiceband", "choicespecs") and (it == "choiceband") != mi.physical:
                it_d = _dmg(mi, view(None, crit), dvm, w, None, me_item, crit)
            elif it == ic.opp_item:
                it_d = d
            else:
                it_d = _dmg(mi, view(it, crit), dvm, w, it, me_item, crit)
            if _ko(it_d, mi, me_c, me_item):
                ko_item[it] = True
    out[_I_THREAT] = min(1.0, best)
    out[_I_THREAT + 1] = 1.0 if ko_assumed else 0.0
    out[_I_THREAT + 2] = sum(p for it, p in dist.items() if ko_item[it])


def _items_speed(battle: Battle, ic: _ItemCtx, out: List[float]) -> None:
    v2 = ic.v2
    me_c, opp_c = v2.me_c, v2.opp_c
    if me_c is None or opp_c is None:
        return
    mine = item_speed(me_c, ic.me_item, v2.my_tailwind)
    if ic.opp_known:
        dist = {ic.opp_item: 1.0}
    else:
        p_scarf = ic.post[_ITEM_IDX["choicescarf"]]
        dist = {"choicescarf": p_scarf, None: 1.0 - p_scarf}
    p = 0.0
    for it, w in dist.items():
        theirs = item_speed(opp_c, it, v2.opp_tailwind)
        a, b = (-mine, -theirs) if v2.trick_room else (mine, theirs)
        p += w * (1.0 if b > a else (0.5 if a == b else 0.0))
    out[_I_SPEED] = p


def _items_boots(battle: Battle, ic: _ItemCtx, out: List[float]) -> None:
    for i, mon in enumerate(list(battle.team.values())[:N_TEAM]):
        if _item_of(mon) == "heavydutyboots":
            out[_I_BOOTS + i] = 1.0


_ITEM_PARTS = (_items_onehots, _items_choice, _items_damage, _items_speed, _items_boots)
_ITEM_PART_SLICES = (slice(0, _I_LOCK), slice(_I_LOCK, _I_DMG), slice(_I_DMG, _I_SPEED),
                     slice(_I_SPEED, _I_BOOTS), slice(_I_BOOTS, ITEMS_BLOCK_SIZE))


def items_block(battle: Battle) -> List[float]:
    """Held-item features (see module docstring for the layout). Never raises."""
    out = [0.0] * ITEMS_BLOCK_SIZE
    try:
        ic = _ItemCtx(battle)
    except Exception:
        return out
    for fn, sl in zip(_ITEM_PARTS, _ITEM_PART_SLICES):
        try:
            fn(battle, ic, out)
        except Exception:
            out[sl] = [0.0] * (sl.stop - sl.start)
    return [_clip(v) for v in out]


# --------------------------------------------------------------------------- #
# Crit block
# --------------------------------------------------------------------------- #
CRIT_SLOT_FEATS = 3
CRIT_BLOCK_SIZE = N_MOVES * CRIT_SLOT_FEATS + 10  # 12 + 10 = 22
_C_OPP = N_MOVES * CRIT_SLOT_FEATS


def _crit_mine(battle: Battle, ic: _ItemCtx, out: List[float]) -> None:
    v2 = ic.v2
    me_c, opp_c = v2.me_c, v2.opp_c
    if me_c is None or opp_c is None:
        return
    w = v2.weather
    opp_item = ic.opp_held_for_knockoff()
    av, dv = _att_view(me_c, ic.me_item, False), _def_view(opp_c, opp_item, False)
    avc, dvc = _att_view(me_c, ic.me_item, True), _def_view(opp_c, opp_item, True)
    for i, m in enumerate(move_slots(battle)[:N_MOVES]):
        mi = move_info(m)
        if mi is None or mi.status:
            continue
        p = ic.my_crit_p(mi)
        dc = _dmg(mi, avc, dvc, w, ic.me_item, opp_item, True)
        d0 = dc if p >= 1.0 else _dmg(mi, av, dv, w, ic.me_item, opp_item, False)
        j = CRIT_SLOT_FEATS * i
        out[j] = p
        out[j + 1] = 1.0 if (p > 0 and not _ko(d0, mi, opp_c, opp_item) and _ko(dc, mi, opp_c, opp_item)) else 0.0
        out[j + 2] = min(1.0, (1.0 - p) * d0 + p * dc)


def _crit_opp(battle: Battle, ic: _ItemCtx, out: List[float]) -> None:
    v2 = ic.v2
    me_c, opp_c = v2.me_c, v2.opp_c
    j = _C_OPP
    if v2.me is not None:
        out[j + 4] = 1.0 if ic.me_immune else 0.0
        out[j + 6] = 1.0 if _focus_energy(v2.me) else 0.0
        out[j + 8] = min(ic.my_bonus, 3) / 3.0
    if v2.opp is not None:
        out[j + 5] = 1.0 if ic.opp_immune else 0.0
        out[j + 7] = 1.0 if _focus_energy(v2.opp) else 0.0
        out[j + 9] = min(ic.opp_bonus, 3) / 3.0
    pool = [mi for mi in v2.opp_pool if not mi.status]
    if v2.opp is not None:
        out[j + 3] = 1.0 if any(move_crit_info(mi.id)[1] for mi in pool) else 0.0
    if me_c is None or opp_c is None or not pool:
        return
    w = v2.weather
    me_item, opp_item = ic.me_item, ic.opp_item
    av, dv = _att_view(opp_c, opp_item, False), _def_view(me_c, me_item, False)
    avc, dvc = _att_view(opp_c, opp_item, True), _def_view(me_c, me_item, True)
    max_p = 0.0
    any_ko = False
    crit_only_p = -1.0
    for mi in pool:
        p = ic.opp_crit_p(mi)
        dc = _dmg(mi, avc, dvc, w, opp_item, me_item, True)
        if dc <= 0.0:
            continue
        d0 = dc if p >= 1.0 else _dmg(mi, av, dv, w, opp_item, me_item, False)
        if p > max_p:
            max_p = p
        if _ko(d0, mi, me_c, me_item):
            any_ko = True
        elif p > 0 and _ko(dc, mi, me_c, me_item) and p > crit_only_p:
            crit_only_p = p
    out[j] = max_p
    if not any_ko and crit_only_p >= 0:
        out[j + 1] = 1.0
        out[j + 2] = crit_only_p


_CRIT_PARTS = (_crit_mine, _crit_opp)
_CRIT_PART_SLICES = (slice(0, _C_OPP), slice(_C_OPP, CRIT_BLOCK_SIZE))


def crit_block(battle: Battle) -> List[float]:
    """Critical-hit features (see module docstring for the layout). Never raises."""
    out = [0.0] * CRIT_BLOCK_SIZE
    try:
        ic = _ItemCtx(battle)
    except Exception:
        return out
    for fn, sl in zip(_CRIT_PARTS, _CRIT_PART_SLICES):
        try:
            fn(battle, ic, out)
        except Exception:
            out[sl] = [0.0] * (sl.stop - sl.start)
    return [_clip(v) for v in out]


__all__ = [
    "ITEMS_BLOCK_SIZE", "CRIT_BLOCK_SIZE", "items_block", "crit_block", "ITEM_VOCAB", "item_index",
    "estimate_damage_items", "item_speed", "species_item_prior", "global_item_prior", "stint_moves",
    "crit_chance", "crit_chance_from_stage", "crit_stage_bonus", "move_crit_info", "is_nfe",
]
