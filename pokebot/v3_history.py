"""v3 history block: a fixed-size summary of the whole battle and of our own past decisions.

Why this exists
---------------
The policy is a feed-forward MLP (MaskablePPO has no recurrent+maskable variant in sb3-contrib), so it
only "remembers" what we put into the observation. Pokemon is a POMDP: the current `Battle` snapshot
does not say what happened on previous turns (did our last move fail? does this opponent switch out
when threatened?). This block turns the protocol log into a fixed-size vector:

  * recent turns, explicit   : the last K=6 turns, most recent first, one 27-float record each
  * whole-battle aggregates  : 27 floats of counts / rates covering *all* turns
  * our last action          : 7-way one-hot (move slot 0-3 / tera flag / switch / other) - the classic
                               "previous action as input" trick for memoryless policies

Source of truth: `battle._replay_data`, the list of split protocol messages that poke-env appends to in
`AbstractBattle.parse_message` (verified for poke-env 0.17.0 and a live local server):
  * every battle-room line that reaches `parse_message` is appended (copied) *before* poke-env's own
    ignore list, so `-crit`, `-supereffective`, `-resisted`, `-immune`, `-miss`, `-fail`, `t:`, blank
    `['', '']` separators etc. are all there; `|request|` JSON is NOT (it goes to `parse_request`),
    nor `expire`/`uhtmlchange`/`tempnotify*`, `error`, `bigerror`, `showteam`. `win`/`tie` are appended
    by `won_by` / `tied`.
  * at a normal decision the log ends with `['', 'turn', 'N']`: the whole previous turn is present and
    nothing of turn N yet. At a forced switch after a faint it ends with `... faint ..., ['', ''],
    ['', 'upkeep']` (turn N in progress, no `turn` marker yet); after our U-turn / Volt Switch it ends
    mid-turn right after the pivot move's messages.
  * HP strings: our own side is absolute `cur/max` (e.g. `179/321`), the opponent's is out of 100
    (`84/100`, HP Percentage Mod); fainted is `0 fnt`; a status may follow (`45/100 par`). Both are
    turned into a fraction, so the two perspectives look the same to the network.
  * identifiers carry the position letter (`p1a: Garchomp`); side conditions use `p1: <username>`.

Everything is mapped to the agent's perspective (we are `battle.player_role`). Only public log
information is used, plus our own team's movesets (which we know anyway) for the slot mapping.

Performance: a streaming parser (`_HistoryState`) consumes each message once. Its state is cached
per battle (LRU of `_CACHE_MAX` battles, validated by log identity + length), so each call only parses
the new messages. Because the parser is a pure function of the message sequence and every
battle-state-dependent quantity (move slot mapping, threat check) is computed at output time, the
cached result equals a fresh parse exactly (tested). Never raises: every sub-part is guarded.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Tuple

from poke_env.data import GenData, to_id_str

from pokebot.features import move_slots

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
K_TURNS = 6  # explicit recent turns. 6 covers a typical "stint" of one matchup; older turns are in aggregates.
N_SLOTS = 4  # move slots, aligned with actions 6..9 (and tera moves 22..25) via features.move_slots

# Per-turn record (27 floats). Index names are used by the tests and the decoder below.
TURN_FIELDS: Tuple[str, ...] = (
    "present",       # 0  1 if this record exists (else the whole record is 0)
    "in_progress",   # 1  1 if this is the current, unfinished turn (forced switch / pivot decision)
    "me_move",       # 2  we used a move (our decision; called moves e.g. via Sleep Talk don't count)
    "me_switch",     # 3  we switched voluntarily (before acting, not a faint replacement / pivot / Eject Button)
    "me_cant",       # 4  we chose to act but couldn't (sleep, paralysis, flinch, recharge, ...)
    "me_tera",       # 5  we terastallized this turn
    "me_bp",         # 6  base power of our move / 150 (0 for status moves)
    "me_slot0",      # 7  one-hot: which current move slot it was (only if that mon is still active)
    "me_slot1",      # 8
    "me_slot2",      # 9
    "me_slot3",      # 10
    "me_eff",        # 11 result on the target: immune -1, resisted -0.5, neutral 0, super-effective +1
    "me_fail",       # 12 our move missed / failed / was protected against / had no target
    "me_dmg",        # 13 opponent HP fraction removed by our direct hits this turn
    "me_ko",         # 14 an opponent mon fainted this turn
    "opp_move",      # 15 opponent used a move
    "opp_switch",    # 16 opponent switched voluntarily
    "opp_cant",      # 17 opponent couldn't act
    "opp_tera",      # 18 opponent terastallized
    "opp_bp",        # 19 opponent move base power / 150
    "opp_eff",       # 20 effectiveness of their move on us
    "opp_fail",      # 21 their move missed / failed / was protected against
    "opp_dmg",       # 22 our HP fraction removed by their direct hits
    "opp_ko",        # 23 one of our mons fainted this turn
    "first",         # 24 who acted first: +1 us, -1 them, 0 unless both acted (move or cant)
    "crit",          # 25 +1 we landed a crit, -1 we were crit, 0 neither (or both, rare)
    "status",        # 26 +1 we inflicted a major status, -1 we received one, 0 neither/both
)
TURN_FEATS = len(TURN_FIELDS)  # 27
_TI = {name: i for i, name in enumerate(TURN_FIELDS)}

AGG_FIELDS: Tuple[str, ...] = (
    "turns",              # 0  turns elapsed / 60 (clipped)
    "my_switch_rate",     # 1  our voluntary switches per turn
    "opp_switch_rate",    # 2  opponent voluntary switches per turn
    "opp_sw_threat",      # 3  P(opp switches | we had a super-effective move on it) = (k+1)/(n+2)
    "opp_sw_threat_conf", # 4  min(n_threatened_turns / 10, 1): how much evidence feeds field 3
    "opp_sw_safe",        # 5  P(opp switches | no SE move on it) = (k+1)/(n+2)
    "crits_by_me",        # 6  / 4
    "crits_on_me",        # 7  / 4
    "dmg_dealt",          # 8  total opponent HP fractions removed by our hits / 10 (heals let it exceed 6)
    "dmg_taken",          # 9  total own HP fractions removed by their hits / 10
    "kos_by_me",          # 10 opponent faints / 6
    "kos_on_me",          # 11 own faints / 6
    "slot0_uses",         # 12 times our *current* active used move slot i this battle / 10
    "slot1_uses",         # 13
    "slot2_uses",         # 14
    "slot3_uses",         # 15
    "me_tera_used",       # 16 we already terastallized (tera is once per battle)
    "me_tera_turn",       # 17 turn we did / 60
    "opp_tera_used",      # 18 opponent already terastallized
    "opp_tera_turn",      # 19 turn they did / 60
    "opp_moves_seen",     # 20 distinct (mon, move) pairs the opponent revealed / 24
    "opp_hazards_set",    # 21 hazard layers the opponent set on our side / 4
    "my_hazards_set",     # 22 hazard layers we set on their side / 4
    "my_side_cleared",    # 23 hazard removals from our side / 4
    "opp_side_cleared",   # 24 hazard removals from their side / 4
    "my_stint",           # 25 turns since our active switched in / 10
    "opp_stint",          # 26 turns since their active switched in / 10
)
AGG_FEATS = len(AGG_FIELDS)  # 27

LAST_ACTION_FIELDS: Tuple[str, ...] = ("slot0", "slot1", "slot2", "slot3", "tera", "switch", "other")
LAST_ACTION_FEATS = len(LAST_ACTION_FIELDS)  # 7

RECENT_SLICE = slice(0, K_TURNS * TURN_FEATS)
AGG_SLICE = slice(RECENT_SLICE.stop, RECENT_SLICE.stop + AGG_FEATS)
LAST_ACTION_SLICE = slice(AGG_SLICE.stop, AGG_SLICE.stop + LAST_ACTION_FEATS)
HISTORY_V3_SIZE: int = LAST_ACTION_SLICE.stop  # 6*27 + 27 + 7 = 196

# --------------------------------------------------------------------------- #
# Static data
# --------------------------------------------------------------------------- #
_GEN = GenData.from_gen(9)
_EFF = {"-supereffective": 1.0, "-resisted": -0.5, "-immune": -1.0}
_HAZARDS = ("stealth rock", "spikes", "toxic spikes", "sticky web", "steelsurge")
_PROTECT_WORDS = ("protect", "detect", "shield", "bunker", "obstruct", "silk trap", "bulwark", "max guard",
                  "mat block", "quick guard", "wide guard", "crafty shield")
_FORCED_OUT_ITEMS = ("eject button", "eject pack")
_FORCED_OUT_ABILITIES = ("emergency exit", "wimp out")
_CACHE_MAX = 32  # live battles tracked at once (training uses <= a handful); LRU-evicted


@lru_cache(maxsize=4096)
def _move_data(move_id: str) -> Tuple[float, bool, str]:
    """(base power / 150 clipped, is_status, TYPE) for a move id; unknown moves -> (0, False, '')."""
    entry = _GEN.moves.get(move_id)
    if not entry:
        return 0.0, False, ""
    bp = min(float(entry.get("basePower") or 0) / 150.0, 1.0)
    return bp, entry.get("category") == "Status", str(entry.get("type", "")).upper()


@lru_cache(maxsize=4096)
def _species_types(species_id: str) -> Tuple[str, ...]:
    entry = _GEN.pokedex.get(species_id)
    if not entry:
        return ()
    return tuple(str(t).upper() for t in entry.get("types", ()))


@lru_cache(maxsize=16384)
def _has_se(move_types: Tuple[str, ...], species_id: str) -> bool:
    """Does any attacking type in `move_types` hit `species_id`'s base types for >= 2x?"""
    defend = _species_types(species_id)
    if not defend:
        return False
    chart = _GEN.type_chart
    for atk in move_types:
        mult = 1.0
        for d in defend:
            mult *= chart.get(d, {}).get(atk, 1.0)
        if mult >= 2.0:
            return True
    return False


def _hp_fraction(hp_status: str) -> Optional[float]:
    """'179/321' -> 0.557, '45/100 par' -> 0.45, '0 fnt' -> 0.0; garbage -> None."""
    try:
        token = hp_status.split(" ")[0]
        if "/" in token:
            cur, mx = token.split("/")
            mx_f = float(mx)
            return max(0.0, min(1.0, float(cur) / mx_f)) if mx_f > 0 else None
        return 0.0 if token == "0" else None
    except Exception:
        return None


def _ident(raw: str) -> str:
    """'p1a: Garchomp' -> 'p1: Garchomp' (the key poke-env uses in battle.team)."""
    if len(raw) > 3 and raw[2] != ":":
        return raw[:2] + raw[3:]
    return raw


def _has_from(ev: List[str], start: int) -> Optional[str]:
    for tok in ev[start:]:
        if tok.startswith("[from]"):
            return tok[6:].strip().lower()
    return None


# --------------------------------------------------------------------------- #
# Streaming parser
# --------------------------------------------------------------------------- #
class _Side:
    """One side's half of a turn record."""

    __slots__ = ("move", "switch", "cant", "tera", "move_id", "mover", "bp", "eff", "fail", "dmg", "ko",
                 "crit", "status", "acted")

    def __init__(self) -> None:
        self.move = self.switch = self.cant = self.tera = False
        self.move_id: Optional[str] = None
        self.mover: Optional[str] = None
        self.bp = 0.0
        self.eff = 0.0
        self.fail = False
        self.dmg = 0.0
        self.ko = False  # this side KO'd an opposing mon
        self.crit = False  # this side landed a crit
        self.status = False  # this side inflicted a status
        self.acted = False  # moved or 'cant'


class _Turn:
    __slots__ = ("number", "me", "opp", "first", "active", "upkeep", "fainted", "key")

    def __init__(self, number: int, key: Tuple[Optional[str], Optional[str]]) -> None:
        self.number = number
        self.me = _Side()
        self.opp = _Side()
        self.first: Optional[str] = None  # 'me' / 'opp'
        self.active = False  # any action happened in this segment
        self.upkeep = False  # past the residual phase (later switches are faint replacements)
        self.fainted = {"me": False, "opp": False}
        self.key = key  # (our active ident, opp active species) at turn start, for the threat stats


class _HistoryState:
    """Consumes protocol messages one at a time; output is computed by `features(battle)`."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.opp_role = "p2" if role == "p1" else "p1"
        self.n = 0  # messages consumed
        self.first_msg: Any = None
        self.last_msg: Any = None
        self.turn_no = 0
        self.cur: Optional[_Turn] = None
        self.done: Deque[_Turn] = deque(maxlen=K_TURNS)
        self.hp: Dict[str, float] = {}
        self.active: Dict[str, Optional[str]] = {"me": None, "opp": None}
        self.species: Dict[str, str] = {}
        self.switch_in_turn = {"me": 0, "opp": 0}
        self.mover: Optional[str] = None  # side whose move's effects are being reported
        self.forced_out = {"me": False, "opp": False}
        self.tera_this_turn = False
        # aggregates
        self.vol_switches = {"me": 0, "opp": 0}
        self.crits = {"me": 0, "opp": 0}  # landed by side
        self.dmg = {"me": 0.0, "opp": 0.0}  # dealt by side
        self.kos = {"me": 0, "opp": 0}  # scored by side
        self.tera_turn = {"me": 0, "opp": 0}
        self.hazards_set = {"me": 0, "opp": 0}  # set by side
        self.cleared = {"me": 0, "opp": 0}  # removed from side
        self.uses: Dict[Tuple[str, str], int] = {}
        self.opp_moves: set = set()
        self.matchups: Dict[Tuple[Optional[str], Optional[str]], List[int]] = {}  # key -> [turns, opp switches]
        self.last_action: Optional[Tuple] = None  # ('move', ident, move_id, tera) / ('switch',) / ('other',)

    # ----------------------------------------------------------------- parse
    def _side(self, raw: str) -> Optional[str]:
        who = raw[:2]
        if who == self.role:
            return "me"
        if who == self.opp_role:
            return "opp"
        return None

    def _rec(self, side: str) -> Optional[_Side]:
        if self.cur is None:
            return None
        return self.cur.me if side == "me" else self.cur.opp

    def feed(self, data: List[List[str]]) -> None:
        for i in range(self.n, len(data)):
            ev = data[i]
            try:
                if isinstance(ev, list) and len(ev) > 1:
                    self._event(ev)
            except Exception:
                pass  # garbage line: skip it (deterministic, so cache == fresh still holds)
        if len(data) > self.n:
            if self.n == 0:
                self.first_msg = data[0]
            self.last_msg = data[-1]
        self.n = len(data)

    def _finish_turn(self) -> None:
        cur = self.cur
        if cur is None:
            return
        self.done.append(cur)
        stats = self.matchups.setdefault(cur.key, [0, 0])
        stats[0] += 1
        stats[1] += 1 if cur.opp.switch else 0

    def _event(self, ev: List[str]) -> None:
        kind = ev[1]
        if kind == "turn":
            self._finish_turn()
            self.turn_no = int(ev[2])
            opp_active = self.active["opp"]
            self.cur = _Turn(self.turn_no, (self.active["me"], self.species.get(opp_active) if opp_active else None))
            self.mover = None
            self.tera_this_turn = False
            return
        if kind == "upkeep":
            if self.cur is not None:
                self.cur.upkeep = True
            self.mover = None
            return
        if len(ev) < 3 or not isinstance(ev[2], str):
            return
        cur = self.cur

        if kind in ("switch", "drag", "replace"):
            side = self._side(ev[2])
            if side is None:
                return
            ident = _ident(ev[2])
            if len(ev) > 3:
                self.species[ident] = to_id_str(ev[3].split(",")[0])
            if len(ev) > 4:
                frac = _hp_fraction(ev[4])
                if frac is not None:
                    self.hp[ident] = frac
            self.active[side] = ident
            if kind == "replace":  # Illusion broken: same slot, real identity revealed
                return
            self.switch_in_turn[side] = self.turn_no
            self.mover = None
            if cur is not None:
                cur.active = True
                rec = self._rec(side)
                voluntary = (kind == "switch" and not rec.acted and not cur.upkeep and not cur.fainted[side]
                             and not self.forced_out[side] and not rec.switch)
                if voluntary:
                    rec.switch = True
                    rec.acted = True
                    self.vol_switches[side] += 1
            self.forced_out[side] = False
            if side == "me" and kind == "switch" and self.turn_no >= 1:
                self.last_action = ("switch",)
            return

        side = self._side(ev[2])
        if kind == "move":
            if side is None or len(ev) < 4:
                return
            self.mover = side
            src = _has_from(ev, 4)
            if cur is None:
                return
            cur.active = True
            rec = self._rec(side)
            if "[miss]" in ev[4:] or "[notarget]" in ev[4:]:
                rec.fail = True
            if src is not None and src != "lockedmove":
                return  # called / copied move (Sleep Talk, Dancer, Magic Bounce...): not a decision
            if rec.move or rec.cant:
                return
            move_id = to_id_str(ev[3])
            ident = _ident(ev[2])
            bp, is_status, _ = _move_data(move_id)
            rec.move = True
            rec.acted = True
            rec.move_id = move_id
            rec.mover = ident
            rec.bp = 0.0 if is_status else bp
            if cur.first is None:
                cur.first = side
            if side == "me":
                self.uses[(ident, move_id)] = self.uses.get((ident, move_id), 0) + 1
                self.last_action = ("move", ident, move_id, self.tera_this_turn)
            else:
                self.opp_moves.add((ident, move_id))
            return
        if kind == "cant":
            self.mover = None
            if side is None or cur is None:
                return
            cur.active = True
            rec = self._rec(side)
            if not rec.move and not rec.cant:
                rec.cant = True
                rec.acted = True
                if cur.first is None:
                    cur.first = side
                if side == "me":
                    self.last_action = ("other",)
            return
        if kind == "-terastallize":
            if side is None:
                return
            if not self.tera_turn[side]:
                self.tera_turn[side] = max(self.turn_no, 1)
            if cur is not None:
                cur.active = True
                self._rec(side).tera = True
            if side == "me":
                self.tera_this_turn = True
            return
        if kind in ("-damage", "-heal", "-sethp"):
            if side is None or len(ev) < 4:
                return
            ident = _ident(ev[2])
            frac = _hp_fraction(ev[3])
            if frac is None:
                return
            old = self.hp.get(ident, 1.0)
            self.hp[ident] = frac
            if kind != "-damage" or cur is None:
                return
            cur.active = True
            mover = self.mover
            if mover is not None and mover != side and _has_from(ev, 4) is None and old > frac:
                self._rec(mover).dmg += old - frac
                self.dmg[mover] += old - frac
            return
        if kind == "faint":
            if side is None:
                return
            self.hp[_ident(ev[2])] = 0.0
            other = "opp" if side == "me" else "me"
            self.kos[other] += 1
            if cur is not None:
                cur.active = True
                cur.fainted[side] = True
                self._rec(other).ko = True
            return
        if kind in _EFF:
            mover = self.mover
            if side is not None and mover is not None and mover != side and cur is not None:
                self._rec(mover).eff = _EFF[kind]
            return
        if kind == "-crit":
            if side is None:
                return
            other = "opp" if side == "me" else "me"
            self.crits[other] += 1
            if cur is not None:
                self._rec(other).crit = True
            return
        if kind == "-miss":
            if side is not None and cur is not None:  # '-miss|USER|TARGET'
                self._rec(side).fail = True
            return
        if kind == "-fail":
            if self.mover is not None and cur is not None:
                self._rec(self.mover).fail = True
            return
        if kind == "-activate":
            text = " ".join(ev[3:]).lower()
            mover = self.mover
            if (side is not None and mover is not None and mover != side and cur is not None
                    and any(w in text for w in _PROTECT_WORDS)):
                self._rec(mover).fail = True
            elif side is not None and any(w in text for w in _FORCED_OUT_ABILITIES):
                self.forced_out[side] = True
            return
        if kind == "-enditem":
            if side is not None and len(ev) > 3 and ev[3].lower() in _FORCED_OUT_ITEMS:
                self.forced_out[side] = True
            return
        if kind == "-status":
            if side is None or cur is None:
                return
            src = _has_from(ev, 4) or ""
            if src.startswith("item") or "rest" in src:
                return  # Flame/Toxic Orb, Rest: self-inflicted
            other = "opp" if side == "me" else "me"
            self._rec(other).status = True
            return
        if kind in ("-sidestart", "-sideend"):
            if side is None or len(ev) < 4:
                return
            cond = ev[3].lower()
            if not any(h in cond for h in _HAZARDS):
                return
            if kind == "-sidestart":
                setter = "opp" if side == "me" else "me"
                self.hazards_set[setter] += 1
            else:
                self.cleared[side] += 1
            return

    # ----------------------------------------------------------------- output
    def features(self, battle: Any) -> List[float]:
        out = [0.0] * HISTORY_V3_SIZE
        try:
            slots = [m.id for m in move_slots(battle)][:N_SLOTS]
        except Exception:
            slots = []
        my_active = self.active["me"]

        def slot_of(ident: Optional[str], move_id: Optional[str]) -> int:
            if ident is None or ident != my_active or move_id not in slots:
                return -1
            return slots.index(move_id)

        try:
            self._recent(out, slot_of)
        except Exception:
            out[RECENT_SLICE] = [0.0] * (RECENT_SLICE.stop - RECENT_SLICE.start)
        try:
            self._aggregates(out, battle, slots)
        except Exception:
            out[AGG_SLICE] = [0.0] * AGG_FEATS
        try:
            self._last_action(out, slot_of)
        except Exception:
            out[LAST_ACTION_SLICE] = [0.0] * LAST_ACTION_FEATS
        return [max(-1.0, min(1.0, v)) if v == v else 0.0 for v in out]  # clip, NaN -> 0

    def _recent(self, out: List[float], slot_of) -> None:
        turns: List[Tuple[_Turn, bool]] = []
        if self.cur is not None and self.cur.active:
            turns.append((self.cur, True))
        for t in reversed(self.done):
            turns.append((t, False))
        for k, (t, in_prog) in enumerate(turns[:K_TURNS]):
            b = k * TURN_FEATS
            me, opp = t.me, t.opp
            out[b + 0] = 1.0
            out[b + 1] = 1.0 if in_prog else 0.0
            out[b + 2] = float(me.move)
            out[b + 3] = float(me.switch)
            out[b + 4] = float(me.cant)
            out[b + 5] = float(me.tera)
            out[b + 6] = me.bp
            s = slot_of(me.mover, me.move_id) if me.move else -1
            if s >= 0:
                out[b + 7 + s] = 1.0
            out[b + 11] = me.eff
            out[b + 12] = float(me.fail)
            out[b + 13] = min(me.dmg, 1.0)
            out[b + 14] = float(me.ko)
            out[b + 15] = float(opp.move)
            out[b + 16] = float(opp.switch)
            out[b + 17] = float(opp.cant)
            out[b + 18] = float(opp.tera)
            out[b + 19] = opp.bp
            out[b + 20] = opp.eff
            out[b + 21] = float(opp.fail)
            out[b + 22] = min(opp.dmg, 1.0)
            out[b + 23] = float(opp.ko)
            if me.acted and opp.acted and not me.switch and not opp.switch:
                out[b + 24] = 1.0 if t.first == "me" else -1.0
            out[b + 25] = float(me.crit) - float(opp.crit)
            out[b + 26] = float(me.status) - float(opp.status)

    def _aggregates(self, out: List[float], battle: Any, slots: List[str]) -> None:
        b = AGG_SLICE.start
        turns = max(self.turn_no, 1)
        out[b + 0] = min(self.turn_no / 60.0, 1.0)
        out[b + 1] = min(self.vol_switches["me"] / turns, 1.0)
        out[b + 2] = min(self.vol_switches["opp"] / turns, 1.0)
        # Opponent switch tendency, conditioned on whether we threatened it with a SE move.
        n_t = k_t = n_s = k_s = 0
        team = getattr(battle, "team", None) or {}
        move_types: Dict[str, Tuple[str, ...]] = {}
        for (my_ident, opp_species), (n, k) in self.matchups.items():
            if my_ident is None or opp_species is None:
                continue
            if my_ident not in move_types:
                mon = team.get(my_ident)
                types = set()
                if mon is not None:
                    for mid in getattr(mon, "moves", {}) or {}:
                        bp, is_status, mtype = _move_data(mid)
                        if bp > 0 and not is_status and mtype:
                            types.add(mtype)
                move_types[my_ident] = tuple(sorted(types))
            if _has_se(move_types[my_ident], opp_species):
                n_t += n
                k_t += k
            else:
                n_s += n
                k_s += k
        out[b + 3] = (k_t + 1.0) / (n_t + 2.0)
        out[b + 4] = min(n_t / 10.0, 1.0)
        out[b + 5] = (k_s + 1.0) / (n_s + 2.0)
        out[b + 6] = min(self.crits["me"] / 4.0, 1.0)
        out[b + 7] = min(self.crits["opp"] / 4.0, 1.0)
        out[b + 8] = min(self.dmg["me"] / 10.0, 1.0)
        out[b + 9] = min(self.dmg["opp"] / 10.0, 1.0)
        out[b + 10] = min(self.kos["me"] / 6.0, 1.0)
        out[b + 11] = min(self.kos["opp"] / 6.0, 1.0)
        my_active = self.active["me"]
        if my_active is not None:
            for i, mid in enumerate(slots):
                out[b + 12 + i] = min(self.uses.get((my_active, mid), 0) / 10.0, 1.0)
        out[b + 16] = 1.0 if self.tera_turn["me"] else 0.0
        out[b + 17] = min(self.tera_turn["me"] / 60.0, 1.0)
        out[b + 18] = 1.0 if self.tera_turn["opp"] else 0.0
        out[b + 19] = min(self.tera_turn["opp"] / 60.0, 1.0)
        out[b + 20] = min(len(self.opp_moves) / 24.0, 1.0)
        out[b + 21] = min(self.hazards_set["opp"] / 4.0, 1.0)
        out[b + 22] = min(self.hazards_set["me"] / 4.0, 1.0)
        out[b + 23] = min(self.cleared["me"] / 4.0, 1.0)
        out[b + 24] = min(self.cleared["opp"] / 4.0, 1.0)
        if self.active["me"] is not None:
            out[b + 25] = min((self.turn_no - self.switch_in_turn["me"]) / 10.0, 1.0)
        if self.active["opp"] is not None:
            out[b + 26] = min((self.turn_no - self.switch_in_turn["opp"]) / 10.0, 1.0)

    def _last_action(self, out: List[float], slot_of) -> None:
        b = LAST_ACTION_SLICE.start
        la = self.last_action
        if la is None:
            return
        if la[0] == "move":
            s = slot_of(la[1], la[2])
            if s >= 0:
                out[b + s] = 1.0
            else:
                out[b + 6] = 1.0
            if la[3]:
                out[b + 4] = 1.0
        elif la[0] == "switch":
            out[b + 5] = 1.0
        else:
            out[b + 6] = 1.0


# --------------------------------------------------------------------------- #
# Cache + public API
# --------------------------------------------------------------------------- #
_CACHE: "OrderedDict[Tuple[Any, ...], _HistoryState]" = OrderedDict()


def _state_for(battle: Any, data: List[List[str]], role: str, use_cache: bool) -> _HistoryState:
    if not use_cache:
        st = _HistoryState(role)
        st.feed(data)
        return st
    key = (getattr(battle, "battle_tag", None), id(battle), id(data), role)
    st = _CACHE.get(key)
    if st is not None:
        n = st.n
        valid = (n <= len(data) and (n == 0 or (data[0] is st.first_msg and data[n - 1] is st.last_msg)))
        if not valid:
            st = None
    if st is None:
        st = _HistoryState(role)
        _CACHE[key] = st
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    st.feed(data)
    return st


def history_block(battle: Any, use_cache: bool = True) -> List[float]:
    """HISTORY_V3_SIZE floats in [-1, 1] summarising the battle so far from our perspective. Never raises."""
    try:
        role = getattr(battle, "player_role", None)
        data = getattr(battle, "_replay_data", None)
        if role not in ("p1", "p2") or not isinstance(data, list) or not data:
            return [0.0] * HISTORY_V3_SIZE
        st = _state_for(battle, data, role, use_cache)
        out = st.features(battle)
        if len(out) != HISTORY_V3_SIZE:
            return [0.0] * HISTORY_V3_SIZE
        return out
    except Exception:
        return [0.0] * HISTORY_V3_SIZE


def clear_cache() -> None:
    _CACHE.clear()


def decode(vec: List[float]) -> Dict[str, Any]:
    """Human-readable view of a history_block vector (for debugging / eyeballing)."""
    recent = []
    for k in range(K_TURNS):
        rec = vec[k * TURN_FEATS:(k + 1) * TURN_FEATS]
        if rec[0] <= 0:
            break
        recent.append({name: round(v, 3) for name, v in zip(TURN_FIELDS, rec) if v != 0})
    agg = {name: round(v, 3) for name, v in zip(AGG_FIELDS, vec[AGG_SLICE])}
    last = [name for name, v in zip(LAST_ACTION_FIELDS, vec[LAST_ACTION_SLICE]) if v > 0]
    return {"recent": recent, "aggregates": agg, "last_action": last}


__all__ = ["HISTORY_V3_SIZE", "history_block", "decode", "clear_cache", "TURN_FIELDS", "AGG_FIELDS",
           "LAST_ACTION_FIELDS", "K_TURNS", "TURN_FEATS", "RECENT_SLICE", "AGG_SLICE", "LAST_ACTION_SLICE"]
