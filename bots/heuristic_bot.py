"""
Week 1: a hand-written heuristic bot, before any RL.

Why start here? An RL agent only "sees" the numbers you extract from the
battle. Writing a heuristic forces you to learn the Battle API, which is
exactly what you'll turn into an observation vector in Week 2.

Run (with the Showdown server running on localhost:8000):
    .venv/bin/python bots/heuristic_bot.py --battles 100
    .venv/bin/python bots/heuristic_bot.py --battles 1 --verbose
"""

import argparse
import asyncio

from poke_env.battle import Battle, MoveCategory, Pokemon
from poke_env.battle.move import Move
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer


class HeuristicBot(Player):
    """Picks the move with the highest estimated damage, and switches out of bad matchups."""

    # If our best move's score is below this AND a bench mon has a better
    # matchup, switch. Tune it and watch the win rate change.
    SWITCH_THRESHOLD = 40.0

    def __init__(self, *args, verbose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Scoring helpers: these are the "features" you'll reuse for RL later
    # ------------------------------------------------------------------ #
    @staticmethod
    def move_score(move: Move, attacker: Pokemon, defender: Pokemon) -> float:
        """Rough expected damage of a move. Not real damage, just a relative score."""
        if move.category == MoveCategory.STATUS:
            return 0.0

        power = move.base_power * move.expected_hits    # multi-hit moves
        stab = 1.5 if move.type in attacker.types else 1.0
        effectiveness = defender.damage_multiplier(move)  # 0, 0.25, 0.5, 1, 2, 4
        accuracy = move.accuracy                          # 0..1

        # Physical moves use Atk vs Def, special use SpA vs SpD (base stats).
        if move.category == MoveCategory.PHYSICAL:
            stat_ratio = attacker.base_stats["atk"] / defender.base_stats["def"]
        else:
            stat_ratio = attacker.base_stats["spa"] / defender.base_stats["spd"]

        return power * stab * effectiveness * accuracy * stat_ratio

    @staticmethod
    def matchup_score(mon: Pokemon, opponent: Pokemon) -> float:
        """Positive = mon is favoured. Offense (our types vs them) minus defense (theirs vs us)."""
        offense = max(opponent.damage_multiplier(t) for t in mon.types)
        defense = max(mon.damage_multiplier(t) for t in opponent.types)
        return offense - defense + 0.5 * mon.current_hp_fraction

    # ------------------------------------------------------------------ #
    # The one method every poke-env Player must implement
    # ------------------------------------------------------------------ #
    def choose_move(self, battle: Battle):
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        if self.verbose:
            self.print_state(battle)

        # Forced switch (our mon fainted) or no info yet: switch to best matchup.
        if not battle.available_moves or me is None or opp is None:
            if battle.available_switches and opp is not None:
                best = max(battle.available_switches, key=lambda m: self.matchup_score(m, opp))
                return self.create_order(best)
            return self.choose_random_move(battle)

        # 1) Score every legal move.
        scored = [(self.move_score(m, me, opp), m) for m in battle.available_moves]
        best_score, best_move = max(scored, key=lambda x: x[0])

        # 2) Consider switching if our attacks are weak and someone on the bench is better.
        if best_score < self.SWITCH_THRESHOLD and battle.available_switches:
            best_switch = max(battle.available_switches, key=lambda m: self.matchup_score(m, opp))
            if self.matchup_score(best_switch, opp) > self.matchup_score(me, opp) + 1:
                return self.create_order(best_switch)

        # 3) Terastallize if our tera type gives STAB on a super-effective move.
        tera = (
            battle.can_tera
            and me.tera_type is not None
            and best_move.type == me.tera_type
            and opp.damage_multiplier(best_move) >= 2
        )
        return self.create_order(best_move, terastallize=tera)

    # ------------------------------------------------------------------ #
    # Debug printout: read this to learn what a Battle object contains
    # ------------------------------------------------------------------ #
    def print_state(self, battle: Battle):
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        print(f"\n=== Turn {battle.turn} ===")
        if me:
            print(f"  Me:  {me.species:<14} HP {me.current_hp_fraction:4.0%}  types={[t.name for t in me.types]}  boosts={ {k: v for k, v in me.boosts.items() if v} }")
        if opp:
            print(f"  Opp: {opp.species:<14} HP {opp.current_hp_fraction:4.0%}  types={[t.name for t in opp.types]}")
        if me and opp:
            for m in battle.available_moves:
                print(f"    move {m.id:<16} bp={m.base_power:<4} {m.type.name:<9} x{opp.damage_multiplier(m):<4} score={self.move_score(m, me, opp):6.1f}")
            for s in battle.available_switches:
                print(f"    switch {s.species:<14} matchup={self.matchup_score(s, opp):+.2f}")


async def main(n_battles: int, verbose: bool):
    bot = HeuristicBot(battle_format="gen9randombattle", max_concurrent_battles=10, verbose=verbose)

    for Opponent in (RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer):
        opponent = Opponent(battle_format="gen9randombattle", max_concurrent_battles=10)
        wins_before = bot.n_won_battles
        await bot.battle_against(opponent, n_battles=n_battles)
        wins = bot.n_won_battles - wins_before
        print(f"vs {Opponent.__name__:<24} {wins}/{n_battles} wins ({wins / n_battles:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--verbose", action="store_true", help="print the battle state every turn")
    args = parser.parse_args()
    asyncio.run(main(args.battles, args.verbose))
