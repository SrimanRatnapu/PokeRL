"""Evaluate a trained checkpoint (or a scripted bot) against baseline opponents.

Every player here is a normal *networked* poke-env Player that talks to the local
Showdown server (localhost:8000), so this measures exactly what a human opponent
would face.

Examples (run from the project root, server running):
    .venv/bin/python -m pokebot.evaluate --model checkpoints/run1/final.zip --battles 200
    .venv/bin/python -m pokebot.evaluate --model heuristic --battles 20      # sanity baseline
    .venv/bin/python -m pokebot.evaluate --model ckpt_a.zip --compare ckpt_b.zip --battles 100
    .venv/bin/python -m pokebot.evaluate --model ckpt.zip --opponents heuristic --save-replays replays/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from poke_env import AccountConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from bots.heuristic_bot import HeuristicBot
from pokebot.config import BATTLE_FORMAT

# Baseline opponents, keyed by the names used in --opponents (same names as
# make_opponent in pokebot/env.py).
BASELINES: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "maxpower": MaxBasePowerPlayer,
    "heuristic": HeuristicBot,
    "simple_heuristic": SimpleHeuristicsPlayer,
}
DEFAULT_OPPONENTS = "random,maxpower,heuristic,simple_heuristic"

# Short username prefixes (Showdown names are max 18 chars; a random 5-char
# suffix is appended so evaluation never collides with training players).
_NAME_PREFIX = {
    "random": "EvRandom",
    "maxpower": "EvMaxPower",
    "heuristic": "EvHeuristic",
    "simple_heuristic": "EvSimpleHeur",
}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class MatchResult:
    opponent: str
    battles: int
    wins: int
    losses: int
    ties: int
    mean_turns: float
    seconds: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.battles if self.battles else 0.0


# --------------------------------------------------------------------------- #
# Player construction
# --------------------------------------------------------------------------- #
def _account(prefix: str) -> AccountConfiguration:
    return AccountConfiguration.generate(prefix[:12], rand=True)


def build_player(
    spec: str,
    *,
    name_prefix: str,
    concurrency: int,
    deterministic: bool = True,
    save_replays: str | bool = False,
    log_level: int = logging.WARNING,
) -> Player:
    """Build a networked player from a spec.

    spec is either a baseline name (random / maxpower / heuristic /
    simple_heuristic) or a path to a MaskablePPO checkpoint (.zip).
    """
    common = dict(
        account_configuration=_account(name_prefix),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        save_replays=save_replays,
        log_level=log_level,
    )
    if spec in BASELINES:
        return BASELINES[spec](**common)

    # Imported lazily so the baseline modes work even before env.py exists.
    from pokebot.env import PolicyPlayer

    return PolicyPlayer(model_path=spec, deterministic=deterministic, **common)


def check_model_compat(player: Player, label: str) -> bool:
    """Warn loudly if a checkpoint's observation size matches no known feature version.

    PolicyPlayer picks the feature version (v1, v2, ...) from the model's observation
    size, so any registered version is fine. An unknown size makes PolicyPlayer fall
    back to random moves on every turn, which would silently produce meaningless results.
    """
    model = getattr(player, "model", None)
    if model is None:
        return True
    from pokebot.feature_sets import FEATURE_SETS, version_for_model

    shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    version = version_for_model(model)
    if version is None:
        known = ", ".join(f"{v}={size}" for v, (size, _) in FEATURE_SETS.items())
        print(
            f"WARNING: {label} expects observations of shape {shape}, which matches no known feature "
            f"version ({known}). The bot will fall back to random moves.",
            file=sys.stderr,
            flush=True,
        )
        return False
    print(f"{label}: feature set {version} ({shape[0]} dims)", flush=True)
    return True


def build_opponent(kind: str, concurrency: int, log_level: int) -> Player:
    if kind not in BASELINES:
        raise SystemExit(f"Unknown opponent '{kind}'. Choose from: {', '.join(BASELINES)}")
    return BASELINES[kind](
        account_configuration=_account(_NAME_PREFIX[kind]),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        log_level=log_level,
    )


# --------------------------------------------------------------------------- #
# Running matches
# --------------------------------------------------------------------------- #
async def play_match(agent: Player, opponent: Player, label: str, n_battles: int) -> MatchResult:
    """Play n_battles between agent and opponent and summarise the agent's results."""
    before = set(agent.battles)
    t0 = time.perf_counter()
    await agent.battle_against(opponent, n_battles=n_battles)
    elapsed = time.perf_counter() - t0

    new = [b for tag, b in agent.battles.items() if tag not in before and b.finished]
    wins = sum(1 for b in new if b.won)
    losses = sum(1 for b in new if b.lost)
    turns = [b.turn for b in new]
    return MatchResult(
        opponent=label,
        battles=len(new),
        wins=wins,
        losses=losses,
        ties=len(new) - wins - losses,
        mean_turns=sum(turns) / len(turns) if turns else 0.0,
        seconds=elapsed,
    )


async def _close(player: Player) -> None:
    try:
        await player.ps_client.stop_listening()
    except Exception:  # already closed / never connected
        pass


def print_table(agent_label: str, results: list[MatchResult]) -> None:
    print()
    print(f"Agent: {agent_label}   format: {BATTLE_FORMAT}")
    header = f"{'opponent':<34} {'wins/battles':>13} {'win rate':>9} {'95% CI (Wilson)':>17} {'turns':>7} {'ties':>5} {'time':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        lo, hi = wilson_ci(r.wins, r.battles)
        print(
            f"{r.opponent:<34} {f'{r.wins}/{r.battles}':>13} {r.win_rate:>9.1%} "
            f"{f'[{lo:.1%}, {hi:.1%}]':>17} {r.mean_turns:>7.1f} {r.ties:>5} {r.seconds:>6.1f}s"
        )
    print()


async def main_async(args: argparse.Namespace) -> list[MatchResult]:
    log_level = logging.DEBUG if args.debug else logging.ERROR
    # The agent logs at WARNING so PolicyPlayer's "falling back to a random move" is visible.
    agent_log_level = logging.DEBUG if args.debug else logging.WARNING
    deterministic = not args.stochastic

    agent = build_player(
        args.model,
        name_prefix="EvAgent",
        concurrency=args.concurrency,
        deterministic=deterministic,
        save_replays=args.save_replays or False,
        log_level=agent_log_level,
    )
    check_model_compat(agent, args.model)

    if args.opponents is None:
        opponents = [] if args.compare else DEFAULT_OPPONENTS.split(",")
    else:
        opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]

    results: list[MatchResult] = []
    try:
        if args.compare:
            other = build_player(
                args.compare,
                name_prefix="EvCompare",
                concurrency=args.concurrency,
                deterministic=deterministic,
                log_level=agent_log_level,
            )
            check_model_compat(other, args.compare)
            print(f"Playing {args.battles} battles: {args.model}  vs  {args.compare} ...", flush=True)
            try:
                results.append(await play_match(agent, other, f"compare: {args.compare}", args.battles))
            finally:
                await _close(other)

        for kind in opponents:
            opp = build_opponent(kind, args.concurrency, log_level)
            print(f"Playing {args.battles} battles vs {kind} ...", flush=True)
            try:
                results.append(await play_match(agent, opp, kind, args.battles))
            finally:
                await _close(opp)
            r = results[-1]
            print(f"  -> {r.wins}/{r.battles} wins ({r.win_rate:.1%}), {r.seconds:.1f}s", flush=True)
    finally:
        await _close(agent)

    print_table(args.model, results)
    if args.save_replays:
        print(f"Replays saved to {args.save_replays}/")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint against baseline bots on the local Showdown server.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path to a MaskablePPO .zip checkpoint, or a baseline name "
        f"({', '.join(BASELINES)}) for a sanity check without a model.",
    )
    p.add_argument("--battles", type=int, default=200, help="battles per opponent (default 200)")
    p.add_argument(
        "--opponents",
        default=None,
        help=f"comma-separated baselines (default: {DEFAULT_OPPONENTS}; none when --compare is given)",
    )
    p.add_argument("--concurrency", type=int, default=10, help="max concurrent battles (default 10)")
    p.add_argument("--compare", metavar="PATH2", default=None, help="pit --model against this second checkpoint")
    p.add_argument("--save-replays", metavar="DIR", default=None, help="save the agent's battle replays (HTML) here")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--debug", action="store_true", help="verbose poke-env logging")
    args = p.parse_args(argv)
    if args.battles < 1:
        p.error("--battles must be >= 1")
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    for flag, spec in (("--model", args.model), ("--compare", args.compare)):
        if spec is not None and not is_valid_spec(spec):
            p.error(f"{flag} {spec!r}: not a baseline name ({', '.join(BASELINES)}) and no such checkpoint file")
    return args


def is_valid_spec(spec: str) -> bool:
    """A baseline name, or an existing checkpoint path (SB3 also accepts it without '.zip')."""
    return spec in BASELINES or Path(spec).is_file() or Path(spec + ".zip").is_file()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
