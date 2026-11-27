"""Put the bot online on the local Showdown server so a human can battle it.

Examples (run from the project root, server running at localhost:8000):
    .venv/bin/python -m pokebot.play --model checkpoints/run1/final.zip
    .venv/bin/python -m pokebot.play --model checkpoints/run1/final.zip --n-challenges 0 --verbose
    .venv/bin/python -m pokebot.play --model heuristic                      # no model needed
    .venv/bin/python -m pokebot.play --model ckpt.zip --challenge YourName  # bot challenges you

By default the bot waits for challenges. Open http://localhost:8000, pick any
name (no password with --no-security) and challenge the bot to
"[Gen 9] Random Battle".
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from poke_env import AccountConfiguration
from poke_env.player import Player

from pokebot.config import BATTLE_FORMAT
from pokebot.evaluate import BASELINES, check_model_compat, is_valid_spec

SERVER_URL = "http://localhost:8000"
LOGIN_TIMEOUT_S = 15.0


# --------------------------------------------------------------------------- #
# Pretty-printing helpers
# --------------------------------------------------------------------------- #
def order_text(order) -> str:
    msg = getattr(order, "message", None) or str(order)
    return msg.replace("/choose ", "")


def describe_action(action: int, battle) -> str:
    """Human-readable label for an index in the 26-action SinglesEnv space."""
    try:
        if action < 6:
            mon = list(battle.team.values())[action]
            return f"switch {mon.species}"
        if battle.active_pokemon is None:
            return f"action {action}"
        moves = list(battle.active_pokemon.moves.values())[:4]
        known_ids = {m.id for m in moves}
        if len(battle.available_moves) == 1 and battle.available_moves[0].id not in known_ids:
            moves = [battle.available_moves[0]]  # e.g. Struggle / locked move -> slot 0
        if 6 <= action <= 9:
            return f"move {moves[action - 6].id}"
        if 22 <= action <= 25:
            return f"move {moves[action - 22].id} +tera"
    except (IndexError, AttributeError):
        pass
    return f"action {action}"


def log_policy_probs(model, battle, top_k: int = 3) -> str:
    """Top-k masked action probabilities of a MaskablePPO policy for this battle."""
    import numpy as np
    import torch as th
    from poke_env.environment import SinglesEnv

    from pokebot.feature_sets import get_feature_set, version_for_model

    version = version_for_model(model)
    if version is None:
        raise ValueError(f"model observation shape {model.observation_space.shape} matches no feature version")
    obs = get_feature_set(version)[1](battle)
    mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=bool)
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=mask[None, :])
        probs = dist.distribution.probs[0].cpu().numpy()
    legal = [int(a) for a in np.argsort(-probs) if mask[a]][:top_k]
    return ", ".join(f"{describe_action(a, battle)} {probs[a]:.0%}" for a in legal)


# --------------------------------------------------------------------------- #
# Player construction
# --------------------------------------------------------------------------- #
def _with_reporting(cls: type[Player], verbose: bool, model=None) -> type[Player]:
    """Subclass cls so it prints battle results (and, if verbose, each turn's choice)."""

    class Reporting(cls):  # type: ignore[misc, valid-type]
        def choose_move(self, battle):
            order = super().choose_move(battle)
            if verbose:
                line = f"[{battle.battle_tag}] turn {battle.turn:>3}: {order_text(order)}"
                if model is not None:
                    try:
                        line += f"   | top-3: {log_policy_probs(model, battle)}"
                    except Exception as e:  # never let logging break a live battle
                        line += f"   | (probs unavailable: {e})"
                print(line, flush=True)
            return order

        def _battle_finished_callback(self, battle):
            outcome = "WON" if battle.won else ("LOST" if battle.lost else "TIED")
            print(
                f"Battle {battle.battle_tag} vs {battle.opponent_username}: {outcome} in {battle.turn} turns",
                flush=True,
            )
            super()._battle_finished_callback(battle)

    Reporting.__name__ = f"Reporting{cls.__name__}"
    return Reporting


def build_bot(args: argparse.Namespace) -> Player:
    common = dict(
        account_configuration=AccountConfiguration(args.username, None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=args.concurrency,
        save_replays=args.save_replays or False,
        # WARNING so PolicyPlayer's "falling back to a random move" messages are visible.
        log_level=logging.DEBUG if args.debug else logging.WARNING,
    )
    if args.model in BASELINES:
        cls = _with_reporting(BASELINES[args.model], args.verbose)
        if args.model == "heuristic" and args.verbose:
            common["verbose"] = True  # HeuristicBot prints its full state every turn
        return cls(**common)

    from pokebot.env import PolicyPlayer

    if args.verbose:
        # Load the model ourselves so we can also inspect its action distribution.
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(args.model, device="cpu", custom_objects={"seed": None})
        cls = _with_reporting(PolicyPlayer, True, model=model)
        return cls(model=model, deterministic=not args.stochastic, **common)

    cls = _with_reporting(PolicyPlayer, False)
    return cls(model_path=args.model, deterministic=not args.stochastic, **common)


async def wait_for_login(player: Player, timeout: float = LOGIN_TIMEOUT_S) -> bool:
    # logged_in is an Event on poke-env's own loop, so poll it rather than await it.
    deadline = time.monotonic() + timeout
    while not player.ps_client.logged_in.is_set():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.1)
    return True


def print_instructions(args: argparse.Namespace, name: str) -> None:
    mode = "forever" if args.n_challenges == 0 else f"{args.n_challenges} battle(s)"
    print()
    print(f"Bot '{name}' is online on {SERVER_URL}  (model: {args.model}, {mode})")
    if args.challenge:
        print(f"Sending challenge(s) to '{args.challenge}' in [Gen 9] Random Battle.")
        print(f"  -> In your browser at {SERVER_URL}, logged in as '{args.challenge}', click Accept.")
    else:
        print("To battle it:")
        print(f"  1. Open {SERVER_URL} in your browser.")
        print("  2. Click 'Choose name' (top right) and pick any name - no password needed.")
        print(f"  3. Click 'Find a user', type '{name}', then 'Challenge'.")
        print("  4. Set the format to '[Gen 9] Random Battle' and click 'Challenge'.")
        print("     (Challenges in any other format are ignored.)")
    print("Press Ctrl+C to quit.")
    print(flush=True)


async def main_async(args: argparse.Namespace) -> int:
    bot = build_bot(args)
    check_model_compat(bot, args.model)
    if not await wait_for_login(bot):
        print(
            f"Could not log in as '{args.username}' within {LOGIN_TIMEOUT_S:.0f}s. Is the Showdown "
            f"server running at {SERVER_URL}, and is that name free? Try --username OtherName.",
            file=sys.stderr,
        )
        return 1
    print_instructions(args, bot.username)

    try:
        if args.challenge:
            if args.n_challenges == 0:
                while True:
                    await bot.send_challenges(args.challenge, 1)
            else:
                await bot.send_challenges(args.challenge, args.n_challenges)
        elif args.n_challenges == 0:
            while True:
                await bot.accept_challenges(None, 1)
        else:
            await bot.accept_challenges(None, args.n_challenges)
    finally:
        finished = bot.n_finished_battles
        if finished:
            print(f"\nDone: {bot.n_won_battles}/{finished} battles won.")
        if args.save_replays and finished:
            print(f"Replays saved to {args.save_replays}/")
        try:
            await bot.ps_client.stop_listening()
        except Exception:
            pass
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log the bot into the local Showdown server to play humans.")
    p.add_argument(
        "--model",
        required=True,
        help=f"MaskablePPO .zip checkpoint, or a baseline name ({', '.join(BASELINES)})",
    )
    p.add_argument("--username", default="PokeBotRL", help="bot's name on the server (max 18 chars)")
    p.add_argument("--n-challenges", type=int, default=1, help="battles to play before exiting; 0 = forever")
    p.add_argument("--challenge", metavar="USER", default=None, help="send challenges to USER instead of waiting")
    p.add_argument("--verbose", action="store_true", help="log each turn's action (and top-3 policy probabilities)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--concurrency", type=int, default=1, help="max simultaneous battles (default 1)")
    p.add_argument("--save-replays", metavar="DIR", default=None, help="save replays (HTML) here")
    p.add_argument("--debug", action="store_true", help="verbose poke-env logging")
    args = p.parse_args(argv)
    if not is_valid_spec(args.model):
        p.error(f"--model {args.model!r}: not a baseline name and no such checkpoint file")
    if len(args.username) > 18:
        p.error("--username must be at most 18 characters")
    if args.n_challenges < 0:
        p.error("--n-challenges must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nBye.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
