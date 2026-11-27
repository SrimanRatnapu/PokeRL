"""
Behavior cloning (BC) warm start: "SFT before RL" for the Pokémon bot.

1) collect: a teacher bot (default poke-env's SimpleHeuristicsPlayer) plays networked
   battles against a mix of opponents; every decision is recorded as
   (observation, action mask, teacher action, value target).
2) train:   a fresh MaskablePPO policy (same architecture as pokebot.train) is trained
   with supervised learning:
     - policy head: cross-entropy toward the teacher's action (masked distribution)
     - value head:  regression onto the discounted shaped return actually obtained
   The result is a normal MaskablePPO .zip that `pokebot.train --init-from` can
   fine-tune with PPO, and that evaluate.py / play.py can run directly.

Usage (from project root, server running):
    .venv/bin/python -m pokebot.bc collect --battles 3000 --out data/bc_v2.npz
    .venv/bin/python -m pokebot.bc train --data data/bc_v2.npz --out checkpoints/bc_v2/bc.zip
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.spaces import Box, Discrete

from pokebot.config import ACTION_SPACE_SIZE, BATTLE_FORMAT, RewardConfig
from pokebot.feature_sets import get_feature_set

GAMMA = 0.99
POLICY_KWARGS = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))  # must match pokebot.train


def _rand_name(prefix: str) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def make_recording_teacher(teacher_cls, feature_version: str):
    from poke_env.environment import SinglesEnv
    from poke_env.battle import Battle

    from pokebot.env import PokeBotEnv

    _, embed = get_feature_set(feature_version)
    reward_cfg = RewardConfig()

    class RecordingTeacher(teacher_cls):
        """Plays like `teacher_cls`, logging (obs, mask, action, potential) per decision."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps = defaultdict(list)   # battle_tag -> [(obs, mask, action, phi)]
            self.finished: list[tuple[list, float, int]] = []  # (steps, final_phi, outcome)
            self.skipped = 0

        def choose_move(self, battle):
            order = super().choose_move(battle)
            if isinstance(battle, Battle):
                try:
                    mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=bool)
                    action = int(SinglesEnv.order_to_action(order, battle, fake=False, strict=False))
                    if 0 <= action < ACTION_SPACE_SIZE and mask[action] and mask.sum() > 1:
                        phi = PokeBotEnv._potential(battle, reward_cfg)
                        self.steps[battle.battle_tag].append((embed(battle), mask, action, phi))
                    else:
                        self.skipped += 1  # forced (only one legal action) or unmappable
                except Exception:
                    self.skipped += 1
            return order

        def _battle_finished_callback(self, battle):
            steps = self.steps.pop(battle.battle_tag, [])
            outcome = 1 if battle.won else (-1 if battle.lost else 0)
            if steps:
                self.finished.append((steps, PokeBotEnv._potential(battle, reward_cfg), outcome))
            super()._battle_finished_callback(battle)

    return RecordingTeacher


def returns_for_episode(phis: list[float], final_phi: float, outcome: int, cfg: RewardConfig) -> np.ndarray:
    """Discounted shaped return from each decision, matching PokeBotEnv.calc_reward (shaping_scale=1)."""
    nxt = phis[1:] + [final_phi]
    rewards = [cfg.shaping_scale * (b - a) for a, b in zip(phis, nxt)]
    rewards[-1] += cfg.victory * outcome
    g, out = 0.0, []
    for r in reversed(rewards):
        g = r + GAMMA * g
        out.append(g)
    return np.asarray(out[::-1], dtype=np.float32)


async def _collect(args) -> None:
    from poke_env import AccountConfiguration
    from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

    from bots.heuristic_bot import HeuristicBot

    teachers = {"simple_heuristic": SimpleHeuristicsPlayer, "heuristic": HeuristicBot}
    Teacher = make_recording_teacher(teachers[args.teacher], args.features)
    teacher = Teacher(account_configuration=AccountConfiguration(_rand_name("BCT"), None),
                      battle_format=BATTLE_FORMAT, max_concurrent_battles=args.concurrency)

    opp_classes = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer,
                   "heuristic": HeuristicBot, "simple_heuristic": SimpleHeuristicsPlayer}
    mix = {k: float(v) for k, v in (kv.split(":") for kv in args.opponents.split(","))}
    total = sum(mix.values())
    t0 = time.time()
    for kind, w in mix.items():
        n = int(round(args.battles * w / total))
        if n <= 0:
            continue
        opp = opp_classes[kind](account_configuration=AccountConfiguration(_rand_name("BCO"), None),
                                battle_format=BATTLE_FORMAT, max_concurrent_battles=args.concurrency)
        before = teacher.n_won_battles
        await teacher.battle_against(opp, n_battles=n)
        print(f"[collect] teacher vs {kind:<16} {teacher.n_won_battles - before}/{n} won "
              f"({time.time() - t0:.0f}s, {sum(len(s) for s, _, _ in teacher.finished):,} samples)", flush=True)

    cfg = RewardConfig()
    obs, masks, acts, rets, ep_ids = [], [], [], [], []
    for ep, (steps, final_phi, outcome) in enumerate(teacher.finished):
        o, m, a, phis = zip(*steps)
        obs.extend(o); masks.extend(m); acts.extend(a)
        rets.append(returns_for_episode(list(phis), final_phi, outcome, cfg))
        ep_ids.extend([ep] * len(steps))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=np.asarray(obs, np.float32), masks=np.asarray(masks, bool),
                        actions=np.asarray(acts, np.int64), returns=np.concatenate(rets),
                        episode=np.asarray(ep_ids, np.int32), features=args.features, teacher=args.teacher)
    print(f"[collect] saved {len(acts):,} samples from {len(teacher.finished)} battles to {out} "
          f"(skipped {teacher.skipped} forced/unmappable decisions)")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class _SpacesOnlyEnv(gym.Env):
    """Placeholder env so MaskablePPO can be constructed without a server connection."""

    def __init__(self, obs_size: int):
        self.observation_space = Box(-1.0, 1.0, (obs_size,), np.float32)
        self.action_space = Discrete(ACTION_SPACE_SIZE)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, np.float32), 0.0, True, False, {}

    def action_masks(self):
        return np.ones(ACTION_SPACE_SIZE, bool)


def build_model(obs_size: int):
    from sb3_contrib import MaskablePPO

    return MaskablePPO("MlpPolicy", _SpacesOnlyEnv(obs_size), policy_kwargs=POLICY_KWARGS,
                       seed=None, device="cpu", verbose=0)


def train_bc(args) -> None:
    th.set_num_threads(args.torch_threads)
    d = np.load(args.data)
    obs, masks, acts, rets, ep = d["obs"], d["masks"], d["actions"], d["returns"], d["episode"]
    n_eps = int(ep.max()) + 1
    rng = np.random.default_rng(0)
    val_eps = set(rng.choice(n_eps, size=max(1, n_eps // 10), replace=False).tolist())
    is_val = np.isin(ep, list(val_eps))
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    print(f"[bc] {len(acts):,} samples ({len(tr):,} train / {len(va):,} val, split by battle), obs {obs.shape[1]}")

    model = build_model(obs.shape[1])
    policy = model.policy
    opt = th.optim.Adam(policy.parameters(), lr=args.lr)
    O, M, A, R = (th.as_tensor(x) for x in (obs, masks, acts, rets))

    def evaluate(idx):
        policy.eval()
        with th.no_grad():
            dist = policy.get_distribution(O[idx], action_masks=M[idx].numpy())
            pred = dist.distribution.probs.argmax(-1)
            acc = (pred == A[idx]).float().mean().item()
            nll = -dist.log_prob(A[idx]).mean().item()
            v = policy.predict_values(O[idx]).squeeze(-1)
            vmse = th.mean((v - R[idx]) ** 2).item()
        policy.train()
        return acc, nll, vmse

    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            dist = policy.get_distribution(O[b], action_masks=M[b].numpy())
            pi_loss = -dist.log_prob(A[b]).mean()
            v_loss = th.mean((policy.predict_values(O[b]).squeeze(-1) - R[b]) ** 2)
            loss = pi_loss + args.vf_weight * v_loss
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
        tr_acc, tr_nll, _ = evaluate(tr[:20000])
        va_acc, va_nll, va_v = evaluate(va)
        print(f"[bc] epoch {epoch:2d} | train acc {tr_acc:.1%} nll {tr_nll:.3f} | "
              f"val acc {va_acc:.1%} nll {va_nll:.3f} value-mse {va_v:.1f}", flush=True)
        if va_nll < best_val:  # early stopping on held-out battles
            best_val = va_nll
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"[bc] saved best-val policy (val nll {best_val:.3f}) to {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--battles", type=int, default=3000)
    c.add_argument("--teacher", choices=["simple_heuristic", "heuristic"], default="simple_heuristic")
    c.add_argument("--opponents", default="random:0.1,maxpower:0.15,heuristic:0.35,simple_heuristic:0.4")
    c.add_argument("--features", choices=["v1", "v2"], default="v2")
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--out", default="data/bc_v2.npz")
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/bc_v2.npz")
    t.add_argument("--out", default="checkpoints/bc_v2/bc.zip")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--vf-weight", type=float, default=0.01)
    t.add_argument("--torch-threads", type=int, default=2)
    args = p.parse_args(argv)
    if args.cmd == "collect":
        asyncio.run(_collect(args))
    else:
        train_bc(args)


if __name__ == "__main__":
    main()
