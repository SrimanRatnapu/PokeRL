"""Train a MaskablePPO agent for gen9randombattle with an opponent curriculum.

Run from the project root (local Showdown server must be up on localhost:8000):

    .venv/bin/python -m pokebot.train --smoke                                  # ~30 s sanity check (2 envs)
    .venv/bin/python -m pokebot.train --timesteps 1000000 --n-envs 4 --run-name v1
    .venv/bin/python -m pokebot.train --timesteps 500000 --n-envs 4 --run-name v1 \
        --resume checkpoints/v1/latest.zip                                     # +500k more steps

Observation features: --features v1|v2 (default v2 for new runs). The version is stored in
train_state.json; --resume reuses it (falling back to the checkpoint's observation size for
old state files) and errors if an explicit --features disagrees.

Curriculum (stage advances at the forced timestep fraction OR earlier once the rolling
win rate vs that stage's opponents clears the threshold):
    1 random            -> advance at >85% or 10% of training
    2 maxpower          -> advance at >70% or 25%
    3 heuristic mix     -> advance at >60% or 50%
    4 self-play pool (65% past checkpoints, 35% scripted opponents) until the end
Reward shaping_scale anneals linearly 1.0 -> 0.1 over the run.

Outputs: checkpoints/<run>/{step_N.zip, latest.zip, best.zip, final.zip, train_state.json}
         logs/<run>/ (TensorBoard + progress.csv): tensorboard --logdir logs
"""

from __future__ import annotations

import argparse
import functools
import os
import random
import signal
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch as th

from pokebot.callbacks import (
    CheckpointEvalCallback,
    CurriculumCallback,
    ProgressCallback,
    ShapingAnnealCallback,
    TrainState,
    build_train_env,
    default_stages,
)
from pokebot.config import BATTLE_FORMAT, CHECKPOINT_DIR, LOG_DIR, RewardConfig
from pokebot.feature_sets import DEFAULT_FEATURE_VERSION, FEATURE_VERSIONS, version_for_obs_size

MAX_ENVS = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--timesteps", type=int, default=1_000_000, help="env steps to train (added on top when resuming)")
    p.add_argument("--n-envs", type=int, default=4, help=f"parallel battles (capped at {MAX_ENVS})")
    p.add_argument("--run-name", default=None, help="output subdir under checkpoints/ and logs/ (default: run_<time>)")
    p.add_argument("--resume", default=None, help="checkpoint .zip to continue training from")
    p.add_argument("--init-from", default=None,
                   help="start a NEW run from these policy weights (e.g. a behavior-cloning model from pokebot.bc); "
                        "optimizer state and step count start fresh")
    p.add_argument("--device", default="cpu", help="torch device (cpu is fastest for this small MLP)")
    p.add_argument("--seed", type=int, default=None, help="seed numpy/torch/opponent sampling (default: random)")
    p.add_argument("--subproc", action="store_true", help="use SubprocVecEnv instead of DummyVecEnv (experimental)")
    p.add_argument("--torch-threads", type=int, default=min(4, os.cpu_count() or 1), help="cap torch CPU threads")
    p.add_argument("--features", choices=FEATURE_VERSIONS, default=None,
                   help=f"observation feature version (default: {DEFAULT_FEATURE_VERSION}; on --resume: the run's stored version)")
    p.add_argument("--smoke", action="store_true", help="tiny config for quick end-to-end verification")
    p.add_argument("--verbose", action="store_true", help="also print SB3's per-iteration stats table")

    h = p.add_argument_group("PPO")
    h.add_argument("--lr", type=float, default=3e-4)
    h.add_argument("--lr-final-frac", type=float, default=0.1, help="linear LR decay to lr*frac (1.0 = constant)")
    h.add_argument("--n-steps", type=int, default=1024, help="rollout steps per env")
    h.add_argument("--batch-size", type=int, default=256)
    h.add_argument("--n-epochs", type=int, default=10)
    h.add_argument("--gamma", type=float, default=0.99)
    h.add_argument("--gae-lambda", type=float, default=0.95)
    h.add_argument("--clip-range", type=float, default=0.2)
    h.add_argument("--ent-coef", type=float, default=0.01)
    h.add_argument("--vf-coef", type=float, default=0.5)

    c = p.add_argument_group("curriculum / reward")
    c.add_argument("--stage-fracs", default="0.10,0.25,0.50",
                   help="forced end of stages 1-3 as fractions of total timesteps")
    c.add_argument("--start-stage", type=int, default=None, help="force starting stage (1-4)")
    c.add_argument("--no-winrate-advance", action="store_true", help="advance only by timestep fractions")
    c.add_argument("--winrate-window", type=int, default=200, help="rolling window (battles) for advancement")
    c.add_argument("--min-episodes", type=int, default=100, help="min in-stage battles before win-rate advance")
    c.add_argument("--min-stage-frac", type=float, default=0.02, help="min fraction of training spent per stage")
    c.add_argument("--pool-size", type=int, default=20, help="max checkpoints kept in the self-play pool")
    c.add_argument("--shaping-start", type=float, default=1.0)
    c.add_argument("--shaping-end", type=float, default=0.1)

    e = p.add_argument_group("checkpoints / eval")
    e.add_argument("--ckpt-freq", type=int, default=50_000, help="env steps between checkpoints")
    e.add_argument("--eval-freq", type=int, default=50_000, help="env steps between evals (0 = off)")
    e.add_argument("--eval-battles", type=int, default=50)
    e.add_argument("--eval-opponent", default="simple_heuristic")

    args = p.parse_args(argv)
    if args.smoke:
        args.timesteps = 4096
        args.n_envs = min(args.n_envs, 2)
        args.n_steps = 256
        args.batch_size = 256
        args.n_epochs = 4
        args.stage_fracs = "0.15,0.30,0.45"
        args.winrate_window = 10
        args.min_episodes = 4
        args.min_stage_frac = 0.0
        args.ckpt_freq = 1024
        args.eval_freq = 2048
        args.eval_battles = 5
        args.run_name = args.run_name or "smoke"
    if args.n_envs > MAX_ENVS:
        print(f"[train] capping --n-envs {args.n_envs} -> {MAX_ENVS} (laptop-friendly limit)")
        args.n_envs = MAX_ENVS
    args.n_envs = max(1, args.n_envs)
    args.run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    fr = tuple(float(x) for x in args.stage_fracs.split(","))
    if len(fr) != 3 or not (0 <= fr[0] <= fr[1] <= fr[2] <= 1):
        p.error("--stage-fracs must be three increasing fractions, e.g. 0.1,0.25,0.5")
    args.stage_fracs = fr
    return args


def make_vec_env(args, mix, pool, reward_cfg):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    fns = [functools.partial(build_train_env, i, mix, pool, args.seed, reward_cfg, BATTLE_FORMAT, args.features)
           for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if args.subproc and args.n_envs > 1 else DummyVecEnv(fns)
    return VecMonitor(venv)


def _close_with_timeout(obj, what: str, timeout: float = 20.0) -> bool:
    """Close envs in a helper thread so a wedged websocket can't hang the process forever."""
    if obj is None:
        return True
    err: list[BaseException] = []

    def _run():
        try:
            obj.close()
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"[train] warning: closing {what} timed out after {timeout:.0f}s")
        return False
    if err:
        print(f"[train] warning: error closing {what}: {err[0]!r}")
    return True


def checkpoint_feature_version(path: str | Path) -> str | None:
    """Feature version of a saved MaskablePPO .zip, from its stored observation space."""
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _, _ = load_from_zip_file(str(path), load_data=True, custom_objects={"seed": None}, device="cpu")
    shape = tuple(getattr((data or {}).get("observation_space"), "shape", ()) or ())
    return version_for_obs_size(shape[0]) if len(shape) == 1 else None


def resolve_feature_version(args, state: TrainState) -> str:
    """Pick the run's feature version: explicit flag / default for new runs; stored (or inferred) on resume."""
    if not args.resume:
        return args.features or DEFAULT_FEATURE_VERSION
    stored = state.features
    source = "train_state.json"
    if stored is None:
        stored = checkpoint_feature_version(args.resume)
        source = "checkpoint observation size"
    if stored is None:
        raise SystemExit(f"[train] cannot determine the feature version of {args.resume}; pass --features")
    if args.features is not None and args.features != stored:
        raise SystemExit(f"[train] --features {args.features} conflicts with the resumed run's features "
                         f"{stored} (from {source}); drop the flag or start a new run")
    return stored


def _raise_interrupt(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Ctrl+C / `kill` -> KeyboardInterrupt -> save a checkpoint and close envs (also when
    # launched from a shell that would otherwise ignore SIGINT, e.g. `nohup ... &`).
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    th.set_num_threads(max(1, args.torch_threads))

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import LinearSchedule, set_random_seed

    from pokebot.env import make_env, make_opponent

    # Seed numpy/torch only. Python's global `random` must stay on fresh entropy: poke-env
    # draws Showdown usernames from it, and re-seeding it makes new envs (eval env, a second
    # run) reuse names already logged in -> |nametaken|. For the same reason models are built
    # with seed=None, since SB3 re-seeds `random` whenever a seeded model is created or loaded.
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "little") % 1_000_000
    set_random_seed(args.seed)
    random.seed()
    ckpt_dir = CHECKPOINT_DIR / args.run_name
    log_dir = LOG_DIR / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- run state (curriculum stage, self-play pool, best eval) ------------ #
    state = TrainState()
    if args.resume:
        for cand in (Path(args.resume).parent / "train_state.json", ckpt_dir / "train_state.json"):
            if cand.exists():
                state = TrainState.load(cand)
                print(f"[train] restored run state from {cand} (stage {state.stage_idx + 1}, pool {len(state.pool)})")
                break
        else:
            ckpts = sorted(Path(args.resume).parent.glob("step_*.zip"), key=lambda q: int(q.stem.split("_")[1]))
            state.pool = [str(q) for q in ckpts][-args.pool_size:]
    if args.start_stage is not None:
        state.stage_idx = int(np.clip(args.start_stage - 1, 0, 3))
        state.stage_history = []
    args.features = resolve_feature_version(args, state)
    state.features = args.features

    stages = default_stages(args.stage_fracs)
    base_reward = RewardConfig()
    init_reward = replace(base_reward, shaping_scale=args.shaping_start)

    lr = LinearSchedule(args.lr, args.lr * args.lr_final_frac, 1.0) if args.lr_final_frac != 1.0 else args.lr

    venv = eval_cb = None
    model = None
    exit_code = 0
    interrupted = False
    try:
        print(f"[train] run={args.run_name} seed={args.seed} envs={args.n_envs} ({'subproc' if args.subproc else 'dummy'}) "
              f"device={args.device} features={args.features} timesteps={args.timesteps:,} -> {ckpt_dir}", flush=True)
        venv = make_vec_env(args, stages[state.stage_idx].mix, state.pool, init_reward)

        if args.resume:
            model = MaskablePPO.load(args.resume, env=venv, device=args.device, custom_objects={"learning_rate": lr, "seed": None})
            print(f"[train] resumed from {args.resume} at {model.num_timesteps:,} steps")
        else:
            model = MaskablePPO(
                "MlpPolicy",
                venv,
                learning_rate=lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                seed=None,  # see seeding note above
                device=args.device,
                verbose=0,
            )
        if args.init_from and not args.resume:
            src = MaskablePPO.load(args.init_from, device=args.device, custom_objects={"seed": None})
            if src.observation_space.shape != model.observation_space.shape:
                raise SystemExit(f"[train] --init-from {args.init_from} has obs {src.observation_space.shape}, "
                                 f"run uses {model.observation_space.shape} ({args.features})")
            model.policy.load_state_dict(src.policy.state_dict())
            del src
            print(f"[train] initialized policy + value weights from {args.init_from}")
        formats = ["tensorboard", "csv"] + (["stdout"] if args.verbose else [])
        model.set_logger(configure(str(log_dir), formats))

        curriculum = CurriculumCallback(
            stages, state, ckpt_dir,
            window=args.winrate_window, min_episodes=args.min_episodes, min_stage_frac=args.min_stage_frac,
            winrate_advance=not args.no_winrate_advance, pool_size=args.pool_size,
        )
        shaping = ShapingAnnealCallback(base_reward, state, start=args.shaping_start, end=args.shaping_end)
        eval_factory = ((lambda: make_env(opponent=make_opponent(args.eval_opponent), feature_version=args.features))
                        if args.eval_freq > 0 else None)
        eval_cb = CheckpointEvalCallback(
            ckpt_dir, state, curriculum,
            ckpt_freq=args.ckpt_freq, eval_freq=args.eval_freq, eval_battles=args.eval_battles,
            eval_env_factory=eval_factory, eval_opponent=args.eval_opponent,
        )
        progress = ProgressCallback(args.run_name, state, curriculum)
        callbacks = CallbackList([curriculum, shaping, eval_cb, progress])

        t0 = time.time()
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callbacks,
                reset_num_timesteps=not args.resume,
                log_interval=1,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\n[train] interrupted - saving checkpoint ...", flush=True)
            eval_cb.save_checkpoint(name=f"interrupted_step_{model.num_timesteps}.zip", add_to_pool=False)
        else:
            elapsed = time.time() - t0
            model.save(ckpt_dir / "final.zip")
            model.save(ckpt_dir / "latest.zip")
            eval_cb.save_state()
            done = model.num_timesteps - (model._num_timesteps_at_start or 0)
            print(f"[train] done: {model.num_timesteps:,} steps ({done / max(elapsed, 1e-6):.0f} sps avg, "
                  f"{elapsed / 60:.1f} min); saved {ckpt_dir / 'final.zip'}", flush=True)
            if state.best_eval >= 0:
                print(f"[train] best eval win rate vs {args.eval_opponent}: {state.best_eval:.1%} -> best.zip")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[train] interrupted during setup", flush=True)
        if model is not None:
            model.save(ckpt_dir / f"interrupted_step_{model.num_timesteps}.zip")
    except Exception:
        exit_code = 1
        traceback.print_exc()
        if model is not None:
            try:
                path = ckpt_dir / f"crash_step_{model.num_timesteps}.zip"
                model.save(path)
                print(f"[train] error - emergency checkpoint saved to {path}", file=sys.stderr)
            except Exception:  # noqa: BLE001
                pass
    finally:
        if model is not None:
            try:
                model.logger.close()
            except Exception:  # noqa: BLE001
                pass
        ok = True
        if eval_cb is not None:
            ok &= _close_with_timeout(eval_cb, "eval env")
        ok &= _close_with_timeout(venv, "training envs")
        if not ok:  # something is wedged; don't hang at interpreter shutdown
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(130 if interrupted else exit_code)
    return 130 if interrupted else exit_code


if __name__ == "__main__":
    sys.exit(main())
