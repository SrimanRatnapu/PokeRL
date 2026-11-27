"""Curriculum, reward-anneal, checkpoint/eval and progress callbacks for pokebot.train.

Also provides ``OpponentMixWrapper``: a thin gymnasium wrapper around ``MaskedPokeEnv``
that samples a fresh opponent from a weighted "mix" right before every ``reset()``.
Opponents are described by string keys so the same mechanism works under both
``DummyVecEnv`` and ``SubprocVecEnv`` (no Player objects cross process boundaries):

    "random" | "maxpower" | "heuristic" | "simple_heuristic"   -> make_opponent(kind)
    "policy:<path/to/ckpt.zip>"                               -> make_opponent("policy", model=...)
    "pool"   (mix entry only)  -> sample a checkpoint from the self-play pool

Every episode-ending ``info`` dict gets ``info["opponent"] = <key>`` so win rates can be
attributed to the exact opponent the battle was played against.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from pokebot.config import BATTLE_FORMAT, RewardConfig

POLICY_PREFIX = "policy:"
POOL = "pool"


def opponent_kind(key: str | None) -> str:
    """Collapse an opponent key to its kind ("policy:<path>" -> "policy")."""
    if not key:
        return "unknown"
    return "policy" if key.startswith(POLICY_PREFIX) else key


# --------------------------------------------------------------------------- #
# Opponent construction (runs inside whichever process owns the env)
# --------------------------------------------------------------------------- #
_MODEL_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_MODEL_CACHE_CAP = 8


def load_policy_model(path: str):
    """Load (and LRU-cache per process) a MaskablePPO checkpoint for use as an opponent."""
    from sb3_contrib import MaskablePPO

    path = str(path)
    if path in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(path)
        return _MODEL_CACHE[path]
    # seed=None: a seeded load would re-seed python/numpy/torch global RNGs mid-training.
    model = MaskablePPO.load(path, device="cpu", custom_objects={"seed": None})
    _MODEL_CACHE[path] = model
    while len(_MODEL_CACHE) > _MODEL_CACHE_CAP:
        _MODEL_CACHE.popitem(last=False)
    return model


def build_opponent(key: str):
    from pokebot.env import make_opponent

    if key.startswith(POLICY_PREFIX):
        # Stochastic past selves give a more diverse (harder to exploit) self-play opponent.
        return make_opponent("policy", model=load_policy_model(key[len(POLICY_PREFIX):]), deterministic=False)
    return make_opponent(key)


class OpponentMixWrapper(gym.Wrapper):
    """Samples the opponent for each new battle from a weighted mix of opponent kinds."""

    def __init__(
        self,
        env: gym.Env,
        mix: dict[str, float] | None = None,
        pool: list[str] | None = None,
        seed: int | None = None,
        initial_key: str = "random",
        initial_player: Any = None,
        recent_k: int = 3,
        recent_prob: float = 0.5,
        max_cached_policies: int = 6,
    ):
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self._mix: dict[str, float] = dict(mix or {initial_key: 1.0})
        self._pool: list[str] = list(pool or [])
        self._recent_k = recent_k
        self._recent_prob = recent_prob
        self._max_cached_policies = max_cached_policies
        self._players: "OrderedDict[str, Any]" = OrderedDict()
        if initial_player is not None:
            self._players[initial_key] = initial_player
        self._env_key = initial_key  # opponent currently installed in the inner env
        self._active_key = initial_key  # opponent of the battle in progress

    # ---- control surface (called through VecEnv.env_method) ---------------- #
    def set_opponent_mix(self, mix: dict[str, float], pool: list[str] | None = None) -> None:
        self._mix = dict(mix)
        if pool is not None:
            self._pool = list(pool)

    def set_opponent_pool(self, pool: list[str]) -> None:
        self._pool = list(pool)

    def current_opponent(self) -> str:
        return self._active_key

    def action_masks(self) -> np.ndarray:
        return self.env.get_wrapper_attr("action_masks")()

    # ---- internals ---------------------------------------------------------- #
    def _sample_key(self) -> str:
        items = [(k, w) for k, w in self._mix.items() if w > 0 and (k != POOL or self._pool)]
        if not items:
            return self._env_key
        keys, weights = zip(*items)
        p = np.asarray(weights, dtype=np.float64)
        kind = keys[int(self._rng.choice(len(keys), p=p / p.sum()))]
        if kind != POOL:
            return kind
        # Recency-biased checkpoint choice: often the latest few, otherwise uniform.
        if len(self._pool) > self._recent_k and self._rng.random() < self._recent_prob:
            candidates = self._pool[-self._recent_k:]
        else:
            candidates = self._pool
        return POLICY_PREFIX + candidates[int(self._rng.integers(len(candidates)))]

    def _get_player(self, key: str):
        if key in self._players:
            self._players.move_to_end(key)
            return self._players[key]
        player = build_opponent(key)
        self._players[key] = player
        policy_keys = [k for k in self._players if k.startswith(POLICY_PREFIX) and k != self._env_key and k != key]
        while sum(k.startswith(POLICY_PREFIX) for k in self._players) > self._max_cached_policies and policy_keys:
            self._players.pop(policy_keys.pop(0), None)
        return player

    def reset(self, **kwargs):
        key = self._sample_key()
        if key != self._env_key:
            try:
                player = self._get_player(key)
                self.env.get_wrapper_attr("set_opponent")(player)  # takes effect on this reset
                self._env_key = key
            except Exception as exc:  # never kill training over a bad opponent checkpoint
                warnings.warn(f"could not switch opponent to {key!r} ({exc!r}); keeping {self._env_key!r}")
        self._active_key = self._env_key
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["opponent"] = self._active_key
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info)
            info["opponent"] = self._active_key
        return obs, reward, terminated, truncated, info


def build_train_env(
    idx: int,
    mix: dict[str, float],
    pool: list[str],
    seed: int,
    reward_config: RewardConfig,
    battle_format: str = BATTLE_FORMAT,
    feature_version: str = "v1",
) -> gym.Env:
    """Env factory (module-level so it pickles for SubprocVecEnv).

    `feature_version` defaults to "v1" here only for backward compatibility with old
    callers; pokebot.train always passes it explicitly. Self-play opponents
    (PolicyPlayer) detect their own version from the checkpoint.
    """
    from pokebot.env import make_env, make_opponent

    initial = make_opponent("random")
    env = make_env(opponent=initial, reward_config=replace(reward_config), battle_format=battle_format,
                   feature_version=feature_version)
    return OpponentMixWrapper(env, mix=mix, pool=pool, seed=seed + 1000 * idx, initial_key="random", initial_player=initial)


# --------------------------------------------------------------------------- #
# Shared run state (persisted to checkpoints/<run>/train_state.json)
# --------------------------------------------------------------------------- #
@dataclass
class TrainState:
    stage_idx: int = 0
    stage_start_ts: int = 0
    pool: list[str] = field(default_factory=list)
    best_eval: float = -1.0
    last_eval: float | None = None
    shaping_scale: float = 1.0
    num_timesteps: int = 0
    stage_history: list[dict] = field(default_factory=list)
    features: str | None = None  # observation feature version ("v1"/"v2"); None in pre-v2 state files

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TrainState":
        data = json.loads(path.read_text())
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        state = cls(**known)
        state.pool = [p for p in state.pool if Path(p).exists()]
        return state


@dataclass
class Stage:
    name: str
    mix: dict[str, float]
    advance_win_rate: float | None  # early-advance threshold on rolling in-stage win rate
    end_frac: float  # forced advance once training progress reaches this fraction

    @property
    def uses_pool(self) -> bool:
        return self.mix.get(POOL, 0) > 0

    def counts(self, key: str | None) -> bool:
        kind = opponent_kind(key)
        return (kind == "policy" and self.uses_pool) or self.mix.get(kind, 0) > 0


def default_stages(fracs: tuple[float, float, float] = (0.10, 0.25, 0.50)) -> list[Stage]:
    f1, f2, f3 = fracs
    return [
        Stage("random", {"random": 1.0}, 0.85, f1),
        Stage("maxpower", {"maxpower": 1.0}, 0.70, f2),
        Stage("heuristics", {"heuristic": 0.45, "simple_heuristic": 0.45, "maxpower": 0.10}, 0.60, f3),
        # ~30% scripted opponents in self-play to avoid forgetting how to beat them.
        Stage("selfplay", {POOL: 0.65, "heuristic": 0.15, "simple_heuristic": 0.15, "maxpower": 0.05}, None, 1.01),
    ]


def _progress(cb: BaseCallback) -> float:
    total = getattr(cb.model, "_total_timesteps", 0) or 1
    return min(1.0, cb.model.num_timesteps / total)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
class CurriculumCallback(BaseCallback):
    """Tracks win rates from info["won"] and moves through the opponent curriculum."""

    def __init__(
        self,
        stages: list[Stage],
        state: TrainState,
        ckpt_dir: Path,
        window: int = 200,
        min_episodes: int = 100,
        min_stage_frac: float = 0.02,
        winrate_advance: bool = True,
        pool_size: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stages = stages
        self.state = state
        self.ckpt_dir = Path(ckpt_dir)
        self.window = window
        self.min_episodes = min_episodes
        self.min_stage_frac = min_stage_frac
        self.winrate_advance = winrate_advance
        self.pool_size = pool_size
        self.stage_results: deque[float] = deque(maxlen=window)  # in-stage opponents only
        self.all_results: deque[float] = deque(maxlen=window)
        self.by_kind: dict[str, deque[float]] = {}
        self.episodes = 0
        self.unknown_outcomes = 0

    @property
    def stage(self) -> Stage:
        return self.stages[self.state.stage_idx]

    def rolling_win_rate(self) -> tuple[float | None, int]:
        n = len(self.stage_results)
        return (float(np.mean(self.stage_results)) if n else None), n

    def _on_training_start(self) -> None:
        self.state.stage_idx = min(max(self.state.stage_idx, 0), len(self.stages) - 1)
        forced = self._forced_stage()
        if forced > self.state.stage_idx:
            self._enter_stage(forced, reason="timestep fraction (resume)")
        else:
            self._enter_stage(self.state.stage_idx, reason="start", record=not self.state.stage_history)

    def _forced_stage(self) -> int:
        prog = _progress(self)
        idx = 0
        while idx < len(self.stages) - 1 and prog >= self.stages[idx].end_frac:
            idx += 1
        return idx

    def _enter_stage(self, idx: int, reason: str, record: bool = True) -> None:
        changed = idx != self.state.stage_idx
        self.state.stage_idx = idx
        if changed or not self.state.stage_history:
            self.state.stage_start_ts = self.model.num_timesteps
        stage = self.stage
        if stage.uses_pool and not self.state.pool:
            path = self.ckpt_dir / f"pool_seed_{self.model.num_timesteps}.zip"
            self.model.save(path)
            self.state.pool.append(str(path))
        self.training_env.env_method("set_opponent_mix", stage.mix, list(self.state.pool))
        self.stage_results.clear()
        if record:
            self.state.stage_history.append({"stage": stage.name, "ts": self.model.num_timesteps, "reason": reason})
        if self.verbose:
            print(f"[curriculum] stage {idx + 1}/{len(self.stages)} '{stage.name}' at {self.model.num_timesteps:,} steps "
                  f"({reason}); mix={stage.mix}", flush=True)

    def add_to_pool(self, path: str | Path) -> None:
        self.state.pool.append(str(path))
        self.state.pool = self.state.pool[-self.pool_size:]
        self.training_env.env_method("set_opponent_pool", list(self.state.pool))

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        if dones is None or not np.any(dones):
            return True
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.episodes += 1
            won = info.get("won")
            if won is None:
                self.unknown_outcomes += 1
                continue
            r = 1.0 if won else 0.0
            key = info.get("opponent")
            self.all_results.append(r)
            self.by_kind.setdefault(opponent_kind(key), deque(maxlen=self.window)).append(r)
            if self.stage.counts(key):
                self.stage_results.append(r)
        self._maybe_advance()
        return True

    def _maybe_advance(self) -> None:
        idx = self.state.stage_idx
        if idx >= len(self.stages) - 1:
            return
        forced = self._forced_stage()
        if forced > idx:
            self._enter_stage(forced, reason=f"timestep fraction {self.stages[idx].end_frac:.2f}")
            return
        thr = self.stage.advance_win_rate
        if not self.winrate_advance or thr is None:
            return
        wr, n = self.rolling_win_rate()
        in_stage = self.model.num_timesteps - self.state.stage_start_ts
        if (wr is not None and n >= self.min_episodes and wr >= thr
                and in_stage >= self.min_stage_frac * getattr(self.model, "_total_timesteps", 0)):
            self._enter_stage(idx + 1, reason=f"win rate {wr:.1%} >= {thr:.0%} over {n} battles")

    def _on_rollout_end(self) -> None:
        wr, n = self.rolling_win_rate()
        self.logger.record("curriculum/stage", self.state.stage_idx + 1)
        self.logger.record("curriculum/pool_size", len(self.state.pool))
        self.logger.record("curriculum/episodes", self.episodes)
        if wr is not None:
            self.logger.record("winrate/rolling_stage", wr)
        if self.all_results:
            self.logger.record("winrate/rolling_all", float(np.mean(self.all_results)))
        for kind, dq in self.by_kind.items():
            if dq:
                self.logger.record(f"winrate/{kind}", float(np.mean(dq)))
        if self.unknown_outcomes:
            self.logger.record("curriculum/unknown_outcomes", self.unknown_outcomes)


class ShapingAnnealCallback(BaseCallback):
    """Linearly anneals RewardConfig.shaping_scale from `start` to `end` over training progress."""

    def __init__(self, base: RewardConfig, state: TrainState, start: float = 1.0, end: float = 0.1,
                 end_frac: float = 1.0, verbose: int = 0):
        super().__init__(verbose)
        self.base = base
        self.state = state
        self.start, self.end, self.end_frac = start, end, max(end_frac, 1e-8)
        self._last: float | None = None

    def _apply(self) -> None:
        t = min(1.0, _progress(self) / self.end_frac)
        scale = self.start + (self.end - self.start) * t
        if self._last is None or abs(scale - self._last) > 1e-4:
            self.training_env.env_method("set_reward_config", replace(self.base, shaping_scale=scale))
            self._last = scale
        self.state.shaping_scale = scale

    def _on_training_start(self) -> None:
        self._apply()

    def _on_rollout_start(self) -> None:
        self._apply()

    def _on_rollout_end(self) -> None:
        self.logger.record("reward/shaping_scale", self.state.shaping_scale)

    def _on_step(self) -> bool:
        return True


def run_eval(model, env, n_battles: int, max_steps: int = 2000) -> tuple[int, int]:
    """Play `n_battles` with the deterministic policy on a plain MaskedPokeEnv. Returns (wins, played)."""
    wins = played = 0
    for _ in range(n_battles):
        obs, _info = env.reset()
        info: dict = {}
        for _step in range(max_steps):
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _r, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        played += 1
        wins += int(bool(info.get("won", False)))
    return wins, played


class CheckpointEvalCallback(BaseCallback):
    """Periodic checkpoints (fed into the self-play pool) and periodic deterministic eval (best.zip)."""

    def __init__(
        self,
        ckpt_dir: Path,
        state: TrainState,
        curriculum: CurriculumCallback,
        ckpt_freq: int,
        eval_freq: int,
        eval_battles: int,
        eval_env_factory: Callable[[], gym.Env] | None,
        eval_opponent: str = "simple_heuristic",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.ckpt_dir = Path(ckpt_dir)
        self.state = state
        self.curriculum = curriculum
        self.ckpt_freq = ckpt_freq
        self.eval_freq = eval_freq
        self.eval_battles = eval_battles
        self.eval_env_factory = eval_env_factory
        self.eval_opponent = eval_opponent
        self.eval_env: gym.Env | None = None
        self._last_ckpt = 0
        self._last_eval = 0

    @property
    def state_path(self) -> Path:
        return self.ckpt_dir / "train_state.json"

    def _on_training_start(self) -> None:
        self._last_ckpt = self._last_eval = self.model.num_timesteps

    def save_state(self) -> None:
        self.state.num_timesteps = self.model.num_timesteps
        self.state.save(self.state_path)

    def save_checkpoint(self, name: str | None = None, add_to_pool: bool = True) -> Path:
        path = self.ckpt_dir / (name or f"step_{self.model.num_timesteps}.zip")
        self.model.save(path)
        self.model.save(self.ckpt_dir / "latest.zip")
        if add_to_pool:
            self.curriculum.add_to_pool(path)
        self.save_state()
        if self.verbose:
            print(f"[checkpoint] saved {path.name} (pool={len(self.state.pool)})", flush=True)
        return path

    def evaluate(self) -> float:
        if self.eval_env is None:
            self.eval_env = self.eval_env_factory()
        t0 = time.time()
        wins, played = run_eval(self.model, self.eval_env, self.eval_battles)
        wr = wins / max(played, 1)
        self.state.last_eval = wr
        self.logger.record(f"eval/win_rate_vs_{self.eval_opponent}", wr)
        self.logger.record("eval/battles", played)
        is_best = wr > self.state.best_eval
        if is_best:
            self.state.best_eval = wr
            self.model.save(self.ckpt_dir / "best.zip")
        self.logger.record("eval/best_win_rate", self.state.best_eval)
        self.save_state()
        if self.verbose:
            print(f"[eval] {self.model.num_timesteps:,} steps: {wins}/{played} = {wr:.1%} vs {self.eval_opponent} "
                  f"({time.time() - t0:.0f}s){'  -> new best.zip' if is_best else ''}", flush=True)
        return wr

    def _on_step(self) -> bool:
        ts = self.model.num_timesteps
        if self.ckpt_freq > 0 and ts - self._last_ckpt >= self.ckpt_freq:
            self._last_ckpt = ts
            self.save_checkpoint()
        if self.eval_freq > 0 and self.eval_env_factory is not None and ts - self._last_eval >= self.eval_freq:
            self._last_eval = ts
            self.evaluate()
        return True

    def close(self) -> None:
        if self.eval_env is not None:
            try:
                self.eval_env.close()
            finally:
                self.eval_env = None


class ProgressCallback(BaseCallback):
    """Prints one concise status line after every rollout."""

    def __init__(self, run_name: str, state: TrainState, curriculum: CurriculumCallback, verbose: int = 1):
        super().__init__(verbose)
        self.run_name = run_name
        self.state = state
        self.curriculum = curriculum
        self._t0 = 0.0
        self._ts0 = 0
        self._t_last = 0.0
        self._ts_last = 0

    def _on_training_start(self) -> None:
        self._t0 = self._t_last = time.time()
        self._ts0 = self._ts_last = self.model.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        now, ts = time.time(), self.model.num_timesteps
        total = getattr(self.model, "_total_timesteps", ts) or ts
        sps_recent = (ts - self._ts_last) / max(now - self._t_last, 1e-6)
        sps_avg = (ts - self._ts0) / max(now - self._t0, 1e-6)
        self._t_last, self._ts_last = now, ts
        eta = (total - ts) / max(sps_avg, 1e-6)
        self.logger.record("time/sps_recent", sps_recent)

        wr, n = self.curriculum.rolling_win_rate()
        stage = self.curriculum.stage
        buf = self.model.ep_info_buffer or []
        ep_rew = np.mean([e["r"] for e in buf]) if buf else float("nan")
        ep_len = np.mean([e["l"] for e in buf]) if buf else float("nan")
        wr_s = f"{wr:.1%} (n={n})" if wr is not None else "n/a"
        ev = "" if self.state.last_eval is None else f" | eval {self.state.last_eval:.0%} (best {self.state.best_eval:.0%})"
        print(
            f"[{self.run_name}] {ts:,}/{total:,} ({ts / total:.1%}) | stage {self.state.stage_idx + 1}/"
            f"{len(self.curriculum.stages)} {stage.name} | win {wr_s} | shaping {self.state.shaping_scale:.2f} | "
            f"ep_rew {ep_rew:.2f} len {ep_len:.0f} | {sps_recent:.0f} sps | eta {_fmt_secs(eta)}{ev}",
            flush=True,
        )


def _fmt_secs(s: float) -> str:
    s = int(max(s, 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"
