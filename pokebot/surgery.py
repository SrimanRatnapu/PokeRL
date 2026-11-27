"""
Network surgery: grow a trained MaskablePPO's observation input without changing its behaviour.

When a new feature version is "the old vector followed by new dims" (v3 = v2 + extras),
a trained policy can be reused instead of re-trained from scratch:

  * build a fresh MaskablePPO whose observation is N dims (same net_arch / hyperparameters),
  * copy every parameter of the source policy,
  * for every Linear layer that reads the observation (the first layer of the policy MLP and
    of the value MLP; the features extractor is a parameter-free Flatten) set
        new_W[:, :old] = old_W      new_W[:, old:] = 0   (or N(0, new_init_std))
    and keep the bias.

With zero init the new network computes exactly the old function of obs[:old] and ignores
obs[old:], so it plays identically to the source; PPO fine-tuning then learns weights for the
new inputs (the gradient wrt a weight is input x upstream gradient, which is non-zero even
when the weight itself is zero). This is the input-widening case of Net2Net-style
function-preserving expansion.

A built-in check evaluates both models on random observations and reports the max abs
difference of action logits / probabilities (all-true mask) and values; with
new_init_std == 0 it must be < 1e-5 or surgery fails.

The output is a normal MaskablePPO .zip: `pokebot.train --init-from` loads it as long as the
run's feature version has N dims, and evaluate.py / play.py can load it once a feature set of
size N is registered in pokebot.feature_sets.

Usage (from project root):
    .venv/bin/python -m pokebot.surgery --src checkpoints/champion.zip --obs-size N \
        --out checkpoints/v3_init/init.zip [--new-init-std 0.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.spaces import Box

PARITY_TOL = 1e-5

# Hyperparameters copied from the source model into the new one (all accepted by the
# MaskablePPO constructor). Schedules (learning_rate, clip_range) are passed through as-is.
_COPIED_HPARAMS = (
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range",
    "clip_range_vf", "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
)


class _SpacesOnlyEnv(gym.Env):
    """Placeholder env so MaskablePPO can be built without a Showdown server connection.

    (Same idea as pokebot.bc._SpacesOnlyEnv; duplicated so this module imports no feature code.)
    """

    def __init__(self, observation_space: Box, action_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, np.float32), 0.0, True, False, {}

    def action_masks(self):
        return np.ones(self.action_space.n, bool)


def _expanded_obs_space(src_space: Box, new_obs_size: int) -> Box:
    """Box of size N with the source's (scalar) bounds, e.g. Box(-1, 1, (N,), float32)."""
    low, high = np.unique(src_space.low), np.unique(src_space.high)
    if low.size != 1 or high.size != 1:
        raise ValueError("source observation space has per-dimension bounds; cannot infer bounds for new dims")
    return Box(low=float(low[0]), high=float(high[0]), shape=(new_obs_size,), dtype=src_space.dtype)


def _obs_input_layers(policy) -> list[str]:
    """Names of the Linear layers that consume the (flattened) observation directly."""
    names = []
    mlp = policy.mlp_extractor
    for net_name in ("shared_net", "policy_net", "value_net"):
        net = getattr(mlp, net_name, None)
        if net is None or len(net) == 0:
            continue
        first = net[0]
        if isinstance(first, th.nn.Linear):
            names.append(f"mlp_extractor.{net_name}.0")
        if net_name == "shared_net":
            break  # with a shared trunk only its first layer sees the observation
    return names


def _check_feature_extractors(policy, old_size: int) -> None:
    """The observation must reach the MLPs unchanged: parameter-free Flatten extractors only."""
    for name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        fe = getattr(policy, name, None)
        if fe is None:
            continue
        n_params = sum(p.numel() for p in fe.parameters())
        if n_params or type(fe).__name__ != "FlattenExtractor":
            raise ValueError(f"{name} is {type(fe).__name__} with {n_params} params; "
                             "surgery only supports a parameter-free FlattenExtractor")
        if fe.features_dim != old_size:
            raise ValueError(f"{name}.features_dim={fe.features_dim} != obs size {old_size}")


def build_like(src, new_obs_size: int):
    """Fresh MaskablePPO with obs size N and the source's net_arch / hyperparameters (seed=None, cpu)."""
    from sb3_contrib import MaskablePPO

    env = _SpacesOnlyEnv(_expanded_obs_space(src.observation_space, new_obs_size), src.action_space)
    kwargs = {k: getattr(src, k) for k in _COPIED_HPARAMS if hasattr(src, k)}
    return MaskablePPO("MlpPolicy", env, policy_kwargs=dict(src.policy_kwargs), seed=None,
                       device="cpu", verbose=0, **kwargs)


def expand_policy_(src_policy, dst_policy, old_size: int, new_size: int, new_init_std: float = 0.0,
                   generator: th.Generator | None = None) -> list[tuple[str, tuple, tuple, str]]:
    """Copy src -> dst in place, widening observation-input weights. Returns (key, old, new, action) rows."""
    _check_feature_extractors(src_policy, old_size)
    expand_layers = set(_obs_input_layers(src_policy))
    src_sd, dst_sd = src_policy.state_dict(), dst_policy.state_dict()
    if set(src_sd) != set(dst_sd):
        raise ValueError(f"parameter names differ: only in src {sorted(set(src_sd) - set(dst_sd))}, "
                         f"only in dst {sorted(set(dst_sd) - set(src_sd))}")
    new_sd, rows = {}, []
    for key, w_old in src_sd.items():
        w_dst = dst_sd[key]
        layer = key.rsplit(".", 1)[0]
        if layer in expand_layers and key.endswith(".weight") and new_size != old_size:
            if w_old.shape[1] != old_size or w_dst.shape != (w_old.shape[0], new_size):
                raise ValueError(f"{key}: unexpected shapes {tuple(w_old.shape)} -> {tuple(w_dst.shape)}")
            w_new = th.zeros(w_dst.shape, dtype=w_old.dtype)
            if new_init_std > 0 and new_size > old_size:
                w_new[:, old_size:] = th.randn((w_dst.shape[0], new_size - old_size), generator=generator,
                                               dtype=w_old.dtype) * new_init_std
            w_new[:, :old_size] = w_old
            new_sd[key] = w_new
            init = f"N(0, {new_init_std:g})" if new_init_std > 0 else "zeros"
            rows.append((key, tuple(w_old.shape), tuple(w_new.shape), f"expanded, new cols = {init}"))
        else:
            if w_old.shape != w_dst.shape:
                raise ValueError(f"{key}: shape {tuple(w_old.shape)} -> {tuple(w_dst.shape)} but it is not an "
                                 f"observation-input weight (input layers: {sorted(expand_layers)})")
            new_sd[key] = w_old.clone()
            rows.append((key, tuple(w_old.shape), tuple(w_dst.shape), "copied"))
    dst_policy.load_state_dict(new_sd, strict=True)
    return rows


@th.no_grad()
def parity_diffs(src_policy, dst_policy, old_size: int, new_size: int, n: int = 2048,
                 seed: int = 0, low: float = -1.0, high: float = 1.0) -> dict[str, float]:
    """Max abs diff of logits / probs (all-true mask) / values: dst(obs) vs src(obs[:, :old])."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(low, high, size=(n, new_size)).astype(np.float32)
    # also include the all-zero, all-low and all-high corners
    obs[:3] = np.stack([np.zeros(new_size), np.full(new_size, low), np.full(new_size, high)])
    n_actions = src_policy.action_space.n
    mask = np.ones((n, n_actions), dtype=bool)
    o_new = th.as_tensor(obs)
    o_old = th.as_tensor(obs[:, :old_size])
    src_policy.set_training_mode(False)
    dst_policy.set_training_mode(False)
    d_old = src_policy.get_distribution(o_old, action_masks=mask).distribution
    d_new = dst_policy.get_distribution(o_new, action_masks=mask).distribution
    v_old, v_new = src_policy.predict_values(o_old), dst_policy.predict_values(o_new)
    return {
        "logits": float((d_old.logits - d_new.logits).abs().max()),
        "probs": float((d_old.probs - d_new.probs).abs().max()),
        "values": float((v_old - v_new).abs().max()),
        "argmax_agree": float((d_old.probs.argmax(-1) == d_new.probs.argmax(-1)).float().mean()),
    }


def expand_model(src, new_obs_size: int, new_init_std: float = 0.0, seed: int | None = None,
                 verbose: bool = True):
    """In-memory surgery on a loaded MaskablePPO. Returns (new_model, summary_rows, parity_diffs)."""
    if len(src.observation_space.shape) != 1:
        raise ValueError(f"expected a 1-D observation space, got {src.observation_space.shape}")
    old = int(src.observation_space.shape[0])
    new_obs_size = int(new_obs_size)
    if new_obs_size < old:
        raise ValueError(f"new obs size {new_obs_size} < source obs size {old}: surgery only grows the input")
    if new_init_std < 0:
        raise ValueError("new_init_std must be >= 0")
    dst = build_like(src, new_obs_size)
    gen = th.Generator().manual_seed(seed) if seed is not None else None
    rows = expand_policy_(src.policy, dst.policy, old, new_obs_size, new_init_std, generator=gen)
    space = src.observation_space
    diffs = parity_diffs(src.policy, dst.policy, old, new_obs_size,
                         low=float(np.min(space.low)), high=float(np.max(space.high)))
    if verbose:
        n_src = sum(p.numel() for p in src.policy.parameters())
        n_dst = sum(p.numel() for p in dst.policy.parameters())
        print(f"[surgery] obs {old} -> {new_obs_size} (+{new_obs_size - old} dims); "
              f"params {n_src:,} -> {n_dst:,} (+{n_dst - n_src:,})")
        for key, s_old, s_new, action in rows:
            print(f"  {key:36s} {str(s_old):>12s} -> {str(s_new):<12s} {action}")
        print(f"[surgery] parity on 2048 random obs: max|dlogits|={diffs['logits']:.3e} "
              f"max|dprobs|={diffs['probs']:.3e} max|dvalue|={diffs['values']:.3e} "
              f"argmax agreement={diffs['argmax_agree']:.4f}")
    if new_init_std == 0 and max(diffs["logits"], diffs["probs"], diffs["values"]) >= PARITY_TOL:
        raise AssertionError(f"function-preservation check failed (tol {PARITY_TOL}): {diffs}")
    return dst, rows, diffs


def expand_obs(src_model_path: str | Path, new_obs_size: int, out_path: str | Path,
               new_init_std: float = 0.0, seed: int | None = None, verbose: bool = True) -> dict[str, float]:
    """Load `src_model_path`, grow its observation input to `new_obs_size`, save to `out_path`.

    Returns the parity diffs (see `parity_diffs`). Raises if N < old size, or if
    new_init_std == 0 and the new model does not reproduce the source within 1e-5.
    """
    from sb3_contrib import MaskablePPO

    src = MaskablePPO.load(str(src_model_path), device="cpu", custom_objects={"seed": None})
    dst, _, diffs = expand_model(src, new_obs_size, new_init_std, seed=seed, verbose=verbose)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst.save(str(out_path))
    if verbose:
        print(f"[surgery] saved {out_path} (source {src_model_path} had {src.num_timesteps:,} training steps; "
              "new model starts at 0 with a fresh optimizer)")
    return diffs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grow a MaskablePPO's observation input, preserving its behaviour.")
    p.add_argument("--src", required=True, help="source MaskablePPO .zip (e.g. checkpoints/champion.zip)")
    p.add_argument("--obs-size", type=int, required=True, help="new observation size N (>= source size)")
    p.add_argument("--out", required=True, help="output .zip path")
    p.add_argument("--new-init-std", type=float, default=0.0,
                   help="std of the N(0, std) init for weights on the new inputs (0 = exact function copy)")
    p.add_argument("--seed", type=int, default=None, help="torch seed for the --new-init-std noise")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    th.set_num_threads(1)
    try:
        expand_obs(args.src, args.obs_size, args.out, args.new_init_std, seed=args.seed)
    except (ValueError, AssertionError) as exc:
        raise SystemExit(f"[surgery] error: {exc}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
