"""Tests for pokebot.surgery (function-preserving observation expansion).

Uses small freshly built models only (no server, no champion, no training):
    .venv/bin/python -m pytest tests/test_surgery.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th
from gymnasium.spaces import Box, Discrete

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sb3_contrib import MaskablePPO  # noqa: E402

from pokebot.surgery import (  # noqa: E402
    PARITY_TOL, _SpacesOnlyEnv, expand_model, expand_obs, main, parity_diffs,
)

OLD, NEW, N_ACT = 12, 20, 26
NET_ARCH = dict(net_arch=dict(pi=[16, 16], vf=[16, 16]))
INPUT_LAYERS = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")


def _small_model(obs_size: int = OLD) -> MaskablePPO:
    env = _SpacesOnlyEnv(Box(-1.0, 1.0, (obs_size,), np.float32), Discrete(N_ACT))
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=NET_ARCH, n_steps=64, batch_size=32,
                        ent_coef=0.02, gamma=0.97, seed=None, device="cpu", verbose=0)
    # give biases non-trivial values so the parity check is meaningful
    with th.no_grad():
        for p in model.policy.parameters():
            p.add_(0.1 * th.randn_like(p))
    return model


@pytest.fixture()
def src_path(tmp_path):
    path = tmp_path / "src.zip"
    _small_model().save(str(path))
    return path


def test_zero_init_preserves_function():
    src = _small_model()
    dst, _, diffs = expand_model(src, NEW, verbose=False)
    assert max(diffs["logits"], diffs["probs"], diffs["values"]) < PARITY_TOL
    assert diffs["argmax_agree"] == 1.0
    # independent check with a different batch
    again = parity_diffs(src.policy, dst.policy, OLD, NEW, n=512, seed=123)
    assert max(again["logits"], again["probs"], again["values"]) < PARITY_TOL
    # new inputs are ignored entirely: changing obs[OLD:] does not change the outputs
    obs = th.rand(64, NEW) * 2 - 1
    obs2 = obs.clone()
    obs2[:, OLD:] = th.rand(64, NEW - OLD) * 2 - 1
    with th.no_grad():
        assert th.equal(dst.policy.predict_values(obs), dst.policy.predict_values(obs2))


def test_noise_init_is_nonzero_but_bounded():
    src = _small_model()
    _, _, diffs = expand_model(src, NEW, new_init_std=0.01, seed=0, verbose=False)
    assert diffs["logits"] > 0
    assert diffs["probs"] < 0.1
    # bounded by the size of the perturbation: |dW| ~ 0.01 on 8 inputs in [-1, 1]
    assert diffs["logits"] < 1.0


def test_refuses_smaller_obs():
    with pytest.raises(ValueError, match="< source obs size"):
        expand_model(_small_model(), OLD - 1, verbose=False)


def test_same_size_is_plain_copy():
    src = _small_model()
    dst, rows, diffs = expand_model(src, OLD, verbose=False)
    assert all(action == "copied" for *_, action in rows)
    assert diffs["logits"] == 0.0


def test_shapes_param_counts_and_hparams():
    src = _small_model()
    dst, rows, _ = expand_model(src, NEW, verbose=False)
    expanded = {key for key, *_, action in rows if action.startswith("expanded")}
    assert expanded == set(INPUT_LAYERS)
    s_sd, d_sd = src.policy.state_dict(), dst.policy.state_dict()
    assert set(s_sd) == set(d_sd)
    for key, w in s_sd.items():
        if key in INPUT_LAYERS:
            assert d_sd[key].shape == (w.shape[0], NEW)
            assert th.equal(d_sd[key][:, :OLD], w)
            assert th.count_nonzero(d_sd[key][:, OLD:]) == 0
        else:
            assert th.equal(d_sd[key], w)
    n_src = sum(p.numel() for p in src.policy.parameters())
    n_dst = sum(p.numel() for p in dst.policy.parameters())
    hidden = NET_ARCH["net_arch"]["pi"][0] + NET_ARCH["net_arch"]["vf"][0]
    assert n_dst - n_src == hidden * (NEW - OLD)
    assert dst.observation_space == Box(-1.0, 1.0, (NEW,), np.float32)
    assert (dst.n_steps, dst.batch_size, dst.ent_coef, dst.gamma) == (64, 32, 0.02, 0.97)
    assert dst.policy_kwargs == src.policy_kwargs


def test_new_input_weights_receive_gradient():
    """Zero weights on new inputs still learn: dL/dW[:, j] = upstream_grad * obs[j]."""
    dst, _, _ = expand_model(_small_model(), NEW, verbose=False)
    obs = th.rand(32, NEW) * 2 - 1
    dist = dst.policy.get_distribution(obs, action_masks=np.ones((32, N_ACT), bool))
    loss = -dist.log_prob(th.zeros(32, dtype=th.long)).mean() + dst.policy.predict_values(obs).pow(2).mean()
    loss.backward()
    for name, p in dst.policy.named_parameters():
        if name in INPUT_LAYERS:
            assert p.grad[:, OLD:].abs().sum() > 0


def test_saved_model_reloads_and_init_from_path(src_path, tmp_path):
    out = tmp_path / "out" / "init.zip"
    diffs = expand_obs(src_path, NEW, out, verbose=False)
    assert diffs["logits"] < PARITY_TOL
    loaded = MaskablePPO.load(str(out), device="cpu", custom_objects={"seed": None})
    assert loaded.observation_space.shape == (NEW,)
    src = MaskablePPO.load(str(src_path), device="cpu", custom_objects={"seed": None})
    # what pokebot.train --init-from does: fresh model on the run's env, shape check, load_state_dict
    run_model = _small_model(NEW)
    assert loaded.observation_space.shape == run_model.observation_space.shape
    run_model.policy.load_state_dict(loaded.policy.state_dict())
    again = parity_diffs(src.policy, run_model.policy, OLD, NEW, n=256)
    assert max(again["logits"], again["probs"], again["values"]) < PARITY_TOL


def test_cli(src_path, tmp_path, capsys):
    out = tmp_path / "cli.zip"
    assert main(["--src", str(src_path), "--obs-size", str(NEW), "--out", str(out)]) == 0
    assert out.exists()
    assert "expanded" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--src", str(src_path), "--obs-size", str(OLD - 2), "--out", str(tmp_path / "bad.zip")])
