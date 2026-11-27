"""Policy + search at decision time: blend a trained policy with the one-turn lookahead.

For every legal action a:

    score(a) = log pi(a | s) + beta * lookahead(a)

and play the argmax. `log pi` is the trained MaskablePPO policy's (masked) log-probability,
`lookahead(a)` is `LookaheadScorer.action_scores` (units: HP of a full-health mon). beta = 0 is
the plain policy; a very large beta is the lookahead bot with the policy only breaking ties.
This is the "learned prior + search" idea of AlphaGo-style programs, with a one-ply search.

Evaluate (Showdown server on localhost:8000):
    .venv/bin/python -m pokebot.evaluate --model checkpoints/v4ft/final.zip --search-beta 2 \
        --opponents lookahead --battles 2000
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder

from bots.lookahead_bot import LookaheadScorer
from pokebot.env import PolicyPlayer


def policy_log_probs(model, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked log-probabilities of every action (−inf for illegal ones)."""
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_t, action_masks=mask.reshape(1, -1))
        logits = dist.distribution.logits.cpu().numpy().reshape(-1).astype(np.float64)
    logits[~mask] = -np.inf
    return logits


def blend(log_probs: np.ndarray, scores: np.ndarray, mask: np.ndarray, beta: float) -> int:
    """argmax over legal actions of log_probs + beta * scores (unscored legal actions get the worst score)."""
    s = scores.astype(np.float64).copy()
    legal_scored = mask & ~np.isnan(s)
    fill = s[legal_scored].min() if legal_scored.any() else 0.0
    s[np.isnan(s)] = fill
    total = np.where(mask, log_probs + beta * s, -np.inf)
    return int(np.argmax(total))


class SearchPlayer(PolicyPlayer):
    """PolicyPlayer that re-ranks the policy's actions with the lookahead (see module docstring)."""

    def __init__(self, *args: Any, beta: float = 2.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        self._lookahead = LookaheadScorer()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, Battle) or self._embed_fn is None:
            return super().choose_move(battle)
        try:
            mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=bool)
            if not mask.any():
                return self.choose_random_move(battle)
            logp = policy_log_probs(self.model, self._embed_fn(battle), mask)
            scores = self._lookahead.action_scores(battle)
            action = blend(logp, scores, mask, self.beta)
            return SinglesEnv.action_to_order(np.int64(action), battle, fake=False, strict=False)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("SearchPlayer.choose_move failed (%s); using the plain policy", e)
            return super().choose_move(battle)


__all__ = ["SearchPlayer", "blend", "policy_log_probs"]
