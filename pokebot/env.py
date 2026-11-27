"""Environment layer: poke-env SinglesEnv subclass + a masked single-agent Gymnasium env.

See pokebot/INTERFACE.md for the contract. Summary:

* `PokeBotEnv`     - PettingZoo-style SinglesEnv (two internal websocket agents),
                     observations from the chosen feature version (see
                     `pokebot.feature_sets`; "v1" = 226 dims, "v2" = superset), shaped
                     potential-based reward per `RewardConfig`.
* `MaskedPokeEnv`  - Gymnasium env for sb3_contrib.MaskablePPO: Box obs,
                     Discrete(26) actions, `action_masks()`, swappable opponent.
* `make_env`, `make_opponent`, `PolicyPlayer`.

Feature versions: envs take `feature_version` (default "v2"). `PolicyPlayer`
infers the version from its model's observation size, so a v1 checkpoint plays
correctly anywhere (networked, or as a self-play opponent inside a v2 env).

Each `MaskedPokeEnv` owns its own `PokeBotEnv`, which owns its own asyncio loop
thread and two players with random usernames, so several can coexist in one
process (e.g. inside a DummyVecEnv).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from weakref import WeakKeyDictionary

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.player.battle_order import BattleOrder

from pokebot.config import ACTION_SPACE_SIZE, BATTLE_FORMAT, PROJECT_ROOT, RewardConfig
from pokebot.feature_sets import DEFAULT_FEATURE_VERSION, get_feature_set, version_for_model

# `bots/` lives at the project root; make it importable even when the caller's
# cwd / sys.path doesn't include the root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("pokebot.env")

N_POKEMON = 6


def _observation_space(feature_version: str = DEFAULT_FEATURE_VERSION) -> Box:
    obs_size, _ = get_feature_set(feature_version)
    return Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)


# --------------------------------------------------------------------------- #
# PettingZoo-level env
# --------------------------------------------------------------------------- #
class PokeBotEnv(SinglesEnv):
    """SinglesEnv with our features and a scaled potential-based reward.

    reward_t = shaping_scale * [Phi(s_t) - Phi(s_{t-1})] + (+victory | -victory | 0 at the end)

    Phi = sum over our mons of (hp * hp_frac - fainted * is_fainted - status * has_status)
          minus the same for the opponent's mons (unrevealed mons count as full HP),
    exactly like poke-env's `reward_computing_helper`, but the victory term is kept
    outside the shaping scale.

    Defaults differ from SinglesEnv in one place: `strict=False`, so an illegal
    action (should not happen with masking) or an inconvertible opponent order
    falls back to a random legal move with a logged warning instead of crashing
    a long training run. Pass `strict=True` to get exceptions instead.
    """

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        feature_version: str = DEFAULT_FEATURE_VERSION,
        **singles_env_kwargs: Any,
    ):
        self.feature_version = feature_version
        _, self._embed_fn = get_feature_set(feature_version)  # validates the version early
        singles_env_kwargs.setdefault("battle_format", BATTLE_FORMAT)
        singles_env_kwargs.setdefault("strict", False)
        super().__init__(**singles_env_kwargs)
        self.reward_config: RewardConfig = reward_config if reward_config is not None else RewardConfig()
        self.observation_spaces = {agent: _observation_space(feature_version) for agent in self.possible_agents}
        # battle -> last potential; battle -> whether the terminal reward was paid
        self._potentials: WeakKeyDictionary[AbstractBattle, float] = WeakKeyDictionary()
        self._terminal_paid: WeakKeyDictionary[AbstractBattle, bool] = WeakKeyDictionary()

    # ---- PokeEnv abstract methods ------------------------------------------ #
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return self._embed_fn(battle)

    def calc_reward(self, battle: AbstractBattle) -> float:
        cfg = self.reward_config
        phi = self._potential(battle, cfg)
        prev = self._potentials.get(battle, 0.0)
        self._potentials[battle] = phi
        reward = float(cfg.shaping_scale) * (phi - prev)
        if battle.finished and not self._terminal_paid.get(battle, False):
            self._terminal_paid[battle] = True
            if battle.won:
                reward += cfg.victory
            elif battle.lost:
                reward -= cfg.victory
        return float(reward)

    @staticmethod
    def _potential(battle: AbstractBattle, cfg: RewardConfig) -> float:
        value = 0.0
        for mon in battle.team.values():
            value += mon.current_hp_fraction * cfg.hp
            if mon.fainted:
                value -= cfg.fainted
            elif mon.status is not None:
                value -= cfg.status
        value += (N_POKEMON - len(battle.team)) * cfg.hp
        for mon in battle.opponent_team.values():
            value -= mon.current_hp_fraction * cfg.hp
            if mon.fainted:
                value += cfg.fainted
            elif mon.status is not None:
                value += cfg.status
        value -= (N_POKEMON - len(battle.opponent_team)) * cfg.hp
        return value

    # ---- cleanup ------------------------------------------------------------ #
    def shutdown(self, timeout: float = 5.0) -> None:
        """Close websockets and stop this env's private event loop."""
        for agent in (self.agent1, self.agent2):
            # poke-env logs cancelled message handlers at CRITICAL; silence this env's agents.
            agent.logger.setLevel(logging.CRITICAL + 1)
        for agent in (self.agent1, self.agent2):
            try:
                fut = asyncio.run_coroutine_threadsafe(agent.ps_client._stop_listening(), self._loop)
                fut.result(timeout=timeout)
            except Exception:  # already closed / never connected
                pass

        async def _cancel_remaining() -> None:
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_cancel_remaining(), self._loop).result(timeout=timeout)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass


class _SafeSingleAgentWrapper(SingleAgentWrapper):
    """SingleAgentWrapper whose reset tolerates an opponent Player shared between envs.

    The stock reset calls `opponent.reset_battles()`, which raises if the same
    opponent object is currently registered in another env's unfinished battle.
    We only drop that opponent's *finished* battles instead.
    """

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed, options)
        opp = self.opponent
        try:
            opp._battles = {tag: b for tag, b in opp._battles.items() if not b.finished}
        except Exception:
            pass
        assert self.env.battle2 is not None
        opp._battles[self.env.battle2.battle_tag] = self.env.battle2
        self._np_random = self.env._np_random
        return obs[self.env.agent1.username], infos[self.env.agent1.username]


# --------------------------------------------------------------------------- #
# Gymnasium single-agent env for MaskablePPO
# --------------------------------------------------------------------------- #
class MaskedPokeEnv(gym.Env):
    """Single-agent Gymnasium env usable by sb3_contrib.MaskablePPO.

    Observations: Box(-1, 1, (obs_size of `feature_version`,), float32). Actions: Discrete(26).
    `action_masks()` returns the legal-action mask for the current observation.
    At episode end `info` contains `won` (bool), plus `lost`, `tie`, `turns`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: Player | None = None,
        reward_config: RewardConfig | None = None,
        battle_format: str = BATTLE_FORMAT,
        feature_version: str = DEFAULT_FEATURE_VERSION,
        **poke_env_kwargs: Any,
    ):
        super().__init__()
        self.battle_format = battle_format
        self.feature_version = feature_version
        self.poke_env = PokeBotEnv(
            reward_config=reward_config, battle_format=battle_format, feature_version=feature_version, **poke_env_kwargs
        )
        if opponent is None:
            opponent = make_opponent("random", battle_format=battle_format)
        self._wrapper = _SafeSingleAgentWrapper(self.poke_env, opponent)
        self._pending_opponent: Player | None = None
        self.observation_space = _observation_space(feature_version)
        self.action_space = Discrete(ACTION_SPACE_SIZE)
        self._mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        self._mask[0] = True
        self._closed = False

    # ---- properties ---------------------------------------------------------- #
    @property
    def opponent(self) -> Player:
        return self._wrapper.opponent

    @property
    def reward_config(self) -> RewardConfig:
        return self.poke_env.reward_config

    @property
    def battle(self) -> Optional[AbstractBattle]:
        """Our side's current battle object (None before the first reset)."""
        return self.poke_env.battle1

    # ---- contract methods --------------------------------------------------- #
    def set_opponent(self, player: Player) -> None:
        """Swap the opponent; takes effect on the next reset()."""
        self._pending_opponent = player

    def set_reward_config(self, cfg: RewardConfig) -> None:
        """Takes effect immediately (next reward computation)."""
        self.poke_env.reward_config = cfg

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self._pending_opponent is not None:
            self._wrapper.opponent = self._pending_opponent
            self._pending_opponent = None
        obs, info = self._wrapper.reset(seed=seed, options=options)
        return self._unpack(obs), dict(info)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.int64(int(np.asarray(action).reshape(-1)[0]))
        obs, reward, terminated, truncated, info = self._wrapper.step(action)
        info = dict(info)
        if terminated or truncated:
            battle = self.poke_env.battle1
            won = bool(battle is not None and battle.won)
            lost = bool(battle is not None and battle.lost)
            info["won"] = won
            info["lost"] = lost
            info["tie"] = not won and not lost
            info["turns"] = int(battle.turn) if battle is not None else 0
        return self._unpack(obs), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.poke_env.close(force=True, wait=True)
        except Exception as e:  # never let cleanup crash the caller
            logger.warning("MaskedPokeEnv.close: env close failed: %s", e)
        self.poke_env.shutdown()

    def render(self):
        return None

    # ---- helpers ------------------------------------------------------------- #
    def _unpack(self, obs: Dict[str, Any]) -> np.ndarray:
        mask = np.asarray(obs["action_mask"]).astype(bool)
        if not mask.any():
            # No legal action reported (e.g. a transient state). Any action is
            # converted with strict=False -> random legal move, so allow slot 0.
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            mask[0] = True
        self._mask = mask
        return np.asarray(obs["observation"], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def make_env(
    opponent: Player | None = None,
    reward_config: RewardConfig | None = None,
    battle_format: str = BATTLE_FORMAT,
    feature_version: str = DEFAULT_FEATURE_VERSION,
) -> MaskedPokeEnv:
    """Build a MaskedPokeEnv. Opponent defaults to a (non-listening) RandomPlayer.

    `feature_version` picks the observation features ("v1" or "v2"); it only affects
    the learning agent's observations, not the opponent (a PolicyPlayer opponent
    always uses its own model's version).
    """
    return MaskedPokeEnv(
        opponent=opponent, reward_config=reward_config, battle_format=battle_format, feature_version=feature_version
    )


def make_opponent(kind: str, **kwargs: Any) -> Player:
    """Build an opponent Player for use inside MaskedPokeEnv.

    kind: "random" | "maxpower" | "heuristic" | "simple_heuristic" | "policy".
    For "policy": pass `model=` or `model_path=` (and optionally `deterministic=`).
    Remaining kwargs go to the Player constructor. start_listening defaults to
    False and battle_format to BATTLE_FORMAT.
    """
    kwargs.setdefault("battle_format", BATTLE_FORMAT)
    kwargs.setdefault("start_listening", False)
    kind = kind.lower()
    if kind == "random":
        return RandomPlayer(**kwargs)
    if kind == "maxpower":
        return MaxBasePowerPlayer(**kwargs)
    if kind == "simple_heuristic":
        return SimpleHeuristicsPlayer(**kwargs)
    if kind == "heuristic":
        from bots.heuristic_bot import HeuristicBot

        return HeuristicBot(**kwargs)
    if kind == "policy":
        return PolicyPlayer(
            model=kwargs.pop("model", None),
            model_path=kwargs.pop("model_path", None),
            deterministic=kwargs.pop("deterministic", True),
            **kwargs,
        )
    raise ValueError(f"Unknown opponent kind {kind!r}")


# --------------------------------------------------------------------------- #
# Trained-policy player
# --------------------------------------------------------------------------- #
class PolicyPlayer(Player):
    """A Player driven by a trained MaskablePPO model.

    Works as a sync opponent inside MaskedPokeEnv (self-play; build with
    start_listening=False) and as a normal networked Player (battle_against,
    accept_challenges, ladder).

    The feature version is detected from `model.observation_space.shape[0]`
    (`self.feature_version`), so v1 and v2 checkpoints both embed correctly. If the
    size matches no known version, every move falls back to a random legal move.
    """

    def __init__(
        self,
        model=None,
        model_path: Union[str, Path, None] = None,
        deterministic: bool = True,
        **player_kwargs: Any,
    ):
        if model is None:
            if model_path is None:
                raise ValueError("PolicyPlayer needs model= or model_path=")
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(str(model_path), device="cpu", custom_objects={"seed": None})
        self.model = model
        self.model_path = str(model_path) if model_path is not None else None
        self.deterministic = deterministic
        self.feature_version = version_for_model(model)
        self._embed_fn = get_feature_set(self.feature_version)[1] if self.feature_version else None
        player_kwargs.setdefault("battle_format", BATTLE_FORMAT)
        super().__init__(**player_kwargs)
        if self._embed_fn is None:
            shape = getattr(getattr(model, "observation_space", None), "shape", None)
            self.logger.warning(
                "PolicyPlayer: model observation shape %s matches no known feature version; playing random moves",
                shape,
            )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        if self._embed_fn is None:
            return self.choose_random_move(battle)
        try:
            obs = self._embed_fn(battle)
            mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=bool)
            if not mask.any():
                return self.choose_random_move(battle)
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
            action = np.int64(int(np.asarray(action).reshape(-1)[0]))
            return SinglesEnv.action_to_order(action, battle, fake=False, strict=False)
        except Exception as e:
            self.logger.warning("PolicyPlayer.choose_move failed (%s); playing a random move", e)
            return self.choose_random_move(battle)


__all__ = ["PokeBotEnv", "MaskedPokeEnv", "make_env", "make_opponent", "PolicyPlayer"]
