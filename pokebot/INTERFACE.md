# Module contract (shared by all contributors)

Python: `.venv/bin/python` (3.12). poke-env 0.17.0, gymnasium 1.4, stable-baselines3 / sb3-contrib 2.9.2, torch 2.15.
Server: local Showdown at `localhost:8000` (already running; do not start/stop it).
Format: `gen9randombattle`. Run everything from the project root.

## `pokebot/config.py` (exists)
`BATTLE_FORMAT`, `ACTION_SPACE_SIZE = 26`, `RewardConfig`, `CHECKPOINT_DIR`, `LOG_DIR`.

## `pokebot/features.py` (owner: env agent)
```python
OBS_SIZE: int
def embed_battle(battle: poke_env.battle.Battle) -> np.ndarray  # float32, shape (OBS_SIZE,), roughly in [-1, 1]
```
Pure function of the battle; must never raise on missing info (unknown opp mons, None active, etc.).

## Feature versions: `pokebot/feature_sets.py`
```python
FEATURE_SETS = {"v1": (226, features.embed_battle), "v2": (OBS_SIZE_V2, features_v2.embed_battle_v2)}
def get_feature_set(version: str) -> tuple[int, Callable]   # (obs_size, embed_fn)
def version_for_obs_size(n: int) -> str | None
def version_for_model(model) -> str | None                  # from model.observation_space.shape[0]
```
`features.py` is v1 and frozen (`OBS_SIZE`/`embed_battle` remain the v1 names; also `OBS_SIZE_V1`/`embed_battle_v1`).
`features_v2.py`: `OBS_SIZE_V2` (346), `embed_battle_v2` = v1 vector + damage/threat/team/history blocks (`BLOCK_SLICES_V2`).

## `pokebot/env.py` (owner: env agent)
```python
class PokeBotEnv(poke_env.environment.SinglesEnv):
    def __init__(self, reward_config: RewardConfig | None = None, **singles_env_kwargs): ...
    # embed_battle -> feature_sets[feature_version] embed fn; calc_reward -> shaped reward per RewardConfig
    reward_config: RewardConfig   # mutable at runtime (train.py anneals shaping_scale)

class MaskedPokeEnv(gymnasium.Env):
    """Single-agent Gymnasium env usable by sb3_contrib.MaskablePPO."""
    observation_space: Box(shape=(OBS_SIZE,), float32)
    action_space: Discrete(ACTION_SPACE_SIZE)
    def reset(self, *, seed=None, options=None) -> (obs, info)
    def step(self, action) -> (obs, reward, terminated, truncated, info)
        # info on episode end includes {"won": bool}
    def action_masks(self) -> np.ndarray[bool]     # shape (ACTION_SPACE_SIZE,)
    def set_opponent(self, player: poke_env.player.Player) -> None  # takes effect on next reset()
    def set_reward_config(self, cfg: RewardConfig) -> None
    def close(self) -> None

def make_env(opponent: Player | None = None, reward_config: RewardConfig | None = None,
             battle_format: str = BATTLE_FORMAT, feature_version: str = "v2") -> MaskedPokeEnv
    # opponent default: RandomPlayer. Opponents passed here are driven synchronously by
    # SingleAgentWrapper, so build them with start_listening=False (see make_opponent).

def make_opponent(kind: str, **kwargs) -> Player
    # kind in {"random", "maxpower", "heuristic", "simple_heuristic", "policy"}
    # "heuristic"        -> bots/heuristic_bot.py HeuristicBot
    # "simple_heuristic" -> poke_env SimpleHeuristicsPlayer
    # "policy"           -> PolicyPlayer(model=kwargs["model"] or model_path=kwargs["model_path"])
    # always start_listening=False, battle_format=BATTLE_FORMAT

class PolicyPlayer(poke_env.player.Player):
    """A Player driven by a trained MaskablePPO model. Usable both as a sync
    opponent inside MaskedPokeEnv (self-play) and as a normal networked Player
    (battle_against / accept_challenges / ladder) for evaluation and play."""
    def __init__(self, model=None, model_path: str | None = None, deterministic: bool = True, **player_kwargs): ...
    feature_version: str | None   # auto-detected from model.observation_space.shape[0] (v1 checkpoints keep working)
    def choose_move(self, battle) -> BattleOrder   # sync; uses its version's embed fn + SinglesEnv.get_action_mask + SinglesEnv.action_to_order
```

## `pokebot/train.py` (owner: training agent)
CLI: `python -m pokebot.train --timesteps N --n-envs K --run-name NAME [--resume PATH] [--features v1|v2]`
(`--features` defaults to v2, is stored in train_state.json, and is taken from the run on `--resume`.)
MaskablePPO, curriculum random -> maxpower -> heuristic/simple_heuristic -> self-play pool of past checkpoints,
shaping anneal, checkpoints to `checkpoints/<run>/`, TensorBoard to `logs/<run>/`, periodic eval win rate.

## `pokebot/evaluate.py`, `pokebot/play.py` (owner: eval agent)
`python -m pokebot.evaluate --model PATH --battles N` -> win rate + 95% CI vs each baseline.
`python -m pokebot.play --model PATH ...` -> bot logs in to local server and accepts challenges from a human.

## v3 (`pokebot/features_v3.py`)
`OBS_SIZE_V3` (635), `embed_battle_v3` = v2 vector (346, byte-for-byte) + `items` (71) + `crit` (22) + `history` (196);
offsets in `BLOCK_SLICES_V3`. Blocks: `v3_items.items_block / crit_block`, `v3_history.history_block` — each
`fn(battle) -> list[float]` of fixed size, in [-1, 1], never raises. Registered as `"v3"` in `feature_sets`.
`pokebot/surgery.py`: `expand_obs(src, new_obs_size, out)` widens a model's input layers with zero weights
(function-preserving) so `train.py --init-from` can fine-tune it on the larger feature set.

## v4 (`pokebot/features_v4.py`)
- `embed_battle_v4(battle)` = the 635 v3 entries unchanged + `v4_abilities.abilities_block(battle)` (117) = 752 floats in [-1, 1]; never raises.
- Registered as `"v4"` in `feature_sets.FEATURE_SETS`; `version_for_obs_size(752) == "v4"`, so PolicyPlayer / evaluate / play auto-detect v4 checkpoints.
- Layout of the ability block is documented at the top of `pokebot/v4_abilities.py`. Team-slot features (indices 75–116) follow `battle.team` order, like switch actions 0–5.
- Warm start: `checkpoints/v4_init/init.zip` (v3ft widened via `pokebot.surgery`).
