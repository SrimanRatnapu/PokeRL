"""Shared constants and configuration for the Pokémon Showdown RL bot."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

BATTLE_FORMAT = "gen9randombattle"

# poke-env SinglesEnv action space for gen 9:
#   0-5   switch to team slot i (order of battle.team)
#   6-9   use move i
#   10-21 mega / z-move / dynamax variants (always masked in gen 9)
#   22-25 use move i + terastallize
ACTION_SPACE_SIZE = 26


@dataclass
class RewardConfig:
    """Potential-based shaped reward weights (see poke-env reward_computing_helper).

    reward_t = shaping_scale * [Φ(s_{t+1}) - Φ(s_t)] (HP/faint/status terms)
               + terminal ±victory
    Anneal shaping_scale toward ~0.1 late in training so win/loss dominates.
    """

    fainted: float = 2.0
    hp: float = 1.0
    status: float = 0.5
    victory: float = 30.0
    shaping_scale: float = 1.0
