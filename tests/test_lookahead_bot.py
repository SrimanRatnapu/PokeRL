"""Tests for the one-turn lookahead bot (bots/lookahead_bot.py).

    .venv/bin/python -m pytest tests/test_lookahead_bot.py -q -s
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bots.lookahead_bot import LookaheadBot  # noqa: E402
from pokebot.env import make_opponent  # noqa: E402
from tests.test_v4_abilities import _team_battle  # noqa: E402


def _bot():
    return make_opponent("lookahead")


def _scores(bot, battle):
    return {getattr(a, "species", getattr(a, "id", None)): v for v, a, _ in bot.score_actions(battle)}


def test_factory_builds_it():
    assert isinstance(_bot(), LookaheadBot)


def test_lost_game_sends_tyranitar():
    """The E8 trigger position: forced switch, their Gyarados on 2%, no weather. Sand KOs it."""
    b = _team_battle([("Tyranitar", "sandstream"), ("Espeon", "magicbounce"), ("Eiscue", "iceface")],
                     "Gyarados", "2/100", fainted=["Eiscue"])
    b._force_switch = True
    b._available_switches = [m for m in b.team.values() if not m.fainted]
    b._available_moves = []
    s = _scores(_bot(), b)
    print("\n", s)
    assert s["tyranitar"] > s["espeon"]
