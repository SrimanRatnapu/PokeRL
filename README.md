# Pokémon Showdown RL Bot

A reinforcement-learning agent for Pokémon Showdown **gen9randombattle** singles. It is
trained with masked PPO (sb3-contrib `MaskablePPO`) on top of
[poke-env](https://github.com/hsahovic/poke-env), using a local Showdown server. The
curriculum runs from scripted baselines to self-play. Once it's trained, you can battle it
in your browser.

## Results

The best model is **`checkpoints/champion.zip`**: v2 features, PPO from scratch, 4M steps (about 2.5 hours of training on 4 in-process envs). Over 500 battles per opponent:

| opponent | win rate (95% CI) |
|---|---|
| RandomPlayer | 98.6% [97.1, 99.3] |
| MaxBasePowerPlayer | 92.0% [89.3, 94.1] |
| HeuristicBot (`bots/heuristic_bot.py`) | 73.2% [69.2, 76.9] |
| SimpleHeuristicsPlayer | 70.6% [66.5, 74.4] |
| first RL agent (v1, 3M steps), head-to-head | 65.2% [60.9, 69.2] |

Play it with `.venv/bin/python -m pokebot.play --model checkpoints/champion.zip --verbose`. See
`docs/MORNING_REPORT.md` for the summary and `docs/DECISIONS.md` for every experiment.

## Setup

Requirements: Python 3.12, Node.js (tested with v26), and macOS or Linux.

**1. Showdown server.** It is a separate checkout of [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown) in `pokemon-showdown/` (not tracked in this repo). The random-battle set data in it is also read by the v2+ features. Set it up once, pinned to the commit this project was built against:

```bash
scripts/setup_showdown.sh                   # clone + npm install + config (first time only)
cd pokemon-showdown
node pokemon-showdown start --no-security   # serves http://localhost:8000
```

`--no-security` turns off authentication, so bots and humans can log in with any name and
no password. Keep the server running in its own terminal while you train, evaluate or play.

> Note: the setup script sets `exports.repl = false` in `pokemon-showdown/config/config.js`. The project path
> is long, and that made the REPL socket path too long for macOS (Unix socket paths have a
> length limit), which broke server startup. The REPL is only a debugging aid, so turning it
> off is harmless.

**2. Python environment:**

```bash
python3.12 -m venv .venv
.venv/bin/pip install poke-env==0.17.0 stable-baselines3==2.9.2 sb3-contrib==2.9.2 \
    gymnasium torch tensorboard pytest
```

Run every command from the project root with `.venv/bin/python`.

## Layout

```
bots/heuristic_bot.py   hand-written damage/matchup heuristic bot (baseline + example poke-env Player)
pokebot/config.py       shared constants: BATTLE_FORMAT, ACTION_SPACE_SIZE=26, RewardConfig, dirs
pokebot/INTERFACE.md    module contract shared by all the pokebot modules
pokebot/features.py     v1 features: embed_battle(battle) -> 226 floats (frozen; v1 checkpoints use it)
pokebot/features_v2.py  v2 features: v1 + damage-%/KO flags, opponent movepool threats, switch-in
                        matchups, opponent team, history -> 346 floats (default for new runs)
pokebot/features_v3.py  v3 features: v2 + held items, critical hits, battle history / past decisions -> 635 floats
pokebot/v3_items.py     v3 item + crit blocks (item priors, Choice lock, item-aware damage, crit pricing)
pokebot/v3_history.py   v3 memory block (last 6 turns, whole-battle tallies, previous action)
pokebot/features_v4.py  v4 features: v3 + abilities -> 752 floats
pokebot/v4_abilities.py v4 ability block (effects, immunities, residual damage, weather on switch-in, Intimidate)
pokebot/surgery.py      widen a trained model's input layer for a bigger feature set (starts identical)
pokebot/feature_sets.py registry: version <-> obs size; PolicyPlayer auto-detects a checkpoint's version
pokebot/bc.py           behavior cloning: record a teacher bot's battles, train a policy to imitate it
docs/DECISIONS.md       design & decision log with every experiment and its results (start here to learn)
data/                   recorded teacher battles for behavior cloning
pokebot/env.py          PokeBotEnv / MaskedPokeEnv (Gymnasium env with action masks),
                        make_env, make_opponent, PolicyPlayer (a Player driven by a checkpoint)
pokebot/train.py        MaskablePPO training: curriculum, reward annealing, self-play, checkpoints
pokebot/evaluate.py     win rate + 95% CI against the baseline bots, checkpoint vs checkpoint
pokebot/play.py         put the bot online so a human can challenge it (or have it challenge you)
tests/                  pytest tests
checkpoints/<run>/      saved models (created by training; only champion.zip, v2ft/v3ft finals and the
                        v3/v4 warm starts are tracked in git)
logs/<run>/             TensorBoard logs (created by training)
```

## Commands

### Training

```bash
# Smoke test: a tiny run that checks the whole pipeline end to end
.venv/bin/python -m pokebot.train --smoke --run-name smoke

# Real training run
.venv/bin/python -m pokebot.train --timesteps 2000000 --n-envs 4 --run-name run1

# Resume from a checkpoint
.venv/bin/python -m pokebot.train --timesteps 2000000 --n-envs 4 --run-name run1 \
    --resume checkpoints/run1/<checkpoint>.zip
```

Full CLI: `python -m pokebot.train --timesteps N --n-envs K --run-name NAME [--resume PATH] [--smoke]
[--features v1|v2|v3|v4] [--init-from PATH] [--start-stage 1-4] [--lr X] [--shaping-start X --shaping-end Y]`.

**When resuming**, the learning-rate and shaping schedules are computed over the *new* call, so pass
`--shaping-start 0.1 --shaping-end 0.1` (and the run's `--lr`) to avoid re-inflating them — see
`docs/DECISIONS.md` D14.

### Behavior cloning warm start ("SFT before RL")

```bash
# 1) record 10k battles of SimpleHeuristicsPlayer (~3 min)
.venv/bin/python -m pokebot.bc collect --battles 10000 --out data/bc_v2_10k.npz
# 2) supervised training of policy + value heads (~5 min)
.venv/bin/python -m pokebot.bc train --data data/bc_v2_10k.npz --out checkpoints/bc_v2/bc.zip --epochs 20 --vf-weight 0.05
# 3) PPO fine-tuning from those weights, skipping the easy curriculum stages
.venv/bin/python -m pokebot.train --run-name v2bc --init-from checkpoints/bc_v2/bc.zip --start-stage 3 --lr 1e-4 --timesteps 2000000
```

Training is safe to run on a laptop. It uses a few CPU cores and a modest amount of memory,
and you can stop it at any time and resume later. Long runs keep the CPU busy for hours, so
**keep the laptop plugged in** and somewhere it can stay cool. Use a lower `--n-envs` if
the machine gets hot.

### v3 features: warm start from the champion (trained in E7: no gain over the champion yet)

```bash
# widen the champion's input layer 346 -> 635 (already done: checkpoints/v3_init/init.zip)
.venv/bin/python -m pokebot.surgery --src checkpoints/champion.zip --obs-size 635 --out checkpoints/v3_init/init.zip
# fine-tune it with the new features (see docs/DECISIONS.md E7 for the matched v2 control run)
.venv/bin/python -m pokebot.train --run-name v3ft --features v3 --init-from checkpoints/v3_init/init.zip \
    --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 --n-envs 4
```

### v4 features: abilities (trained in E8: no gain over the champion yet)

```bash
# widen the v3 fine-tune 635 -> 752 (already done: checkpoints/v4_init/init.zip)
.venv/bin/python -m pokebot.surgery --src checkpoints/v3ft/final.zip --obs-size 752 --out checkpoints/v4_init/init.zip
# fine-tune with ability features (see docs/DECISIONS.md E8 for the matched v3 control run)
.venv/bin/python -m pokebot.train --run-name v4ft --features v4 --init-from checkpoints/v4_init/init.zip \
    --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 --n-envs 4
```

### TensorBoard

```bash
.venv/bin/tensorboard --logdir logs      # then open http://localhost:6006
```

### Evaluation

```bash
# Sanity check with no model: the heuristic bot against all baselines
.venv/bin/python -m pokebot.evaluate --model heuristic --battles 20

# A checkpoint against random, maxpower, heuristic and simple_heuristic (200 battles each)
.venv/bin/python -m pokebot.evaluate --model checkpoints/run1/<checkpoint>.zip

# Choose opponents and save replays
.venv/bin/python -m pokebot.evaluate --model ckpt.zip --opponents heuristic,simple_heuristic \
    --battles 100 --save-replays replays/

# Pit two checkpoints against each other
.venv/bin/python -m pokebot.evaluate --model new.zip --compare old.zip --battles 200
```

Options: `--battles N` (per opponent, default 200), `--opponents a,b,...`,
`--concurrency N` (default 10), `--save-replays DIR`, `--compare PATH2`, `--stochastic`
(sample actions instead of always picking the most likely one). `--model` also accepts a
baseline name (`random`, `maxpower`, `heuristic`, `simple_heuristic`). The output table shows
wins/battles, win rate, a 95% Wilson confidence interval and the mean battle length in turns.

### Playing against the bot

```bash
.venv/bin/python -m pokebot.play --model checkpoints/run1/<checkpoint>.zip
```

Then open http://localhost:8000 and click **Choose name** to pick any name (no password).
Click **Find a user**, enter `PokeBotRL`, then **Challenge**, and pick
**[Gen 9] Random Battle**.

- `--username NAME` changes the bot's name (default `PokeBotRL`)
- `--n-challenges N` sets how many battles to play before exiting (default 1; `0` means forever)
- `--challenge USER` makes the bot challenge you instead (you must be logged in as USER first)
- `--verbose` prints each turn's chosen action and the policy's top-3 action probabilities
- `--model heuristic` plays the scripted bot instead of a checkpoint

## How it works

- **Observation**: `features.embed_battle` turns the battle state into a fixed-size float
  vector, with every entry clipped to [-1, 1]. It covers both active Pokémon (HP, types,
  base stats, boosts, status), the four moves (power, type, category, effectiveness), the
  bench, weather, terrain, hazards and tera availability. The vector is built from
  fixed-size blocks, and any block that hits missing information is left as zeros, so the
  function never fails.
- **Actions**: poke-env's 26-action singles space. Actions 0–5 switch to a team slot, 6–9
  use a move, 22–25 use a move and terastallize, and 10–21 (mega, Z-move, dynamax) are never
  legal in gen 9. Every turn, an action mask marks the illegal actions, and `MaskablePPO`
  only samples legal ones.
- **Reward**: potential-based shaping, `shaping_scale * [Φ(s') − Φ(s)]`. Φ combines HP,
  faint and status differences between the two sides. A terminal ±victory bonus is added at
  the end. Potential-based shaping leaves the optimal policy unchanged. `shaping_scale` is
  annealed toward ~0.1 during training, so later on the agent optimises mostly for winning.
- **Learning**: `MaskablePPO` (sb3-contrib) runs over several parallel environments. Each
  environment is a live battle on the local server against an opponent driven in the same
  process.
- **Curriculum → self-play**: opponents get harder in stages: `RandomPlayer` →
  `MaxBasePowerPlayer` → the heuristic bots (`HeuristicBot`, `SimpleHeuristicsPlayer`) → a
  self-play pool of the agent's own past checkpoints, mixed with baselines so it doesn't
  forget how to beat them. Training periodically measures the win rate against the baselines
  and logs it to TensorBoard.
