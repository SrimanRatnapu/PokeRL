# Morning report (overnight run, finished 06:20)

## TL;DR
- **The best bot is `checkpoints/champion.zip`** (a copy of `checkpoints/v2scratch/final.zip`, 4M PPO steps).
- Over 500 battles per opponent it wins **98.6% vs random, 92.0% vs max-power, 73.2% vs the hand-written HeuristicBot and 70.6% vs poke-env's SimpleHeuristicsPlayer**. That last one is the strongest scripted bot we have, and it is also the teacher used for behavior cloning.
- Head-to-head it beats every earlier model: **65% vs the first (v1) agent**, 60% vs the BC-warm-started agent and 61% vs its own 1M-step self.
- All training stopped. Nothing is running except the local Showdown server and a `caffeinate` (both stop when you close the laptop).

## Play against it

The Showdown server must be running. If you rebooted, start it first:

```bash
cd pokemon-showdown && node pokemon-showdown start --no-security
```

Then, in a second terminal, from the project root:

```bash
.venv/bin/python -m pokebot.play --model checkpoints/champion.zip --verbose
```

Open http://localhost:8000, click **Choose name** and pick any name. Then **Find a user** → `PokeBotRL` → **Challenge** → **[Gen 9] Random Battle**. `--verbose` prints the bot's top-3 action probabilities every turn, which shows you what it is "thinking".

## What happened overnight (the story)

| run | what changed | result vs SimpleHeuristics (500 battles) |
|---|---|---|
| v1 @1M | baseline: 226 features, curriculum → self-play | ~49% |
| v1 @3M | +2M self-play steps | 54.8% |
| BC clone | imitate SimpleHeuristics (90.7% agreement) | 50.6%, which hits the imitation ceiling |
| v2bc @2M | v2 features + BC warm start + PPO | 62.6% |
| v2scratch @1M | v2 features, no BC (ablation) | 64.2% |
| **v2scratch @4M** | +3M self-play steps | **70.6%** |

The three biggest lessons are written up in `docs/DECISIONS.md`.
1. **Representation beat everything.** Telling the network "this move does ~60% damage and can KO", "their mon probably carries a Ground move" and "this switch-in takes 20% from their best move" was worth about +15 points. BC warm-starting changed how fast it learned, not how good it got (E5).
2. **Eval noise is huge.** A 200-battle eval has about ±7% noise, so `best.zip` (picked by its highest noisy score) was worse than `final.zip` three times out of three. This is the *winner's curse*. Re-test on a large fresh sample before picking a model (E2, E6).
3. **Self-play keeps improving slowly.** It is invisible on small evals, but real: the 4M model beats its 1M self 61% of the time.

## Where to read next
- `docs/DECISIONS.md`: every design decision (D1–D18) and experiment (E1–E6), each with the RL concept it teaches. Start at the top.
- TensorBoard curves: `.venv/bin/tensorboard --logdir logs`, then open http://localhost:6006.
- Training logs: `logs/*-train.log`. The final evaluation output is in `logs/final-eval.log`.

## Housekeeping
- `checkpoints/` is about 690 MB, mostly step snapshots. The ones that matter: `champion.zip`, `v2scratch/final_1M.zip`, `v2bc/final.zip`, `v1/final.zip` and `bc_v2/bc.zip`. The others (`step_*.zip`, `smoke/`) are safe to delete if you want the space.
- The CPU budget rule (one training job, 4 envs, single process) was kept for the whole night after you asked.
