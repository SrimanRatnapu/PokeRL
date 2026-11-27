#!/bin/bash
# E9: league training = PFSP + strong fixed pool + lookahead bot + one exploiter.
# Matched control: v4ft (E8), the same start (v4_init), features, steps and LR schedule, old opponent mix.
# One job at a time, 4 in-process envs, 1 torch thread (the user asked for ~1 core, no fan).
set -u
cd "$(dirname "$0")/.."
caffeinate -i -s -w $$ &   # keep the Mac awake until this script exits (never use a bare `wait` here)
PY=.venv/bin/python
ts() { date "+%H:%M:%S"; }
COMMON="--features v4 --start-stage 4 --shaping-start 0.1 --shaping-end 0.1 --n-envs 4 --torch-threads 1"
MIX="pool=0.55,lookahead=0.20,heuristic=0.10,simple_heuristic=0.10,maxpower=0.05"
EXTRA="checkpoints/champion.zip,checkpoints/v2ft/final.zip,checkpoints/v3ft/final.zip,checkpoints/v3ft2/final.zip"
echo "[$(ts)] E9 start"

# Phase A: first half of the main agent. LR 1e-4 -> 5.5e-5 = the first half of v4ft's 1e-4 -> 1e-5 over 2M.
echo "[$(ts)] league phase A (1M)"
$PY -m pokebot.train --run-name league --init-from checkpoints/v4_init/init.zip $COMMON \
    --lr 1e-4 --lr-final-frac 0.55 --timesteps 1000000 --eval-battles 200 \
    --pfsp --opponent-mix "$MIX" --pool-extra "$EXTRA" > logs/league-a-train.log 2>&1
echo "[$(ts)] phase A exit $?"
cp checkpoints/league/final.zip checkpoints/league/phase_a.zip

# Exploiter: starts as a copy of phase A and only tries to beat it (85% of games).
echo "[$(ts)] exploiter vs phase A (500k)"
$PY -m pokebot.train --run-name exploiter --init-from checkpoints/league/phase_a.zip $COMMON \
    --lr 1e-4 --timesteps 500000 --eval-battles 100 \
    --opponent-mix "policy:checkpoints/league/phase_a.zip=0.85,simple_heuristic=0.15" > logs/exploiter-train.log 2>&1
echo "[$(ts)] exploiter exit $?"
echo "[$(ts)] h2h exploiter vs phase A (500)"
$PY -m pokebot.evaluate --model checkpoints/exploiter/final.zip --compare checkpoints/league/phase_a.zip \
    --battles 500 > logs/e9-h2h-exploiter-vs-a.log 2>&1

# Phase B: second half, with the exploiter added to the fixed pool. Resuming continues the LR schedule
# from 5.5e-5 down to 1e-5 (SB3 counts progress over the combined 2M steps).
echo "[$(ts)] league phase B (1M, exploiter in pool)"
$PY -m pokebot.train --run-name league --resume checkpoints/league/phase_a.zip $COMMON \
    --lr 1e-4 --lr-final-frac 0.1 --timesteps 1000000 --eval-battles 200 \
    --pfsp --opponent-mix "$MIX" --pool-extra "$EXTRA,checkpoints/exploiter/final.zip" > logs/league-b-train.log 2>&1
echo "[$(ts)] phase B exit $?"

# Evals, one at a time.
echo "[$(ts)] eval league (500/opponent incl. lookahead)"
$PY -m pokebot.evaluate --model checkpoints/league/final.zip --battles 500 \
    --opponents random,maxpower,heuristic,simple_heuristic,lookahead > logs/e9-eval-league.log 2>&1
for m in v4ft/final champion; do
  n=$(basename "$(dirname "checkpoints/$m.zip")"); [ "$m" = champion ] && n=champion
  echo "[$(ts)] eval $n vs lookahead (500)"
  $PY -m pokebot.evaluate --model "checkpoints/$m.zip" --battles 500 --opponents lookahead > "logs/e9-eval-$n-lookahead.log" 2>&1
done
echo "[$(ts)] h2h league vs v4ft (held-out matched control, 500)"
$PY -m pokebot.evaluate --model checkpoints/league/final.zip --compare checkpoints/v4ft/final.zip \
    --battles 500 > logs/e9-h2h-v4ft.log 2>&1
echo "[$(ts)] h2h league vs champion (500)"
$PY -m pokebot.evaluate --model checkpoints/league/final.zip --compare checkpoints/champion.zip \
    --battles 500 > logs/e9-h2h-champ.log 2>&1
echo "[$(ts)] h2h league vs exploiter (500)"
$PY -m pokebot.evaluate --model checkpoints/league/final.zip --compare checkpoints/exploiter/final.zip \
    --battles 500 > logs/e9-h2h-exploiter.log 2>&1
echo "[$(ts)] E9 done"
