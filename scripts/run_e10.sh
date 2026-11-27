#!/bin/bash
# E10 step 2: v5 = v4 + the lookahead's score for every action.
# Matched control: v4ft (E8): same warm start (v4_init, widened), steps, LR schedule, opponent mix.
# One job at a time, 4 in-process envs, 1 torch thread (the user asked for ~1 core, no fan).
# BETA = the best blend weight from scripts/e10_beta_sweep.sh (used for the "+ search" rows).
set -u
cd "$(dirname "$0")/.."
caffeinate -i -s -w $$ &   # keep the Mac awake until this script exits (never use a bare `wait` here)
PY=.venv/bin/python
BETA=${BETA:-2}
N=2000
ts() { date "+%H:%M:%S"; }
echo "[$(ts)] E10 start (BETA=$BETA)"

echo "[$(ts)] train v5ft (2M)"
$PY -m pokebot.train --run-name v5ft --features v5 --init-from checkpoints/v5_init/init.zip \
    --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 \
    --n-envs 4 --torch-threads 1 --eval-battles 200 > logs/v5ft-train.log 2>&1
echo "[$(ts)] train exit $?"

V5=checkpoints/v5ft/final.zip
echo "[$(ts)] eval v5ft vs bots ($N each)"
$PY -m pokebot.evaluate --model $V5 --battles $N \
    --opponents random,maxpower,heuristic,simple_heuristic,lookahead > logs/e10-eval-v5ft.log 2>&1
echo "[$(ts)] eval v4ft vs bots ($N each)"
$PY -m pokebot.evaluate --model checkpoints/v4ft/final.zip --battles $N \
    --opponents heuristic,simple_heuristic,lookahead > logs/e10-eval-v4ft.log 2>&1
echo "[$(ts)] h2h v5ft vs v4ft (matched control)"
$PY -m pokebot.evaluate --model $V5 --compare checkpoints/v4ft/final.zip --battles $N > logs/e10-h2h-v5-v4ft.log 2>&1
echo "[$(ts)] h2h v5ft vs champion"
$PY -m pokebot.evaluate --model $V5 --compare checkpoints/champion.zip --battles $N > logs/e10-h2h-v5-champ.log 2>&1
echo "[$(ts)] v5ft + search (beta $BETA) vs lookahead and champion"
$PY -m pokebot.evaluate --model $V5 --search-beta $BETA --opponents lookahead --battles $N > logs/e10-eval-v5ft-search.log 2>&1
$PY -m pokebot.evaluate --model $V5 --search-beta $BETA --compare checkpoints/champion.zip --battles $N > logs/e10-h2h-v5search-champ.log 2>&1
echo "[$(ts)] h2h v5ft + search vs v4ft + search (both beta $BETA)"
$PY -m pokebot.evaluate --model $V5 --search-beta $BETA --compare checkpoints/v4ft/final.zip \
    --compare-search-beta $BETA --battles $N > logs/e10-h2h-v5search-v4search.log 2>&1
echo "[$(ts)] E10 done"
