#!/bin/bash
# E7: v3 fine-tune vs matched v2 control, then 500-battle evals + head-to-head.
# Sequential, one job at a time, 4 in-process envs (CPU budget rule).
set -u
cd "$(dirname "$0")/.."
caffeinate -i -s -w $$ &   # keep the Mac awake until this script exits
PY=.venv/bin/python
ts() { date "+%H:%M:%S"; }
echo "[$(ts)] E7 start"

echo "[$(ts)] v3ft train"
$PY -m pokebot.train --run-name v3ft --features v3 --init-from checkpoints/v3_init/init.zip \
    --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 \
    --n-envs 4 --eval-battles 200 > logs/v3ft-train.log 2>&1
echo "[$(ts)] v3ft exit $?"

echo "[$(ts)] v2ft train"
$PY -m pokebot.train --run-name v2ft --features v2 --init-from checkpoints/champion.zip \
    --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 \
    --n-envs 4 --eval-battles 200 > logs/v2ft-train.log 2>&1
echo "[$(ts)] v2ft exit $?"

for run in v3ft v2ft; do
  echo "[$(ts)] eval $run (500/opponent)"
  $PY -m pokebot.evaluate --model checkpoints/$run/final.zip --battles 500 > logs/e7-eval-$run.log 2>&1
  echo "[$(ts)] eval $run exit $?"
done
echo "[$(ts)] head-to-head v3ft vs v2ft (500)"
$PY -m pokebot.evaluate --model checkpoints/v3ft/final.zip --compare checkpoints/v2ft/final.zip --battles 500 > logs/e7-h2h.log 2>&1
echo "[$(ts)] h2h exit $?"
echo "[$(ts)] head-to-head v3ft vs champion (500)"
$PY -m pokebot.evaluate --model checkpoints/v3ft/final.zip --compare checkpoints/champion.zip --battles 500 > logs/e7-h2h-champ.log 2>&1
echo "[$(ts)] h2h-champ exit $?"
echo "[$(ts)] E7 done"
