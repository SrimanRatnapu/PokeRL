#!/bin/bash
# E8: v4 (abilities) fine-tune vs matched v3 control, then 500-battle evals + head-to-heads.
# Two training jobs in parallel (user OK'd, 2026-11-27), 4 in-process envs each; evals in parallel.
set -u
cd "$(dirname "$0")/.."
caffeinate -i -s -w $$ &   # keep the Mac awake until this script exits
PY=.venv/bin/python
ts() { date "+%H:%M:%S"; }
echo "[$(ts)] E8 start"

# The two training runs are independent, so they run side by side (about 2 cores).
# Set E8_WAIT_PIDS="pid1 pid2" to skip training and wait on runs already in progress.
if [ -z "${E8_WAIT_PIDS:-}" ]; then
  echo "[$(ts)] v4ft + v3ft2 (control) train, in parallel"
  $PY -m pokebot.train --run-name v4ft --features v4 --init-from checkpoints/v4_init/init.zip \
      --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 \
      --n-envs 4 --eval-battles 200 > logs/v4ft-train.log 2>&1 &
  P1=$!
  $PY -m pokebot.train --run-name v3ft2 --features v3 --init-from checkpoints/v3ft/final.zip \
      --start-stage 4 --lr 1e-4 --shaping-start 0.1 --shaping-end 0.1 --timesteps 2000000 \
      --n-envs 4 --eval-battles 200 > logs/v3ft2-train.log 2>&1 &
  P2=$!
  wait $P1; echo "[$(ts)] v4ft exit $?"
  wait $P2; echo "[$(ts)] v3ft2 exit $?"
else
  echo "[$(ts)] waiting on running training pids: $E8_WAIT_PIDS"
  for pid in $E8_WAIT_PIDS; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
  echo "[$(ts)] training done"
fi

# Evals: the 4 jobs are independent (random player names), so they run side by side.
echo "[$(ts)] evals: v4ft, v3ft2 (500/opponent), h2h v4ft vs v3ft2, h2h v4ft vs champion (500) - in parallel"
PIDS=()
for run in v4ft v3ft2; do
  $PY -m pokebot.evaluate --model checkpoints/$run/final.zip --battles 500 > logs/e8-eval-$run.log 2>&1 &
  PIDS+=($!)
done
$PY -m pokebot.evaluate --model checkpoints/v4ft/final.zip --compare checkpoints/v3ft2/final.zip --battles 500 > logs/e8-h2h.log 2>&1 &
PIDS+=($!)
$PY -m pokebot.evaluate --model checkpoints/v4ft/final.zip --compare checkpoints/champion.zip --battles 500 > logs/e8-h2h-champ.log 2>&1 &
PIDS+=($!)
wait "${PIDS[@]}"   # not a bare `wait`: that would also wait on caffeinate, which waits on us
echo "[$(ts)] evals finished"
echo "[$(ts)] E8 done"
