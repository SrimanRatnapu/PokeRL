#!/bin/bash
# Wait for the beta sweep, pick the best beta, then run the E10 training + evals.
cd "$(dirname "$0")/.."
caffeinate -i -s -w $$ &
while pgrep -f e10_beta_sweep.sh > /dev/null; do sleep 30; done
BETA=$(.venv/bin/python scripts/e10_pick_beta.py logs/e10-beta-sweep.log)
echo "picked BETA=$BETA"
BETA=$BETA scripts/run_e10.sh
