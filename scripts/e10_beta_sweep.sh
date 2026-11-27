#!/bin/bash
# E10 step 1: policy + lookahead blend. Base policy v4ft; 2000 battles per cell; one job at a time.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=checkpoints/v4ft/final.zip
row() { tail -2 | head -1; }
echo "lookahead (new) vs champion: $($PY -m pokebot.evaluate --model lookahead --compare checkpoints/champion.zip --battles 2000 2>&1 | row)"
echo "lookahead (new) vs v4ft:     $($PY -m pokebot.evaluate --model lookahead --compare $M --battles 2000 2>&1 | row)"
for B in 0 0.5 1 2 4 8 100; do
  echo "beta=$B vs lookahead: $($PY -m pokebot.evaluate --model $M --search-beta $B --opponents lookahead --battles 2000 2>&1 | row)"
  echo "beta=$B vs champion:  $($PY -m pokebot.evaluate --model $M --search-beta $B --compare checkpoints/champion.zip --battles 2000 2>&1 | row)"
done
echo done
