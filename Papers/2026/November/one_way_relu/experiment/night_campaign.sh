#!/bin/bash
# Night campaign 2026-11-15 (Heath asleep; Bocher attending).
# Phase 0: wait for any running factorial (incl. alpha=1 arm) to finish.
# Phase 1: statistical power — relu/leaky_10/oneway_10 seeds 5..24 (20 more each).
# Phase 2: grokking probe — 100 epochs, 3 arms x 3 seeds, checkpoints saved.
# Phase 3: oneway_5 + oneway_1000 seed extension (5..14) for the dose curve.
cd /mnt/data/shared/one_way_leaky
while pgrep -f 'factorial_emnist[.]py --' > /dev/null; do sleep 30; done

echo "══ PHASE 1: seed extension to n=25 ══" >> night.log
python3 factorial_emnist.py --arms relu leaky_10 oneway_10 \
  --seed-start 5 --seeds 20 --epochs 10 --device cuda \
  --out results_emnist.jsonl >> night.log 2>&1

echo "══ PHASE 2: 100-epoch grokking probe ══" >> night.log
python3 factorial_emnist.py --arms relu leaky_10 oneway_10 \
  --seed-start 100 --seeds 3 --epochs 100 --device cuda \
  --out results_long.jsonl --save-ckpt ckpt_long >> night.log 2>&1

echo "══ PHASE 3: dose-curve seed extension ══" >> night.log
python3 factorial_emnist.py --arms oneway_5 oneway_1000 \
  --seed-start 5 --seeds 10 --epochs 10 --device cuda \
  --out results_emnist.jsonl >> night.log 2>&1

echo "══ CAMPAIGN COMPLETE $(date -u) ══" >> night.log
