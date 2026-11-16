#!/bin/bash
# Campaign 2 (Heath, 2026-11-15 evening): n=50 everywhere + alpha grid {2.5,5,10,20} thousandths.
# All runs save checkpoints for classical-stats + latent analysis at scale.
cd "${ONEWAY_DIR:-$(dirname "$0")}"
R="--epochs 10 --device cuda --out results_emnist.jsonl --save-ckpt ckpt_e10"
echo "══ C2 PHASE A: core arms to n=50 ══" >> night.log
python3 factorial_emnist.py --arms relu leaky_10 oneway_10 --seed-start 25 --seeds 25 $R >> night.log 2>&1
echo "══ C2 PHASE B: new alpha points n=50 ══" >> night.log
python3 factorial_emnist.py --arms oneway_20 oneway_2.5 --seed-start 0 --seeds 50 $R >> night.log 2>&1
echo "══ C2 PHASE C: fill oneway_5 + leaky_5 to n=50 ══" >> night.log
python3 factorial_emnist.py --arms oneway_5 --seed-start 15 --seeds 35 $R >> night.log 2>&1
python3 factorial_emnist.py --arms leaky_5 --seed-start 5 --seeds 45 $R >> night.log 2>&1
echo "══ C2 PHASE D: relu ckpts for 3-way classical table (seeds 200-204) ══" >> night.log
python3 factorial_emnist.py --arms relu --seed-start 200 --seeds 5 $R --out results_ckpt10.jsonl >> night.log 2>&1
echo "══ CAMPAIGN 2 COMPLETE $(date -u) ══" >> night.log
