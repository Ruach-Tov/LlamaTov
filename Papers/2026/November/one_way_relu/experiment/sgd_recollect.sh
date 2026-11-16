#!/bin/bash
# SGD recollection (Heath, 2026-11-16): n=50 pairwise-matched to Adam results.
# Same seeds 0-49 as the Adam arms -> paired cross-optimizer differences.
# Checkpoints saved for latent analysis (does the latent inversion survive SGD?).
cd "${ONEWAY_DIR:-$(dirname "$0")}"
R="--epochs 10 --device cuda --optimizer sgd --amp-log --out results_sgd.jsonl --save-ckpt ckpt_sgd_e10"
echo "== SGD RECOLLECT n=50 start $(date -u) ==" >> night.log
python3 factorial_emnist.py --arms relu leaky_10 oneway_10 --seed-start 10 --seeds 40 $R >> night.log 2>&1
echo "== SGD RECOLLECT: ckpt backfill seeds 0-9 ==" >> night.log
python3 factorial_emnist.py --arms relu leaky_10 oneway_10 --seed-start 0 --seeds 10 --epochs 10 --device cuda --optimizer sgd --out /dev/null --save-ckpt ckpt_sgd_e10 >> night.log 2>&1
echo "== SGD RECOLLECT COMPLETE $(date -u) ==" >> night.log
