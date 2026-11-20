#!/usr/bin/env bash
# Show the full-model op listing. Usage: ops.sh [--summary|--f16-only|--layer N]
python3 /mnt/data/home/iyun/Ruach-Tov/bpd-substrate/tools/model_op_listing.py \
  /mnt/data/home/iyun/n64chk/manifest.tsv "$@"
