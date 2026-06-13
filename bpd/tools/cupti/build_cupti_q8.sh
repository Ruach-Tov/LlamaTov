#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# build_cupti_q8.sh — build cupti_q8.so, the CUPTI-from-Prolog stall instrument for our q8 GEMVs.
# Combines: cupti_bridge.c + bpd_cupti_profile.c (the PC-sampling stall bridge) + q8_gemv_launcher.c
# (run_q8_gemv / run_q8_gemv_tiled foreign preds) + combined_install.c (the install entry).
#
# Foreign preds exposed: run_q8_gemv(Cubin,M,K,Iters), run_q8_gemv_tiled(Cubin,M,K,BM,Iters),
# cupti_init, cupti_flush, cupti_stall_report(SL), cupti_total_samples(N).
#
# Usage from Prolog:
#   :- use_foreign_library('/tmp/cupti_q8.so', install_cupti_q8).
#   ... run_q8_gemv_tiled(Cubin, M, K, BM, 300), cupti_stall_report(SL).
# See bpd/tools/cupti/stall_v4_compare.pl for a worked example.
#
# ⚠️ The CRITICAL link detail (cost a debug cycle to find): libcuda is the DRIVER stub at
# /run/opengl-driver/lib on this enclave — NOT in the cuda-merged tree (whose -lcuda is missing).
# Link with BOTH -L paths.
set -e

CUDA_MERGED="${CUDA_MERGED:-/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8}"
SWI="${SWI:-/nix/store/jn4yixfq3qjdl3d4g6hfvl8nnn2pjhc5-swi-prolog-9.2.9}"
LIBCUDA_DIR="${LIBCUDA_DIR:-/run/opengl-driver/lib}"   # the driver stub, NOT the merged tree
OUT="${1:-/tmp/cupti_q8.so}"

HERE="$(cd "$(dirname "$0")" && pwd)"
SUBLIB="$HERE/../../../bpd-substrate/lib"
SWIINC="$(find "$SWI" -name 'SWI-Prolog.h' -printf '%h\n' 2>/dev/null | head -1)"

echo "SWI include: $SWIINC ; libcuda: $LIBCUDA_DIR ; out: $OUT"
gcc -shared -fPIC -O2 \
  "$HERE/combined_install.c" "$HERE/q8_gemv_launcher.c" \
  "$SUBLIB/cupti_bridge.c" "$SUBLIB/bpd_cupti_profile.c" \
  -o "$OUT" \
  -I"$CUDA_MERGED/include" -I"$SWIINC" \
  -L"$CUDA_MERGED/lib" -L"$LIBCUDA_DIR" -lcuda -lcupti
echo "built $OUT"
strings "$OUT" | grep -q run_q8_gemv_tiled && echo "  ✓ run_q8_gemv_tiled present" || echo "  ✗ tiled pred MISSING"
