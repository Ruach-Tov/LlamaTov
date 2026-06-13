#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# Build the CUPTI activity tracer (the acquisition layer for cupti-from-prolog).
# Produces libcupti_trace.so, which you LD_PRELOAD into a CUDA target to capture
# memcpy + kernel activity as Prolog-ingestible facts.
set -euo pipefail

# CUDA 12.8 merged tree has both cupti.h and cuda.h in one include dir on this host.
# Override CUDA_MERGED for other layouts.
CUDA_MERGED="${CUDA_MERGED:-/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8}"
HERE="$(cd "$(dirname "$0")" && pwd)"

gcc -shared -fPIC -O2 "$HERE/cupti_trace.c" -o "$HERE/libcupti_trace.so" \
    -I"$CUDA_MERGED/include" -L"$CUDA_MERGED/lib" -lcupti

echo "built: $HERE/libcupti_trace.so"
echo
echo "Usage:"
echo "  CUPTI_TRACE_OUT=/tmp/cupti.facts \\"
echo "  LD_PRELOAD=$HERE/libcupti_trace.so \\"
echo "  LD_LIBRARY_PATH=$CUDA_MERGED/lib \\"
echo "  python3 your_driver.py"
echo
echo "Then analyse:"
echo "  swipl -g 'style_check(-discontiguous)' run_cupti.pl"
