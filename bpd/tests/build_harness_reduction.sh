#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# build_harness_reduction.sh — Build the C-host reduction harness.
#
# Generates the CUDA kernel + launcher via kernel_emit_bridge, then
# compiles harness_reduction.cu + npy_io.c + emitted kernel into one
# executable via nvcc.
#
# Requires:
#   nvcc (CUDA toolkit, available on P4)
#   python3 with numpy + torch (for kernel_emit_bridge invocation)
#   swipl (the bridge runs swipl as a subprocess)
#
# Usage:
#   ./build_harness_reduction.sh <op_kind> [output_binary]
#
# Examples:
#   ./build_harness_reduction.sh ggml_sum_rows /tmp/harness_reduce_sum
#
# Default output: /tmp/harness_<op_kind>
#
# Author: metayen 2026-05-17
# Per Heath's cross-language correctness matrix vision (1.c.ii/b).

set -euo pipefail

OP_KIND="${1:-ggml_sum_rows}"
OUTPUT_BIN="${2:-/tmp/harness_$OP_KIND}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BPD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$BPD_DIR/lib"
TESTS_DIR="$BPD_DIR/tests"

TMPDIR="$(mktemp -d)"
trap "rm -rf $TMPDIR" EXIT

KERNEL_SRC="$TMPDIR/emitted_kernel.cu"

echo "Step 1: Emit kernel + launcher for op_kind=$OP_KIND"
cd "$LIB_DIR"
LAUNCHER_NAME=$(python3 -c "
import kernel_emit_bridge as keb
combined, kn, ln = keb.emit_kernel_with_launcher('reduction', '$OP_KIND')
with open('$KERNEL_SRC', 'w') as f:
    f.write(combined)
import sys
print(f'  Wrote {len(combined)} chars to $KERNEL_SRC', file=sys.stderr)
print(f'  kernel:   {kn}', file=sys.stderr)
print(f'  launcher: {ln}', file=sys.stderr)
print(ln)  # this is what gets captured into LAUNCHER_NAME
")
echo "  (launcher symbol passed via -DREDUCTION_LAUNCHER=$LAUNCHER_NAME)"

echo
echo "Step 2: Compile harness with nvcc"
# Allow override via env vars for environments where nvcc / CUDA headers
# aren't on the default search paths (e.g., NixOS where they live at
# /nix/store/... paths).
NVCC_BIN="${NVCC:-nvcc}"
# CUDA include dir (cuda_runtime.h). Empty by default = let nvcc find it.
CUDA_INC_FLAG=""
if [ -n "${CUDA_INC:-}" ]; then
    CUDA_INC_FLAG="-I $CUDA_INC"
fi
# CUDA lib dir (libcudart.so). Empty by default = let nvcc find it.
CUDA_LIB_FLAG=""
if [ -n "${CUDA_LIB:-}" ]; then
    CUDA_LIB_FLAG="-L $CUDA_LIB"
fi
# P4 (Tesla P4) is Pascal compute capability 6.1. nvcc 12.x deprecates
# sm_61 but still supports it; -Wno-deprecated-gpu-targets silences the
# deprecation warning. Per mavchin's enclave-build guidance 2026-05-17.
"$NVCC_BIN" -O2 -arch=sm_61 -Wno-deprecated-gpu-targets \
    $CUDA_INC_FLAG $CUDA_LIB_FLAG \
    -I "$TESTS_DIR" \
    -DREDUCTION_LAUNCHER="$LAUNCHER_NAME" \
    -o "$OUTPUT_BIN" \
    "$TESTS_DIR/harness_reduction.cu" \
    "$KERNEL_SRC" \
    "$TESTS_DIR/npy_io.c"

echo
echo "Built: $OUTPUT_BIN"
echo "Run example:"
echo "  $OUTPUT_BIN $TESTS_DIR/fixtures/${OP_KIND}_8x16.npy /tmp/${OP_KIND}_cell2_c_gpu.npy $OP_KIND"
echo "  python3 $TESTS_DIR/matrix_verify.py \\"
echo "      $TESTS_DIR/fixtures/${OP_KIND}_8x16_cell3_python_cpu.npy \\"
echo "      /tmp/${OP_KIND}_cell2_c_gpu.npy"
echo
echo "  # The comparison above is cell [3] (Python CPU) vs cell [2] (C GPU)."
echo "  # Per mavchin's 7-cell-per-kernel framing (2026-05-17 21:42 UTC)."
