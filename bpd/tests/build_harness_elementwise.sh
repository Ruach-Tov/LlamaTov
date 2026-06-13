#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# build_harness_elementwise.sh — Build the C-host elementwise harness.
#
# Sibling to build_harness_reduction.sh. Generates the CUDA kernel +
# launcher for an activation family op (k_silu, k_sigmoid, k_relu,
# k_gelu, k_tanh) via kernel_emit_bridge, then compiles
# harness_elementwise.cu + npy_io.c + emitted kernel into one
# executable via nvcc.
#
# Requires:
#   nvcc (CUDA toolkit, available on P4)
#   python3 with numpy + torch (for kernel_emit_bridge invocation)
#   swipl (the bridge runs swipl as a subprocess)
#
# Usage:
#   ./build_harness_elementwise.sh <op_kind> [output_binary]
#
# Examples:
#   ./build_harness_elementwise.sh k_silu /tmp/harness_k_silu
#
# Default output: /tmp/harness_<op_kind>
#
# Environment overrides (mirrors build_harness_reduction.sh):
#   NVCC:     absolute path to nvcc (default 'nvcc' from PATH)
#   CUDA_INC: path containing cuda_runtime.h (passed as -I)
#   CUDA_LIB: path containing libcudart.so (passed as -L)
#
# Author: metayen 2026-05-17
# Per Heath's cross-language correctness matrix vision (T8.c).
# Per mavchin's 7-cell framing: this produces cell [2] outputs for
# the activation column.

set -euo pipefail

OP_KIND="${1:-k_silu}"
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
combined, kn, ln = keb.emit_kernel_with_launcher('activation', '$OP_KIND')
with open('$KERNEL_SRC', 'w') as f:
    f.write(combined)
import sys
print(f'  Wrote {len(combined)} chars to $KERNEL_SRC', file=sys.stderr)
print(f'  kernel:   {kn}', file=sys.stderr)
print(f'  launcher: {ln}', file=sys.stderr)
print(ln)  # captured into LAUNCHER_NAME
")
echo "  (launcher symbol passed via -DACTIVATION_LAUNCHER=$LAUNCHER_NAME)"

echo
echo "Step 2: Compile harness with nvcc"
# Allow override via env vars for environments where nvcc / CUDA paths
# aren't on the default search list (e.g., NixOS where they live at
# /nix/store/... paths).
NVCC_BIN="${NVCC:-nvcc}"
CUDA_INC_FLAG=""
if [ -n "${CUDA_INC:-}" ]; then
    CUDA_INC_FLAG="-I $CUDA_INC"
fi
CUDA_LIB_FLAG=""
if [ -n "${CUDA_LIB:-}" ]; then
    CUDA_LIB_FLAG="-L $CUDA_LIB"
fi
# Tesla P4 is Pascal compute capability 6.1. nvcc 12.x deprecates sm_61
# but still supports it; -Wno-deprecated-gpu-targets silences the
# deprecation warning. Per mavchin's enclave-build guidance.
"$NVCC_BIN" -O2 -arch=sm_61 -Wno-deprecated-gpu-targets \
    $CUDA_INC_FLAG $CUDA_LIB_FLAG \
    -I "$TESTS_DIR" \
    -DACTIVATION_LAUNCHER="$LAUNCHER_NAME" \
    -o "$OUTPUT_BIN" \
    "$TESTS_DIR/harness_elementwise.cu" \
    "$KERNEL_SRC" \
    "$TESTS_DIR/npy_io.c"

echo
echo "Built: $OUTPUT_BIN"
echo "Run example:"
echo "  $OUTPUT_BIN $TESTS_DIR/fixtures/${OP_KIND}_128.npy /tmp/${OP_KIND}_cell2_c_gpu.npy $OP_KIND"
echo "  python3 $TESTS_DIR/matrix_verify.py --strict \\"
echo "      $TESTS_DIR/fixtures/${OP_KIND}_128_cell3_python_cpu.npy \\"
echo "      /tmp/${OP_KIND}_cell2_c_gpu.npy"
echo
echo "  # Per Heath's --strict-maxxing: try --strict first. For"
echo "  # transcendentals (silu/sigmoid/gelu/tanh), --strict may fail by"
echo "  # 1-2 ULPs due to SFU precision; fall back to --ulp 2 then."
echo "  # For relu, expect --strict to pass — no transcendental involved."
