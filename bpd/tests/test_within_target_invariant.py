# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""F2 Within-Target Invariant Tests — GPU bit-identical verification.

Verifies that BPD-emitted CUDA kernels produce bit-identical output
regardless of host dispatch path (C vs Python). Any within-GPU
divergence is a substrate bug, NOT a tunable.

Per docs/methodology/within-target-bit-identical-baseline.md (Heath's
2026-05-17 reframe): cells sharing dispatch hardware MUST produce
ULP=0. Cross-axis divergence is allowed and bounded; within-target
divergence is structural code generation failure.

Empirical basis: F2.a (commit 2835eaf72) — 36/36 bit-identical on Tesla P4.

Test design: medayek's 3fdc9f239 concise pattern, refactored to use
shared _matrix_test_helpers.py for measurement. Tests describe WHAT;
helpers describe HOW.

Authors: medayek (test design) + metayen (curriculum_log + helpers)
Date: 2026-05-18
"""

import pytest
from curriculum_log import log_measurement
from _matrix_test_helpers import measure_within_gpu_ulp


# ═══════════════════════════════════════════════════════════════════
# Kernel registry — extend as the curriculum grows
# ═══════════════════════════════════════════════════════════════════

ACTIVATIONS = ["k_silu", "k_sigmoid", "k_relu", "k_tanh", "k_gelu_tanh", "k_gelu_erf"]
SIZES = [128, 256, 257, 1000, 1023, 1024]
SMOKE_SIZE = 1024  # representative size for CI smoke runs

# Within-GPU invariant: ALL kernels must be bit-identical (0 ULP)
WITHIN_GPU_BOUND = {op: 0 for op in ACTIVATIONS}


# ═══════════════════════════════════════════════════════════════════
# Smoke tests — CI default (1 representative size per kernel)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.smoke
@pytest.mark.tier3
@pytest.mark.parametrize("kernel", ACTIVATIONS)
def test_within_gpu_smoke(kernel):
    """Within-GPU invariant at SMOKE_SIZE. Hard: ULP == 0."""
    actual = measure_within_gpu_ulp(kernel, SMOKE_SIZE)
    bound = WITHIN_GPU_BOUND[kernel]
    print(f"  {kernel} @ {SMOKE_SIZE}: max_ULP={actual} (bound={bound})")
    log_measurement(kernel, SMOKE_SIZE, "cell2_vs_cell4", "strict", actual, bound)
    assert actual <= bound, (
        f"WITHIN-GPU SUBSTRATE BUG for {kernel} @ {SMOKE_SIZE}: "
        f"{actual} ULP > bound {bound}. "
        f"Cells sharing dispatch hardware MUST be bit-identical; "
        f"investigate code generation, not tunables."
    )


# ═══════════════════════════════════════════════════════════════════
# Full sweep — periodic / pre-release (all sizes × all kernels)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.full_sweep
@pytest.mark.tier3
@pytest.mark.parametrize("kernel", ACTIVATIONS)
@pytest.mark.parametrize("size", SIZES)
def test_within_gpu_full(kernel, size):
    """Within-GPU invariant at every size. Catches boundary issues."""
    actual = measure_within_gpu_ulp(kernel, size)
    bound = WITHIN_GPU_BOUND[kernel]
    print(f"  {kernel} @ {size}: max_ULP={actual} (bound={bound})")
    log_measurement(kernel, size, "cell2_vs_cell4", "strict", actual, bound)
    assert actual <= bound, (
        f"WITHIN-GPU SUBSTRATE BUG for {kernel} @ {size}: "
        f"{actual} ULP > bound {bound}"
    )
