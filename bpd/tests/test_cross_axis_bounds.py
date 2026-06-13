# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""F2 Cross-Axis Bound Tests — CPU-vs-GPU ULP verification.

Verifies that cross-axis divergence (cell [2] C-host GPU vs cell [3]
Python-host CPU reference) stays within empirically characterized
bounds per kernel. Cross-axis ULPs are EXPECTED (different math
libraries, FMA semantics, reduction orders); the test ensures they
stay bounded.

Bounds TIGHTEN over time as F2.b tunables get standardized. Each PR
that closes a tunable gap can lower a bound.

Empirical basis: F2.b (commit 36b0e255e) — six tunables characterized.
Attribution: bpd/lib/ulp_attribution.pl (Prolog source of truth).

Test design: medayek's 3fdc9f239 concise pattern with parallel
BOUND + ATTRIBUTION dicts. Tests describe WHAT; helpers describe HOW.

Authors: medayek (test design + bounds) + metayen (curriculum_log + helpers)
Date: 2026-05-18
"""

import pytest
from curriculum_log import log_measurement
from _matrix_test_helpers import measure_cross_axis_ulp


# ═══════════════════════════════════════════════════════════════════
# Kernel registry + empirical cross-axis bounds (from F2.b on Tesla P4)
# Bounds are UPPER limits — actuals should be at or below.
# Update as tunables tighten (PR-level bound reduction).
# ═══════════════════════════════════════════════════════════════════

ACTIVATIONS = ["k_silu", "k_sigmoid", "k_relu", "k_tanh", "k_gelu_tanh", "k_gelu_erf"]
REDUCTIONS = ["ggml_sum_rows", "ggml_mean", "ggml_max", "ggml_min", "ggml_argmax", "ggml_argmin"]
SIZES = [128, 256, 257, 1000, 1023, 1024]
SMOKE_SIZE = 1024

CROSS_AXIS_BOUND = {
    # Activations
    "k_relu":        0,      # no transcendental, bit-identical
    "k_silu":        2,      # Class 1 transcendental (expf)
    "k_sigmoid":     2,      # Class 1 transcendental (expf)
    "k_tanh":        2,      # Class 1 transcendental (tanhf)
    "k_gelu_tanh":   200,    # Class 2 tight (tanhf + polynomial FMA)
    "k_gelu_erf":    10000,  # Class 2 loose (erff precision gap near zero)
    # Reductions
    "ggml_sum_rows": 4,      # reduction_order: linear vs PyTorch tree
    "ggml_mean":     4,      # reduction_order: linear vs PyTorch tree
    "ggml_max":      0,      # selection op, no accumulation
    "ggml_min":      0,      # selection op, no accumulation
    "ggml_argmax":   0,      # selection op
    "ggml_argmin":   0,      # selection op
}

ATTRIBUTION = {
    "k_relu":        "no_transcendental",
    "k_silu":        "math_library(expf)",
    "k_sigmoid":     "math_library(expf)",
    "k_tanh":        "math_library(tanhf)",
    "k_gelu_tanh":   "math_library(tanhf) + fma + constant_precision",
    "k_gelu_erf":    "math_library(erff)",
    "ggml_sum_rows": "reduction_order",
    "ggml_mean":     "reduction_order",
    "ggml_max":      "selection",
    "ggml_min":      "selection",
    "ggml_argmax":   "selection",
    "ggml_argmin":   "selection",
}


# ═══════════════════════════════════════════════════════════════════
# Smoke tests — activations (CI default)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.smoke
@pytest.mark.tier3
@pytest.mark.parametrize("kernel", ACTIVATIONS)
def test_activation_cross_axis_smoke(kernel):
    """Cross-axis bound at SMOKE_SIZE per activation."""
    actual = measure_cross_axis_ulp(kernel, SMOKE_SIZE)
    bound = CROSS_AXIS_BOUND[kernel]
    attr = ATTRIBUTION[kernel]
    print(f"  {kernel} @ {SMOKE_SIZE}: max_ULP={actual} (bound={bound}, attr={attr})")
    log_measurement(
        kernel, SMOKE_SIZE, "cell2_vs_cell3", "ulp_bounded", actual, bound,
        extra_fields={"attribution": attr},
    )
    assert actual <= bound, (
        f"{kernel} @ {SMOKE_SIZE} cross-axis: {actual} ULP > bound {bound} "
        f"(attribution: {attr}). Substrate regressed OR bound needs widening."
    )


# ═══════════════════════════════════════════════════════════════════
# Full sweep — activations (periodic)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.full_sweep
@pytest.mark.tier3
@pytest.mark.parametrize("kernel", ACTIVATIONS)
@pytest.mark.parametrize("size", SIZES)
def test_activation_cross_axis_full(kernel, size):
    """Cross-axis bound at every size. Catches size-specific issues."""
    actual = measure_cross_axis_ulp(kernel, size)
    bound = CROSS_AXIS_BOUND[kernel]
    attr = ATTRIBUTION[kernel]
    print(f"  {kernel} @ {size}: max_ULP={actual} (bound={bound}, attr={attr})")
    log_measurement(
        kernel, size, "cell2_vs_cell3", "ulp_bounded", actual, bound,
        extra_fields={"attribution": attr},
    )
    assert actual <= bound, (
        f"{kernel} @ {size} cross-axis: {actual} ULP > bound {bound} "
        f"(attribution: {attr})"
    )


# ═══════════════════════════════════════════════════════════════════
# Smoke tests — reductions (single shape per kernel)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.smoke
@pytest.mark.tier3
@pytest.mark.parametrize("kernel", REDUCTIONS)
def test_reduction_cross_axis_smoke(kernel):
    """Cross-axis bound for reductions (single 8x16 shape)."""
    actual = measure_cross_axis_ulp(kernel, size=None, shape_suffix='_8x16')
    bound = CROSS_AXIS_BOUND[kernel]
    attr = ATTRIBUTION[kernel]
    print(f"  {kernel} (8x16): max_ULP={actual} (bound={bound}, attr={attr})")
    log_measurement(
        kernel, None, "cell2_vs_cell3", "ulp_bounded", actual, bound,
        extra_fields={"attribution": attr, "shape": "8x16"},
    )
    assert actual <= bound, (
        f"{kernel} (8x16) cross-axis: {actual} ULP > bound {bound} "
        f"(attribution: {attr})"
    )
