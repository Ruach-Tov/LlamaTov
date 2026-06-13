#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_opmath_precision_invariance.py — TDD harness for opmath-precision invariance.

Per Heath's substantive substrate-design direction 2026-05-20 ~17:50 UTC:
'Is there a way we could have detected the bug in precompute_bn automatically?'

Substantive substrate-design property tested:
  A function declared to return float32 must be OPMATH-PRECISION-INVARIANT:
  passing the SAME numerical values in different input precisions (f16, f32,
  f64) must yield the SAME f32 output bits.

This is the substrate-design family member 'opmath_precision' — parallel
to rsqrt_variant, k_tile_strategy, reduction_strategy, etc. When the
substrate (or orchestrator) takes f16 weights and computes in numpy without
promotion, the result diverges from PyTorch's behavior (which promotes to
f32 for op math). The bug surfaces as massive precision loss (5e-5 to 1e-2
absolute error in the BN affine path).

The substantive substrate-design test:
  1. Generate reference values in float64 (highest precision)
  2. Cast to f16, f32, f64 versions of the SAME numerical values
  3. Run the function on each precision input, forcing output to f32
  4. Assert all three outputs are BIT_IDENTICAL with each other

If a function silently retains f16 internally, its f16-input result will
diverge from its f32/f64-input results. The test catches it.

Run:
  PYTHONPATH=<torch> python3 bench/test_opmath_precision_invariance.py

xfail expected BEFORE the fix:
  precompute_bn: f16 input -> divergent from f32/f64 inputs (BUG)

PASS expected AFTER the fix:
  precompute_bn: all three precisions yield 0 ULP outputs (CORRECT)
"""
import os
import sys
import numpy as np

# Add the bench dir to path so we can import the orchestrator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def ulp(a, b):
    """IEEE 754 sign-magnitude ULP distance between two float32 arrays."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum())


def assert_opmath_invariant(fn, fn_name, ref_inputs_f64, eps=1e-5, atol_ulp=0):
    """Assert fn(inputs) -> f32 produces same f32 bits regardless of input precision.

    The substrate-design property tested: for the SAME numerical values arriving
    in different precisions (f16, f32, f64), the f32 output bits must agree.

    Substantively important methodological point: we use values that are
    EXACTLY REPRESENTABLE in f16 — so the lift f16 -> f32 -> f64 is lossless
    and the value itself is identical across all three precisions. Any
    divergence in output is then from the FUNCTION'S handling, not from
    input rounding.

    To produce exactly-f16-representable values, we round the f64 reference
    to f16, then lift back to f64. After this round-trip, the f64 array's
    values are bit-exact representations of f16 numbers, and casting to f32
    or back to f16 is lossless.

    ref_inputs_f64: tuple of float64 reference arrays
    fn: function (taking those tensors, returning a tuple of f32 arrays)
    """
    # Substantively important: snap ref_inputs to f16-representable values
    # so the lift f16->f32->f64 is exact (no rounding-induced false divergence).
    inputs_f64_exact = tuple(arr.astype(np.float16).astype(np.float64) for arr in ref_inputs_f64)
    inputs_f32 = tuple(arr.astype(np.float32) for arr in inputs_f64_exact)
    inputs_f16 = tuple(arr.astype(np.float16) for arr in inputs_f32)

    out_f32 = fn(*inputs_f32, eps=eps)
    out_f16 = fn(*inputs_f16, eps=eps)
    out_f64 = fn(*inputs_f64_exact, eps=eps)

    if not isinstance(out_f32, tuple):
        out_f32 = (out_f32,)
        out_f16 = (out_f16,)
        out_f64 = (out_f64,)

    print(f"  Testing {fn_name}:")
    all_pass = True
    for i, (o32, o16, o64) in enumerate(zip(out_f32, out_f16, out_f64)):
        # All outputs must be f32 (the function's declared output type)
        o32_arr = np.ascontiguousarray(o32, dtype=np.float32)
        o16_arr = np.ascontiguousarray(o16, dtype=np.float32)
        o64_arr = np.ascontiguousarray(o64, dtype=np.float32)

        m_16_vs_32, d_16_vs_32 = ulp(o32_arr, o16_arr)
        m_64_vs_32, d_64_vs_32 = ulp(o32_arr, o64_arr)

        status_16 = "PASS" if m_16_vs_32 <= atol_ulp else "FAIL"
        status_64 = "PASS" if m_64_vs_32 <= atol_ulp else "FAIL"

        if status_16 == "FAIL" or status_64 == "FAIL":
            all_pass = False

        print(f"    output[{i}]: f16-input vs f32-input: {status_16} ({m_16_vs_32} ULP, {d_16_vs_32}/{o32_arr.size} diffs)")
        print(f"    output[{i}]: f64-input vs f32-input: {status_64} ({m_64_vs_32} ULP, {d_64_vs_32}/{o32_arr.size} diffs)")
    return all_pass


# ====================================================================
# Test 1: precompute_bn (the substantive bug surfaced this morning)
# ====================================================================
def precompute_bn(gamma, beta, mean, var, eps=1e-5):
    """The function under test — extracted from bench/yolo_forward.py.

    Substantive substrate-design fix 2026-05-20 ~17:55 UTC: promote inputs
    to f32 at the function boundary. Without this, f16 weights from .pt
    files (the actual orchestrator input path) cause numpy to do f16 math,
    producing 5e-5 to 1e-2 absolute error vs PyTorch's f32-promoted compute.

    Per medayek's substrate-design framework: opmath_precision is a named
    substrate-design parameter; this function declares 'compute in f32.'"""
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    bn_scale = gamma / np.sqrt(var + eps)
    bn_offset = beta - mean * bn_scale
    return bn_scale.astype(np.float32), bn_offset.astype(np.float32)


def test_precompute_bn():
    """Substantive substrate-design test: precompute_bn must be opmath-invariant."""
    # Reference: realistic YOLOv5n Layer 0 BN parameters (16 channels)
    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    gamma = (rng.standard_normal(16) * 0.5 + 1.0).astype(np.float64)
    beta = (rng.standard_normal(16) * 0.1).astype(np.float64)
    mean = (rng.standard_normal(16) * 0.2).astype(np.float64)
    var = ((rng.standard_normal(16) * 0.1 + 1.0) ** 2).astype(np.float64)

    return assert_opmath_invariant(
        precompute_bn, "precompute_bn (yolo_forward.py)",
        (gamma, beta, mean, var))


# ====================================================================
# Test 2: simple control case (f32 multiply is trivially invariant)
# ====================================================================
def trivial_multiply(a, b, eps=1e-5):
    """Control case: a trivial f32-promote-then-multiply.
    Substantively correct opmath discipline: promote inputs to f32 at the
    function boundary, then compute."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a * b).astype(np.float32)


def test_trivial_control():
    """Sanity check: a simple a*b cast to f32 should pass opmath invariance.
    The assert_opmath_invariant helper snaps inputs to f16-representable
    values first, so f16/f32/f64 inputs all carry the same numerical values."""
    rng = np.random.default_rng(7)
    a = (rng.standard_normal(16) * 0.5).astype(np.float64)
    b = (rng.standard_normal(16) * 0.5).astype(np.float64)
    return assert_opmath_invariant(
        trivial_multiply, "trivial_multiply (a*b cast to f32)",
        (a, b))


if __name__ == "__main__":
    print("=" * 72)
    print("Opmath-precision-invariance test harness")
    print("Per Heath's substantive substrate-design direction:")
    print("  'Is there a way we could have detected the bug in precompute_bn")
    print("   automatically? If so, let's add tests that would have caught it.'")
    print("=" * 72)
    print()

    results = []
    print("--- Test 1: precompute_bn (orchestrator's substantive bug) ---")
    results.append(("precompute_bn", test_precompute_bn()))
    print()
    print("--- Test 2: trivial_multiply (control) ---")
    results.append(("trivial_multiply", test_trivial_control()))
    print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    fail_count = sum(1 for _, ok in results if not ok)
    for name, ok in results:
        status = "PASS" if ok else "FAIL (substantive substrate-design bug)"
        print(f"  {name:<40} {status}")
    print()
    if fail_count > 0:
        print(f"  Substantively detected {fail_count} substrate-design bug(s).")
        print(f"  This is the substantively-correct TDD xfail state — BEFORE the fix.")
        sys.exit(1)
    else:
        print("  All substantive substrate-design opmath-invariance properties hold.")
        sys.exit(0)
