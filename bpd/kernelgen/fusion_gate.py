# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""fusion_gate.py — the apply-time 0-ULP gate for kernel fusion.

The runtime verification that makes Heath's "all automatic, provably 0-ULP before/after"
requirement real. Generalizes Mavchin's test_fusion_bitidentical.py XOR=0 harness (which was a
test on specific cases) into a reusable callable that apply_fusion can invoke before committing
ANY fusion: run the unfused reference and the fused candidate on IDENTICAL inputs, compare the
f32 OUTPUT bytes, and only pass if the declared equivalence class holds.

  bit_exact     -> XOR == 0 for every element (memcmp == 0)
  tolerance(eps)-> max abs diff <= eps (per element)

Lift credit: Mavchin (the XOR=0 comparison core + the "compare the f32 OUTPUT not the int8
intermediate" discipline). This wraps the kernel build/launch in the working _build_inline path
(standalone nvcc is broken on the enclave; the inline path is the one that compiles).

A fusion is described declaratively so any fusion in the series uses one gate:
  FusionSpec(
    name, K, M,
    unfused_launch(ctx) -> writes f32 output to a device buffer, returns it,
    fused_launch(ctx)   -> same,
    output_size,
    equiv_class)   # 'bit_exact' or ('tolerance', eps)
"""
import ctypes
import numpy as np


class GateResult:
    def __init__(self, passed, equiv_class, mismatches, max_ulp, max_abs, n):
        self.passed = passed
        self.equiv_class = equiv_class
        self.mismatches = mismatches
        self.max_ulp = max_ulp
        self.max_abs = max_abs
        self.n = n

    def __repr__(self):
        verdict = "PASS" if self.passed else "FAIL"
        return (f"GateResult({verdict} {self.equiv_class}: {self.mismatches}/{self.n} "
                f"mismatches, max_ulp={self.max_ulp}, max_abs={self.max_abs:.3e})")


def compare_outputs(unfused_out, fused_out, equiv_class):
    """The comparison core (Mavchin's XOR=0, generalized to carry a tolerance class).
    unfused_out, fused_out: np.float32 arrays (the f32 OUTPUT of each path).
    equiv_class: 'bit_exact' or ('tolerance', eps).
    Returns GateResult.
    """
    a = np.ascontiguousarray(unfused_out, np.float32)
    b = np.ascontiguousarray(fused_out, np.float32)
    n = a.size
    # XOR on the raw bits — Mavchin's core: "both must produce XOR=0x00000000 for every element"
    ab = a.view(np.uint32)
    bb = b.view(np.uint32)
    xor = ab ^ bb
    mismatches = int((xor != 0).sum())
    max_ulp = int(np.abs(ab.astype(np.int64) - bb.astype(np.int64)).max()) if n else 0
    max_abs = float(np.abs(a - b).max()) if n else 0.0
    if equiv_class == 'bit_exact':
        passed = (mismatches == 0)
    elif isinstance(equiv_class, (tuple, list)) and equiv_class[0] == 'tolerance':
        passed = (max_abs <= float(equiv_class[1]))
    else:
        raise ValueError(f"unknown equivalence class: {equiv_class!r}")
    return GateResult(passed, equiv_class, mismatches, max_ulp, max_abs, n)


def gate_bitexact(unfused_fn, fused_fn, output_size, equiv_class='bit_exact', seed=7):
    """The apply-time gate. unfused_fn() and fused_fn() each launch their path (on identical
    device inputs they set up internally or share) and RETURN the f32 output as a numpy array.
    Only commit the fusion if the result passes the declared equivalence class.

    This is the callable apply_fusion invokes before committing a fusion: build both kernels,
    run on identical inputs, compare f32 output bits, gate on the class. Returns GateResult.
    """
    np.random.seed(seed)
    unfused_out = unfused_fn()
    fused_out = fused_fn()
    return compare_outputs(unfused_out, fused_out, equiv_class)
