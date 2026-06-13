#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_fusion_F3.py — verify Conv+BN+SiLU fused kernel is BIT_IDENTICAL with unfused.

For each test shape, runs:
  reference: bpd_conv2d_full_cpu(x, w) -> bpd_batchnorm_cpu_affine_fused -> bpd_silu_cpu
  fused:     bpd_conv2d_bn_silu_fused_cpu(x, w, alpha, beta)

Where alpha, beta are precomputed from gamma, bn_beta, mean, var, eps such that
the fused kernel's epilogue y = silu(alpha*acc + beta) is algebraically identical
to the unfused composition.

If the fused kernel doesn't yet exist, the script reports MISSING_KERNEL and is
a structural placeholder for when we implement it.

This is the Phase 3.0.a scaffolding that gates Phase 3.1 (F3 implementation).
"""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

# Avoid pulling torch by accident — we want substrate-vs-substrate
SO = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")


def ulp_distance(a, b):
    """Max-ULP distance, num diverging positions, total count."""
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


def setup_lib():
    lib = ctypes.CDLL(SO)
    lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
    lib.bpd_conv2d_full_cpu.restype = None
    lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3 + [ctypes.c_float]
    lib.bpd_batchnorm_cpu_affine_fused.restype = None
    lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
    lib.bpd_silu_cpu.restype = None
    has_fused = hasattr(lib, "bpd_conv2d_bn_silu_fused_cpu")
    if has_fused:
        # Signature: (in, weight, alpha, beta, out, N, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW)
        lib.bpd_conv2d_bn_silu_fused_cpu.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
        lib.bpd_conv2d_bn_silu_fused_cpu.restype = None
    return lib, has_fused


def precompute_alpha_beta(gamma, bn_beta, mean, var, eps=1e-5):
    """BN-fold: out = gamma * (x - mean) / sqrt(var + eps) + bn_beta
              = alpha * x + beta
       where alpha = gamma / sqrt(var + eps)
             beta  = bn_beta - mean * alpha
    All computed in float32 to match the substrate's pipeline exactly.
    """
    gamma = np.asarray(gamma, dtype=np.float32)
    bn_beta = np.asarray(bn_beta, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    inv_std = (1.0 / np.sqrt(var + np.float32(eps))).astype(np.float32)
    alpha = (gamma * inv_std).astype(np.float32)
    beta = (bn_beta - mean * alpha).astype(np.float32)
    return alpha, beta


def run_unfused(lib, x, w, gamma, bn_beta, mean, var, stride, pad, eps=1e-5):
    """Reference: bpd_conv2d_full_cpu -> bpd_batchnorm_cpu_affine_fused -> bpd_silu_cpu."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = w.shape
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1

    conv_out = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_full_cpu(
        x.ctypes.data, w.ctypes.data, 0,  # bias = NULL
        conv_out.ctypes.data,
        N, Cin, H, W, Cout, kH, kW,
        stride, stride, pad, pad, 1, 1, 1)

    bn_out = np.zeros_like(conv_out)
    lib.bpd_batchnorm_cpu_affine_fused(
        conv_out.ctypes.data,
        gamma.ctypes.data, bn_beta.ctypes.data,
        mean.ctypes.data, var.ctypes.data,
        bn_out.ctypes.data, 0, 0,
        N, Cout, H_out * W_out, ctypes.c_float(eps))

    silu_out = np.zeros_like(bn_out)
    lib.bpd_silu_cpu(bn_out.ctypes.data, silu_out.ctypes.data, bn_out.size)
    return silu_out


def run_fused(lib, x, w, alpha, beta, stride, pad):
    """Fused: bpd_conv2d_bn_silu_fused_cpu(x, w, alpha, beta) -> silu(alpha*acc + beta)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = w.shape
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1

    out = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_bn_silu_fused_cpu(
        x.ctypes.data, w.ctypes.data,
        alpha.ctypes.data, beta.ctypes.data,
        out.ctypes.data,
        N, Cin, H, W, Cout, kH, kW,
        stride, stride, pad, pad)
    return out


# Test shapes — representative CBS configurations from YOLOv5n
TEST_SHAPES = [
    # (label, N, Cin, H, W, Cout, kH, kW, stride, pad)
    ("L0 focus 6x6 s=2",   1,   3, 640, 640,  16, 6, 6, 2, 2),
    ("L1 cbs 3x3 s=2",     1,  16, 320, 320,  32, 3, 3, 2, 1),
    ("L2 c3 cv1 1x1",      1,  32, 160, 160,  16, 1, 1, 1, 0),
    ("L3 cbs 3x3 s=2",     1,  32, 160, 160,  64, 3, 3, 2, 1),
    ("L5 cbs 3x3 s=2",     1,  64,  80,  80, 128, 3, 3, 2, 1),
    ("L7 cbs 3x3 s=2",     1, 128,  40,  40, 256, 3, 3, 2, 1),
    ("L9 sppf cv2 1x1",    1, 512,  20,  20, 256, 1, 1, 1, 0),
    ("L10 head cbs 1x1",   1, 256,  20,  20, 128, 1, 1, 1, 0),
    ("L13 c3 cv3 1x1",     1, 128,  40,  40, 128, 1, 1, 1, 0),
    ("L23 P5 cv3 1x1",     1, 256,  20,  20, 256, 1, 1, 1, 0),
]


def main():
    lib, has_fused = setup_lib()
    print(f"Substrate: {SO}")
    print(f"Fused kernel available: {has_fused}")
    print()

    if not has_fused:
        print("⚠️  bpd_conv2d_bn_silu_fused_cpu not yet in substrate.")
        print("    This script is the Phase 3.0.a gate \u2014 it will report PASS/FAIL once")
        print("    Phase 3.1.a (the kernel itself) lands. For now, every shape reports")
        print("    MISSING_KERNEL.")
        print()

    print(f"{'Shape':<22} {'In dims':<22} {'Out dims':<22} {'Status':<28}")
    print("-" * 100)

    rng = np.random.default_rng(2026)
    all_pass = True
    for label, N, Cin, H, W, Cout, kH, kW, stride, pad in TEST_SHAPES:
        # Realistic random tensors
        x = (rng.standard_normal((N, Cin, H, W)) * 0.5).astype(np.float32)
        w = (rng.standard_normal((Cout, Cin, kH, kW)) * (1.0/np.sqrt(Cin*kH*kW))).astype(np.float32)
        gamma = (rng.standard_normal(Cout) * 0.3 + 1.0).astype(np.float32)
        bn_beta = (rng.standard_normal(Cout) * 0.1).astype(np.float32)
        mean = (rng.standard_normal(Cout) * 0.2).astype(np.float32)
        var = (np.abs(rng.standard_normal(Cout) * 0.5) + 0.1).astype(np.float32)

        # Reference (unfused)
        ref = run_unfused(lib, x, w, gamma, bn_beta, mean, var, stride, pad)

        # Fused
        in_shape = (N, Cin, H, W)
        out_shape = ref.shape
        if not has_fused:
            status = "MISSING_KERNEL"
            all_pass = False
        else:
            alpha, beta = precompute_alpha_beta(gamma, bn_beta, mean, var)
            fused = run_fused(lib, x, w, alpha, beta, stride, pad)
            max_ulp, n_diff, n_total = ulp_distance(ref, fused)
            if max_ulp == 0:
                status = "BIT_IDENTICAL"
            else:
                status = f"DIVERGENT max_ulp={max_ulp} n_diff={n_diff}/{n_total}"
                all_pass = False
        print(f"{label:<22} {str(in_shape):<22} {str(out_shape):<22} {status:<28}")

    print()
    if has_fused and all_pass:
        print("✅ F3 fused kernel is BIT_IDENTICAL across all tested shapes.")
        sys.exit(0)
    elif not has_fused:
        print("⚠️  Phase 3.1.a not yet landed. Build the fused kernel first.")
        sys.exit(2)
    else:
        print("❌ F3 fused kernel DIVERGES on at least one shape. Investigate via gdb.")
        sys.exit(1)


if __name__ == "__main__":
    main()
