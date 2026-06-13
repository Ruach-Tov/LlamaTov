#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_fusion_F3_v2.py — verify bpd_conv2d_bn_silu_fused_cpu_v2 BIT_IDENTICAL.

Substrate-design substantive substantive Tier 1.5 gate for the
CAT-scan-informed fused CBS kernel (Phase 3.CAT.FUSE.a).

Reference (already verified BIT_IDENTICAL with PyTorch CPU):
  bpd_conv2d_bn_silu_fused_cpu (F3 v1, scalar epilogue)
  When SUBSTRATE_AVX1_GEMM_V2=1 is in effect, this calls v2 GEMM internally,
  so v1 fused == v2-GEMM + scalar-epilogue.

Fused-v2:
  bpd_conv2d_bn_silu_fused_cpu_v2 (Phase 3.CAT.FUSE.a)
  Inlines v2 GEMM with SIMD epilogue applied directly in registers,
  eliminating the GEMM-output round-trip.

Tests YOLOv5n CBS shapes covering both single-K-block (K<=384, 45/57 sites)
and multi-K-block (K>384, 12/57 sites) cases per the empirical inventory.

Each shape must be 0 ULP.
"""
import ctypes
import os
import sys

import numpy as np

SO = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")


def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


def precompute_alpha_beta(gamma, bn_beta, mean, var, eps=1e-5):
    """BN-fold: alpha = gamma / sqrt(var + eps); beta = bn_beta - mean * alpha."""
    gamma = np.asarray(gamma, dtype=np.float32)
    bn_beta = np.asarray(bn_beta, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    inv_std = (1.0 / np.sqrt(var + np.float32(eps))).astype(np.float32)
    alpha = (gamma * inv_std).astype(np.float32)
    beta = (bn_beta - mean * alpha).astype(np.float32)
    return alpha, beta


def main():
    lib = ctypes.CDLL(SO)
    lib.bpd_conv2d_bn_silu_fused_cpu.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
    lib.bpd_conv2d_bn_silu_fused_cpu.restype = None
    has_v2 = hasattr(lib, 'bpd_conv2d_bn_silu_fused_cpu_v2')
    if has_v2:
        lib.bpd_conv2d_bn_silu_fused_cpu_v2.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
        lib.bpd_conv2d_bn_silu_fused_cpu_v2.restype = None

    print(f"Substrate: {SO}")
    print(f"F3-v2 kernel available: {has_v2}")
    print()
    if not has_v2:
        print("\u26a0\ufe0f  F3-v2 not built. Build first.")
        sys.exit(2)

    # YOLOv5n CBS shapes — both single-K-block (K<=384) and multi-K-block (K>384)
    shapes = [
        # (label, N, Cin, H, W, Cout, kH, kW, stride, pad, K_value, K<=384?)
        ("L0 focus 6x6 s=2",   1,   3, 640, 640,  16, 6, 6, 2, 2),
        ("L1 cbs 3x3 s=2",     1,  16, 320, 320,  32, 3, 3, 2, 1),
        ("L3 cbs 3x3 s=2",     1,  32, 160, 160,  64, 3, 3, 2, 1),
        ("L5 cbs 3x3 s=2",     1,  64,  80,  80, 128, 3, 3, 2, 1),  # K=576 multi-block
        ("L7 cbs 3x3 s=2",     1, 128,  40,  40, 256, 3, 3, 2, 1),  # K=1152 multi-block
        ("L9 sppf cv1 1x1",    1, 256,  20,  20, 128, 1, 1, 1, 0),
        ("L9 sppf cv2 1x1",    1, 512,  20,  20, 256, 1, 1, 1, 0),  # K=512 multi-block
        ("L10 head 1x1",       1, 256,  20,  20, 128, 1, 1, 1, 0),
        ("L13 c3 cv3 1x1",     1, 128,  40,  40, 128, 1, 1, 1, 0),
        ("L21 head 3x3 s=2",   1, 128,  40,  40, 128, 3, 3, 2, 1),  # K=1152 multi-block
    ]

    print(f"{'Shape':<22} {'In':<22} {'Out':<22} {'K':<6} {'Status':<25}")
    print("-" * 102)

    rng = np.random.default_rng(2026)
    all_pass = True
    for label, N, Cin, H, W, Cout, kH, kW, stride, pad in shapes:
        K_value = Cin * kH * kW
        H_out = (H + 2*pad - kH) // stride + 1
        W_out = (W + 2*pad - kW) // stride + 1

        x = (rng.standard_normal((N, Cin, H, W)) * 0.3).astype(np.float32)
        w = (rng.standard_normal((Cout, Cin, kH, kW)) * (1.0/np.sqrt(K_value))).astype(np.float32)
        gamma = (rng.standard_normal(Cout) * 0.3 + 1.0).astype(np.float32)
        bn_beta = (rng.standard_normal(Cout) * 0.1).astype(np.float32)
        mean = (rng.standard_normal(Cout) * 0.2).astype(np.float32)
        var = (np.abs(rng.standard_normal(Cout) * 0.5) + 0.1).astype(np.float32)

        alpha, beta = precompute_alpha_beta(gamma, bn_beta, mean, var)
        alpha = np.ascontiguousarray(alpha, dtype=np.float32)
        beta = np.ascontiguousarray(beta, dtype=np.float32)

        # Reference: F3 v1 (uses v2 GEMM internally via dispatcher, scalar epilogue)
        ref = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
        lib.bpd_conv2d_bn_silu_fused_cpu(
            x.ctypes.data, w.ctypes.data,
            alpha.ctypes.data, beta.ctypes.data,
            ref.ctypes.data,
            N, Cin, H, W, Cout, kH, kW, stride, stride, pad, pad)

        # Fused-v2
        fused = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
        lib.bpd_conv2d_bn_silu_fused_cpu_v2(
            x.ctypes.data, w.ctypes.data,
            alpha.ctypes.data, beta.ctypes.data,
            fused.ctypes.data,
            N, Cin, H, W, Cout, kH, kW, stride, stride, pad, pad)

        max_ulp, n_diff, n_total = ulp_distance(ref, fused)
        in_s = f"({N},{Cin},{H},{W})"
        out_s = f"({N},{Cout},{H_out},{W_out})"
        if max_ulp == 0:
            status = "BIT_IDENTICAL"
        else:
            status = f"DIVERGENT max={max_ulp} n={n_diff}/{n_total}"
            all_pass = False
        kb_note = f"{K_value}" + (" *" if K_value > 384 else "")
        print(f"{label:<22} {in_s:<22} {out_s:<22} {kb_note:<6} {status:<25}")

    print()
    print("* = multi K-block (K > 384)")
    print()
    if all_pass:
        print("\u2705 F3-v2 BIT_IDENTICAL with F3 v1 across all YOLOv5n CBS shapes.")
        sys.exit(0)
    else:
        print("\u274c F3-v2 DIVERGES on at least one shape. gdb per the discipline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
