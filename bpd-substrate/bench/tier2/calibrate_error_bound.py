#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""calibrate_error_bound.py — Empirical calibration of the Tier 2 GEMM error
bound factor.

Per medayek's substrate-design direction 2026-05-20 ~01:07 UTC:
"For random inputs, empirical calibration will tell us the right factor.
 Start with 8, measure actual max_err/theoretical_bound ratios across all
 shapes, tighten to the smallest factor that covers 100% of cases. That
 calibrated factor becomes a substrate constant."

The theoretical bound is `factor * sqrt(K) * eps * max|A| * max|B|`.
We measure `max_err / (sqrt(K) * eps * max|A| * max|B|)` per shape; the
maximum of those ratios across all shapes is the minimum substrate-safe
factor.
"""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    sys.exit("error: torch with CUDA required.")

REPO_DIR = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("BPD_BUILD_DIR", REPO_DIR / "build"))
SO_PATH = os.environ.get("BPD_MM_SO", str(BUILD_DIR / "bpd_mm.so"))


def bpd_matmul(lib, A_np, B_np):
    M, K = A_np.shape
    K2, N = B_np.shape
    assert K == K2
    mn = M * N
    dA = lib.gpu_alloc(M * K * 4)
    dB = lib.gpu_alloc(K * N * 4)
    dC = lib.gpu_alloc(mn * 4)
    lib.gpu_h2d(dA, A_np.ctypes.data, M * K * 4)
    lib.gpu_h2d(dB, B_np.ctypes.data, K * N * 4)
    lib.bpd_sgemm(dA, dB, dC, M, N, K)
    lib.gpu_sync()
    out = np.zeros((M, N), dtype=np.float32)
    lib.gpu_d2h(out.ctypes.data, dC, mn * 4)
    lib.gpu_free(dA)
    lib.gpu_free(dB)
    lib.gpu_free(dC)
    return out


def main():
    if not os.path.exists(SO_PATH):
        sys.exit(f"error: {SO_PATH} not found. Run `make bit_identical` to build.")
    lib = ctypes.CDLL(SO_PATH)
    lib.gpu_alloc.restype = ctypes.c_void_p
    lib.gpu_alloc.argtypes = [ctypes.c_int]
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_sync.argtypes = []
    lib.bpd_sgemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    eps = float(np.finfo(np.float32).eps)
    print(f"Tier 2 error bound calibration (eps={eps:.3e})")
    print(f"Goal: find smallest 'factor' such that max_err < factor*sqrt(K)*eps*max|A|*max|B| holds for all shapes\n")

    shapes = [
        # (M, N, K, label)
        (64, 64, 64, "square small"),
        (128, 128, 128, "square small"),
        (256, 256, 256, "square small"),
        (512, 512, 512, "square mid"),
        (1024, 1024, 1024, "square large"),
        (2048, 2048, 2048, "square xlarge"),
        (64, 1024, 1024, "non-square M small"),
        (128, 512, 256, "non-square mixed"),
        (1024, 512, 2048, "non-square N small"),
        (2048, 1024, 512, "non-square K small"),
    ]

    # Add several random seeds per shape to estimate the spread
    seeds = [42, 137, 7, 12345]

    print(f"{'shape':<22} {'seed':<5} {'max_err':>11} {'normalized':>11} {'ratio_to_unit'}")
    print("─" * 75)

    worst_ratio = 0.0
    worst_shape = None
    all_ratios = []
    for M, N, K, label in shapes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            A = rng.standard_normal((M, K)).astype(np.float32)
            B = rng.standard_normal((K, N)).astype(np.float32)
            # Truth: f64 then round to f32
            truth = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
            # BPD output
            bpd_out = bpd_matmul(lib, A, B)
            max_err = float(np.abs(truth - bpd_out).max())
            # Unit-error bound: sqrt(K) * eps * max|A| * max|B|
            unit_bound = float(np.sqrt(K)) * eps * float(np.abs(A).max()) * float(np.abs(B).max())
            ratio = max_err / unit_bound if unit_bound > 0 else 0.0
            all_ratios.append(ratio)
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_shape = (M, N, K, seed, label)
            print(f"  {M}x{N}x{K:<11} {seed:<5} {max_err:11.3e}  {unit_bound:11.3e}  {ratio:7.3f}")

    print()
    print("=" * 75)
    print(f"Worst-case ratio: {worst_ratio:.3f}")
    print(f"  at shape {worst_shape[0]}x{worst_shape[1]}x{worst_shape[2]} (seed={worst_shape[3]}, {worst_shape[4]})")
    print()
    print(f"Statistics across {len(all_ratios)} shape*seed combinations:")
    print(f"  min:    {min(all_ratios):.3f}")
    print(f"  median: {sorted(all_ratios)[len(all_ratios)//2]:.3f}")
    print(f"  max:    {max(all_ratios):.3f}")
    print()
    # Suggest substrate constant: ceiling of max with safety margin
    suggested = max(1.0, np.ceil(worst_ratio * 1.5))
    print(f"Suggested substrate factor: {suggested:.0f}")
    print(f"  (worst observed * 1.5 safety margin, rounded up)")
    print()
    print("This factor goes into bench/bit_identical.py:error_bound_gemm().")
    print("Per medayek 2026-05-20: 'calibrated factor becomes a substrate constant'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
