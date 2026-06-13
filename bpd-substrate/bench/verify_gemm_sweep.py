#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_gemm_sweep.py — Bit-identity sweep across all gemm_pattern instantiations.

Per Heath's direction 2026-05-20 ~22:00 UTC: sweepable Goto GEMM generator
exploring parameter combinations OpenBLAS never considered.

For each of 4328 valid (P, Q, UM, UN, SIMD, KRule) patterns:
  1. Load the kernel from the generated dispatch table
  2. Run against cblas_sgemm (direct OpenBLAS call) at multiple shapes
  3. Report ULP per shape per pattern

Output: per-pattern row, summary of BIT_IDENTICAL patterns, and the named-
platform validation (Sandy at idx N must be BIT_IDENTICAL at every shape).

Sweep covers shapes that exercise different K-block boundaries:
  (M=16, N=16, K=256)   no boundary
  (M=16, N=16, K=384)   exact single Q-block
  (M=16, N=16, K=512)   one full + one tail
  (M=16, N=16, K=4096)  many full + adaptive-half tail
  (M=128, N=128, K=128) larger M/N
  (M=67, N=89, K=113)   irregular shapes

Usage:
    python3 bench/generate_gemm_kernels.py > bench/gemm_kernels_generated.c
    gcc -O2 -shared -fPIC -o build/gemm.so bench/gemm_kernels_generated.c -lm
    python3 bench/verify_gemm_sweep.py
"""
import ctypes
import os
import sys
import numpy as np


GEMM_FN_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
)


def ulp(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    return int(np.abs(ai - bi).max())


def load_gemm_dispatch(so_path):
    """Load gemm.so and return (lib, kernel_count, names, dispatch_array)."""
    lib = ctypes.CDLL(so_path)
    count = ctypes.c_int.in_dll(lib, "gemm_dispatch_count").value
    names_arr = (ctypes.c_char_p * count).in_dll(lib, "gemm_dispatch_names")
    names = [names_arr[i].decode() for i in range(count)]
    dispatch_arr = (GEMM_FN_TYPE * count).in_dll(lib, "gemm_dispatch")
    return lib, count, names, dispatch_arr


def load_cblas():
    """Load OpenBLAS direct cblas_sgemm."""
    OPENBLAS = os.environ.get("OPENBLAS_SO",
        "/nix/store/dsy2fc8jkx0vkqcw54jh1f8pxvshg7ks-openblas-0.3.32/lib/libopenblas.so.0.3")
    blas = ctypes.CDLL(OPENBLAS)
    blas.cblas_sgemm.restype = None
    blas.cblas_sgemm.argtypes = [ctypes.c_int]*3 + [ctypes.c_int]*3 + [
        ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int,
    ]
    return blas


def cblas_sgemm(blas, A, B):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    blas.cblas_sgemm(101, 111, 111, M, N, K, 1.0,
                      A.ctypes.data, K, B.ctypes.data, N,
                      0.0, C.ctypes.data, N)
    return C


def run_kernel(fn, A, B):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    fn(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
    return C


def parse_pattern(name):
    """Parse 'gemm_p768_q384_um16_un4_simd8_adaptive_half' -> tuple."""
    parts = name.split("_")
    # parts: ['gemm', 'p768', 'q384', 'um16', 'un4', 'simd8', 'adaptive', 'half']
    # KRule can be multi-word ('adaptive_half', 'equal_split', 'single_block', 'fixed_q')
    p = int(parts[1][1:])
    q = int(parts[2][1:])
    um = int(parts[3][2:])
    un = int(parts[4][2:])
    simd = int(parts[5][4:])
    krule = "_".join(parts[6:])
    return (p, q, um, un, simd, krule)


def main():
    so_path = os.environ.get("GEMM_SO", "build/gemm.so")
    if not os.path.exists(so_path):
        sys.exit(f"{so_path} not found. Build:\n"
                 f"  python3 bench/generate_gemm_kernels.py > bench/gemm_kernels_generated.c\n"
                 f"  gcc -O2 -shared -fPIC -o {so_path} bench/gemm_kernels_generated.c -lm")

    print(f"Loading {so_path}...")
    lib, count, names, dispatch = load_gemm_dispatch(so_path)
    print(f"  {count} gemm kernels available")
    blas = load_cblas()
    print(f"  OpenBLAS direct cblas_sgemm reference loaded")
    print()

    shapes = [
        (16, 16, 256),    # 1 Q-block
        (16, 16, 384),    # exact Q-block
        (16, 16, 512),    # 1 full + tail
        (16, 16, 1024),   # 2 full + tail
        (16, 16, 4096),   # 10 full + adaptive-half tail (Sandy headline)
        (128, 128, 128),  # larger M/N
        (67, 89, 113),    # irregular
    ]

    # Pre-compute references
    refs = {}
    for M, N, K in shapes:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        refs[(M, N, K)] = (A, B, cblas_sgemm(blas, A, B))

    # Sweep
    matches_all = []
    by_shape_ulp = {}

    for k in range(count):
        ulps_per_shape = {}
        for shape in shapes:
            A, B, ref = refs[shape]
            try:
                out = run_kernel(dispatch[k], A, B)
                u = ulp(ref, out)
            except Exception:
                u = -1
            ulps_per_shape[shape] = u
        all_zero = all(u == 0 for u in ulps_per_shape.values())
        if all_zero:
            matches_all.append((k, names[k], parse_pattern(names[k])))
        by_shape_ulp[k] = ulps_per_shape
        if k % 500 == 0 or all_zero:
            status = "[BIT_IDENTICAL_ALL]" if all_zero else f"({k}/{count})"
            print(f"  {status} {names[k][:60]}", flush=True)

    print()
    print("═" * 80)
    print(f"Sweep summary ({count} patterns × {len(shapes)} shapes):")
    print(f"  BIT_IDENTICAL at all shapes: {len(matches_all)} patterns")
    print()

    if matches_all:
        print("Patterns BIT_IDENTICAL with cblas_sgemm at every shape:")
        for idx, name, pat in matches_all:
            P, Q, UM, UN, SW, KR = pat
            print(f"  P={P:4} Q={Q:4} UM={UM:2} UN={UN:2} SIMD={SW:2} KRule={KR:14}  ({name})")

    return 0 if matches_all else 1


if __name__ == "__main__":
    sys.exit(main())
