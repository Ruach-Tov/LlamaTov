#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""perf_gemm_sweep.py — Performance sweep across BIT_IDENTICAL gemm patterns.

Per Heath's direction 2026-05-20 ~22:00 UTC: surpass OpenBLAS by sweeping
the parameter space to extremes they never explored.

For each of the 107 patterns that achieved BIT_IDENTICAL with cblas_sgemm
in verify_gemm_sweep.py:
  1. Measure GFLOPS at multiple (M, N, K) shapes
  2. Compare to direct cblas_sgemm baseline
  3. Identify which (P, UM, UN, SIMD) ratios are most efficient

Note: our scalar-mimic kernels do NOT use SIMD intrinsics, so they will
be slower than OpenBLAS's hand-tuned assembly. The substantive value is:
  - Identifying which scalar-mimic combinations are most efficient
  - Establishing a performance baseline for future SIMD-aware emitters
  - Confirming the parameter that crystallizes substrate-design choice

Usage:
    python3 bench/perf_gemm_sweep.py
"""
import ctypes
import os
import sys
import time
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
    lib = ctypes.CDLL(so_path)
    count = ctypes.c_int.in_dll(lib, "gemm_dispatch_count").value
    names_arr = (ctypes.c_char_p * count).in_dll(lib, "gemm_dispatch_names")
    names = [names_arr[i].decode() for i in range(count)]
    dispatch_arr = (GEMM_FN_TYPE * count).in_dll(lib, "gemm_dispatch")
    return lib, count, names, dispatch_arr


def load_cblas():
    p = os.environ.get("OPENBLAS_SO",
        "/nix/store/dsy2fc8jkx0vkqcw54jh1f8pxvshg7ks-openblas-0.3.32/lib/libopenblas.so.0.3")
    blas = ctypes.CDLL(p)
    blas.cblas_sgemm.restype = None
    blas.cblas_sgemm.argtypes = [ctypes.c_int]*3 + [ctypes.c_int]*3 + [
        ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_int]
    return blas


def cblas_sgemm(blas, A, B, C):
    M, K = A.shape
    _, N = B.shape
    blas.cblas_sgemm(101, 111, 111, M, N, K, 1.0,
                      A.ctypes.data, K, B.ctypes.data, N,
                      0.0, C.ctypes.data, N)


def parse_pattern(name):
    parts = name.split("_")
    p = int(parts[1][1:])
    q = int(parts[2][1:])
    um = int(parts[3][2:])
    un = int(parts[4][2:])
    simd = int(parts[5][4:])
    krule = "_".join(parts[6:])
    return (p, q, um, un, simd, krule)


def gflops(M, N, K, time_seconds):
    """2*M*N*K FMA-equivalents → GFLOPS."""
    return (2.0 * M * N * K) / (time_seconds * 1e9)


def bench_kernel(fn, A, B, C, repeats=3, target_time=0.05):
    """Run kernel until total time >= target_time, return median GFLOPS."""
    M, K = A.shape
    _, N = B.shape

    # Warmup
    fn(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)

    # Calibrate: how many iters fit in target_time?
    t0 = time.perf_counter()
    fn(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
    dt = time.perf_counter() - t0
    if dt < 1e-6:
        dt = 1e-6
    iters = max(1, min(100, int(target_time / dt)))

    # Measure repeats × iters
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
        elapsed = time.perf_counter() - t0
        samples.append(gflops(M, N, K, elapsed / iters))
    return sorted(samples)[len(samples) // 2]  # median


def main():
    so_path = os.environ.get("GEMM_SO", "build/gemm.so")
    if not os.path.exists(so_path):
        sys.exit(f"{so_path} not found. Run verify_gemm_sweep.py first to build it.")

    print(f"Loading {so_path}...")
    lib, count, names, dispatch = load_gemm_dispatch(so_path)
    print(f"  {count} kernels available")
    blas = load_cblas()

    shapes = [
        (16, 16, 4096),     # Sandy's L1 #6 headline
        (128, 128, 128),    # square mid
        (256, 256, 256),    # square larger
        (67, 89, 113),      # irregular
    ]

    # Find BIT_IDENTICAL patterns (filter to those that match cblas_sgemm at all shapes)
    # We re-verify here rather than relying on prior output.
    print("Filtering to BIT_IDENTICAL patterns (this may take ~30s)...")
    refs = {}
    for M, N, K in shapes:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        C_ref = np.zeros((M, N), dtype=np.float32)
        cblas_sgemm(blas, A, B, C_ref)
        refs[(M, N, K)] = (A, B, C_ref)

    bi_indices = []
    for k in range(count):
        ok = True
        for shape in shapes:
            A, B, ref = refs[shape]
            out = np.zeros(ref.shape, dtype=np.float32)
            try:
                dispatch[k](A.ctypes.data, B.ctypes.data, out.ctypes.data, *ref.shape, A.shape[1])
            except Exception:
                ok = False
                break
            if ulp(ref, out) != 0:
                ok = False
                break
        if ok:
            bi_indices.append(k)
    print(f"  {len(bi_indices)} BIT_IDENTICAL patterns to benchmark")
    print()

    # Baseline: cblas_sgemm
    print("Baseline GFLOPS (direct cblas_sgemm, hand-tuned assembly):")
    print(f"  {'shape':<24} {'GFLOPS':<10}")
    cblas_gflops = {}
    for shape in shapes:
        A, B, _ = refs[shape]
        C = np.zeros(refs[shape][2].shape, dtype=np.float32)
        def run_cblas(_aa, _bb, _cc, _M, _N, _K):
            cblas_sgemm(blas, A, B, C)
        g = bench_kernel(run_cblas, A, B, C, repeats=5)
        cblas_gflops[shape] = g
        print(f"  {str(shape):<24} {g:.2f}")
    print()

    # Per-pattern benchmarks
    print(f"Sweep across {len(bi_indices)} BIT_IDENTICAL patterns...")
    print(f"  {'pattern':<60}", end="")
    for s in shapes:
        print(f" {f'{s[0]}x{s[1]}x{s[2]}':<14}", end="")
    print()

    results = []
    for k in bi_indices:
        gs = []
        for shape in shapes:
            A, B, _ = refs[shape]
            C = np.zeros(refs[shape][2].shape, dtype=np.float32)
            g = bench_kernel(dispatch[k], A, B, C, repeats=3, target_time=0.02)
            gs.append(g)
        results.append((k, names[k], gs))
        # Print a compact row
        short = names[k].replace("gemm_", "")
        print(f"  {short:<60}", end="")
        for g in gs:
            print(f" {g:<14.3f}", end="")
        print()

    print()
    print("═" * 80)
    print("Top 5 patterns by total GFLOPS across shapes:")
    results.sort(key=lambda r: -sum(r[2]))
    for rank, (k, name, gs) in enumerate(results[:5], 1):
        pat = parse_pattern(name)
        total = sum(gs)
        # vs cblas baseline
        ratios = [g / cblas_gflops[s] for g, s in zip(gs, shapes)]
        ratio_str = " ".join(f"{r:.3f}x" for r in ratios)
        print(f"  #{rank}: P={pat[0]:4} Q={pat[1]:3} UM={pat[2]:2} UN={pat[3]:2} SIMD={pat[4]:2}")
        print(f"       GFLOPS={[f'{g:.2f}' for g in gs]}  total={total:.2f}")
        print(f"       vs cblas: {ratio_str}")
    print()

    # Find any pattern that BEATS cblas at any shape
    print("Patterns that outperform cblas_sgemm at some shape:")
    found_win = False
    for k, name, gs in results:
        wins = [(s, g, cblas_gflops[s]) for s, g in zip(shapes, gs) if g > cblas_gflops[s]]
        if wins:
            pat = parse_pattern(name)
            for s, g, c in wins:
                print(f"  {name}: shape={s} {g:.2f} GFLOPS vs cblas {c:.2f} (+{(g/c-1)*100:.0f}%)")
                found_win = True
    if not found_win:
        print("  None — cblas_sgemm wins on every shape (expected: hand-tuned assembly vs scalar-mimic).")
        print("  This is the expected baseline. Substrate-design value: the BIT_IDENTICAL set")
        print("  identifies WHICH parameter combinations carry substrate meaning vs performance dials.")


if __name__ == "__main__":
    main()
