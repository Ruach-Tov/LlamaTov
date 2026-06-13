#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""bench_L0_spec.py \u2014 Phase 3.CAT.SPEC.a empirical experiment.

Measures whether GCC's const-propagation on compile-time-fixed M, N, K
yields a measurable wall-clock improvement on the L0_focus shape
(M=16, K=108, N=102400 \u2014 the dominant time-consumer in YOLOv5n).

Compares:
  bpd_mm_cpu_avx1_v2(A, B, C, 16, 102400, 108)    \u2014 generic
  bpd_mm_cpu_avx1_v2_L0(A, B, C)                  \u2014 specialized

Bit-identity required first; if any divergence, gdb time.
"""
import ctypes
import os
import sys
import time

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


def main():
    lib = ctypes.CDLL(SO)
    lib.bpd_mm_cpu_avx1_v2.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    lib.bpd_mm_cpu_avx1_v2.restype = None
    if not hasattr(lib, 'bpd_mm_cpu_avx1_v2_L0'):
        print("ERROR: bpd_mm_cpu_avx1_v2_L0 not in substrate")
        sys.exit(2)
    lib.bpd_mm_cpu_avx1_v2_L0.argtypes = [ctypes.c_void_p]*3
    lib.bpd_mm_cpu_avx1_v2_L0.restype = None

    # L0_focus dimensions
    M, K, N = 16, 108, 102400
    print(f"L0_focus shape: M={M}, K={K}, N={N} ({M*N*4 / 1024:.0f} KB output)")
    print()

    rng = np.random.default_rng(2026)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)

    # Bit-identity check
    C_generic = np.zeros((M, N), dtype=np.float32)
    lib.bpd_mm_cpu_avx1_v2(A.ctypes.data, B.ctypes.data, C_generic.ctypes.data, M, N, K)
    C_spec = np.zeros((M, N), dtype=np.float32)
    lib.bpd_mm_cpu_avx1_v2_L0(A.ctypes.data, B.ctypes.data, C_spec.ctypes.data)

    max_ulp, n_diff, n_total = ulp_distance(C_generic, C_spec)
    if max_ulp == 0:
        print(f"BIT-IDENTITY: \u2705 0 ULP / {n_total}")
    else:
        print(f"BIT-IDENTITY: \u274c max_ulp={max_ulp} n_diff={n_diff}/{n_total}")
        sys.exit(1)
    print()

    # Wall-clock measurement (best-of-N runs to filter noise)
    runs = 20
    times_generic = []
    times_spec = []

    # Warmups
    lib.bpd_mm_cpu_avx1_v2(A.ctypes.data, B.ctypes.data, C_generic.ctypes.data, M, N, K)
    lib.bpd_mm_cpu_avx1_v2_L0(A.ctypes.data, B.ctypes.data, C_spec.ctypes.data)

    # Interleave to avoid systematic thermal bias
    for i in range(runs):
        t0 = time.perf_counter()
        lib.bpd_mm_cpu_avx1_v2(A.ctypes.data, B.ctypes.data, C_generic.ctypes.data, M, N, K)
        times_generic.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        lib.bpd_mm_cpu_avx1_v2_L0(A.ctypes.data, B.ctypes.data, C_spec.ctypes.data)
        times_spec.append(time.perf_counter() - t0)

    min_g = min(times_generic) * 1000
    min_s = min(times_spec) * 1000
    med_g = sorted(times_generic)[runs // 2] * 1000
    med_s = sorted(times_spec)[runs // 2] * 1000

    print(f"Runs: {runs}, interleaved")
    print()
    print(f"{'Variant':<20} {'min ms':>10} {'median ms':>12}")
    print("-" * 45)
    print(f"{'generic v2':<20} {min_g:>9.3f}   {med_g:>10.3f}")
    print(f"{'specialized L0':<20} {min_s:>9.3f}   {med_s:>10.3f}")
    print()
    speedup_min = min_g / min_s
    speedup_med = med_g / med_s
    print(f"Speedup (min): {speedup_min:.3f}x")
    print(f"Speedup (med): {speedup_med:.3f}x")
    print()
    if speedup_min > 1.10:
        print(f"\u2705 Specialization wins substantively (>{(speedup_min-1)*100:.1f}%% faster at min)")
        print("   The compile-time-const M, N, K enable GCC optimizations beyond generic.")
    elif speedup_min > 1.02:
        print(f"\u26a0\ufe0f  Specialization wins marginally ({(speedup_min-1)*100:.1f}%% at min)")
        print("   Substrate-design decision: is the maintenance cost worth this?")
    else:
        print(f"\u274c Specialization does NOT win (only {(speedup_min-1)*100:.1f}%% at min)")
        print("   Empirical finding: GCC -O2 already extracts what it can from the generic kernel.")
        print("   Substrate-design conclusion: template-style specialization is not the path.")


if __name__ == "__main__":
    main()
