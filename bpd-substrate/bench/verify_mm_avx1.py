# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify bpd_mm_cpu_avx1 BIT_IDENTICAL with bpd_mm_cpu (scalar).

Test at YOLOv5n GEMM shapes: M=Cout, N=spatial_out, K=Cin*kH*kW.
For 640x640 input:
  L0 focus 6x6  M=16,  K=108,   N=102400
  L1 cbs 3x3    M=32,  K=144,   N=25600
  L3 cbs 3x3    M=64,  K=288,   N=6400
  L5 cbs 3x3    M=128, K=576,   N=1600
  L7 cbs 3x3    M=256, K=1152,  N=400
  L9 sppf cv2   M=256, K=512,   N=400
  L13 c3 cv3    M=128, K=128,   N=1600
  L17 c3 cv3    M=64,  K=128,   N=6400
"""
import ctypes, os, sys
import numpy as np
import time

SO = "/tmp/bpd_test/build/bpd_cpu.so"
lib = ctypes.CDLL(SO)
lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
lib.bpd_mm_cpu.restype = None
has_avx1 = hasattr(lib, 'bpd_mm_cpu_avx1')
if has_avx1:
    lib.bpd_mm_cpu_avx1.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    lib.bpd_mm_cpu_avx1.restype = None


def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64); bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai); bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


SHAPES = [
    ("L0 focus 6x6",   16,  108,  102400),
    ("L1 cbs 3x3",     32,  144,   25600),
    ("L3 cbs 3x3",     64,  288,    6400),
    ("L5 cbs 3x3",    128,  576,    1600),
    ("L7 cbs 3x3",    256, 1152,     400),
    ("L9 sppf cv2",   256,  512,     400),
    ("L13 c3 cv3",    128,  128,    1600),
    ("L17 c3 cv3",     64,  128,    6400),
    ("Small odd",      17,  100,    1003),  # exercise the tail
    ("Tiny",            5,    7,      11),  # purely scalar tail
]

print(f"AVX1 kernel available: {has_avx1}")
print()
print(f"{'Label':<18} {'M':>6} {'K':>6} {'N':>8} {'Scalar ms':>10} {'AVX1 ms':>10} {'Speedup':>8} {'ULP':>6}")
print("-" * 88)

rng = np.random.default_rng(2026)
for label, M, K, N in SHAPES:
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)

    # Scalar
    C_scalar = np.zeros((M, N), dtype=np.float32)
    # warm
    lib.bpd_mm_cpu(A.ctypes.data, B.ctypes.data, C_scalar.ctypes.data, M, N, K)
    t0 = time.perf_counter()
    lib.bpd_mm_cpu(A.ctypes.data, B.ctypes.data, C_scalar.ctypes.data, M, N, K)
    t_scalar = time.perf_counter() - t0

    # AVX1
    if has_avx1:
        C_avx = np.zeros((M, N), dtype=np.float32)
        lib.bpd_mm_cpu_avx1(A.ctypes.data, B.ctypes.data, C_avx.ctypes.data, M, N, K)
        t0 = time.perf_counter()
        lib.bpd_mm_cpu_avx1(A.ctypes.data, B.ctypes.data, C_avx.ctypes.data, M, N, K)
        t_avx = time.perf_counter() - t0
        max_ulp, n_diff, n_total = ulp_distance(C_scalar, C_avx)
        speedup = t_scalar / t_avx if t_avx > 0 else 0
        status = "BIT_IDENTICAL" if max_ulp == 0 else f"DIVERGENT max={max_ulp} n={n_diff}/{n_total}"
        print(f"{label:<18} {M:>6} {K:>6} {N:>8} {t_scalar*1000:>8.2f}   {t_avx*1000:>8.2f}    {speedup:>6.2f}x   {status}")
    else:
        print(f"{label:<18}  AVX1 kernel not available")
