#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_f3_v2_tdd.py — TDD harness for the F3-v2 multi-K-block primitives.

Per Heath: 'decompose that path into lots of smaller subtasks, and use TDD.
We want to have a bit_identical output that fuses the SIMD for any shape of
(K,Q), and so we can break that down into whatever primitives compose to
perform that calculation, and we can test them all in independent test
cases, and tdd that into precision existence.'

Each primitive is tested in isolation against a reference implementation.
Bit-identity required at 0 ULP.

Primitives (built up from simplest):
  P1. bpd_gemm_v2_init(C, M, N)
  P2. bpd_gemm_v2_kblock_accumulate(A, B, C, M, N, K_total, k_start, k_end)
  P3. bpd_gemm_v2_kblock_accumulate_mtail(...)
  P4. bpd_gemm_v2_kblock_accumulate_ntail(...)
  P5. bpd_gemm_v2_full(A, B, C, M, N, K)  [composition: P1 + P2 + P3 + P4]
  P6. bpd_bn_silu_epilogue_simd(C, M, N, alpha, beta)
  P7. bpd_conv2d_bn_silu_fused_cpu_v2  [composition: P5 + P6]

Run: $PY bench/test_f3_v2_tdd.py [test_name]
Or:  $PY bench/test_f3_v2_tdd.py           (runs all)
"""
import ctypes
import os
import sys

import numpy as np

SO = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")


# ──────────────────────────── Test infrastructure ────────────────────────────

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


class TestStatus:
    PASS = "\u2705 PASS"
    FAIL = "\u274c FAIL"
    SKIP = "\u23ed\ufe0f SKIP"
    MISSING = "\u26a0\ufe0f  MISSING"


def assert_bit_identical(name, ref, got, *, max_print=8):
    """Assert ref and got are bit-identical. Return (status, message)."""
    max_ulp, n_diff, n_total = ulp_distance(ref, got)
    if max_ulp == 0:
        return TestStatus.PASS, f"0 ULP / {n_total}"
    # Show first few divergent positions for diagnosis
    ref_flat = np.asarray(ref, dtype=np.float32).reshape(-1)
    got_flat = np.asarray(got, dtype=np.float32).reshape(-1)
    diff_mask = (ref_flat.view(np.uint32) != got_flat.view(np.uint32))
    diff_idx = np.where(diff_mask)[0][:max_print]
    samples = []
    for i in diff_idx:
        samples.append(f"  [{i}]: ref={ref_flat[i]:.7e} got={got_flat[i]:.7e}")
    msg = f"max_ulp={max_ulp} n_diff={n_diff}/{n_total}\n" + "\n".join(samples)
    return TestStatus.FAIL, msg


def setup_lib():
    lib = ctypes.CDLL(SO)
    # Already-existing primitives
    lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    lib.bpd_mm_cpu.restype = None
    if hasattr(lib, 'bpd_mm_cpu_avx1_v2'):
        lib.bpd_mm_cpu_avx1_v2.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
        lib.bpd_mm_cpu_avx1_v2.restype = None
    if hasattr(lib, 'bpd_silu_cpu'):
        lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
        lib.bpd_silu_cpu.restype = None
    # New primitives (TDD targets)
    for fname, sig in [
        ('bpd_gemm_v2_init',                          ([ctypes.c_void_p] + [ctypes.c_int]*2, None)),
        ('bpd_gemm_v2_kblock_accumulate',             ([ctypes.c_void_p]*3 + [ctypes.c_int]*5, None)),
        ('bpd_gemm_v2_kblock_accumulate_mtail',       ([ctypes.c_void_p]*3 + [ctypes.c_int]*5, None)),
        ('bpd_gemm_v2_kblock_accumulate_ntail',       ([ctypes.c_void_p]*3 + [ctypes.c_int]*5, None)),
        ('bpd_gemm_v2_full',                          ([ctypes.c_void_p]*3 + [ctypes.c_int]*3, None)),
        ('bpd_bn_silu_epilogue_simd',                 ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p], None)),
    ]:
        if hasattr(lib, fname):
            argtypes, restype = sig
            setattr(getattr(lib, fname), 'argtypes', argtypes)
            setattr(getattr(lib, fname), 'restype', restype)
    return lib


# ──────────────────────────── Reference impls (Python) ────────────────────────────

def py_scalar_gemm_kblock_accumulate(A, B, C, k_start, k_end):
    """Reference: accumulate C[i,j] += sum_{k=k_start..k_end-1} A[i,k] * B[k,j].
    Linear left-fold per (i, j). Uses float32 throughout.
    Mutates C in place.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    M, K_full = A.shape
    _, N = B.shape
    for i in range(M):
        for j in range(N):
            partial = np.float32(0.0)
            for k in range(k_start, k_end):
                partial = np.float32(partial + np.float32(A[i, k] * B[k, j]))
            C[i, j] = np.float32(C[i, j] + partial)
    return C


# ──────────────────────────── Tests ────────────────────────────

def test_p1_gemm_v2_init(lib):
    """P1: bpd_gemm_v2_init zeros out a buffer."""
    if not hasattr(lib, 'bpd_gemm_v2_init'):
        return TestStatus.MISSING, "bpd_gemm_v2_init not in substrate yet"
    M, N = 17, 23
    # Pre-fill with non-zero garbage
    C = (np.random.default_rng(1).standard_normal((M, N)) * 100).astype(np.float32)
    lib.bpd_gemm_v2_init(C.ctypes.data, M, N)
    expected = np.zeros((M, N), dtype=np.float32)
    return assert_bit_identical("p1", expected, C)


def test_p2_gemm_v2_kblock_accumulate_simple(lib):
    """P2 (basic): single full K-block (k_start=0, k_end=K), M and N divisible by tile."""
    if not hasattr(lib, 'bpd_gemm_v2_kblock_accumulate'):
        return TestStatus.MISSING, "bpd_gemm_v2_kblock_accumulate not in substrate yet"
    rng = np.random.default_rng(2)
    M, N, K = 8, 32, 100  # M divisible by 4 (MR=4), N divisible by 16 (NR=16)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
    
    # Reference: scalar K-block accumulate using Python
    C_ref = np.zeros((M, N), dtype=np.float32)
    py_scalar_gemm_kblock_accumulate(A, B, C_ref, 0, K)
    
    # Substrate
    C = np.zeros((M, N), dtype=np.float32)
    lib.bpd_gemm_v2_kblock_accumulate(
        A.ctypes.data, B.ctypes.data, C.ctypes.data,
        M, N, K, 0, K)
    
    return assert_bit_identical("p2_simple", C_ref, C)


def test_p2_gemm_v2_kblock_accumulate_partial(lib):
    """P2 (partial): partial K-block [k_start=20, k_end=80) within larger K=100."""
    if not hasattr(lib, 'bpd_gemm_v2_kblock_accumulate'):
        return TestStatus.MISSING, "bpd_gemm_v2_kblock_accumulate not in substrate yet"
    rng = np.random.default_rng(3)
    M, N, K = 8, 32, 100
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
    k_start, k_end = 20, 80
    
    C_ref = np.zeros((M, N), dtype=np.float32)
    py_scalar_gemm_kblock_accumulate(A, B, C_ref, k_start, k_end)
    
    C = np.zeros((M, N), dtype=np.float32)
    lib.bpd_gemm_v2_kblock_accumulate(
        A.ctypes.data, B.ctypes.data, C.ctypes.data,
        M, N, K, k_start, k_end)
    
    return assert_bit_identical("p2_partial", C_ref, C)


def test_p2_gemm_v2_kblock_accumulate_into_nonzero(lib):
    """P2 (accumulate into nonzero): two successive partial K-blocks compose correctly."""
    if not hasattr(lib, 'bpd_gemm_v2_kblock_accumulate'):
        return TestStatus.MISSING, "bpd_gemm_v2_kblock_accumulate not in substrate yet"
    rng = np.random.default_rng(4)
    M, N, K = 8, 32, 200
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
    
    # Reference: two K-blocks [0, 100) then [100, 200)
    C_ref = np.zeros((M, N), dtype=np.float32)
    py_scalar_gemm_kblock_accumulate(A, B, C_ref, 0, 100)
    py_scalar_gemm_kblock_accumulate(A, B, C_ref, 100, 200)
    
    # Substrate: same two K-blocks
    C = np.zeros((M, N), dtype=np.float32)
    lib.bpd_gemm_v2_kblock_accumulate(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K, 0, 100)
    lib.bpd_gemm_v2_kblock_accumulate(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K, 100, 200)
    
    return assert_bit_identical("p2_two_blocks", C_ref, C)


def test_p3_gemm_v2_kblock_accumulate_mtail(lib):
    """P3: M-tail handler — accumulate rows [M_blocks*4, M) for a K-range.

    Test exercises M=5, M=7, M=17 (each leaves 1, 3, 1 row in the tail).
    """
    if not hasattr(lib, 'bpd_gemm_v2_kblock_accumulate_mtail'):
        return TestStatus.MISSING, "bpd_gemm_v2_kblock_accumulate_mtail not in substrate yet"
    rng = np.random.default_rng(30)
    K = 100
    failures = []
    for M, N in [(5, 32), (7, 32), (17, 32)]:
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
        A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
        # Reference: scalar accumulation only on the tail rows
        C_ref = (rng.standard_normal((M, N)) * 0.01).astype(np.float32)
        M_blocks = M // 4
        row_start = M_blocks * 4
        for i in range(row_start, M):
            for j in range(N):
                partial = np.float32(0.0)
                for k in range(0, K):
                    partial = np.float32(partial + np.float32(A[i, k] * B[k, j]))
                C_ref[i, j] = np.float32(C_ref[i, j] + partial)
        # Substrate (start C with the SAME prefill as reference)
        C = C_ref.copy()
        # Re-extract pre-call state by recomputing — actually we want the same start state.
        # Simpler: re-init both to the same buffer.
        C_pre = (rng.standard_normal((M, N)) * 0.01).astype(np.float32)
        # Redo reference
        C_ref = C_pre.copy()
        for i in range(row_start, M):
            for j in range(N):
                partial = np.float32(0.0)
                for k in range(0, K):
                    partial = np.float32(partial + np.float32(A[i, k] * B[k, j]))
                C_ref[i, j] = np.float32(C_ref[i, j] + partial)
        # Substrate call
        C = C_pre.copy()
        lib.bpd_gemm_v2_kblock_accumulate_mtail(
            A.ctypes.data, B.ctypes.data, C.ctypes.data,
            M, N, K, 0, K)
        status, msg = assert_bit_identical(f"p3_M{M}", C_ref, C)
        if status != TestStatus.PASS:
            failures.append(f"M={M}: {msg.splitlines()[0]}")
    if failures:
        return TestStatus.FAIL, "; ".join(failures)
    return TestStatus.PASS, "M=5, M=7, M=17 all 0 ULP"


def test_p4_gemm_v2_kblock_accumulate_ntail(lib):
    """P4: N-tail handler — accumulate cols [N_blocks*16, N) for all rows.

    Test exercises N=15, N=17, N=23 (each leaves 15, 1, 7 col in the tail).
    """
    if not hasattr(lib, 'bpd_gemm_v2_kblock_accumulate_ntail'):
        return TestStatus.MISSING, "bpd_gemm_v2_kblock_accumulate_ntail not in substrate yet"
    rng = np.random.default_rng(40)
    K = 100
    failures = []
    for M, N in [(8, 15), (8, 17), (8, 23)]:
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
        A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
        C_pre = (rng.standard_normal((M, N)) * 0.01).astype(np.float32)
        # Reference: scalar accumulation only on the tail cols
        C_ref = C_pre.copy()
        N_blocks = N // 16
        col_start = N_blocks * 16
        for i in range(M):
            for j in range(col_start, N):
                partial = np.float32(0.0)
                for k in range(0, K):
                    partial = np.float32(partial + np.float32(A[i, k] * B[k, j]))
                C_ref[i, j] = np.float32(C_ref[i, j] + partial)
        # Substrate
        C = C_pre.copy()
        lib.bpd_gemm_v2_kblock_accumulate_ntail(
            A.ctypes.data, B.ctypes.data, C.ctypes.data,
            M, N, K, 0, K)
        status, msg = assert_bit_identical(f"p4_N{N}", C_ref, C)
        if status != TestStatus.PASS:
            failures.append(f"N={N}: {msg.splitlines()[0]}")
    if failures:
        return TestStatus.FAIL, "; ".join(failures)
    return TestStatus.PASS, "N=15, N=17, N=23 all 0 ULP"


def test_p5_gemm_v2_full(lib):
    """P5: bpd_gemm_v2_full(A, B, C, M, N, K) — full GEMM composing P1+P2+P3+P4.

    Substrate-vs-substrate against bpd_mm_cpu_avx1_v2 (the existing v2 GEMM,
    already verified bit-identical with scalar bpd_mm_cpu).
    Tests M=8/N=32/K=100 (no tails) and edge case M=17/N=23/K=576 (both
    tails AND multi-K-block).
    """
    if not hasattr(lib, 'bpd_gemm_v2_full'):
        return TestStatus.MISSING, "bpd_gemm_v2_full not in substrate yet"
    rng = np.random.default_rng(50)
    failures = []
    for label, M, N, K in [
        ("simple",     8,  32,  100),
        ("mtail+ntail",17, 23,  100),
        ("multi-K",     8,  32,  576),
        ("mtail+ntail+multiK", 17, 23, 576),
    ]:
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
        A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
        # Reference: bpd_mm_cpu_avx1_v2 (existing verified GEMM)
        C_ref = np.zeros((M, N), dtype=np.float32)
        lib.bpd_mm_cpu_avx1_v2(A.ctypes.data, B.ctypes.data, C_ref.ctypes.data, M, N, K)
        # Substrate: P5 composition
        C = np.zeros((M, N), dtype=np.float32)
        lib.bpd_gemm_v2_full(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
        status, msg = assert_bit_identical(f"p5_{label}", C_ref, C)
        if status != TestStatus.PASS:
            failures.append(f"{label}: {msg.splitlines()[0]}")
    if failures:
        return TestStatus.FAIL, "; ".join(failures)
    return TestStatus.PASS, "simple, mtail+ntail, multi-K, all-edges 0 ULP"


def test_p7_conv2d_bn_silu_fused_v2_multi_k(lib):
    """P7: full F3-v2 composition. Tests on YOLOv5n CBS shapes with K > Q
    (the multi-K-block cases that previously diverged).

    Substrate-vs-substrate against bpd_conv2d_bn_silu_fused_cpu (F3 v1, which
    via the dispatcher uses v2 GEMM + scalar epilogue and is already verified
    BIT_IDENTICAL with PyTorch).
    """
    if not hasattr(lib, 'bpd_conv2d_bn_silu_fused_cpu_v2'):
        return TestStatus.MISSING, "bpd_conv2d_bn_silu_fused_cpu_v2 not in substrate yet"
    if not hasattr(lib, 'bpd_conv2d_bn_silu_fused_cpu'):
        return TestStatus.MISSING, "bpd_conv2d_bn_silu_fused_cpu (F3 v1 reference) not in substrate"
    # Register argtypes for both
    lib.bpd_conv2d_bn_silu_fused_cpu.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
    lib.bpd_conv2d_bn_silu_fused_cpu.restype = None
    lib.bpd_conv2d_bn_silu_fused_cpu_v2.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
    lib.bpd_conv2d_bn_silu_fused_cpu_v2.restype = None

    rng = np.random.default_rng(70)
    # Focus on K > Q (the previously-failing cases) + a sanity-check K <= Q
    shapes = [
        # (label, N, Cin, H, W, Cout, kH, kW, stride, pad, K_value)
        ("K<=Q L1 3x3 s=2",     1,  16, 320, 320,  32, 3, 3, 2, 1),  # K=144
        ("K>Q  L5 3x3 s=2",     1,  64,  80,  80, 128, 3, 3, 2, 1),  # K=576
        ("K>Q  L7 3x3 s=2",     1, 128,  40,  40, 256, 3, 3, 2, 1),  # K=1152
        ("K>Q  L9 sppf cv2",    1, 512,  20,  20, 256, 1, 1, 1, 0),  # K=512
        ("K>Q  L21 3x3 s=2",    1, 128,  40,  40, 128, 3, 3, 2, 1),  # K=1152
    ]
    failures = []
    for label, N, Cin, H, W, Cout, kH, kW, stride, pad in shapes:
        K_value = Cin * kH * kW
        x = (rng.standard_normal((N, Cin, H, W)) * 0.3).astype(np.float32)
        w = (rng.standard_normal((Cout, Cin, kH, kW)) * (1.0/np.sqrt(K_value))).astype(np.float32)
        gamma = (rng.standard_normal(Cout) * 0.3 + 1.0).astype(np.float32)
        bn_beta = (rng.standard_normal(Cout) * 0.1).astype(np.float32)
        mean = (rng.standard_normal(Cout) * 0.2).astype(np.float32)
        var = (np.abs(rng.standard_normal(Cout) * 0.5) + 0.1).astype(np.float32)
        eps = 1e-5
        inv_std = (1.0 / np.sqrt(var + np.float32(eps))).astype(np.float32)
        alpha = np.ascontiguousarray((gamma * inv_std).astype(np.float32))
        beta = np.ascontiguousarray((bn_beta - mean * alpha).astype(np.float32))
        H_out = (H + 2*pad - kH) // stride + 1
        W_out = (W + 2*pad - kW) // stride + 1

        # Reference: F3 v1
        ref = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
        lib.bpd_conv2d_bn_silu_fused_cpu(
            x.ctypes.data, w.ctypes.data,
            alpha.ctypes.data, beta.ctypes.data,
            ref.ctypes.data,
            N, Cin, H, W, Cout, kH, kW, stride, stride, pad, pad)

        # Substrate: F3 v2 fully-fused (the TDD target)
        fused = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
        lib.bpd_conv2d_bn_silu_fused_cpu_v2(
            x.ctypes.data, w.ctypes.data,
            alpha.ctypes.data, beta.ctypes.data,
            fused.ctypes.data,
            N, Cin, H, W, Cout, kH, kW, stride, stride, pad, pad)

        status, msg = assert_bit_identical(f"p7_{label}", ref, fused)
        if status != TestStatus.PASS:
            failures.append(f"{label}: {msg.splitlines()[0]}")
    if failures:
        return TestStatus.FAIL, "; ".join(failures)
    return TestStatus.PASS, "All YOLO CBS shapes incl K>Q: 0 ULP"


def test_p6_bn_silu_epilogue_simd(lib):
    """P6: SIMD epilogue applied to a known-value tensor."""
    if not hasattr(lib, 'bpd_bn_silu_epilogue_simd'):
        return TestStatus.MISSING, "bpd_bn_silu_epilogue_simd not in substrate yet"
    rng = np.random.default_rng(6)
    M, N = 4, 32  # M=4 for register block, N=32 = 2 tiles of 16
    C_initial = (rng.standard_normal((M, N)) * 0.5).astype(np.float32)
    alpha = (rng.standard_normal(M) * 0.3 + 1.0).astype(np.float32)
    beta = (rng.standard_normal(M) * 0.1).astype(np.float32)
    alpha = np.ascontiguousarray(alpha); beta = np.ascontiguousarray(beta)
    
    # Reference: scalar epilogue (matches F3 v1's per-element loop)
    C_ref = C_initial.copy()
    for i in range(M):
        a = alpha[i]; b = beta[i]
        for j in range(N):
            x = np.float32(a * C_ref[i, j] + b)
            C_ref[i, j] = np.float32(x / np.float32(1.0 + np.float32(np.exp(-x))))
    
    # Substrate
    C = C_initial.copy()
    lib.bpd_bn_silu_epilogue_simd(
        C.ctypes.data, M, N, alpha.ctypes.data, beta.ctypes.data)
    
    return assert_bit_identical("p6", C_ref, C)


# ──────────────────────────── Test runner ────────────────────────────

TESTS = [
    ("P1 init",                          test_p1_gemm_v2_init),
    ("P2 simple (full K-block)",         test_p2_gemm_v2_kblock_accumulate_simple),
    ("P2 partial (k-range)",             test_p2_gemm_v2_kblock_accumulate_partial),
    ("P2 two-blocks (compose)",          test_p2_gemm_v2_kblock_accumulate_into_nonzero),
    ("P3 M-tail",                        test_p3_gemm_v2_kblock_accumulate_mtail),
    ("P4 N-tail",                        test_p4_gemm_v2_kblock_accumulate_ntail),
    ("P5 full GEMM (compose)",           test_p5_gemm_v2_full),
    ("P6 SIMD epilogue",                 test_p6_bn_silu_epilogue_simd),
    ("P7 F3-v2 multi-K-block",           test_p7_conv2d_bn_silu_fused_v2_multi_k),
]


def main():
    lib = setup_lib()
    print(f"Substrate: {SO}")
    print()
    print(f"{'Test':<40} {'Result':<60}")
    print("-" * 102)
    n_pass = 0
    n_fail = 0
    n_missing = 0
    for name, test_fn in TESTS:
        try:
            status, msg = test_fn(lib)
        except Exception as e:
            status, msg = TestStatus.FAIL, f"exception: {e}"
        first_line = msg.splitlines()[0] if msg else ""
        print(f"{name:<40} {status} {first_line}")
        rest = msg.splitlines()[1:]
        for r in rest:
            print(f"{'':<40} {'':<10}{r}")
        if status == TestStatus.PASS:
            n_pass += 1
        elif status == TestStatus.MISSING:
            n_missing += 1
        else:
            n_fail += 1
    print()
    print(f"PASS: {n_pass}, FAIL: {n_fail}, MISSING: {n_missing}")
    sys.exit(0 if (n_fail == 0 and n_missing == 0) else 1)


if __name__ == "__main__":
    main()
