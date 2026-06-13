#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_correctness.py — BPD CPU kernel correctness harness.

Verifies that BPD CPU kernels produce mathematically correct results using
two complementary strategies:

1. EXACT cases: inputs chosen so the true answer is representable exactly
   in float32 (identity matrix, all-ones, small integer matrices).
   These must be BIT_IDENTICAL.

2. RANDOM cases: verified using the standard Wilkinson backward-error bound
   for sequential float32 GEMM:

       max_i,j |C_bpd[i,j] - C_true[i,j]| / (||A||_1 * ||B||_1) <= K * eps32

   where:
     - C_true is computed in float64 (exact to float32 precision)
     - ||A||_1 = max column 1-norm of A
     - ||B||_1 = max column 1-norm of B
     - K       = inner dimension of the matmul
     - eps32   = machine epsilon for float32 (~5.96e-8)

   This is the correct way to certify a GEMM implementation.  It does NOT
   compare BPD against PyTorch/cblas bit-for-bit, because both use IEEE 754
   accumulation in different (equally valid) orders.  The goal of BPD is to
   subsume BLAS by producing correct results through its own generated
   kernels, not by wrapping BLAS.

Usage:
  python3 bench/test_correctness.py
  BPD_VERBOSE=1 python3 bench/test_correctness.py   # show worst element
"""

import ctypes
import os
import sys
import numpy as np

# ── Load bpd_cpu.so ──────────────────────────────────────────────────────────
_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'bpd_cpu.so')
_lib = ctypes.CDLL(os.path.realpath(_lib_path))

def _reg(name, argtypes, restype=None):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype  = restype
    return fn

_mm      = _reg('bpd_mm_cpu',           [ctypes.c_void_p]*3 + [ctypes.c_int]*3)
_mm_br   = _reg('bpd_mm_bias_relu_cpu', [ctypes.c_void_p]*4 + [ctypes.c_int]*3)
_linear  = _reg('bpd_linear_cpu',       [ctypes.c_void_p]*4 + [ctypes.c_int]*3)

VERBOSE = os.environ.get('BPD_VERBOSE', '0') == '1'
EPS32   = float(np.float32(1.0) - np.nextafter(np.float32(1.0), np.float32(0.0)))

# ── Helpers ──────────────────────────────────────────────────────────────────
pass_count = 0
fail_count = 0

def check_exact(name: str, got: np.ndarray, expected: np.ndarray):
    """Require bit-identical result."""
    global pass_count, fail_count
    if np.array_equal(got, expected):
        print(f"  PASS  {name}  (BIT_IDENTICAL)")
        pass_count += 1
    else:
        print(f"  FAIL  {name}  (not bit-identical)")
        diff_idx = np.where(got.ravel() != expected.ravel())[0]
        for i in diff_idx[:3]:
            print(f"        [{i}] got={got.ravel()[i]:.8g}  "
                  f"expected={expected.ravel()[i]:.8g}")
        fail_count += 1

def check_backward(name: str, got: np.ndarray,
                   A: np.ndarray, B: np.ndarray, K: int,
                   C_true_f64: np.ndarray):
    """
    Wilkinson backward-error check:
      max|got - C_true| / (||A||_1 * ||B||_1) <= K * eps32
    C_true_f64 must be computed in float64.
    """
    global pass_count, fail_count
    abs_err  = float(np.abs(got.astype(np.float64) - C_true_f64).max())
    norm_A   = float(np.abs(A).sum(axis=0).max())   # max column 1-norm
    norm_B   = float(np.abs(B).sum(axis=0).max())
    denom    = norm_A * norm_B
    bwd_err  = abs_err / denom if denom > 0 else 0.0
    bound    = K * EPS32
    ok       = bwd_err <= bound
    tag      = 'PASS' if ok else 'FAIL'
    print(f"  {tag}  {name}  "
          f"(bwd_err={bwd_err:.3e}, bound=K*eps32={bound:.3e})")
    if not ok or VERBOSE:
        diff = np.abs(got.astype(np.float64) - C_true_f64)
        idx  = int(diff.argmax())
        print(f"        worst [{idx}]: got={got.ravel()[idx]:.8g}  "
              f"true={C_true_f64.ravel()[idx]:.8g}")
    if ok:
        pass_count += 1
    else:
        fail_count += 1

# ── BPD wrappers ─────────────────────────────────────────────────────────────

def mm_bpd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M, K = A.shape; _, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    _mm(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
    return C

def mm_bias_relu_bpd(A: np.ndarray, B: np.ndarray,
                      bias: np.ndarray) -> np.ndarray:
    M, K = A.shape; _, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    _mm_br(A.ctypes.data, B.ctypes.data, bias.ctypes.data,
           C.ctypes.data, M, N, K)
    return C

def linear_bpd(A: np.ndarray, W: np.ndarray,
               bias: np.ndarray) -> np.ndarray:
    """
    bpd_linear_cpu(input, weight, bias, output, M, N, K)
    weight layout: (N, K)  — same as PyTorch nn.Linear.weight
    computes: output[m,n] = sum_k input[m,k] * weight[n,k] + bias[n]
    """
    M, K = A.shape
    N    = W.shape[0]     # W is (N, K)
    assert W.shape[1] == K, f"weight shape mismatch: {W.shape} vs K={K}"
    C = np.zeros((M, N), dtype=np.float32)
    _linear(A.ctypes.data, W.ctypes.data, bias.ctypes.data,
            C.ctypes.data, M, N, K)
    return C

# ── Test suites ───────────────────────────────────────────────────────────────

def test_trivial_exact():
    """Cases with known exact float32 answers — require BIT_IDENTICAL."""
    print("\n--- Trivial exact cases (BIT_IDENTICAL required) ---")
    rng = np.random.default_rng(0)

    # Identity
    for M in [1, 2, 4, 8, 16, 32]:
        A = rng.standard_normal((M, M)).astype(np.float32)
        I = np.eye(M, dtype=np.float32)
        check_exact(f"A@I=A  ({M}x{M})",  mm_bpd(A, I), A)
        check_exact(f"I@A=A  ({M}x{M})",  mm_bpd(I, A), A)

    # Zero
    for M in [4, 16, 64]:
        A = rng.standard_normal((M, M)).astype(np.float32)
        Z = np.zeros((M, M), dtype=np.float32)
        check_exact(f"A@0=0  ({M}x{M})", mm_bpd(A, Z), Z)

    # All-ones: C[i,j] = M (integer, exact in float32 for M <= 16M)
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        A = np.ones((M, M), dtype=np.float32)
        expected = np.full((M, M), float(M), dtype=np.float32)
        check_exact(f"ones@ones={M}  ({M}x{M})", mm_bpd(A, A), expected)

    # 2x2 known product
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    check_exact("2x2 [[1,2],[3,4]]@[[5,6],[7,8]]",
                mm_bpd(A, B),
                np.array([[19, 22], [43, 50]], dtype=np.float32))

    # Non-square
    A = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
    B = np.array([[7,8],[9,10],[11,12]], dtype=np.float32)
    check_exact("2x3 @ 3x2",
                mm_bpd(A, B),
                np.array([[58, 64],[139, 154]], dtype=np.float32))

    # Dot product
    check_exact("1x4 dot product = 30",
                mm_bpd(np.array([[1,2,3,4]], dtype=np.float32),
                       np.array([[1],[2],[3],[4]], dtype=np.float32)),
                np.array([[30.0]], dtype=np.float32))

    # Outer product
    e0 = np.array([[1,0,0,0]], dtype=np.float32)
    e2 = np.array([[0,0,1,0]], dtype=np.float32)
    expected_outer = np.zeros((4,4), dtype=np.float32)
    expected_outer[0,2] = 1.0
    check_exact("outer product e0 @ e2^T", mm_bpd(e0.T, e2), expected_outer)

    # Diagonal powers-of-2
    D = np.diag(np.array([2,4,8], dtype=np.float32))
    check_exact("diag(2,4,8)^2 = diag(4,16,64)",
                mm_bpd(D, D),
                np.diag(np.array([4,16,64], dtype=np.float32)))

    # mm_bias_relu exact
    M = 8
    A = np.abs(rng.standard_normal((M, M)).astype(np.float32))
    I = np.eye(M, dtype=np.float32)
    bias = np.zeros(M, dtype=np.float32)
    check_exact("relu(A@I + 0) = A  (A>=0)",
                mm_bias_relu_bpd(A, I, bias), A)

    A_neg = -np.abs(rng.standard_normal((M, M)).astype(np.float32))
    bias_neg = -np.ones(M, dtype=np.float32)
    check_exact("relu(neg@I + neg_bias) = 0",
                mm_bias_relu_bpd(A_neg, I, bias_neg),
                np.zeros((M, M), dtype=np.float32))

    # linear exact: A @ I^T + 0 = A
    M = 16
    A = rng.standard_normal((M, M)).astype(np.float32)
    I = np.eye(M, dtype=np.float32)
    bias = np.zeros(M, dtype=np.float32)
    check_exact("linear(A, I, 0) = A",
                linear_bpd(A, I, bias), A)


def test_random_backward():
    """Random matrices: verify Wilkinson backward error bound."""
    print("\n--- Random matrices (Wilkinson backward-error bound) ---")
    rng = np.random.default_rng(42)

    # mm
    for M, N, K in [(64,64,64), (128,128,128), (256,256,256),
                    (512,512,512), (100,200,300), (33,33,33),
                    (7,13,5), (1,1,1000)]:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        C_true = A.astype(np.float64) @ B.astype(np.float64)
        check_backward(f"mm  ({M}x{K} @ {K}x{N})",
                       mm_bpd(A, B), A, B, K, C_true)

    # mm_bias_relu — check the matmul part only (bias/relu are exact)
    for M, K in [(64,64), (256,256), (512,512)]:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, M)).astype(np.float32)
        bias = rng.standard_normal(M).astype(np.float32)
        # ground truth: f64 matmul + bias + relu
        C_true_f64 = A.astype(np.float64) @ B.astype(np.float64)
        C_true_f64 += bias.astype(np.float64)
        C_true_f64 = np.maximum(C_true_f64, 0.0)
        # For the backward error denominator, use ||A||_1 * ||B||_1
        # (bias and relu don't change the matmul error structure)
        check_backward(f"mm_bias_relu  ({M}x{K})",
                       mm_bias_relu_bpd(A, B, bias), A, B, K, C_true_f64)

    # linear  (weight layout: (N, K) — same as nn.Linear)
    for M, K, N in [(4,32,64), (64,64,64), (256,256,256), (512,512,512)]:
        A = rng.standard_normal((M, K)).astype(np.float32)
        W = rng.standard_normal((N, K)).astype(np.float32)
        bias = rng.standard_normal(N).astype(np.float32)
        # ground truth: f64 matmul (A @ W^T) + bias
        C_true_f64 = (A.astype(np.float64) @ W.astype(np.float64).T
                      + bias.astype(np.float64))
        # For the bound, treat as A @ W^T: inner dim K, B = W^T
        W_T = W.T.copy()  # (K, N)
        check_backward(f"linear  ({M}x{K} -> {N})",
                       linear_bpd(A, W, bias), A, W_T, K, C_true_f64)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== BPD CPU Kernel Correctness Tests ===")
    print(f"eps32={EPS32:.4e}  bound = K * eps32 (Wilkinson backward error)\n")

    test_trivial_exact()
    test_random_backward()

    print(f"\n=== Summary: {pass_count} PASSED, {fail_count} FAILED ===")
    sys.exit(0 if fail_count == 0 else 1)
