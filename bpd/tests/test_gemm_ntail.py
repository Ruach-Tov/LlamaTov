#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_gemm_ntail.py — Regression test for GEMM N-tail handler.

Same bug class as M-tail (commit 494f257) but for the N dimension.
If N % NR != 0, the N-tail loop must process ALL remaining columns.

Tests GEMM with N values around NR=16 boundaries.

Author: medayek
"""
import sys, ctypes, numpy as np

c_float_p = ctypes.POINTER(ctypes.c_float)


def main():
    so_path = sys.argv[1] if len(sys.argv) > 1 else "build/bpd_cpu.so"
    lib = ctypes.CDLL(so_path)
    
    for fn_name in ['bpd_matmul_f32_cpu', 'bpd_3d_tensor_matmul_cpu']:
        if hasattr(lib, fn_name):
            fn = getattr(lib, fn_name)
            fn.restype = None
            fn.argtypes = [c_float_p, c_float_p, c_float_p,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int]
            break
    else:
        print("SKIP: no F32 matmul in .so")
        return
    
    np.random.seed(42)
    M = 16  # fixed
    K = 32  # fixed
    
    # NR=16 typical — test N values around multiples of 16
    n_values = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65]
    
    print(f"{'N':>4}  {'tail':>4}  {'diffs':>8}  {'max_ULP':>8}  {'status'}")
    print("-" * 45)
    
    all_pass = True
    for N in n_values:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        ref = (A @ B).astype(np.float32)
        out = np.zeros((M, N), dtype=np.float32)
        
        fn(A.ctypes.data_as(c_float_p), B.ctypes.data_as(c_float_p),
           out.ctypes.data_as(c_float_p),
           ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K))
        
        # Check for zeros in tail columns
        tail_start = (N // 16) * 16
        tail = N % 16
        if tail > 0 and np.all(out[:, tail_start:] == 0):
            print(f"{N:4d}  {tail:4d}  {'--':>8}  {'--':>8}  FAIL (tail cols ALL ZERO)")
            all_pass = False
            continue
        
        rb = ref.flatten().view(np.uint32)
        ob = out.flatten().view(np.uint32)
        mm = int(np.sum(rb != ob))
        max_ulp = int(np.max(np.abs(rb.astype(np.int64) - ob.astype(np.int64)))) if mm > 0 else 0
        
        status = "PASS" if max_ulp <= 4 else "FAIL"
        if max_ulp > 4:
            all_pass = False
        print(f"{N:4d}  {tail:4d}  {mm:8d}  {max_ulp:8d}  {status}")
    
    print()
    print(f"GEMM N-tail: {'ALL PASS' if all_pass else 'FAILURES'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
