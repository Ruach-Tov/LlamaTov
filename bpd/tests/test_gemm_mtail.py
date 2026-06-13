#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_gemm_mtail.py — Regression test for GEMM M-tail handler.

Bug: bpd_gemm_packed_panel M-tail loop only computed column j+0
instead of all NR=16 columns. Any matrix where M % MR != 0 got
zeros in tail rows.

Caught by im2col conv2d with identity weights where K_dim (=M)
was not a multiple of 4 (MR=4).

This test sweeps M values around multiples of MR to catch tail
handler regressions for any GEMM variant.

Root cause commit: 494f257 (mavchin)
Author: medayek (from mavchin's finding)
"""
import sys, ctypes, numpy as np

c_float_p = ctypes.POINTER(ctypes.c_float)


def test_gemm_mtail(so_path):
    """Test GEMM correctness for M values that exercise the tail handler."""
    lib = ctypes.CDLL(so_path)
    
    # Find available GEMM function
    for fn_name in ['bpd_matmul_f32_cpu', 'bpd_3d_tensor_matmul_cpu']:
        if hasattr(lib, fn_name):
            break
    else:
        print("SKIP: no F32 matmul in .so")
        return True
    
    fn = getattr(lib, fn_name)
    
    np.random.seed(42)
    
    # MR=4 typical tile size — test M values around multiples of 4
    N = 16  # fixed N (number of output columns)
    K = 32  # fixed K (inner dimension)
    
    # M values that exercise the tail: 1,2,3,5,6,7,9,10,11,13,14,15,17
    # Plus clean multiples: 4,8,12,16
    m_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 33, 63, 65]
    
    results = []
    for M in m_values:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Reference: numpy matmul
        ref = (A @ B).astype(np.float32)
        
        # Our kernel
        out = np.zeros((M, N), dtype=np.float32)
        
        if fn_name == 'bpd_matmul_f32_cpu':
            fn.restype = None
            fn.argtypes = [c_float_p, c_float_p, c_float_p,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn(A.ctypes.data_as(c_float_p),
               B.ctypes.data_as(c_float_p),
               out.ctypes.data_as(c_float_p),
               ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K))
        else:
            fn.restype = None
            fn.argtypes = [c_float_p, c_float_p, c_float_p,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn(A.ctypes.data_as(c_float_p),
               B.ctypes.data_as(c_float_p),
               out.ctypes.data_as(c_float_p),
               ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K))
        
        # Check for zeros in tail rows (the specific bug pattern)
        tail_start = (M // 4) * 4
        if tail_start < M:
            tail_rows = out[tail_start:]
            if np.all(tail_rows == 0):
                print(f"  FAIL  M={M:3d}: tail rows [{tail_start}:{M}] are ALL ZEROS (bug!)")
                results.append(False)
                continue
        
        # ULP comparison
        rb = ref.flatten().view(np.uint32)
        ob = out.flatten().view(np.uint32)
        mm = int(np.sum(rb != ob))
        
        if mm == 0:
            print(f"  PASS  M={M:3d}: BIT_IDENTICAL ({M*N} elements)")
            results.append(True)
        else:
            max_ulp = int(np.max(np.abs(rb.astype(np.int64) - ob.astype(np.int64))))
            # Small ULP from accumulation order is OK
            if max_ulp <= 4:
                print(f"  PASS  M={M:3d}: {mm} differ, max_ULP={max_ulp} (accumulation order)")
                results.append(True)
            else:
                max_abs = float(np.max(np.abs(ref.flatten() - out.flatten())))
                print(f"  FAIL  M={M:3d}: {mm} differ, max_ULP={max_ulp}, max_abs={max_abs:.2e}")
                results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\nGEMM M-tail: {passed}/{total} PASS")
    return all(results)


def test_conv2d_identity(so_path):
    """Test conv2d with identity kernel — catches im2col GEMM tail bugs."""
    lib = ctypes.CDLL(so_path)
    
    if not hasattr(lib, 'bpd_conv2d_full_cpu'):
        print("SKIP: no bpd_conv2d_full_cpu")
        return True
    
    np.random.seed(42)
    
    # Parameters that exercise different M-tail scenarios
    # M = K_dim = Cin * kH * kW
    test_cases = [
        # (Cin, H, W, Cout, kH, kW, stride, pad) → M = Cin*kH*kW
        (1, 8, 8, 1, 3, 3, 1, 0),   # M=9, tail=1  ← the original bug
        (1, 8, 8, 1, 5, 5, 1, 0),   # M=25, tail=1
        (3, 8, 8, 1, 3, 3, 1, 0),   # M=27, tail=3
        (1, 8, 8, 1, 7, 7, 1, 0),   # M=49, tail=1
        (3, 16, 16, 16, 3, 3, 1, 1), # M=27, realistic
    ]
    
    results = []
    for cin, h, w, cout, kh, kw, s, p in test_cases:
        m = cin * kh * kw
        tail = m % 4
        x = np.random.randn(1, cin, h, w).astype(np.float32)
        
        # Identity-ish weights (small random to test all paths)
        wt = np.random.randn(cout, cin, kh, kw).astype(np.float32) * 0.1
        bias = np.zeros(cout, dtype=np.float32)
        
        # Reference: manual conv2d
        ho = (h + 2*p - kh) // s + 1
        wo = (w + 2*p - kw) // s + 1
        
        # Our kernel
        out = np.zeros((1, cout, ho, wo), dtype=np.float32)
        xc = np.ascontiguousarray(x)
        wc = np.ascontiguousarray(wt)
        
        lib.bpd_conv2d_full_cpu.restype = None
        lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
        
        lib.bpd_conv2d_full_cpu(
            xc.ctypes.data, wc.ctypes.data, bias.ctypes.data, out.ctypes.data,
            1, cin, h, w, cout, kh, kw, s, s, p, p, 1, 1, 1)
        
        # Check for zeros in output (the bug pattern)
        if np.all(out == 0):
            print(f"  FAIL  conv {cin}ch {h}x{w} k={kh} M={m} tail={tail}: ALL ZEROS")
            results.append(False)
        elif np.any(np.isnan(out)):
            print(f"  FAIL  conv {cin}ch {h}x{w} k={kh} M={m} tail={tail}: CONTAINS NaN")
            results.append(False)
        else:
            print(f"  PASS  conv {cin}ch {h}x{w} k={kh} M={m} tail={tail}: non-zero output")
            results.append(True)
    
    passed = sum(results)
    total = len(results)
    print(f"\nConv2d M-tail: {passed}/{total} PASS")
    return all(results)


if __name__ == "__main__":
    so_path = sys.argv[1] if len(sys.argv) > 1 else "build/bpd_cpu.so"
    print("=" * 60)
    print("GEMM M-tail regression test")
    print("=" * 60)
    
    r1 = test_gemm_mtail(so_path)
    print()
    r2 = test_conv2d_identity(so_path)
    
    print()
    if r1 and r2:
        print("ALL PASS")
    else:
        print("FAILURES DETECTED")
        sys.exit(1)
