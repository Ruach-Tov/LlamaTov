#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_gemm_multi_reference.py — GEMM verification against multiple BLAS references.

Per Heath's direction: don't dismiss hermes-final's 59648 ULP result.
Reproduce it by testing against MULTIPLE OpenBLAS versions on our hardware.

Structure:
  1. System OpenBLAS (what numpy links) — our current reference
  2. Direct cblas_sgemm from libopenblas.so — explicit version
  3. Our scalar K-block GEMM (self-referential, portable)
  4. PyTorch's internal BLAS (MKL on x86, OpenBLAS on others)
  5. TODO: Build older OpenBLAS versions (0.3.26 = Ubuntu 24.04)

If any reference disagrees with another, we surface it — that's
a genuine finding about BLAS version sensitivity, not a test gap.

Usage:
    python3 bench/test_gemm_multi_reference.py [--shapes M,N,K ...]
"""
import ctypes
import numpy as np
import os
import sys
import struct

def ulp_diff(a, b):
    """Per-element ULP difference between two float32 arrays."""
    ai = a.view(np.int32)
    bi = b.view(np.int32)
    return np.abs(ai.astype(np.int64) - bi.astype(np.int64))

def load_bpd_cpu(path=None):
    """Load our bpd_cpu.so with multiple GEMM paths."""
    if path is None:
        path = "/tmp/bpd-generated/build/bpd_cpu.so"
    lib = ctypes.CDLL(path)
    return lib

def load_openblas(path=None):
    """Load OpenBLAS directly for cblas_sgemm."""
    if path is None:
        # Find the system OpenBLAS
        import subprocess
        result = subprocess.run(['find', '/nix/store', '-name', 'libopenblas.so',
                                '-maxdepth', '3'], capture_output=True, text=True, timeout=10)
        paths = result.stdout.strip().split('\n')
        path = paths[0] if paths else None
    if path:
        lib = ctypes.CDLL(path)
        lib.cblas_sgemm.restype = None
        lib.cblas_sgemm.argtypes = [
            ctypes.c_int]*3 + [ctypes.c_int]*3 + [
            ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_float, ctypes.c_void_p, ctypes.c_int
        ]
        return lib, path
    return None, None

def cblas_sgemm_call(blas, A, B):
    """Call cblas_sgemm: C = A @ B."""
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    blas.cblas_sgemm(101, 111, 111,  # CblasRowMajor, NoTrans, NoTrans
                     M, N, K, 1.0,
                     A.ctypes.data, K,
                     B.ctypes.data, N,
                     0.0, C.ctypes.data, N)
    return C

def bpd_mm_call(lib, A, B, env_gemm="1"):
    """Call bpd_mm_cpu with specified dispatch."""
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    os.environ["SUBSTRATE_AVX1_GEMM"] = env_gemm
    lib.bpd_mm_cpu(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
    )
    return C

def bpd_scalar_call(lib, A, B):
    """Call bpd_mm_cpu with scalar K-block (no SIMD, portable reference)."""
    return bpd_mm_call(lib, A, B, env_gemm="0")

def main():
    shapes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),    # hermes-final's failing shape
        (512, 512, 512),
        (256, 256, 384),    # K = Q boundary
        (256, 256, 512),    # K-split boundary
        (67, 89, 113),      # irregular
    ]
    
    bpd = load_bpd_cpu()
    blas, blas_path = load_openblas()
    
    print("=== MULTI-REFERENCE GEMM VERIFICATION ===")
    print(f"  BPD:      bpd_cpu.so (packed panel, AVX1)")
    print(f"  OpenBLAS: {blas_path}")
    print(f"  numpy:    {np.__version__} (uses system BLAS)")
    print()
    
    # Collect references
    refs = {}
    refs['bpd_packed'] = lambda A, B: bpd_mm_call(bpd, A, B, "1")
    refs['bpd_scalar'] = lambda A, B: bpd_scalar_call(bpd, A, B)
    refs['numpy'] = lambda A, B: (A @ B)
    if blas:
        refs['cblas_direct'] = lambda A, B: cblas_sgemm_call(blas, A, B)
    
    try:
        import torch
        refs['torch_mm'] = lambda A, B: torch.mm(
            torch.from_numpy(A), torch.from_numpy(B)).numpy()
        print(f"  PyTorch:  {torch.__version__} ({torch.backends.mkl.is_available() and 'MKL' or 'no MKL'})")
    except ImportError:
        print("  PyTorch:  not available")
    
    print()
    
    # Header
    ref_names = list(refs.keys())
    print(f"{'Shape':>18s}", end="")
    for i, r1 in enumerate(ref_names):
        for r2 in ref_names[i+1:]:
            label = f"{r1[:6]}v{r2[:6]}"
            print(f"  {label:>16s}", end="")
    print()
    print("-" * (18 + 18 * (len(ref_names) * (len(ref_names)-1) // 2)))
    
    # Test each shape
    for M, N, K in shapes:
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        results = {}
        for name, fn in refs.items():
            results[name] = fn(A, B)
        
        print(f"  {M:>4d}x{N:>4d}x{K:>4d}", end="")
        
        for i, r1 in enumerate(ref_names):
            for r2 in ref_names[i+1:]:
                ud = ulp_diff(results[r1], results[r2])
                max_u = ud.max()
                n_mis = (ud > 0).sum()
                if max_u == 0:
                    print(f"  {'0 ULP':>16s}", end="")
                else:
                    print(f"  {n_mis}/{M*N} u{max_u:>5d}", end="")
        print()

if __name__ == '__main__':
    main()
