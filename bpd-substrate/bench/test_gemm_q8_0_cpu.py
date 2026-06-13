# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import ctypes
import numpy as np
import os

# Load the compiled shared library
lib = ctypes.CDLL('./build/bpd_cpu.so')

# We'll use the existing bpd_qmatmul_q8_0_cpu as our reference, 
# as it's known to match ggml's AVX1 fallback path (the one used for single rows)
lib.bpd_qmatmul_q8_0_cpu.argtypes = [
    ctypes.c_void_p,  # W_q8_0
    ctypes.c_void_p,  # X_f32
    ctypes.c_void_p,  # out
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int      # K
]

# The new dispatcher we want to test
lib.bpd_qmatmul_q8_0_dispatch_cpu.argtypes = [
    ctypes.c_void_p,  # W_q8_0
    ctypes.c_void_p,  # X_q8_0
    ctypes.c_void_p,  # out
    ctypes.c_int,     # m_weight
    ctypes.c_int,     # m_tokens
    ctypes.c_int      # K
]

# We need the quantize function to prepare X_q8_0 for the dispatcher
lib.bpd_quant_q8_0_cpu.argtypes = [
    ctypes.c_void_p,  # x
    ctypes.c_void_p,  # y
    ctypes.c_int      # n_elements
]

def test_tile(RM, RN, K=128):
    print(f"Testing tile RM={RM}, RN={RN}, K={K}...")
    
    # Q8_0 block size is 32 elements, 34 bytes
    k_blocks = K // 32
    bytes_per_row = k_blocks * 34
    
    # Generate random weight data (F32) and quantize to Q8_0
    np.random.seed(42 + RM * 10 + RN)
    W_f32 = np.random.randn(RM, K).astype(np.float32)
    W_q8_0 = np.zeros(RM * bytes_per_row, dtype=np.uint8)
    
    for i in range(RM):
        row_f32 = W_f32[i:i+1].copy()
        row_q8_0 = W_q8_0[i*bytes_per_row:(i+1)*bytes_per_row].copy()
        lib.bpd_quant_q8_0_cpu(
            row_f32.ctypes.data,
            row_q8_0.ctypes.data,
            K
        )
        W_q8_0[i*bytes_per_row:(i+1)*bytes_per_row] = row_q8_0
        
    # Generate random activation data (F32)
    X_f32 = np.random.randn(RN, K).astype(np.float32)
    
    # Run reference implementation (bpd_qmatmul_q8_0_cpu)
    # Note: it takes W as (M, K) and X as (N, K), outputs (M, N)
    # Wait, bpd_qmatmul_q8_0_cpu signature: (W, X, out, M, N, K)
    # where out[m, n] = sum_k X[m, k] * W[n, k]
    # M = rows of X (RN), N = rows of W (RM)
    out_ref = np.zeros((RN, RM), dtype=np.float32)
    lib.bpd_qmatmul_q8_0_cpu(
        W_q8_0.ctypes.data,
        X_f32.ctypes.data,
        out_ref.ctypes.data,
        RN, RM, K
    )
    
    # Prepare X_q8_0 for the dispatcher
    X_q8_0 = np.zeros(RN * bytes_per_row, dtype=np.uint8)
    for i in range(RN):
        row_f32 = X_f32[i:i+1].copy()
        row_q8_0 = X_q8_0[i*bytes_per_row:(i+1)*bytes_per_row].copy()
        lib.bpd_quant_q8_0_cpu(
            row_f32.ctypes.data,
            row_q8_0.ctypes.data,
            K
        )
        X_q8_0[i*bytes_per_row:(i+1)*bytes_per_row] = row_q8_0
        
    # Run new dispatcher
    # Signature: (W_q8, X_q8, out, m_weight, m_tokens, K)
    out_new = np.zeros((RN, RM), dtype=np.float32)
    lib.bpd_qmatmul_q8_0_dispatch_cpu(
        W_q8_0.ctypes.data,
        X_q8_0.ctypes.data,
        out_new.ctypes.data,
        RM, RN, K
    )
    
    # Compare
    diff = np.abs(out_ref - out_new)
    max_diff = np.max(diff)
    
    if max_diff == 0.0:
        print(f"  ✅ PASS: 0 ULP difference")
        return True
    else:
        print(f"  ❌ FAIL: Max difference = {max_diff}")
        print(f"  Reference:\n{out_ref}")
        print(f"  New:\n{out_new}")
        return False

def main():
    print("Testing 9 tiled kernels...")
    all_pass = True
    for RM in [1, 2, 4]:
        for RN in [1, 2, 4]:
            if not test_tile(RM, RN, K=128):
                all_pass = False
                
    if all_pass:
        print("\n🎉 All 9 tile kernels pass at 0 ULP vs reference!")
    else:
        print("\n💥 Some tile kernels failed.")
        
if __name__ == "__main__":
    main()
