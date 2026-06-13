#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify Prolog-generated L2 epilogues against PyTorch unfused sequence."""
import numpy as np, ctypes, sys, os

lib = ctypes.CDLL(os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so"))

# Load the generated epilogues
epi = ctypes.CDLL("/tmp/l2_fused_epilogues.so")

# We need to verify each chain against sequential kernel calls
# Setup substrate kernel argtypes
for name in ['bpd_relu_cpu', 'bpd_silu_cpu', 'bpd_mish_cpu', 'bpd_sigmoid_cpu',
             'bpd_tanh_cpu', 'bpd_gelu_cpu', 'bpd_leaky_relu_cpu', 'bpd_elu_cpu',
             'bpd_selu_cpu', 'bpd_neg_cpu', 'bpd_abs_cpu', 'bpd_hardsigmoid_cpu',
             'bpd_softplus_cpu', 'bpd_hardtanh_cpu']:
    if hasattr(lib, name):
        f = getattr(lib, name)
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        f.restype = None

# Map activation names to kernel functions
kern_map = {
    'relu': 'bpd_relu_cpu', 'silu': 'bpd_silu_cpu', 'mish': 'bpd_mish_cpu',
    'sigmoid': 'bpd_sigmoid_cpu', 'tanh': 'bpd_tanh_cpu', 'gelu': 'bpd_gelu_cpu',
    'leaky_relu': 'bpd_leaky_relu_cpu', 'neg': 'bpd_neg_cpu',
    'hardswish': None, 'hardtanh': 'bpd_hardtanh_cpu',
    'hardsigmoid': 'bpd_hardsigmoid_cpu', 'softplus': 'bpd_softplus_cpu',
}

def run_chain_unfused(ops, x):
    """Run a chain of activations through individual substrate kernels."""
    buf = x.copy()
    for op in ops:
        kname = kern_map.get(op)
        if kname and hasattr(lib, kname):
            out = np.zeros_like(buf)
            getattr(lib, kname)(buf.ctypes.data, out.ctypes.data, ctypes.c_int(buf.size))
            buf = out
        elif op == 'hardswish':
            buf = buf * np.clip(buf + 3.0, 0, 6).astype(np.float32) / np.float32(6.0)
        else:
            print(f"  WARNING: no kernel for {op}")
            return buf
    return buf

np.random.seed(42)
n = 10000
inputs = (np.random.randn(n).astype(np.float32) * 5.0)

chains = [
    (1,  [('relu',)]),
    (4,  [('mish', 'mish')]),
    (5,  [('neg', 'tanh')]),
    (7,  [('relu', 'leaky_relu', 'gelu', 'sigmoid')]),
    (9,  [('neg', 'relu')]),
    (12, [('leaky_relu',)]),
    (16, [('mish', 'hardtanh')]),
    (26, [('hardswish',)]),
    (29, [('mish', 'mish')]),
    (47, [('mish', 'tanh')]),
    (48, [('tanh', 'sigmoid')]),
    (53, [('hardtanh', 'gelu')]),
    (54, [('leaky_relu', 'gelu')]),
    (57, [('relu', 'hardswish')]),
    (59, [('silu',)]),
    (63, [('relu',)]),
    (69, [('hardswish', 'relu')]),
    (71, [('leaky_relu',)]),
    (76, [('relu',)]),
    (81, [('silu', 'tanh')]),
    (86, [('gelu',)]),
    (87, [('mish',)]),
    (90, [('leaky_relu', 'gelu')]),
    (95, [('silu', 'tanh', 'gelu', 'hardtanh')]),
]

print(f"=== L2 Fused Epilogue Verification: {n} test values ===")
print()
print(f"{'#':<5} {'Chain':<40} {'ULP':<10} {'Status'}")
print("-" * 70)

n_pass = 0
n_fail = 0

for num, ops_list in chains:
    ops = ops_list[0]
    fn_name = f"l2_{num}_epilogue"
    
    if not hasattr(epi, fn_name):
        print(f"{num:<5} {'+'.join(ops):<40} {'N/A':<10} not in .so")
        continue
    
    fn = getattr(epi, fn_name)
    fn.argtypes = [ctypes.c_float]
    fn.restype = ctypes.c_float
    
    # Run unfused chain through substrate kernels
    unfused = run_chain_unfused(ops, inputs)
    
    # Run fused epilogue
    fused = np.array([fn(ctypes.c_float(x)) for x in inputs], dtype=np.float32)
    
    # Compare ULP
    u_bits = unfused.view(np.int32).astype(np.int64)
    f_bits = fused.view(np.int32).astype(np.int64)
    diffs = np.abs(u_bits - f_bits)
    max_ulp = int(diffs.max())
    n_diffs = int((diffs > 0).sum())
    
    status = "BIT_IDENTICAL" if max_ulp == 0 else f"DIVERGENT ({n_diffs} diffs)"
    if max_ulp == 0:
        n_pass += 1
    else:
        n_fail += 1
    
    chain_str = '+'.join(ops)
    print(f"{num:<5} {chain_str:<40} {max_ulp:<10} {status}")

print()
print(f"PASS: {n_pass}  FAIL: {n_fail}  TOTAL: {n_pass + n_fail}")
