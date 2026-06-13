#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Stanford KernelBench L1 — Bit-Identical Verification vs PyTorch

Generates kernels from BPD facts, compiles as CPU C code, runs,
and compares output to PyTorch's output on the same input.
NO third-party library calls for computation — we generate everything.

Phase 1: Activations (problems 19-32, 88)
Phase 2: Reductions (problems 47-53)

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-20
Per Heath: bit-identical output, no third-party libraries for computation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import subprocess
import ctypes
import tempfile
import os
import sys

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Stanford L1 shapes (from the KernelBench paper)
BATCH = 16
DIM = 16384  # activation input size


def generate_c_kernel(kernel_name, body_expr, includes_math=True):
    """Generate a standalone C file for a unary elementwise kernel."""
    math_include = '#include <math.h>' if includes_math else ''
    return f"""
#include <string.h>
{math_include}

void {kernel_name}(const float *in, float *out, int n) {{
    for (int i = 0; i < n; i++) {{
        float x = in[i];
        out[i] = {body_expr};
    }}
}}
"""


def compile_and_load(c_source, func_name):
    """Compile C source to .so and load via ctypes."""
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
        f.write(c_source)
        c_path = f.name
    so_path = c_path.replace('.c', '.so')
    
    result = subprocess.run(
        ['gcc', '-O2', '-shared', '-fPIC', '-o', so_path, c_path, '-lm'],
        capture_output=True, text=True, timeout=10
    )
    os.unlink(c_path)
    
    if result.returncode != 0:
        raise RuntimeError(f"Compile failed: {result.stderr}")
    
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, func_name)
    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    func.restype = None
    return func, so_path


def run_kernel(func, x_np):
    """Run a compiled C kernel on numpy input."""
    n = x_np.size
    x_flat = x_np.flatten().astype(np.float32)
    out = np.zeros_like(x_flat)
    func(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )
    return out.reshape(x_np.shape)


def classify(ref, our, label=""):
    """Classify result as BIT_IDENTICAL, WITHIN_BOUND, or FAIL."""
    ref_flat = ref.flatten()
    our_flat = our.flatten()
    
    ref_bits = ref_flat.view(np.uint32)
    our_bits = our_flat.view(np.uint32)
    
    mismatches = np.sum(ref_bits != our_bits)
    
    if mismatches == 0:
        return 'BIT_IDENTICAL', 0, 0
    
    # Compute max ULP
    max_ulp = np.max(np.abs(ref_bits.astype(np.int64) - our_bits.astype(np.int64)))
    max_abs = np.max(np.abs(ref_flat - our_flat))
    
    return 'MISMATCH', int(max_ulp), float(max_abs)


# ═══════════════════════════════════════════════════════════════════════
# L1 Activation Kernels — C body expressions matching PyTorch exactly
# ═══════════════════════════════════════════════════════════════════════

L1_ACTIVATIONS = {
    # Problem 19: ReLU
    19: ('k_relu', 'fmaxf(0.0f, x)', lambda x: F.relu(x)),
    
    # Problem 20: LeakyReLU (default alpha=0.01)
    20: ('k_leaky_relu', '(x > 0.0f) ? x : 0.01f * x', lambda x: F.leaky_relu(x, 0.01)),
    
    # Problem 21: Sigmoid
    21: ('k_sigmoid', '1.0f / (1.0f + expf(-x))', lambda x: torch.sigmoid(x)),
    
    # Problem 22: Tanh
    22: ('k_tanh', 'tanhf(x)', lambda x: torch.tanh(x)),
    
    # Problem 25: Swish/SiLU
    25: ('k_silu', 'x / (1.0f + expf(-x))', lambda x: F.silu(x)),
    
    # Problem 26: GELU — PyTorch DEFAULT is the exact (erf) form, NOT tanh
    26: ('k_gelu', '0.5f * x * (1.0f + erff(x * 0.7071067811865476f))',
         lambda x: F.gelu(x)),
    
    # Problem 27: SELU
    27: ('k_selu', '(x > 0.0f) ? 1.0507009873554805f * x : 1.0507009873554805f * 1.6732632423543772f * (expf(x) - 1.0f)',
         lambda x: F.selu(x)),
    
    # Problem 28: HardSigmoid — PyTorch ATen uses clamp(x+3, 0, 6) / 6
    28: ('k_hardsigmoid', 'fminf(6.0f, fmaxf(0.0f, x + 3.0f)) / 6.0f',
         lambda x: F.hardsigmoid(x)),
    
    # Problem 29: Softplus — PyTorch ATen uses log1pf, not logf(1+x)
    29: ('k_softplus', '(x > 20.0f) ? x : log1pf(expf(x))',
         lambda x: F.softplus(x)),
    
    # Problem 30: Softsign
    30: ('k_softsign', 'x / (1.0f + fabsf(x))', lambda x: F.softsign(x)),
    
    # Problem 31: ELU (alpha=1.0)
    31: ('k_elu', '(x > 0.0f) ? x : 1.0f * (expf(x) - 1.0f)', lambda x: F.elu(x, 1.0)),
    
    # Problem 32: HardTanh (min=-1, max=1)
    32: ('k_hardtanh', 'fminf(1.0f, fmaxf(-1.0f, x))', lambda x: F.hardtanh(x)),
    
    # Problem 88: MinGPT NewGELU — this IS the tanh approximation form
    88: ('k_mingpt_gelu', '0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)))',
         lambda x: F.gelu(x, approximate='tanh')),  # MinGPT specifically uses tanh form
}


def run_activation_suite():
    """Run all activation L1 problems."""
    print("=" * 70)
    print("Stanford KernelBench L1 — Activation Bit-Identical Verification")
    print("=" * 70)
    print(f"Input shape: ({BATCH}, {DIM}), seed={SEED}")
    print()
    
    # Generate input (same for all activations)
    x_np = np.random.randn(BATCH, DIM).astype(np.float32)
    x_torch = torch.from_numpy(x_np.copy())
    
    results = {}
    
    for prob_num in sorted(L1_ACTIVATIONS.keys()):
        name, c_body, torch_fn = L1_ACTIVATIONS[prob_num]
        
        # PyTorch reference
        with torch.no_grad():
            ref = torch_fn(x_torch).numpy()
        
        # Our generated kernel
        c_source = generate_c_kernel(name, c_body)
        try:
            func, so_path = compile_and_load(c_source, name)
            our = run_kernel(func, x_np)
            os.unlink(so_path)
            
            status, max_ulp, max_abs = classify(ref, our, name)
            results[prob_num] = status
            
            symbol = '✅' if status == 'BIT_IDENTICAL' else '❌'
            print(f"  {symbol} L1-{prob_num:3d} {name:25s} {status:20s}"
                  f"  max_ULP={max_ulp:>10d}  max_abs={max_abs:.2e}")
            
        except Exception as e:
            results[prob_num] = 'ERROR'
            print(f"  ❌ L1-{prob_num:3d} {name:25s} ERROR: {e}")
    
    # Summary
    print()
    print("-" * 70)
    bit_identical = sum(1 for v in results.values() if v == 'BIT_IDENTICAL')
    total = len(results)
    print(f"  BIT_IDENTICAL: {bit_identical}/{total}")
    print(f"  MISMATCH:      {sum(1 for v in results.values() if v == 'MISMATCH')}/{total}")
    print(f"  ERROR:         {sum(1 for v in results.values() if v == 'ERROR')}/{total}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# L1 Softmax + LogSoftmax (problems 23, 24) — reduction kernels
# ═══════════════════════════════════════════════════════════════════════

def generate_softmax_c():
    """Generate C softmax kernel matching PyTorch's row-wise softmax."""
    return """
#include <math.h>
#include <float.h>

void k_softmax(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row_in = in + r * cols;
        float *row_out = out + r * cols;
        
        // Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int c = 0; c < cols; c++) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row_out[c] = expf(row_in[c] - max_val);
            sum += row_out[c];
        }
        
        // Normalize
        for (int c = 0; c < cols; c++) {
            row_out[c] /= sum;
        }
    }
}

void k_logsoftmax(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row_in = in + r * cols;
        float *row_out = out + r * cols;
        
        float max_val = -FLT_MAX;
        for (int c = 0; c < cols; c++) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }
        
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += expf(row_in[c] - max_val);
        }
        float log_sum = logf(sum);
        
        for (int c = 0; c < cols; c++) {
            row_out[c] = (row_in[c] - max_val) - log_sum;
        }
    }
}
"""


def run_softmax_suite():
    """Run softmax L1 problems (23, 24)."""
    print()
    print("=" * 70)
    print("Stanford KernelBench L1 — Softmax/LogSoftmax Verification")
    print("=" * 70)
    
    SOFTMAX_DIM = 4096
    x_np = np.random.randn(BATCH, SOFTMAX_DIM).astype(np.float32)
    x_torch = torch.from_numpy(x_np.copy())
    
    c_source = generate_softmax_c()
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
        f.write(c_source)
        c_path = f.name
    so_path = c_path.replace('.c', '.so')
    
    subprocess.run(
        ['gcc', '-O2', '-shared', '-fPIC', '-o', so_path, c_path, '-lm'],
        capture_output=True, check=True, timeout=10
    )
    os.unlink(c_path)
    
    lib = ctypes.CDLL(so_path)
    
    results = {}
    
    for prob_num, name, torch_fn in [
        (23, 'k_softmax', lambda x: F.softmax(x, dim=-1)),
        (24, 'k_logsoftmax', lambda x: F.log_softmax(x, dim=-1)),
    ]:
        func = getattr(lib, name)
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int
        ]
        
        with torch.no_grad():
            ref = torch_fn(x_torch).numpy()
        
        out = np.zeros_like(x_np)
        func(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            BATCH, SOFTMAX_DIM
        )
        
        status, max_ulp, max_abs = classify(ref, out, name)
        results[prob_num] = status
        
        symbol = '✅' if status == 'BIT_IDENTICAL' else '❌'
        print(f"  {symbol} L1-{prob_num:3d} {name:25s} {status:20s}"
              f"  max_ULP={max_ulp:>10d}  max_abs={max_abs:.2e}")
    
    os.unlink(so_path)
    return results


if __name__ == '__main__':
    act_results = run_activation_suite()
    soft_results = run_softmax_suite()
    
    all_results = {**act_results, **soft_results}
    total = len(all_results)
    bit_identical = sum(1 for v in all_results.values() if v == 'BIT_IDENTICAL')
    
    print()
    print("=" * 70)
    print(f"TOTAL: {bit_identical}/{total} BIT_IDENTICAL")
    print("=" * 70)
    
    sys.exit(0 if bit_identical == total else 1)
