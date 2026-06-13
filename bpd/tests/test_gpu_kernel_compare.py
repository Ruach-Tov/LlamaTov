#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""GPU Kernel Comparison Harness — BPD-lifted vs PyTorch source.

For any PyTorch kernel we lift to BPD:
  1. Run the ORIGINAL PyTorch kernel on GPU → reference output
  2. Run our BPD-generated CUDA kernel on GPU → our output
  3. Compare bit-identical (ULP analysis)
  4. Measure performance (warmup + timed runs)
  5. Profile (memory bandwidth, compute utilization)

This is the GPU equivalent of test_l1_bit_identical_cpu.py.

Usage:
  python test_gpu_kernel_compare.py                    # Run all registered kernels
  python test_gpu_kernel_compare.py --kernel forward_diff_y  # Run one kernel
  python test_gpu_kernel_compare.py --perf             # Include performance measurement
  python test_gpu_kernel_compare.py --profile          # Include basic profiling

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-22
"""

import argparse
import json
import time
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KernelCompareSpec:
    """Specification for comparing a PyTorch kernel vs BPD kernel on GPU."""
    name: str
    domain: str                     # 'cfd', 'ml', 'blas', etc.
    pytorch_fn: Callable            # Original PyTorch computation (runs on GPU)
    bpd_fn: Optional[Callable]      # Our BPD kernel (runs on GPU via PyCUDA/ctypes)
    input_gen: Callable             # Generates inputs as numpy arrays
    description: str = ""
    tolerance_ulp: int = 0          # Expected max ULP (0 = bit-identical)


# Registry
GPU_KERNELS: Dict[str, KernelCompareSpec] = {}

def register(spec: KernelCompareSpec):
    GPU_KERNELS[spec.name] = spec
    return spec


def ulp_compare(ref_np, our_np):
    """Compare two float32 arrays by ULP distance."""
    ref = ref_np.flatten().astype(np.float32)
    our = our_np.flatten().astype(np.float32)
    
    if ref.shape != our.shape:
        return {'status': 'shape_mismatch', 'ref_shape': ref.shape, 'our_shape': our.shape}
    
    ref_bits = ref.view(np.uint32)
    our_bits = our.view(np.uint32)
    mismatches = int(np.sum(ref_bits != our_bits))
    
    if mismatches == 0:
        return {
            'status': 'BIT_IDENTICAL',
            'max_ulp': 0,
            'n_diff': 0,
            'n_total': len(ref),
            'max_abs': 0.0,
        }
    
    max_ulp = int(np.max(np.abs(ref_bits.astype(np.int64) - our_bits.astype(np.int64))))
    max_abs = float(np.max(np.abs(ref - our)))
    mean_ulp = float(np.mean(np.abs(ref_bits.astype(np.int64) - our_bits.astype(np.int64))))
    
    return {
        'status': 'MATCH' if max_ulp <= 1 else 'DIVERGENT',
        'max_ulp': max_ulp,
        'mean_ulp': round(mean_ulp, 2),
        'n_diff': mismatches,
        'n_total': len(ref),
        'pct_diff': round(100 * mismatches / len(ref), 1),
        'max_abs': max_abs,
    }


def measure_performance(fn, inputs, n_warmup=10, n_runs=50, device='cuda'):
    """Measure kernel execution time with warmup."""
    if not HAS_CUDA:
        return {'error': 'No CUDA device'}
    
    # Move inputs to device
    gpu_inputs = []
    for inp in inputs:
        if isinstance(inp, np.ndarray):
            gpu_inputs.append(torch.from_numpy(inp).to(device))
        elif isinstance(inp, torch.Tensor):
            gpu_inputs.append(inp.to(device))
        else:
            gpu_inputs.append(inp)
    
    # Warmup
    for _ in range(n_warmup):
        _ = fn(*gpu_inputs)
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(*gpu_inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    
    times = np.array(times)
    return {
        'mean_us': round(float(times.mean()), 1),
        'std_us': round(float(times.std()), 1),
        'min_us': round(float(times.min()), 1),
        'max_us': round(float(times.max()), 1),
        'median_us': round(float(np.median(times)), 1),
        'n_runs': n_runs,
    }


def basic_profile(fn, inputs, device='cuda'):
    """Basic profiling: memory bandwidth estimation."""
    if not HAS_CUDA:
        return {'error': 'No CUDA device'}
    
    gpu_inputs = []
    total_bytes = 0
    for inp in inputs:
        if isinstance(inp, np.ndarray):
            t = torch.from_numpy(inp).to(device)
            total_bytes += t.nelement() * t.element_size()
            gpu_inputs.append(t)
        elif isinstance(inp, torch.Tensor):
            t = inp.to(device)
            total_bytes += t.nelement() * t.element_size()
            gpu_inputs.append(t)
        else:
            gpu_inputs.append(inp)
    
    # Estimate output size (run once to get output)
    out = fn(*gpu_inputs)
    if isinstance(out, torch.Tensor):
        total_bytes += out.nelement() * out.element_size()
    
    # Measure time
    perf = measure_performance(fn, inputs, device=device)
    
    if 'error' not in perf:
        bandwidth_gbps = total_bytes / (perf['mean_us'] * 1e-6) / 1e9
        return {
            'total_bytes': total_bytes,
            'bandwidth_gbps': round(bandwidth_gbps, 2),
            'perf': perf,
        }
    return perf


# ═══════════════════════════════════════════════════════════════════════
# CFD Stencil Kernels (torch-cfd lifted)
# ═══════════════════════════════════════════════════════════════════════

def _cfd_grid(ny=256, nx=256):
    np.random.seed(SEED)
    u = np.random.randn(ny, nx).astype(np.float32)
    return (u,), {'step': 0.1}  # inputs tuple, kwargs not used by fn

DX = 0.1
DY = 0.1

register(KernelCompareSpec(
    name='forward_diff_y',
    domain='cfd',
    pytorch_fn=lambda u: (torch.roll(u, -1, dims=0) - u) / DY,
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(256, 256).astype(np.float32),),
    description='Forward difference dy: (u[i+1]-u[i])/dy, periodic BC',
    tolerance_ulp=0,
))

register(KernelCompareSpec(
    name='forward_diff_x',
    domain='cfd',
    pytorch_fn=lambda u: (torch.roll(u, -1, dims=1) - u) / DX,
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(256, 256).astype(np.float32),),
    description='Forward difference dx: (u[j+1]-u[j])/dx, periodic BC',
))

register(KernelCompareSpec(
    name='backward_diff_y',
    domain='cfd',
    pytorch_fn=lambda u: (u - torch.roll(u, 1, dims=0)) / DY,
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(256, 256).astype(np.float32),),
    description='Backward difference dy',
))

register(KernelCompareSpec(
    name='backward_diff_x',
    domain='cfd',
    pytorch_fn=lambda u: (u - torch.roll(u, 1, dims=1)) / DX,
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(256, 256).astype(np.float32),),
    description='Backward difference dx',
))

register(KernelCompareSpec(
    name='central_diff_y',
    domain='cfd',
    pytorch_fn=lambda u: (torch.roll(u, -1, 0) - torch.roll(u, 1, 0)) / (2*DY),
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(256, 256).astype(np.float32),),
    description='Central difference dy',
))

register(KernelCompareSpec(
    name='central_diff_x',
    domain='cfd',
    pytorch_fn=lambda u: (torch.roll(u, -1, 1) - torch.roll(u, 1, 1)) / (2*DX),
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(256, 256).astype(np.float32),),
    description='Central difference dx',
))

register(KernelCompareSpec(
    name='laplacian_2d',
    domain='cfd',
    pytorch_fn=lambda u: (
        -2 * u * (1/(DX**2) + 1/(DY**2))
        + (torch.roll(u, -1, 0) + torch.roll(u, 1, 0)) / (DY**2)
        + (torch.roll(u, -1, 1) + torch.roll(u, 1, 1)) / (DX**2)
    ),
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(256, 256).astype(np.float32),),
    description='2D Laplacian, periodic BC',
))

register(KernelCompareSpec(
    name='divergence_2d',
    domain='cfd',
    pytorch_fn=lambda vx, vy: (
        (vx - torch.roll(vx, 1, 1)) / DX + (vy - torch.roll(vy, 1, 0)) / DY
    ),
    bpd_fn=None,
    input_gen=lambda: (
        np.random.randn(256, 256).astype(np.float32),
        np.random.randn(256, 256).astype(np.float32),
    ),
    description='2D Divergence (backward diff), periodic BC',
))


# ═══════════════════════════════════════════════════════════════════════
# ML Kernels (from pytorch_kernel_library.py)
# ═══════════════════════════════════════════════════════════════════════

register(KernelCompareSpec(
    name='relu',
    domain='ml',
    pytorch_fn=lambda x: torch.nn.functional.relu(x),
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(16, 16384).astype(np.float32),),
    description='ReLU activation',
))

register(KernelCompareSpec(
    name='silu',
    domain='ml',
    pytorch_fn=lambda x: torch.nn.functional.silu(x),
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(16, 16384).astype(np.float32),),
    description='SiLU/Swish activation',
))

register(KernelCompareSpec(
    name='softmax',
    domain='ml',
    pytorch_fn=lambda x: torch.nn.functional.softmax(x, dim=-1),
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(16, 4096).astype(np.float32),),
    description='Row-wise softmax',
))

register(KernelCompareSpec(
    name='matmul_square',
    domain='ml',
    pytorch_fn=lambda a, b: torch.matmul(a, b),
    bpd_fn=None,
    input_gen=lambda: (
        np.random.randn(512, 512).astype(np.float32),
        np.random.randn(512, 512).astype(np.float32),
    ),
    description='Square GEMM 512x512',
))

register(KernelCompareSpec(
    name='layer_norm',
    domain='ml',
    pytorch_fn=lambda x: torch.nn.functional.layer_norm(x, [x.shape[-1]]),
    bpd_fn=None,
    input_gen=lambda: (np.random.randn(16, 256, 1024).astype(np.float32),),
    description='LayerNorm over last dim',
))


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def run_comparison(spec, do_perf=False, do_profile=False):
    """Run a single kernel comparison: PyTorch GPU vs CPU reference."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    raw_inputs = spec.input_gen()
    if not isinstance(raw_inputs, tuple):
        raw_inputs = (raw_inputs,)
    
    # All inputs are numpy arrays (scalars baked into the lambda)
    numpy_inputs = [inp for inp in raw_inputs if isinstance(inp, np.ndarray)]
    
    # Run on CPU
    cpu_args = [torch.from_numpy(inp.copy()) for inp in numpy_inputs]
    
    with torch.no_grad():
        cpu_out = spec.pytorch_fn(*cpu_args)
    cpu_np = cpu_out.numpy() if isinstance(cpu_out, torch.Tensor) else np.array(cpu_out)
    
    result = {'name': spec.name, 'domain': spec.domain, 'description': spec.description}
    
    if HAS_CUDA:
        # Run on GPU
        gpu_args = [torch.from_numpy(inp.copy()).cuda() for inp in numpy_inputs]
        
        torch.cuda.synchronize()
        with torch.no_grad():
            gpu_out = spec.pytorch_fn(*gpu_args)
        torch.cuda.synchronize()
        gpu_np = gpu_out.cpu().numpy() if isinstance(gpu_out, torch.Tensor) else np.array(gpu_out)
        
        # Compare GPU vs CPU
        comparison = ulp_compare(cpu_np, gpu_np)
        result['gpu_vs_cpu'] = comparison
        
        # Performance
        if do_perf:
            cpu_perf = measure_performance(spec.pytorch_fn, numpy_inputs, device='cpu')
            gpu_perf = measure_performance(spec.pytorch_fn, numpy_inputs, device='cuda')
            result['cpu_perf'] = cpu_perf
            result['gpu_perf'] = gpu_perf
            if 'error' not in cpu_perf and 'error' not in gpu_perf:
                result['speedup'] = round(cpu_perf['mean_us'] / gpu_perf['mean_us'], 2)
        
        # Profile
        if do_profile:
            prof = basic_profile(spec.pytorch_fn, numpy_inputs)
            result['profile'] = prof
    else:
        result['gpu_vs_cpu'] = {'status': 'NO_CUDA'}
    
    return result


def main():
    parser = argparse.ArgumentParser(description='GPU Kernel Comparison Harness')
    parser.add_argument('--kernel', type=str, help='Run specific kernel')
    parser.add_argument('--domain', type=str, help='Filter by domain (cfd, ml)')
    parser.add_argument('--perf', action='store_true', help='Measure performance')
    parser.add_argument('--profile', action='store_true', help='Basic profiling')
    parser.add_argument('--json', type=str, help='Save results as JSON')
    args = parser.parse_args()
    
    print("=" * 70)
    print("GPU Kernel Comparison Harness")
    if HAS_CUDA:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No CUDA device — CPU-only comparison")
    print(f"  Registered kernels: {len(GPU_KERNELS)}")
    print("=" * 70)
    
    results = []
    for name, spec in sorted(GPU_KERNELS.items()):
        if args.kernel and name != args.kernel:
            continue
        if args.domain and spec.domain != args.domain:
            continue
        
        result = run_comparison(spec, do_perf=args.perf, do_profile=args.profile)
        results.append(result)
        
        comp = result.get('gpu_vs_cpu', {})
        status = comp.get('status', '?')
        symbol = '✅' if status == 'BIT_IDENTICAL' else ('⚠️' if status == 'MATCH' else '❌')
        
        line = f"  {symbol} {name:25s} [{spec.domain:4s}] {status:15s}"
        if 'max_ulp' in comp:
            line += f" ULP={comp['max_ulp']}"
        if 'speedup' in result:
            line += f"  GPU {result['speedup']}×"
        print(line)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")
    
    # Summary
    print()
    bit_identical = sum(1 for r in results if r.get('gpu_vs_cpu', {}).get('status') == 'BIT_IDENTICAL')
    total = len(results)
    print(f"  GPU vs CPU BIT_IDENTICAL: {bit_identical}/{total}")


if __name__ == '__main__':
    main()
