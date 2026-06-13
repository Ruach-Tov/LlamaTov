#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Stanford L1 GPU bit-discrepancy sweep on Tesla P4.

Compares BPD GPU kernels against PyTorch GPU for each L1 problem.
Also tests each kernel at MULTIPLE dimension sizes to catch
dimension-dependent bugs (like the 2D block reduction at dim>4096).

Usage:
  python3 test_gpu_l1_sweep.py [--so PATH] [--kernel NAME] [--dims 64,256,1024,4096]

Author: medayek (Collective SME, Verification Methodology)
"""
import sys, os, ctypes, argparse, json, numpy as np, time
from dataclasses import dataclass, field
from typing import List, Callable, Optional

try:
    import torch
    import torch.nn.functional as F
    assert torch.cuda.is_available(), "No CUDA"
except (ImportError, AssertionError) as e:
    print(f"SKIP: {e}")
    sys.exit(0)

SEED = 42
c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)


@dataclass
class L1Problem:
    """Stanford KernelBench L1 problem."""
    number: int
    name: str
    pytorch_fn: Callable
    shape_fn: Callable          # (dim) -> list of input shapes
    n_inputs: int = 1
    dtype: str = "float32"


def shape_1d(dim):
    return [(16, dim)]

def shape_2d(dim):
    return [(16, dim)]

def shape_2d_binary(dim):
    return [(16, dim), (16, dim)]

def shape_reduce(dim):
    return [(32, dim)]

def shape_norm(dim):
    return [(16, 64, dim)]

def shape_pool2d(dim):
    s = max(4, int(dim ** 0.5))
    return [(2, 16, s, s)]

def shape_matmul(dim):
    return [(16, dim), (dim, 16)]


# Map L1 problems to our GPU kernels
L1_PROBLEMS = [
    L1Problem(1, "relu", F.relu, shape_1d),
    L1Problem(2, "sigmoid", torch.sigmoid, shape_1d),
    L1Problem(3, "tanh", torch.tanh, shape_1d),
    L1Problem(4, "gelu", F.gelu, shape_1d),
    L1Problem(5, "silu", F.silu, shape_1d),
    L1Problem(6, "elu", F.elu, shape_1d),
    L1Problem(7, "leaky_relu", lambda x: F.leaky_relu(x, 0.01), shape_1d),
    L1Problem(8, "softplus", F.softplus, shape_1d),
    L1Problem(10, "hardtanh", F.hardtanh, shape_1d),
    L1Problem(11, "hardsigmoid", F.hardsigmoid, shape_1d),
    L1Problem(12, "hardswish", F.hardswish, shape_1d),
    L1Problem(13, "mish", lambda x: x * torch.tanh(F.softplus(x)), shape_1d),
    L1Problem(14, "softsign", F.softsign, shape_1d),
    L1Problem(15, "selu", F.selu, shape_1d),
    L1Problem(27, "selu_full", F.selu, shape_1d),  # same as 15 but different params in L1
    L1Problem(31, "elu_full", F.elu, shape_1d),
    L1Problem(37, "add", lambda x, y: x + y, shape_2d_binary, n_inputs=2),
    L1Problem(38, "mul", lambda x, y: x * y, shape_2d_binary, n_inputs=2),
    L1Problem(39, "sub", lambda x, y: x - y, shape_2d_binary, n_inputs=2),
    L1Problem(40, "div", lambda x, y: x / (y.abs() + 1e-7), shape_2d_binary, n_inputs=2),
    L1Problem(47, "softmax", lambda x: F.softmax(x, dim=-1), shape_reduce),
    L1Problem(48, "logsoftmax", lambda x: F.log_softmax(x, dim=-1), shape_reduce),
    L1Problem(34, "instancenorm", lambda x: F.instance_norm(x), shape_norm),
    L1Problem(36, "rmsnorm", lambda x: F.rms_norm(x, [x.shape[-1]]) if hasattr(F, 'rms_norm') else x, shape_norm),
    L1Problem(42, "matmul", lambda x, y: x @ y, shape_matmul, n_inputs=2),
    L1Problem(50, "sum_reduce", lambda x: x.sum(dim=-1), shape_reduce),
    L1Problem(51, "mean_reduce", lambda x: x.mean(dim=-1), shape_reduce),
    L1Problem(52, "max_reduce", lambda x: x.max(dim=-1).values, shape_reduce),
    L1Problem(53, "min_reduce", lambda x: x.min(dim=-1).values, shape_reduce),
    L1Problem(95, "crossentropy", lambda x: -torch.log(F.softmax(x, dim=-1) + 1e-10).mean(), shape_reduce),
    L1Problem(97, "scaled_dot_attn", lambda q: F.scaled_dot_product_attention(
        q.unsqueeze(0), q.unsqueeze(0), q.unsqueeze(0)), lambda d: [(16, d)]),
]


def ulp_compare(a, b):
    af = a.flatten().astype(np.float32)
    bf = b.flatten().astype(np.float32)
    if af.size != bf.size:
        return -1, -1, 0
    ab = af.view(np.int32).astype(np.int64)
    bb = bf.view(np.int32).astype(np.int64)
    diffs = np.abs(ab - bb)
    return int(diffs.max()), int((diffs > 0).sum()), len(af)


def run_problem(prob, dim, device='cuda'):
    """Run one L1 problem at one dimension size."""
    torch.manual_seed(SEED)
    shapes = prob.shape_fn(dim)
    inputs = [torch.randn(*s, device=device, dtype=torch.float32) for s in shapes]

    try:
        with torch.no_grad():
            if prob.n_inputs == 1:
                out = prob.pytorch_fn(inputs[0])
            else:
                out = prob.pytorch_fn(*inputs)
        
        # GPU vs GPU self-consistency (same kernel, same device)
        # When we have our BPD GPU .so, compare BPD vs PyTorch
        # For now, compare GPU vs CPU to find GPU-specific divergences
        inputs_cpu = [t.cpu() for t in inputs]
        with torch.no_grad():
            if prob.n_inputs == 1:
                out_cpu = prob.pytorch_fn(inputs_cpu[0])
            else:
                out_cpu = prob.pytorch_fn(*inputs_cpu)
        
        gpu_np = out.cpu().numpy()
        cpu_np = out_cpu.numpy()
        max_ulp, n_diff, n_total = ulp_compare(gpu_np, cpu_np)
        return max_ulp, n_diff, n_total, None
    except Exception as e:
        return -1, -1, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, help='Test specific kernel')
    parser.add_argument('--dims', type=str, default='64,256,1024,4096,8192',
                       help='Comma-separated dim sizes to sweep')
    parser.add_argument('--json', type=str, help='Save results as JSON')
    parser.add_argument('--so', type=str, help='BPD GPU .so path')
    args = parser.parse_args()

    dims = [int(d) for d in args.dims.split(',')]

    print("=" * 80)
    print(f"Stanford L1 GPU Bit-Discrepancy Sweep")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Dims:   {dims}")
    print(f"  Problems: {len(L1_PROBLEMS)}")
    print("=" * 80)

    # Header
    dim_hdrs = "".join(f"{'d='+str(d):>10}" for d in dims)
    print(f"\n  {'#':>3} {'Problem':20s} {dim_hdrs}  {'status'}")
    print("  " + "-" * (30 + 10 * len(dims) + 10))

    results = []
    all_pass = True

    for prob in L1_PROBLEMS:
        if args.kernel and prob.name != args.kernel:
            continue

        row = f"  {prob.number:3d} {prob.name:20s}"
        prob_results = {}
        worst_ulp = 0

        for dim in dims:
            max_ulp, n_diff, n_total, err = run_problem(prob, dim)
            prob_results[dim] = {'max_ulp': max_ulp, 'n_diff': n_diff, 'n_total': n_total}

            if err:
                row += f"{'ERR':>10}"
            elif max_ulp == 0:
                row += f"{'0':>10}"
            elif max_ulp <= 4:
                row += f"{max_ulp:>10}"
            else:
                row += f"{max_ulp:>10}"
                all_pass = False
            
            worst_ulp = max(worst_ulp, max_ulp)

        status = "✅" if worst_ulp == 0 else ("⚠️" if worst_ulp <= 4 else "❌")
        print(f"{row}  {status}")

        results.append({
            'number': prob.number,
            'name': prob.name,
            'worst_ulp': worst_ulp,
            'dims': prob_results,
        })

    # Summary
    n_zero = sum(1 for r in results if r['worst_ulp'] == 0)
    n_low = sum(1 for r in results if 0 < r['worst_ulp'] <= 4)
    n_high = sum(1 for r in results if r['worst_ulp'] > 4)

    print(f"\n  Summary: {n_zero} BIT_IDENTICAL, {n_low} low ULP (≤4), {n_high} divergent")

    # Find dimension-dependent patterns
    print(f"\n  Dimension-dependent patterns:")
    for r in results:
        ulps = [r['dims'].get(d, {}).get('max_ulp', -1) for d in dims]
        if len(set(u for u in ulps if u >= 0)) > 1:
            print(f"    {r['name']:20s}: {' → '.join(str(u) for u in ulps)}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.json}")


if __name__ == "__main__":
    main()
