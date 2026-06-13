#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""bit_identical_v1.py — Minimum-viable KernelBench L1 verification sweep.

Architecture:
  - test_recipe(kernel_case, shape_spec, ref_fn, kernel_args_fn, alloc_outputs_fn)
  - For each recipe: gen inputs (fixed seed), compute ref, compile+run substrate,
    compare element-wise (allclose + bit-identical + stub detection).

Stub detection: substrate output is all-zero (or nearly so) while reference is
non-trivially non-zero. This catches the conv stubs without needing PyTorch
to dispatch a conv reference for them (since we know they'll fail; the goal
is to *empirically prove* the failure for the substrate-historical record).

Per Heath's plan 8d65ba1c subtask 2-inf-g-1. Starts with 6 reduction cases;
generalizes from there.
"""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

VALIDATION_DIR = Path("/tmp/l1_cuda_validation")
BUILD_DIR = Path("/tmp/tier2_build")
BUILD_DIR.mkdir(exist_ok=True)

SEED = 42
NVIDIA_LIB = "/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib"


def compile_kernel(case_name, kernel_name, params_decl, host_dispatch_body):
    """Compose a .cu with the substrate kernel + extern "C" host dispatch,
    then nvcc-compile to a .so. Returns path to .so or raises."""
    src_path = VALIDATION_DIR / f"{case_name}.cu"
    if not src_path.exists():
        raise FileNotFoundError(f"Substrate emit not found: {src_path}")

    out_so = BUILD_DIR / f"{case_name}.so"
    composed = BUILD_DIR / f"{case_name}_dispatch.cu"

    with open(src_path) as f:
        kernel_src = f.read()

    dispatch_src = f"""
{kernel_src}

extern "C" {{
    int run_{case_name}({params_decl}) {{
{host_dispatch_body}
    }}
}}
"""
    composed.write_text(dispatch_src)

    # nvcc compile to .so
    cmd = ["nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_61",
           str(composed), "-o", str(out_so)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed for {case_name}: {result.stderr[:500]}")
    return out_so


def reduction_dispatch_body(kernel_name):
    """Standard 2-input/1-output dispatch for reductions: X[outer,N] -> Y[outer]."""
    return f"""
        float *d_X = nullptr, *d_Y = nullptr;
        size_t in_bytes = (size_t)outer * (size_t)N * sizeof(float);
        size_t out_bytes = (size_t)outer * sizeof(float);
        if (cudaMalloc(&d_X, in_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, out_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        if (cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {{
            cudaFree(d_X); cudaFree(d_Y); return 3;
        }}
        int block = 256;
        int grid = (outer + block - 1) / block;
        {kernel_name}<<<grid, block>>>(d_X, d_Y, N, outer);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        if (cudaMemcpy(h_Y, d_Y, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {{
            cudaFree(d_X); cudaFree(d_Y); return 6;
        }}
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def run_reduction_case(case_name, kernel_name, ref_fn, outer=64, N=256):
    """Run one reduction case end-to-end."""
    torch.manual_seed(SEED)
    x = torch.randn(outer, N, dtype=torch.float32)
    x_flat = x.reshape(-1).contiguous().numpy()

    # Compute reference
    y_ref = ref_fn(x).numpy()

    # Compile + load substrate kernel
    so_path = compile_kernel(
        case_name, kernel_name,
        "const float *h_X, float *h_Y, int N, int outer",
        reduction_dispatch_body(kernel_name)
    )

    lib = ctypes.CDLL(str(so_path))
    fn = getattr(lib, f"run_{case_name}")
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int

    y_act = np.zeros(outer, dtype=np.float32)
    rc = fn(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N), ctypes.c_int(outer)
    )
    return classify_result(case_name, kernel_name, y_ref, y_act, rc)


def classify_result(case_name, kernel_name, y_ref, y_act, rc):
    """Substrate-design classification of the comparison."""
    if rc != 0:
        return {"case": case_name, "kernel": kernel_name, "status": "CUDA_ERROR",
                "rc": rc, "detail": f"runtime rc={rc}"}

    # Stub detection: substrate output is all-zero while ref is not
    act_all_zero = bool(np.all(y_act == 0.0))
    ref_significant = bool(np.any(np.abs(y_ref) > 1e-6))
    if act_all_zero and ref_significant:
        return {"case": case_name, "kernel": kernel_name, "status": "STUB_DETECTED",
                "rc": 0,
                "detail": f"output all-zero; ref max |x|={np.abs(y_ref).max():.4f}"}

    # ULP comparison
    ref_bits = y_ref.view(np.uint32)
    act_bits = y_act.view(np.uint32)
    ulp_diff = np.abs(ref_bits.astype(np.int64) - act_bits.astype(np.int64))
    n_exact = int((ulp_diff == 0).sum())
    n_total = ulp_diff.size
    max_ulp = int(ulp_diff.max())

    if max_ulp == 0:
        status = "BIT_IDENTICAL"
    elif max_ulp <= 4:
        status = "PASS_WITHIN_4_ULP"
    elif max_ulp <= 64:
        status = "PASS_WITHIN_64_ULP"
    elif max_ulp <= 1024:
        status = "REDUCTION_ORDER_DIVERGENCE"
    else:
        status = "MISMATCH"

    return {"case": case_name, "kernel": kernel_name, "status": status, "rc": 0,
            "max_ulp": max_ulp, "n_exact": n_exact, "n_total": n_total,
            "detail": f"{n_exact}/{n_total} exact, max ULP {max_ulp}"}


# Reduction case definitions: case_name -> (kernel_name, reference_fn)
REDUCTION_CASES = [
    ("reduce_sum_rows", "reduce_sum",   lambda x: torch.sum(x, dim=1)),
    ("reduce_mean",     "reduce_mean",  lambda x: torch.mean(x, dim=1)),
    ("reduce_max",      "reduce_max",   lambda x: torch.max(x, dim=1).values),
    ("reduce_min",      "reduce_min",   lambda x: torch.min(x, dim=1).values),
    ("reduce_argmax",   "reduce_argmax", lambda x: torch.argmax(x, dim=1).float()),
    ("reduce_argmin",   "reduce_argmin", lambda x: torch.argmin(x, dim=1).float()),
]


def main():
    print("=== Tier 2 / Subtask 2-inf-g-1 — bit_identical v1 (reductions) ===")
    print(f"Shape: outer=64, N=256; seed={SEED}")
    print()

    results = []
    for case_name, kernel_name, ref_fn in REDUCTION_CASES:
        try:
            r = run_reduction_case(case_name, kernel_name, ref_fn)
        except Exception as e:
            r = {"case": case_name, "kernel": kernel_name, "status": "HARNESS_ERROR",
                 "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {case_name:<20} {r.get('detail', '')}")

    # Summary
    print()
    print("=== Summary ===")
    by_status = {}
    for r in results:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1
    for status, count in sorted(by_status.items()):
        print(f"  {status:<32} {count}")

    print()
    pass_states = {"BIT_IDENTICAL", "PASS_WITHIN_4_ULP", "PASS_WITHIN_64_ULP",
                   "REDUCTION_ORDER_DIVERGENCE"}
    n_pass = sum(1 for r in results if r["status"] in pass_states)
    print(f"PASS (any tolerance):  {n_pass} / {len(results)}")
    print(f"BIT_IDENTICAL strict:  "
          f"{sum(1 for r in results if r['status'] == 'BIT_IDENTICAL')} / {len(results)}")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
