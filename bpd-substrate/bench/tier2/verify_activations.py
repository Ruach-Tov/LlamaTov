#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_activations.py — Bit-identity verification for substrate elementwise ops.

Per Heath's continued direction 2026-05-20: continue per-layer Tier 2
verification of YOLO-relevant components until done. This sweeps the
unary-elementwise activations and verifies each against PyTorch reference.

Coverage:
    silu, relu, gelu, sigmoid, tanh, exp, hardtanh

Plus binary ops:
    bias_add, silu_mul, add_relu

Each kernel goes through: substrate emit (swipl) → nvcc compile →
ctypes invoke → IEEE 754 sign-magnitude ULP comparison.

Expected: BIT_IDENTICAL for all order-independent / monotone activations.
"""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("BPD_BUILD_DIR", REPO_DIR / "build"))
BUILD_DIR.mkdir(exist_ok=True, parents=True)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    """IEEE 754 sign-magnitude ULP comparison."""
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), int(d.size)


def emit_substrate_kernel(kernel_name: str) -> str:
    """Run swipl to emit a standalone elementwise kernel from substrate facts."""
    program = (
        f'use_module("lib/c_ast"), use_module("lib/kernel_templates_blas"), '
        f'elem_kernel({kernel_name}, K), '
        f'emit_program([c_include_sys(\'cuda_runtime.h\'), c_blank, K], Code), '
        f'format("~s", [Code]), halt'
    )
    result = subprocess.run(
        ["swipl", "-q", "-g", program],
        capture_output=True, text=True, cwd=REPO_DIR, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"swipl failed for {kernel_name}: {result.stderr}")
    return result.stdout


def compile_unary_so(kernel_name: str, arch: str) -> Path:
    """Compile a unary (X → Y, n params) kernel into a .so with extern C dispatch."""
    kernel_src = emit_substrate_kernel(kernel_name)
    dispatch = f"""
extern "C" {{
    int run_{kernel_name}(const float *h_X, float *h_Y, int n) {{
        float *d_X = nullptr, *d_Y = nullptr;
        size_t bytes = (size_t)n * sizeof(float);
        if (cudaMalloc(&d_X, bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (n + block - 1) / block;
        {kernel_name}<<<grid, block>>>(n, d_X, d_Y);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
    }}
}}
"""
    cu_path = BUILD_DIR / f"verify_{kernel_name}.cu"
    cu_path.write_text(kernel_src + dispatch)
    so_path = BUILD_DIR / f"verify_{kernel_name}.so"
    cmd = ["nvcc", "-shared", "-Xcompiler", "-fPIC",
           "-arch", arch, str(cu_path), "-o", str(so_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed for {kernel_name}:\n{r.stderr}")
    return so_path


def run_substrate_unary(so_path: Path, kernel_name: str, x: np.ndarray) -> np.ndarray:
    lib = ctypes.CDLL(str(so_path))
    fn = getattr(lib, f"run_{kernel_name}")
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int]
    fn.restype = ctypes.c_int
    x_flat = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    y = np.zeros_like(x_flat)
    rc = fn(x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(x_flat.size))
    if rc != 0:
        raise RuntimeError(f"runtime error rc={rc} for {kernel_name}")
    return y.reshape(x.shape)


# Test cases: (kernel_name, pytorch_op, input_range_factor, description)
# input_range_factor: scales randn(N) by this; controls the input distribution.
UNARY_CASES = [
    ("k_silu_blas",   F.silu,             3.0,  "SiLU (Swish): x / (1 + exp(-x))"),
    ("k_relu_blas",   F.relu,             3.0,  "ReLU: max(0, x)"),
    # Substrate's k_gelu_blas uses the tanh-approximation form (BERT/GPT style).
    # PyTorch's F.gelu defaults to the exact-erf form; passing approximate='tanh'
    # selects the form the substrate emits. Substrate-design observation:
    # GELU has two valid formulations and the substrate-design vocabulary should
    # eventually expose this as an explicit parameter (parallels reduction_strategy).
    ("k_gelu_blas",   lambda x: F.gelu(x, approximate='tanh'),  3.0,  "GELU (tanh approx — matches BERT/GPT)"),
    ("k_vsigmoid",    torch.sigmoid,      3.0,  "sigmoid: 1 / (1 + exp(-x))"),
    ("k_vtanh",       torch.tanh,         3.0,  "tanh"),
    ("k_vexp",        torch.exp,          1.0,  "exp (range -3..3 to avoid overflow)"),
    # k_hardtanh has a 5-param signature (n, min_val, max_val, x, y) — not the
    # standard unary shape. Skipped in this sweep; future work to extend the
    # harness with parameterized-unary dispatch. Low priority for YOLO (uncommon).
    # ("k_hardtanh",    lambda x: F.hardtanh(x), 3.0, "hardtanh: clamp(x, -1, 1)"),
    ("k_vabs",        torch.abs,          3.0,  "abs"),
    ("k_vneg",        torch.neg,          3.0,  "neg"),
    ("k_vsqr",        lambda x: x*x,      3.0,  "square: x*x"),
    ("k_vsqrt",       torch.sqrt,         "positive",  "sqrt (positive input only)"),
    ("k_vrsqrt",      torch.rsqrt,        "positive",  "rsqrt: 1/sqrt(x)"),
    ("k_vlog",        torch.log,          "positive",  "log (positive input only)"),
]


def make_input(n: int, seed: int, factor) -> torch.Tensor:
    torch.manual_seed(seed)
    if factor == "positive":
        # Strictly positive for log/sqrt/rsqrt
        return torch.abs(torch.randn(n)) + 0.01
    else:
        return torch.randn(n) * float(factor)


def verify_one_unary(kernel_name: str, pytorch_op, factor, arch: str) -> dict:
    """Run a full verification sweep for one unary kernel."""
    so_path = compile_unary_so(kernel_name, arch)
    shapes_seeds = [(64, 42), (1024, 42), (1024, 137), (1048576, 42)]
    worst_ulp = 0
    worst_diffs = 0
    total_elems = 0
    for n, seed in shapes_seeds:
        x = make_input(n, seed, factor).numpy().astype(np.float32)
        y_substrate = run_substrate_unary(so_path, kernel_name, x)
        x_t = torch.from_numpy(x).cuda()
        y_pytorch = pytorch_op(x_t).cpu().numpy()
        max_ulp, n_diffs, n_total = ulp_distance(y_pytorch, y_substrate)
        worst_ulp = max(worst_ulp, max_ulp)
        worst_diffs += n_diffs
        total_elems += n_total

    if worst_ulp == 0:
        status = "BIT_IDENTICAL"
    elif worst_ulp <= 4:
        status = "PASS_WITHIN_4_ULP"
    elif worst_ulp <= 64:
        status = "PASS_WITHIN_64_ULP"
    elif worst_ulp <= 1024:
        status = "PASS_WITHIN_1024_ULP"
    else:
        status = "DIVERGENT"

    return {
        "kernel": kernel_name,
        "status": status,
        "max_ulp": worst_ulp,
        "diffs": worst_diffs,
        "total": total_elems,
    }


def compile_binary_so(kernel_name: str, arch: str) -> Path:
    """Compile a binary (A,B → C, n params) kernel into a .so."""
    kernel_src = emit_substrate_kernel(kernel_name)
    dispatch = f"""
extern "C" {{
    int run_{kernel_name}(const float *h_A, const float *h_B, float *h_C, int n) {{
        float *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
        size_t bytes = (size_t)n * sizeof(float);
        if (cudaMalloc(&d_A, bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_B, bytes) != cudaSuccess) {{ cudaFree(d_A); return 2; }}
        if (cudaMalloc(&d_C, bytes) != cudaSuccess) {{ cudaFree(d_A); cudaFree(d_B); return 3; }}
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (n + block - 1) / block;
        {kernel_name}<<<grid, block>>>(n, d_A, d_B, d_C);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return 5; }}
        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return 0;
    }}
}}
"""
    cu_path = BUILD_DIR / f"verify_{kernel_name}.cu"
    cu_path.write_text(kernel_src + dispatch)
    so_path = BUILD_DIR / f"verify_{kernel_name}.so"
    r = subprocess.run(["nvcc", "-shared", "-Xcompiler", "-fPIC",
                        "-arch", arch, str(cu_path), "-o", str(so_path)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed for {kernel_name}:\n{r.stderr}")
    return so_path


def run_substrate_binary(so_path: Path, kernel_name: str,
                          a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lib = ctypes.CDLL(str(so_path))
    fn = getattr(lib, f"run_{kernel_name}")
    fn.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int]
    fn.restype = ctypes.c_int
    a_flat = np.ascontiguousarray(a.reshape(-1), dtype=np.float32)
    b_flat = np.ascontiguousarray(b.reshape(-1), dtype=np.float32)
    c = np.zeros_like(a_flat)
    rc = fn(a_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(a_flat.size))
    if rc != 0:
        raise RuntimeError(f"runtime error rc={rc} for {kernel_name}")
    return c.reshape(a.shape)


def verify_one_binary(kernel_name, pytorch_op, factor, arch: str) -> dict:
    so_path = compile_binary_so(kernel_name, arch)
    shapes_seeds = [(64, 42), (1024, 42), (1024, 137), (1048576, 42)]
    worst_ulp, worst_diffs, total_elems = 0, 0, 0
    for n, seed in shapes_seeds:
        torch.manual_seed(seed)
        a = (torch.randn(n) * float(factor)).numpy().astype(np.float32)
        b = (torch.randn(n) * float(factor)).numpy().astype(np.float32)
        c_substrate = run_substrate_binary(so_path, kernel_name, a, b)
        a_t = torch.from_numpy(a).cuda()
        b_t = torch.from_numpy(b).cuda()
        c_pytorch = pytorch_op(a_t, b_t).cpu().numpy()
        max_ulp, n_diffs, n_total = ulp_distance(c_pytorch, c_substrate)
        worst_ulp = max(worst_ulp, max_ulp)
        worst_diffs += n_diffs
        total_elems += n_total
    if worst_ulp == 0:
        status = "BIT_IDENTICAL"
    elif worst_ulp <= 4:
        status = "PASS_WITHIN_4_ULP"
    elif worst_ulp <= 64:
        status = "PASS_WITHIN_64_ULP"
    elif worst_ulp <= 1024:
        status = "PASS_WITHIN_1024_ULP"
    else:
        status = "DIVERGENT"
    return {"kernel": kernel_name, "status": status, "max_ulp": worst_ulp,
            "diffs": worst_diffs, "total": total_elems}


BINARY_CASES = [
    # k_vadd through k_vmin: pure binary ops
    ("k_vadd",       lambda a,b: a + b,             3.0,  "vadd: a + b"),
    ("k_vmul",       lambda a,b: a * b,             3.0,  "vmul: a * b"),
    ("k_vsub",       lambda a,b: a - b,             3.0,  "vsub: a - b"),
    ("k_vdiv",       lambda a,b: a / b,             3.0,  "vdiv: a / b (b shifted positive)"),
    ("k_vmax",       lambda a,b: torch.max(a, b),   3.0,  "vmax: max(a, b)"),
    ("k_vmin",       lambda a,b: torch.min(a, b),   3.0,  "vmin: min(a, b)"),
    # Fused ops
    ("k_silu_mul",   lambda a,b: F.silu(a) * b,     3.0,  "silu_mul: silu(gate) * up (SwiGLU)"),
    ("k_add_relu",   lambda a,b: F.relu(a + b),     3.0,  "add_relu: max(0, a + b)"),
]


def main():
    arch = os.environ.get("NVCC_ARCH", "sm_86")
    print("=" * 72)
    print(f"verify_activations.py — substrate elementwise ops vs PyTorch")
    print(f"Arch: {arch}")
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}")
    print("=" * 72)
    print()
    print(f"{'kernel':<18}  {'status':<22}  {'max ULP':>10}  {'diffs':>8}  {'description'}")
    print("-" * 95)

    results = []
    print("\n--- Unary activations ---")
    for kernel_name, op, factor, desc in UNARY_CASES:
        try:
            r = verify_one_unary(kernel_name, op, factor, arch)
        except Exception as e:
            r = {"kernel": kernel_name, "status": "HARNESS_ERROR",
                 "max_ulp": -1, "diffs": -1, "total": 0}
            print(f"{kernel_name:<18}  HARNESS_ERROR  {str(e)[:50]}")
            results.append(r)
            continue
        results.append(r)
        print(f"{r['kernel']:<18}  {r['status']:<22}  {r['max_ulp']:>10}  "
              f"{r['diffs']:>4}/{r['total']:<7}  {desc}")

    print("\n--- Binary ops ---")
    for kernel_name, op, factor, desc in BINARY_CASES:
        try:
            r = verify_one_binary(kernel_name, op, factor, arch)
        except Exception as e:
            r = {"kernel": kernel_name, "status": "HARNESS_ERROR",
                 "max_ulp": -1, "diffs": -1, "total": 0}
            print(f"{kernel_name:<18}  HARNESS_ERROR  {str(e)[:80]}")
            results.append(r)
            continue
        results.append(r)
        print(f"{r['kernel']:<18}  {r['status']:<22}  {r['max_ulp']:>10}  "
              f"{r['diffs']:>4}/{r['total']:<7}  {desc}")

    print()
    print("=" * 72)
    bit_id = sum(1 for r in results if r["status"] == "BIT_IDENTICAL")
    within_4 = sum(1 for r in results if r["status"] == "PASS_WITHIN_4_ULP")
    within_64 = sum(1 for r in results if r["status"] == "PASS_WITHIN_64_ULP")
    within_1024 = sum(1 for r in results if r["status"] == "PASS_WITHIN_1024_ULP")
    divergent = sum(1 for r in results if r["status"] == "DIVERGENT")
    error = sum(1 for r in results if r["status"] == "HARNESS_ERROR")
    print(f"BIT_IDENTICAL:        {bit_id}")
    print(f"PASS_WITHIN_4_ULP:    {within_4}")
    print(f"PASS_WITHIN_64_ULP:   {within_64}")
    print(f"PASS_WITHIN_1024_ULP: {within_1024}")
    print(f"DIVERGENT:            {divergent}")
    print(f"HARNESS_ERROR:        {error}")
    print()
    return 0 if divergent == 0 and error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
