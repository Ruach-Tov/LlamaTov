#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_mish.py — Bit-identity verification for substrate-emitted mish.

Per Heath's "Colossal Importance" framing 2026-05-20 ~02:10 UTC:
mish lifted to first-class elem_op in commit (to come). This script
verifies the substrate's emission produces bit-identical output with
torch.nn.functional.mish.

Math: mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))

The three transcendentals (exp, log, tanh) make this a non-trivial
bit-identity test. Reference: PyTorch's F.mish, which ultimately
routes to ATen's mish kernel using the same CUDA MUFU.{EX2, LG2, TANH}
hardware path.

Expected result: BIT_IDENTICAL (0 ULP) at all test shapes. If divergence
appears, that's substantive substrate-design data — the substrate emit
and ATen's emit differ in their formulation despite using the same
hardware instructions.

Usage:
    python3 bench/tier2/verify_mish.py
"""
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    sys.exit("error: torch with CUDA required.")

REPO_DIR = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("BPD_BUILD_DIR", REPO_DIR / "build"))
BUILD_DIR.mkdir(exist_ok=True, parents=True)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    """IEEE 754 sign-magnitude ULP comparison.

    Returns (max_ulp, n_diffs, n_total).
    """
    assert a.dtype == b.dtype == np.float32
    assert a.shape == b.shape
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    # Sign-magnitude: convert negative-zero-form to monotonic int64
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), int(d.size)


def emit_substrate_mish() -> str:
    """Run swipl to emit the standalone mish .cu source."""
    program = """
    use_module(\"lib/c_ast\"),
    use_module(\"lib/kernel_templates_blas\"),
    elem_kernel(k_mish_blas, K),
    emit_program([
        c_include_sys('cuda_runtime.h'),
        c_blank,
        K
    ], Code),
    format(\"~s\", [Code]),
    halt
    """.strip()
    result = subprocess.run(
        ["swipl", "-q", "-g", program],
        capture_output=True, text=True, cwd=REPO_DIR, timeout=30
    )
    if result.returncode != 0:
        sys.exit(f"swipl failed: {result.stderr}")
    return result.stdout


def compile_mish_so(arch: str = "sm_86") -> Path:
    """Compose the substrate emit + extern C wrapper, nvcc-compile to .so."""
    kernel_src = emit_substrate_mish()
    dispatch = """
extern "C" {
    int run_mish(const float *h_X, float *h_Y, int n) {
        float *d_X = nullptr, *d_Y = nullptr;
        size_t bytes = (size_t)n * sizeof(float);
        if (cudaMalloc(&d_X, bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, bytes) != cudaSuccess) { cudaFree(d_X); return 2; }
        if (cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_X); cudaFree(d_Y); return 3;
        }
        int block = 256;
        int grid = (n + block - 1) / block;
        k_mish_blas<<<grid, block>>>(n, d_X, d_Y);
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(d_X); cudaFree(d_Y); return 4;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(d_X); cudaFree(d_Y); return 5;
        }
        if (cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(d_X); cudaFree(d_Y); return 6;
        }
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
    }
}
"""
    cu_path = BUILD_DIR / "verify_mish.cu"
    cu_path.write_text(kernel_src + dispatch)
    so_path = BUILD_DIR / "verify_mish.so"
    cmd = ["nvcc", "-shared", "-Xcompiler", "-fPIC",
           "-arch", arch, str(cu_path), "-o", str(so_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        sys.exit(f"nvcc failed:\n{result.stderr}")
    return so_path


def run_substrate_mish(so_path: Path, x: np.ndarray) -> np.ndarray:
    lib = ctypes.CDLL(str(so_path))
    lib.run_mish.argtypes = [ctypes.POINTER(ctypes.c_float),
                             ctypes.POINTER(ctypes.c_float),
                             ctypes.c_int]
    lib.run_mish.restype = ctypes.c_int
    x_flat = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    y = np.zeros_like(x_flat)
    rc = lib.run_mish(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(x_flat.size)
    )
    if rc != 0:
        sys.exit(f"substrate kernel error rc={rc}")
    return y.reshape(x.shape)


def run_pytorch_mish(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x).cuda()
    y = F.mish(t)
    return y.cpu().numpy()


def main():
    arch = os.environ.get("NVCC_ARCH", "sm_86")
    print("=" * 64)
    print(f"verify_mish.py — substrate emit vs torch.nn.functional.mish")
    print(f"Arch: {arch}")
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}")
    print("=" * 64)
    print()

    print("[1] Emitting substrate mish kernel from elem_op(k_mish_blas, ...)...")
    print()
    kernel_src = emit_substrate_mish()
    # Show just the kernel body, not the includes
    for line in kernel_src.splitlines():
        print(f"    {line}")
    print()

    print(f"[2] Compiling for {arch}...")
    so_path = compile_mish_so(arch)
    print(f"    {so_path} ({so_path.stat().st_size} bytes)")
    print()

    print("[3] Running bit-identity sweep across diverse shapes...")
    print()
    print(f"    {'Shape':<14} {'Seed':>6}  {'max ULP':>10}  {'diffs':>10}  {'verdict'}")
    print(f"    {'-'*14} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*20}")

    test_cases = [
        (64, 42),
        (256, 42),
        (1024, 42),
        (1024, 137),  # different seed, same shape — sanity
        (1024 * 1024, 42),
        (4096, 7),
    ]

    results = []
    for n, seed in test_cases:
        torch.manual_seed(seed)
        # Range that exercises full transcendentals: not too small (where softplus
        # is approx identity), not too large (where softplus saturates).
        x = (torch.randn(n) * 3.0).numpy().astype(np.float32)
        y_substrate = run_substrate_mish(so_path, x)
        y_pytorch = run_pytorch_mish(x)
        max_ulp, n_diffs, n_total = ulp_distance(y_pytorch, y_substrate)
        verdict = "0 ULP ✓" if max_ulp == 0 else f"divergent"
        results.append((n, seed, max_ulp, n_diffs, n_total))
        print(f"    {f'({n})':<14} {seed:>6}  {max_ulp:>10}  "
              f"{n_diffs:>5}/{n_total:<5}  {verdict}")

    print()
    print("=" * 64)
    all_zero = all(r[2] == 0 for r in results)
    if all_zero:
        print("VERDICT: BIT_IDENTICAL across all test shapes. ✓")
        print()
        print("substrate-design observation: the three transcendentals")
        print("(exp, log, tanh) chained for mish produce bit-identical")
        print("output with PyTorch's F.mish across diverse shapes and seeds.")
        print("The substrate's emit routes to the same CUDA MUFU.{EX2, LG2,")
        print("TANH} hardware instructions ATen uses.")
        return 0
    else:
        worst = max(results, key=lambda r: r[2])
        print(f"VERDICT: substantive divergence — max {worst[2]} ULP")
        print(f"    Worst case: shape {worst[0]}, seed {worst[1]}")
        print(f"    This is substrate-design data. Either the substrate's")
        print(f"    formulation differs from ATen's, or PyTorch is using")
        print(f"    a different precision path. Next: inspect SASS.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
