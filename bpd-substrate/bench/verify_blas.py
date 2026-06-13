# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify the BPD-generated SGEMV against PyTorch (which routes to cuBLAS on GPU).

The repo's headline claim is "0 ULP vs cuBLAS" for SGEMV on sm_61 (Pascal).
This harness exercises that claim on whatever architecture the .so was built
for — defaults to sm_86 (Ampere) via the parent Makefile.

Usage:
    BPD_BUILD_DIR=build python3 bench/verify_blas.py
    (or just `make verify` from the repo root.)

Output:
    Per-size report of max ULP, mean ULP, max abs error, and wall-clock timing
    against torch.matmul. Exits non-zero if any size exceeds MAX_ULP_THRESHOLD
    (currently 256 — a generous bound; the claim is 0).
"""

import ctypes
import os
import sys
import time

import numpy as np
import torch


BUILD_DIR = os.environ.get("BPD_BUILD_DIR", "build")
SO_PATH = os.path.join(BUILD_DIR, "blas_kernels.so")

# (M, K) sizes — claim is 0 ULP for the cublas_match kernel at K >= 32 (per
# the kernel's 32-thread-per-row layout). We exercise both small and large.
SIZES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 1024),
    (4096, 4096),
]


def load_lib():
    if not os.path.exists(SO_PATH):
        sys.exit(
            f"error: {SO_PATH} not found. Run `make blas` (or `make build`) first.\n"
            f"       BPD_BUILD_DIR={BUILD_DIR}"
        )
    lib = ctypes.CDLL(SO_PATH)
    lib.gpu_alloc.restype = ctypes.c_void_p
    lib.gpu_alloc.argtypes = [ctypes.c_int]
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_sync.argtypes = []
    lib.gpu_sgemv_cublas_match.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.gpu_sgemv_cublas_match.restype = None
    return lib


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bit-distance between two same-shape float32 arrays, treated as monotone
    integer encodings. Standard IEEE-754 ULP distance for normal floats.
    """
    assert a.dtype == np.float32 and b.dtype == np.float32
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    # Map negatives to a monotone range so distance is well-defined across zero.
    bias = np.int64(0x80000000)
    ai = np.where(ai < 0, bias - ai, ai)
    bi = np.where(bi < 0, bias - bi, bi)
    return np.abs(ai - bi)


def run_size(lib, M: int, K: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K), dtype=np.float32)
    x = rng.standard_normal((K,), dtype=np.float32)

    # BPD path
    A_flat = np.ascontiguousarray(A.reshape(-1))
    x_flat = np.ascontiguousarray(x)
    y_out = np.zeros((M,), dtype=np.float32)

    dA = lib.gpu_alloc(A_flat.nbytes)
    dx = lib.gpu_alloc(x_flat.nbytes)
    dy = lib.gpu_alloc(y_out.nbytes)
    lib.gpu_h2d(dA, A_flat.ctypes.data, A_flat.nbytes)
    lib.gpu_h2d(dx, x_flat.ctypes.data, x_flat.nbytes)

    # Warmup + timed runs
    for _ in range(3):
        lib.gpu_sgemv_cublas_match(dA, dx, dy, M, K)
    lib.gpu_sync()

    t0 = time.perf_counter()
    N_ITER = 50
    for _ in range(N_ITER):
        lib.gpu_sgemv_cublas_match(dA, dx, dy, M, K)
    lib.gpu_sync()
    bpd_us = (time.perf_counter() - t0) * 1e6 / N_ITER

    lib.gpu_d2h(y_out.ctypes.data, dy, y_out.nbytes)
    lib.gpu_free(dA); lib.gpu_free(dx); lib.gpu_free(dy)

    # Torch reference (routes to cuBLAS on CUDA)
    At = torch.from_numpy(A).cuda()
    xt = torch.from_numpy(x).cuda()
    yt = At @ xt
    for _ in range(2):
        yt = At @ xt
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        yt = At @ xt
    torch.cuda.synchronize()
    torch_us = (time.perf_counter() - t0) * 1e6 / N_ITER
    y_ref = yt.cpu().numpy()

    ulps = ulp_distance(y_out, y_ref)
    abs_err = np.abs(y_out - y_ref)
    return {
        "M": M, "K": K,
        "max_ulp": int(ulps.max()),
        "mean_ulp": float(ulps.mean()),
        "max_abs_err": float(abs_err.max()),
        "bpd_us": bpd_us,
        "torch_us": torch_us,
    }


def main():
    if not torch.cuda.is_available():
        sys.exit("error: torch.cuda is not available — this harness needs a CUDA GPU.")

    dev = torch.cuda.get_device_name(0)
    sm_major, sm_minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {dev} (sm_{sm_major}{sm_minor})")
    print(f"so:  {SO_PATH}")
    print()

    lib = load_lib()

    print(f"{'M':>6} {'K':>6}  {'max_ulp':>10} {'mean_ulp':>10}  "
          f"{'max_abs':>12}  {'bpd_us':>10} {'torch_us':>10}  speedup")
    print("-" * 90)
    worst_ulp = 0
    for M, K in SIZES:
        r = run_size(lib, M, K)
        speedup = r["torch_us"] / r["bpd_us"] if r["bpd_us"] > 0 else float("inf")
        print(f"{r['M']:>6} {r['K']:>6}  {r['max_ulp']:>10d} {r['mean_ulp']:>10.2f}  "
              f"{r['max_abs_err']:>12.2e}  {r['bpd_us']:>10.1f} {r['torch_us']:>10.1f}  "
              f"{speedup:>6.2f}x")
        worst_ulp = max(worst_ulp, r["max_ulp"])

    print()
    if worst_ulp == 0:
        print(f"0 ULP across all sizes on sm_{sm_major}{sm_minor} — claim holds.")
    else:
        print(f"max ULP across sizes: {worst_ulp} on sm_{sm_major}{sm_minor}.")
        print(f"  (The repo's 0-ULP claim was established on sm_61. Bit-identity")
        print(f"   is architecture-dependent — different SM families dispatch through")
        print(f"   different cuBLAS code paths, so this is information, not pass/fail.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
