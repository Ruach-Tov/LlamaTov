#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

"""kernel_harness.py — the SHARED verification harness (fixes infra-debt #1).

Before this, ~15 one-off /tmp scripts each re-implemented "build cubin -> run on the
P4 -> verify bit-exact vs torch -> ulp/max_abs -> time". This factors that logic into
ONE reusable library so every harness (and the multibackend referee) uses the same
verify-vs-oracle + ULP + timing machinery — no copy-paste, no drift.

Shared with multibackend_referee.py / stanford_referee.py: the ulp() function is the
canonical bit-exactness measure (0 ULP = bit-identical, NaN-aware, +-0-aware).

Usage (on the enclave):
    from kernel_harness import Env, build_cubin, run_kernel, verify, time_kernel
    env = Env()
    cubin = build_cubin(env, "/path/k.cu")
    out = run_kernel(env, cubin, "k_name", grid, block, args, out_shape, shmem=...)
    rep = verify(out, reference)          # {max_ulp, n_differ, max_abs, bit_exact}
    ms  = time_kernel(env, cubin, "k_name", grid, block, args, iters=100)

Author: Iyun, 2026-06-08
"""
import os, re, subprocess, ctypes, struct
import numpy as np
import sys as _sys; _sys.path.insert(0, _os.path.join(_BPD, "lib"))
import toolchain as tc

# ── canonical toolchain paths (enclave) ──────────────────────────────────────
CUDA   = tc.cuda_root()  # shared toolchain (ENV-SHIFT defense)
REDIST = "/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib"

def _find_devrt():
    return tc.link_lib_dirs()[1]  # the devrt dir (shared toolchain, found once)

class Env:
    """Toolchain + working-dir context for kernel build/run."""
    def __init__(self, work="/tmp/gpu-work/harness", cuda=CUDA):
        self.cuda = cuda
        self.work = work
        os.makedirs(work, exist_ok=True)
        self.env = tc.nvcc_env()  # shared toolchain env
        self.devrt = _find_devrt()

    def run(self, cmd, shell=False, timeout=120):
        return subprocess.run(cmd, capture_output=True, text=True,
                              env=self.env, cwd=self.work, timeout=timeout, shell=shell)

# ── the canonical ULP measure (shared with multibackend_referee / stanford) ──
def ulp(a, b):
    """0 ULP = bit-identical. NaN-aware, +-0-aware. Returns (max_ulp, n_differ, nan_mismatch, signzero)."""
    a = np.ascontiguousarray(a, np.float32).ravel()
    b = np.ascontiguousarray(b, np.float32).ravel()
    n = min(a.size, b.size); a = a[:n]; b = b[:n]
    nan_a = np.isnan(a); nan_b = np.isnan(b)
    nan_mismatch = int((nan_a != nan_b).sum())
    both_zero = (a == 0) & (b == 0)
    ai = np.frombuffer(a.tobytes(), np.int32).astype(np.int64)
    bi = np.frombuffer(b.tobytes(), np.int32).astype(np.int64)
    signzero = int((both_zero & (ai != bi)).sum())
    mask = ~(nan_a | nan_b) & ~both_zero
    u = np.abs(ai - bi); u[~mask] = 0
    return int(u.max()), int((u > 0).sum()), nan_mismatch, signzero

# ── build ────────────────────────────────────────────────────────────────────
def build_cubin(env, src, arch="sm_61", opts=None, ptxas_v=False):
    """Compile a .cu -> .cubin. Returns the cubin path, or None on failure (prints stderr)."""
    out = src.rsplit(".", 1)[0] + ".cubin"
    _extra = list(opts or [])
    if ptxas_v:
        _extra.append("--ptxas-options=-v")
    cmd = tc.nvcc_compile_cmd(src, out, arch=arch, extra=_extra)  # appends -I, src, -o out
    r = env.run(cmd)
    if not os.path.exists(out):
        print(f"[harness] build FAIL for {src}:\n{r.stderr[:400]}")
        return None
    if ptxas_v:
        for ln in r.stderr.splitlines():
            if "registers" in ln.lower() or "stack frame" in ln.lower():
                print(f"[harness] {os.path.basename(src)}: {ln.strip()}")
    return out

# ── verify ───────────────────────────────────────────────────────────────────
def verify(got, reference, ulp_tol=0, abs_tol=1e-2):
    """Compare got vs reference. Returns a structured report.
    bit_exact = max_ulp <= ulp_tol (default 0). within_tol = max_abs <= abs_tol."""
    got = np.asarray(got, np.float32); reference = np.asarray(reference, np.float32)
    mu, nd, nanm, sz = ulp(got, reference)
    d = np.abs(got.ravel()[:reference.size] - reference.ravel()[:got.size])
    max_abs = float(d.max()) if d.size else 0.0
    return {
        "max_ulp": mu, "n_differ": nd, "nan_mismatch": nanm, "signzero": sz,
        "max_abs": max_abs,
        "bit_exact": mu <= ulp_tol and nanm == 0,
        "within_tol": max_abs <= abs_tol,
    }

# ── run a kernel via the CUDA driver API (ctypes — no nvcc link needed) ──────
_CUDA_LIB = None
def _cuda():
    """Load the real libcuda.so.1 (the driver, not the stubs). On nix the driver
    lives in /run/opengl-driver/lib; the default loader path has only stubs."""
    global _CUDA_LIB
    if _CUDA_LIB is not None:
        return _CUDA_LIB
    for cand in ("/run/opengl-driver/lib/libcuda.so.1", "libcuda.so.1", "libcuda.so"):
        try:
            _CUDA_LIB = ctypes.CDLL(cand)
            return _CUDA_LIB
        except OSError:
            continue
    raise OSError("libcuda.so.1 not found (driver, not stubs) — check /run/opengl-driver/lib")

def run_kernel(env, cubin, fname, grid, block, host_args, out_idx, out_shape,
               shmem=0):
    """Launch a kernel and return its output array.
    host_args: list of (np.ndarray | int | float | 'OUT'); 'OUT' marks the output buffer.
    out_idx: index in host_args of the output; out_shape: shape of the output array.
    Returns the output np.float32 array. Driver-API based (libcuda), no nvcc link."""
    cu = _cuda()
    cu.cuInit(0)
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    mod = ctypes.c_void_p()
    rc = cu.cuModuleLoad(ctypes.byref(mod), cubin.encode())
    if rc != 0:
        raise RuntimeError(f"cuModuleLoad failed rc={rc} for {cubin}")
    fn = ctypes.c_void_p()
    rc = cu.cuModuleGetFunction(ctypes.byref(fn), mod, fname.encode())
    if rc != 0:
        raise RuntimeError(f"cuModuleGetFunction({fname}) failed rc={rc}")
    devptrs, kargs, out_dptr, out_nbytes = [], [], None, 0
    for i, a in enumerate(host_args):
        if isinstance(a, str) and a == "OUT":
            nbytes = int(np.prod(out_shape)) * 4
            dptr = ctypes.c_void_p()
            cu.cuMemAlloc_v2(ctypes.byref(dptr), nbytes)
            devptrs.append(dptr); kargs.append(ctypes.byref(dptr))
            out_dptr, out_nbytes = dptr, nbytes
        elif isinstance(a, np.ndarray):
            arr = np.ascontiguousarray(a, np.float32)
            nbytes = arr.nbytes
            dptr = ctypes.c_void_p()
            cu.cuMemAlloc_v2(ctypes.byref(dptr), nbytes)
            cu.cuMemcpyHtoD_v2(dptr, arr.ctypes.data_as(ctypes.c_void_p), nbytes)
            devptrs.append(dptr); kargs.append(ctypes.byref(dptr))
        elif isinstance(a, int):
            ci = ctypes.c_int(a); kargs.append(ctypes.byref(ci)); devptrs.append(ci)
        elif isinstance(a, float):
            cf = ctypes.c_float(a); kargs.append(ctypes.byref(cf)); devptrs.append(cf)
    argv = (ctypes.c_void_p * len(kargs))(*[ctypes.cast(k, ctypes.c_void_p) for k in kargs])
    gx, gy, gz = (grid + (1, 1, 1))[:3]
    bx, by, bz = (block + (1, 1, 1))[:3]
    rc = cu.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shmem, None, argv, None)
    cu.cuCtxSynchronize()
    out = np.empty(out_shape, np.float32)
    cu.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), out_dptr, out_nbytes)
    cu.cuCtxDestroy_v2(ctx)
    return out

# ── attention oracle (used by every flash harness — was re-written each time) ─
def attention_reference(Q, K, V, scale=None):
    """The torch SDPA-equivalent oracle: softmax(scale * Q@K^T) @ V. CPU, f32."""
    import torch, torch.nn.functional as F
    D = Q.shape[-1]
    sc = scale if scale is not None else 1.0 / np.sqrt(D)
    s = torch.from_numpy(np.ascontiguousarray(Q, np.float32)) @ \
        torch.from_numpy(np.ascontiguousarray(K, np.float32)).T * sc
    return (F.softmax(s, dim=-1) @ torch.from_numpy(np.ascontiguousarray(V, np.float32))).numpy()

def deterministic_qkv(S, D):
    """The canonical deterministic test input used across the flash harnesses."""
    idx = np.arange(S * D, dtype=np.uint64)
    h = ((idx * np.uint64(2654435761)) % 1000).astype(np.float32) / 500.0 - 1.0
    Q = h.reshape(S, D)
    return Q, Q.copy(), Q.copy()

# ── timing (driver-API CUDA events) ──────────────────────────────────────────
def time_kernel(env, cubin, fname, grid, block, host_args, out_idx, out_shape,
                shmem=0, iters=100):
    """Time a kernel over `iters` launches via CUDA events. Returns ms/launch.
    Reuses run_kernel's arg marshalling by doing one setup then a timed loop."""
    cu = _cuda()
    cu.cuInit(0)
    dev = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    mod = ctypes.c_void_p(); cu.cuModuleLoad(ctypes.byref(mod), cubin.encode())
    fn = ctypes.c_void_p(); cu.cuModuleGetFunction(ctypes.byref(fn), mod, fname.encode())
    devptrs, kargs = [], []
    for a in host_args:
        if isinstance(a, str) and a == "OUT":
            nbytes = int(np.prod(out_shape)) * 4
            dptr = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dptr), nbytes)
            devptrs.append(dptr); kargs.append(ctypes.byref(dptr))
        elif isinstance(a, np.ndarray):
            arr = np.ascontiguousarray(a, np.float32)
            dptr = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dptr), arr.nbytes)
            cu.cuMemcpyHtoD_v2(dptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)
            devptrs.append(dptr); kargs.append(ctypes.byref(dptr))
        elif isinstance(a, int):
            ci = ctypes.c_int(a); kargs.append(ctypes.byref(ci)); devptrs.append(ci)
        elif isinstance(a, float):
            cf = ctypes.c_float(a); kargs.append(ctypes.byref(cf)); devptrs.append(cf)
    argv = (ctypes.c_void_p * len(kargs))(*[ctypes.cast(k, ctypes.c_void_p) for k in kargs])
    gx, gy, gz = (grid + (1, 1, 1))[:3]; bx, by, bz = (block + (1, 1, 1))[:3]
    e0 = ctypes.c_void_p(); e1 = ctypes.c_void_p()
    cu.cuEventCreate(ctypes.byref(e0), 0); cu.cuEventCreate(ctypes.byref(e1), 0)
    for _ in range(5):  # warmup
        cu.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shmem, None, argv, None)
    cu.cuCtxSynchronize()
    cu.cuEventRecord(e0, None)
    for _ in range(iters):
        cu.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shmem, None, argv, None)
    cu.cuEventRecord(e1, None); cu.cuEventSynchronize(e1)
    ms = ctypes.c_float(); cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1)
    cu.cuCtxDestroy_v2(ctx)
    return ms.value / iters
