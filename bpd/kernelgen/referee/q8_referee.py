#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""q8_referee.py — standing differential gate for the fact-derived Q8_0 GEMV.

Builds the q8_0_from_facts emitter's output (scalar + dp4a) and verifies both on the
P4 vs the dequant-then-dot oracle, via the shared kernel_harness. Brings the Q8_0
quantized dot — the novel kernel of the llama-Q8_0 path — under a standing referee
(infra-debt #2: route new work through the gate, not /tmp scripts).

Run on enclave:  python3 q8_referee.py
Author: Iyun, 2026-06-08
"""
import os, sys, subprocess, ctypes
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernel_harness import Env, build_cubin
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EMIT = os.path.join(REPO, "bpd", "kernelgen", "emitters")
SWIPL = "/run/current-system/sw/bin/swipl"

M, K = 64, 256
nblk = K // 32

def _emit(mode, out):
    g = (f"use_module('{EMIT}/q8_0_from_facts.pl'), "
         f'q8_0_op_expr(E), emit_from_fact(E, [mode({mode})], "{out}"), halt')
    subprocess.run([SWIPL, "-q", "-g", g, "-t", "halt"], capture_output=True, text=True, timeout=30)
    return out if os.path.exists(out) else None

def _q8_quant(mat):
    rows = mat.shape[0]
    q = np.zeros((rows, K), np.int8); d = np.zeros((rows, nblk), np.float16)
    for r in range(rows):
        for b in range(nblk):
            blk = mat[r, b*32:(b+1)*32]; amax = float(np.max(np.abs(blk)))
            dd = np.float16(amax/127.0) if amax > 0 else np.float16(1.0)
            d[r, b] = dd
            q[r, b*32:(b+1)*32] = np.clip(np.round(blk/np.float32(dd)), -127, 127).astype(np.int8)
    return q, d

def _run(env, cubin):
    cu = ctypes.CDLL("/run/opengl-driver/lib/libcuda.so.1")
    cu.cuInit(0); dev = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p(); cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    mod = ctypes.c_void_p(); cu.cuModuleLoad(ctypes.byref(mod), cubin.encode())
    fn = ctypes.c_void_p(); cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"k_q8_0_gemv")
    def up(a, n):
        p = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p), n)
        cu.cuMemcpyHtoD_v2(p, a.ctypes.data_as(ctypes.c_void_p), n); return p
    dWq = up(np.ascontiguousarray(Wq, np.int8), Wq.nbytes)
    dWd = up(np.ascontiguousarray(Wd, np.float16), Wd.nbytes)
    dXq = up(np.ascontiguousarray(Xq, np.int8), Xq.nbytes)
    dXd = up(np.ascontiguousarray(Xd, np.float16), Xd.nbytes)
    dY = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY), M*4)
    Mi = ctypes.c_int(M); Ki = ctypes.c_int(K)
    args = [dWq, dWd, dXq, dXd, dY, Mi, Ki]
    argv = (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    blk = 64; grid = (M+blk-1)//blk
    cu.cuLaunchKernel(fn, grid, 1, 1, blk, 1, 1, 0, None, argv, None)
    cu.cuCtxSynchronize()
    Y = np.empty(M, np.float32); cu.cuMemcpyDtoH_v2(Y.ctypes.data_as(ctypes.c_void_p), dY, M*4)
    cu.cuCtxDestroy_v2(ctx)
    return Y

np.random.seed(11)
W = (np.random.randn(M, K).astype(np.float32)) * 0.3
X = (np.random.randn(K).astype(np.float32)) * 0.5
Wq, Wd = _q8_quant(W)
Xq, Xd = _q8_quant(X.reshape(1, K)); Xq = Xq.reshape(K); Xd = Xd.reshape(nblk)
Wdeq = (Wd[:, :, None].astype(np.float32) * Wq.reshape(M, nblk, 32).astype(np.float32)).reshape(M, K)
Xdeq = (Xd[:, None].astype(np.float32) * Xq.reshape(nblk, 32).astype(np.float32)).reshape(K)
ORACLE = (Wdeq @ Xdeq).astype(np.float32)

def main():
    env = Env(work="/tmp/gpu-work/q8_referee")
    print(f"=== Q8_0 REFEREE (M={M} K={K}, P4) — fact-derived GEMV vs dequant-dot ===")
    print(f"{'variant':<14}{'max_abs':<11}{'within_tol':<11}{'verdict'}")
    all_ok = True
    for mode in ("scalar", "dp4a"):
        src = _emit(mode, os.path.join(env.work, f"q8_{mode}.cu"))
        if not src:
            print(f"{mode:<14}EMIT-FAIL"); all_ok = False; continue
        cubin = build_cubin(env, src)
        if not cubin:
            print(f"{mode:<14}BUILD-FAIL"); all_ok = False; continue
        Y = _run(env, cubin)
        d = np.abs(Y - ORACLE); ok = bool(d.max() < 1e-2)
        if not ok: all_ok = False
        print(f"{mode:<14}{d.max():<11.6f}{str(ok):<11}{'PASS' if ok else 'FAIL'}")
    print("\n" + ("Q8_0 GEMV VERIFIED ✓" if all_ok else "SOME FAILED ✗"))
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
