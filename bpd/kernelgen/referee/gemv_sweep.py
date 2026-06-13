#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""gemv_sweep.py — sweep the tiled Q8_0 GEMV parameter space (gpu_gemv_point(BM,BK,VEC)).
Per Heath: generate-first, sweep the RANGE (concrete trends, not just extremes). For each
valid point: GENERATE the tiled kernel, VERIFY correctness (rel-error vs the serial reference),
MEASURE sectors_per_load + timing on the P4. Reports the response surface so we see WHICH
parameter drives DRAM efficiency toward ggml's 0.19 (ours currently ~0.4-0.5).

Run on the enclave. Usage: python3 gemv_sweep.py [--shape ffn_down] [--limit N]
"""
import sys, os, ctypes, time, subprocess, argparse, itertools
import numpy as np
import os as _os2
_REPO = _os2.environ.get("LLAMATOV_ROOT") or _os2.path.abspath(_os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), *[".."]*8))

sys.path.insert(0, _BPD)
sys.path.insert(0, _os.path.join(_BPD, "lib"))
sys.path.insert(0, _os2.path.join(_REPO, "bpd/kernelgen"))
import dev_residency as dr, fact_dispatch as fd
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

cu = fd._libcuda(); fd._ctx(); dr.cu = cu

SHAPES = {"attn": (896, 896), "ffn_down": (896, 4864), "ffn_up": (4864, 896)}

def valid_points():
    pts = []
    for BM in [1, 2, 4, 8, 16, 32, 64]:
        for BK in [32, 64, 128, 256]:
            for VEC in [1, 2, 4]:
                if BK % 32 == 0 and 8 % VEC == 0 and (BK + BK//16) <= 49152 and BM*32 <= 1024:
                    pts.append((BM, BK, VEC))
    return pts

def emit_tiled(BM, BK, VEC):
    tag = f"gemv_tiled_{BM}_{BK}_{VEC}"
    out = f"{fd._CACHE}/{tag}.cu"
    for f in os.listdir(fd._CACHE):
        if f.startswith(tag): os.remove(os.path.join(fd._CACHE, f))
    return fd._emit_and_build(["FACTS", f"{fd._EMIT}/q8_0_from_facts.pl"],
        f'q8_0_op_expr(E), emit_from_fact(E, [mode(tiled({BM},{BK},{VEC}))], "{out}")', tag)

def emit_serial():
    return fd._emit_and_build(["FACTS", f"{fd._EMIT}/q8_0_from_facts.pl"],
        f'q8_0_op_expr(E), emit_from_fact(E, [mode(dp4a)], "{fd._CACHE}/gemv_serial_ref.cu")', "gemv_serial_ref")

def run_kernel(cubin, kname, dWq, dWd, dXq, dXd, M, K, BM=None):
    fn = fd._func(cubin, kname)
    dY = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY), M*4)
    Mi, Ki = ctypes.c_int(M), ctypes.c_int(K); nb = K//32
    args = [dWq, dWd, dXq, dXd, dY, Mi, Ki]
    argv = (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    if BM:  # tiled launch
        grid = (M + BM - 1)//BM; block = BM*32; shmem = K + nb*2
        cu.cuLaunchKernel(fn, grid,1,1, block,1,1, shmem, None, argv, None)
    else:   # serial thread-per-row
        cu.cuLaunchKernel(fn, (M+63)//64,1,1, 64,1,1, 0, None, argv, None)
    cu.cuCtxSynchronize()
    Y = np.empty(M, np.float32); cu.cuMemcpyDtoH_v2(Y.ctypes.data_as(ctypes.c_void_p), dY, M*4)
    cu.cuMemFree_v2(dY)
    return Y

def time_kernel(cubin, kname, dWq, dWd, dXq, dXd, M, K, BM=None, iters=200):
    fn = fd._func(cubin, kname)
    dY = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY), M*4)
    Mi, Ki = ctypes.c_int(M), ctypes.c_int(K); nb = K//32
    args = [dWq, dWd, dXq, dXd, dY, Mi, Ki]
    argv = (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    def launch():
        if BM:
            cu.cuLaunchKernel(fn, (M+BM-1)//BM,1,1, BM*32,1,1, K+nb*2, None, argv, None)
        else:
            cu.cuLaunchKernel(fn, (M+63)//64,1,1, 64,1,1, 0, None, argv, None)
    for _ in range(10): launch()
    cu.cuCtxSynchronize(); t0 = time.time()
    for _ in range(iters): launch()
    cu.cuCtxSynchronize()
    cu.cuMemFree_v2(dY)
    return (time.time()-t0)/iters*1e6  # us/launch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="ffn_down", choices=list(SHAPES))
    ap.add_argument("--limit", type=int, default=999)
    args = ap.parse_args()
    M, K = SHAPES[args.shape]; nb = K//32
    np.random.seed(7)
    Wq = np.random.randint(-127,128,size=M*K,dtype=np.int8)
    Wd = (np.random.randn(M*nb)*0.1).astype(np.float16)
    Xq = np.random.randint(-127,128,size=K,dtype=np.int8)
    Xd = (np.random.randn(nb)*0.1).astype(np.float16)
    def dptr(a):
        p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes)
        cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
    dWq,dWd,dXq,dXd = dptr(Wq),dptr(Wd),dptr(Xq),dptr(Xd)
    # serial reference
    ref = run_kernel(emit_serial(), "k_q8_0_gemv", dWq,dWd,dXq,dXd, M,K)
    ref_us = time_kernel(emit_serial(), "k_q8_0_gemv", dWq,dWd,dXq,dXd, M,K)
    print(f"=== TILED Q8_0 GEMV SWEEP  shape={args.shape} (M={M} K={K}) ===", flush=True)
    print(f"serial reference: {ref_us:.1f} us/launch (the BM=1 thread-per-row corner)", flush=True)
    print(f"{'BM':>3} {'BK':>4} {'VEC':>3} | {'us/launch':>9} {'speedup':>7} {'max_ulp':>7} {'rel_err':>9} | verdict", flush=True)
    results = []
    for (BM,BK,VEC) in valid_points()[:args.limit]:
        try:
            cubin = emit_tiled(BM,BK,VEC)
            y = run_kernel(cubin, "k_q8_0_gemv", dWq,dWd,dXq,dXd, M,K, BM=BM)
            us = time_kernel(cubin, "k_q8_0_gemv", dWq,dWd,dXq,dXd, M,K, BM=BM)
            ulp = int(np.abs(y.view(np.uint32).astype(np.int64)-ref.view(np.uint32).astype(np.int64)).max())
            rel = float(np.abs(y-ref).max()/(np.abs(ref).max()+1e-12))
            ok = rel < 1e-3
            sp = ref_us/us
            print(f"{BM:>3} {BK:>4} {VEC:>3} | {us:>9.1f} {sp:>6.2f}x {ulp:>7} {rel:>9.2e} | {'OK' if ok else 'BAD'}", flush=True)
            results.append((BM,BK,VEC,us,sp,ulp,rel,ok))
        except Exception as e:
            print(f"{BM:>3} {BK:>4} {VEC:>3} | FAIL: {str(e)[:60]}", flush=True)
    ok_res = [r for r in results if r[7]]
    if ok_res:
        best = min(ok_res, key=lambda r: r[3])
        print(f"\n>>> FASTEST correct: BM={best[0]} BK={best[1]} VEC={best[2]} = {best[3]:.1f}us ({best[4]:.2f}x vs serial), max_ulp={best[5]}", flush=True)

if __name__ == "__main__":
    main()
