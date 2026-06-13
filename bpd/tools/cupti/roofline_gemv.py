# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Roofline check: are the GEMVs at the DRAM bandwidth roofline, or wasting it?
For each shape, compute the minimum-possible time (weight bytes / peak BW) and compare to actual.
If actual >> roofline, the memory_dependency is recoverable (L2/coalescing). If actual ~= roofline,
we're bandwidth-bound and the only lever is reading FEWER bytes (e.g. better cache reuse across rows)."""
import sys, ctypes, time, numpy as np
sys.path.insert(0, _BPD); sys.path.insert(0, _os.path.join(_BPD, "lib"))
import dev_residency as dr, fact_dispatch as fd
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

cu = fd._libcuda(); fd._ctx(); dr.cu = cu

PEAK_GBs = 192.0  # Tesla P4 memory bandwidth

def build_v4():
    return fd._emit_and_build(["FACTS", f"{fd._EMIT}/q8_0_from_facts.pl"],
        f'q8_0_op_expr(E), emit_from_fact(E, [mode(tiled_v4(16,256))], "{fd._CACHE}/rl_v4.cu")', "rl_v4")

def bench(cubin, M, K, iters=300):
    nb = K // 32
    Wq = np.zeros(M*K, np.int8); Wd = np.zeros(M*nb, np.float16)
    Xq = np.zeros(K, np.int8); Xd = np.zeros(nb, np.float16)
    def d(a):
        p = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p), a.nbytes); cu.cuMemcpyHtoD_v2(p, a.ctypes.data_as(ctypes.c_void_p), a.nbytes); return p
    dWq, dWd, dXq, dXd = d(Wq), d(Wd), d(Xq), d(Xd)
    dY = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY), M*4)
    fn = fd._func(cubin, "k_q8_0_gemv"); Mi, Ki = ctypes.c_int(M), ctypes.c_int(K)
    args = [dWq, dWd, dXq, dXd, dY, Mi, Ki]
    av = (ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    BM = 16; grid = (M+BM-1)//BM; block = BM*32; shmem = K + nb*2
    for _ in range(20): cu.cuLaunchKernel(fn, grid,1,1, block,1,1, shmem, None, av, None)
    cu.cuCtxSynchronize(); t0 = time.time()
    for _ in range(iters): cu.cuLaunchKernel(fn, grid,1,1, block,1,1, shmem, None, av, None)
    cu.cuCtxSynchronize()
    us = (time.time()-t0)/iters*1e6
    # weight bytes read: M*K int8 + M*nb fp16 scales (the dominant DRAM traffic)
    wbytes = M*K + M*nb*2
    roofline_us = wbytes / PEAK_GBs / 1e3  # bytes / (GB/s) -> ns -> /1e3 = us... = wbytes/192e9*1e6
    roofline_us = wbytes / (PEAK_GBs*1e9) * 1e6
    achieved_gbs = wbytes / (us*1e-6) / 1e9
    return us, wbytes/1e6, roofline_us, achieved_gbs

v4 = build_v4()
print(f"{'shape':12s} {'M':>7} {'K':>5} {'wMB':>7} {'us':>8} {'roof_us':>8} {'%roof':>6} {'GB/s':>6}")
for name, M, K in [("attn", 896, 896), ("ffn_down", 896, 4864), ("ffn_up", 4864, 896), ("vocab", 151936, 896)]:
    us, wMB, roof, gbs = bench(v4, M, K)
    print(f"{name:12s} {M:7d} {K:5d} {wMB:7.1f} {us:8.1f} {roof:8.1f} {100*roof/us:6.1f} {gbs:6.1f}", flush=True)
print(f"\nP4 peak: {PEAK_GBs} GB/s. %roof = roofline_us/actual_us = fraction of peak BW achieved.")
print("High %roof = near bandwidth limit (read fewer bytes is the only lever).")
print("Low %roof  = wasting bandwidth (coalescing/L2/occupancy recoverable).")
