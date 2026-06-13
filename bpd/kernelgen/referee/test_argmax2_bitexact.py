# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""W2 argmax: (1) bit-exact vs single-block on random + tie cases, (2) speedup."""
import sys, ctypes, time, numpy as np
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import dev_residency as dr, fact_dispatch as fd
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

cu=fd._libcuda(); fd._ctx(); dr.cu=cu
def run(x, two):
    n=len(x); dr._ARGMAX2=two; dr._ARGMAX_PARTIALS=None; dr._ARGMAX_BUF=None
    dx=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dx),n*4); cu.cuMemcpyHtoD_v2(dx,x.ctypes.data_as(ctypes.c_void_p),n*4)
    out=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(out),4)
    dr.argmax_dev(dx, n, out); cu.cuCtxSynchronize()
    r=ctypes.c_int(); cu.cuMemcpyDtoH_v2(ctypes.byref(r),out,4); return r.value
V=151936
# random
np.random.seed(9); x=(np.random.randn(V)*5).astype(np.float32)
s=run(x,False); t=run(x,True)
print(f"random: single={s} two-stage={t} match={s==t} (np.argmax={int(x.argmax())})",flush=True)
# tie case: two equal maxima, lower index must win
x2=x.copy(); x2[1000]=100.0; x2[50000]=100.0
s2=run(x2,False); t2=run(x2,True)
print(f"tie@1000,50000: single={s2} two-stage={t2} match={s2==t2} (expect 1000, lower idx)",flush=True)
# speed
def bench(two):
    n=V; dr._ARGMAX2=two; dr._ARGMAX_PARTIALS=None; dr._ARGMAX_BUF=None
    dx=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dx),n*4); cu.cuMemcpyHtoD_v2(dx,x.ctypes.data_as(ctypes.c_void_p),n*4)
    out=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(out),4)
    for _ in range(10): dr.argmax_dev(dx,n,out)
    cu.cuCtxSynchronize(); t0=time.time()
    for _ in range(200): dr.argmax_dev(dx,n,out)
    cu.cuCtxSynchronize(); return (time.time()-t0)/200*1e6
us1=bench(False); us2=bench(True)
print(f"SPEED: single-block={us1:.1f}us two-stage={us2:.1f}us speedup={us1/us2:.1f}x",flush=True)
