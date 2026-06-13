# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Gate tiled_v4 (int4 loads) vs tiled (int32 loads): MUST be bit-exact (same arithmetic, only
load width changes). + speedup."""
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
def build(mode,tag): return fd._emit_and_build(["FACTS",f"{fd._EMIT}/q8_0_from_facts.pl"],
    f'q8_0_op_expr(E), emit_from_fact(E, [mode({mode})], "{fd._CACHE}/{tag}.cu")', tag)
def run(cubin,M,K,BM):
    nb=K//32; np.random.seed(4)
    Wq=np.random.randint(-127,128,M*K,dtype=np.int8); Wd=(np.random.randn(M*nb)*0.1).astype(np.float16)
    Xq=np.random.randint(-127,128,K,dtype=np.int8); Xd=(np.random.randn(nb)*0.1).astype(np.float16)
    def d(a):
        p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
    dWq,dWd,dXq,dXd=d(Wq),d(Wd),d(Xq),d(Xd); dY=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY),M*4)
    fn=fd._func(cubin,"k_q8_0_gemv"); Mi,Ki=ctypes.c_int(M),ctypes.c_int(K)
    args=[dWq,dWd,dXq,dXd,dY,Mi,Ki]; argv=(ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
    g=(M+BM-1)//BM; cu.cuLaunchKernel(fn,g,1,1,BM*32,1,1,K+nb*2,None,argv,None); cu.cuCtxSynchronize()
    Y=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(Y.ctypes.data_as(ctypes.c_void_p),dY,M*4)
    def bench():
        for _ in range(10): cu.cuLaunchKernel(fn,g,1,1,BM*32,1,1,K+nb*2,None,argv,None)
        cu.cuCtxSynchronize(); t0=time.time()
        for _ in range(200): cu.cuLaunchKernel(fn,g,1,1,BM*32,1,1,K+nb*2,None,argv,None)
        cu.cuCtxSynchronize(); return (time.time()-t0)/200*1e6
    return Y,bench()
til=build("tiled(16,256,1)","t_til"); v4=build("tiled_v4(16,256)","t_v4")
for name,M,K in [("ffn_down",896,4864),("vocab",151936,896)]:
    yt,ut=run(til,M,K,16); yv,uv=run(v4,M,K,16)
    ulp=int(np.abs(yt.view(np.uint32).astype(np.int64)-yv.view(np.uint32).astype(np.int64)).max())
    print(f"{name}: tiled vs v4 max_ulp={ulp} ({'BIT-EXACT' if ulp==0 else 'DIFFERS'})  tiled={ut:.1f}us v4={uv:.1f}us speedup={ut/uv:.2f}x",flush=True)
