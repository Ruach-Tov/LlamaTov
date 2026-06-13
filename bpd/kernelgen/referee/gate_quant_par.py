# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""W1 gate: parallel quant vs serial quant. Bit-exact (XOR=0) since amax is order-insensitive?
+ speedup."""
import sys, ctypes, time, numpy as np
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
sys.path.insert(0,_os2.path.join(_REPO, "bpd/kernelgen"))
import dev_residency as dr, fact_dispatch as fd
from fusion_gate import compare_outputs
import os as _os, sys as _sys
import os as _os2
_REPO = _os2.environ.get("LLAMATOV_ROOT") or _os2.path.abspath(_os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), *[".."]*8))

def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

cu=fd._libcuda(); fd._ctx(); dr.cu=cu
for K in [896, 4864]:
    nb=K//32
    np.random.seed(7)
    X=(np.random.randn(K)*2).astype(np.float32)
    dX=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dX),K*4); cu.cuMemcpyHtoD_v2(dX,X.ctypes.data_as(ctypes.c_void_p),K*4)
    def run(parallel):
        dXq=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dXq),K)
        dXd=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dXd),nb*2)
        Ki=ctypes.c_int(K)
        args=[dX,dXq,dXd,Ki]; argv=(ctypes.c_void_p*4)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
        if parallel:
            fn=fd._func(dr._quant_cubin_par(),"k_quant_q8")
            warps_per_block=8; block=warps_per_block*32; grid=(nb+warps_per_block-1)//warps_per_block
            cu.cuLaunchKernel(fn,grid,1,1,block,1,1,0,None,argv,None)
        else:
            fn=fd._func(dr._quant_cubin(),"k_quant_q8")
            cu.cuLaunchKernel(fn,(nb+63)//64,1,1,64,1,1,0,None,argv,None)
        cu.cuCtxSynchronize()
        Xq=np.empty(K,np.int8); Xd=np.empty(nb,np.float16)
        cu.cuMemcpyDtoH_v2(Xq.ctypes.data_as(ctypes.c_void_p),dXq,K)
        cu.cuMemcpyDtoH_v2(Xd.ctypes.data_as(ctypes.c_void_p),dXd,nb*2)
        return Xq,Xd
    qs,ds=run(False); qp,dp=run(True)
    q_mism=int((qs!=qp).sum()); d_mism=int((ds.view(np.uint16)!=dp.view(np.uint16)).sum())
    print(f"K={K}: Xq mismatches={q_mism}/{K}  Xd(scale) mismatches={d_mism}/{nb}  -> {'BIT_EXACT' if q_mism==0 and d_mism==0 else 'DIFFERS'}",flush=True)
    # speed
    def bench(parallel):
        dXq=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dXq),K); dXd=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dXd),nb*2)
        Ki=ctypes.c_int(K); args=[dX,dXq,dXd,Ki]; argv=(ctypes.c_void_p*4)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
        if parallel:
            fn=fd._func(dr._quant_cubin_par(),"k_quant_q8"); wpb=8; bl=wpb*32; gr=(nb+wpb-1)//wpb
            L=lambda: cu.cuLaunchKernel(fn,gr,1,1,bl,1,1,0,None,argv,None)
        else:
            fn=fd._func(dr._quant_cubin(),"k_quant_q8"); L=lambda: cu.cuLaunchKernel(fn,(nb+63)//64,1,1,64,1,1,0,None,argv,None)
        for _ in range(10): L()
        cu.cuCtxSynchronize(); t0=time.time()
        for _ in range(500): L()
        cu.cuCtxSynchronize(); return (time.time()-t0)/500*1e6
    us_s=bench(False); us_p=bench(True)
    print(f"        serial={us_s:.1f}us  parallel={us_p:.1f}us  speedup={us_s/us_p:.1f}x",flush=True)
