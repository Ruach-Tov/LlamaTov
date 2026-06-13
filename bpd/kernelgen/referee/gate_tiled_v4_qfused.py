# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Gate tiled_v4_qfused vs (k_quant_q8 then tiled_v4): MUST be bit-exact. The fused kernel quantizes
f32 X in-kernel; the unfused path quantizes via k_quant_q8 then runs v4 on the result. Same quant
arithmetic + same v4 float DAG -> same bits."""
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
v4=build("tiled_v4(16,256)","gqf_v4"); qf=build("tiled_v4_qfused(16,256)","gqf_qf")
for name,M,K in [("ffn_down",896,4864),("vocab",151936,896),("attn",896,896)]:
    nb=K//32; np.random.seed(8)
    Wq=np.random.randint(-127,128,M*K,dtype=np.int8); Wd=(np.random.randn(M*nb)*0.1).astype(np.float16)
    X=(np.random.randn(K)*0.5).astype(np.float32)   # f32 activation
    # quantize X via the standalone k_quant_q8 (for the UNFUSED v4 path)
    dX=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dX),K*4); cu.cuMemcpyHtoD_v2(dX,X.ctypes.data_as(ctypes.c_void_p),K*4)
    dXq=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dXq),K); dXd=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dXd),nb*2)
    qfn=fd._func(dr._quant_cubin(),"k_quant_q8"); Ki=ctypes.c_int(K)
    qa=[dX,dXq,dXd,Ki]; qav=(ctypes.c_void_p*4)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in qa])
    cu.cuLaunchKernel(qfn,(nb+63)//64,1,1,64,1,1,0,None,qav,None); cu.cuCtxSynchronize()
    def d(a):
        p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
    dWq,dWd=d(Wq),d(Wd); dY1=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY1),M*4); dY2=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY2),M*4)
    Mi,Ki2=ctypes.c_int(M),ctypes.c_int(K)
    # UNFUSED v4: reads pre-quantized dXq/dXd
    fn1=fd._func(v4,"k_q8_0_gemv"); a1=[dWq,dWd,dXq,dXd,dY1,Mi,Ki2]; av1=(ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a1])
    cu.cuLaunchKernel(fn1,(M+15)//16,1,1,16*32,1,1,K+nb*2,None,av1,None); cu.cuCtxSynchronize()
    Y1=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(Y1.ctypes.data_as(ctypes.c_void_p),dY1,M*4)
    # FUSED: reads f32 dX, quantizes in-kernel
    fn2=fd._func(qf,"k_q8_0_gemv"); a2=[dWq,dWd,dX,dY2,Mi,Ki2]; av2=(ctypes.c_void_p*6)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a2])
    cu.cuLaunchKernel(fn2,(M+15)//16,1,1,16*32,1,1,K+nb*2,None,av2,None); cu.cuCtxSynchronize()
    Y2=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(Y2.ctypes.data_as(ctypes.c_void_p),dY2,M*4)
    ulp=int(np.abs(Y1.view(np.uint32).astype(np.int64)-Y2.view(np.uint32).astype(np.int64)).max())
    print(f"{name}: unfused(quant+v4) vs fused max_ulp={ulp} ({'BIT-EXACT' if ulp==0 else 'DIFFERS'})",flush=True)
