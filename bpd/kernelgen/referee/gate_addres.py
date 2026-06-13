# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Gate tiled_v4_addres vs (tiled_v4 then k_add): MUST be bit-exact. Fused: Y=gemv(x)+resid in
one kernel. Unfused: v4 -> Y0, then k_add(Y0, resid). Same v4 DAG + same f32 add -> same bits."""
import sys, ctypes, numpy as np
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import dev_residency as dr, fact_dispatch as fd
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

cu=fd._libcuda(); fd._ctx(); dr.cu=cu
def build(mode,tag,name): return fd._emit_and_build(["FACTS",f"{fd._EMIT}/q8_0_from_facts.pl"],
    f'q8_0_op_expr(E), emit_from_fact(E, [mode({mode})], "{fd._CACHE}/{tag}.cu")', tag)
v4=build("tiled_v4(16,256)","ga_v4","k_q8_0_gemv")
ar=build("tiled_v4_addres(16,256)","ga_ar","k_q8_0_gemv_addres")
# k_add kernel source (the existing residual add y=a+b)
addsrc=r'extern "C" __global__ void k_add(const float* a, const float* b, float* y, int N){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) y[i]=a[i]+b[i];}'
addcub=dr._build_inline("k_add_test", addsrc)
for name,M,K in [("o_proj",896,896),("down_proj",896,4864)]:
    nb=K//32; np.random.seed(8)
    Wq=np.random.randint(-127,128,M*K,dtype=np.int8); Wd=(np.random.randn(M*nb)*0.1).astype(np.float16)
    Xq=np.random.randint(-127,128,K,dtype=np.int8); Xd=(np.random.randn(nb)*0.1).astype(np.float16)
    Resid=(np.random.randn(M)*0.3).astype(np.float32)
    def d(a):
        p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
    dWq,dWd,dXq,dXd,dR=d(Wq),d(Wd),d(Xq),d(Xd),d(Resid)
    Mi,Ki=ctypes.c_int(M),ctypes.c_int(K)
    # UNFUSED: v4 -> Y0, then k_add(Y0, Resid)
    dY0=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY0),M*4)
    fn1=fd._func(v4,"k_q8_0_gemv"); a1=[dWq,dWd,dXq,dXd,dY0,Mi,Ki]; av1=(ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a1])
    cu.cuLaunchKernel(fn1,(M+15)//16,1,1,16*32,1,1,K+nb*2,None,av1,None); cu.cuCtxSynchronize()
    dY1=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY1),M*4)
    fa=fd._func(addcub,"k_add"); Ni=ctypes.c_int(M); aa=[dY0,dR,dY1,Ni]; ava=(ctypes.c_void_p*4)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in aa])
    cu.cuLaunchKernel(fa,(M+63)//64,1,1,64,1,1,0,None,ava,None); cu.cuCtxSynchronize()
    Y_unf=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(Y_unf.ctypes.data_as(ctypes.c_void_p),dY1,M*4)
    # FUSED: tiled_v4_addres (Y = gemv + resid)
    dY2=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY2),M*4)
    fn2=fd._func(ar,"k_q8_0_gemv_addres"); a2=[dWq,dWd,dXq,dXd,dR,dY2,Mi,Ki]; av2=(ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a2])
    cu.cuLaunchKernel(fn2,(M+15)//16,1,1,16*32,1,1,K+nb*2,None,av2,None); cu.cuCtxSynchronize()
    Y_fus=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(Y_fus.ctypes.data_as(ctypes.c_void_p),dY2,M*4)
    ulp=int(np.abs(Y_unf.view(np.uint32).astype(np.int64)-Y_fus.view(np.uint32).astype(np.int64)).max())
    print(f"{name}: (v4+k_add) vs addres-fused max_ulp={ulp} ({'BIT-EXACT' if ulp==0 else 'DIFFERS'})",flush=True)
