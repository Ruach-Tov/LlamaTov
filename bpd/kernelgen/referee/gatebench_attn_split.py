# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Gate + bench split-K v2 (local-max, scores-once) vs original k_attn_decode_masked."""
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
nh,nkv,hd,MAXT=14,2,64,256; scale=1.0/np.sqrt(hd)
orig=fd._func(dr._attn_decode_masked_cubin(MAXT),"k_attn_decode_masked")
def build(NS):
    src=open("/tmp/attn_split2.cu").read()%{"MAXT":MAXT,"NSPLIT":NS}; cub=dr._build_inline(f"attn_split2_{NS}",src)
    return fd._func(cub,"k_attn_decode_split"),fd._func(cub,"k_attn_decode_combine")
def d(a):
    p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p

def run_orig(Q,K,V,dL):
    dOUT=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dOUT),nh*hd*4)
    ao=[Q,K,V,dL,dOUT,ctypes.c_int(hd),ctypes.c_int(nh),ctypes.c_int(nkv),ctypes.c_float(scale)]
    av=(ctypes.c_void_p*9)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in ao])
    cu.cuLaunchKernel(orig,nh,1,1,64,1,1,MAXT*4+1024*4,None,av,None); cu.cuCtxSynchronize()
    o=np.empty(nh*hd,np.float32); cu.cuMemcpyDtoH_v2(o.ctypes.data_as(ctypes.c_void_p),dOUT,nh*hd*4); return o,dOUT

def run_split(sf,cf,NS,Q,K,V,dL):
    dpM=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dpM),nh*NS*4)
    dpZ=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dpZ),nh*NS*4)
    dpO=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dpO),nh*NS*hd*4)
    dOUT=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dOUT),nh*hd*4)
    a1=[Q,K,V,dL,dpM,dpZ,dpO,ctypes.c_int(hd),ctypes.c_int(nh),ctypes.c_int(nkv),ctypes.c_float(scale)]
    av1=(ctypes.c_void_p*11)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a1])
    a2=[dpM,dpZ,dpO,dOUT,ctypes.c_int(hd),ctypes.c_int(nh)]
    av2=(ctypes.c_void_p*6)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a2])
    def launch():
        cu.cuLaunchKernel(sf,nh*NS,1,1,64,1,1,MAXT*4+1024*4,None,av1,None)
        cu.cuLaunchKernel(cf,nh,1,1,hd,1,1,0,None,av2,None)
    launch(); cu.cuCtxSynchronize()
    o=np.empty(nh*hd,np.float32); cu.cuMemcpyDtoH_v2(o.ctypes.data_as(ctypes.c_void_p),dOUT,nh*hd*4)
    return o,launch

for L in [40,64,120]:
    np.random.seed(L)
    Q=d(np.random.randn(nh*hd).astype(np.float32)*0.3); K=d(np.random.randn(MAXT*nkv*hd).astype(np.float32)*0.3); V=d(np.random.randn(MAXT*nkv*hd).astype(np.float32)*0.3); dL=d(np.array([L],np.int32))
    o0,_=run_orig(Q,K,V,dL)
    # time original
    ao=[Q,K,V,dL,_,ctypes.c_int(hd),ctypes.c_int(nh),ctypes.c_int(nkv),ctypes.c_float(scale)]
    avo=(ctypes.c_void_p*9)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in ao])
    for _ in range(20): cu.cuLaunchKernel(orig,nh,1,1,64,1,1,MAXT*4+1024*4,None,avo,None)
    cu.cuCtxSynchronize(); t0=time.time()
    for _ in range(500): cu.cuLaunchKernel(orig,nh,1,1,64,1,1,MAXT*4+1024*4,None,avo,None)
    cu.cuCtxSynchronize(); us_o=(time.time()-t0)/500*1e6
    print(f"L={L}: original={us_o:.2f}us", flush=True)
    for NS in [2,4]:
        sf,cf=build(NS)
        os,launch=run_split(sf,cf,NS,Q,K,V,dL)
        ulp=int(np.abs(o0.view(np.uint32).astype(np.int64)-os.view(np.uint32).astype(np.int64)).max())
        maxabs=np.abs(o0-os).max()
        for _ in range(20): launch()
        cu.cuCtxSynchronize(); t0=time.time()
        for _ in range(500): launch()
        cu.cuCtxSynchronize(); us=(time.time()-t0)/500*1e6
        print(f"  NSPLIT={NS}: {us:.2f}us ({us_o/us:.2f}x)  max_ulp={ulp} max_abs={maxabs:.2e}", flush=True)
