# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import sys, ctypes, os, subprocess
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import numpy as np, fact_dispatch as fd, dev_residency as dr
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

cu=fd._libcuda(); fd._ctx()
K=896; nb=K//32; eps=1e-5
np.random.seed(7)
X=(np.random.randn(K)*2.0).astype(np.float32)
NW=(np.random.randn(K)*0.5+1.0).astype(np.float32)
def dev(a):
    p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
def devz(n):
    p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),n); return p
# --- REFERENCE: standalone k_rmsnorm then k_quant_q8 ---
# k_rmsnorm (serial thread_per_row form, M=1)
rms_cu=os.path.join(fd._CACHE,"chk_rms.cu")
subprocess.run(["swipl","-q","-g",f'consult("{fd._EMIT}/norm_softmax_from_facts"), emit_rmsnorm_cuda({eps},"{rms_cu}"), halt'],capture_output=True,text=True,timeout=60)
def build(cuf,name):
    out=cuf.replace(".cu",".cubin")
    if os.path.exists(out): os.remove(out)
    r=subprocess.run([f"{fd._CUDA}/bin/nvcc","-arch=sm_61","-cubin","-O3",f"-I{fd._CUDA}/include",cuf,"-o",out],capture_output=True,text=True,env=fd._ENV,timeout=120)
    if not os.path.exists(out): print(f"BUILD FAIL {name}:\n{r.stderr[:400]}"); sys.exit(1)
    return fd._func(out,name)
import glob
print("rms emit:", glob.glob(rms_cu) and "ok" or "MISSING", flush=True)
rms_fn=build(rms_cu,"k_rmsnorm")
# quant: use dev_residency's _QUANT_SRC_PAR
quant_fn=fd._func(dr._build_inline("k_quant_q8_chk", dr._QUANT_SRC_PAR), "k_quant_q8")
dX=dev(X); dNW=dev(NW); dO=devz(K*4)
# rmsnorm: launch thread_per_row, M=1 -> grid=1 block=1 (serial). sig (x,w,y,M,N)
Mi,Ni=ctypes.c_int(1),ctypes.c_int(K)
a=[dX,dNW,dO,Mi,Ni]; av=(ctypes.c_void_p*5)(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in a])
cu.cuLaunchKernel(rms_fn,1,1,1,1,1,1,0,None,av,None); cu.cuCtxSynchronize()
# quant: reads dO -> Xq,Xd. sig (X,Xq,Xd,K). one warp per block, wpb=8
dXq_ref=devz(K); dXd_ref=devz(nb*2); Ki=ctypes.c_int(K)
a=[dO,dXq_ref,dXd_ref,Ki]; av=(ctypes.c_void_p*4)(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in a])
cu.cuLaunchKernel(quant_fn,(nb+7)//8,1,1,256,1,1,0,None,av,None); cu.cuCtxSynchronize()
Xq_ref=np.empty(K,np.int8); Xd_ref=np.empty(nb,np.float16)
cu.cuMemcpyDtoH_v2(Xq_ref.ctypes.data_as(ctypes.c_void_p),dXq_ref,K); cu.cuMemcpyDtoH_v2(Xd_ref.ctypes.data_as(ctypes.c_void_p),dXd_ref,nb*2)
# --- FUSED k_rms_quant ---
fq_cu=os.path.join(fd._CACHE,"chk_rmsquant.cu")
subprocess.run(["swipl","-q","-g",f'consult("{fd._EMIT}/fused_rms_quant"), emit_fused_rms_quant({eps},"{fq_cu}"), halt'],capture_output=True,text=True,timeout=60)
print("fused emit:", os.path.exists(fq_cu) and "ok" or "MISSING", flush=True)
fq_fn=build(fq_cu,"k_rms_quant")
dXq=devz(K); dXd=devz(nb*2)
# sig (X,NW,Xq,Xd,K). grid=1 block=256
a=[dX,dNW,dXq,dXd,Ki]; av=(ctypes.c_void_p*5)(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in a])
cu.cuLaunchKernel(fq_fn,1,1,1,256,1,1,0,None,av,None); cu.cuCtxSynchronize()
Xq=np.empty(K,np.int8); Xd=np.empty(nb,np.float16)
cu.cuMemcpyDtoH_v2(Xq.ctypes.data_as(ctypes.c_void_p),dXq,K); cu.cuMemcpyDtoH_v2(Xd.ctypes.data_as(ctypes.c_void_p),dXd,nb*2)
# compare
xq_diff=int((Xq!=Xq_ref).sum())
xd_diff=int((Xd.view(np.uint16)!=Xd_ref.view(np.uint16)).sum())
print(f"Xq differing: {xq_diff}/{K}   Xd differing: {xd_diff}/{nb}", flush=True)
print(f"sample Xq fused[:8]={Xq[:8]} ref[:8]={Xq_ref[:8]}", flush=True)
if xq_diff==0 and xd_diff==0: print("*** FUSED k_rms_quant 0-ULP vs (k_rmsnorm then k_quant_q8) ***", flush=True)
else: print(f"DIFFERS", flush=True)
