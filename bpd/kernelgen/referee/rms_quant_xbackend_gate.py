# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# Cross-backend gate: oxide k_rms_quant vs CUDA k_rms_quant on IDENTICAL inputs (from oxide dump).
# The true "both lowerings agree with each other from birth" for the rms->quant seam.
import sys, ctypes, os, subprocess, struct
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import numpy as np, fact_dispatch as fd
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

cu=fd._libcuda(); fd._ctx()
DUMP=sys.argv[1] if len(sys.argv)>1 else "/tmp/rq_oxide.bin"
raw=open(DUMP,"rb").read(); off=0
K,=struct.unpack_from("<i",raw,off); off+=4
nb=K//32
x=np.frombuffer(raw,np.float32,K,off).copy(); off+=K*4
nw=np.frombuffer(raw,np.float32,K,off).copy(); off+=K*4
xq_ox=np.frombuffer(raw,np.int8,K,off).copy(); off+=K
xd_ox=np.frombuffer(raw,np.uint16,nb,off).copy(); off+=nb*2
print(f"loaded oxide dump: K={K} nb={nb}  xq_ox[:8]={xq_ox[:8]}",flush=True)
eps=1e-5
# emit + build the CUDA k_rms_quant from the SAME fact (eps must match the oxide's 1e-5)
cuf=os.path.join(fd._CACHE,"gate_rms_quant.cu"); out=cuf.replace(".cu",".cubin")
subprocess.run(["swipl","-q","-g",f'consult("{fd._EMIT}/fused_rms_quant"), emit_fused_rms_quant({eps},"{cuf}"), halt'],capture_output=True,text=True,timeout=60)
if os.path.exists(out): os.remove(out)
r=subprocess.run([f"{fd._CUDA}/bin/nvcc","-arch=sm_61","-cubin","-O3",f"-I{fd._CUDA}/include",cuf,"-o",out],capture_output=True,text=True,env=fd._ENV,timeout=120)
if not os.path.exists(out): print(f"NVCC FAIL:\n{r.stderr[:400]}"); sys.exit(1)
fn=fd._func(out,"k_rms_quant")
def dev(a):
    p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
def devz(n):
    p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),n); return p
dX=dev(x); dNW=dev(nw); dXq=devz(K); dXd=devz(nb*2)
# sig (X,NW,Xq,Xd,K). grid=1 block=256
Ki=ctypes.c_int(K)
a=[dX,dNW,dXq,dXd,Ki]; av=(ctypes.c_void_p*5)(*[ctypes.cast(ctypes.byref(z),ctypes.c_void_p) for z in a])
cu.cuLaunchKernel(fn,1,1,1,256,1,1,256*4,None,av,None); cu.cuCtxSynchronize()
xq_cu=np.empty(K,np.int8); xd_cu=np.empty(nb,np.uint16)
cu.cuMemcpyDtoH_v2(xq_cu.ctypes.data_as(ctypes.c_void_p),dXq,K)
cu.cuMemcpyDtoH_v2(xd_cu.ctypes.data_as(ctypes.c_void_p),dXd,nb*2)
xq_diff=int((xq_ox!=xq_cu).sum()); xd_diff=int((xd_ox!=xd_cu).sum())
print(f"CUDA xq[:8]={xq_cu[:8]}  oxide xq[:8]={xq_ox[:8]}",flush=True)
print(f"Xq differing (oxide vs CUDA): {xq_diff}/{K}   Xd differing: {xd_diff}/{nb}",flush=True)
if xq_diff==0 and xd_diff==0:
    print("*** CROSS-BACKEND 0-ULP: oxide k_rms_quant == CUDA k_rms_quant, BIT-IDENTICAL ***",flush=True)
    print("*** the rms->quant seam: born polyglot, both lowerings agree from ONE fact. ***",flush=True)
else: print(f"DIFFERS",flush=True)
