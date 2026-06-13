# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""The CORRECT gate: silu-fused gate GEMV vs the production path (plain gate GEMV -> k_silu_mul with
u=ones). Both use the SAME device expf, so the silu part must be 0 ULP. (The earlier 2 ULP was my
host np.exp f64 vs device __expf f32 — a reference artifact, not a kernel error.)
k_silu_mul does y[i]=(g[i]/(1+expf(-g[i])))*u[i]. With u=1.0, that's exactly silu(g[i]).
Our fused kernel does Y[row]=acc/(1+__expf(-acc)). Same expr, same device expf -> must be 0 ULP."""
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

def build(mode, tag):
    return fd._emit_and_build(["FACTS", f"{fd._EMIT}/q8_0_from_facts.pl"],
        f'q8_0_op_expr(E), emit_from_fact(E, [mode({mode})], "/tmp/{tag}.cu")', tag)
plain = build("tiled_v4(16,256)", "silu_gate_plain")
silu  = build("tiled_v4_silu(16,256)", "silu_gate_silu")
silumul = dr._build_inline("silu_mul_ref", dr._SILU_MUL_SRC)  # the production k_silu_mul

M,K = 4864, 896; nb=K//32; np.random.seed(7)
Wq = np.random.randint(-8,8,M*K,dtype=np.int8); Wd = (np.random.rand(M*nb).astype(np.float32)*0.02+0.01).astype(np.float16)
Xq = np.random.randint(-8,8,K,dtype=np.int8); Xd = (np.random.rand(nb).astype(np.float32)*0.02+0.01).astype(np.float16)
def d(a):
    p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
dWq,dWd,dXq,dXd=d(Wq),d(Wd),d(Xq),d(Xd)
Mi,Ki=ctypes.c_int(M),ctypes.c_int(K); BM=16; grid=(M+BM-1)//BM; block=BM*32; shmem=K+nb*2

# 1) plain gemv -> g
dG=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dG),M*4)
a1=[dWq,dWd,dXq,dXd,dG,Mi,Ki]; av1=(ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a1])
cu.cuLaunchKernel(fd._func(plain,"k_q8_0_gemv"),grid,1,1,block,1,1,shmem,None,av1,None); cu.cuCtxSynchronize()
# 2) k_silu_mul(g, ones) -> ref
ones = np.ones(M,np.float32); dU=d(ones); dRef=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dRef),M*4)
Ni=ctypes.c_int(M); a2=[dG,dU,dRef,Ni]; av2=(ctypes.c_void_p*4)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a2])
cu.cuLaunchKernel(fd._func(silumul,"k_silu_mul"),(M+255)//256,1,1,256,1,1,0,None,av2,None); cu.cuCtxSynchronize()
ref=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(ref.ctypes.data_as(ctypes.c_void_p),dRef,M*4)
# 3) silu-fused gemv -> fused
dF=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dF),M*4)
a3=[dWq,dWd,dXq,dXd,dF,Mi,Ki]; av3=(ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in a3])
cu.cuLaunchKernel(fd._func(silu,"k_q8_0_gemv_silu"),grid,1,1,block,1,1,shmem,None,av3,None); cu.cuCtxSynchronize()
fused=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(fused.ctypes.data_as(ctypes.c_void_p),dF,M*4)

ulp=int(np.abs(fused.view(np.uint32).astype(np.int64)-ref.view(np.uint32).astype(np.int64)).max())
print(f"silu-fused gemv vs (plain gemv -> k_silu_mul(g,1)): max_ulp={ulp}  {'BIT-EXACT' if ulp==0 else 'DIFFERS'}", flush=True)
print(f"  sample fused={fused[:3]}  ref={ref[:3]}", flush=True)
