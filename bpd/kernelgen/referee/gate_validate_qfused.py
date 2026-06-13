# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Validate the reusable gate on the KNOWN-GOOD quant+gemv fusion: it should reproduce XOR=0.
This proves gate_bitexact works as the apply-time gate before wiring it into apply_fusion."""
import sys, ctypes, numpy as np
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
sys.path.insert(0,_os2.path.join(_REPO, "bpd/kernelgen"))
import dev_residency as dr, fact_dispatch as fd
from fusion_gate import gate_bitexact
import os as _os, sys as _sys
import os as _os2
_REPO = _os2.environ.get("LLAMATOV_ROOT") or _os2.path.abspath(_os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), *[".."]*8))

def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

cu=fd._libcuda(); fd._ctx(); dr.cu=cu
M,K=896,4864; nb=K//32
# shared fixed inputs (the gate's job: identical inputs to both paths)
np.random.seed(7)
X=(np.random.randn(K)*2).astype(np.float32)
Wq=np.random.randint(-127,128,size=M*K,dtype=np.int8)
Wd=(np.random.randn(M*nb).astype(np.float32)*0.1).astype(np.float16)
def dptr(arr):
    p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),arr.nbytes)
    cu.cuMemcpyHtoD_v2(p,arr.ctypes.data_as(ctypes.c_void_p),arr.nbytes); return p
dWq,dWd,dX=dptr(Wq),dptr(Wd),dptr(X)
def launch(fn,grid,blk,args,sh=0):
    argv=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
    cu.cuLaunchKernel(fn,grid,1,1,blk,1,1,sh,None,argv,None)
# emit both kernels via the emitter (the gate runs GENERATED kernels)
import subprocess
def emit(opts,out):
    fd._emit_and_build(["FACTS",f"{fd._EMIT}/q8_0_from_facts.pl"],
        f'q8_0_op_expr(E), emit_from_fact(E, [{opts}], "{out}")', out.split("/")[-1].replace(".cu",""))
def unfused():
    dXq=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dXq),K)
    dXd=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dXd),nb*2)
    dY=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY),M*4)
    qfn=fd._func(dr._quant_cubin(),"k_quant_q8")
    Ki=ctypes.c_int(K); launch(qfn,(nb+63)//64,64,[dX,dXq,dXd,Ki])
    gemv=fd._emit_and_build(["FACTS",f"{fd._EMIT}/q8_0_from_facts.pl"],
        f'q8_0_op_expr(E), emit_from_fact(E, [mode(dp4a)], "{fd._CACHE}/q8_gemv_dp4a.cu")',"q8_gemv_dp4a")
    gfn=fd._func(gemv,"k_q8_0_gemv"); Mi=ctypes.c_int(M)
    launch(gfn,(M+63)//64,64,[dWq,dWd,dXq,dXd,dY,Mi,Ki]); cu.cuCtxSynchronize()
    Y=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(Y.ctypes.data_as(ctypes.c_void_p),dY,M*4); return Y
def fused():
    dY=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY),M*4)
    ff=fd._emit_and_build(["FACTS",f"{fd._EMIT}/q8_0_from_facts.pl"],
        f'q8_0_op_expr(E), emit_from_fact(E, [prologue(quant)], "{fd._CACHE}/q8_gemv_qfused.cu")',"q8_gemv_qfused")
    ffn=fd._func(ff,"k_q8_0_gemv_qfused"); Mi=ctypes.c_int(M); Ki=ctypes.c_int(K)
    launch(ffn,(M+63)//64,64,[dWq,dWd,dX,dY,Mi,Ki],sh=K+nb*2); cu.cuCtxSynchronize()
    Y=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(Y.ctypes.data_as(ctypes.c_void_p),dY,M*4); return Y
res=gate_bitexact(unfused, fused, M, equiv_class='bit_exact')
print(f"GATE on quant+gemv: {res}", flush=True)
print(f">>> gate verdict: {'PASS (commit the fusion)' if res.passed else 'FAIL (reject)'}", flush=True)
