# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""GEMV canonicalization pair-gate (the rmsnorm move): canonical_serial vs v4 tiled MUST be 0-ULP.
Both render from reduction_order(q8_gemv_dp4a,...). + migration delta vs old serial dp4a."""
import sys, ctypes, numpy as np
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
def build(mode,tag): return fd._emit_and_build(["FACTS",f"{fd._EMIT}/q8_0_from_facts.pl"],
    f'q8_0_op_expr(E), emit_from_fact(E, [mode({mode})], "{fd._CACHE}/{tag}.cu")', tag)
def run(cubin,M,K,tiled,BM=16):
    nb=K//32; np.random.seed(8)
    Wq=np.random.randint(-127,128,M*K,dtype=np.int8); Wd=(np.random.randn(M*nb)*0.1).astype(np.float16)
    Xq=np.random.randint(-127,128,K,dtype=np.int8); Xd=(np.random.randn(nb)*0.1).astype(np.float16)
    def d(a):
        p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
    dWq,dWd,dXq,dXd=d(Wq),d(Wd),d(Xq),d(Xd); dY=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY),M*4)
    fn=fd._func(cubin,"k_q8_0_gemv"); Mi,Ki=ctypes.c_int(M),ctypes.c_int(K)
    args=[dWq,dWd,dXq,dXd,dY,Mi,Ki]; argv=(ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
    if tiled: cu.cuLaunchKernel(fn,(M+BM-1)//BM,1,1,BM*32,1,1,K+nb*2,None,argv,None)
    else: cu.cuLaunchKernel(fn,(M+63)//64,1,1,64,1,1,0,None,argv,None)
    cu.cuCtxSynchronize()
    Y=np.empty(M,np.float32); cu.cuMemcpyDtoH_v2(Y.ctypes.data_as(ctypes.c_void_p),dY,M*4); return Y
v4=build("tiled_v4(16,256)","pg_v4"); can=build("canonical_serial_gemv","pg_can"); old=build("dp4a","pg_old")
for name,M,K in [("ffn_down",896,4864),("vocab",151936,896)]:
    y_v4=run(v4,M,K,True); y_can=run(can,M,K,False); y_old=run(old,M,K,False)
    print(f"=== {name} (M={M} K={K}) ===",flush=True)
    print(f"  PAIR-GATE canonical_serial vs v4: {compare_outputs(y_can,y_v4,'bit_exact')}",flush=True)
    mig=int(np.abs(y_can.view(np.uint32).astype(np.int64)-y_old.view(np.uint32).astype(np.int64)).max())
    print(f"  MIGRATION canonical vs old-serial: max_ulp={mig} max_abs={np.abs(y_can-y_old).max():.3e}",flush=True)
