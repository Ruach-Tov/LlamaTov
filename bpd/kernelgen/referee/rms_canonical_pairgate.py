# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Pair-gate (Bocher's requirement): canonical_serial vs block_row must be 0-ULP. Both emitted
from the one reduction_order spec. Also measure the one-time migration delta (canonical vs the
OLD torch-matched serial = the bounded reclassification artifact)."""
import sys, ctypes, numpy as np, torch
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
N=896; eps=1e-6
np.random.seed(7)
x=(np.random.randn(N)*2).astype(np.float32); w=(np.random.randn(N)*0.5+1).astype(np.float32)
def gen(mode,tag):
    opt = f", mode({mode})" if mode else ""
    return fd._emit_and_build(["FACTS", f"{fd._EMIT}/norm_softmax_from_facts.pl"],
        f'op_expr(bpd_rmsnorm, R), emit_from_fact(R, [eps({eps}){opt}], "{fd._CACHE}/{tag}.cu")', tag)
dW=dr._dev_const(w); dx=dr.DevTensor.from_host(x)
def launch(cubin, blockrow):
    fn=fd._func(cubin,"k_rmsnorm"); y=dr._empty_dev((N,))
    Mi,Ni=ctypes.c_int(1),ctypes.c_int(N)
    args=[dx.ptr,dW,y.ptr,Mi,Ni]
    argv=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
    if blockrow: cu.cuLaunchKernel(fn,1,1,1,256,1,1,256*4,None,argv,None)
    else: cu.cuLaunchKernel(fn,1,1,1,1,1,1,0,None,argv,None)
    cu.cuCtxSynchronize()
    o=np.empty(N,np.float32); cu.cuMemcpyDtoH_v2(o.ctypes.data_as(ctypes.c_void_p),y.ptr,N*4); return o
y_br=launch(gen("block_row","rms_br_g"), True)
y_can=launch(gen("canonical_serial","rms_can_g"), False)
y_old=launch(gen(None,"rms_serial_g"), False)  # old torch-matched serial
y_torch=fd.rms_norm_fact(torch.from_numpy(x).reshape(1,-1),torch.from_numpy(w),eps).reshape(-1).numpy()
print("=== PAIR-GATE (Bocher): canonical_serial vs block_row ===",flush=True)
print(f"  {compare_outputs(y_can,y_br,'bit_exact')}",flush=True)
print("=== MIGRATION DELTA (one-time, bounded): canonical vs OLD torch-serial ===",flush=True)
r=compare_outputs(y_old,y_can,('tolerance',1e-3))
print(f"  canonical vs old-serial: mismatches={r.mismatches}/{N} max_ulp={r.max_ulp} max_abs={r.max_abs:.3e}",flush=True)
print(f"  old-serial vs torch: max_ulp={int(np.abs(y_old.view(np.uint32).astype(np.int64)-y_torch.view(np.uint32).astype(np.int64)).max())}",flush=True)
