# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Cross-backend pair-gate: oxide q8_0 GEMV vs the CUDA emitter's canonical_serial GEMV, IDENTICAL
inputs (read from the oxide dump), bit-for-bit compare. The true thesis claim: same fact, two
backends (Rust/cuda-oxide vs CUDA/nvcc), 0 ULP."""
import sys, struct, os, ctypes
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import numpy as np
import fact_dispatch as fd
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()


DUMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/q8_xcheck.bin"
raw = open(DUMP, "rb").read()
off = 0
M, K = struct.unpack_from("<ii", raw, off); off += 8
nb = K // 32
wq = np.frombuffer(raw, np.int8, M*K, off).copy(); off += M*K
wd_h = np.frombuffer(raw, np.uint16, M*nb, off).copy(); off += M*nb*2   # fp16 bits
xq = np.frombuffer(raw, np.int8, K, off).copy(); off += K
xd_h = np.frombuffer(raw, np.uint16, nb, off).copy(); off += nb*2       # fp16 bits
y_oxide = np.frombuffer(raw, np.float32, M, off).copy(); off += M*4
print(f"loaded: M={M} K={K} nb={nb}  oxide y[0:3]={y_oxide[0:3]}", flush=True)

# emit the CUDA canonical_serial GEMV from the SAME fact
cu = fd._libcuda(); fd._ctx()
cache = fd._CACHE
cuf = os.path.join(cache, "xcheck_q8_gemv_canonical.cu")
import subprocess
# drive the emitter: q8_0_op_expr(E) -> emit with mode(canonical_serial_gemv) (the fact-driven path)
emit_goal = f'q8_0_op_expr(E), emit_from_fact(E, [mode(canonical_serial_gemv)], "{cuf}")'
r = subprocess.run(["swipl","-q","-g",
    f'consult("{fd._EMIT}/q8_0_from_facts.pl"), ({emit_goal} -> true ; true), halt'],
    capture_output=True, text=True, timeout=60)
print(f"emit: {r.stdout.strip()[:120]} {r.stderr.strip()[:120]}", flush=True)
if not os.path.exists(cuf):
    print("EMIT FAILED — trying direct emit_q8_0_gemv_canonical_serial", flush=True)
    r = subprocess.run(["swipl","-q","-g",
        f'consult("{fd._EMIT}/q8_0_from_facts.pl"), emit_q8_0_gemv_canonical_serial("{cuf}"), halt'],
        capture_output=True, text=True, timeout=60)
    print(f"  retry: {r.stdout.strip()[:120]} {r.stderr.strip()[:120]}", flush=True)

# build the cubin
out = os.path.join(cache, "xcheck_q8_gemv_canonical.cubin")
if os.path.exists(out): os.remove(out)
rb = subprocess.run([f"{fd._CUDA}/bin/nvcc","-arch=sm_61","-cubin","-O3",
    f"-I{fd._CUDA}/include", cuf, "-o", out], capture_output=True, text=True, env=fd._ENV, timeout=120)
if not os.path.exists(out):
    print(f"NVCC FAILED:\n{rb.stderr[:500]}", flush=True); sys.exit(1)
fn = fd._func(out, "k_q8_0_gemv")

# upload identical inputs (scales as raw fp16 bits -> __half)
def dev(arr):
    p = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p), arr.nbytes)
    cu.cuMemcpyHtoD_v2(p, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes); return p
d_wq, d_wd, d_xq, d_xd = dev(wq), dev(wd_h), dev(xq), dev(xd_h)
d_y = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(d_y), M*4)
# k_q8_0_gemv(Wq, Wd, Xq, Xd, Y, M, K)
Mi, Ki = ctypes.c_int(M), ctypes.c_int(K)
args = [d_wq, d_wd, d_xq, d_xd, d_y, Mi, Ki]
argv = (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
blk = 128; grid = (M + blk - 1)//blk
cu.cuLaunchKernel(fn, grid,1,1, blk,1,1, 0, None, argv, None)
cu.cuCtxSynchronize()
y_cuda = np.empty(M, np.float32)
cu.cuMemcpyDtoH_v2(y_cuda.ctypes.data_as(ctypes.c_void_p), d_y, M*4)

# bit-for-bit compare
ob = y_oxide.view(np.uint32).astype(np.int64)
cb = y_cuda.view(np.uint32).astype(np.int64)
ulp = np.abs(ob - cb)
max_ulp = int(ulp.max()); n_diff = int((ulp != 0).sum())
print(f"\nCUDA y[0:3]  = {y_cuda[0:3]}", flush=True)
print(f"oxide y[0:3] = {y_oxide[0:3]}", flush=True)
print(f"max_ulp(oxide vs CUDA) = {max_ulp}   rows differing = {n_diff}/{M}", flush=True)
print(f"max_abs = {np.abs(y_oxide - y_cuda).max():.3e}", flush=True)
if max_ulp == 0:
    print("\n*** CROSS-BACKEND 0-ULP: oxide-Rust q8_0 GEMV == CUDA-nvcc q8_0 GEMV, BIT-IDENTICAL ***", flush=True)
    print("*** same reduction_order FACT, two backends, zero ULP. THESIS PROVEN. ***", flush=True)
else:
    print(f"\nDIFFERS by {max_ulp} ULP — orders not yet matched", flush=True)
