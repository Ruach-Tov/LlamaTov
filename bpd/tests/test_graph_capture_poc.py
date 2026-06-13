# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Proof-of-mechanism: capture a chain of device kernel launches into a CUDA graph via our
ctypes driver path, replay it, and measure per-launch overhead eliminated. De-risks the
cuGraph binding before integrating into forward_pass_resident."""
import sys, time, ctypes, numpy as np
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import fact_dispatch as fd
import dev_residency as dr
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()


cu = fd._libcuda(); fd._ctx()

# A trivial device kernel we can launch many times (add 1.0 in place).
SRC = r'''
extern "C" __global__ void k_inc(float* x, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f; }
'''
cubin = dr._build_inline("inc_test", SRC)
fn = fd._func(cubin, "k_inc")

N = 896
x = dr.DevTensor.from_host(np.zeros(N, np.float32))
Ni = ctypes.c_int(N)
args = [x.ptr, Ni]
argv = (ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
blk=256; grid=(N+blk-1)//blk
NLAUNCH = 336   # ~ our per-token launch count (24 layers * ~14 kernels)

# 1. create a non-default stream
stream = ctypes.c_void_p()
r = cu.cuStreamCreate(ctypes.byref(stream), 0); assert r==0, f"cuStreamCreate={r}"

def eager():
    for _ in range(NLAUNCH):
        cu.cuLaunchKernel(fn, grid,1,1, blk,1,1, 0, stream, argv, None)
    cu.cuStreamSynchronize(stream)

# warmup
eager()
# time eager
t0=time.time()
for _ in range(50): eager()
eager_ms = (time.time()-t0)/50*1000

# 2. CAPTURE the same chain into a graph
graph = ctypes.c_void_p()
CU_STREAM_CAPTURE_MODE_GLOBAL = 0
r = cu.cuStreamBeginCapture_v2(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) if hasattr(cu,"cuStreamBeginCapture_v2") else cu.cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL)
assert r==0, f"BeginCapture={r}"
for _ in range(NLAUNCH):
    cu.cuLaunchKernel(fn, grid,1,1, blk,1,1, 0, stream, argv, None)
r = cu.cuStreamEndCapture(stream, ctypes.byref(graph)); assert r==0, f"EndCapture={r}"

# 3. instantiate
exec_ = ctypes.c_void_p()
if hasattr(cu, "cuGraphInstantiateWithFlags"):
    r = cu.cuGraphInstantiateWithFlags(ctypes.byref(exec_), graph, 0)
else:
    r = cu.cuGraphInstantiate_v2(ctypes.byref(exec_), graph, None, None, 0)
assert r==0, f"Instantiate={r}"

def replay():
    cu.cuGraphLaunch(exec_, stream)
    cu.cuStreamSynchronize(stream)

# warmup + time graph replay
replay()
t0=time.time()
for _ in range(50): replay()
graph_ms = (time.time()-t0)/50*1000

print(f"eager {NLAUNCH} launches:  {eager_ms:.3f} ms", flush=True)
print(f"graph replay (1 launch): {graph_ms:.3f} ms", flush=True)
print(f">>> speedup {eager_ms/graph_ms:.2f}x — per-launch overhead {'ELIMINATED' if graph_ms < eager_ms*0.7 else 'not much'}", flush=True)
print(f">>> CUDA-graph capture/replay WORKS via ctypes" if graph_ms>0 else "FAIL", flush=True)
