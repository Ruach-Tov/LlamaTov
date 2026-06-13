# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Load-bearing test: does SiLU fold into the gate GEMV's memory shadow?
Compare plain tiled-v4 GEMV (store acc) vs SiLU-epilogue GEMV (store silu(acc)) at the gate shape
(ffn: M=4864, K=896). If SiLU-fused ~= plain, the transcendental is HIDDEN under memory latency."""
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

# plain v4 (store acc) -- hand-written to control exactly, matching the emitter's v4 body
def make_kernel(epi):
    # epi is the store expression in terms of `acc`
    src = f'''
#include <cuda_fp16.h>
extern "C" __global__ void k_gemv(const signed char* Wq, const __half* Wd,
    const signed char* Xq, const __half* Xd, float* Y, int M, int K) {{
  const int BM = 16;
  int warp = threadIdx.x >> 5; int lane = threadIdx.x & 31;
  int row = blockIdx.x*BM + warp; int nblk = K / 32;
  extern __shared__ char smem[];
  signed char* sXq = (signed char*)smem; __half* sXd = (__half*)(smem + K);
  for (int i = threadIdx.x; i < K; i += blockDim.x) sXq[i] = Xq[i];
  for (int i = threadIdx.x; i < nblk; i += blockDim.x) sXd[i] = Xd[i];
  __syncthreads();
  if (row >= M) return;
  float acc = 0.0f;
  for (int b = lane; b < nblk; b += 32) {{
    const int4* wq16 = (const int4*)(Wq + (long)row*K + b*32);
    const int4* xq16 = (const int4*)(sXq + b*32);
    int4 w0=wq16[0],w1=wq16[1],x0=xq16[0],x1=xq16[1]; int isum=0;
    isum=__dp4a(w0.x,x0.x,isum);isum=__dp4a(w0.y,x0.y,isum);isum=__dp4a(w0.z,x0.z,isum);isum=__dp4a(w0.w,x0.w,isum);
    isum=__dp4a(w1.x,x1.x,isum);isum=__dp4a(w1.y,x1.y,isum);isum=__dp4a(w1.z,x1.z,isum);isum=__dp4a(w1.w,x1.w,isum);
    float wd=__half2float(Wd[(long)row*nblk+b]); float xd=__half2float(sXd[b]);
    acc += (wd*xd)*(float)isum;
  }}
  #pragma unroll
  for (int s=16;s>0;s>>=1) acc += __shfl_down_sync(0xffffffff, acc, s);
  if (lane==0) Y[row] = {epi};
}}'''
    return fd._func(dr._build_inline(f"gemv_{abs(hash(epi))%10000}", src), "k_gemv")

def bench(fn, M, K, iters=300):
    nb=K//32; np.random.seed(1)
    Wq=np.zeros(M*K,np.int8); Wd=np.zeros(M*nb,np.float16); Xq=np.zeros(K,np.int8); Xd=np.zeros(nb,np.float16)
    def d(a):
        p=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p),a.nbytes); cu.cuMemcpyHtoD_v2(p,a.ctypes.data_as(ctypes.c_void_p),a.nbytes); return p
    dWq,dWd,dXq,dXd=d(Wq),d(Wd),d(Xq),d(Xd); dY=ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(dY),M*4)
    Mi,Ki=ctypes.c_int(M),ctypes.c_int(K)
    args=[dWq,dWd,dXq,dXd,dY,Mi,Ki]; av=(ctypes.c_void_p*7)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
    BM=16; grid=(M+BM-1)//BM; block=BM*32; shmem=K+nb*2
    for _ in range(20): cu.cuLaunchKernel(fn,grid,1,1,block,1,1,shmem,None,av,None)
    cu.cuCtxSynchronize(); t0=time.time()
    for _ in range(iters): cu.cuLaunchKernel(fn,grid,1,1,block,1,1,shmem,None,av,None)
    cu.cuCtxSynchronize(); return (time.time()-t0)/iters*1e6

plain = make_kernel("acc")
silu  = make_kernel("acc / (1.0f + __expf(-acc))")
for name, M, K in [("gate(4864x896)", 4864, 896), ("vocab(151936x896)", 151936, 896)]:
    up = bench(plain, M, K); us = bench(silu, M, K)
    print(f"{name}: plain={up:.1f}us  silu-epilogue={us:.1f}us  overhead={100*(us-up)/up:+.1f}%  ({'HIDDEN' if (us-up)/up < 0.03 else 'visible'})", flush=True)
