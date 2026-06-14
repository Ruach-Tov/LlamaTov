#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""kv_q8_kernel.py — Milestone (1): the q8_quantize -> q8_dequant ROUND-TRIP KERNEL, on GPU.

Builds the K/V Q8_0 round-trip as real device kernels (the engine's trusted k_quant_q8 quantize +
a matching k_dequant_q8) and verifies on ACTUAL hardware that the round-trip is bounded by d/2 — the
same contract the CPU referee proved. This makes kv_quantize_q8 a transform that RUNS, not just a graph
rewrite. The quantize is byte-identical to dev_residency.k_quant_q8; the dequant inverts it exactly.
"""
import sys, os, ctypes, subprocess
def _root():
    p = os.path.dirname(os.path.abspath(__file__))
    while p != "/" and os.path.basename(p) != "bpd":
        p = os.path.dirname(p)
    return p if os.path.basename(p) == "bpd" else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root()); sys.path.insert(0, os.path.join(_root(), "lib"))
import numpy as np, fact_dispatch as fd
cu = fd._libcuda(); fd._ctx()

# The round-trip kernels. Quantize = the engine's trusted scheme (amax/127, fp16 scale, rint).
# Dequant = multiply int8 by the fp16-rounded scale. One warp per Q8_0 block (32 lanes).
SRC = r'''#include <cuda_fp16.h>
extern "C" __global__ void k_quant_q8(const float* X, signed char* Xq, __half* Xd, int K) {
  int nb = K/32; int b = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
  if (b >= nb) return; int lane = threadIdx.x & 31;
  float a = fabsf(X[b*32+lane]);
  for (int o=16;o>0;o>>=1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));  // order-insensitive max
  float d = (a > 0.0f) ? a/127.0f : 1.0f; __half dh = __float2half(d);
  if (lane==0) Xd[b] = dh;
  float dq = __half2float(dh);
  int q = (int)rintf(X[b*32+lane]/dq); q = q<-127?-127:(q>127?127:q);
  Xq[b*32+lane] = (signed char)q;
}
extern "C" __global__ void k_dequant_q8(const signed char* Xq, const __half* Xd, float* Y, int K) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= K) return;
  Y[i] = (float)Xq[i] * __half2float(Xd[i>>5]);   // block scale = Xd[i/32]
}
'''

def build():
    cuf = "/tmp/fact_cubins/kv_q8.cu"; out = cuf.replace(".cu", ".cubin")
    open(cuf, "w").write(SRC)
    if os.path.exists(out): os.remove(out)
    r = subprocess.run([f"{fd._CUDA}/bin/nvcc", "-arch=sm_61", "-cubin", "-O3",
                        f"-I{fd._CUDA}/include", cuf, "-o", out],
                       capture_output=True, text=True, env=fd._ENV, timeout=120)
    if not os.path.exists(out):
        print("BUILD FAIL:", r.stderr[:400]); sys.exit(1)
    return fd._func(out, "k_quant_q8"), fd._func(out, "k_dequant_q8")

def dev(a):
    p = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p), a.nbytes)
    cu.cuMemcpyHtoD_v2(p, a.ctypes.data_as(ctypes.c_void_p), a.nbytes); return p
def devz(n):
    p = ctypes.c_void_p(); cu.cuMemAlloc_v2(ctypes.byref(p), n); return p

def roundtrip_gpu(fq, fdq, x):
    K = x.shape[0]; nb = K // 32
    dX = dev(x); dXq = devz(K); dXd = devz(nb*2); dY = devz(K*4)
    Kc = ctypes.c_int(K)
    # quantize: one warp per block, 8 warps/block
    aq = (ctypes.c_void_p*4)(*[ctypes.cast(ctypes.byref(z), ctypes.c_void_p) for z in (dX,dXq,dXd,Kc)])
    grid_q = (nb + 7)//8
    cu.cuLaunchKernel(fq, grid_q,1,1, 256,1,1, 0, None, aq, None)
    # dequant: one thread per element
    adq = (ctypes.c_void_p*4)(*[ctypes.cast(ctypes.byref(z), ctypes.c_void_p) for z in (dXq,dXd,dY,Kc)])
    cu.cuLaunchKernel(fdq, (K+255)//256,1,1, 256,1,1, 0, None, adq, None)
    cu.cuCtxSynchronize()
    y = np.empty(K, np.float32); cu.cuMemcpyDtoH_v2(y.ctypes.data_as(ctypes.c_void_p), dY, K*4)
    xd = np.empty(nb, np.float16); cu.cuMemcpyDtoH_v2(xd.ctypes.data_as(ctypes.c_void_p), dXd, nb*2)
    for d in (dX,dXq,dXd,dY): cu.cuMemFree_v2(d)
    return y, xd.astype(np.float32)

def main():
    fq, fdq = build()
    np.random.seed(11)
    print("=== Milestone (1): q8 quantize->dequant ROUND-TRIP KERNEL on GPU ===\n")
    all_ok = True
    for label, sig in [("sigma=0.3",0.3),("sigma=1.0",1.0),("sigma=3.0",3.0)]:
        oks, maxerrs, snrs = [], [], []
        for _ in range(100):
            x = (np.random.randn(128)*sig).astype(np.float32)  # qwen2 K/V = 128 elems
            y, dh = roundtrip_gpu(fq, fdq, x)
            err = np.abs(y - x)
            bound = np.repeat(dh, 32)/2.0 + 1e-6
            oks.append(bool((err <= bound).all())); maxerrs.append(float(err.max()))
            snrs.append(10*np.log10((x**2).mean()/((err**2).mean()+1e-30)))
        passed = all(oks); all_ok = all_ok and passed
        print(f"  {label:10}: |err|<=d/2 on {sum(oks)}/100  worst={max(maxerrs):.5f}  "
              f"SNR={np.mean(snrs):.1f}dB  {'PASS' if passed else 'FAIL'}")
    # cross-check: GPU round-trip must match the CPU referee bit-for-bit (same arithmetic)
    x = (np.random.randn(128)*1.0).astype(np.float32)
    yg, _ = roundtrip_gpu(fq, fdq, x)
    # CPU reference
    xb = x.reshape(-1,32); amax = np.abs(xb).max(1,keepdims=True)
    d = np.where(amax>0, amax/127,1).astype(np.float32); dh = d.astype(np.float16).astype(np.float32)
    qc = np.round(xb/dh).clip(-127,127); yc = (qc*dh).reshape(-1)
    match = int(np.abs(yg-yc).max() == 0.0)
    print(f"\n  GPU round-trip == CPU referee arithmetic, bit-for-bit: {'PASS' if match else 'FAIL'}")
    print(f"\n{'KERNEL PASS' if all_ok and match else 'KERNEL FAIL'}: the q8 round-trip RUNS on GPU, "
          f"bounded by d/2, matching the referee.")
    sys.exit(0 if (all_ok and match) else 1)

if __name__ == "__main__":
    main()
