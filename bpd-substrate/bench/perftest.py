#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""BPD performance test — three paths, same hardware, same data.

Path A: PyTorch unfused    cuBLAS matmul + ATen bias + ATen relu  (3 launches)
Path B: BPD fused          our kernel via PyTorch cpp_extension   (1 launch)
Path C: BPD unfused        our matmul via cpp_extension + ATen    (3 launches)

All paths run inside PyTorch's CUDA runtime.
Same allocator, same context, same driver.

Exit 0 if BPD fused (B) >= PyTorch unfused (A) at all sizes.
Exit 1 if any regression detected.
"""

import os
import sys
import time
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    assert torch.cuda.is_available(), "CUDA required"
except (ImportError, AssertionError) as e:
    sys.exit(f"error: {e}\n  pip install torch numpy")

# ── BPD kernel source (embedded) ──────────────────────────────
# This is the output of our Prolog epilogue_generator for L2 #76.
# In production, `make build` generates this from BPD facts.

BPD_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void k_mm(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {
    __shared__ float As[64][8], Bs[8][64];
    int ty=threadIdx.y, tx=threadIdx.x, tid=ty*16+tx;
    int row0=blockIdx.y*64+ty*4, col0=blockIdx.x*64+tx*4;
    float c00=0,c01=0,c02=0,c03=0,c10=0,c11=0,c12=0,c13=0;
    float c20=0,c21=0,c22=0,c23=0,c30=0,c31=0,c32=0,c33=0;
    for(int tile=0;tile<(K+7)/8;tile++){int bk=tile*8;
    {int idx=tid;int ar=idx/8,ac=idx%8;As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;idx+=256;ar=idx/8;ac=idx%8;As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;}
    {int idx=tid;int br=idx/64,bc=idx%64;Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;idx+=256;br=idx/64;bc=idx%64;Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;}
    __syncthreads();
    #pragma unroll
    for(int i=0;i<8;i++){float a0=As[ty*4][i],a1=As[ty*4+1][i],a2=As[ty*4+2][i],a3=As[ty*4+3][i];float b0=Bs[i][tx*4],b1=Bs[i][tx*4+1],b2=Bs[i][tx*4+2],b3=Bs[i][tx*4+3];c00+=a0*b0;c01+=a0*b1;c02+=a0*b2;c03+=a0*b3;c10+=a1*b0;c11+=a1*b1;c12+=a1*b2;c13+=a1*b3;c20+=a2*b0;c21+=a2*b1;c22+=a2*b2;c23+=a2*b3;c30+=a3*b0;c31+=a3*b1;c32+=a3*b2;c33+=a3*b3;}
    __syncthreads();}
    if(row0+0<M&&col0+0<N)C[(row0+0)*N+col0+0]=c00;if(row0+0<M&&col0+1<N)C[(row0+0)*N+col0+1]=c01;if(row0+0<M&&col0+2<N)C[(row0+0)*N+col0+2]=c02;if(row0+0<M&&col0+3<N)C[(row0+0)*N+col0+3]=c03;
    if(row0+1<M&&col0+0<N)C[(row0+1)*N+col0+0]=c10;if(row0+1<M&&col0+1<N)C[(row0+1)*N+col0+1]=c11;if(row0+1<M&&col0+2<N)C[(row0+1)*N+col0+2]=c12;if(row0+1<M&&col0+3<N)C[(row0+1)*N+col0+3]=c13;
    if(row0+2<M&&col0+0<N)C[(row0+2)*N+col0+0]=c20;if(row0+2<M&&col0+1<N)C[(row0+2)*N+col0+1]=c21;if(row0+2<M&&col0+2<N)C[(row0+2)*N+col0+2]=c22;if(row0+2<M&&col0+3<N)C[(row0+2)*N+col0+3]=c23;
    if(row0+3<M&&col0+0<N)C[(row0+3)*N+col0+0]=c30;if(row0+3<M&&col0+1<N)C[(row0+3)*N+col0+1]=c31;if(row0+3<M&&col0+2<N)C[(row0+3)*N+col0+2]=c32;if(row0+3<M&&col0+3<N)C[(row0+3)*N+col0+3]=c33;
}

__global__ void k_mm_bias_relu(const float* __restrict__ A,
                                const float* __restrict__ B,
                                const float* __restrict__ bias,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[64][8], Bs[8][64];
    int ty=threadIdx.y, tx=threadIdx.x, tid=ty*16+tx;
    int row0=blockIdx.y*64+ty*4, col0=blockIdx.x*64+tx*4;
    float c00=0,c01=0,c02=0,c03=0,c10=0,c11=0,c12=0,c13=0;
    float c20=0,c21=0,c22=0,c23=0,c30=0,c31=0,c32=0,c33=0;
    for(int tile=0;tile<(K+7)/8;tile++){int bk=tile*8;
    {int idx=tid;int ar=idx/8,ac=idx%8;As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;idx+=256;ar=idx/8;ac=idx%8;As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;}
    {int idx=tid;int br=idx/64,bc=idx%64;Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;idx+=256;br=idx/64;bc=idx%64;Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;}
    __syncthreads();
    #pragma unroll
    for(int i=0;i<8;i++){float a0=As[ty*4][i],a1=As[ty*4+1][i],a2=As[ty*4+2][i],a3=As[ty*4+3][i];float b0=Bs[i][tx*4],b1=Bs[i][tx*4+1],b2=Bs[i][tx*4+2],b3=Bs[i][tx*4+3];c00+=a0*b0;c01+=a0*b1;c02+=a0*b2;c03+=a0*b3;c10+=a1*b0;c11+=a1*b1;c12+=a1*b2;c13+=a1*b3;c20+=a2*b0;c21+=a2*b1;c22+=a2*b2;c23+=a2*b3;c30+=a3*b0;c31+=a3*b1;c32+=a3*b2;c33+=a3*b3;}
    __syncthreads();}
    // FUSED EPILOGUE: bias + relu in registers
    #define STORE_FUSED(r,cl) if(row0+r<M&&col0+cl<N) C[(row0+r)*N+col0+cl]=fmaxf(0.0f, c##r##cl + bias[col0+cl]);
    STORE_FUSED(0,0)STORE_FUSED(0,1)STORE_FUSED(0,2)STORE_FUSED(0,3)
    STORE_FUSED(1,0)STORE_FUSED(1,1)STORE_FUSED(1,2)STORE_FUSED(1,3)
    STORE_FUSED(2,0)STORE_FUSED(2,1)STORE_FUSED(2,2)STORE_FUSED(2,3)
    STORE_FUSED(3,0)STORE_FUSED(3,1)STORE_FUSED(3,2)STORE_FUSED(3,3)
    #undef STORE_FUSED
}

torch::Tensor bpd_mm(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0), K=A.size(1), N=B.size(1);
    auto C = torch::empty({M,N}, A.options());
    dim3 grid((N+63)/64,(M+63)/64), block(16,16);
    k_mm<<<grid,block>>>(A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),M,N,K);
    return C;
}

torch::Tensor bpd_mm_bias_relu(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    int M=A.size(0), K=A.size(1), N=B.size(1);
    auto C = torch::empty({M,N}, A.options());
    dim3 grid((N+63)/64,(M+63)/64), block(16,16);
    k_mm_bias_relu<<<grid,block>>>(A.data_ptr<float>(),B.data_ptr<float>(),
                                    bias.data_ptr<float>(),C.data_ptr<float>(),M,N,K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm", &bpd_mm, "BPD matmul");
    m.def("mm_bias_relu", &bpd_mm_bias_relu, "BPD fused matmul+bias+relu");
}
'''

# ── Timing helper ─────────────────────────────────────────────

N_WARMUP = 20
N_ITER   = 100

def bench(fn):
    for _ in range(N_WARMUP): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / N_ITER

def ulp_max(a, b):
    a_np, b_np = a.cpu().numpy(), b.cpu().numpy()
    ai = a_np.view(np.int32).astype(np.int64)
    bi = b_np.view(np.int32).astype(np.int64)
    bias = np.int64(0x80000000)
    ai = np.where(ai < 0, bias - ai, ai)
    bi = np.where(bi < 0, bias - bi, bi)
    return int(np.abs(ai - bi).max())

# ── Main ──────────────────────────────────────────────────────

def main():
    dev = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability()
    print(f"GPU:  {dev} (sm_{sm[0]}{sm[1]})")
    print()

    print("Compiling BPD kernels as PyTorch extension...")
    from torch.utils.cpp_extension import load_inline
    bpd = load_inline(
        name="bpd_perftest",
        cpp_sources="",
        cuda_sources=[BPD_CUDA],
        extra_cuda_cflags=["-O3", f"-arch=sm_{sm[0]}{sm[1]}",
                           "-Wno-deprecated-gpu-targets"],
        verbose=False,
    )
    print("  OK")
    print()

    sizes = [(512,512,512), (1024,1024,1024), (2048,2048,2048)]
    all_pass = True

    hdr = f"{'SIZE':>14}  {'PATH':>35}  {'us':>8}  {'GFLOPS':>7}  {'ULP':>5}  {'vs A':>7}"
    print(hdr)
    print("=" * len(hdr))

    for M, N, K in sizes:
        A    = torch.randn(M, K, device="cuda")
        B    = torch.randn(K, N, device="cuda")
        bias = torch.randn(N, device="cuda")
        flops = 2.0 * M * N * K

        # Path A: PyTorch unfused (cuBLAS + ATen)
        us_a = bench(lambda: torch.relu(A @ B + bias))
        ref  = torch.relu(A @ B + bias)
        gf_a = flops / (us_a * 1e3)

        # Path B: BPD fused (our kernel inside PyTorch)
        us_b = bench(lambda: bpd.mm_bias_relu(A, B, bias))
        out_b = bpd.mm_bias_relu(A, B, bias)
        ulp_b = ulp_max(ref, out_b)
        gf_b = flops / (us_b * 1e3)

        # Path C: BPD matmul + ATen bias + ATen relu (unfused, isolates matmul)
        us_c = bench(lambda: torch.relu(bpd.mm(A, B) + bias))
        out_c = torch.relu(bpd.mm(A, B) + bias)
        ulp_c = ulp_max(ref, out_c)
        gf_c = flops / (us_c * 1e3)

        # Speedups
        ba = us_a / us_b   # >1 = BPD fused faster
        ca = us_a / us_c   # ≈1 = matmul parity
        bc = us_c / us_b   # >1 = pure fusion win

        label = f"{M}x{N}x{K}"
        print(f"{label:>14}  {'A: PT unfused (cuBLAS+ATen)':>35}  {us_a:>7.0f}  {gf_a:>6.0f}  {'ref':>5}  {'base':>7}")
        print(f"{'':>14}  {'B: BPD FUSED (1 kernel)':>35}  {us_b:>7.0f}  {gf_b:>6.0f}  {ulp_b:>5}  {ba:>6.2f}x")
        print(f"{'':>14}  {'C: BPD mm + ATen bias+relu':>35}  {us_c:>7.0f}  {gf_c:>6.0f}  {ulp_c:>5}  {ca:>6.2f}x")
        print(f"{'':>14}  {'fusion win (B vs C)':>35}  {'':>7}  {'':>6}  {'':>5}  {bc:>6.2f}x")
        print()

        if ba < 0.95:
            print(f"  *** REGRESSION at {label}: BPD fused slower than PyTorch ***")
            all_pass = False

    print("─" * len(hdr))
    print()
    print("PATHS:")
    print("  A = PyTorch eager: torch.relu(A @ B + bias)")
    print("      3 kernel launches: cuBLAS sgemm + ATen add + ATen relu")
    print("  B = BPD fused: bpd.mm_bias_relu(A, B, bias)")
    print("      1 kernel launch: matmul with bias+relu epilogue in registers")
    print("  C = BPD matmul + PyTorch: torch.relu(bpd.mm(A, B) + bias)")
    print("      3 kernel launches: BPD sgemm + ATen add + ATen relu")
    print()
    print("  B/A > 1 = BPD fused FASTER than PyTorch unfused")
    print("  C/A ≈ 1 = BPD matmul matches cuBLAS throughput")
    print("  B/C > 1 = pure fusion speedup (DRAM round-trips eliminated)")
    print()
    print("  All paths run inside PyTorch: same runtime, same CUDA context.")

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
