# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Benchmark BPD-fused vs PyTorch-unfused on L2 chains.

Three paths, same hardware, same data:
  Path A: PyTorch unfused    — cuBLAS matmul + ATen elementwise (N launches)
  Path B: PyTorch fused      — loads our BPD-generated fused .cu via cpp_extension
  Path C: BPD native         — our .so via ctypes (same kernel as Path B)

Path B is the airtight comparison: PyTorch runs OUR kernel.
Same runtime. Same driver. Same CUDA context.
Only the kernel code differs.

If Path B > Path A: fusion wins (our kernel is faster in PyTorch's runtime).
If Path B ≈ Path C: no overhead from PyTorch's launcher vs ctypes.

Usage:
    BPD_BUILD_DIR=build python3 bench/bench_fusion.py
    (or: make bench)
"""

import os
import sys
import time
import numpy as np

HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    if torch.cuda.is_available():
        HAS_TORCH = True
except ImportError:
    pass

FUSED_CUDA_SRC = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

// BPD-generated fused kernel: matmul + bias + relu (L2 #76)
// Identical to what our Prolog epilogue_generator emits.
__global__ void k_mm_bias_relu(const float* __restrict__ A,
                                const float* __restrict__ B,
                                const float* __restrict__ bias,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[64][8], Bs[8][64];
    int ty=threadIdx.y, tx=threadIdx.x, tid=ty*16+tx;
    int row0=blockIdx.y*64+ty*4, col0=blockIdx.x*64+tx*4;
    float c[4][4]; for(int r=0;r<4;r++) for(int cl=0;cl<4;cl++) c[r][cl]=0;
    for(int tile=0;tile<(K+7)/8;tile++){
        int bk=tile*8;
        {int idx=tid;int ar=idx/8,ac=idx%8;
         As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;
         idx+=256;ar=idx/8;ac=idx%8;
         As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;}
        {int idx=tid;int br=idx/64,bc=idx%64;
         Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;
         idx+=256;br=idx/64;bc=idx%64;
         Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;}
        __syncthreads();
        #pragma unroll
        for(int i=0;i<8;i++){
            float a0=As[ty*4][i],a1=As[ty*4+1][i],a2=As[ty*4+2][i],a3=As[ty*4+3][i];
            float b0=Bs[i][tx*4],b1=Bs[i][tx*4+1],b2=Bs[i][tx*4+2],b3=Bs[i][tx*4+3];
            c[0][0]+=a0*b0;c[0][1]+=a0*b1;c[0][2]+=a0*b2;c[0][3]+=a0*b3;
            c[1][0]+=a1*b0;c[1][1]+=a1*b1;c[1][2]+=a1*b2;c[1][3]+=a1*b3;
            c[2][0]+=a2*b0;c[2][1]+=a2*b1;c[2][2]+=a2*b2;c[2][3]+=a2*b3;
            c[3][0]+=a3*b0;c[3][1]+=a3*b1;c[3][2]+=a3*b2;c[3][3]+=a3*b3;
        }
        __syncthreads();
    }
    // FUSED EPILOGUE: bias + relu in registers, single DRAM write
    for(int r=0;r<4;r++) for(int cl=0;cl<4;cl++)
        if(row0+r<M&&col0+cl<N)
            C[(row0+r)*N+col0+cl]=fmaxf(0.0f, c[r][cl]+bias[col0+cl]);
}

// Same kernel WITHOUT epilogue (unfused matmul only)
__global__ void k_mm_raw(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    __shared__ float As[64][8], Bs[8][64];
    int ty=threadIdx.y, tx=threadIdx.x, tid=ty*16+tx;
    int row0=blockIdx.y*64+ty*4, col0=blockIdx.x*64+tx*4;
    float c[4][4]; for(int r=0;r<4;r++) for(int cl=0;cl<4;cl++) c[r][cl]=0;
    for(int tile=0;tile<(K+7)/8;tile++){
        int bk=tile*8;
        {int idx=tid;int ar=idx/8,ac=idx%8;
         As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;
         idx+=256;ar=idx/8;ac=idx%8;
         As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;}
        {int idx=tid;int br=idx/64,bc=idx%64;
         Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;
         idx+=256;br=idx/64;bc=idx%64;
         Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;}
        __syncthreads();
        #pragma unroll
        for(int i=0;i<8;i++){
            float a0=As[ty*4][i],a1=As[ty*4+1][i],a2=As[ty*4+2][i],a3=As[ty*4+3][i];
            float b0=Bs[i][tx*4],b1=Bs[i][tx*4+1],b2=Bs[i][tx*4+2],b3=Bs[i][tx*4+3];
            c[0][0]+=a0*b0;c[0][1]+=a0*b1;c[0][2]+=a0*b2;c[0][3]+=a0*b3;
            c[1][0]+=a1*b0;c[1][1]+=a1*b1;c[1][2]+=a1*b2;c[1][3]+=a1*b3;
            c[2][0]+=a2*b0;c[2][1]+=a2*b1;c[2][2]+=a2*b2;c[2][3]+=a2*b3;
            c[3][0]+=a3*b0;c[3][1]+=a3*b1;c[3][2]+=a3*b2;c[3][3]+=a3*b3;
        }
        __syncthreads();
    }
    for(int r=0;r<4;r++) for(int cl=0;cl<4;cl++)
        if(row0+r<M&&col0+cl<N) C[(row0+r)*N+col0+cl]=c[r][cl];
}

// PyTorch-callable wrappers
torch::Tensor bpd_mm_bias_relu(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N+63)/64, (M+63)/64), block(16, 16);
    k_mm_bias_relu<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                     bias.data_ptr<float>(), C.data_ptr<float>(),
                                     M, N, K);
    return C;
}

torch::Tensor bpd_mm_raw(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 grid((N+63)/64, (M+63)/64), block(16, 16);
    k_mm_raw<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                               C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_bias_relu", &bpd_mm_bias_relu, "BPD fused matmul+bias+relu");
    m.def("mm_raw", &bpd_mm_raw, "BPD matmul (unfused)");
}
'''

N_WARMUP = 20
N_ITER = 100


def load_bpd_extension():
    """JIT-compile our fused kernel as a PyTorch extension."""
    try:
        from torch.utils.cpp_extension import load_inline
        sm = torch.cuda.get_device_capability()
        arch = f"-arch=sm_{sm[0]}{sm[1]}"
        mod = load_inline(
            name="bpd_fused",
            cpp_sources="",
            cuda_sources=[FUSED_CUDA_SRC],
            extra_cuda_cflags=["-O3", arch, "-Wno-deprecated-gpu-targets"],
            verbose=False,
        )
        return mod
    except Exception as e:
        print(f"  (JIT compile failed: {e})")
        return None


def time_fn(fn, n_warmup=N_WARMUP, n_iter=N_ITER):
    """Time a CUDA function, return microseconds per call."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_iter


def ulp_distance(a, b):
    """Max ULP distance between two float32 tensors."""
    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    ai = a_np.view(np.int32).astype(np.int64)
    bi = b_np.view(np.int32).astype(np.int64)
    bias = np.int64(0x80000000)
    ai = np.where(ai < 0, bias - ai, ai)
    bi = np.where(bi < 0, bias - bi, bi)
    return int(np.abs(ai - bi).max())


def main():
    if not HAS_TORCH:
        sys.exit("error: torch with CUDA required. pip install torch numpy")

    dev = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability()
    print(f"GPU: {dev} (sm_{sm[0]}{sm[1]})")
    print()

    print("Loading BPD fused kernels as PyTorch extension...")
    bpd = load_bpd_extension()
    if not bpd:
        sys.exit("Failed to load BPD extension.")
    print("  OK")
    print()

    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

    print(f"{'SIZE':>14}  {'PATH':>30}  {'us':>8}  {'GFLOPS':>8}  {'ULP':>6}  {'speedup':>8}")
    print("-" * 92)

    for M, N, K in sizes:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        bias = torch.randn(N, device="cuda")
        flops = 2.0 * M * N * K

        # Path A: PyTorch unfused (cuBLAS + ATen)
        def pt_unfused():
            return torch.relu(A @ B + bias)
        us_a = time_fn(pt_unfused)
        ref = pt_unfused()

        # Path B: PyTorch running BPD fused kernel (same runtime!)
        def pt_bpd_fused():
            return bpd.mm_bias_relu(A, B, bias)
        us_b = time_fn(pt_bpd_fused)
        out_b = pt_bpd_fused()
        ulp_b = ulp_distance(ref, out_b)

        # Path C: PyTorch running BPD unfused (our matmul + ATen elem)
        def pt_bpd_unfused():
            return torch.relu(bpd.mm_raw(A, B) + bias)
        us_c = time_fn(pt_bpd_unfused)
        out_c = pt_bpd_unfused()
        ulp_c = ulp_distance(ref, out_c)

        gf_a = flops / (us_a * 1e3)
        gf_b = flops / (us_b * 1e3)
        gf_c = flops / (us_c * 1e3)

        label = f"{M}x{N}x{K}"
        print(f"{label:>14}  {'A: PT unfused (cuBLAS+ATen)':>30}  {us_a:>8.0f}  {gf_a:>7.0f}  {'ref':>6}  {'1.00x':>8}")
        print(f"{'':>14}  {'B: PT running BPD FUSED':>30}  {us_b:>8.0f}  {gf_b:>7.0f}  {ulp_b:>6}  {us_a/us_b:>7.2f}x")
        print(f"{'':>14}  {'C: PT running BPD unfused+ATen':>30}  {us_c:>8.0f}  {gf_c:>7.0f}  {ulp_c:>6}  {us_a/us_c:>7.2f}x")
        print()

    print("INTERPRETATION:")
    print("  B/A > 1.0 = BPD fused kernel is FASTER than PyTorch unfused")
    print("  C/A ≈ 1.0 = BPD matmul matches cuBLAS (same runtime, fair test)")
    print("  B/C > 1.0 = pure fusion speedup (eliminated DRAM round-trips)")
    print()
    print("  The comparison is airtight: all paths run inside PyTorch's runtime.")
    print("  Same CUDA context, same driver, same memory allocator.")
    print("  Only the kernel code differs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
