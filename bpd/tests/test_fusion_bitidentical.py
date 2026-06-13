#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
test_fusion_bitidentical.py — Prove kernel fusion preserves bit-identical output.

Demonstrates that fusing N separate kernel launches into one
produces THE SAME BITS as running them separately, while
eliminating intermediate memory traffic.

Test case: the FFN gate path from transformer inference
  step 1: silu(x)        — activation
  step 2: x * gate       — element-wise multiply  
  Unfused: 2 launches, intermediate silu output stored to DRAM
  Fused:   1 launch, silu result stays in register

Both must produce XOR = 0x00000000 for every element.
"""
import ctypes, numpy as np, struct, os, tempfile, subprocess

# CUDA source: unfused (2 separate kernels) and fused (1 kernel)
CUDA_SOURCE = r'''
#include <cuda_runtime.h>
#include <stdio.h>

// === UNFUSED: two separate kernels ===

__global__ void k_silu(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));  // silu result → DRAM
    }
}

__global__ void k_mul(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];  // read silu from DRAM, multiply
    }
}

// === FUSED: one kernel, no intermediate DRAM ===

__global__ void k_silu_mul_fused(const float *x, const float *gate, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float silu_xi = xi / (1.0f + expf(-xi));  // silu in REGISTER
        out[i] = silu_xi * gate[i];                // multiply, write ONCE
        // silu_xi never touched DRAM — it lived and died in a register
    }
}

int main() {
    const int N = 2048;  // typical transformer hidden dim
    
    // Deterministic input
    float h_x[2048], h_gate[2048];
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i * 0.001f - 1.024f;
        h_gate[i] = (float)i * 0.0005f + 0.5f;
    }
    
    float *d_x, *d_gate, *d_silu_tmp, *d_unfused_out, *d_fused_out;
    cudaMalloc(&d_x, N*4);
    cudaMalloc(&d_gate, N*4);
    cudaMalloc(&d_silu_tmp, N*4);      // intermediate for unfused path
    cudaMalloc(&d_unfused_out, N*4);
    cudaMalloc(&d_fused_out, N*4);
    
    cudaMemcpy(d_x, h_x, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, h_gate, N*4, cudaMemcpyHostToDevice);
    
    int block = 256;
    int grid = (N + block - 1) / block;
    
    // UNFUSED: two launches, intermediate in DRAM
    k_silu<<<grid, block>>>(d_x, d_silu_tmp, N);       // x → silu → DRAM
    k_mul<<<grid, block>>>(d_silu_tmp, d_gate, d_unfused_out, N);  // DRAM → mul → out
    cudaDeviceSynchronize();
    
    // FUSED: one launch, no intermediate
    k_silu_mul_fused<<<grid, block>>>(d_x, d_gate, d_fused_out, N);
    cudaDeviceSynchronize();
    
    // Download both
    float h_unfused[2048], h_fused[2048];
    cudaMemcpy(h_unfused, d_unfused_out, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fused, d_fused_out, N*4, cudaMemcpyDeviceToHost);
    
    // Compare in hex
    unsigned int *u_bits = (unsigned int*)h_unfused;
    unsigned int *f_bits = (unsigned int*)h_fused;
    
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (u_bits[i] != f_bits[i]) mismatches++;
    }
    
    // Output first 5
    printf("FUSION BIT-IDENTICAL TEST: silu + mul\n");
    printf("  N = %d elements\n\n", N);
    printf("  First 5 (canonical hex):\n");
    for (int i = 0; i < 5; i++) {
        unsigned int diff = u_bits[i] ^ f_bits[i];
        printf("    [%d] unfused=%08x fused=%08x xor=%08x %s\n",
               i, u_bits[i], f_bits[i], diff,
               diff == 0 ? "BIT-IDENTICAL" : "DIFFER");
    }
    
    printf("\n  RESULT: ");
    if (mismatches == 0) {
        printf("BIT-IDENTICAL (%d elements, 0 mismatches)\n", N);
        printf("  Fusion eliminated intermediate DRAM traffic.\n");
        printf("  Same bits. Fewer memory ops. Same correctness.\n");
    } else {
        printf("%d / %d elements differ\n", mismatches, N);
    }
    
    // Memory traffic comparison
    printf("\n  MEMORY TRAFFIC:\n");
    printf("    Unfused: read %d + write %d + read %d + read %d + write %d = %d bytes\n",
           N*4, N*4, N*4, N*4, N*4, N*4*5);
    printf("    Fused:   read %d + read %d + write %d = %d bytes\n",
           N*4, N*4, N*4, N*4*3);
    printf("    Savings: %.0f%% less memory traffic\n",
           (1.0 - 3.0/5.0) * 100);
    
    cudaFree(d_x); cudaFree(d_gate); cudaFree(d_silu_tmp);
    cudaFree(d_unfused_out); cudaFree(d_fused_out);
    return mismatches;
}
'''

if __name__ == "__main__":
    src_path = "/tmp/test_fusion.cu"
    bin_path = "/tmp/test_fusion"
    
    with open(src_path, 'w') as f:
        f.write(CUDA_SOURCE)
    
    NVCC = "/nix/store/m6dcnzyvyxsqn3kylql78c9nrk0bib6r-cuda_nvcc-12.8.93/bin/nvcc"
    CUDA_INC = "/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/include"
    CUDA_LIB = "/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib"
    
    os.system(f"{NVCC} -arch=sm_61 -I{CUDA_INC} -L{CUDA_LIB} -o {bin_path} {src_path} -Wno-deprecated-gpu-targets")
    
    NVIDIA_LIB = "/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib"
    os.environ["LD_LIBRARY_PATH"] = f"{NVIDIA_LIB}:{CUDA_LIB}"
    os.system(bin_path)
