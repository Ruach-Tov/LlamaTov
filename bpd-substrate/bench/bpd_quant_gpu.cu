// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_quant_gpu.cu — Q4_K dequantization + quantized matmul on GPU
 *
 * CUDA implementation of Q4_K_M dequantization matching llama.cpp.
 * Each thread block processes one Q4_K block (256 elements, 144 bytes).
 *
 * Build: nvcc -O2 -shared -Xcompiler -fPIC -o build/bpd_quant_gpu.so bench/bpd_quant_gpu.cu
 */

#include <cuda_runtime.h>
// cuda_fp16.h removed — using manual half decode
#include <math.h>
#include <stdint.h>

#define Q4K_BLOCK_SIZE 256
#define Q4K_BYTES 144

// ── Half-precision decode ──

__device__ float half_to_float_dev(uint16_t h) {
    // Manual half-to-float (no cuda_fp16.h needed)
    // IEEE 754 binary16 decode
    unsigned sign = (h >> 15) & 1;
    unsigned exp = (h >> 10) & 0x1F;
    unsigned mant = h & 0x3FF;
    float result;
    if (exp == 0) result = (mant == 0) ? 0.0f : ldexpf((float)mant / 1024.0f, -14);
    else if (exp == 31) result = (mant == 0) ? INFINITY : NAN;
    else result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    return sign ? -result : result;
}

// ── Q4_K dequantization kernel ──
// One thread per element. Each warp (32 threads) handles one sub-block.

__global__ void k_dequant_q4k(const uint8_t* __restrict__ qdata,
                               float* __restrict__ output,
                               int n_blocks) {
    int block_idx = blockIdx.x;
    int elem_in_block = threadIdx.x;  // 0..255
    
    if (block_idx >= n_blocks) return;
    
    const uint8_t* block = qdata + block_idx * Q4K_BYTES;
    
    // Read d and dmin (half-precision)
    uint16_t d_half, dmin_half;
    memcpy(&d_half, block, 2);
    memcpy(&dmin_half, block + 2, 2);
    float d = half_to_float_dev(d_half);
    float dmin = half_to_float_dev(dmin_half);
    
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;
    
    // Which sub-block (0..7) and position within it (0..31)
    int j = elem_in_block / 32;
    int pos = elem_in_block % 32;
    
    // Extract 6-bit scale and min for this sub-block
    uint8_t sc, m;
    if (j < 4) {
        sc = scales[j] & 0x3F;
        m = scales[j + 4] & 0x3F;
    } else {
        sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
    
    float scale = d * sc;
    float min = dmin * m;
    
    // Read nibble
    uint8_t byte;
    float val;
    if (pos < 16) {
        byte = qs[j * 16 + pos];
        val = scale * (float)(byte & 0x0F) - min;
    } else {
        byte = qs[j * 16 + (pos - 16)];
        val = scale * (float)(byte >> 4) - min;
    }
    
    output[block_idx * Q4K_BLOCK_SIZE + elem_in_block] = val;
}

// ── Quantized matrix-vector multiply (GPU) ──
// Each block handles one output row.
// Threads within a block collaboratively dequant + accumulate.

__global__ void k_qmatmul_q4k(const uint8_t* __restrict__ qweight,
                                const float* __restrict__ x,
                                float* __restrict__ output,
                                int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    int n_blocks_per_row = K / Q4K_BLOCK_SIZE;
    int tid = threadIdx.x;
    
    // Shared memory for partial sums
    __shared__ float sdata[256];
    sdata[tid] = 0.0f;
    
    const uint8_t* row_data = qweight + row * n_blocks_per_row * Q4K_BYTES;
    
    // Each thread handles multiple blocks
    for (int b = tid; b < n_blocks_per_row; b += blockDim.x) {
        const uint8_t* block = row_data + b * Q4K_BYTES;
        
        uint16_t d_half, dmin_half;
        memcpy(&d_half, block, 2);
        memcpy(&dmin_half, block + 2, 2);
        float d = half_to_float_dev(d_half);
        float dmin = half_to_float_dev(dmin_half);
        
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        
        float block_sum = 0.0f;
        int k_base = b * Q4K_BLOCK_SIZE;
        
        for (int j = 0; j < 8; j++) {
            uint8_t sc, m;
            if (j < 4) {
                sc = scales[j] & 0x3F;
                m = scales[j + 4] & 0x3F;
            } else {
                sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }
            float scale = d * sc;
            float min_val = dmin * m;
            
            for (int k = 0; k < 16; k++) {
                uint8_t byte = qs[j * 16 + k];
                float v0 = scale * (float)(byte & 0x0F) - min_val;
                float v1 = scale * (float)(byte >> 4) - min_val;
                block_sum += v0 * x[k_base + j * 32 + k];
                block_sum += v1 * x[k_base + j * 32 + k + 16];
            }
        }
        sdata[tid] += block_sum;
    }
    
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) output[row] = sdata[0];
}

// ═══════════════════════════════════════════════════════════════
// Host wrappers
// ═══════════════════════════════════════════════════════════════

extern "C" {

void bpd_dequant_q4k_gpu(const uint8_t* qdata, float* output, int n_blocks) {
    k_dequant_q4k<<<n_blocks, Q4K_BLOCK_SIZE>>>(qdata, output, n_blocks);
}

void bpd_qmatmul_q4k_gpu(const uint8_t* qweight, const float* x, float* output,
                           int M, int K) {
    k_qmatmul_q4k<<<M, 256>>>(qweight, x, output, M, K);
}

// GPU memory helpers (reuse from bpd_gpu_kernels.cu if linked separately)
void* qgpu_alloc(int bytes) { void* p; cudaMalloc(&p, bytes); return p; }
void  qgpu_free(void* p) { cudaFree(p); }
void  qgpu_h2d(void* d, const void* s, int bytes) { cudaMemcpy(d, s, bytes, cudaMemcpyHostToDevice); }
void  qgpu_d2h(void* d, const void* s, int bytes) { cudaMemcpy(d, s, bytes, cudaMemcpyDeviceToHost); }
void  qgpu_sync() { cudaDeviceSynchronize(); }

} // extern "C"
