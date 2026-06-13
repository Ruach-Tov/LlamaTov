// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_quant_gpu_sweep.cu — Q4_K GPU dequant with sweepable parameters
 *
 * Parameters to sweep:
 *   BLOCK_SIZE: threads per block (32, 64, 128, 256, 512)
 *   BLOCKS_PER_SM: occupancy hint
 *   Elements per thread: 1 vs 2 vs 4 (ILP)
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define Q4K_BLOCK_SIZE 256
#define Q4K_BYTES 144

__device__ float half_to_float_dev(uint16_t h) {
    unsigned sign = (h >> 15) & 1;
    unsigned exp = (h >> 10) & 0x1F;
    unsigned mant = h & 0x3FF;
    float result;
    if (exp == 0) result = (mant == 0) ? 0.0f : ldexpf((float)mant / 1024.0f, -14);
    else if (exp == 31) result = (mant == 0) ? INFINITY : NAN;
    else result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    return sign ? -result : result;
}

__device__ inline void dev_get_scale_min(int j, const uint8_t* scales, uint8_t* sc, uint8_t* m) {
    if (j < 4) { *sc = scales[j] & 0x3F; *m = scales[j+4] & 0x3F; }
    else { *sc = (scales[j+4]&0x0F)|((scales[j-4]>>6)<<4); *m = (scales[j+4]>>4)|((scales[j]>>6)<<4); }
}

/* Variant A: 1 element per thread, BLOCK_SIZE threads per Q4_K block */
template<int BLOCK>
__global__ void k_dequant_q4k_v1(const uint8_t* __restrict__ qdata,
                                   float* __restrict__ output,
                                   int n_blocks) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (block_idx >= n_blocks || tid >= 256) return;

    const uint8_t* block = qdata + block_idx * Q4K_BYTES;
    uint16_t d_half, dmin_half;
    memcpy(&d_half, block, 2);
    memcpy(&dmin_half, block + 2, 2);
    float d = half_to_float_dev(d_half);
    float dmin = half_to_float_dev(dmin_half);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    int j = tid / 32;
    int pos = tid % 32;
    uint8_t sc, m;
    dev_get_scale_min(j, scales, &sc, &m);
    float scale = d * sc;
    float min_val = dmin * m;

    if (pos < 16) {
        output[block_idx * 256 + tid] = scale * (float)(qs[j*16+pos] & 0x0F) - min_val;
    } else {
        output[block_idx * 256 + tid] = scale * (float)(qs[j*16+(pos-16)] >> 4) - min_val;
    }
}

/* Variant B: 2 elements per thread (ILP=2), half the threads */
__global__ void k_dequant_q4k_ilp2(const uint8_t* __restrict__ qdata,
                                     float* __restrict__ output,
                                     int n_blocks) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;  /* 0..127 */
    if (block_idx >= n_blocks || tid >= 128) return;

    const uint8_t* block = qdata + block_idx * Q4K_BYTES;
    uint16_t d_half, dmin_half;
    memcpy(&d_half, block, 2);
    memcpy(&dmin_half, block + 2, 2);
    float d = half_to_float_dev(d_half);
    float dmin = half_to_float_dev(dmin_half);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    /* Each thread handles one byte (two nibbles = two elements) */
    int j = tid / 16;  /* sub-block 0..7 */
    int k = tid % 16;  /* byte within sub-block */
    uint8_t sc, m;
    dev_get_scale_min(j, scales, &sc, &m);
    float scale = d * sc;
    float min_val = dmin * m;
    uint8_t byte = qs[j*16+k];

    output[block_idx * 256 + j*32 + k]      = scale * (float)(byte & 0x0F) - min_val;
    output[block_idx * 256 + j*32 + k + 16] = scale * (float)(byte >> 4) - min_val;
}

/* Variant C: Multiple Q4_K blocks per thread block */
template<int BLOCKS_PER_TB>
__global__ void k_dequant_q4k_multi(const uint8_t* __restrict__ qdata,
                                      float* __restrict__ output,
                                      int n_blocks) {
    int base_block = blockIdx.x * BLOCKS_PER_TB;
    int tid = threadIdx.x;
    
    for (int b = 0; b < BLOCKS_PER_TB; b++) {
        int block_idx = base_block + b;
        if (block_idx >= n_blocks) return;
        
        const uint8_t* block = qdata + block_idx * Q4K_BYTES;
        uint16_t d_half, dmin_half;
        memcpy(&d_half, block, 2);
        memcpy(&dmin_half, block + 2, 2);
        float d = half_to_float_dev(d_half);
        float dmin = half_to_float_dev(dmin_half);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        
        if (tid < 128) {
            int j = tid / 16;
            int k = tid % 16;
            uint8_t sc, m;
            dev_get_scale_min(j, scales, &sc, &m);
            float scale = d * sc;
            float min_val = dmin * m;
            uint8_t byte = qs[j*16+k];
            output[block_idx * 256 + j*32 + k]      = scale * (float)(byte & 0x0F) - min_val;
            output[block_idx * 256 + j*32 + k + 16] = scale * (float)(byte >> 4) - min_val;
        }
    }
}

extern "C" {

/* Sweep entry points */
void dequant_v1_256(const uint8_t* q, float* o, int nb) { k_dequant_q4k_v1<256><<<nb, 256>>>(q, o, nb); }
void dequant_v1_128(const uint8_t* q, float* o, int nb) { k_dequant_q4k_v1<128><<<nb, 128>>>(q, o, nb); }
void dequant_v1_64(const uint8_t* q, float* o, int nb)  { k_dequant_q4k_v1<64><<<nb, 64>>>(q, o, nb); }
void dequant_v1_32(const uint8_t* q, float* o, int nb)  { k_dequant_q4k_v1<32><<<nb, 32>>>(q, o, nb); }

void dequant_ilp2(const uint8_t* q, float* o, int nb) { k_dequant_q4k_ilp2<<<nb, 128>>>(q, o, nb); }

void dequant_multi2(const uint8_t* q, float* o, int nb) { k_dequant_q4k_multi<2><<<(nb+1)/2, 128>>>(q, o, nb); }
void dequant_multi4(const uint8_t* q, float* o, int nb) { k_dequant_q4k_multi<4><<<(nb+3)/4, 128>>>(q, o, nb); }
void dequant_multi8(const uint8_t* q, float* o, int nb) { k_dequant_q4k_multi<8><<<(nb+7)/8, 128>>>(q, o, nb); }

void* sgpu_alloc(int bytes) { void* p; cudaMalloc(&p, bytes); return p; }
void  sgpu_free(void* p) { cudaFree(p); }
void  sgpu_h2d(void* d, const void* s, int bytes) { cudaMemcpy(d, s, bytes, cudaMemcpyHostToDevice); }
void  sgpu_d2h(void* d, const void* s, int bytes) { cudaMemcpy(d, s, bytes, cudaMemcpyDeviceToHost); }
void  sgpu_sync() { cudaDeviceSynchronize(); }

} /* extern "C" */
