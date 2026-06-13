// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_radix_sort_gpu_v2.cu — Correct GPU radix sort using split primitive
 *
 * Based on GPU Gems 3, Chapter 39 (Blelloch scan + split-based radix sort).
 * Key insight: use scan to compute per-element destination addresses.
 * NO atomicAdd in scatter — each element gets a unique address.
 *
 * Algorithm: 1-bit radix sort (32 passes for 32-bit keys)
 *   Per pass: 
 *     1. Flag false keys (bit=0) → flag array
 *     2. Exclusive scan of flag array → false-key addresses
 *     3. totalFalses = scan[n-1] + flag[n-1]
 *     4. True keys: dest = i - scan[i] + totalFalses
 *     5. False keys: dest = scan[i]
 *     6. Scatter (no race — every dest is unique)
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_radix_sort_v2.so bench/bpd_radix_sort_gpu_v2.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

/* ================================================================
 * Block-level exclusive scan (Blelloch, work-efficient)
 * Operates on shared memory array of 'count' elements.
 * ================================================================ */
__device__ void block_exclusive_scan(volatile uint32_t* data, int count) {
    int tid = threadIdx.x;
    int offset = 1;

    /* Up-sweep (reduce) */
    for (int d = count >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            data[bi] += data[ai];
        }
        offset <<= 1;
    }

    /* Clear last element */
    if (tid == 0) data[count - 1] = 0;

    /* Down-sweep */
    for (int d = 1; d < count; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            uint32_t t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }
    }
    __syncthreads();
}

/* ================================================================
 * Single-bit split: partition elements by one bit position
 * Processes one block of up to BLOCK_SIZE elements.
 * ================================================================ */
#define MAX_BLOCK_SIZE 512

__global__ void k_split_scatter(
    const uint32_t* __restrict__ keys_in,
    const int* __restrict__ vals_in,
    uint32_t* __restrict__ keys_out,
    int* __restrict__ vals_out,
    int n,
    int bit)
{
    __shared__ uint32_t flags[MAX_BLOCK_SIZE];      /* 1 if bit=0 (false key) */
    __shared__ uint32_t scan_out[MAX_BLOCK_SIZE];
    __shared__ uint32_t total_falses_shared;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    /* Step 1: Flag false keys */
    uint32_t key = 0;
    int val = 0;
    int active = (gid < n);

    if (active) {
        key = keys_in[gid];
        val = vals_in[gid];
        flags[tid] = ((key >> bit) & 1) == 0 ? 1 : 0;
    } else {
        flags[tid] = 0;
    }
    __syncthreads();

    /* Step 2: Copy flags to scan buffer and scan */
    scan_out[tid] = flags[tid];
    __syncthreads();

    /* Blelloch exclusive scan — need power-of-2 size */
    block_exclusive_scan(scan_out, bs);

    /* Step 3: Compute totalFalses */
    if (tid == bs - 1) {
        total_falses_shared = scan_out[bs - 1] + flags[bs - 1];
    }
    __syncthreads();

    uint32_t total_falses = total_falses_shared;

    /* Step 4-5: Compute destination address */
    if (active) {
        uint32_t dest;
        if (flags[tid]) {
            /* False key (bit=0): goes to scan position */
            dest = scan_out[tid];
        } else {
            /* True key (bit=1): goes to tid - scan[tid] + totalFalses */
            dest = tid - scan_out[tid] + total_falses;
        }

        /* Step 6: Scatter — within this block's output region */
        uint32_t global_dest = blockIdx.x * blockDim.x + dest;
        if (global_dest < (uint32_t)n) {
            keys_out[global_dest] = key;
            vals_out[global_dest] = val;
        }
    }
}

/* ================================================================
 * Host-callable: radix sort using split primitive
 * 32 passes (one per bit), each pass uses scan+scatter.
 * ================================================================ */
extern "C" {

void bpd_radix_sort_gpu_v2(
    uint32_t* keys, int* values, int n,
    int block_size)
{
    if (block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;

    /* Round block_size to power of 2 for Blelloch scan */
    int bs = 1;
    while (bs < block_size) bs <<= 1;

    int num_blocks = (n + bs - 1) / bs;

    uint32_t *d_keys_in, *d_keys_out;
    int *d_vals_in, *d_vals_out;

    cudaMalloc(&d_keys_in, n * sizeof(uint32_t));
    cudaMalloc(&d_keys_out, n * sizeof(uint32_t));
    cudaMalloc(&d_vals_in, n * sizeof(int));
    cudaMalloc(&d_vals_out, n * sizeof(int));

    cudaMemcpy(d_keys_in, keys, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals_in, values, n * sizeof(int), cudaMemcpyHostToDevice);

    /* 32 passes, one per bit (LSB first) */
    for (int bit = 0; bit < 32; bit++) {
        k_split_scatter<<<num_blocks, bs>>>(
            d_keys_in, d_vals_in,
            d_keys_out, d_vals_out,
            n, bit);
        cudaDeviceSynchronize();

        /* Swap buffers */
        uint32_t* tmp_k = d_keys_in; d_keys_in = d_keys_out; d_keys_out = tmp_k;
        int* tmp_v = d_vals_in; d_vals_in = d_vals_out; d_vals_out = tmp_v;
    }

    /* Result is in d_keys_in after 32 swaps */
    cudaMemcpy(keys, d_keys_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, d_vals_in, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_keys_in); cudaFree(d_keys_out);
    cudaFree(d_vals_in); cudaFree(d_vals_out);
}

/* Suffix array via radix sort */
void bpd_build_suffix_array_gpu_v2(
    const char* text, int n,
    int* SA,
    int block_size)
{
    uint32_t* keys = (uint32_t*)malloc(n * sizeof(uint32_t));
    int* indices = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        uint32_t key = 0;
        for (int j = 0; j < 4 && i + j < n; j++) {
            key = (key << 8) | (uint8_t)text[i + j];
        }
        for (int j = n - i; j < 4; j++) {
            key <<= 8;
        }
        keys[i] = key;
        indices[i] = i;
    }

    bpd_radix_sort_gpu_v2(keys, indices, n, block_size);
    memcpy(SA, indices, n * sizeof(int));

    free(keys);
    free(indices);
}

} /* extern "C" */
