// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_radix_sort_gpu_v3.cu — Correct GPU radix sort with global split
 *
 * Fix: use GLOBAL prefix sum across all elements, not per-block.
 * Each bit-pass produces globally correct ordering.
 *
 * Strategy:
 *   1. k_flag: each thread writes 1 if bit=0, else 0
 *   2. Global exclusive scan of flag array (multi-block Blelloch)
 *   3. k_scatter: each thread computes unique global dest from scan result
 *
 * The global scan is the key — it gives every element a globally
 * unique rank within the false-keys and true-keys partitions.
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_radix_sort_v3.so bench/bpd_radix_sort_gpu_v3.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

/* ================================================================
 * Global exclusive scan using block-level scans + block sums
 * Three-kernel approach:
 *   1. Per-block scan → local scans + block totals
 *   2. Scan the block totals
 *   3. Add block totals back to each block's elements
 * ================================================================ */

#define SCAN_BLOCK_SIZE 256

/* Blelloch exclusive scan within shared memory */
__device__ void blelloch_scan_shared(volatile uint32_t* data, int n) {
    int tid = threadIdx.x;
    int offset = 1;

    /* Up-sweep */
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (bi < n) data[bi] += data[ai];
        }
        offset <<= 1;
    }

    __syncthreads();
    if (tid == 0) data[n - 1] = 0;

    /* Down-sweep */
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (bi < n) {
                uint32_t t = data[ai];
                data[ai] = data[bi];
                data[bi] += t;
            }
        }
    }
    __syncthreads();
}

/* Kernel 1: per-block exclusive scan, output block sums */
__global__ void k_block_scan(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* __restrict__ block_sums,
    int n)
{
    __shared__ uint32_t temp[SCAN_BLOCK_SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    temp[tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    /* Save original value for computing block sum */
    uint32_t orig = temp[tid];

    blelloch_scan_shared(temp, blockDim.x);

    if (gid < n) output[gid] = temp[tid];

    /* Last thread computes block total */
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = temp[tid] + orig;
    }
}

/* Kernel 2: scan block sums (single block — works for up to SCAN_BLOCK_SIZE^2 elements) */
__global__ void k_scan_block_sums(
    uint32_t* __restrict__ block_sums,
    int num_blocks)
{
    __shared__ uint32_t temp[SCAN_BLOCK_SIZE];
    int tid = threadIdx.x;
    temp[tid] = (tid < num_blocks) ? block_sums[tid] : 0;
    __syncthreads();

    blelloch_scan_shared(temp, blockDim.x);

    if (tid < num_blocks) block_sums[tid] = temp[tid];
}

/* Kernel 3: add scanned block sums back */
__global__ void k_add_block_sums(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ block_sums,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        data[gid] += block_sums[blockIdx.x];
    }
}

/* ================================================================
 * Three-phase global exclusive scan
 * ================================================================ */
static void global_exclusive_scan(uint32_t* d_data, int n) {
    int num_blocks = (n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;

    uint32_t* d_output;
    uint32_t* d_block_sums;
    cudaMalloc(&d_output, n * sizeof(uint32_t));
    cudaMalloc(&d_block_sums, num_blocks * sizeof(uint32_t));

    /* Phase 1: per-block scan */
    k_block_scan<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_data, d_output, d_block_sums, n);

    /* Phase 2: scan block sums */
    int scan_threads = 1;
    while (scan_threads < num_blocks) scan_threads <<= 1;
    if (scan_threads > SCAN_BLOCK_SIZE) scan_threads = SCAN_BLOCK_SIZE;
    k_scan_block_sums<<<1, scan_threads>>>(d_block_sums, num_blocks);

    /* Phase 3: add block sums */
    k_add_block_sums<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_output, d_block_sums, n);

    /* Copy result back to d_data */
    cudaMemcpy(d_data, d_output, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    cudaFree(d_output);
    cudaFree(d_block_sums);
}

/* ================================================================
 * Flag kernel: write 1 for false keys (bit=0), 0 for true keys
 * ================================================================ */
__global__ void k_flag_false(
    const uint32_t* __restrict__ keys,
    uint32_t* __restrict__ flags,
    int n, int bit)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        flags[gid] = ((keys[gid] >> bit) & 1) == 0 ? 1 : 0;
    }
}

/* ================================================================
 * Scatter kernel: use globally-scanned flags to compute dest address
 * ================================================================ */
__global__ void k_scatter_global(
    const uint32_t* __restrict__ keys_in,
    const int* __restrict__ vals_in,
    uint32_t* __restrict__ keys_out,
    int* __restrict__ vals_out,
    const uint32_t* __restrict__ flags,     /* original flags (before scan) */
    const uint32_t* __restrict__ scan_out,  /* exclusive scan of flags */
    uint32_t total_falses,
    int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    uint32_t dest;
    if (flags[gid]) {
        /* False key: destination = scan result */
        dest = scan_out[gid];
    } else {
        /* True key: destination = gid - scan[gid] + totalFalses */
        dest = gid - scan_out[gid] + total_falses;
    }

    if (dest < (uint32_t)n) {
        keys_out[dest] = keys_in[gid];
        vals_out[dest] = vals_in[gid];
    }
}

/* ================================================================
 * Host API: full radix sort
 * ================================================================ */
extern "C" {

void bpd_radix_sort_gpu_v3(
    uint32_t* keys, int* values, int n,
    int block_size_unused)  /* block_size fixed to SCAN_BLOCK_SIZE */
{
    int num_blocks = (n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;

    uint32_t *d_keys_in, *d_keys_out;
    int *d_vals_in, *d_vals_out;
    uint32_t *d_flags, *d_scan;

    cudaMalloc(&d_keys_in, n * sizeof(uint32_t));
    cudaMalloc(&d_keys_out, n * sizeof(uint32_t));
    cudaMalloc(&d_vals_in, n * sizeof(int));
    cudaMalloc(&d_vals_out, n * sizeof(int));
    cudaMalloc(&d_flags, n * sizeof(uint32_t));
    cudaMalloc(&d_scan, n * sizeof(uint32_t));

    cudaMemcpy(d_keys_in, keys, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals_in, values, n * sizeof(int), cudaMemcpyHostToDevice);

    for (int bit = 0; bit < 32; bit++) {
        /* Step 1: Flag false keys */
        k_flag_false<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_keys_in, d_flags, n, bit);

        /* Step 2: Copy flags and compute global exclusive scan */
        cudaMemcpy(d_scan, d_flags, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        global_exclusive_scan(d_scan, n);

        /* Step 3: Compute totalFalses = scan[n-1] + flags[n-1] */
        uint32_t last_scan, last_flag;
        cudaMemcpy(&last_scan, d_scan + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_flag, d_flags + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t total_falses = last_scan + last_flag;

        /* Step 4: Scatter to globally correct positions */
        k_scatter_global<<<num_blocks, SCAN_BLOCK_SIZE>>>(
            d_keys_in, d_vals_in, d_keys_out, d_vals_out,
            d_flags, d_scan, total_falses, n);

        /* Swap */
        uint32_t* tmp_k = d_keys_in; d_keys_in = d_keys_out; d_keys_out = tmp_k;
        int* tmp_v = d_vals_in; d_vals_in = d_vals_out; d_vals_out = tmp_v;
    }

    cudaMemcpy(keys, d_keys_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, d_vals_in, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_keys_in); cudaFree(d_keys_out);
    cudaFree(d_vals_in); cudaFree(d_vals_out);
    cudaFree(d_flags); cudaFree(d_scan);
}

} /* extern "C" */
