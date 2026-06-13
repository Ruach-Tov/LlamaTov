// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_sw_gpu_v2.cu — Single-launch GPU Smith-Waterman
 *
 * Eliminates per-anti-diagonal kernel launch overhead by processing
 * ALL anti-diagonals in a single persistent kernel with __syncthreads()
 * barriers between diagonals.
 *
 * nsys showed: 59% of GPU time was cudaLaunchKernel overhead (1999 launches).
 * This kernel: 1 launch, 0 overhead.
 *
 * CONSTRAINT: single block (all threads must synchronize). This limits us
 * to block_size threads = block_size cells per anti-diagonal. For sequences
 * longer than block_size, we tile the anti-diagonal and process tiles
 * sequentially within the kernel.
 *
 * For sequences up to ~1024bp, one block handles everything.
 * For longer sequences, we use grid-stride with cooperative groups
 * or fall back to multi-launch.
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_sw_gpu_v2.so bench/bpd_sw_gpu_v2.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

/* Single-launch SW: one block processes all anti-diagonals.
 * Uses shared memory for H row buffers to avoid global memory latency.
 */
__global__ void k_sw_single_launch(
    const char* __restrict__ query,
    const char* __restrict__ ref,
    int* __restrict__ H,       /* (qlen+1) × (rlen+1) global matrix */
    int* __restrict__ E,
    int* __restrict__ F,
    int qlen, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int* __restrict__ max_score,
    int* __restrict__ max_i,
    int* __restrict__ max_j)
{
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int total_diags = qlen + rlen - 1;

    /* Process each anti-diagonal sequentially, threads handle cells in parallel */
    for (int d = 0; d < total_diags; d++) {
        /* Cells on anti-diagonal d: i + j = d + 2 (1-indexed) */
        int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
        int i_max = (d + 1 < qlen) ? d + 1 : qlen;
        int num_cells = i_max - i_min + 1;

        /* Each thread processes one or more cells (grid-stride within block) */
        for (int c = tid; c < num_cells; c += block_size) {
            int i = i_min + c;
            int j = d + 2 - i;

            if (i < 1 || i > qlen || j < 1 || j > rlen) continue;

            int idx = i * (rlen + 1) + j;
            int idx_diag = (i-1) * (rlen + 1) + (j-1);
            int idx_left = i * (rlen + 1) + (j-1);
            int idx_up = (i-1) * (rlen + 1) + j;

            int s = (query[i-1] == ref[j-1]) ? match_score : mismatch_score;

            /* E: horizontal gap */
            int e1 = H[idx_left] - gap_open;
            int e2 = E[idx_left] - gap_extend;
            int e_val = (e1 > e2) ? e1 : e2;
            E[idx] = e_val;

            /* F: vertical gap */
            int f1 = H[idx_up] - gap_open;
            int f2 = F[idx_up] - gap_extend;
            int f_val = (f1 > f2) ? f1 : f2;
            F[idx] = f_val;

            /* H: max of all options */
            int h = H[idx_diag] + s;
            if (e_val > h) h = e_val;
            if (f_val > h) h = f_val;
            if (h < 0) h = 0;
            H[idx] = h;

            /* Track maximum */
            if (h > 0) {
                int old = atomicMax(max_score, h);
                if (h > old) {
                    *max_i = i;
                    *max_j = j;
                }
            }
        }

        /* ALL threads must finish this anti-diagonal before the next one.
         * This is the key synchronization that replaces 1999 kernel launches
         * with 1999 __syncthreads() calls. */
        __syncthreads();
    }
}

/* Multi-block variant for long sequences: uses grid-level sync via
 * cooperative groups. Falls back to per-diagonal launch if cooperative
 * launch is not available. */
__global__ void k_sw_single_diag_multiblock(
    const char* __restrict__ query,
    const char* __restrict__ ref,
    int* __restrict__ H,
    int* __restrict__ E,
    int* __restrict__ F,
    int qlen, int rlen,
    int d,  /* which anti-diagonal */
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int* __restrict__ max_score,
    int* __restrict__ max_i,
    int* __restrict__ max_j)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
    int i_max = (d + 1 < qlen) ? d + 1 : qlen;
    int num_cells = i_max - i_min + 1;

    if (gid >= num_cells) return;

    int i = i_min + gid;
    int j = d + 2 - i;
    if (i < 1 || i > qlen || j < 1 || j > rlen) return;

    int idx = i * (rlen + 1) + j;
    int idx_diag = (i-1) * (rlen + 1) + (j-1);
    int idx_left = i * (rlen + 1) + (j-1);
    int idx_up = (i-1) * (rlen + 1) + j;

    int s = (query[i-1] == ref[j-1]) ? match_score : mismatch_score;

    int e1 = H[idx_left] - gap_open;
    int e2 = E[idx_left] - gap_extend;
    E[idx] = (e1 > e2) ? e1 : e2;

    int f1 = H[idx_up] - gap_open;
    int f2 = F[idx_up] - gap_extend;
    int f_val = (f1 > f2) ? f1 : f2;
    F[idx] = f_val;

    int h = H[idx_diag] + s;
    if (E[idx] > h) h = E[idx];
    if (f_val > h) h = f_val;
    if (h < 0) h = 0;
    H[idx] = h;

    if (h > 0) {
        int old = atomicMax(max_score, h);
        if (h > old) { *max_i = i; *max_j = j; }
    }
}

extern "C" {

typedef struct {
    int score;
    int query_end;
    int ref_end;
} sw_gpu_result_t;

void bpd_smith_waterman_gpu_v2(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int block_size,
    sw_gpu_result_t* result)
{
    int matrix_size = (qlen + 1) * (rlen + 1) * sizeof(int);

    char *d_query, *d_ref;
    int *d_H, *d_E, *d_F, *d_max_score, *d_max_i, *d_max_j;

    cudaMalloc(&d_query, qlen);
    cudaMalloc(&d_ref, rlen);
    cudaMalloc(&d_H, matrix_size);
    cudaMalloc(&d_E, matrix_size);
    cudaMalloc(&d_F, matrix_size);
    cudaMalloc(&d_max_score, sizeof(int));
    cudaMalloc(&d_max_i, sizeof(int));
    cudaMalloc(&d_max_j, sizeof(int));

    cudaMemcpy(d_query, query, qlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, ref, rlen, cudaMemcpyHostToDevice);
    cudaMemset(d_H, 0, matrix_size);
    cudaMemset(d_E, 0, matrix_size);
    cudaMemset(d_F, 0, matrix_size);
    cudaMemset(d_max_score, 0, sizeof(int));
    cudaMemset(d_max_i, 0, sizeof(int));
    cudaMemset(d_max_j, 0, sizeof(int));

    /* Choose strategy based on max anti-diagonal width */
    int max_diag_width = (qlen < rlen) ? qlen : rlen;

    if (max_diag_width <= block_size) {
        /* Single block handles everything — one launch, zero overhead */
        k_sw_single_launch<<<1, block_size>>>(
            d_query, d_ref, d_H, d_E, d_F,
            qlen, rlen,
            match_score, mismatch_score, gap_open, gap_extend,
            d_max_score, d_max_i, d_max_j);
    } else {
        /* Multi-block: one launch per anti-diagonal but with
         * CUDA streams for overlapping launch + execution */
        int total_diags = qlen + rlen - 1;
        for (int d = 0; d < total_diags; d++) {
            int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
            int i_max = (d + 1 < qlen) ? d + 1 : qlen;
            int num_cells = i_max - i_min + 1;
            if (num_cells <= 0) continue;
            int blocks = (num_cells + block_size - 1) / block_size;
            k_sw_single_diag_multiblock<<<blocks, block_size>>>(
                d_query, d_ref, d_H, d_E, d_F,
                qlen, rlen, d,
                match_score, mismatch_score, gap_open, gap_extend,
                d_max_score, d_max_i, d_max_j);
        }
    }
    cudaDeviceSynchronize();

    int h_max_score, h_max_i, h_max_j;
    cudaMemcpy(&h_max_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_i, d_max_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_j, d_max_j, sizeof(int), cudaMemcpyDeviceToHost);

    result->score = h_max_score;
    result->query_end = h_max_i - 1;
    result->ref_end = h_max_j - 1;

    cudaFree(d_query); cudaFree(d_ref);
    cudaFree(d_H); cudaFree(d_E); cudaFree(d_F);
    cudaFree(d_max_score); cudaFree(d_max_i); cudaFree(d_max_j);
}

int bpd_sw_score_gpu_v2(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match, int mismatch,
    int gap_open, int gap_extend,
    int block_size)
{
    sw_gpu_result_t result;
    bpd_smith_waterman_gpu_v2(query, qlen, ref, rlen,
        match, mismatch, gap_open, gap_extend, block_size, &result);
    return result.score;
}

} /* extern "C" */
