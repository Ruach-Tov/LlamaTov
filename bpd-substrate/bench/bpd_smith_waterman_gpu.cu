// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_smith_waterman_gpu.cu — GPU Smith-Waterman via anti-diagonal wavefront
 *
 * Parallelism: cells on the same anti-diagonal (i+j = const) are independent.
 * Each anti-diagonal is launched as one kernel invocation (or one sync point).
 * Within an anti-diagonal, one thread per cell.
 *
 * Derived from Prolog fact:
 *   sw_parallelism(anti_diagonal, parallelism_facts(
 *       parallel_dimension(anti_diagonal),
 *       gpu_mapping(one_thread_per_cell))).
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_sw_gpu.so bench/bpd_smith_waterman_gpu.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

/* One anti-diagonal kernel invocation.
 * d = anti-diagonal index (0..qlen+rlen-2)
 * Each thread handles one cell on this anti-diagonal.
 *
 * Cell (i,j) is on anti-diagonal d when i+j-2 = d (1-indexed)
 * so i ranges from max(1, d+2-rlen) to min(d+1, qlen)
 */
__global__ void k_sw_antidiag(
    const char* __restrict__ query,
    const char* __restrict__ ref,
    int* __restrict__ H,
    int* __restrict__ E,
    int* __restrict__ F,
    int qlen, int rlen,
    int d,  /* anti-diagonal index 0-based: d = 0..qlen+rlen-2 */
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int* __restrict__ max_score,
    int* __restrict__ max_i,
    int* __restrict__ max_j)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Map thread to (i,j) on anti-diagonal d */
    /* d = (i-1) + (j-1), so i+j = d+2 */
    /* i ranges from max(1, d+2-rlen) to min(d+1, qlen) */
    int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
    int i_max = (d + 1 < qlen) ? d + 1 : qlen;
    int num_cells = i_max - i_min + 1;

    if (tid >= num_cells) return;

    int i = i_min + tid;
    int j = d + 2 - i;

    /* Bounds check */
    if (i < 1 || i > qlen || j < 1 || j > rlen) return;

    int idx = i * (rlen + 1) + j;
    int idx_diag = (i-1) * (rlen + 1) + (j-1);
    int idx_left = i * (rlen + 1) + (j-1);
    int idx_up = (i-1) * (rlen + 1) + j;

    /* Scoring function — from sw_scoring_function(dna_simple, ...) */
    int s = (query[i-1] == ref[j-1]) ? match_score : mismatch_score;

    /* E update: horizontal gap */
    int e1 = H[idx_left] - gap_open;
    int e2 = E[idx_left] - gap_extend;
    int e_val = (e1 > e2) ? e1 : e2;
    E[idx] = e_val;

    /* F update: vertical gap */
    int f1 = H[idx_up] - gap_open;
    int f2 = F[idx_up] - gap_extend;
    int f_val = (f1 > f2) ? f1 : f2;
    F[idx] = f_val;

    /* H update: max of all options, floored at 0 */
    int h = H[idx_diag] + s;
    if (e_val > h) h = e_val;
    if (f_val > h) h = f_val;
    if (h < 0) h = 0;
    H[idx] = h;

    /* Track global maximum (atomic) */
    if (h > 0) {
        int old = atomicMax(max_score, h);
        if (h > old) {
            /* Race condition on position is acceptable —
             * we just need ANY cell achieving max_score.
             * For exact traceback, we rescan on CPU. */
            *max_i = i;
            *max_j = j;
        }
    }
}

/* Host-callable wrapper: runs the full Smith-Waterman on GPU */
extern "C" {

typedef struct {
    int score;
    int query_end;
    int ref_end;
} sw_gpu_result_t;

void bpd_smith_waterman_gpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int block_size,     /* sweepable parameter */
    sw_gpu_result_t* result)
{
    int matrix_size = (qlen + 1) * (rlen + 1) * sizeof(int);

    /* Allocate device memory */
    char *d_query, *d_ref;
    int *d_H, *d_E, *d_F;
    int *d_max_score, *d_max_i, *d_max_j;

    cudaMalloc(&d_query, qlen);
    cudaMalloc(&d_ref, rlen);
    cudaMalloc(&d_H, matrix_size);
    cudaMalloc(&d_E, matrix_size);
    cudaMalloc(&d_F, matrix_size);
    cudaMalloc(&d_max_score, sizeof(int));
    cudaMalloc(&d_max_i, sizeof(int));
    cudaMalloc(&d_max_j, sizeof(int));

    /* Copy sequences to device */
    cudaMemcpy(d_query, query, qlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, ref, rlen, cudaMemcpyHostToDevice);

    /* Zero matrices and max trackers */
    cudaMemset(d_H, 0, matrix_size);
    cudaMemset(d_E, 0, matrix_size);
    cudaMemset(d_F, 0, matrix_size);
    cudaMemset(d_max_score, 0, sizeof(int));
    cudaMemset(d_max_i, 0, sizeof(int));
    cudaMemset(d_max_j, 0, sizeof(int));

    /* Process anti-diagonals sequentially */
    int total_diags = qlen + rlen - 1;
    for (int d = 0; d < total_diags; d++) {
        /* Number of cells on this anti-diagonal */
        int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
        int i_max = (d + 1 < qlen) ? d + 1 : qlen;
        int num_cells = i_max - i_min + 1;

        if (num_cells <= 0) continue;

        int blocks = (num_cells + block_size - 1) / block_size;
        k_sw_antidiag<<<blocks, block_size>>>(
            d_query, d_ref, d_H, d_E, d_F,
            qlen, rlen, d,
            match_score, mismatch_score,
            gap_open, gap_extend,
            d_max_score, d_max_i, d_max_j);
    }
    cudaDeviceSynchronize();

    /* Copy results back */
    int h_max_score, h_max_i, h_max_j;
    cudaMemcpy(&h_max_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_i, d_max_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_j, d_max_j, sizeof(int), cudaMemcpyDeviceToHost);

    result->score = h_max_score;
    result->query_end = h_max_i - 1;
    result->ref_end = h_max_j - 1;

    /* Cleanup */
    cudaFree(d_query); cudaFree(d_ref);
    cudaFree(d_H); cudaFree(d_E); cudaFree(d_F);
    cudaFree(d_max_score); cudaFree(d_max_i); cudaFree(d_max_j);
}

/* Score-only convenience wrapper */
int bpd_sw_score_gpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match, int mismatch,
    int gap_open, int gap_extend,
    int block_size)
{
    sw_gpu_result_t result;
    bpd_smith_waterman_gpu(query, qlen, ref, rlen,
                            match, mismatch, gap_open, gap_extend,
                            block_size, &result);
    return result.score;
}

} /* extern "C" */
