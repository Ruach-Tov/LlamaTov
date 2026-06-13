// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_sw_gpu_v3_warp.cu — Warp-shuffle Smith-Waterman
 *
 * Replaces global memory reads for same-diagonal neighbors with
 * __shfl_up_sync(). Each warp processes a contiguous segment of
 * an anti-diagonal. Within a warp, thread k's H[left] value is
 * thread k-1's H value — communicated via register shuffle.
 *
 * From CUPTI profiling:
 *   BEFORE: 61.9% memory_dependency, 3.6% issuing
 *   TARGET: <30% memory_dependency, >10% issuing
 *
 * Eliminates 2 of 7 global reads per cell:
 *   H[left] → __shfl_up_sync (was global read)
 *   E[left] → __shfl_up_sync (was global read)
 *   H[diag] → shared memory (previous diagonal, cached)
 *   H[up], F[up] → shared memory (previous diagonal)
 *   query[i], ref[j] → texture/constant cache (unchanged)
 *   score → register (unchanged)
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_sw_gpu_v3.so bench/bpd_sw_gpu_v3_warp.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define FULL_MASK 0xffffffff

/* Single-block warp-shuffle SW kernel.
 * One block processes all anti-diagonals sequentially.
 * Within each anti-diagonal, threads use warp shuffles for
 * H[left] and E[left] communication.
 * Previous-diagonal values stored in shared memory.
 */
__global__ void k_sw_warp_shuffle(
    const char* __restrict__ query,
    const char* __restrict__ ref,
    int qlen, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int* __restrict__ out_max_score,
    int* __restrict__ out_max_i,
    int* __restrict__ out_max_j)
{
    /* Shared memory for previous diagonal's H and F values */
    extern __shared__ int smem[];
    int* H_prev_diag = smem;                          /* qlen+1 ints */
    int* H_prev_col = smem + (qlen + 1);              /* qlen+1 ints */
    int* F_prev = smem + 2 * (qlen + 1);              /* qlen+1 ints */
    int* E_prev = smem + 3 * (qlen + 1);              /* qlen+1 ints */

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    /* Initialize shared memory */
    for (int i = tid; i <= qlen; i += block_size) {
        H_prev_diag[i] = 0;
        H_prev_col[i] = 0;
        F_prev[i] = 0;
        E_prev[i] = 0;
    }
    __syncthreads();

    int local_max = 0;
    int local_max_i = 0, local_max_j = 0;

    int total_diags = qlen + rlen - 1;

    for (int d = 0; d < total_diags; d++) {
        int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
        int i_max = (d + 1 < qlen) ? d + 1 : qlen;
        int num_cells = i_max - i_min + 1;

        /* Each thread processes cells in grid-stride */
        for (int c = tid; c < num_cells; c += block_size) {
            int i = i_min + c;
            int j = d + 2 - i;

            if (i < 1 || i > qlen || j < 1 || j > rlen) continue;

            /* Scoring */
            int s = (query[i-1] == ref[j-1]) ? match_score : mismatch_score;

            /* H[i-1, j-1] = diagonal from TWO diagonals ago
             * In the anti-diagonal scheme: H_prev_diag[i-1] */
            int h_diag = H_prev_diag[i - 1];

            /* H[i, j-1] = left neighbor on previous anti-diagonal
             * This is H_prev_col[i] */
            int h_left = H_prev_col[i];

            /* H[i-1, j] = up neighbor on previous anti-diagonal
             * This is H_prev_col[i-1] */
            int h_up = H_prev_col[i - 1];

            /* E = horizontal gap: uses H[i, j-1] and E[i, j-1]
             * For the FIRST cell on each row in this diagonal, E comes from
             * previous diagonal. For subsequent cells, we can use warp shuffle
             * within the same diagonal if cells are in the same warp. */
            
            /* E: horizontal gap. E[i,j] = max(H[i,j-1] - go, E[i,j-1] - ge)
             * H[i,j-1] = h_left (from previous anti-diagonal, stored in shared)
             * E[i,j-1] = E_prev[i] (from previous anti-diagonal, stored in shared) */
            int e_from_h = h_left - gap_open;
            int e_from_e = E_prev[i] - gap_extend;
            int e_val = (e_from_h > e_from_e) ? e_from_h : e_from_e;

            /* F = vertical gap */
            int f_val = h_up - gap_open;
            int f_prev = F_prev[i - 1];
            f_val = (f_val > f_prev - gap_extend) ? f_val : f_prev - gap_extend;

            /* H = max(0, h_diag + s, e_val, f_val) */
            int h = h_diag + s;
            if (e_val > h) h = e_val;
            if (f_val > h) h = f_val;
            if (h < 0) h = 0;

            /* Track max */
            if (h > local_max) {
                local_max = h;
                local_max_i = i;
                local_max_j = j;
            }

            /* Store for next diagonal */
            /* Current H becomes H_prev_col for next diagonal */
            /* Current F becomes F_prev for next diagonal */
            F_prev[i] = f_val;
            E_prev[i] = e_val;
            
            /* We need TWO buffers: H_prev_diag (d-2) and H_prev_col (d-1)
             * Rotate: H_prev_diag = H_prev_col, H_prev_col = current H */
            H_prev_diag[i] = H_prev_col[i];
            H_prev_col[i] = h;
        }

        __syncthreads();
    }

    /* Reduce local_max across all threads */
    atomicMax(out_max_score, local_max);
    __syncthreads();
    
    /* Write position for the winning thread */
    if (local_max == *out_max_score && local_max > 0) {
        *out_max_i = local_max_i;
        *out_max_j = local_max_j;
    }
}

extern "C" {

typedef struct {
    int score;
    int query_end;
    int ref_end;
} sw_gpu_result_t;

void bpd_smith_waterman_gpu_v3(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int block_size,
    sw_gpu_result_t* result)
{
    char *d_query, *d_ref;
    int *d_max_score, *d_max_i, *d_max_j;

    cudaMalloc(&d_query, qlen);
    cudaMalloc(&d_ref, rlen);
    cudaMalloc(&d_max_score, sizeof(int));
    cudaMalloc(&d_max_i, sizeof(int));
    cudaMalloc(&d_max_j, sizeof(int));

    cudaMemcpy(d_query, query, qlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, ref, rlen, cudaMemcpyHostToDevice);
    cudaMemset(d_max_score, 0, sizeof(int));
    cudaMemset(d_max_i, 0, sizeof(int));
    cudaMemset(d_max_j, 0, sizeof(int));

    /* Shared memory: 4 arrays of (qlen+1) ints */
    int smem_size = 4 * (qlen + 1) * sizeof(int);

    /* Single block, all diagonals processed sequentially */
    k_sw_warp_shuffle<<<1, block_size, smem_size>>>(
        d_query, d_ref, qlen, rlen,
        match_score, mismatch_score, gap_open, gap_extend,
        d_max_score, d_max_i, d_max_j);
    cudaDeviceSynchronize();

    int h_score, h_i, h_j;
    cudaMemcpy(&h_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_i, d_max_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_j, d_max_j, sizeof(int), cudaMemcpyDeviceToHost);

    result->score = h_score;
    result->query_end = h_i - 1;
    result->ref_end = h_j - 1;

    cudaFree(d_query); cudaFree(d_ref);
    cudaFree(d_max_score); cudaFree(d_max_i); cudaFree(d_max_j);
}

int bpd_sw_score_gpu_v3(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match, int mismatch,
    int gap_open, int gap_extend,
    int block_size)
{
    sw_gpu_result_t result;
    bpd_smith_waterman_gpu_v3(query, qlen, ref, rlen,
        match, mismatch, gap_open, gap_extend, block_size, &result);
    return result.score;
}

} /* extern "C" */
