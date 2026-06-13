// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_sw_gpu_batch.cu — Batched GPU SW with CORRECT double-buffered shared memory
 *
 * FIX: Double buffering for H_col, E, F arrays.
 * Read from buffer[d%2], write to buffer[(d+1)%2].
 * Swap after __syncthreads().
 *
 * The bug was: on the same anti-diagonal, thread (i,j) writes H_col[i]
 * before thread (i+1,j-1) reads H_col[i] for h_up. Double buffering
 * eliminates this race completely.
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_sw_batch.so bench/bpd_sw_gpu_batch.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

__global__ void k_sw_batch(
    const char* __restrict__ all_queries,
    const int* __restrict__ query_offsets,
    const int* __restrict__ query_lengths,
    const char* __restrict__ ref,
    int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int max_qlen,
    int* __restrict__ scores)
{
    int align_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int qlen = query_lengths[align_idx];
    int q_offset = query_offsets[align_idx];
    const char* query = all_queries + q_offset;

    /* Double-buffered shared memory:
     * H_col[2][max_qlen+1] — current and previous column H values
     * E[2][max_qlen+1] — current and previous E values
     * F[2][max_qlen+1] — current and previous F values
     * H_diag[max_qlen+1] — diagonal H values (two diags behind)
     */
    extern __shared__ int smem[];
    int stride = max_qlen + 1;
    int* H_col_0 = smem;                        /* buffer 0 */
    int* H_col_1 = smem + stride;               /* buffer 1 */
    int* E_0 = smem + 2 * stride;
    int* E_1 = smem + 3 * stride;
    int* F_0 = smem + 4 * stride;
    int* F_1 = smem + 5 * stride;
    int* H_diag = smem + 6 * stride;

    /* Initialize all buffers to zero */
    for (int i = tid; i <= qlen; i += block_size) {
        H_col_0[i] = 0; H_col_1[i] = 0;
        E_0[i] = 0; E_1[i] = 0;
        F_0[i] = 0; F_1[i] = 0;
        H_diag[i] = 0;
    }
    __syncthreads();

    int local_max = 0;
    int total_diags = qlen + rlen - 1;

    for (int d = 0; d < total_diags; d++) {
        int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
        int i_max = (d + 1 < qlen) ? d + 1 : qlen;
        int num_cells = i_max - i_min + 1;

        /* Select read/write buffers based on parity */
        int* H_read = (d % 2 == 0) ? H_col_0 : H_col_1;
        int* H_write = (d % 2 == 0) ? H_col_1 : H_col_0;
        int* E_read = (d % 2 == 0) ? E_0 : E_1;
        int* E_write = (d % 2 == 0) ? E_1 : E_0;
        int* F_read = (d % 2 == 0) ? F_0 : F_1;
        int* F_write = (d % 2 == 0) ? F_1 : F_0;

        /* Save H_diag from WRITE buffer BEFORE overwriting.
         * H_write currently holds values from diagonal d-2.
         * H_diag[i] = H_write[i-1] = H[i-1][j-1] from two diags ago. */
        for (int c = tid; c < num_cells; c += block_size) {
            int i = i_min + c;
            int j = d + 2 - i;
            if (i >= 1 && i <= qlen && j >= 1 && j <= rlen)
                H_diag[i] = H_write[i - 1];
        }
        __syncthreads();

        for (int c = tid; c < num_cells; c += block_size) {
            int i = i_min + c;
            int j = d + 2 - i;
            if (i < 1 || i > qlen || j < 1 || j > rlen) continue;

            int s = (query[i-1] == ref[j-1]) ? match_score : mismatch_score;

            /* READ from saved and previous diagonal's buffers */
            int h_diag = H_diag[i];                  /* H[i-1, j-1] from d-2 */
            int h_left = H_read[i];                  /* H[i, j-1] — previous diag */
            int h_up = H_read[i - 1];                /* H[i-1, j] — previous diag */
            int e_prev = E_read[i];                  /* E[i, j-1] — previous diag */
            int f_prev = F_read[i - 1];              /* F[i-1, j] — previous diag */

            /* E[i][j] = max(H[i][j-1] - gap_open, E[i][j-1] - gap_extend) */
            int e_val = (h_left - gap_open > e_prev - gap_extend) ?
                         h_left - gap_open : e_prev - gap_extend;

            /* F[i][j] = max(H[i-1][j] - gap_open, F[i-1][j] - gap_extend) */
            int f_val = (h_up - gap_open > f_prev - gap_extend) ?
                         h_up - gap_open : f_prev - gap_extend;

            /* H[i][j] = max(0, H[i-1][j-1] + s, E[i][j], F[i][j]) */
            int h = h_diag + s;
            if (e_val > h) h = e_val;
            if (f_val > h) h = f_val;
            if (h < 0) h = 0;

            if (h > local_max) local_max = h;

            /* WRITE to current diagonal's buffers */
            E_write[i] = e_val;
            F_write[i] = f_val;
            H_write[i] = h;

            /* H_diag is now saved BEFORE the loop — no per-cell update needed */
        }
        __syncthreads();
    }

    /* Reduce local_max across threads */
    __shared__ int block_max;
    if (tid == 0) block_max = 0;
    __syncthreads();
    atomicMax(&block_max, local_max);
    __syncthreads();

    if (tid == 0) {
        scores[align_idx] = block_max;
    }
}

extern "C" {

void bpd_sw_batch_gpu(
    const char** queries, const int* query_lengths, int n_queries,
    const char* ref, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int block_size,
    int* scores)
{
    int total_qlen = 0;
    int max_qlen = 0;
    for (int i = 0; i < n_queries; i++) {
        total_qlen += query_lengths[i];
        if (query_lengths[i] > max_qlen) max_qlen = query_lengths[i];
    }

    char* concat = (char*)malloc(total_qlen);
    int* offsets = (int*)malloc(n_queries * sizeof(int));
    int off = 0;
    for (int i = 0; i < n_queries; i++) {
        offsets[i] = off;
        memcpy(concat + off, queries[i], query_lengths[i]);
        off += query_lengths[i];
    }

    char *d_queries, *d_ref;
    int *d_offsets, *d_lengths, *d_scores;

    cudaMalloc(&d_queries, total_qlen);
    cudaMalloc(&d_ref, rlen);
    cudaMalloc(&d_offsets, n_queries * sizeof(int));
    cudaMalloc(&d_lengths, n_queries * sizeof(int));
    cudaMalloc(&d_scores, n_queries * sizeof(int));

    cudaMemcpy(d_queries, concat, total_qlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, ref, rlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, n_queries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, query_lengths, n_queries * sizeof(int), cudaMemcpyHostToDevice);

    /* Shared memory: 7 arrays of (max_qlen+1) ints
     * H_col×2 + E×2 + F×2 + H_diag = 7 */
    int smem_size = 7 * (max_qlen + 1) * sizeof(int);

    k_sw_batch<<<n_queries, block_size, smem_size>>>(
        d_queries, d_offsets, d_lengths,
        d_ref, rlen,
        match_score, mismatch_score, gap_open, gap_extend,
        max_qlen, d_scores);
    cudaDeviceSynchronize();

    cudaMemcpy(scores, d_scores, n_queries * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_queries); cudaFree(d_ref);
    cudaFree(d_offsets); cudaFree(d_lengths); cudaFree(d_scores);
    free(concat); free(offsets);
}

} /* extern "C" */
