// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_sw_profile.cu — Standalone profiling target for ncu
 *
 * Single binary that runs one SW alignment for profiling.
 * Build: nvcc -O3 -arch=sm_61 -o /tmp/sw_profile bench/bpd_sw_profile.cu
 * Profile: ncu --set full /tmp/sw_profile
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same kernel as bpd_smith_waterman_gpu.cu */
__global__ void k_sw_antidiag(
    const char* __restrict__ query,
    const char* __restrict__ ref,
    int* __restrict__ H,
    int* __restrict__ E,
    int* __restrict__ F,
    int qlen, int rlen,
    int d,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int* __restrict__ max_score,
    int* __restrict__ max_i,
    int* __restrict__ max_j)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
    int i_max = (d + 1 < qlen) ? d + 1 : qlen;
    int num_cells = i_max - i_min + 1;
    if (tid >= num_cells) return;

    int i = i_min + tid;
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

int main() {
    int qlen = 1000, rlen = 1000;
    int matrix_size = (qlen + 1) * (rlen + 1) * sizeof(int);

    /* Generate random sequences */
    char* query = (char*)malloc(qlen);
    char* ref = (char*)malloc(rlen);
    srand(42);
    const char bases[] = "ACGT";
    for (int i = 0; i < qlen; i++) query[i] = bases[rand() % 4];
    for (int i = 0; i < rlen; i++) ref[i] = bases[rand() % 4];

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

    int block_size = 128;
    int total_diags = qlen + rlen - 1;

    printf("Profiling SW: %dx%d, %d anti-diagonals, block=%d\n",
           qlen, rlen, total_diags, block_size);

    for (int d = 0; d < total_diags; d++) {
        int i_min = (d + 2 - rlen > 1) ? d + 2 - rlen : 1;
        int i_max = (d + 1 < qlen) ? d + 1 : qlen;
        int num_cells = i_max - i_min + 1;
        if (num_cells <= 0) continue;
        int blocks = (num_cells + block_size - 1) / block_size;
        k_sw_antidiag<<<blocks, block_size>>>(
            d_query, d_ref, d_H, d_E, d_F,
            qlen, rlen, d, 2, -1, 3, 1,
            d_max_score, d_max_i, d_max_j);
    }
    cudaDeviceSynchronize();

    int h_score;
    cudaMemcpy(&h_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Score: %d\n", h_score);

    cudaFree(d_query); cudaFree(d_ref);
    cudaFree(d_H); cudaFree(d_E); cudaFree(d_F);
    cudaFree(d_max_score); cudaFree(d_max_i); cudaFree(d_max_j);
    free(query); free(ref);
    return 0;
}
