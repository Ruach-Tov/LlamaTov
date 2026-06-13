// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_lcp_gpu.cu — GPU LCP array construction
 *
 * Given a suffix array SA and the original text, compute the LCP
 * array in parallel. Uses the phi-array approach (parallelizable
 * variant of Kasai's algorithm):
 *   1. Build phi[SA[i]] = SA[i-1] (the "previous suffix" mapping)
 *   2. For each position i in text order, compute LCP by comparing
 *      text[i..] with text[phi[i]..] — this is embarrassingly parallel
 *   3. Permute results back to SA order
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_lcp_gpu.so bench/bpd_lcp_gpu.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>

/* Step 1: Build phi array — phi[SA[i]] = SA[i-1] */
__global__ void k_build_phi(
    const int* __restrict__ SA,
    int* __restrict__ phi,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0) {
        phi[SA[0]] = -1;  /* no predecessor for first suffix */
    } else {
        phi[SA[i]] = SA[i - 1];
    }
}

/* Step 2: Compute PLCP (permuted LCP) — compare text[i..] with text[phi[i]..] 
 * Each position is independent — embarrassingly parallel */
__global__ void k_compute_plcp(
    const char* __restrict__ text,
    const int* __restrict__ phi,
    int* __restrict__ plcp,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (phi[i] == -1) {
        plcp[i] = 0;
        return;
    }

    int len = 0;
    int j = phi[i];
    while (i + len < n && j + len < n && text[i + len] == text[j + len]) {
        len++;
    }
    plcp[i] = len;
}

/* Step 3: Permute PLCP back to SA order → LCP */
__global__ void k_plcp_to_lcp(
    const int* __restrict__ SA,
    const int* __restrict__ plcp,
    int* __restrict__ LCP,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    LCP[i] = plcp[SA[i]];
}

extern "C" {

void bpd_build_lcp_gpu(
    const char* text, int n,
    const int* SA,
    int* LCP,
    int block_size)
{
    int num_blocks = (n + block_size - 1) / block_size;

    char* d_text;
    int *d_SA, *d_phi, *d_plcp, *d_LCP;

    cudaMalloc(&d_text, n);
    cudaMalloc(&d_SA, n * sizeof(int));
    cudaMalloc(&d_phi, n * sizeof(int));
    cudaMalloc(&d_plcp, n * sizeof(int));
    cudaMalloc(&d_LCP, n * sizeof(int));

    cudaMemcpy(d_text, text, n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SA, SA, n * sizeof(int), cudaMemcpyHostToDevice);

    k_build_phi<<<num_blocks, block_size>>>(d_SA, d_phi, n);
    k_compute_plcp<<<num_blocks, block_size>>>(d_text, d_phi, d_plcp, n);
    k_plcp_to_lcp<<<num_blocks, block_size>>>(d_SA, d_plcp, d_LCP, n);

    cudaDeviceSynchronize();
    cudaMemcpy(LCP, d_LCP, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_text); cudaFree(d_SA);
    cudaFree(d_phi); cudaFree(d_plcp); cudaFree(d_LCP);
}

} /* extern "C" */
