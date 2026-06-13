// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* relu_nvcc.cu — nvcc CUDA-C ReLU reference for P4 (same-device perf baseline).
 * Identical work + timing methodology as the cuda-oxide relu_bench:
 * 16M f32, WARMUP=10, ITERS=100, CUDA-event timing, report ms/iter + GB/s.
 * This is the apples-to-apples vendor-toolchain reference on the SAME P4
 * (since PyTorch 2.7 cannot run on Pascal sm_61).
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        c[i] = v > 0.0f ? v : 0.0f;
    }
}

int main() {
    const int N = 16 * 1024 * 1024;
    const int WARMUP = 10, ITERS = 100;
    size_t bytes = (size_t)N * sizeof(float);

    float* x_host = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) x_host[i] = (float)i * 0.001f - 8000.0f;

    float *x_dev, *c_dev;
    cudaMalloc(&x_dev, bytes);
    cudaMalloc(&c_dev, bytes);
    cudaMemcpy(x_dev, x_host, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;

    // warmup
    for (int i = 0; i < WARMUP; i++) relu_kernel<<<grid, block>>>(x_dev, c_dev, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) relu_kernel<<<grid, block>>>(x_dev, c_dev, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    double per_iter = total_ms / ITERS;
    double gbps = (double)bytes * 2.0 / (per_iter * 1e-3) / 1e9;

    printf("=== nvcc CUDA-C ReLU on Tesla P4 (sm_61) ===\n\n");
    printf("N = %d (%zu MB per buffer)\n", N, bytes / (1024 * 1024));
    printf("kernel time : %.4f ms/iter  (%d iters, %d warmup)\n", per_iter, ITERS, WARMUP);
    printf("bandwidth   : %.1f GB/s  (read+write %.0f MB)\n", gbps, (double)bytes * 2 / 1e6);

    // correctness sample
    float* c_host = (float*)malloc(bytes);
    cudaMemcpy(c_host, c_dev, bytes, cudaMemcpyDeviceToHost);
    int diffs = 0;
    for (int i = 0; i < N; i += N / 1000) {
        float want = x_host[i] > 0.0f ? x_host[i] : 0.0f;
        if (*(uint32_t*)&c_host[i] != *(uint32_t*)&want) diffs++;
    }
    printf("correctness : %s (sampled 1000 elems, bitwise)\n",
           diffs == 0 ? "*** 0 diffs ***" : "DIVERGENT");

    cudaFree(x_dev); cudaFree(c_dev); free(x_host); free(c_host);
    return 0;
}
