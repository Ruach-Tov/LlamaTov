// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// Register-heavy vecmat kernel — proof of concept
// Eliminates shared memory for the input vector by tiling into registers.
//
// Original k_vecmat: loads A[K] into shared memory, all threads broadcast-read.
// This version: each thread loads a CHUNK of A into registers, accumulates
// partial sums, then the results are identical (each thread computes one output col).
//
// For K=2048, CHUNK=8: each thread holds 8 floats in registers = 32 bytes.
// 256 threads × 8 elements = 2048 = full vector covered per tile.
// Wait — that's not right. Each thread needs ALL K elements for its column.
//
// Actually, for vecmat (M=1), every thread computing a different output column
// needs the ENTIRE input vector A[0..K-1]. The shared memory approach loads A
// once cooperatively and broadcasts. The register approach would require each
// thread to load the ENTIRE vector into its own registers — K floats = K*4 bytes.
// For K=2048: 8192 bytes = 2048 registers. Way too many (255 max per thread).
//
// So pure register caching of the FULL vector doesn't work for K=2048.
//
// BUT: we can TILE over K. Process the vector in chunks of CHUNK elements.
// Each thread loads CHUNK elements of A into registers, multiplies with
// CHUNK elements of its B column, accumulates. Then loads the next chunk.
//
// This eliminates shared memory entirely AND eliminates __syncthreads.
// The A values are loaded from L1/L2 cache (warm after first thread reads them).
//
// Trade-off: each thread issues its own global loads for A (vs cooperative load
// to shared memory). But L1 cache handles the broadcast — 256 threads reading
// the same A[k] will hit L1 after the first thread loads it.

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define CHUNK 8  // Elements of A loaded into registers per tile iteration

// Register-tiled vecmat: no shared memory, no __syncthreads
extern "C"
__global__ void k_vecmat_reg(const float * __restrict__ A,
                              const float * __restrict__ B,
                              float * __restrict__ C,
                              int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;

    // Process A in chunks of CHUNK elements
    int k = 0;
    for (; k + CHUNK <= K; k += CHUNK) {
        // Load CHUNK elements of A into registers
        float a0 = A[k + 0];
        float a1 = A[k + 1];
        float a2 = A[k + 2];
        float a3 = A[k + 3];
        float a4 = A[k + 4];
        float a5 = A[k + 5];
        float a6 = A[k + 6];
        float a7 = A[k + 7];

        // Multiply with B column elements (coalesced across threads for each k)
        sum += a0 * B[(k + 0) * N + col];
        sum += a1 * B[(k + 1) * N + col];
        sum += a2 * B[(k + 2) * N + col];
        sum += a3 * B[(k + 3) * N + col];
        sum += a4 * B[(k + 4) * N + col];
        sum += a5 * B[(k + 5) * N + col];
        sum += a6 * B[(k + 6) * N + col];
        sum += a7 * B[(k + 7) * N + col];
    }
    // Handle remainder
    for (; k < K; k++) {
        sum += A[k] * B[k * N + col];
    }

    C[col] = sum;
}

// Original shared-memory vecmat for comparison
extern "C"
__global__ void k_vecmat_smem(const float * __restrict__ A,
                               const float * __restrict__ B,
                               float * __restrict__ C,
                               int K, int N) {
    extern __shared__ float sA[];
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        sA[i] = A[i];
    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += sA[k] * B[k * N + col];
    C[col] = sum;
}

// Benchmark harness
extern "C"
void bench_vecmat_variants(int K, int N, int iters) {
    float *h_A, *h_B, *h_C_reg, *h_C_smem;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(K * sizeof(float));
    h_B = (float*)malloc(K * N * sizeof(float));
    h_C_reg = (float*)malloc(N * sizeof(float));
    h_C_smem = (float*)malloc(N * sizeof(float));

    // Initialize with small values
    for (int i = 0; i < K; i++) h_A[i] = (float)(i % 7) * 0.01f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(i % 11) * 0.01f;

    cudaMalloc(&d_A, K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    cudaMemcpy(d_A, h_A, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int smem = K * sizeof(float);

    // Warmup
    k_vecmat_reg<<<blocks, BLOCK_SIZE>>>(d_A, d_B, d_C, K, N);
    k_vecmat_smem<<<blocks, BLOCK_SIZE, smem>>>(d_A, d_B, d_C, K, N);
    cudaDeviceSynchronize();

    // Benchmark register version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 10; i++)
        k_vecmat_reg<<<blocks, BLOCK_SIZE>>>(d_A, d_B, d_C, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; i++)
        k_vecmat_reg<<<blocks, BLOCK_SIZE>>>(d_A, d_B, d_C, K, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float reg_ms;
    cudaEventElapsedTime(&reg_ms, start, stop);

    cudaMemcpy(h_C_reg, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Warmup
    for (int i = 0; i < 10; i++)
        k_vecmat_smem<<<blocks, BLOCK_SIZE, smem>>>(d_A, d_B, d_C, K, N);
    cudaDeviceSynchronize();

    // Benchmark shared memory version
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; i++)
        k_vecmat_smem<<<blocks, BLOCK_SIZE, smem>>>(d_A, d_B, d_C, K, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float smem_ms;
    cudaEventElapsedTime(&smem_ms, start, stop);

    cudaMemcpy(h_C_smem, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness
    float max_diff = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_C_reg[i] - h_C_smem[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("K=%d, N=%d, %d iterations:\n", K, N, iters);
    printf("  Register: %.3f ms (%.3f ms/iter)\n", reg_ms, reg_ms / iters);
    printf("  SharedMem: %.3f ms (%.3f ms/iter)\n", smem_ms, smem_ms / iters);
    printf("  Speedup: %.2fx\n", smem_ms / reg_ms);
    printf("  Max diff: %.6e (correctness check)\n", max_diff);
    printf("\n");

    free(h_A); free(h_B); free(h_C_reg); free(h_C_smem);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    // Check CUDA device
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (sm_%d%d, %d MB)\n\n", prop.name, prop.major, prop.minor,
           (int)(prop.totalGlobalMem / (1024*1024)));

    // Test on LlamaTov-relevant shapes
    printf("=== Vecmat Register vs SharedMem Benchmark ===\n\n");

    // Q/K/V projection: [1,2048] x [2048,2048]
    bench_vecmat_variants(2048, 2048, 1000);

    // FFN up/gate: [1,2048] x [2048,8192]
    bench_vecmat_variants(2048, 8192, 1000);

    // FFN down: [1,8192] x [8192,2048]
    bench_vecmat_variants(8192, 2048, 1000);

    return 0;
}
