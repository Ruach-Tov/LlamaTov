// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* tcheck_rect — rectangular GEMM test harness for C[MxN] = A[MxK] * B[KxN].
 * The general case (square is M=N=K). Loads A.bin (M*K), B.bin (K*N) for verify;
 * dumps Csw.bin (M*N). Perf mode times `iters` launches.
 *
 * usage: tcheck_rect <cubin> <NTH> <BM> <BN> <M> <N> <K> <mode> [iters]
 *   mode = "verify" (load A.bin/B.bin, run, dump Csw.bin)
 *        | "perf"   (random data, time iters launches, print ms)
 *
 * Launch geometry (matches emit_gemm_tiled_rect): grid.x=ceil(N/BN),
 * grid.y=ceil(M/BM), block=NTH threads. Kernel ABI: k_gemm(A,B,C, int M,int N,int K).
 * By: Iyun, 2026-06-08
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)

int main(int c, char** v) {
  if (c < 9) { printf("usage: tcheck_rect cubin NTH BM BN M N K mode [iters]\n"); return 1; }
  const char* cub = v[1];
  int NTH = atoi(v[2]), BM = atoi(v[3]), BN = atoi(v[4]);
  int M = atoi(v[5]), N = atoi(v[6]), K = atoi(v[7]);
  const char* mode = v[8];
  int iters = c > 9 ? atoi(v[9]) : 30;

  size_t bA = (size_t)M * K * 4;
  size_t bB = (size_t)K * N * 4;
  size_t bC = (size_t)M * N * 4;

  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0));
  CUcontext ctx; CK(cuCtxCreate(&ctx, 0, d));
  CUmodule m; CK(cuModuleLoad(&m, cub));
  CUfunction fn; CK(cuModuleGetFunction(&fn, m, "k_gemm"));
  CUdeviceptr A, B, C; CK(cuMemAlloc(&A, bA)); CK(cuMemAlloc(&B, bB)); CK(cuMemAlloc(&C, bC));

  void* args[] = { &A, &B, &C, &M, &N, &K };
  unsigned gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;

  if (strcmp(mode, "verify") == 0) {
    float* Ah = (float*)malloc(bA);
    float* Bh = (float*)malloc(bB);
    float* Ch = (float*)malloc(bC);
    FILE* fa = fopen("A.bin", "rb"); fread(Ah, 4, (size_t)M * K, fa); fclose(fa);
    FILE* fb = fopen("B.bin", "rb"); fread(Bh, 4, (size_t)K * N, fb); fclose(fb);
    CK(cuMemcpyHtoD(A, Ah, bA)); CK(cuMemcpyHtoD(B, Bh, bB));
    CK(cuLaunchKernel(fn, gx, gy, 1, NTH, 1, 1, 0, 0, args, 0));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(Ch, C, bC));
    FILE* fo = fopen("Csw.bin", "wb"); fwrite(Ch, 4, (size_t)M * N, fo); fclose(fo);
    printf("verify ok: M=%d N=%d K=%d grid=(%u,%u) block=%d\n", M, N, K, gx, gy, NTH);
  } else { // perf
    // random fill on device via host (small overhead, amortized over iters)
    float* Ah = (float*)malloc(bA); float* Bh = (float*)malloc(bB);
    for (size_t i = 0; i < (size_t)M * K; i++) Ah[i] = (float)(i % 13) * 0.01f;
    for (size_t i = 0; i < (size_t)K * N; i++) Bh[i] = (float)(i % 7) * 0.02f;
    CK(cuMemcpyHtoD(A, Ah, bA)); CK(cuMemcpyHtoD(B, Bh, bB));
    // warmup
    CK(cuLaunchKernel(fn, gx, gy, 1, NTH, 1, 1, 0, 0, args, 0));
    CK(cuCtxSynchronize());
    CUevent t0, t1; CK(cuEventCreate(&t0, 0)); CK(cuEventCreate(&t1, 0));
    CK(cuEventRecord(t0, 0));
    for (int it = 0; it < iters; it++)
      CK(cuLaunchKernel(fn, gx, gy, 1, NTH, 1, 1, 0, 0, args, 0));
    CK(cuEventRecord(t1, 0)); CK(cuEventSynchronize(t1));
    float ms = 0; CK(cuEventElapsedTime(&ms, t0, t1)); ms /= iters;
    double gflops = 2.0 * M * N * K / (ms * 1e-3) / 1e9;
    printf("perf: %.4f ms  %.1f GFLOPS  (M=%d N=%d K=%d grid=(%u,%u) thr=%d)\n",
           ms, gflops, M, N, K, gx, gy, NTH);
  }
  return 0;
}
