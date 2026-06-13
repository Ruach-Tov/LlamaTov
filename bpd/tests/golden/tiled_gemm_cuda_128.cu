// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* GENERATED from SCHEDULE-IR tiled_gemm(BM=128 BN=128 BK=32 TM=8 TN=4, 512 threads) -> CUDA-C. */
#ifndef GEMM_EPILOGUE
#define GEMM_EPILOGUE (v)
#endif
extern "C" __global__ void k_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  const int BM=128, BN=128, BK=32, TM=8, TN=4;
  __shared__ float As[BM*BK];
  __shared__ float Bs[BK*BN];
  int tid = threadIdx.x, nthreads = blockDim.x;
  int tRow = tid / (BN/TN);
  int tCol = tid % (BN/TN);
  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;
  float acc[TM][TN];
  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j]=0.0f;
  for(int k0=0; k0<K; k0+=BK) {
    for(int idx=tid; idx<BM*BK; idx+=nthreads) {
      int r=idx/BK, c=idx%BK; int gr=blockRow+r, gc=k0+c;
      As[idx] = (gr<M && gc<K) ? A[(long)gr*K + gc] : 0.0f;
    }
    for(int idx=tid; idx<BK*BN; idx+=nthreads) {
      int r=idx/BN, c=idx%BN; int gr=k0+r, gc=blockCol+c;
      Bs[idx] = (gr<K && gc<N) ? B[(long)gr*N + gc] : 0.0f;
    }
    __syncthreads();
    for(int kk=0; kk<BK; kk++) {
      float a_reg[TM], b_reg[TN];
      for(int i=0;i<TM;i++) a_reg[i] = As[(tRow*TM+i)*BK + kk];
      for(int j=0;j<TN;j++) b_reg[j] = Bs[kk*BN + (tCol*TN+j)];
      for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
    }
    __syncthreads();
  }
  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) {
    int gr=blockRow+tRow*TM+i, gc=blockCol+tCol*TN+j;
    if(gr<M && gc<N) { float v = acc[i][j]; C[(long)gr*N + gc] = GEMM_EPILOGUE; }
  }
}
