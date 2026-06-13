// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* GENERATED from op_expr q8_0_dot(block(32), scale(fp16), quant(int8)) — dp4a (Pascal __dp4a 4-way int8 dot, sm_61+ native int8 path).
 * per block: (xd*yd) * sum_i(xq[i]*yq[i]). Algebraically == dequant-then-dot
 * (verified bit-exact, memory 4727ad11). Fact-derived (Iyun, 2026-06-08). */
#include <cuda_fp16.h>
extern "C" __global__ void k_q8_0_gemv(
    const signed char* Wq, const __half* Wd,
    const signed char* Xq, const __half* Xd,
    float* Y, int M, int K) {
  int row = blockIdx.x*blockDim.x + threadIdx.x; if (row >= M) return;
  int nblk = K / 32; float acc = 0.0f;
  for (int b = 0; b < nblk; b++) {
    const int* wq4 = (const int*)(Wq + (long)row*K + b*32);  // 8 packed int8x4
    const int* xq4 = (const int*)(Xq + b*32);
    int isum = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) isum = __dp4a(wq4[j], xq4[j], isum);  // 4-way int8 dot
    float wd = __half2float(Wd[(long)row*nblk + b]);
    float xd = __half2float(Xd[b]);
    acc += (wd * xd) * (float)isum;
  }
  Y[row] = acc;
}
// LAUNCH: thread_per_row total=M
