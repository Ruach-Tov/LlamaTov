// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* GENERATED from op_expr q8_0_dot(block(32), scale(fp16), quant(int8)) — scalar (int32 accumulation loop, portable).
 * per block: (xd*yd) * sum_i(xq[i]*yq[i]). Algebraically == dequant-then-dot
 * (verified bit-exact, memory 4727ad11). Fact-derived (Iyun, 2026-06-08). */
#include <cuda_fp16.h>
extern "C" __global__ void k_q8_0_gemv(
    const signed char* Wq, const __half* Wd,   // weight: [M*K/32 blocks] int8 + fp16 scales
    const signed char* Xq, const __half* Xd,   // activation: [K/32 blocks] int8 + scales
    float* Y, int M, int K) {                  // Y[M], K = cols (mult of 32)
  int row = blockIdx.x*blockDim.x + threadIdx.x; if (row >= M) return;
  int nblk = K / 32; float acc = 0.0f;
  for (int b = 0; b < nblk; b++) {
    const signed char* wq = Wq + (long)row*K + b*32;
    const signed char* xq = Xq + b*32;
    int isum = 0;
    for (int i = 0; i < 32; i++) isum += (int)wq[i] * (int)xq[i];  // int32 accumulate
    float wd = __half2float(Wd[(long)row*nblk + b]);
    float xd = __half2float(Xd[b]);
    acc += (wd * xd) * (float)isum;            // fp scale per block
  }
  Y[row] = acc;
}
// LAUNCH: thread_per_row total=M
