// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* GENERATED from SCHEDULE-IR (tiled_row_reduce, sum) -> CUDA-C. One schedule, N backends. */
extern "C" __global__ void k_reduce(const float* x, float* out, int R, int C) {
  int r = blockIdx.x;  if (r >= R) return;
  int t = threadIdx.x;
  const float* row = x + (long)r * C;
  float acc = 0.0;
  for (int c = t; c < C; c += blockDim.x) { float v = row[c]; acc = acc + v; }
  for (int o = 16; o > 0; o >>= 1) { float v = __shfl_down_sync(0xffffffff, acc, o); acc = acc + v; }
  __shared__ float sh[32];
  int lane = t & 31, wid = t >> 5;
  if (lane == 0) sh[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = (t < (blockDim.x + 31) / 32) ? sh[lane] : (0.0);
    for (int o = 16; o > 0; o >>= 1) { float v = __shfl_down_sync(0xffffffff, acc, o); acc = acc + v; }
    if (lane == 0) out[r] = acc;
  }
}
