// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* GENERATED from op_expr rmsnorm(1, const(1.0e-5), var) — row reduction + scale.
 * y[i,:] = x[i,:] * rsqrt(mean(x[i,:]^2) + eps) * w[:]. Fact-derived (Iyun). */
extern "C" __global__ void k_rmsnorm(const float* x, const float* w, float* y, int M, int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= M) return;
  const float eps = 1.0e-5f;
  const float* row = x + (long)i*N; float ss = 0.0f;
  for (int j=0;j<N;j++) ss += row[j]*row[j];        // reduction: sum of squares
  float inv = rsqrtf(ss/N + eps);                   // rsqrt(mean + eps)
  float* o = y + (long)i*N;
  for (int j=0;j<N;j++) o[j] = row[j]*inv*w[j];     // scale by inv * weight
}
// LAUNCH: thread_per_row total=M
