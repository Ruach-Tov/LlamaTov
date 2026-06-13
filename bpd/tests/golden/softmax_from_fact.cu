// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* GENERATED from op_expr softmax(1, var) — row max/sum reductions.
 * y[i,:] = exp(x[i,:]-max) / sum(exp(x[i,:]-max)). Fact-derived (Iyun). */
extern "C" __global__ void k_softmax(const float* x, float* y, int M, int N, float scale) {
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= M) return;
  const float* row = x + (long)i*N; float* o = y + (long)i*N;
  float mx = -3.4e38f;
  for (int j=0;j<N;j++){ float v=row[j]*scale; if(v>mx) mx=v; }   // reduction: row max
  float sum = 0.0f;
  for (int j=0;j<N;j++){ float e=expf(row[j]*scale-mx); o[j]=e; sum+=e; } // exp-shift + sum
  float inv = 1.0f/sum;
  for (int j=0;j<N;j++) o[j] *= inv;                              // normalize
}
// LAUNCH: thread_per_row total=M
