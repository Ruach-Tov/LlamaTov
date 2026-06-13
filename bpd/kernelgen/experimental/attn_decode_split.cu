// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// k_attn_decode_split2: flash-decode split-K, LOCAL-max single-pass (no redundant global-max).
// Fixes the v1 killer: v1 had every split recompute QK^T over ALL L for the global max, so score
// compute was (NSPLIT+1)x. Here each split computes its OWN range's scores ONCE, finds its LOCAL
// max, exps locally, accumulates partial Z and partial O. The combine rescales across splits using
// the per-split max (the online-softmax / flash combine). Since the V-sum is ALREADY being
// re-canonicalized (declared order attn_decode_split), the local-max rescale costs no ULP beyond
// the already-accepted re-parenthesization — it's part of the declared order.
//
// reduction_order(attn_decode_split, splits(NSPLIT), per_split(local_max, sequential_left_fold),
//                 combine(flash_rescale, split_index_order)).

extern "C" __global__ void k_attn_decode_split(
    const float* Q, const float* K, const float* V,
    const int* len_ptr,
    float* partM,   // [nh*NSPLIT]      per-split local max
    float* partZ,   // [nh*NSPLIT]      per-split denominator (sum exp(s - localmax))
    float* partO,   // [nh*NSPLIT*hd]   per-split numerator (sum exp*V, un-normalized by Z)
    int hd, int nh, int nkv, float scale) {
  const int MAXT = %(MAXT)d;
  const int NSPLIT = %(NSPLIT)d;
  int h = blockIdx.x %% nh;
  int sp = blockIdx.x / nh;
  if (h >= nh || sp >= NSPLIT) return;
  int d = threadIdx.x;
  int L = *len_ptr;
  int rep = nh / nkv; int hk = h / rep;
  const float* qh = Q + h*hd;
  int per = (L + NSPLIT - 1) / NSPLIT;
  int p0 = sp*per; int p1 = p0 + per; if (p1 > L) p1 = L;
  int rangeN = p1 - p0;

  extern __shared__ float sh[];      // scores for this split's range (indexed t-p0)
  __shared__ float red[1024];

  // compute THIS range's scores ONCE, store in sh, find LOCAL max.
  float m = -1e30f;
  for (int t = p0 + d; t < p1; t += blockDim.x) {
    const float* Kt = K + ((long)t*nkv + hk)*hd;
    float s = 0.0f;
    for (int i = 0; i < hd; i++) s += qh[i] * Kt[i];
    s *= scale;
    sh[t - p0] = s;          // store raw score; exp later (after we know local max)
    m = fmaxf(m, s);
  }
  red[d] = m; __syncthreads();
  for (int s2 = blockDim.x/2; s2 > 0; s2 >>= 1) { if (d < s2) red[d] = fmaxf(red[d], red[d+s2]); __syncthreads(); }
  float lmx = (rangeN > 0) ? red[0] : -1e30f; __syncthreads();
  if (d == 0) partM[h*NSPLIT + sp] = lmx;

  // exp against LOCAL max, store back in sh
  for (int t = d; t < rangeN; t += blockDim.x) sh[t] = __expf(sh[t] - lmx);
  __syncthreads();
  // partial Z (per-thread tree, same as original)
  float ls = 0.0f;
  for (int t = d; t < rangeN; t += blockDim.x) ls += sh[t];
  red[d] = ls; __syncthreads();
  for (int s2 = blockDim.x/2; s2 > 0; s2 >>= 1) { if (d < s2) red[d] += red[d+s2]; __syncthreads(); }
  if (d == 0) partZ[h*NSPLIT + sp] = red[0];
  // partial O (sequential left-fold over this range)
  if (d < hd) {
    float acc = 0.0f;
    for (int t = p0; t < p1; t++) {
      const float* Vt = V + ((long)t*nkv + hk)*hd;
      acc += sh[t - p0] * Vt[d];
    }
    partO[(h*NSPLIT + sp)*hd + d] = acc;
  }
}

// combine: flash-style rescale across splits in split-index order.
// global max M = max_sp partM[sp].  For each split: w_sp = exp(partM[sp] - M).
// Z = sum_sp w_sp * partZ[sp].   O[d] = sum_sp w_sp * partO[sp][d].   OUT = O / Z.
extern "C" __global__ void k_attn_decode_combine(
    const float* partM, const float* partZ, const float* partO,
    float* OUT, int hd, int nh) {
  const int NSPLIT = %(NSPLIT)d;
  int h = blockIdx.x; if (h >= nh) return;
  int d = threadIdx.x; if (d >= hd) return;
  // global max over splits (associative, exact)
  float M = -1e30f;
  for (int sp = 0; sp < NSPLIT; sp++) M = fmaxf(M, partM[h*NSPLIT + sp]);
  float Z = 0.0f, acc = 0.0f;
  for (int sp = 0; sp < NSPLIT; sp++) {
    float w = __expf(partM[h*NSPLIT + sp] - M);   // rescale weight for this split
    Z   += w * partZ[h*NSPLIT + sp];
    acc += w * partO[(h*NSPLIT + sp)*hd + d];
  }
  OUT[h*hd + d] = acc / Z;
}
