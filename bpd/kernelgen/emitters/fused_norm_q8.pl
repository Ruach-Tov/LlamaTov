%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% fused_norm_q8.pl — derive a FULLY-FUSED rms_norm -> Q8_0 GEMV kernel.
%%
%% The thesis fusion: rms_norm fact + q8_0_dot fact -> ONE kernel. The activation is
%% normalized, quantized to Q8_0, AND dotted with the weight WITHOUT ever leaving the
%% kernel (no intermediate activation in global/host memory, no separate quant launch,
%% no host roundtrip). This is "no glue between ops" — the ops are fused at the source.
%%
%% k_rmsnorm_q8_gemv(const float* X, const float* NW, const signed char* Wq,
%%                   const __half* Wd, float* Y, int M, int K, float eps)
%%   X[K]   = raw activation (one token)         NW[K] = rms_norm weight
%%   Wq/Wd  = Q8_0 weight (M rows x K)            Y[M]  = output
%%   Per block: the kernel computes rms = rsqrt(mean(X^2)+eps), then for each thread's
%%   output row, normalizes+quantizes X on-the-fly per 32-block and dot-products with Wq.
%%   The normalized+quantized activation lives in SHARED (computed once per block) — the
%%   norm reduction is shared across all output rows.
%%
%% emit_fused_norm_q8(+OutFile)
%% Author: Iyun, 2026-06-09
%% ═══════════════════════════════════════════════════════════════════════════
:- module(fused_norm_q8, [
    emit_fused_norm_q8/1,
    fused_norm_q8_op_expr/1
]).

%% the fused op-graph fact: rms_norm composed into the q8_0 dot's activation load.
fused_norm_q8_op_expr(fused(rmsnorm(1, const(1.0e-5), var), q8_0_dot(block(32), scale(fp16), quant(int8)))).

emit_fused_norm_q8(OutFile) :-
    open(OutFile, write, S),
    format(S, "/* GENERATED fully-fused kernel from op-graph~n", []),
    format(S, " *   fused(rmsnorm(eps), q8_0_dot(block(32),scale(fp16),quant(int8)))~n", []),
    format(S, " * rms_norm folded into the Q8_0 GEMV's activation load: X is normalized,~n", []),
    format(S, " * quantized to Q8_0, and dotted with W — all in ONE kernel, no intermediate~n", []),
    format(S, " * activation leaving the kernel, no host roundtrip, no glue. (Iyun, 2026-06-09) */~n", []),
    format(S, "#include <cuda_fp16.h>~n", []),
    format(S, "#define BLK 32~n", []),
    format(S, "extern \"C\" __global__ void k_rmsnorm_q8_gemv(~n", []),
    format(S, "    const float* X, const float* NW,            // raw activation[K] + rmsnorm weight[K]~n", []),
    format(S, "    const signed char* Wq, const __half* Wd,    // Q8_0 weight [M*K] int8 + [M*nblk] fp16~n", []),
    format(S, "    float* Y, int M, int K, float eps) {~n", []),
    format(S, "  extern __shared__ float sh[];~n", []),
    format(S, "  signed char* xq = (signed char*)sh;           // [K] quantized normed activation~n", []),
    format(S, "  float* xd = (float*)(xq + ((K+15)/16*16));     // [nblk] activation block scales~n", []),
    format(S, "  int nblk = K / BLK; int tid = threadIdx.x, nth = blockDim.x;~n", []),
    format(S, "  // ── PHASE 1: rms over X (mean of squares), block-cooperative ──~n", []),
    format(S, "  __shared__ float ssq[256];~n", []),
    format(S, "  float local = 0.0f;~n", []),
    format(S, "  for (int i = tid; i < K; i += nth) local += X[i]*X[i];~n", []),
    format(S, "  ssq[tid] = local; __syncthreads();~n", []),
    format(S, "  for (int s = nth/2; s > 0; s >>= 1) { if (tid < s) ssq[tid]+=ssq[tid+s]; __syncthreads(); }~n", []),
    format(S, "  float rms = rsqrtf(ssq[0]/K + eps);~n", []),
    format(S, "  __syncthreads();~n", []),
    format(S, "  // ── PHASE 2: normalize + quantize X to Q8_0 into shared (once per block) ──~n", []),
    format(S, "  for (int b = tid; b < nblk; b += nth) {~n", []),
    format(S, "    float amax = 0.0f; float tmp[BLK];~n", []),
    format(S, "    for (int i = 0; i < BLK; i++) {~n", []),
    format(S, "      int idx = b*BLK + i; float v = X[idx]*rms*NW[idx];   // rms_norm folded in~n", []),
    format(S, "      tmp[i] = v; float a = fabsf(v); if (a > amax) amax = a;~n", []),
    format(S, "    }~n", []),
    format(S, "    float d = (amax > 0.0f) ? amax/127.0f : 1.0f; xd[b] = d;~n", []),
    format(S, "    for (int i = 0; i < BLK; i++) {~n", []),
    format(S, "      int q = (int)rintf(tmp[i]/d); q = q<-127?-127:(q>127?127:q); xq[b*BLK+i] = (signed char)q;~n", []),
    format(S, "    }~n", []),
    format(S, "  }~n", []),
    format(S, "  __syncthreads();~n", []),
    format(S, "  // ── PHASE 3: each thread = one output row, dot its W against the shared quantized X ──~n", []),
    format(S, "  for (int row = tid; row < M; row += nth) {~n", []),
    format(S, "    float acc = 0.0f;~n", []),
    format(S, "    for (int b = 0; b < nblk; b++) {~n", []),
    format(S, "      const signed char* wq = Wq + (long)row*K + b*BLK;~n", []),
    format(S, "      const signed char* xqb = xq + b*BLK;~n", []),
    format(S, "      int isum = 0;~n", []),
    format(S, "      for (int i = 0; i < BLK; i++) isum += (int)wq[i] * (int)xqb[i];~n", []),
    format(S, "      acc += __half2float(Wd[(long)row*nblk + b]) * xd[b] * (float)isum;~n", []),
    format(S, "    }~n", []),
    format(S, "    Y[row] = acc;~n", []),
    format(S, "  }~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: one block, nth threads (e.g. 256), shmem = K(int8)+nblk(float)~n", []),
    close(S),
    format("Generated FULLY-FUSED rmsnorm->q8_0 GEMV kernel -> ~w~n", [OutFile]).
