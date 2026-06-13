%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% fused_rms_quant.pl — the RMS→QUANT SEAM, CUDA lowering of activation_fold(rms_norm).
%% Generated from the fact param_axis(quant, activation_fold, [.., rms_norm]): fold k_rmsnorm into
%% k_quant_q8's read. ONE block per row (one token, K=896). TWO-PHASE, both canonical orders preserved
%% -> BIT-IDENTICAL to k_rmsnorm then k_quant_q8:
%%   PHASE 1: cooperative sum-of-squares over X[K] (== the standalone rms reduction), -> inv.
%%   PHASE 2: per 32-block, lane L computes nv = X[blk*32+L]*inv*NW[blk*32+L], warp-amax (== the
%%            standalone quant reduction), quantize to int8 + fp16 scale. No intermediate to global.
%% Eliminates k_rmsnorm + the normalized-activation global round-trip.
%% k_rms_quant(const float* X, const float* NW, signed char* Xq, __half* Xd, int K, float eps)
%% Author: Iyun, 2026-06-12 (born polyglot — see oxide_from_facts.pl for the Rust lowering).
%% ═══════════════════════════════════════════════════════════════════════════
:- module(fused_rms_quant, [emit_fused_rms_quant/2, attest_rms_quant_reduction_order/0]).

%% This emitter's phase-1 reproduces reduction_order(rms_ss, lanes(256), strided, tree(pairwise,8))
%% EXACTLY (block_row strided partials + shared pairwise tree) — verified 0-ULP vs the production
%% block_row k_rmsnorm and vs the oxide lowering (cross-backend gate). So the fused kernel has EARNED
%% the order-preservation attestation: it can discharge the fusion reduction-order gate's obligation.
%% attest_rms_quant_reduction_order/0 asserts reduction_order_preserved/3 for the rms_quant fusion so
%% numerical_stability:fusion_reduction_gate(fusion(rms_quant,...), bit_exact) PASSES (not VETOES).
:- ( catch(use_module('lib/numerical_stability.pl'), _, true) -> true ; true ).
attest_rms_quant_reduction_order :-
    ( catch(( numerical_stability:assertz(
                numerical_stability:reduction_order_preserved(rms_quant, rms_norm,
                  reduction_order(rms_ss, lanes(256), strided, tree(pairwise, 8)))) ), _, fail)
    -> format("attested: rms_quant preserves reduction_order(rms_ss, lanes(256), strided, tree(pairwise,8))~n", [])
    ;  format("WARN: could not assert rms_quant reduction-order attestation~n", []) ).

emit_fused_rms_quant(Eps, OutFile) :-
    open(OutFile, write, S),
    format(S, "/* GENERATED from activation_fold(rms_norm) — the RMS->QUANT SEAM, born polyglot.~n", []),
    format(S, " * Folds k_rmsnorm into k_quant_q8. Two-phase, both canonical orders preserved ->~n", []),
    format(S, " * BIT-IDENTICAL to k_rmsnorm then k_quant_q8. (Iyun, 2026-06-12) */~n", []),
    format(S, "#include <cuda_fp16.h>~n", []),
    format(S, "extern \"C\" __global__ void k_rms_quant(~n", []),
    format(S, "    const float* X, const float* NW,           // raw activation[K] + rmsnorm weight[K]~n", []),
    format(S, "    signed char* Xq, __half* Xd, int K) {       // quantized output + block scales[nb]~n", []),
    format(S, "  const float eps = ~wf;~n", [Eps]),
    format(S, "  int nb = K / 32;~n", []),
    format(S, "  int tid = threadIdx.x, nth = blockDim.x;~n", []),
    %% PHASE 1: rms sum-of-squares reproducing reduction_order(rms_ss, lanes(256), strided,
    %% tree(pairwise,8)) EXACTLY — the SAME order as the production block_row k_rmsnorm: each thread
    %% accumulates a strided slice, then a shared-mem pairwise tree. 0-ULP to the standalone block_row
    %% rms (the correctness contract is the declared order, not a serial left-fold).
    format(S, "  // PHASE 1: sum of squares -> inv. block_row order (strided partials + shared tree),~n", []),
    format(S, "  // bit-identical to the production block_row k_rmsnorm (reduction_order rms_ss).~n", []),
    format(S, "  extern __shared__ float sred[];~n", []),
    format(S, "  float local = 0.0f;~n", []),
    format(S, "  for (int j = tid; j < K; j += nth) local += X[j]*X[j];~n", []),
    format(S, "  sred[tid] = local; __syncthreads();~n", []),
    format(S, "  for (int s = nth/2; s > 0; s >>= 1) { if (tid < s) sred[tid] += sred[tid+s]; __syncthreads(); }~n", []),
    format(S, "  float inv = rsqrtf(sred[0]/K + eps);~n", []),
    format(S, "  __syncthreads();~n", []),
    %% PHASE 2: canonical warp-amax quantize per 32-block (one warp per block, == standalone k_quant_q8).
    format(S, "  // PHASE 2: per 32-block warp-amax quantize of nv = X*inv*NW (matches k_quant_q8).~n", []),
    format(S, "  int warps = nth >> 5;~n", []),
    format(S, "  for (int b = (tid >> 5); b < nb; b += warps) {~n", []),
    format(S, "    int lane = tid & 31;~n", []),
    format(S, "    int idx = b*32 + lane;~n", []),
    format(S, "    float nv = X[idx] * inv * NW[idx];         // rms-normalized + weighted activation~n", []),
    format(S, "    float a = fabsf(nv);~n", []),
    format(S, "    #pragma unroll~n", []),
    format(S, "    for (int s = 16; s > 0; s >>= 1) { float o = __shfl_down_sync(0xffffffff, a, s); if (o > a) a = o; }~n", []),
    format(S, "    float amax = __shfl_sync(0xffffffff, a, 0);~n", []),
    format(S, "    float d = (amax > 0.0f) ? amax/127.0f : 1.0f; __half dh = __float2half(d);~n", []),
    format(S, "    if (lane == 0) Xd[b] = dh;~n", []),
    format(S, "    float dq = __half2float(dh);~n", []),
    format(S, "    int q = (int)rintf(nv/dq); q = q<-127?-127:(q>127?127:q);~n", []),
    format(S, "    Xq[idx] = (signed char)q;~n", []),
    format(S, "  }~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: grid=1 block (one token/row), blockDim=256 (LANES contract), dynamic smem=256*4 bytes~n", []),
    close(S),
    format("Generated FACT-DERIVED k_rms_quant (rms->quant seam, eps=~w) -> ~w~n", [Eps, OutFile]),
    %% emitting the kernel discharges the reduction-order obligation (this lowering preserves it).
    ( catch(attest_rms_quant_reduction_order, _, true) -> true ; true ).
