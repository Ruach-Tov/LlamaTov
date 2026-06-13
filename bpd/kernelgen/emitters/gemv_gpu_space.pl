%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% gemv_gpu_space.pl — sweepable GPU Q8_0 GEMV parameter space.
%% The GEMV (N=1 decode) analog of gemm_gpu_space.pl's gpu_gemm_point/5. For a
%% matrix-VECTOR product (one activation column) the GEMM 5D space collapses: BN
%% and TN drop out (no column tile, no column micro-tile). The free axes are how
%% a BLOCK of threads co-processes BM output rows while staging a BK-deep strip
%% of the (shared) activation — the mechanism that lifts weight L2-reuse and
%% load coalescing (the measured gap vs ggml: ours sectors_per_load ~0.4-0.5,
%% ggml ~0.19 — ggml gets ~2x better DRAM efficiency via cache reuse).
%%
%% gpu_gemv_point(+BM, +BK, +VEC)
%%   BM  = output rows co-processed per block   in {1,2,4,8,16,32,64}
%%         (1 = the degenerate thread-per-row corner = our CURRENT kernel)
%%   BK  = K-strip depth staged in shared mem    in {32,64,128,256} (mult of 32)
%%   VEC = dp4a load width (int32 words/step)     in {1,2,4}
%%         (how many 4xint8 packed words each thread loads per inner step)
%% Threads/block = BM * (something) — the block has BM "row lanes"; each row's
%% dp4a accumulation is done by the lane(s) assigned to it. The sweep reveals
%% which axis moves sectors_per_load toward ggml's 0.19.
%%
%% This is the spec; q8_0_from_facts projects it to a tiled Q8_0 dp4a kernel
%% (CUDA first target; the spec is backend-portable like gpu_gemm_point).
%% Author: Iyun, 2026-06-12 (per Heath: generate-first, sweep the range)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(gemv_gpu_space, [gpu_gemv_point/3, valid_gpu_gemv/3, gpu_gemv_count/1]).
:- use_module(library(lists)).

gpu_gemv_point(BM, BK, VEC) :-
    member(BM,  [1, 2, 4, 8, 16, 32, 64]),
    member(BK,  [32, 64, 128, 256]),
    member(VEC, [1, 2, 4]),
    valid_gpu_gemv(BM, BK, VEC).

valid_gpu_gemv(BM, BK, VEC) :-
    %% 1. BK is a multiple of 32 (one Q8_0 block = 32 int8 + 1 fp16 scale).
    0 =:= BK mod 32,
    %% 2. VEC int32-words per step must fit BK: BK/32 blocks * 8 words/block,
    %%    VEC divides the 8 words-per-block unroll.
    0 =:= 8 mod VEC,
    %% 3. shared-mem budget: the staged activation strip is BK int8 + (BK/32)
    %%    fp16 scales = BK + BK/16 bytes; must fit sm_61 48KB with margin.
    SMEM is BK + BK // 16,
    SMEM =< 49152,
    %% 4. threads/block = BM row-lanes * 32-thread reduction group (a warp per
    %%    row-strip), capped at 1024 threads/block (sm_61 limit).
    Threads is BM * 32,
    Threads =< 1024.

gpu_gemv_count(N) :-
    aggregate_all(count, gpu_gemv_point(_, _, _), N).
