%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% gemm_gpu_space.pl — sweepable GPU GEMM parameter space (the GPU analog of
%% bpd/lib/gemm_kernel.pl's gemm_pattern/6 for CPU/AVX).
%%
%% v2 (register-blocked): the classic high-performance SGEMM shape. A block of
%% (BN/TN)x(BM/TM) threads computes a BM x BN output tile; each thread computes
%% a TM x TN register micro-tile; the K dimension is processed in BK-deep strips
%% loaded cooperatively into shared memory. This is the structure that closes the
%% gap to cuBLAS (register reuse: each smem value feeds TM or TN FMAs).
%%
%%   CPU gemm_pattern         ->  GPU gemm_point
%%   P,Q (L2/L1 cache blocks) ->  BM,BN (block output tile) + BK (K strip)
%%   UM,UN (ymm micro-kernel) ->  TM,TN (register micro-tile per thread)
%%   SimdWidth (AVX lanes)    ->  (vectorized loads — follow-up)
%%
%% gpu_gemm_point(+BM, +BN, +BK, +TM, +TN)
%%   BM,BN = block output tile (rows,cols of C per block)   in {32,64,128}
%%   BK    = K-strip depth (shared tile K dim)              in {8,16,32}
%%   TM,TN = register micro-tile per thread                in {2,4,8}
%% Threads/block = (BM/TM) * (BN/TN). Each thread does TM*TN FMAs per K step.
%%
%% Author: Iyun, 2026-06-07 (v2 register-blocked, per Heath)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(gemm_gpu_space, [gpu_gemm_point/5, valid_gpu_gemm/5, gpu_gemm_count/1]).

gpu_gemm_point(BM, BN, BK, TM, TN) :-
    member(BM, [32, 64, 128]),
    member(BN, [32, 64, 128]),
    member(BK, [8, 16, 32]),
    member(TM, [2, 4, 8]),
    member(TN, [2, 4, 8]),
    valid_gpu_gemm(BM, BN, BK, TM, TN).

valid_gpu_gemm(BM, BN, BK, TM, TN) :-
    %% 1. micro-tile must evenly tile the block output tile
    0 =:= BM mod TM,
    0 =:= BN mod TN,
    NThreads is (BM // TM) * (BN // TN),
    %% 2. threads/block in [64, 1024] and a multiple of 32 (full warps)
    NThreads >= 64, NThreads =< 1024,
    0 =:= NThreads mod 32,
    %% 3. cooperative load must cover the smem tiles with the thread count:
    %%    A tile is BM x BK, B tile is BK x BN; each loadable in whole steps
    0 =:= (BM * BK) mod NThreads,
    0 =:= (BK * BN) mod NThreads,
    %% 4. shared memory budget: (BM*BK + BK*BN) f32 <= 48KB/block (P4)
    (BM*BK + BK*BN) * 4 =< 49152,
    %% 5. register budget: TM*TN accumulators + TM+TN operands <= ~96 (headroom under 255)
    TM*TN + TM + TN =< 96.

gpu_gemm_count(N) :-
    findall(_, gpu_gemm_point(_,_,_,_,_), L), length(L, N).
