%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% matmul_layout.pl — memory-access SHAPE as a BPD parameter on matmul ops.
%%
%% Per Heath: the tile/access shape is NOT a hardcoded dispatch heuristic — it is a PARAMETER
%% of the BPD compute-graph spec that transforms the computation in a STABLE (referee-verified,
%% bit-identical) way toward better memory bandwidth. The matmul op carries its problem shape
%% (m_weight x K, n_tokens); the access_shape parameter is DERIVED from it (or swept), and
%% drives codegen tile selection. Bit-identity holds because tile shape only reorders the
%% accumulation grouping, not the per-output-element math (same FP order within a row).
%%
%% Grounded in measurement (Iyun 2026-06-01, mavchin cpu_profile, cache-cold 134MB):
%%   4_1 tile (RM=4,RN=1, column-major 4-row):  4.02 GB/s, IPC 0.98, L1-miss/dot 12.63  (THRASH)
%%   row-sequential (RM=1, bpd_qdot):            7.79 GB/s, IPC 3.20, L1-miss/dot ~3      (STREAM)
%% ggml decode = num_rows_per_vec_dot=1 (row-sequential). Confirmed via asm_facts on ggml-cpu.so.

:- module(matmul_layout, [
    access_shape/3,            % access_shape(Problem, Shape, Rationale)
    matmul_access_param/2,     % matmul_access_param(matmul_op(MW,K,NT), access(RM,RN))
    predicted_bw/3,            % predicted_bw(access(RM,RN), Regime, GBps)  -- from measurement
    select_tile/2              % select_tile(matmul_op(MW,K,NT), tile(RM,RN))  -- the codegen choice
]).

%% ── The access-shape PARAMETER: (RM, RN) = weight-rows x activation-tokens per tile ──
%% This is the knob. It does NOT change the result (bit-identical) — only the memory walk.

%% ── DERIVE the access shape from the problem shape (the stable transform) ──
%% Decode (mat-VECTOR, n_tokens=1): NO token reuse to amortize multi-row column-major loads,
%%   so row-sequential (RM=1) wins — stream each contiguous weight row fully. (7.79 vs 4.02 GB/s.)
%% Prefill (mat-MAT, n_tokens>=4): many tokens reuse each loaded weight block, so a wider tile
%%   (RM=4,RN=4) amortizes the load and wins on compute density.
matmul_access_param(matmul_op(_MW, _K, NT), access(RM, RN)) :-
    ( NT =:= 1   -> RM = 1, RN = 1                 % decode: row-sequential mat-vector
    ; NT  <  4   -> RM = 4, RN = NT                 % small batch
    ;               RM = 4, RN = 4 ).               % prefill: wide tile

select_tile(Problem, tile(RM, RN)) :-
    matmul_access_param(Problem, access(RM, RN)).

%% ── access_shape/3: human-readable rationale for the chosen shape ──
access_shape(matmul_op(MW,K,1), row_sequential, 'mat-vector (decode): no token reuse; stream each contiguous weight row (RM=1) for max BW') :-
    integer(MW), integer(K).
access_shape(matmul_op(MW,K,NT), wide_tile, 'mat-mat (prefill): token reuse amortizes multi-row loads; wide tile (RM=4,RN=4) for compute density') :-
    integer(MW), integer(K), NT >= 4.

%% ── predicted_bw/3: the MEASURED bandwidth of each access shape (the cost model) ──
%% regime = cold (weights streamed once, like real inference).
predicted_bw(access(1,1),  cold, 7.79).   %% row-sequential = bpd_qdot, measured
predicted_bw(access(4,1),  cold, 4.02).   %% 4-row column-major decode tile, measured (THRASH)
predicted_bw(access(4,4),  cold, 8.21).   %% wide tile, prefill (per-shape aggregate, measured)

%% ── The bit-identity guarantee (why this is a STABLE transform): a matmul's output element
%% out[n,r] = sum_k X[n,k]*W[r,k] is computed by the SAME per-(n,r) reduction regardless of
%% (RM,RN) — the tile only groups which (n,r) pairs are computed together / in what memory
%% order. The FP accumulation order WITHIN a row is unchanged. So tokens stay bit-identical
%% (referee-gated). The parameter transforms ONLY the memory access pattern. ──
