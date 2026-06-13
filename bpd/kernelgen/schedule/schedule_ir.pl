%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% schedule_ir.pl — a SCHEDULE IR between the algorithm-AST (op_expr) and the
%% per-backend lowering. The Halide/TVM "algorithm / schedule" separation, in the
%% TRANSFORM stage: op_expr says WHAT (the math), the schedule says HOW (tile,
%% thread-map, stage-to-shared). ONE schedule lowers to ALL backends.
%%
%% Scope (incremental, measure-first): starts with the REDUCTION class only —
%% the tiled "one-block-per-row + warp-shuffle" schedule we proved in cuda-c
%% (143 GB/s, parity with torch). If this lowers to BOTH cuda-c (reproducing the
%% existing tuned kernel) AND MLIR (matching its perf), the shared-tiling-layer
%% idea is validated and generalizes to pool/conv/matmul.
%%
%% THE SCHEDULE-NEUTRAL PRIMITIVES (the whole point — same concept, N syntaxes):
%%   block_map(IV, Bound)           — one threadblock per IV (blockIdx / ctaid / block_idx)
%%   thread_strided(IV, Lo,Hi,Step) — each thread strides over [Lo,Hi) by blockDim
%%   accumulate(Acc, Init, Comb, V) — fold V into Acc (the op_expr combine)
%%   warp_shuffle(Acc, Comb)        — intra-warp tree reduce (shfl.down / nvvm.shfl)
%%   stage_shared(Buf, Size, Vals)  — cross-warp via shared mem (__shared__ / memref)
%%   barrier                        — __syncthreads / gpu.barrier
%%   guarded_store(Dst, IV, Bound, V)
%%
%% Author: Iyun, 2026-06-08 (the shared-tiling-layer prototype, Heath's design)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(schedule_ir, [tile_schedule/3, schedule_combine/3]).
:- use_module(library(lists)).

%% op_expr/2 (the algorithm AST) is provided by the facts module
%% (bpd/lib/robust_op_match.pl), which callers use_module BEFORE this one — the
%% same load-order pattern the cuda-c/mlir emitters use. The lint 'undefined
%% predicate' note for op_expr/2 is expected (resolved at call time, not load).

%% tile_schedule(+Op, +ScheduleName, -Schedule)
%% Produce the schedule (a list of primitives) for Op under a named schedule.
%% For reductions: the proven tiled-reduce schedule.
tile_schedule(Op, tiled_row_reduce, schedule(reduce, Kind, Prims)) :-
    op_expr(Op, axis_reduce(Kind, _Axis, _Body)),
    schedule_combine(Kind, Init, Comb),
    Prims = [
        %% grid: one block per row r; thread t within block
        block_map(r, 'R'),
        %% each thread folds a strided slice of the row's C columns (coalesced)
        accumulate_strided(acc, Init, Comb, row(r), c, 'C'),
        %% intra-warp tree reduce of acc
        warp_shuffle(acc, Comb),
        %% cross-warp: lane0 of each warp -> shared[wid]; barrier; warp0 re-reduces
        stage_shared(sh, 32, acc),
        barrier,
        warp0_reduce(acc, sh, Comb),
        %% final (mean divides by C); lane0 stores out[r]
        finalize(Kind, acc, 'C', res),
        guarded_store(out, r, 'R', res)
    ].

%% per-kind init + combine, shared by ALL backends (the algorithm's fold)
schedule_combine(sum,  "0.0",             add).
schedule_combine(mean, "0.0",             add).
schedule_combine(max,  "-3.40282347e+38", maxf).
schedule_combine(min,  "3.40282347e+38",  minf).

%% ── TILED GEMM schedule (the L3-critical kernel) ────────────────────────────
%% tile_schedule(Op, tiled_gemm(BM,BN,BK,TM,TN), Schedule): the register-blocked
%% shared-memory GEMM schedule we autotuned (BM128 BN128 BK32 TM8 TN4 = 41% cuBLAS).
%% Backend-neutral primitives -> cuda-c reproduces the tuned rect GEMM; MLIR gets
%% the same tiling (the L3 perf-parity path). One schedule, both backends.
tile_schedule(Op, tiled_gemm(BM,BN,BK,TM,TN), schedule(gemm, contract, Prims)) :-
    op_expr(Op, axis_reduce(_, _, _)) ; true,  % matmul is a reduction over k
    Prims = [
        %% 2D grid: one block per BMxBN output tile; thread owns a TMxTN micro-tile
        block_map_2d(bi, bj, BM, BN),
        thread_tile(TM, TN),
        %% accumulators zeroed
        register_init(acc, TM, TN, "0.0"),
        %% loop over K in BK-wide strips, staging A,B tiles to shared each step
        k_loop(kt, 'K', BK, [
            stage_shared(as_tile, BM, BK, from_a),
            stage_shared(bs_tile, BK, BN, from_b),
            barrier,
            %% inner: accumulate the TMxTN micro-tile from the shared tiles
            register_accumulate(acc, as_tile, bs_tile, TM, TN, BK),
            barrier
        ]),
        %% guarded store with the fusion epilogue hook
        guarded_store_2d(c_out, acc, TM, TN, 'M', 'N', epilogue)
    ].

%% ═══════════════════════════════════════════════════════════════════════════
%% MOVE 4 (thesis fidelity): complete the schedule vocabulary so EVERY op class is
%% tileable via a Prolog schedule term (parallel to tiled_gemm / tiled_reduce).
%% These are the backend-neutral tiling primitives; lowerings follow per-backend.
%% ═══════════════════════════════════════════════════════════════════════════

%% tiled_elementwise(VEC, GRID) — vectorized grid-strided elementwise (the simplest
%% tiling: float4 loads, grid-stride loop). The epilogue-fusion tail rides on this.
tile_schedule(Op, tiled_elementwise(VEC, grid_stride), schedule(elementwise, map, Prims)) :-
    op_is_elementwise(Op),
    Prims = [
        grid_stride_loop(i, n),
        vectorize(VEC),                 % float4 when n%4==0, else scalar
        apply_expr(Op, i),              % the op_expr lowered at element i
        epilogue_hook(i)                % the folded tail (Move 3) inlined here
    ].

%% tiled_pool(BH, BW, EPILOGUE) — pooling tiled by output tile, epilogue-capable.
tile_schedule(Op, tiled_pool(BH, BW), schedule(pool, window_reduce, Prims)) :-
    op_is_pool(Op),
    Prims = [
        block_map_2d(bi, bj, BH, BW),
        thread_per_output,
        window_reduce(Op),              % max/avg over the pooling window
        guarded_store_2d(out, epilogue) % epilogue hook (backend-neutral, Move 3)
    ].

%% tiled_conv(BM, BN, BK, TM, TN) — conv-as-implicit-GEMM, reuses the gemm tiling.
%% The im2col-free implicit GEMM: conv windows map to the GEMM's A-tile loads.
tile_schedule(Op, tiled_conv(BM, BN, BK, TM, TN), schedule(conv, implicit_gemm, Prims)) :-
    op_is_conv(Op),
    Prims = [
        block_map_2d(bi, bj, BM, BN),
        thread_tile(TM, TN),
        register_init(acc, TM, TN, "0.0"),
        implicit_gemm_kloop(BK, im2col_inline),  % conv windows as A-tile, no materialized im2col
        guarded_store_2d(out, epilogue)
    ].

%% tiled_flash(WPB, BC, DPL) — the flash schedule, registered in the schedule-IR
%% family. The detailed levers live in flash_attention.pl:flash_attn_schedule/2
%% (Move 1); this entry makes flash a first-class member of the tiling vocabulary
%% so the autotuner enumerates it alongside the others.
tile_schedule(Op, tiled_flash(WPB, BC, DPL), schedule(flash, online_softmax, Prims)) :-
    op_is_attention(Op),
    Prims = [
        warp_per_query(WPB),
        d_split_warp(DPL),              % acc across 32 lanes -> register-resident
        kv_tile_shared(BC),             % block-shared K/V tiles
        online_softmax_accumulate,      % running m,l + rescale; no [SxS]
        vectorize_when(DPL == 4)        % float4 when D=128
    ].

%% op-class predicates (loose — match common op_expr kinds + ggml names)
op_is_elementwise(Op) :- member(Op, [bpd_relu,bpd_silu,bpd_gelu,bpd_tanh,bpd_scaling,
                                     bpd_add,bpd_mul,bpd_scalar_add]) ; true.
op_is_pool(Op) :- ( atom(Op), sub_atom(Op,_,_,_,pool) -> true ; member(Op,[bpd_maxpool2d,bpd_avgpool2d]) ).
op_is_conv(Op) :- ( atom(Op), sub_atom(Op,_,_,_,conv) -> true ; member(Op,[bpd_conv2d]) ).
op_is_attention(Op) :- member(Op, [bpd_attention, attention, flash_attn]) ; true.
