%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% fusion_cost.pl — the PROFITABILITY dimension for the fusion framework.
%%
%% The existing fusion machinery (symbolic_fusion / fusion_rules / iterative_fusion)
%% decides whether a fusion is VALID — region-compatible, non-escaping, class-
%% compatible. But validity is not profitability. Fusion trades away an
%% intermediate's MEMORY TRAFFIC in exchange for RECOMPUTATION of that
%% intermediate inside the consumer. Whether that trade wins is SHAPE-DEPENDENT.
%%
%% Worked example that motivated this module (measured on the Tesla P4):
%%   conv2d as im2col(produce col[K,Nn]) -> gemm(consume col).
%%   FUSING im2col into the GEMM's B-load eliminates the 215MB col write
%%   (~5.6ms) but RECOMPUTES each col entry once per M-block (2x at Cout=128,
%%   8x at Cout=512) with heavy per-element index math (3 div + 3 mod). The
%%   recompute cost EXCEEDED the memory saved -> fused 22.8ms vs 2-stage 15.8ms.
%%   The fusion was VALID but NOT PROFITABLE; a generic pass must DECLINE it.
%%
%% Contrast: elementwise-chain / matmul-epilogue fusions are ALWAYS profitable
%% because the recomputed intermediate is consumed exactly once per output
%% element (recompute factor 1.0) and the per-element work is trivial. Those
%% rules need no cost gate. GENERATOR-PROLOGUE fusions (im2col, broadcast-expand,
%% gather) DO, because the generated tile is reloaded by multiple output blocks.
%%
%% Author: Iyun, 2026-06-08 (the cost dimension, from the conv im2col finding)

:- module(fusion_cost, [
    fusion_profitable/3,        % fusion_profitable(+Rule, +Bindings, -Verdict)
    recompute_factor/2,         % recompute_factor(+FusionShape, -Factor)
    bytes_saved/2,              % bytes_saved(+IntermediateShape, -Bytes)
    recompute_cost/4            % recompute_cost(+Shape, +GenKind, +Factor, -Cost)
]).

%% ── Per-element recompute weight, CALIBRATED against measured P4 results. ──
%% Expressed as the byte-time cost of regenerating ONE element, relative to the
%% per-element cost of MATERIALIZING it (one coalesced write + read = 8 bytes f32).
%% Calibration (im2col conv, measured): one full regen pass cost ~1.13x the
%% materialization it replaced — because the inline index math (3 div + 3 mod)
%% is ALU-serial AND the recompute's global load of x is NON-coalesced (gather),
%% unlike the materialized col's coalesced GEMM read. So an index-heavy generator
%% regen is MORE expensive per element than just materializing+reading it; fusion
%% can only win when the recompute_factor is < 1 (impossible) OR the generator is
%% trivial (broadcast/copy: re-read is a coalesced cache hit, cheaper than a
%% round-trip to global memory).
%% Weight units: byte-equivalents per regenerated element.
gen_regen_bytes(im2col,     9).   % > 8 (materialize cost): index ALU + gather load
gen_regen_bytes(gather,     9).
gen_regen_bytes(broadcast,  2).   % cheap re-read (cache-friendly) << 8
gen_regen_bytes(copy,       2).
gen_regen_bytes(elementwise,2).   % the always-profitable case
gen_regen_bytes(_,          6).   % default: assume close to break-even

%% ── recompute_factor(+fusion(Producer, ConsumerTiling), -Factor) ──
%% How many times the fused intermediate gets regenerated vs materialized once.
%% For a generator feeding a tiled GEMM B-input: the B-tile is recomputed once
%% per block-ROW of the output (= ceil(M/BM)) — every M-block re-loads the same
%% B columns. So Factor = number of output M-blocks.
recompute_factor(gemm_b_input(M, BM), Factor) :-
    Factor is max(1, (M + BM - 1) // BM).
%% Epilogue / elementwise-chain: consumed once per element -> factor 1.
recompute_factor(epilogue, 1).
recompute_factor(elementwise_chain, 1).

%% ── bytes_saved: the intermediate we DON'T materialize (write + read). ──
%% Materializing costs one coalesced write (producer) + one coalesced read
%% (consumer) = 8 bytes/elem (f32 round-trip).
bytes_saved(elems(NElems), Bytes) :- Bytes is 8 * NElems.

%% ── recompute_cost: total cost of regenerating the intermediate in the fused
%%    kernel = recompute_factor full regenerations, each costing
%%    gen_regen_bytes/elem (calibrated byte-equivalents). ──
recompute_cost(elems(NElems), GenKind, Factor, CostBytes) :-
    gen_regen_bytes(GenKind, W),
    CostBytes is Factor * NElems * W.

%% ── fusion_profitable(+Rule, +Bindings, -Verdict) ──
%% Verdict = profitable(MarginBytes) | unprofitable(DeficitBytes) | always.
%% always: epilogue/elementwise-chain (recompute factor 1, trivial per-elem).
fusion_profitable(epilogue_matmul_elementwise, _, always) :- !.
fusion_profitable(elementwise_chain, _, always) :- !.
fusion_profitable(layout_transparent, _, always) :- !.   % pure rewrite, no recompute

%% Generator-prologue (im2col, broadcast-expand, gather): cost-gated.
%% Bindings carry: gen_kind, intermediate_elems, consumer_m, consumer_bm.
%% PROFITABLE iff the total recompute cost < the materialization saved:
%%   Factor * NElems * gen_regen_bytes  <  8 * NElems
%%   <=> Factor * gen_regen_bytes < 8.
%% For im2col (regen 9 > 8): unprofitable at ANY factor>=1 (matches measurement).
%% For broadcast (regen 2): profitable up to factor 3 (8/2=4 -> factor<=3).
fusion_profitable(generator_prologue,
                  binds(GenKind, NElems, M, BM), Verdict) :-
    bytes_saved(elems(NElems), Saved),
    recompute_factor(gemm_b_input(M, BM), Factor),
    recompute_cost(elems(NElems), GenKind, Factor, TotalRegen),
    ( TotalRegen < Saved
    -> Margin is Saved - TotalRegen, Verdict = profitable(Margin)
    ;  Deficit is TotalRegen - Saved, Verdict = unprofitable(Deficit) ).
