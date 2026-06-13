%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% fusible.pl — Declarative fusion legality predicate
%%
%% Determines whether two operations can be legally fused into a single kernel,
%% based on their operation classification, semantic regions, and platform constraints.

/** <module> Declarative Fusion Legality

Four fusion rules expressed as Prolog clauses. Adding a new fusion
pattern = adding one clause. Each rule specifies when two adjacent
operations in a compute graph can be fused into a single kernel.

Rules:
  1. Epilogue fusion — spatial op followed by elementwise (matmul+relu)
  2. Elementwise chain — elementwise followed by elementwise (silu+mul = SwiGLU)
  3. Layout transparent — reshape elimination between compatible ops
  4. Fused epilogue — epilogue to an already-fused op (matmul+bias+relu)

@author Ruach Tov Collective
@license RTAAL-1.0
@see graph_optimizer.pl for the fixed-point iteration that applies these rules
@see valid_tile.pl for constraint-based tile selection
*/

:- module(fusible, [
    fusible/3,
    fusible_pair/4
]).

:- use_module(auto_fuser, [classify_op/2]).
:- use_module(region_inference, [infer_region_from_facts/5]).

%% ────────────────────────────────────────────────────────────────────
%% fusible(+GraphFacts, +Op1, +Op2)
%% ────────────────────────────────────────────────────────────────────
%% True if Op1 and Op2 can be legally fused given the graph facts.
%% This is a wrapper that delegates to fusible_pair/4, extracting kinds.

%! fusible(+GraphFacts, ?Op1, ?Op2) is nondet.
%  True if Op1 and Op2 can be fused in the compute graph.
%  Enumerates all fusible pairs via backtracking.
fusible(GraphFacts, Op1, Op2) :-
    member(op_kind(Op1, _Kind1), GraphFacts),
    member(op_kind(Op2, _Kind2), GraphFacts),
    fusible_pair(GraphFacts, Op1, Op2, fusion(_RuleName, [Op1, Op2], _EqClass)).

%% ────────────────────────────────────────────────────────────────────
%% fusible_pair(+GraphFacts, +Op1, +Op2, -Fusion)
%% ────────────────────────────────────────────────────────────────────
%% Determines if Op1 and Op2 are fusible and returns the fusion term.

%% Rule 1: Spatial operation followed by elementwise operation (epilogue fusion).
%% Op1's output is consumed by Op2.
fusible_pair(GraphFacts, Op1, Op2, fusion(epilogue_matmul_elementwise, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, Kind1), GraphFacts),
    member(op_kind(Op2, Kind2), GraphFacts),
    classify_op(Kind1, spatial),
    classify_op(Kind2, elementwise),
    %% Op1 output is consumed by Op2
    member(op_output(Op1, Intermediate), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Intermediate, Inputs2),
    %% Op2 must read Intermediate elementwise or broadcast
    infer_region_from_facts(GraphFacts, Op2, Intermediate, read, region(RegionType, _)),
    (RegionType = elementwise ; RegionType = broadcast).

%% Rule 2: Elementwise chain fusion.
fusible_pair(GraphFacts, Op1, Op2, fusion(elementwise_chain, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, Kind1), GraphFacts),
    member(op_kind(Op2, Kind2), GraphFacts),
    classify_op(Kind1, elementwise),
    classify_op(Kind2, elementwise),
    %% Op1 output is consumed by Op2
    member(op_output(Op1, Intermediate), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Intermediate, Inputs2),
    %% Op2 must read Intermediate elementwise or broadcast
    infer_region_from_facts(GraphFacts, Op2, Intermediate, read, region(RegionType, _)),
    (RegionType = elementwise ; RegionType = broadcast).

%% Rule 4: Epilogue elementwise fusion to already fused matmul
fusible_pair(GraphFacts, Op1, Op2, fusion(epilogue_matmul_elementwise, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, fused(Kind1A, _Kind1B)), GraphFacts),
    %% Check if the original base op was spatial
    classify_op(Kind1A, spatial),
    member(op_kind(Op2, Kind2), GraphFacts),
    classify_op(Kind2, elementwise),
    member(op_output(Op1, Intermediate), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Intermediate, Inputs2),
    infer_region_from_facts(GraphFacts, Op2, Intermediate, read, region(RegionType, _)),
    (RegionType = elementwise ; RegionType = broadcast).

%% Rule 3: Layout transparent fusion (reshape elimination).
%% Op1 is reshape, Op2 is a consumer.
fusible_pair(GraphFacts, Op1, Op2, fusion(layout_transparent, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, ggml_reshape_3d), GraphFacts),
    member(op_output(Op1, Reshaped), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Reshaped, Inputs2).

%% Rule 5: Scale absorption — matmul followed by scalar multiply (QK^T scaling).
%% When matmul → scale(1/sqrt(d)) → softmax, the scale can be absorbed
%% into the matmul by pre-scaling Q. This converts post_scaled to pre_scaled.
%%
%% This fusion is NOT bit-exact: it changes floating-point evaluation order.
%% The result is scale_application_path=pre_scaled vs post_scaled.
%% Both are correct; they produce different bits.
%%
%% The substrate can emit EITHER form:
%%   post_scaled (unfused): matmul → scale → softmax  (matches ggml)
%%   pre_scaled (fused):    matmul(Q/sqrt(d), K) → softmax  (fewer ops)
%%
%% This is a SWEEPABLE PARAMETER, not a fixed fusion decision.
fusible_pair(GraphFacts, Op1, Op2, fusion(scale_absorption, [Op1, Op2], parameterized(scale_application_path))) :-
    member(op_kind(Op1, Kind1), GraphFacts),
    (Kind1 = ggml_mul_mat ; Kind1 = matmul),
    member(op_kind(Op2, Kind2), GraphFacts),
    (Kind2 = ggml_scale ; Kind2 = ggml_mul ; Kind2 = scale ; Kind2 = mul),
    %% Op1 output consumed by Op2
    member(op_output(Op1, Intermediate), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Intermediate, Inputs2),
    %% Op2 has a scalar input (the scale factor)
    member(op_inputs(Op2, AllInputs), GraphFacts),
    length(AllInputs, 2).  %% binary op: tensor + scalar

%% Rule 6: Spatial + reduction + elementwise (the Conv+BN+Act pattern).
%% BatchNorm is a reduction (per-channel statistics), but when its
%% parameters are known (inference mode), it reduces to an affine
%% transform: y = alpha * x + beta. This is a CHEAP epilogue
%% that can be folded into the spatial op's output loop.
%%
%% Detects: conv2d → batchnorm → silu (YOLO CBS)
%%          conv2d → batchnorm → relu
%%          conv2d → groupnorm → tanh
%%          etc.
fusible_pair(GraphFacts, Op1, Op3, 
    fusion(spatial_reduction_elementwise, [Op1, Op2, Op3], bit_exact)) :-
    member(op_kind(Op1, Kind1), GraphFacts),
    member(op_kind(Op2, Kind2), GraphFacts),
    member(op_kind(Op3, Kind3), GraphFacts),
    classify_op(Kind1, spatial),
    classify_op(Kind2, reduction),
    classify_op(Kind3, elementwise),
    %% Chain: Op1 → Op2 → Op3
    member(op_output(Op1, T1), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(T1, Inputs2),
    member(op_output(Op2, T2), GraphFacts),
    member(op_inputs(Op3, Inputs3), GraphFacts),
    member(T2, Inputs3).

%% Rule 7: Spatial + reduction (Conv+BN without activation).
%% BatchNorm in inference mode is alpha*x + beta — a cheap affine
%% epilogue even without a following activation.
%% Detects: conv2d → batchnorm (no activation after)
fusible_pair(GraphFacts, Op1, Op2,
    fusion(spatial_reduction, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, Kind1), GraphFacts),
    member(op_kind(Op2, Kind2), GraphFacts),
    classify_op(Kind1, spatial),
    classify_op(Kind2, reduction),
    member(op_output(Op1, T1), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(T1, Inputs2).
