%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% symbolic_fusion.pl — Stage 1 symbolic fusion correctness analyzer.
%%
%% TRIVIAL CASE per SYMBOLIC_FUSION_CORRECTNESS.md Section 9.2:
%% prove fusion validity for matmul → bias_add by checking:
%%   (1) Op1's write region matches Op2's read region (shape compatibility)
%%   (2) Intermediate tensor doesn't escape to other consumers
%%   (3) Operation classes are compatible (matmul → elementwise)
%%
%% This is region-level dependence analysis applied to compute graphs —
%% the substrate-honest minimum that proves a fusion is STRUCTURALLY
%% safe. Numerical-equivalence and IO-cost are separate layers
%% deferred to later stages.
%%
%% Heath's discipline: write a trivial case end-to-end, then extend.

:- module(symbolic_fusion, [
    fusion_valid/2,
    region_matches/2,
    no_escape/2,
    op_class_compatible/2,
    subset_list/2
]).

%% ────────────────────────────────────────────────────────────────────
%% Region vocabulary
%% ────────────────────────────────────────────────────────────────────
%%
%% Region terms describe what part of a tensor a particular op
%% reads or writes. Per Hall's symbolic subscript analysis at coarse
%% granularity — sufficient for the regular access patterns of
%% transformer compute graphs.
%%
%%   region(elementwise, Shape)
%%       - Op produces/consumes one output per position
%%       - Shape is a list of dim symbols, e.g. [n_tokens, n_embd]
%%
%%   region(matmul_output, [M, N])
%%       - Op produces an M×N tensor from a matmul-shaped computation
%%
%%   region(matmul_input, [M, K])
%%       - Op consumes an M×K tensor as one of the matmul operands
%%
%%   region(matmul_weights, [K, N])
%%       - Op consumes a K×N weight matrix in a matmul
%%
%%   region(broadcast, Shape)
%%       - Op consumes a tensor of Shape, broadcast across other dims
%%
%%   region(row_reduction, Shape)
%%       - Op reduces across the last dimension of Shape (e.g., softmax)

%% region_matches(+RegionA, +RegionB) succeeds when two regions describe
%% the same set of elements in compatible access patterns. For fusion
%% the typical case: elementwise output of Op1 matches elementwise input
%% of Op2 when shapes are equal.

region_matches(region(elementwise, Shape), region(elementwise, Shape)) :- !.

region_matches(region(matmul_output, [M, N]), region(elementwise, [M, N])) :- !.
region_matches(region(matmul_output, [M, N]), region(elementwise, [M, N | _])) :- !.

region_matches(region(broadcast, BcastShape), region(elementwise, FullShape)) :-
    % Broadcast is compatible if the bcast shape is a suffix of the full shape.
    append(_, BcastShape, FullShape), !.

%% Anything else is incompatible at this stage — conservative.
%% (Future stages can add more region-matching rules.)

%% ────────────────────────────────────────────────────────────────────
%% Op-class compatibility for fusion
%% ────────────────────────────────────────────────────────────────────
%%
%% Two ops can be fused if their classes admit fusion AT ALL. This is
%% the existing fusion analyzer's job — we reuse those rules here.
%% For the trivial case we only handle epilogue fusion: matmul-class
%% op followed by elementwise op.

op_class_compatible(matmul, elementwise).
op_class_compatible(elementwise, elementwise).
op_class_compatible(elementwise, layout).
op_class_compatible(layout, elementwise).
op_class_compatible(layout, layout).
% Future extensions:
%   op_class_compatible(normalization, elementwise).
%   op_class_compatible(elementwise, normalization).  % only sometimes; needs care
%   Conservative: do NOT make matmul→matmul compatible (different fusion class)
%   Conservative: do NOT make reduction→anything compatible (needs streaming)

%% ────────────────────────────────────────────────────────────────────
%% Escape analysis (region-level)
%% ────────────────────────────────────────────────────────────────────
%%
%% no_escape(+Tensor, +AllowedConsumerOps) succeeds when Tensor is only
%% consumed by ops in AllowedConsumerOps. Any consumer NOT in the
%% allowed list means the tensor escapes and must materialize to memory.

no_escape(Tensor, AllowedOps) :-
    findall(Op, op_input(Op, Tensor), Consumers),
    subset_list(Consumers, AllowedOps).

%% subset_list(+L1, +L2) — every element of L1 is in L2
subset_list([], _).
subset_list([H | T], L) :-
    member(H, L),
    subset_list(T, L).

%% ────────────────────────────────────────────────────────────────────
%% Main fusion-validity predicate
%% ────────────────────────────────────────────────────────────────────
%%
%% fusion_valid(+OpPair, -Reason) is the trivial-case fusion validator.
%% Succeeds when the two ops in OpPair can be SAFELY fused; Reason
%% is the validity classification.

fusion_valid([Op1, Op2], epilogue_fusion) :-
    % Op2 is the next op in sequence after Op1, and consumes Op1's output
    op_output(Op1, T),
    op_input(Op2, T),

    % Op1's write region matches Op2's read region
    op_writes(Op1, T, WriteRegion),
    op_reads(Op2, T, ReadRegion),
    region_matches(WriteRegion, ReadRegion),

    % The intermediate tensor T doesn't escape — only Op2 consumes it
    no_escape(T, [Op2]),

    % Op classes are fusion-compatible
    op_class(Op1, C1),
    op_class(Op2, C2),
    op_class_compatible(C1, C2),

    !.

%% Failure case: explicitly report why fusion isn't valid.
%% This lets us debug WHY a fusion was rejected, supporting the
%% mutation-analysis discipline.

fusion_invalid([Op1, Op2], shape_mismatch) :-
    op_output(Op1, T),
    op_input(Op2, T),
    op_writes(Op1, T, WriteRegion),
    op_reads(Op2, T, ReadRegion),
    \+ region_matches(WriteRegion, ReadRegion).

fusion_invalid([Op1, Op2], tensor_escapes) :-
    op_output(Op1, T),
    op_input(Op2, T),
    \+ no_escape(T, [Op2]).

fusion_invalid([Op1, Op2], op_class_incompatible) :-
    op_class(Op1, C1),
    op_class(Op2, C2),
    \+ op_class_compatible(C1, C2).
