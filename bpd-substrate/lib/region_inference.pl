%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% region_inference.pl — Stage 2: derive region facts from op_kind + shape.
%%
%% Stage 1 (symbolic_fusion.pl) consumed hand-encoded op_reads/op_writes
%% facts. Stage 2 derives those facts from the more compact substrate
%% of op_kind + tensor shapes.
%%
%% Per SYMBOLIC_FUSION_CORRECTNESS.md Section 9.3: extension path from
%% trivial case. The substrate gets smaller (BPD authors don't have to
%% hand-write region facts) while the analysis gets richer (regions
%% are derived consistently from a single source of truth).

:- module(region_inference, [
    infer_region/4,
    infer_region_from_facts/5
]).

%% ────────────────────────────────────────────────────────────────────
%% Region inference rules
%% ────────────────────────────────────────────────────────────────────
%%
%% infer_region(+Op, +Tensor, +Role, -Region)
%%   Op:     the operation (e.g., wq_mul)
%%   Tensor: the tensor being read or written (e.g., cur_after_norm)
%%   Role:   read | write
%%   Region: the inferred region term
%%
%% This module is the substrate-honest version of what Stage 1 did
%% by hand. The rules encode the SEMANTICS of each op_kind: given
%% the kind, what region does each of its inputs and outputs occupy?

%% ────────────────────────────────────────────────────────────────────
%% Matmul: build_lora_mm(W, X) → Y
%% ────────────────────────────────────────────────────────────────────
%% Inputs:
%%   W is read as matmul weights, shape [K, N]
%%   X is read as matmul input,    shape [M, K]
%% Output:
%%   Y is written as matmul output, shape [M, N]

%% Note: Since region_inference expects global facts (op_kind/2, etc.)
%% we should define a version that takes GraphFacts or assert them.
%% For now, we will add infer_region_from_facts/5 to avoid breaking existing code.

infer_region_from_facts(Facts, Op, Tensor, Role, Region) :-
    member(op_kind(Op, Kind), Facts),
    infer_region_kind(Facts, Kind, Op, Tensor, Role, Region).

infer_region_kind(Facts, build_lora_mm, Op, W, read, region(matmul_weights, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(1, Inputs, W).

infer_region_kind(Facts, build_lora_mm, Op, X, read, region(matmul_input, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(2, Inputs, X).

infer_region_kind(Facts, build_lora_mm, Op, Y, write, region(matmul_output, _)) :-
    member(op_output(Op, Y), Facts).

infer_region_kind(Facts, ggml_add, Op, A, read, region(elementwise, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(1, Inputs, A).

infer_region_kind(Facts, ggml_add, Op, B, read, region(broadcast, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(2, Inputs, B).

infer_region_kind(Facts, ggml_add, Op, C, write, region(elementwise, _)) :-
    member(op_output(Op, C), Facts).

infer_region_kind(Facts, ggml_silu, Op, A, read, region(elementwise, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(1, Inputs, A).

infer_region_kind(Facts, ggml_silu, Op, C, write, region(elementwise, _)) :-
    member(op_output(Op, C), Facts).

infer_region_kind(Facts, bias_add, Op, A, read, region(elementwise, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(1, Inputs, A).

infer_region_kind(Facts, bias_add, Op, C, write, region(elementwise, _)) :-
    member(op_output(Op, C), Facts).

infer_region_kind(Facts, matmul, Op, W, read, region(matmul_weights, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(1, Inputs, W).

infer_region_kind(Facts, matmul, Op, X, read, region(matmul_input, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(2, Inputs, X).

infer_region_kind(Facts, matmul, Op, Y, write, region(matmul_output, _)) :-
    member(op_output(Op, Y), Facts).

infer_region_kind(Facts, conv2d, Op, W, read, region(matmul_weights, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(1, Inputs, W).

infer_region_kind(Facts, conv2d, Op, X, read, region(matmul_input, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(2, Inputs, X).

infer_region_kind(Facts, conv2d, Op, Y, write, region(matmul_output, _)) :-
    member(op_output(Op, Y), Facts).

infer_region_kind(Facts, gemm, Op, W, read, region(matmul_weights, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(1, Inputs, W).

infer_region_kind(Facts, gemm, Op, X, read, region(matmul_input, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(2, Inputs, X).

infer_region_kind(Facts, gemm, Op, Y, write, region(matmul_output, _)) :-
    member(op_output(Op, Y), Facts).

infer_region_kind(Facts, ggml_mul, Op, A, read, region(elementwise, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(1, Inputs, A).

infer_region_kind(Facts, ggml_mul, Op, B, read, region(broadcast, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    nth1(2, Inputs, B).

infer_region_kind(Facts, ggml_mul, Op, C, write, region(elementwise, _)) :-
    member(op_output(Op, C), Facts).

%% Fused ops: the output of a fused op is read elementwise by the next consumer
%% (since fusion absorbs the epilogue into registers)
infer_region_kind(Facts, fused(_, _), Op, Y, write, region(elementwise, _)) :-
    member(op_output(Op, Y), Facts).

infer_region_kind(Facts, fused(_, _), Op, A, read, region(elementwise, _)) :-
    member(op_inputs(Op, Inputs), Facts),
    member(A, Inputs).

infer_region(Op, W, read, region(matmul_weights, [K, N])) :-
    op_kind(Op, build_lora_mm),
    op_input_position(Op, W, 1),       % W is the first input
    tensor_shape(W, [K, N]).

infer_region(Op, X, read, region(matmul_input, [M, K])) :-
    op_kind(Op, build_lora_mm),
    op_input_position(Op, X, 2),       % X is the second input
    tensor_shape(X, [M, K]).

infer_region(Op, Y, write, region(matmul_output, [M, N])) :-
    op_kind(Op, build_lora_mm),
    op_output(Op, Y),
    op_input_position(Op, W, 1), tensor_shape(W, [_, N]),
    op_input_position(Op, X, 2), tensor_shape(X, [M, _]).

%% ────────────────────────────────────────────────────────────────────
%% Elementwise add: ggml_add(A, B) → C
%% ────────────────────────────────────────────────────────────────────
%% Inputs:
%%   A is read elementwise over full shape
%%   B is broadcast (might be smaller — last dims must match A's last dims)
%% Output:
%%   C is written elementwise, same shape as A

infer_region(Op, A, read, region(elementwise, ShapeA)) :-
    op_kind(Op, ggml_add),
    op_input_position(Op, A, 1),
    tensor_shape(A, ShapeA).

infer_region(Op, B, read, region(broadcast, ShapeB)) :-
    op_kind(Op, ggml_add),
    op_input_position(Op, B, 2),
    tensor_shape(B, ShapeB).

infer_region(Op, C, write, region(elementwise, ShapeA)) :-
    op_kind(Op, ggml_add),
    op_output(Op, C),
    op_input_position(Op, A, 1),
    tensor_shape(A, ShapeA).

%% ────────────────────────────────────────────────────────────────────
%% Norm: build_norm(rms)(X, weights) → Y
%% ────────────────────────────────────────────────────────────────────
%% RMS norm reads X row-wise (needs full row to compute norm), reads
%% weights as broadcast, writes Y elementwise.

infer_region(Op, X, read, region(row_reduction, ShapeX)) :-
    op_kind(Op, build_norm(rms)),
    op_input_position(Op, X, 1),
    tensor_shape(X, ShapeX).

infer_region(Op, W, read, region(broadcast, ShapeW)) :-
    op_kind(Op, build_norm(rms)),
    op_input_position(Op, W, 2),
    tensor_shape(W, ShapeW).

infer_region(Op, Y, write, region(elementwise, ShapeX)) :-
    op_kind(Op, build_norm(rms)),
    op_output(Op, Y),
    op_input_position(Op, X, 1),
    tensor_shape(X, ShapeX).

%% ────────────────────────────────────────────────────────────────────
%% Reshape: ggml_reshape_3d(X, D1, D2, D3) → Y
%% ────────────────────────────────────────────────────────────────────
%% Reshape is a layout transformation — same elements, different shape.
%% Read region is elementwise over X's shape; write region is elementwise
%% over the new shape. The "same elements, different layout" property
%% is critical for fusion: reshape between two ops can often be elided
%% if the consuming op can address the source layout directly.

infer_region(Op, X, read, region(elementwise, ShapeX)) :-
    op_kind(Op, ggml_reshape_3d),
    op_input_position(Op, X, 1),
    tensor_shape(X, ShapeX).

infer_region(Op, Y, write, region(elementwise, NewShape)) :-
    op_kind(Op, ggml_reshape_3d),
    op_output(Op, Y),
    tensor_shape(Y, NewShape).

%% ────────────────────────────────────────────────────────────────────
%% Notes for future stages
%% ────────────────────────────────────────────────────────────────────
%%
%% Operations NOT yet handled (defer to Stage 3+):
%%   - ggml_rope_ext: elementwise BUT depends on position; needs care
%%   - ggml_silu: pure elementwise (easy add)
%%   - ggml_mul: pure elementwise (easy add)
%%   - ggml_soft_max: row-reduction read, row-elementwise write (hard for fusion)
%%   - matmul output → attention: needs multi-head reshape semantics
%%
%% Operations that are FUNDAMENTALLY harder:
%%   - Softmax (the diamond pattern's middle): full-row dependency
%%   - Attention combined matmul+softmax+matmul: requires graph patterns
%%   - Any reduction op: the "barrier" mavchin identified
%%
%% These all become straightforward to add once the trivial cases
%% above are validated. Stage 2 is foundational; Stage 4 will extend
%% to the harder patterns.

%% Extended region inference for all elementwise activations
%% Any elementwise op reads and writes elementwise
infer_region_kind(_, relu, _, _, read, region(elementwise, _)).
infer_region_kind(_, relu, _, _, write, region(elementwise, _)).
infer_region_kind(_, leaky_relu, _, _, read, region(elementwise, _)).
infer_region_kind(_, leaky_relu, _, _, write, region(elementwise, _)).
infer_region_kind(_, gelu, _, _, read, region(elementwise, _)).
infer_region_kind(_, gelu, _, _, write, region(elementwise, _)).
infer_region_kind(_, sigmoid, _, _, read, region(elementwise, _)).
infer_region_kind(_, sigmoid, _, _, write, region(elementwise, _)).
infer_region_kind(_, tanh, _, _, read, region(elementwise, _)).
infer_region_kind(_, tanh, _, _, write, region(elementwise, _)).
infer_region_kind(_, mish, _, _, read, region(elementwise, _)).
infer_region_kind(_, mish, _, _, write, region(elementwise, _)).
infer_region_kind(_, hardswish, _, _, read, region(elementwise, _)).
infer_region_kind(_, hardswish, _, _, write, region(elementwise, _)).
infer_region_kind(_, hardtanh, _, _, read, region(elementwise, _)).
infer_region_kind(_, hardtanh, _, _, write, region(elementwise, _)).
infer_region_kind(_, hardsigmoid, _, _, read, region(elementwise, _)).
infer_region_kind(_, hardsigmoid, _, _, write, region(elementwise, _)).
infer_region_kind(_, elu, _, _, read, region(elementwise, _)).
infer_region_kind(_, elu, _, _, write, region(elementwise, _)).
infer_region_kind(_, selu, _, _, read, region(elementwise, _)).
infer_region_kind(_, selu, _, _, write, region(elementwise, _)).
infer_region_kind(_, softplus, _, _, read, region(elementwise, _)).
infer_region_kind(_, softplus, _, _, write, region(elementwise, _)).
infer_region_kind(_, silu, _, _, read, region(elementwise, _)).
infer_region_kind(_, silu, _, _, write, region(elementwise, _)).
infer_region_kind(_, swish, _, _, read, region(elementwise, _)).
infer_region_kind(_, swish, _, _, write, region(elementwise, _)).
infer_region_kind(_, scale, _, _, read, region(elementwise, _)).
infer_region_kind(_, scale, _, _, write, region(elementwise, _)).
infer_region_kind(_, clamp, _, _, read, region(elementwise, _)).
infer_region_kind(_, clamp, _, _, write, region(elementwise, _)).
infer_region_kind(_, dropout, _, _, read, region(elementwise, _)).
infer_region_kind(_, dropout, _, _, write, region(elementwise, _)).
infer_region_kind(_, mul, _, _, read, region(elementwise, _)).
infer_region_kind(_, mul, _, _, write, region(elementwise, _)).
infer_region_kind(_, div, _, _, read, region(elementwise, _)).
infer_region_kind(_, div, _, _, write, region(elementwise, _)).
infer_region_kind(_, sub, _, _, read, region(elementwise, _)).
infer_region_kind(_, sub, _, _, write, region(elementwise, _)).
infer_region_kind(_, add, _, _, read, region(elementwise, _)).
infer_region_kind(_, add, _, _, write, region(elementwise, _)).
