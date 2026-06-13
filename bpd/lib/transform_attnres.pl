%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% transform_attnres.pl — Attention Residuals model transformation
%%
%% AttnRes (arXiv:2603.15031) replaces standard residual connections
%% with a learned softmax attention over preceding layer outputs.
%% This transform finds standard `ggml_add` operations that act as
%% residual connections and replaces them with `attn_residual` ops.

:- module(transform_attnres, []).

:- use_module(model_transform).

%% ────────────────────────────────────────────────────────────────────
%% AttnRes Transform: Replace Residual Add
%% ────────────────────────────────────────────────────────────────────
%% Matches an addition where one input is a layer output and the other
%% is the original input to that layer (a skip connection).

model_transform:transform_pattern(attnres, Facts, Subgraph, Context) :-
    %% Find an addition operation
    model_transform:find_op(Facts, ggml_add, AddOp),
    model_transform:get_inputs(Facts, AddOp, [SkipInput, LayerOutput]),
    
    %% Verify it's a residual connection:
    %% The LayerOutput must come from an op that consumed SkipInput
    %% (either directly or via a short chain like norm -> matmul)
    %% For simplicity in this proof-of-concept, we just look for an op
    %% that produces LayerOutput and has SkipInput somewhere in its ancestry.
    depends_on(Facts, LayerOutput, SkipInput, Path),
    
    %% The subgraph to replace is the op_kind and inputs
    Subgraph = [
        op_kind(AddOp, ggml_add),
        op_inputs(AddOp, [SkipInput, LayerOutput])
    ],
    
    %% Context is the op ID and the inputs
    Context = ctx(AddOp, SkipInput, LayerOutput, Path).

model_transform:transform_replacement(attnres, _Facts, _Subgraph, ctx(AddOp, SkipInput, LayerOutput, _Path), Replacement) :-
    %% In a full implementation, we would gather ALL preceding layer outputs.
    %% For this structural rewrite, we replace the `ggml_add` with `attn_residual`
    %% and explicitly pass the block state (which represents the history).
    %% We assume a `block_state` tensor is maintained in the graph.
    
    %% If there's no block_state yet, we just use the two inputs we know about
    %% as a 2-element history.
    Replacement = [
        op_kind(AddOp, attn_residual),
        op_inputs(AddOp, [SkipInput, LayerOutput, block_history])
    ].

%% Helper: Simple dependency check (is Out derived from In?)
depends_on(Facts, Out, In, [Op]) :-
    member(op_output(Op, Out), Facts),
    member(op_inputs(Op, Inputs), Facts),
    member(In, Inputs).
depends_on(Facts, Out, In, [Op|Path]) :-
    member(op_output(Op, Out), Facts),
    member(op_inputs(Op, Inputs), Facts),
    member(Mid, Inputs),
    depends_on(Facts, Mid, In, Path).
