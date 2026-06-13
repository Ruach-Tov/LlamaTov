%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% transform_turboquant.pl — TurboQuant model transformation
%%
%% TurboQuant replaces standard KV caching with a polar quantization scheme.
%% This transform looks for matmul operations that produce Key or Value tensors,
%% and rewrites them to output turboquant-encoded tensors instead.
%% The attention mechanism is also rewritten to accept turboquant inputs.

:- module(transform_turboquant, []).

:- use_module(model_transform).

%% ────────────────────────────────────────────────────────────────────
%% TurboQuant Transform: K/V Projections
%% ────────────────────────────────────────────────────────────────────
%% Matches a matmul that produces a tensor used as a Key or Value in attention.
%% Replaces it with `turboquant_matmul` which implies the random rotation (S)
%% and polar quantization are fused into the projection.

model_transform:transform_pattern(turboquant, Facts, Subgraph, Context) :-
    %% Find a matmul operation
    model_transform:find_op(Facts, matmul, MmOp),
    model_transform:get_output(Facts, MmOp, MmOut),
    
    %% Verify it feeds into an attention operation as K or V
    model_transform:find_op(Facts, flash_attention, AttnOp),
    model_transform:get_inputs(Facts, AttnOp, AttnInputs),
    %% flash_attention inputs are typically [Q, K, V]
    ( nth1(2, AttnInputs, MmOut) ; nth1(3, AttnInputs, MmOut) ),
    
    %% Ensure it hasn't already been transformed
    \+ member(tensor_encoding(MmOut, turboquant), Facts),
    
    %% The subgraph to replace is just the op_kind
    Subgraph = [op_kind(MmOp, matmul)],
    
    %% Context is the op ID and output tensor
    Context = ctx(MmOp, MmOut).

model_transform:transform_replacement(turboquant, _Facts, _Subgraph, ctx(MmOp, MmOut), Replacement) :-
    %% Replace matmul with turboquant_matmul
    %% Add a fact indicating the output tensor is now turboquant encoded
    Replacement = [
        op_kind(MmOp, turboquant_matmul),
        tensor_encoding(MmOut, turboquant)
    ].

%% ────────────────────────────────────────────────────────────────────
%% TurboQuant Transform: Attention Consumer
%% ────────────────────────────────────────────────────────────────────
%% Matches an attention operation that consumes turboquant-encoded K/V.
%% Replaces it with `turboquant_attention` which knows how to compute
%% scores using the polar representation.

model_transform:transform_pattern(turboquant, Facts, Subgraph, Context) :-
    %% Find an attention operation
    model_transform:find_op(Facts, flash_attention, AttnOp),
    model_transform:get_inputs(Facts, AttnOp, AttnInputs),
    
    %% Check if any input (K or V) is turboquant encoded
    member(Input, AttnInputs),
    member(tensor_encoding(Input, turboquant), Facts),
    
    %% The subgraph to replace is the op_kind
    Subgraph = [op_kind(AttnOp, flash_attention)],
    Context = ctx(AttnOp).

model_transform:transform_replacement(turboquant, _Facts, _Subgraph, ctx(AttnOp), Replacement) :-
    %% Replace with turboquant_attention
    Replacement = [
        op_kind(AttnOp, turboquant_attention)
    ].
