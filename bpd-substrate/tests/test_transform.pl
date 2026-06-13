%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

:- use_module(library(plunit)).
:- use_module('../lib/model_transform').
:- use_module('../lib/transform_turboquant').
:- use_module('../lib/transform_attnres').
:- use_module('../lib/transform_search').

:- begin_tests(model_transform).

test(turboquant_transform) :-
    %% Mock facts for an attention block that produces K/V via matmul
    Facts = [
        op(k_proj), op_kind(k_proj, matmul), op_output(k_proj, k_tensor),
        op(v_proj), op_kind(v_proj, matmul), op_output(v_proj, v_tensor),
        op(q_proj), op_kind(q_proj, matmul), op_output(q_proj, q_tensor),
        op(attn), op_kind(attn, flash_attention), op_inputs(attn, [q_tensor, k_tensor, v_tensor])
    ],
    
    %% Apply TurboQuant transform
    apply_all_transforms([turboquant], Facts, TransformedFacts),
    
    %% Verify k_proj and v_proj became turboquant_matmul
    member(op_kind(k_proj, turboquant_matmul), TransformedFacts),
    member(op_kind(v_proj, turboquant_matmul), TransformedFacts),
    
    %% Verify q_proj remained standard matmul
    member(op_kind(q_proj, matmul), TransformedFacts),
    
    %% Verify attention became turboquant_attention
    member(op_kind(attn, turboquant_attention), TransformedFacts),
    
    %% Verify tensor encoding facts were added
    member(tensor_encoding(k_tensor, turboquant), TransformedFacts),
    member(tensor_encoding(v_tensor, turboquant), TransformedFacts), !.

test(attnres_transform) :-
    %% Mock facts for a residual connection
    Facts = [
        op(layer_norm), op_kind(layer_norm, rmsnorm), op_inputs(layer_norm, [x_in]), op_output(layer_norm, norm_out),
        op(layer_compute), op_kind(layer_compute, matmul), op_inputs(layer_compute, [norm_out]), op_output(layer_compute, layer_out),
        op(residual_add), op_kind(residual_add, ggml_add), op_inputs(residual_add, [x_in, layer_out]), op_output(residual_add, next_x)
    ],
    
    %% Apply AttnRes transform
    apply_all_transforms([attnres], Facts, TransformedFacts),
    
    %% Verify the ggml_add was replaced with attn_residual
    \+ member(op_kind(residual_add, ggml_add), TransformedFacts),
    member(op_kind(residual_add, attn_residual), TransformedFacts),
    
    %% Verify the inputs were updated to include block_history
    member(op_inputs(residual_add, [x_in, layer_out, block_history]), TransformedFacts), !.

test(transform_search) :-
    %% Very simple graph that can take both transforms
    Facts = [
        %% Residual path
        op(norm), op_kind(norm, rmsnorm), op_inputs(norm, [x]), op_output(norm, n_out),
        
        %% K proj
        op(k_proj), op_kind(k_proj, matmul), op_inputs(k_proj, [n_out]), op_output(k_proj, k_out),
        
        %% Attn
        op(attn), op_kind(attn, flash_attention), op_inputs(attn, [q, k_out, v]), op_output(attn, a_out),
        
        %% Residual add
        op(res_add), op_kind(res_add, ggml_add), op_inputs(res_add, [x, a_out])
    ],
    
    %% Search for optimal transformed graph
    optimal_transformed_graph([turboquant, attnres], Facts, cuBLAS, BestPlan),
    
    %% Verify both transforms were applied in the best plan
    member(op_kind(k_proj, turboquant_matmul), BestPlan),
    member(op_kind(attn, turboquant_attention), BestPlan),
    member(op_kind(res_add, attn_residual), BestPlan), !.

:- end_tests(model_transform).

:- initialization(run_tests, main).
