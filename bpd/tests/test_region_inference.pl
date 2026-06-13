%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_region_inference.pl — Stage 2 derivation rules test
%%
%% Verifies that region facts derived from op_kind + tensor shapes
%% match what Stage 1 had hand-encoded for the QKV BPD.

:- use_module('../lib/region_inference').

%% Declare predicates dynamic (test data is dynamic)
:- dynamic op_kind/2.
:- dynamic op_input_position/3.
:- dynamic op_output/2.
:- dynamic tensor_shape/2.

%% Wire region_inference's expected predicates
region_inference:op_kind(Op, K) :- op_kind(Op, K).
region_inference:op_input_position(Op, T, P) :- op_input_position(Op, T, P).
region_inference:op_output(Op, T) :- op_output(Op, T).
region_inference:tensor_shape(T, S) :- tensor_shape(T, S).

%% ────────────────────────────────────────────────────────────────────
%% Test fixture: wq_mul + wq_bias_add (the QKV BPD's trivial fusion)
%% ────────────────────────────────────────────────────────────────────

% Tensor shapes — symbolic in the architectural parameters
tensor_shape(wq, [n_embd, n_head_x_head_dim]).
tensor_shape(cur_after_norm, [n_tokens, n_embd]).
tensor_shape(qcur_pre_bias, [n_tokens, n_head_x_head_dim]).
tensor_shape(bq, [n_head_x_head_dim]).
tensor_shape(qcur_post_bias, [n_tokens, n_head_x_head_dim]).

% wq_mul: build_lora_mm with W=wq (pos 1), X=cur_after_norm (pos 2), output=qcur_pre_bias
op_kind(wq_mul, build_lora_mm).
op_input_position(wq_mul, wq, 1).
op_input_position(wq_mul, cur_after_norm, 2).
op_output(wq_mul, qcur_pre_bias).

% wq_bias_add: ggml_add with A=qcur_pre_bias (pos 1), B=bq (pos 2), output=qcur_post_bias
op_kind(wq_bias_add, ggml_add).
op_input_position(wq_bias_add, qcur_pre_bias, 1).
op_input_position(wq_bias_add, bq, 2).
op_output(wq_bias_add, qcur_post_bias).

%% Also add an RMS norm op for testing those rules
tensor_shape(inpL, [n_tokens, n_embd]).
tensor_shape(attn_norm_w, [n_embd]).
op_kind(qkv_norm, build_norm(rms)).
op_input_position(qkv_norm, inpL, 1).
op_input_position(qkv_norm, attn_norm_w, 2).
op_output(qkv_norm, cur_after_norm).

%% ────────────────────────────────────────────────────────────────────
%% Test runner
%% ────────────────────────────────────────────────────────────────────

run_tests :-
    Tests = [
        test_matmul_weight_region,
        test_matmul_input_region,
        test_matmul_output_region,
        test_elementwise_add_input_region,
        test_elementwise_add_broadcast_region,
        test_elementwise_add_output_region,
        test_rms_norm_input_region,
        test_rms_norm_output_region,
        test_stage1_compatibility
    ],
    run_each(Tests, 0, 0, P, F),
    format("~n=============================================~n", []),
    format("RESULTS: ~d passed, ~d failed~n", [P, F]),
    format("=============================================~n", []),
    ( F > 0 -> halt(1) ; true ).

run_each([], P, F, P, F).
run_each([T | Rest], P0, F0, P, F) :-
    ( catch(call(T), Err, (format("  FAIL ~w: error ~w~n", [T, Err]), fail))
    -> ( format("  PASS ~w~n", [T]), P1 is P0 + 1, F1 = F0 )
    ; ( format("  FAIL ~w~n", [T]), P1 = P0, F1 is F0 + 1 )
    ),
    run_each(Rest, P1, F1, P, F).

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

test_matmul_weight_region :-
    region_inference:infer_region(wq_mul, wq, read, R),
    R == region(matmul_weights, [n_embd, n_head_x_head_dim]).

test_matmul_input_region :-
    region_inference:infer_region(wq_mul, cur_after_norm, read, R),
    R == region(matmul_input, [n_tokens, n_embd]).

test_matmul_output_region :-
    region_inference:infer_region(wq_mul, qcur_pre_bias, write, R),
    R == region(matmul_output, [n_tokens, n_head_x_head_dim]).

test_elementwise_add_input_region :-
    region_inference:infer_region(wq_bias_add, qcur_pre_bias, read, R),
    R == region(elementwise, [n_tokens, n_head_x_head_dim]).

test_elementwise_add_broadcast_region :-
    region_inference:infer_region(wq_bias_add, bq, read, R),
    R == region(broadcast, [n_head_x_head_dim]).

test_elementwise_add_output_region :-
    region_inference:infer_region(wq_bias_add, qcur_post_bias, write, R),
    R == region(elementwise, [n_tokens, n_head_x_head_dim]).

test_rms_norm_input_region :-
    region_inference:infer_region(qkv_norm, inpL, read, R),
    R == region(row_reduction, [n_tokens, n_embd]).

test_rms_norm_output_region :-
    region_inference:infer_region(qkv_norm, cur_after_norm, write, R),
    R == region(elementwise, [n_tokens, n_embd]).

%% Cross-stage compatibility: the regions derived in Stage 2 should
%% match the regions Stage 1 used as hand-encoded facts. This ensures
%% the substrate stays self-consistent.
test_stage1_compatibility :-
    region_inference:infer_region(wq_mul, qcur_pre_bias, write, R1),
    region_inference:infer_region(wq_bias_add, qcur_pre_bias, read, R2),
    % These should satisfy region_matches/2 from symbolic_fusion
    % (matmul_output → elementwise with same shape)
    R1 = region(matmul_output, S),
    R2 = region(elementwise, S).

:- initialization(run_tests, main).
