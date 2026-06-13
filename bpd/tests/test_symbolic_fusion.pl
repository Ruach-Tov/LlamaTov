%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_symbolic_fusion.pl — Stage 1 trivial case test suite
%%
%% Tests the symbolic_fusion analyzer on the QKV BPD's wq_mul + wq_bias_add
%% fusion plus negative cases that should be rejected:
%%   - Shape mismatch
%%   - Tensor escapes to multiple consumers
%%   - Op class incompatible (e.g., matmul → matmul)

:- use_module('../lib/symbolic_fusion').

%% Declare fact predicates dynamic so tests can assertz/retract for diagnosis cases.
:- dynamic op_input/2.
:- dynamic op_output/2.
:- dynamic op_reads/3.
:- dynamic op_writes/3.
:- dynamic op_class/2.
:- discontiguous op_input/2.
:- discontiguous op_output/2.
:- discontiguous op_reads/3.
:- discontiguous op_writes/3.
:- discontiguous op_class/2.

%% ────────────────────────────────────────────────────────────────────
%% Test fixture: the QKV portion of Qwen2's attention block
%% ────────────────────────────────────────────────────────────────────
%%
%% These facts encode the same compute graph as compute-graphs/qwen2/qkv.bpd
%% but in the format symbolic_fusion expects: op_input/2, op_output/2,
%% op_reads/3, op_writes/3, op_class/2.
%%
%% NOTE: in production, these would be DERIVED from the BPD via the
%% bpd_to_prolog.py compiler. For Stage 1, we hand-encode them.

% wq_mul: Qcur = matmul(wq, cur_after_norm)
op_input(wq_mul, wq).
op_input(wq_mul, cur_after_norm).
op_output(wq_mul, Qcur_pre_bias).
% (Using atoms; in the real BPD these are ground tensor IDs)
op_reads(wq_mul, wq, region(matmul_weights, [n_embd, n_head_x_head_dim])).
op_reads(wq_mul, cur_after_norm, region(matmul_input, [n_tokens, n_embd])).
op_writes(wq_mul, qcur_pre_bias, region(matmul_output, [n_tokens, n_head_x_head_dim])).
op_class(wq_mul, matmul).

% Fix the variable in op_output for the test (Prolog var won't unify with ground)
op_output(wq_mul, qcur_pre_bias) :- !.

% wq_bias_add: Qcur_post_bias = qcur_pre_bias + bq
op_input(wq_bias_add, qcur_pre_bias).
op_input(wq_bias_add, bq).
op_output(wq_bias_add, qcur_post_bias).
op_reads(wq_bias_add, qcur_pre_bias, region(elementwise, [n_tokens, n_head_x_head_dim])).
op_reads(wq_bias_add, bq, region(broadcast, [n_head_x_head_dim])).
op_writes(wq_bias_add, qcur_post_bias, region(elementwise, [n_tokens, n_head_x_head_dim])).
op_class(wq_bias_add, elementwise).

% wk_mul: similar to wq_mul (negative-case neighbor — should NOT be fusible
% with wq_mul because they don't have a producer/consumer relationship)
op_input(wk_mul, wk).
op_input(wk_mul, cur_after_norm).
op_output(wk_mul, kcur_pre_bias).
op_reads(wk_mul, wk, region(matmul_weights, [n_embd, n_head_kv_x_head_dim])).
op_reads(wk_mul, cur_after_norm, region(matmul_input, [n_tokens, n_embd])).
op_writes(wk_mul, kcur_pre_bias, region(matmul_output, [n_tokens, n_head_kv_x_head_dim])).
op_class(wk_mul, matmul).

% Wire the symbolic_fusion module's expected predicates to our fixtures.
% (In production, these would all live in the compute: namespace.)
symbolic_fusion:op_input(Op, T) :- op_input(Op, T).
symbolic_fusion:op_output(Op, T) :- op_output(Op, T).
symbolic_fusion:op_reads(Op, T, R) :- op_reads(Op, T, R).
symbolic_fusion:op_writes(Op, T, R) :- op_writes(Op, T, R).
symbolic_fusion:op_class(Op, C) :- op_class(Op, C).

%% ────────────────────────────────────────────────────────────────────
%% Test runner
%% ────────────────────────────────────────────────────────────────────

run_tests :-
    Tests = [
        test_wq_mul_bias_add_is_fusible,
        test_wq_mul_alone_not_fusion,
        test_unrelated_ops_not_fusible,
        test_diagnosis_of_shape_mismatch_scenario,
        test_diagnosis_of_op_class_incompatible
    ],
    run_each(Tests, 0, 0, Passed, Failed),
    format("~n=============================================~n", []),
    format("RESULTS: ~d passed, ~d failed~n", [Passed, Failed]),
    format("=============================================~n", []),
    ( Failed > 0 -> halt(1) ; true ).

run_each([], P, F, P, F).
run_each([T | Rest], P0, F0, P, F) :-
    ( catch(call(T), Err, (format("  FAIL ~w: error ~w~n", [T, Err]), fail))
    -> ( format("  PASS ~w~n", [T]),
         P1 is P0 + 1,
         F1 = F0 )
    ; ( format("  FAIL ~w~n", [T]),
        P1 = P0,
        F1 is F0 + 1 )
    ),
    run_each(Rest, P1, F1, P, F).

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

%% POSITIVE: wq_mul → wq_bias_add is fusible (the trivial case)
test_wq_mul_bias_add_is_fusible :-
    symbolic_fusion:fusion_valid([wq_mul, wq_bias_add], Reason),
    Reason == epilogue_fusion.

%% NEGATIVE: wq_mul alone (singleton) is not a fusion pair
%% Validation: fusion_valid expects a 2-element list
test_wq_mul_alone_not_fusion :-
    \+ symbolic_fusion:fusion_valid([wq_mul], _).

%% NEGATIVE: wq_mul and wk_mul are not fusible (they're parallel branches,
%% wk_mul doesn't consume wq_mul's output)
test_unrelated_ops_not_fusible :-
    \+ symbolic_fusion:fusion_valid([wq_mul, wk_mul], _).

%% DIAGNOSIS: when an op pair COULD be classes-compatible but has
%% shape mismatch, the fusion_invalid predicate should diagnose it.
%% We construct a synthetic pair to test:
test_diagnosis_of_shape_mismatch_scenario :-
    assertz(op_output(synth_op_a, synth_t)),
    assertz(op_input(synth_op_b, synth_t)),
    assertz(op_writes(synth_op_a, synth_t, region(matmul_output, [10, 20]))),
    % wrong shape for the read:
    assertz(op_reads(synth_op_b, synth_t, region(elementwise, [10, 30]))),
    % Should be diagnosed as shape mismatch
    symbolic_fusion:fusion_invalid([synth_op_a, synth_op_b], shape_mismatch),
    % Cleanup
    retract(op_output(synth_op_a, synth_t)),
    retract(op_input(synth_op_b, synth_t)),
    retract(op_writes(synth_op_a, synth_t, _)),
    retract(op_reads(synth_op_b, synth_t, _)).

%% DIAGNOSIS: op classes incompatible
test_diagnosis_of_op_class_incompatible :-
    assertz(op_class(synth_op_c, matmul)),
    assertz(op_class(synth_op_d, matmul)),  % matmul→matmul: not in our compatibility table
    symbolic_fusion:fusion_invalid([synth_op_c, synth_op_d], op_class_incompatible),
    retract(op_class(synth_op_c, _)),
    retract(op_class(synth_op_d, _)).

:- initialization(run_tests, main).
