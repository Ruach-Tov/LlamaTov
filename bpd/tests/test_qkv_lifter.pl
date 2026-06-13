%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_qkv_lifter.pl — Verify lifter extracts BPD facts from qwen2.cpp QKV section.
%%
%% Per mavchin's direction: the lifter is the convergence work. This test
%% exercises the full loop end-to-end on real qwen2.cpp Q-projection C.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/qkv_lifter').

run_tests :-
    Tests = [
        test_lift_norm_statement,
        test_lift_q_projection_assignment,
        test_lift_conditional_bias_add,
        test_lift_full_q_projection_section,
        test_op_level_classification,
        test_arch_param_recognition,
        test_lift_k_projection_section,
        test_lift_v_projection_section,
        test_lift_full_qkv_block,
        test_lifter_generalizes_across_QKV,
        test_lift_reshape_op,
        test_lift_rope_op,
        test_lifter_handles_primitive_op_kinds,
        test_known_limitation_reassignment_produces_duplicate_outputs
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

%% Norm: cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
test_lift_norm_statement :-
    qkv_lifter:lift_qkv_section(
        'cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);',
        Facts),
    member(op_kind(cur, build_norm), Facts),
    member(op_level(cur, builder), Facts),
    member(op_output(cur, cur), Facts).

%% Q projection: Qcur = build_lora_mm(model.layers[il].wq, cur);
test_lift_q_projection_assignment :-
    qkv_lifter:lift_qkv_section(
        'Qcur = build_lora_mm(model.layers[il].wq, cur);',
        Facts),
    member(op_kind('Qcur', build_lora_mm), Facts),
    member(op_level('Qcur', builder), Facts),
    member(op_inputs('Qcur', [wq, cur]), Facts).

%% Conditional bias: if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); cb(Qcur, "Qcur", il); }
test_lift_conditional_bias_add :-
    qkv_lifter:lift_qkv_section(
        'if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); cb(Qcur, "Qcur", il); }',
        Facts),
    %% Should produce a tensor_join fact
    member(tensor_join('Qcur', [if(present(bq), Qcur_post_bias, 'Qcur')]),
           Facts),
    %% And the conditional op_kind
    member(op_kind(Qcur_post_bias, ggml_add), Facts),
    member(op_condition(Qcur_post_bias, present(bq)), Facts).

%% Full Q projection section: 3 statements (matmul + conditional bias)
test_lift_full_q_projection_section :-
    Source = 'Qcur = build_lora_mm(model.layers[il].wq, cur); if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); cb(Qcur, "Qcur", il); }',
    qkv_lifter:lift_qkv_section(Source, Facts),
    %% Verify both the matmul AND the conditional bias are extracted
    member(op_kind('Qcur', build_lora_mm), Facts),
    member(op_inputs('Qcur', [wq, cur]), Facts),
    member(tensor_join('Qcur', _), Facts).

%% op_level classification: builders vs primitives
test_op_level_classification :-
    qkv_lifter:op_level_of(build_norm, builder),
    qkv_lifter:op_level_of(build_lora_mm, builder),
    qkv_lifter:op_level_of(ggml_add, primitive),
    qkv_lifter:op_level_of(ggml_reshape_3d, primitive),
    qkv_lifter:op_level_of(ggml_rope_ext, primitive).

%% Architecture parameter recognition
test_arch_param_recognition :-
    qkv_lifter:lift_qkv_section(
        'x = build_lora_mm(model.layers[il].wq, cur);',
        Facts),
    %% Verify a parameter fact is emitted for wq
    member(parameter(wq, layer(il), from_hparams), Facts).

%% K projection: same structure as Q, but with wk and bk
test_lift_k_projection_section :-
    Source = 'Kcur = build_lora_mm(model.layers[il].wk, cur); if (model.layers[il].bk) { Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk); cb(Kcur, "Kcur", il); }',
    qkv_lifter:lift_qkv_section(Source, Facts),
    member(op_kind('Kcur', build_lora_mm), Facts),
    member(op_inputs('Kcur', [wk, cur]), Facts),
    member(tensor_join('Kcur', [if(present(bk), Kcur_post_bias, 'Kcur')]),
           Facts),
    member(op_kind(Kcur_post_bias, ggml_add), Facts).

%% V projection: same structure as Q and K, but with wv and bv
test_lift_v_projection_section :-
    Source = 'Vcur = build_lora_mm(model.layers[il].wv, cur); if (model.layers[il].bv) { Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv); cb(Vcur, "Vcur", il); }',
    qkv_lifter:lift_qkv_section(Source, Facts),
    member(op_kind('Vcur', build_lora_mm), Facts),
    member(op_inputs('Vcur', [wv, cur]), Facts),
    member(tensor_join('Vcur', [if(present(bv), Vcur_post_bias, 'Vcur')]),
           Facts),
    member(op_kind(Vcur_post_bias, ggml_add), Facts).

%% Full QKV block: norm + Q + K + V projections in sequence
test_lift_full_qkv_block :-
    Source = 'cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il); Qcur = build_lora_mm(model.layers[il].wq, cur); if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); cb(Qcur, "Qcur", il); } Kcur = build_lora_mm(model.layers[il].wk, cur); if (model.layers[il].bk) { Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk); cb(Kcur, "Kcur", il); } Vcur = build_lora_mm(model.layers[il].wv, cur); if (model.layers[il].bv) { Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv); cb(Vcur, "Vcur", il); }',
    qkv_lifter:lift_qkv_section(Source, Facts),
    %% Should have the norm
    member(op_kind(cur, build_norm), Facts),
    %% All three projections
    member(op_kind('Qcur', build_lora_mm), Facts),
    member(op_kind('Kcur', build_lora_mm), Facts),
    member(op_kind('Vcur', build_lora_mm), Facts),
    %% All three bias tensor_joins
    member(tensor_join('Qcur', _), Facts),
    member(tensor_join('Kcur', _), Facts),
    member(tensor_join('Vcur', _), Facts),
    %% All three parameter declarations
    member(parameter(wq, layer(il), from_hparams), Facts),
    member(parameter(wk, layer(il), from_hparams), Facts),
    member(parameter(wv, layer(il), from_hparams), Facts).

%% Reshape: Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens)
test_lift_reshape_op :-
    qkv_lifter:lift_qkv_section(
        'Qcur_3d = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);',
        Facts),
    member(op_kind('Qcur_3d', ggml_reshape_3d), Facts),
    member(op_level('Qcur_3d', primitive), Facts),
    member(op_inputs('Qcur_3d', [ctx0, 'Qcur', n_embd_head, n_head, n_tokens]),
           Facts).

%% Rope: Qcur = ggml_rope_ext(ctx0, Qcur_3d, inp_pos, ...)
test_lift_rope_op :-
    qkv_lifter:lift_qkv_section(
        'Qcur_rope = ggml_rope_ext(ctx0, Qcur_3d, inp_pos, NULL, n_rot, rope_type);',
        Facts),
    member(op_kind('Qcur_rope', ggml_rope_ext), Facts),
    member(op_level('Qcur_rope', primitive), Facts).

%% The op_level classification is comprehensive
test_lifter_handles_primitive_op_kinds :-
    qkv_lifter:op_level_of(ggml_reshape_3d, primitive),
    qkv_lifter:op_level_of(ggml_rope_ext, primitive),
    qkv_lifter:op_level_of(ggml_mul, primitive),
    qkv_lifter:op_level_of(ggml_silu, primitive).

%% KNOWN LIMITATION: when C re-assigns to the same name, the lifter
%% emits multiple op facts with the same output. Documenting via test.
%% (This is structurally faithful; semantic disambiguation would need
%% a post-lift SSA pass.)
test_known_limitation_reassignment_produces_duplicate_outputs :-
    qkv_lifter:lift_qkv_section(
        'Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens); Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, NULL, n_rot, rope_type);',
        Facts),
    %% Both ops should emit op('Qcur') (the re-assignment behavior)
    findall(O, member(op(O), Facts), Outputs),
    length(Outputs, NumOps),
    %% Two ops, both targeting 'Qcur'
    NumOps == 2,
    forall(member(O, Outputs), O == 'Qcur').

%% Generality: same lifter handles all three projections without modification
test_lifter_generalizes_across_QKV :-
    %% Count facts from each projection's lift — should be the same shape
    qkv_lifter:lift_qkv_section(
        'X1 = build_lora_mm(model.layers[il].wq, cur); if (model.layers[il].bq) { X1 = ggml_add(ctx0, X1, model.layers[il].bq); cb(X1, "X1", il); }',
        Facts1),
    qkv_lifter:lift_qkv_section(
        'X2 = build_lora_mm(model.layers[il].wk, cur); if (model.layers[il].bk) { X2 = ggml_add(ctx0, X2, model.layers[il].bk); cb(X2, "X2", il); }',
        Facts2),
    qkv_lifter:lift_qkv_section(
        'X3 = build_lora_mm(model.layers[il].wv, cur); if (model.layers[il].bv) { X3 = ggml_add(ctx0, X3, model.layers[il].bv); cb(X3, "X3", il); }',
        Facts3),
    length(Facts1, L1),
    length(Facts2, L2),
    length(Facts3, L3),
    L1 == L2,
    L2 == L3.

:- initialization(run_tests, main).
