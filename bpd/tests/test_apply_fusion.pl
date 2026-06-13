%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_apply_fusion.pl — Stage 4 trivial-case apply-fusion tests.
%%
%% Per boneh's facilitation: close the validate→apply gap. This test
%% suite verifies the simplest case: applying an epilogue fusion
%% transforms BPD facts coherently.

:- use_module('../lib/apply_fusion').

run_tests :-
    Tests = [
        test_apply_creates_fused_op,
        test_apply_removes_original_ops,
        test_apply_removes_intermediate_tensor_facts,
        test_apply_preserves_external_facts,
        test_apply_records_provenance,
        test_apply_merged_inputs_correct,
        test_apply_preserves_sequence_position,
        test_apply_to_qkv_q_projection
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
%% Test fixture: wq_mul + wq_bias_add from QKV BPD
%% ────────────────────────────────────────────────────────────────────

input_facts([
    op(wq_mul),
    op_kind(wq_mul, build_lora_mm),
    op_inputs(wq_mul, [wq, cur_after_norm]),
    op_output(wq_mul, qcur_pre_bias),
    op_level(wq_mul, builder),
    sequence(qkv_block, wq_mul, 1),
    op_writes(wq_mul, qcur_pre_bias, region(matmul_output, [n_tokens, n_head_x_head_dim])),

    op(wq_bias_add),
    op_kind(wq_bias_add, ggml_add),
    op_inputs(wq_bias_add, [qcur_pre_bias, bq]),
    op_output(wq_bias_add, qcur_post_bias),
    op_level(wq_bias_add, primitive),
    sequence(qkv_block, wq_bias_add, 2),
    op_reads(wq_bias_add, qcur_pre_bias, region(elementwise, [n_tokens, n_head_x_head_dim])),

    %% External fact that should be preserved
    parameter(wq, layer(il), from_hparams),
    parameter(bq, layer(il), from_hparams)
]).

fusion_to_apply(fusion(epilogue_matmul_elementwise,
                       [wq_mul, wq_bias_add],
                       bit_exact)).

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

%% Test 1: applying creates a new fused op with appropriate kind
test_apply_creates_fused_op :-
    input_facts(In),
    fusion_to_apply(F),
    apply_fusion:apply_fusion_to_facts(In, F, Out),
    member(op(FusedName), Out),
    %% The fused name should reference both originals
    atom_concat('wq_mul_fused_', _, FusedName),
    %% The kind should be fused(build_lora_mm, ggml_add)
    member(op_kind(FusedName, fused(build_lora_mm, ggml_add)), Out).

%% Test 2: applying removes the original two ops
test_apply_removes_original_ops :-
    input_facts(In),
    fusion_to_apply(F),
    apply_fusion:apply_fusion_to_facts(In, F, Out),
    \+ member(op(wq_mul), Out),
    \+ member(op(wq_bias_add), Out),
    \+ member(op_kind(wq_mul, _), Out),
    \+ member(op_kind(wq_bias_add, _), Out),
    \+ member(sequence(_, wq_mul, _), Out),
    \+ member(sequence(_, wq_bias_add, _), Out).

%% Test 3: applying removes the intermediate tensor's region facts
test_apply_removes_intermediate_tensor_facts :-
    input_facts(In),
    fusion_to_apply(F),
    apply_fusion:apply_fusion_to_facts(In, F, Out),
    %% qcur_pre_bias was the intermediate; its op_writes/op_reads should be gone
    \+ member(op_writes(_, qcur_pre_bias, _), Out),
    \+ member(op_reads(_, qcur_pre_bias, _), Out).

%% Test 4: applying preserves external facts (parameter declarations etc.)
test_apply_preserves_external_facts :-
    input_facts(In),
    fusion_to_apply(F),
    apply_fusion:apply_fusion_to_facts(In, F, Out),
    member(parameter(wq, layer(il), from_hparams), Out),
    member(parameter(bq, layer(il), from_hparams), Out).

%% Test 5: applying records provenance (which ops were fused)
test_apply_records_provenance :-
    input_facts(In),
    fusion_to_apply(F),
    apply_fusion:apply_fusion_to_facts(In, F, Out),
    member(fused_from(_FusedName, [wq_mul, wq_bias_add]), Out).

%% Test 6: applying merges inputs correctly
%% wq_mul has [wq, cur_after_norm]; wq_bias_add has [qcur_pre_bias, bq].
%% After fusion: qcur_pre_bias is consumed within the kernel, so fused
%% inputs should be [wq, cur_after_norm, bq] (no qcur_pre_bias).
test_apply_merged_inputs_correct :-
    input_facts(In),
    fusion_to_apply(F),
    apply_fusion:apply_fusion_to_facts(In, F, Out),
    member(op(FusedName), Out),
    member(op_inputs(FusedName, Merged), Out),
    %% Should contain wq, cur_after_norm, bq — NOT qcur_pre_bias
    member(wq, Merged),
    member(cur_after_norm, Merged),
    member(bq, Merged),
    \+ member(qcur_pre_bias, Merged).

%% Test 7: applying preserves the sequence position (uses Op1's seq num)
test_apply_preserves_sequence_position :-
    input_facts(In),
    fusion_to_apply(F),
    apply_fusion:apply_fusion_to_facts(In, F, Out),
    member(op(FusedName), Out),
    %% wq_mul was at sequence 1; fused op should be at sequence 1 (the earlier of the two)
    member(sequence(qkv_block, FusedName, 1), Out).

%% Test 8: full end-to-end — verify the output BPD is coherent enough
%% to be processed downstream. Specifically: every op() has matching
%% op_kind, op_inputs, op_output, op_level, sequence.
test_apply_to_qkv_q_projection :-
    input_facts(In),
    fusion_to_apply(F),
    apply_fusion:apply_fusion_to_facts(In, F, Out),
    %% For every op in Out, verify it has the required matching facts
    findall(O, member(op(O), Out), Ops),
    forall(member(O, Ops),
           ( member(op_kind(O, _), Out),
             member(op_inputs(O, _), Out),
             member(op_output(O, _), Out),
             member(op_level(O, _), Out),
             member(sequence(_, O, _), Out)
           )).

:- initialization(run_tests, main).
