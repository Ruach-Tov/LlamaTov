%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_apply_fusion_multi_rule.pl — verify apply_fusion handles ALL THREE rule kinds
%%
%% Per Heath's "proceed to other rule kinds" directive: extend apply_fusion
%% from epilogue_matmul_elementwise to also handle:
%%   - elementwise_chain (silu → mul, etc.)
%%   - layout_transparent (reshape elimination)
%%
%% Three rule kinds covered; iterative_fusion can compose them.

:- use_module('../lib/apply_fusion').
:- use_module('../lib/iterative_fusion').

run_tests :-
    Tests = [
        test_elementwise_chain_creates_fused_op,
        test_elementwise_chain_removes_originals,
        test_elementwise_chain_correct_inputs,
        test_elementwise_chain_via_fixpoint,

        test_layout_transparent_eliminates_reshape,
        test_layout_transparent_rewires_consumer,
        test_layout_transparent_excludes_builder,

        test_iterative_applies_all_rule_kinds
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
%% Fixtures
%% ────────────────────────────────────────────────────────────────────

%% Two elementwise ops in a chain: silu producing intermediate, then mul.
%% This is the FFN SwiGLU gate pattern: silu(gate_proj) → mul(gate_silu, up_proj)
silu_mul_chain([
    op(silu_op),
    op_kind(silu_op, ggml_silu),
    op_inputs(silu_op, [gate_proj]),
    op_output(silu_op, gate_silu),
    op_level(silu_op, primitive),
    sequence(ffn_block, silu_op, 1),

    op(mul_op),
    op_kind(mul_op, ggml_mul),
    op_inputs(mul_op, [gate_silu, up_proj]),
    op_output(mul_op, fused_intermediate),
    op_level(mul_op, primitive),
    sequence(ffn_block, mul_op, 2)
]).

%% A reshape op feeding into a rope op (real qwen2 pattern)
reshape_rope_chain([
    op(qcur_reshape),
    op_kind(qcur_reshape, ggml_reshape_3d),
    op_inputs(qcur_reshape, [ctx0, qcur_pre_reshape, n_embd_head, n_head, n_tokens]),
    op_output(qcur_reshape, qcur_reshaped),
    op_level(qcur_reshape, primitive),
    sequence(qkv_block, qcur_reshape, 1),

    op(qcur_rope),
    op_kind(qcur_rope, ggml_rope_ext),
    op_inputs(qcur_rope, [ctx0, qcur_reshaped, inp_pos, freq_base]),
    op_output(qcur_rope, qcur_final),
    op_level(qcur_rope, primitive),
    sequence(qkv_block, qcur_rope, 2)
]).

%% Reshape feeding a BUILDER (should NOT be eligible for layout elimination)
reshape_to_builder([
    op(some_reshape),
    op_kind(some_reshape, ggml_reshape_3d),
    op_inputs(some_reshape, [ctx0, input_tensor, d1, d2, d3]),
    op_output(some_reshape, reshaped_for_builder),
    op_level(some_reshape, primitive),
    sequence(blk, some_reshape, 1),

    op(builder_op),
    op_kind(builder_op, build_norm(rms)),
    op_inputs(builder_op, [reshaped_for_builder, weights]),
    op_output(builder_op, normed),
    op_level(builder_op, builder),
    sequence(blk, builder_op, 2)
]).

%% Mixed: matmul→add (epilogue) + silu→mul (elementwise_chain)
mixed_rule_kinds([
    %% Matmul + bias (epilogue fusion target)
    op(matmul_op),
    op_kind(matmul_op, build_lora_mm),
    op_inputs(matmul_op, [weights, input]),
    op_output(matmul_op, matmul_out),
    op_level(matmul_op, builder),
    sequence(blk, matmul_op, 1),
    op_writes(matmul_op, matmul_out, region(matmul_output, [m, n])),

    op(bias_op),
    op_kind(bias_op, ggml_add),
    op_inputs(bias_op, [matmul_out, bias]),
    op_output(bias_op, biased),
    op_level(bias_op, primitive),
    sequence(blk, bias_op, 2),
    op_reads(bias_op, matmul_out, region(elementwise, [m, n])),

    %% Silu + Mul (elementwise_chain fusion target)
    op(silu_op),
    op_kind(silu_op, ggml_silu),
    op_inputs(silu_op, [biased]),
    op_output(silu_op, after_silu),
    op_level(silu_op, primitive),
    sequence(blk, silu_op, 3),

    op(mul_op),
    op_kind(mul_op, ggml_mul),
    op_inputs(mul_op, [after_silu, gate]),
    op_output(mul_op, final),
    op_level(mul_op, primitive),
    sequence(blk, mul_op, 4)
]).

%% ────────────────────────────────────────────────────────────────────
%% Tests for elementwise_chain
%% ────────────────────────────────────────────────────────────────────

test_elementwise_chain_creates_fused_op :-
    silu_mul_chain(In),
    Fusion = fusion(elementwise_chain, [silu_op, mul_op], bit_exact),
    apply_fusion:apply_fusion_to_facts(In, Fusion, Out),
    member(op(_FusedName), Out),
    member(op_kind(_, fused(ggml_silu, ggml_mul)), Out).

test_elementwise_chain_removes_originals :-
    silu_mul_chain(In),
    Fusion = fusion(elementwise_chain, [silu_op, mul_op], bit_exact),
    apply_fusion:apply_fusion_to_facts(In, Fusion, Out),
    \+ member(op(silu_op), Out),
    \+ member(op(mul_op), Out).

test_elementwise_chain_correct_inputs :-
    silu_mul_chain(In),
    Fusion = fusion(elementwise_chain, [silu_op, mul_op], bit_exact),
    apply_fusion:apply_fusion_to_facts(In, Fusion, Out),
    member(op_inputs(_FusedName, MergedInputs), Out),
    member(gate_proj, MergedInputs),
    member(up_proj, MergedInputs),
    %% Intermediate (gate_silu) should NOT appear
    \+ member(gate_silu, MergedInputs).

test_elementwise_chain_via_fixpoint :-
    silu_mul_chain(In),
    iterative_fusion:fixpoint_fuse(In, [elementwise_chain], Out, Count),
    Count == 1,
    member(op_kind(_, fused(ggml_silu, ggml_mul)), Out).

%% ────────────────────────────────────────────────────────────────────
%% Tests for layout_transparent
%% ────────────────────────────────────────────────────────────────────

test_layout_transparent_eliminates_reshape :-
    reshape_rope_chain(In),
    Fusion = fusion(layout_transparent, [qcur_reshape, qcur_rope], bit_exact),
    apply_fusion:apply_fusion_to_facts(In, Fusion, Out),
    %% Reshape op should be gone
    \+ member(op(qcur_reshape), Out),
    \+ member(op_kind(qcur_reshape, _), Out),
    %% Provenance fact should record the elimination
    member(layout_eliminated(qcur_reshape, _), Out).

test_layout_transparent_rewires_consumer :-
    reshape_rope_chain(In),
    Fusion = fusion(layout_transparent, [qcur_reshape, qcur_rope], bit_exact),
    apply_fusion:apply_fusion_to_facts(In, Fusion, Out),
    %% Rope op should now read from qcur_pre_reshape (the source)
    member(op_inputs(qcur_rope, NewInputs), Out),
    member(qcur_pre_reshape, NewInputs),
    %% qcur_reshaped (the eliminated intermediate) should NOT appear
    \+ member(qcur_reshaped, NewInputs).

test_layout_transparent_excludes_builder :-
    %% A reshape feeding a builder should NOT match layout_transparent
    %% (per medayek's P3 finding, fixed in commit 37c198162)
    reshape_to_builder(In),
    iterative_fusion:enumerate_with_facts([layout_transparent], In, Fusions),
    %% No layout_transparent pair should be admissible
    Fusions == [].

%% ────────────────────────────────────────────────────────────────────
%% Integration test: iterative fusion applies ALL rule kinds
%% ────────────────────────────────────────────────────────────────────

test_iterative_applies_all_rule_kinds :-
    mixed_rule_kinds(In),
    AllRules = [epilogue_matmul_elementwise, elementwise_chain,
                layout_transparent],
    iterative_fusion:fixpoint_fuse(In, AllRules, Out, Count),
    %% Should apply 2 fusions: matmul+bias (epilogue) and silu+mul (elementwise_chain)
    Count >= 2,
    %% Output should contain BOTH fused ops
    member(op_kind(_, fused(build_lora_mm, ggml_add)), Out),
    member(op_kind(_, fused(ggml_silu, ggml_mul)), Out),
    format("    Applied ~d fusions across multiple rule kinds~n", [Count]).

:- initialization(run_tests, main).
