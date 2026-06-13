%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_iterative_fusion.pl — verify fixpoint fusion iteration
%%
%% Trivial case: a BPD with THREE fusion candidates (Q, K, V bias-add
%% pairs from real qwen2.cpp QKV) should produce three fusions when
%% iterated to fixpoint.

:- use_module('../lib/iterative_fusion').

run_tests :-
    Tests = [
        test_fixpoint_on_empty_facts,
        test_fixpoint_on_no_candidates,
        test_fixpoint_one_candidate_one_fusion,
        test_fixpoint_qkv_block_three_fusions,
        test_fixpoint_terminates,
        test_iteration_preserves_op_count_invariant,
        test_fixpoint_provenance
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

%% A QKV block with three fusable bias-add pairs
qkv_block_facts([
    %% Q-projection + bias-add
    op(wq_mul),
    op_kind(wq_mul, build_lora_mm),
    op_inputs(wq_mul, [wq, cur]),
    op_output(wq_mul, qcur_pre_bias),
    op_level(wq_mul, builder),
    sequence(qkv_block, wq_mul, 1),
    op_writes(wq_mul, qcur_pre_bias, region(matmul_output, [n_tokens, dim])),

    op(wq_bias_add),
    op_kind(wq_bias_add, ggml_add),
    op_inputs(wq_bias_add, [qcur_pre_bias, bq]),
    op_output(wq_bias_add, qcur_post_bias),
    op_level(wq_bias_add, primitive),
    sequence(qkv_block, wq_bias_add, 2),
    op_reads(wq_bias_add, qcur_pre_bias, region(elementwise, [n_tokens, dim])),

    %% K-projection + bias-add
    op(wk_mul),
    op_kind(wk_mul, build_lora_mm),
    op_inputs(wk_mul, [wk, cur]),
    op_output(wk_mul, kcur_pre_bias),
    op_level(wk_mul, builder),
    sequence(qkv_block, wk_mul, 3),
    op_writes(wk_mul, kcur_pre_bias, region(matmul_output, [n_tokens, dim])),

    op(wk_bias_add),
    op_kind(wk_bias_add, ggml_add),
    op_inputs(wk_bias_add, [kcur_pre_bias, bk]),
    op_output(wk_bias_add, kcur_post_bias),
    op_level(wk_bias_add, primitive),
    sequence(qkv_block, wk_bias_add, 4),
    op_reads(wk_bias_add, kcur_pre_bias, region(elementwise, [n_tokens, dim])),

    %% V-projection + bias-add
    op(wv_mul),
    op_kind(wv_mul, build_lora_mm),
    op_inputs(wv_mul, [wv, cur]),
    op_output(wv_mul, vcur_pre_bias),
    op_level(wv_mul, builder),
    sequence(qkv_block, wv_mul, 5),
    op_writes(wv_mul, vcur_pre_bias, region(matmul_output, [n_tokens, dim])),

    op(wv_bias_add),
    op_kind(wv_bias_add, ggml_add),
    op_inputs(wv_bias_add, [vcur_pre_bias, bv]),
    op_output(wv_bias_add, vcur_post_bias),
    op_level(wv_bias_add, primitive),
    sequence(qkv_block, wv_bias_add, 6),
    op_reads(wv_bias_add, vcur_pre_bias, region(elementwise, [n_tokens, dim]))
]).

%% No-candidates facts: just two ops with no fusable relationship
no_candidate_facts([
    op(some_input),
    op_kind(some_input, ggml_silu),
    op_inputs(some_input, [x]),
    op_output(some_input, y),
    op_level(some_input, primitive),
    sequence(blk, some_input, 1),

    op(some_output),
    op_kind(some_output, build_norm(rms)),
    op_inputs(some_output, [z]),
    op_output(some_output, w),
    op_level(some_output, builder),
    sequence(blk, some_output, 2)
]).

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

test_fixpoint_on_empty_facts :-
    iterative_fusion:fixpoint_fuse([], Out),
    Out == [].

test_fixpoint_on_no_candidates :-
    no_candidate_facts(In),
    iterative_fusion:fixpoint_fuse(In, Out),
    %% Output should equal input (no fusions applied)
    length(In, L1),
    length(Out, L2),
    L1 == L2.

test_fixpoint_one_candidate_one_fusion :-
    qkv_block_facts(All),
    %% Extract only the Q-projection pair (first 14 facts of the fixture)
    length(QOnly, 14),
    append(QOnly, _, All),
    iterative_fusion:fixpoint_fuse(QOnly, [epilogue_matmul_elementwise],
                                   Out, Count),
    Count == 1,
    %% Result should have a fused op
    member(op_kind(_, fused(build_lora_mm, ggml_add)), Out).

test_fixpoint_qkv_block_three_fusions :-
    qkv_block_facts(In),
    iterative_fusion:fixpoint_fuse(In, [epilogue_matmul_elementwise],
                                   Out, Count),
    Count == 3,
    %% Three fused ops should exist
    findall(F, member(fused_from(F, _), Out), FusedOps),
    length(FusedOps, 3),
    format("    Fused ~d ops; output has ~d fused_from records~n",
           [Count, 3]).

test_fixpoint_terminates :-
    qkv_block_facts(In),
    %% Just verify fixpoint_fuse returns; if it didn't terminate this would timeout
    iterative_fusion:fixpoint_fuse(In, Out),
    is_list(Out).

test_iteration_preserves_op_count_invariant :-
    qkv_block_facts(In),
    iterative_fusion:fixpoint_fuse(In, [epilogue_matmul_elementwise],
                                   Out, Count),
    %% Each fusion REDUCES op count by 1 (two ops → one)
    findall(O, member(op(O), In), InOps),
    findall(O, member(op(O), Out), OutOps),
    length(InOps, InCount),
    length(OutOps, OutCount),
    Expected is InCount - Count,
    OutCount == Expected.

test_fixpoint_provenance :-
    qkv_block_facts(In),
    iterative_fusion:fixpoint_fuse(In, [epilogue_matmul_elementwise],
                                   Out, _Count),
    %% Each fused op should have provenance referring to its original ops
    findall(Sources, member(fused_from(_, Sources), Out), AllSources),
    forall(member(S, AllSources),
           (length(S, 2),  %% each fusion combines exactly 2 ops
            forall(member(Op, S), member(op(Op), In))   %% sources existed in input
           )).

:- initialization(run_tests, main).
