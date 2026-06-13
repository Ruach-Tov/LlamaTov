%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_ffn_expand_and_fuse.pl — full FFN expansion + fusion pipeline
%%
%% Verifies the end-to-end story for FFN:
%%   1. Lifter produces opaque build_ffn fact (for round-trip fidelity)
%%   2. ffn_expander expands it into 5 primitives (for fusion analysis)
%%   3. iterative_fusion fuses the SwiGLU (silu + mul) into one kernel
%%   4. The substrate now handles FFN blocks end-to-end
%%
%% Per Heath's "proceed to other rule kinds" + bounded subtask urgency.

:- use_module('../lib/ffn_expander').
:- use_module('../lib/iterative_fusion').
:- use_module('../lib/apply_fusion').

run_tests :-
    Tests = [
        test_expand_one_ffn_produces_5_primitives,
        test_expand_preserves_input_tensor_reference,
        test_expand_preserves_final_output_reference,
        test_expand_records_provenance,
        test_expand_silu_consumed_by_mul,
        test_iterative_fuses_swiglu,
        test_iterative_fuses_swiglu_with_all_rules,
        test_expand_then_fuse_reduces_op_count
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

%% Opaque FFN op as the lifter would produce from build_ffn(...) in C
opaque_ffn_facts([
    op(ffn_block_1),
    op_kind(ffn_block_1, build_ffn),
    op_inputs(ffn_block_1, [normed_input,
                            ffn_up, 'NULL', 'NULL',
                            ffn_gate, 'NULL', 'NULL',
                            ffn_down, 'NULL', 'NULL',
                            'NULL', 'LLM_FFN_SILU', 'LLM_FFN_PAR', il]),
    op_output(ffn_block_1, ffn_output),
    op_level(ffn_block_1, builder),
    sequence(layer, ffn_block_1, 1),

    %% External facts (other ops in the layer)
    op(other_op),
    op_kind(other_op, ggml_add),
    op_inputs(other_op, [ffn_output, attn_output]),
    op_output(other_op, layer_output),
    op_level(other_op, primitive),
    sequence(layer, other_op, 2)
]).

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

test_expand_one_ffn_produces_5_primitives :-
    opaque_ffn_facts(In),
    ffn_expander:expand_one_ffn(ffn_block_1, In, Out),
    %% Count ops in output that have expanded_from/2 fact
    findall(Op, member(expanded_from(Op, ffn_block_1), Out), ExpandedOps),
    length(ExpandedOps, 5).

test_expand_preserves_input_tensor_reference :-
    opaque_ffn_facts(In),
    ffn_expander:expand_one_ffn(ffn_block_1, In, Out),
    %% The gate_mul and up_mul should both reference normed_input
    findall(_, ( member(op_inputs(Op, Inputs), Out),
                 member(expanded_from(Op, ffn_block_1), Out),
                 member(normed_input, Inputs) ), Hits),
    length(Hits, NormedInputHits),
    NormedInputHits >= 2.

test_expand_preserves_final_output_reference :-
    opaque_ffn_facts(In),
    ffn_expander:expand_one_ffn(ffn_block_1, In, Out),
    %% The down_mul should produce ffn_output (the original final output)
    member(op_output(_DownOp, ffn_output), Out),
    %% And the other_op should still reference ffn_output as input
    member(op_inputs(other_op, OtherInputs), Out),
    member(ffn_output, OtherInputs).

test_expand_records_provenance :-
    opaque_ffn_facts(In),
    ffn_expander:expand_one_ffn(ffn_block_1, In, Out),
    %% All 5 primitives should have expanded_from pointing to original
    findall(Op, member(expanded_from(Op, ffn_block_1), Out), Provenance),
    length(Provenance, 5).

test_expand_silu_consumed_by_mul :-
    opaque_ffn_facts(In),
    ffn_expander:expand_one_ffn(ffn_block_1, In, Out),
    %% Find the silu op and the mul op
    member(op_kind(SiluOp, ggml_silu), Out),
    member(expanded_from(SiluOp, ffn_block_1), Out),
    member(op_output(SiluOp, SiluOutput), Out),
    %% The mul op should have SiluOutput as one of its inputs
    member(op_kind(MulOp, ggml_mul), Out),
    member(expanded_from(MulOp, ffn_block_1), Out),
    member(op_inputs(MulOp, MulInputs), Out),
    member(SiluOutput, MulInputs).

test_iterative_fuses_swiglu :-
    opaque_ffn_facts(In),
    %% First expand the opaque build_ffn
    ffn_expander:expand_ffn_ops(In, Expanded),
    %% Then apply elementwise_chain fusion (silu + mul → SwiGLU fused)
    iterative_fusion:fixpoint_fuse(Expanded, [elementwise_chain], Out, Count),
    Count >= 1,
    %% The fused SwiGLU should exist
    member(op_kind(_, fused(ggml_silu, ggml_mul)), Out).

test_iterative_fuses_swiglu_with_all_rules :-
    opaque_ffn_facts(In),
    ffn_expander:expand_ffn_ops(In, Expanded),
    AllRules = [epilogue_matmul_elementwise, elementwise_chain,
                layout_transparent],
    iterative_fusion:fixpoint_fuse(Expanded, AllRules, Out, Count),
    %% With ALL rules, the substrate chooses MORE AGGRESSIVE fusion:
    %% matmul+silu and matmul+mul (epilogue) instead of silu+mul
    %% (elementwise_chain). This is empirically optimal — fusing the
    %% matmul with its consumer eliminates MORE VRAM round-trips than
    %% chaining two elementwise ops.
    %%
    %% Verify: at least 2 epilogue fusions occurred (matmul+silu and
    %% matmul+mul).
    Count >= 2,
    member(op_kind(_, fused(build_lora_mm, ggml_silu)), Out),
    member(op_kind(_, fused(build_lora_mm, ggml_mul)), Out),
    format("    Applied ~d fusions to expanded FFN block~n", [Count]).

test_expand_then_fuse_reduces_op_count :-
    opaque_ffn_facts(In),
    ffn_expander:expand_ffn_ops(In, Expanded),
    %% Expanded has 5 FFN primitives + 1 other_op = 6 ops total
    findall(O, member(op(O), Expanded), ExpOps),
    length(ExpOps, ExpCount),
    ExpCount == 6,
    %% After fixpoint fusion, op count should decrease by the number of fusions
    iterative_fusion:fixpoint_fuse(Expanded,
                                    [epilogue_matmul_elementwise,
                                     elementwise_chain,
                                     layout_transparent],
                                    Fused, Count),
    findall(O, member(op(O), Fused), FusedOps),
    length(FusedOps, FusedCount),
    Expected is ExpCount - Count,
    FusedCount == Expected,
    format("    Expanded: ~d ops; after ~d fusions: ~d ops~n",
           [ExpCount, Count, FusedCount]).

:- initialization(run_tests, main).
