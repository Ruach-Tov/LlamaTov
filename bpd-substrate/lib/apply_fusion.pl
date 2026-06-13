%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% apply_fusion.pl — Stage 4: APPLY a fusion rule to transform BPD facts.
%%
%% Per boneh's facilitation: this closes the validate→apply gap, which
%% is the last piece before "Prolog generates an optimal kernel from
%% BPD" works end-to-end for the QKV case.
%%
%% Trivial-case-first scope per Heath's methodology:
%%   - One fusion rule (epilogue_matmul_elementwise)
%%   - One fusion: wq_mul + wq_bias_add → fused_wq_mul_bias_add
%%   - Verify the post-fusion BPD facts are coherent and validate
%%
%% Future extensions (not this commit):
%%   - Multi-rule application (try multiple rules, pick best)
%%   - Cost-driven application (apply fusion that maximizes objective)
%%   - Conflict resolution between overlapping fusions
%%   - Iterative fusion (keep applying until no more rules match)

:- module(apply_fusion, [
    apply_epilogue_fusion/3,
    apply_elementwise_chain_fusion/3,
    apply_layout_transparent_fusion/3,
    apply_fusion_to_facts/3
]).

%% ────────────────────────────────────────────────────────────────────
%% apply_epilogue_fusion(+InputFacts, +FusionPair, -OutputFacts)
%% ────────────────────────────────────────────────────────────────────
%%
%% Given a list of BPD facts and a fusion pair, produce a new list of
%% BPD facts where the two fused ops are replaced by a single fused op.
%%
%% FusionPair: fusion(RuleName, [Op1, Op2], EquivalenceClass)
%%   from enumerate_valid_fusions/2 in fusion_rules.pl
%%
%% Transformation:
%%   - Remove op(Op1), op_kind(Op1, _), op_inputs(Op1, _), op_output(Op1, _),
%%     op_level(Op1, _), sequence(_, Op1, _), op_writes(Op1, ...),
%%     op_reads(Op1, ...) — i.e., everything pertaining to Op1
%%   - Remove the same for Op2
%%   - Add facts for the new fused op:
%%       op(FusedName)
%%       op_kind(FusedName, fused(Kind1, Kind2))
%%       op_inputs(FusedName, MergedInputs)
%%       op_output(FusedName, Op2's output)
%%       op_level(FusedName, matmul)  % still matmul-bound for cost purposes
%%       sequence(Block, FusedName, Op1's sequence)
%%   - Remove the intermediate tensor's region facts (op_writes/op_reads)

apply_epilogue_fusion(InputFacts, fusion(epilogue_matmul_elementwise,
                                          [Op1, Op2], _EqClass),
                       OutputFacts) :-
    %% Extract data about Op1 (the matmul) and Op2 (the elementwise)
    member(op_kind(Op1, Kind1), InputFacts),
    member(op_inputs(Op1, Inputs1), InputFacts),
    member(op_output(Op1, Intermediate), InputFacts),
    member(op_kind(Op2, Kind2), InputFacts),
    member(op_inputs(Op2, Inputs2), InputFacts),
    member(op_output(Op2, FinalOutput), InputFacts),
    member(sequence(Block, Op1, Seq1), InputFacts),

    %% Generate the fused op name
    atom_concat(Op1, '_fused_', Stem),
    atom_concat(Stem, Op2, FusedName),

    %% Merge inputs: Op1's inputs + Op2's inputs minus the intermediate
    delete(Inputs2, Intermediate, RemainingInputs2),
    append(Inputs1, RemainingInputs2, MergedInputs),

    %% Construct the new fact set:
    %% Remove everything pertaining to Op1, Op2, and Intermediate.
    %% Add fused-op facts.
    remove_op_facts(InputFacts, Op1, F1),
    remove_op_facts(F1, Op2, F2),
    remove_tensor_facts(F2, Intermediate, F3),

    NewFacts = [
        op(FusedName),
        op_kind(FusedName, fused(Kind1, Kind2)),
        op_inputs(FusedName, MergedInputs),
        op_output(FusedName, FinalOutput),
        op_level(FusedName, matmul),
        sequence(Block, FusedName, Seq1),
        %% Provenance: track which ops were fused
        fused_from(FusedName, [Op1, Op2])
    ],
    append(NewFacts, F3, OutputFacts).

%% ────────────────────────────────────────────────────────────────────
%% Helper: remove all facts pertaining to a specific op
%% ────────────────────────────────────────────────────────────────────

remove_op_facts(Facts, Op, Cleaned) :-
    exclude(fact_concerns_op(Op), Facts, Cleaned).

fact_concerns_op(Op, op(Op)).
fact_concerns_op(Op, op_kind(Op, _)).
fact_concerns_op(Op, op_inputs(Op, _)).
fact_concerns_op(Op, op_output(Op, _)).
fact_concerns_op(Op, op_level(Op, _)).
fact_concerns_op(Op, op_condition(Op, _)).
fact_concerns_op(Op, sequence(_, Op, _)).
fact_concerns_op(Op, op_writes(Op, _, _)).
fact_concerns_op(Op, op_reads(Op, _, _)).
fact_concerns_op(Op, cb_after_v2(Op, _, _)).

%% ────────────────────────────────────────────────────────────────────
%% Helper: remove all facts pertaining to a specific tensor
%% ────────────────────────────────────────────────────────────────────
%%
%% After fusion, the intermediate tensor (e.g., Qcur_pre_bias) no longer
%% exists — it's been absorbed into the fused op's register-only epilogue.
%% Remove any facts that reference it.

remove_tensor_facts(Facts, Tensor, Cleaned) :-
    exclude(fact_mentions_tensor(Tensor), Facts, Cleaned).

fact_mentions_tensor(Tensor, op_writes(_, Tensor, _)).
fact_mentions_tensor(Tensor, op_reads(_, Tensor, _)).

%% ────────────────────────────────────────────────────────────────────
%% apply_fusion_to_facts(+InputFacts, +Fusion, -OutputFacts)
%% ────────────────────────────────────────────────────────────────────
%%
%% Dispatch wrapper: routes a fusion to the appropriate apply predicate
%% based on the rule name. Currently only epilogue_matmul_elementwise
%% is supported; future extensions add more rule-name clauses.

apply_fusion_to_facts(InputFacts,
                      fusion(epilogue_matmul_elementwise, Ops, EqClass),
                      OutputFacts) :-
    apply_epilogue_fusion(InputFacts,
                           fusion(epilogue_matmul_elementwise, Ops, EqClass),
                           OutputFacts).

apply_fusion_to_facts(InputFacts,
                      fusion(elementwise_chain, Ops, EqClass),
                      OutputFacts) :-
    apply_elementwise_chain_fusion(InputFacts,
                                    fusion(elementwise_chain, Ops, EqClass),
                                    OutputFacts).

apply_fusion_to_facts(InputFacts,
                      fusion(layout_transparent, Ops, EqClass),
                      OutputFacts) :-
    apply_layout_transparent_fusion(InputFacts,
                                     fusion(layout_transparent, Ops, EqClass),
                                     OutputFacts).

%% ────────────────────────────────────────────────────────────────────
%% apply_elementwise_chain_fusion/3
%% ────────────────────────────────────────────────────────────────────
%%
%% Fuse two consecutive elementwise ops where Op2 consumes Op1's output.
%% The intermediate tensor is absorbed into the fused kernel's register-only
%% computation.
%%
%% Structurally identical to apply_epilogue_fusion EXCEPT:
%%   - Both ops are elementwise-class (not matmul + elementwise)
%%   - The fused op is elementwise-class (not matmul)
%%   - Equivalence class: bit_exact (no reordering of floating-point ops)
%%
%% Example: silu(matmul_output) → mul(silu_output, gate)
%%   Op1 = ggml_silu, Op2 = ggml_mul
%%   Fused = single kernel doing silu then mul in registers

apply_elementwise_chain_fusion(
        InputFacts,
        fusion(elementwise_chain, [Op1, Op2], _EqClass),
        OutputFacts) :-
    %% Extract data about Op1 and Op2
    member(op_kind(Op1, Kind1), InputFacts),
    member(op_inputs(Op1, Inputs1), InputFacts),
    member(op_output(Op1, Intermediate), InputFacts),
    member(op_kind(Op2, Kind2), InputFacts),
    member(op_inputs(Op2, Inputs2), InputFacts),
    member(op_output(Op2, FinalOutput), InputFacts),
    member(sequence(Block, Op1, Seq1), InputFacts),

    %% Generate the fused op name
    atom_concat(Op1, '_fused_', Stem),
    atom_concat(Stem, Op2, FusedName),

    %% Merge inputs: Op1's inputs + Op2's inputs minus the intermediate
    delete(Inputs2, Intermediate, RemainingInputs2),
    append(Inputs1, RemainingInputs2, MergedInputs),

    %% Remove everything pertaining to Op1, Op2, and Intermediate
    remove_op_facts(InputFacts, Op1, F1),
    remove_op_facts(F1, Op2, F2),
    remove_tensor_facts(F2, Intermediate, F3),

    NewFacts = [
        op(FusedName),
        op_kind(FusedName, fused(Kind1, Kind2)),
        op_inputs(FusedName, MergedInputs),
        op_output(FusedName, FinalOutput),
        op_level(FusedName, primitive),    % elementwise stays primitive
        sequence(Block, FusedName, Seq1),
        fused_from(FusedName, [Op1, Op2])
    ],
    append(NewFacts, F3, OutputFacts).

%% ────────────────────────────────────────────────────────────────────
%% apply_layout_transparent_fusion/3
%% ────────────────────────────────────────────────────────────────────
%%
%% Eliminate a reshape op between a producer and consumer. The reshape
%% op disappears; the consumer is REWIRED to read the original source
%% directly (since reshape is just a view of the same memory).
%%
%% Structurally different from epilogue/elementwise_chain:
%%   - The reshape op is ELIMINATED, not merged
%%   - The CONSUMER op is preserved (its inputs change)
%%   - No new fused op is created
%%   - Just an input rewire on the consumer
%%
%% Example: reshape_3d(Qcur_pre, n_embd_head, n_head, n_tokens) → output
%%          rope_ext(reshaped) → rotated
%%   After: rope_ext(Qcur_pre directly) → rotated (rope reads source layout)
%%
%% The consumer must be capable of addressing the source's layout. The
%% rule's precondition guarantees this is safe.
%%
%% Equivalence class: bit_exact (no computation changes; just view).

apply_layout_transparent_fusion(
        InputFacts,
        fusion(layout_transparent, [ReshapeOp, ConsumerOp], _EqClass),
        OutputFacts) :-
    %% Extract ReshapeOp's source and output tensors
    member(op_inputs(ReshapeOp, ReshapeInputs), InputFacts),
    member(op_output(ReshapeOp, Reshaped), InputFacts),
    %% First input of ggml_reshape_3d after ctx0 is the source tensor.
    %% Pattern: op_inputs(reshape_op, [ctx0, Source, dim1, dim2, dim3]).
    %% For simpler reshape forms, just take the input that isn't a dim
    %% spec (i.e., a tensor reference). Conservative: use second arg.
    reshape_source(ReshapeInputs, Source),

    %% Extract ConsumerOp's inputs and rewire Reshaped → Source
    member(op_inputs(ConsumerOp, ConsumerInputs), InputFacts),
    rewire_input_list(ConsumerInputs, Reshaped, Source, NewConsumerInputs),

    %% Remove the reshape op entirely
    remove_op_facts(InputFacts, ReshapeOp, F1),
    %% Remove the Reshaped tensor's region facts (no longer exists)
    remove_tensor_facts(F1, Reshaped, F2),
    %% Replace ConsumerOp's op_inputs/2 with the rewired version
    select(op_inputs(ConsumerOp, ConsumerInputs), F2, F3),

    NewFacts = [
        op_inputs(ConsumerOp, NewConsumerInputs),
        %% Provenance: record that the layout op was eliminated
        layout_eliminated(ReshapeOp, between(Source, ConsumerOp))
    ],
    append(NewFacts, F3, OutputFacts).

%% reshape_source(+ReshapeInputs, -Source)
%%   Extract the source tensor from a reshape op's input list.
%%   ggml_reshape_3d typically has inputs [ctx0, Source, dim1, dim2, dim3].
%%   The source is the input that's a tensor (not a dim spec).
%%   Conservative heuristic: take the second element (after ctx0).
reshape_source([_Ctx, Source | _Dims], Source) :- !.
reshape_source([Source | _], Source).

%% rewire_input_list(+OldInputs, +OldName, +NewName, -NewInputs)
%%   Replace OldName with NewName in the inputs list.
rewire_input_list([], _, _, []).
rewire_input_list([OldName | Rest], OldName, NewName, [NewName | Rest1]) :-
    !,
    rewire_input_list(Rest, OldName, NewName, Rest1).
rewire_input_list([X | Rest], OldName, NewName, [X | Rest1]) :-
    rewire_input_list(Rest, OldName, NewName, Rest1).
