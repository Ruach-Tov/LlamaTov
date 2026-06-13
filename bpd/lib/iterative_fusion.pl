%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% iterative_fusion.pl — Apply fusion rules until fixpoint reached.
%%
%% Per Heath's "treat bounded subtasks as urgent" directive: extends
%% apply_fusion from one-fusion-at-a-time to FIXPOINT iteration.
%% Real BPDs have multiple fusion candidates; the substrate should
%% find them all and apply them.
%%
%% Algorithm (substrate-honest, terminating):
%%   1. Enumerate all valid fusions in current BPD facts
%%   2. If none, return current facts (fixpoint reached)
%%   3. Pick one fusion (currently: first; future: cost-driven)
%%   4. Apply it via apply_fusion
%%   5. Recurse with the transformed facts
%%
%% Termination guarantee: each fusion REDUCES the op count by 1
%% (two ops become one). Starting from N ops, at most N-1 iterations.
%%
%% This is the fixpoint computation that closes the "one fusion at a time"
%% gap to "fully fused BPD."

:- module(iterative_fusion, [
    fixpoint_fuse/2,
    fixpoint_fuse/4,
    enumerate_with_facts/3,
    fusion_count/3
]).

:- use_module('apply_fusion').
:- use_module('fusion_rules').
:- use_module('fusion_cost').    % the PROFITABILITY gate (validity != profitability)

%% ────────────────────────────────────────────────────────────────────
%% fixpoint_fuse(+InputFacts, -OutputFacts)
%% ────────────────────────────────────────────────────────────────────
%%
%% Apply fusion rules iteratively until no more candidates exist.
%% Returns the final fused BPD facts.

fixpoint_fuse(InputFacts, OutputFacts) :-
    fixpoint_fuse(InputFacts, [epilogue_matmul_elementwise],
                  OutputFacts, _ApplicationCount).

%% fixpoint_fuse(+InputFacts, +RuleNames, -OutputFacts, -ApplicationCount)
%% Like fixpoint_fuse/2 but allows specifying which rules to apply
%% and returns the count of fusions applied.

fixpoint_fuse(InputFacts, RuleNames, OutputFacts, ApplicationCount) :-
    fixpoint_iterate(InputFacts, RuleNames, 0, OutputFacts, ApplicationCount).

fixpoint_iterate(Facts, RuleNames, CountSoFar, OutputFacts, FinalCount) :-
    enumerate_with_facts(RuleNames, Facts, AllFusions),
    %% PROFITABILITY GATE: a fusion can be VALID (enumerate_with_facts only
    %% checks validity) yet NOT PROFITABLE — generator-prologue fusions
    %% (im2col/gather/broadcast-expand) trade memory traffic for recomputation,
    %% and the trade is shape-dependent. Filter out fusions the cost model
    %% declines, BEFORE picking one. Epilogue/elementwise/layout rules return
    %% 'always' and pass through unchanged; only cost-dependent rules are gated.
    include(fusion_is_profitable(Facts), AllFusions, Fusions),
    (   Fusions = []
    ->  %% Fixpoint reached (no more PROFITABLE fusions — valid-but-unprofitable
        %% candidates are intentionally left un-applied).
        OutputFacts = Facts,
        FinalCount = CountSoFar
    ;   %% Apply first profitable fusion
        Fusions = [Fusion | _],
        apply_fusion:apply_fusion_to_facts(Facts, Fusion, NewFacts),
        Count1 is CountSoFar + 1,
        fixpoint_iterate(NewFacts, RuleNames, Count1, OutputFacts, FinalCount)
    ).

%% fusion_is_profitable(+Facts, +Fusion) — true iff the cost model approves.
%% Maps the enumerated fusion(RuleName, [Op1,Op2], _) to a fusion_profitable/3
%% query. Rules with an 'always' verdict (epilogue/elementwise/layout) pass
%% unconditionally; generator-prologue rules consult the calibrated cost model
%% with bindings extracted from the facts.
%% Generator-prologue: cost-gated (consult the calibrated model with bindings).
fusion_is_profitable(Facts, fusion(generator_prologue, [Op1, Op2], _Eq)) :- !,
    fusion_binds(Facts, Op1, Op2, Binds),
    fusion_cost:fusion_profitable(generator_prologue, Binds, Verdict),
    Verdict \= unprofitable(_).
%% Always-profitable rules (epilogue/elementwise/layout): the cost model returns
%% 'always' for these named rules (queried with a ground rule name only).
fusion_is_profitable(_Facts, fusion(RuleName, _Ops, _Eq)) :-
    fusion_cost:fusion_profitable(RuleName, _, always), !.
%% Any other rule (validity already held, no cost entry): allow it.
fusion_is_profitable(_Facts, _Fusion).

%% fusion_binds(+Facts, +GenOp, +ConsumerOp, -binds(GenKind, NElems, M, BM))
%% Extract the cost-model bindings: the generator kind, the intermediate's
%% element count, and the consuming GEMM's M / tile-BM. Falls back to a
%% conservative shape if the facts don't carry full metadata.
fusion_binds(Facts, GenOp, ConsumerOp, binds(GenKind, NElems, M, BM)) :-
    ( member(op_kind(GenOp, GenKind0), Facts) -> GenKind = GenKind0 ; GenKind = im2col ),
    ( member(op_output_elems(GenOp, NElems0), Facts) -> NElems = NElems0 ; NElems = 1000000 ),
    ( member(op_gemm_m(ConsumerOp, M0), Facts) -> M = M0 ; M = 128 ),
    ( member(op_gemm_bm(ConsumerOp, BM0), Facts) -> BM = BM0 ; BM = 64 ).

%% ────────────────────────────────────────────────────────────────────
%% enumerate_with_facts(+RuleNames, +Facts, -Fusions)
%% ────────────────────────────────────────────────────────────────────
%%
%% Variant of enumerate_valid_fusions that takes facts as input
%% rather than querying global predicate scope. This allows iteration
%% over transformed fact bases without mutating global state.
%%
%% Currently supports only epilogue_matmul_elementwise rule. Extension
%% to other rules is bounded follow-on work.

enumerate_with_facts(RuleNames, Facts, Fusions) :-
    findall(
        Fusion,
        ( member(RuleName, RuleNames),
          valid_pair_for_rule(RuleName, Facts, Op1, Op2),
          Fusion = fusion(RuleName, [Op1, Op2], bit_exact)
        ),
        Fusions).

%% valid_pair_for_rule(+RuleName, +Facts, -Op1, -Op2)
%%   Dispatch by rule name to the appropriate matcher.
valid_pair_for_rule(epilogue_matmul_elementwise, Facts, Op1, Op2) :-
    valid_epilogue_pair(Facts, Op1, Op2).
valid_pair_for_rule(elementwise_chain, Facts, Op1, Op2) :-
    valid_elementwise_chain_pair(Facts, Op1, Op2).
valid_pair_for_rule(layout_transparent, Facts, ReshapeOp, ConsumerOp) :-
    valid_layout_transparent_pair(Facts, ReshapeOp, ConsumerOp).

%% valid_epilogue_pair(+Facts, -Op1, -Op2) — find a matmul→elementwise pair
%% in the given facts. The pair must satisfy:
%%   - Op1 is matmul-class, Op2 is elementwise-class
%%   - Op2 reads Op1's output as one of its inputs
%%   - Op1's output has no other consumers
%%   - Region shapes match

valid_epilogue_pair(Facts, Op1, Op2) :-
    member(op(Op1), Facts),
    member(op_kind(Op1, Kind1), Facts),
    fusion_rules:op_class(Kind1, matmul),
    member(op_output(Op1, Intermediate), Facts),
    member(op(Op2), Facts),
    Op1 \== Op2,
    member(op_kind(Op2, Kind2), Facts),
    fusion_rules:op_class(Kind2, elementwise),
    member(op_inputs(Op2, Inputs2), Facts),
    member(Intermediate, Inputs2),
    %% Check no other consumers
    no_other_consumers(Facts, Intermediate, Op2).

%% valid_elementwise_chain_pair(+Facts, -Op1, -Op2) — both ops elementwise
%% Same structure as valid_epilogue_pair but both classes are elementwise.
%%
%% Critical: exclude pairs already covered by epilogue (matmul→elementwise)
%% so the two rules don't double-match on the same op pair. Since this
%% matcher requires BOTH ops to be elementwise-class, the matmul→elementwise
%% case is naturally excluded.

valid_elementwise_chain_pair(Facts, Op1, Op2) :-
    member(op(Op1), Facts),
    member(op_kind(Op1, Kind1), Facts),
    fusion_rules:op_class(Kind1, elementwise),
    member(op_output(Op1, Intermediate), Facts),
    member(op(Op2), Facts),
    Op1 \== Op2,
    member(op_kind(Op2, Kind2), Facts),
    fusion_rules:op_class(Kind2, elementwise),
    member(op_inputs(Op2, Inputs2), Facts),
    member(Intermediate, Inputs2),
    no_other_consumers(Facts, Intermediate, Op2).

%% valid_layout_transparent_pair(+Facts, -ReshapeOp, -ConsumerOp)
%%   Find a reshape op whose output is consumed by exactly one op,
%%   eligible for layout elimination.
%%
%% Excludes BUILDER consumers per medayek's property test finding
%% (cannot_fuse(opaque_builder) requires builders to NOT participate
%% in layout transparency — this was the root cause of the bug fixed
%% in commit 37c198162).

valid_layout_transparent_pair(Facts, ReshapeOp, ConsumerOp) :-
    member(op(ReshapeOp), Facts),
    member(op_kind(ReshapeOp, ggml_reshape_3d), Facts),
    member(op_output(ReshapeOp, Reshaped), Facts),
    member(op(ConsumerOp), Facts),
    ConsumerOp \== ReshapeOp,
    %% Consumer must NOT be a builder (medayek's P3 finding, commit 37c198162).
    %% Use op_level/2 (metayen's vocabulary): builders are opaque to fusion.
    member(op_level(ConsumerOp, ConsumerLevel), Facts),
    ConsumerLevel \== builder,
    member(op_inputs(ConsumerOp, ConsumerInputs), Facts),
    member(Reshaped, ConsumerInputs),
    no_other_consumers(Facts, Reshaped, ConsumerOp).

%% no_other_consumers(+Facts, +Tensor, +AllowedConsumer)
%%   Succeeds when no op besides AllowedConsumer consumes Tensor.
no_other_consumers(Facts, Tensor, AllowedConsumer) :-
    findall(Other,
            ( member(op_inputs(Other, OtherInputs), Facts),
              member(Tensor, OtherInputs),
              Other \== AllowedConsumer
            ),
            OtherConsumers),
    OtherConsumers = [].

%% ────────────────────────────────────────────────────────────────────
%% fusion_count(+InputFacts, +OutputFacts, -Count)
%% ────────────────────────────────────────────────────────────────────
%%
%% Count how many fusions occurred between input and output facts.
%% Equal to the number of fused_from/2 facts in output minus input.

fusion_count(InputFacts, OutputFacts, Count) :-
    findall(F, member(fused_from(F, _), InputFacts), InFused),
    findall(F, member(fused_from(F, _), OutputFacts), OutFused),
    length(OutFused, OutCount),
    length(InFused, InCount),
    Count is OutCount - InCount.
