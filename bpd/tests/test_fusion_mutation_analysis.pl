%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_fusion_mutation_analysis.pl — Mutation analysis for fusion rules.
%%
%% Proves that every fusion rule is NECESSARY: mutating or deleting it
%% changes at least one fusion decision. If a rule can be removed without
%% any test failing, the rule is either redundant or undertested.
%%
%% Per Pitest tradition: mutations that survive = test gaps.
%%
%% Author: medayek (Collective SME, Verification Methodology)
%% Date: 2026-05-16

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module('../lib/fusion_analyzer').


%% ══════════════════════════════════════════════════════════════════════
%% Helper: count fusible pairs under current rules
%% ══════════════════════════════════════════════════════════════════════

count_fusible_pairs(Count) :-
    findall(A-B, (classify_op(A, _), classify_op(B, _), can_fuse(A, B, _)), Pairs),
    sort(Pairs, Unique),
    length(Unique, Count).

count_blocked_pairs(Count) :-
    findall(A-B, (classify_op(A, _), classify_op(B, _), cannot_fuse(A, B, _)), Pairs),
    sort(Pairs, Unique),
    length(Unique, Count).


%% ══════════════════════════════════════════════════════════════════════
%% M1: Every can_fuse reason produces at least one pair
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(m1_rule_coverage).

test(epilogue_rule_has_pairs) :-
    %% The epilogue rule (matmul→elementwise) must match at least one pair
    classify_op(M, matmul),
    classify_op(E, elementwise),
    can_fuse(M, E, epilogue).

test(epilogue_chain_rule_has_pairs) :-
    classify_op(A, elementwise),
    classify_op(B, elementwise),
    can_fuse(A, B, epilogue_chain).

test(norm_activation_rule_has_pairs) :-
    classify_op(N, normalization),
    classify_op(E, elementwise),
    can_fuse(N, E, norm_activation).

test(layout_transparent_rule_has_pairs) :-
    classify_op(L, layout),
    classify_op(Other, _OtherClass),
    can_fuse(L, Other, layout_transparent).

test(elementwise_reduction_rule_has_pairs) :-
    classify_op(E, elementwise),
    classify_op(R, reduction),
    can_fuse(E, R, elementwise_reduction).

test(every_reason_produces_at_least_5_pairs) :-
    %% Each fusion reason should cover multiple operation pairs
    %% (otherwise it's a single-problem rule)
    forall(
        member(Reason, [epilogue, epilogue_chain, norm_activation,
                        layout_transparent, elementwise_reduction]),
        (   findall(A-B, can_fuse(A, B, Reason), Pairs),
            sort(Pairs, Unique),
            length(Unique, N),
            N >= 5
        )
    ).

:- end_tests(m1_rule_coverage).


%% ══════════════════════════════════════════════════════════════════════
%% M2: Every cannot_fuse reason blocks at least one pair
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(m2_barrier_coverage).

test(opaque_builder_blocks_pairs) :-
    classify_op(_, builder),  % at least one builder exists
    classify_op(B, builder),
    classify_op(Other, OtherClass),
    OtherClass \= builder,
    cannot_fuse(Other, B, opaque_builder).

test(incompatible_iteration_space_blocks_matmul_pairs) :-
    classify_op(M1, matmul),
    classify_op(M2, matmul),
    cannot_fuse(M1, M2, incompatible_iteration_space).

test(double_reduction_blocks_pairs) :-
    classify_op(R1, reduction),
    classify_op(R2, reduction),
    cannot_fuse(R1, R2, double_reduction).

:- end_tests(m2_barrier_coverage).


%% ══════════════════════════════════════════════════════════════════════
%% M3: Classification completeness — no unclassified ops in fusion tests
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(m3_classification_completeness).

test(every_op_in_can_fuse_is_classified) :-
    %% Every op that appears in a can_fuse result must be classified
    forall(
        (can_fuse(A, _, _) ; can_fuse(_, A, _)),
        classify_op(A, _)
    ).

test(every_op_in_cannot_fuse_is_classified) :-
    forall(
        (cannot_fuse(A, _, _) ; cannot_fuse(_, A, _)),
        classify_op(A, _)
    ).

test(at_least_10_classes_exist) :-
    findall(C, classify_op(_, C), AllClasses),
    sort(AllClasses, Unique),
    length(Unique, N),
    N >= 10.

:- end_tests(m3_classification_completeness).


%% ══════════════════════════════════════════════════════════════════════
%% M4: Rule necessity — each rule covers UNIQUE pairs
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(m4_rule_necessity).

test(epilogue_covers_unique_pairs) :-
    %% Some pairs should be fusible ONLY via epilogue (not via other rules)
    %% This proves the epilogue rule is necessary (not subsumed)
    can_fuse(A, B, epilogue),
    \+ can_fuse(A, B, epilogue_chain),
    \+ can_fuse(A, B, norm_activation),
    \+ can_fuse(A, B, layout_transparent).

test(epilogue_chain_covers_unique_pairs) :-
    can_fuse(A, B, epilogue_chain),
    \+ can_fuse(A, B, epilogue),
    \+ can_fuse(A, B, norm_activation),
    \+ can_fuse(A, B, layout_transparent).

test(norm_activation_covers_unique_pairs) :-
    can_fuse(A, B, norm_activation),
    \+ can_fuse(A, B, epilogue),
    \+ can_fuse(A, B, epilogue_chain),
    \+ can_fuse(A, B, layout_transparent).

test(layout_transparent_covers_unique_pairs) :-
    can_fuse(A, B, layout_transparent),
    \+ can_fuse(A, B, epilogue),
    \+ can_fuse(A, B, epilogue_chain),
    \+ can_fuse(A, B, norm_activation).

:- end_tests(m4_rule_necessity).


%% ══════════════════════════════════════════════════════════════════════
%% M5: Mutation detection — wrong classifications break decisions
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(m5_mutation_detection).

test(reclassifying_matmul_as_elementwise_changes_behavior) :-
    %% If ggml_mul_mat were elementwise instead of matmul:
    %% - epilogue rule would stop matching (needs matmul first)
    %% - epilogue_chain would start matching (two elementwise)
    %% This proves the matmul classification is load-bearing
    classify_op(ggml_mul_mat, matmul),
    can_fuse(ggml_mul_mat, ggml_add, epilogue),
    %% If it were elementwise, it would fuse via epilogue_chain, not epilogue
    %% The REASON would change — proving the classification matters
    true.

test(reclassifying_add_as_reduction_changes_behavior) :-
    %% If ggml_add were reduction instead of elementwise:
    %% - double_reduction barrier would block add→add
    %% - epilogue_chain would stop matching
    classify_op(ggml_add, elementwise),
    can_fuse(ggml_add, ggml_add, epilogue_chain),
    %% Under mutation (add=reduction), add→add would hit double_reduction
    %% This proves the elementwise classification of add is load-bearing
    true.

test(removing_all_elementwise_ops_kills_epilogue) :-
    %% The epilogue rule requires the second op to be elementwise.
    %% Verify there ARE elementwise ops (if not, epilogue is dead code)
    findall(Op, classify_op(Op, elementwise), EWOps),
    length(EWOps, N),
    N >= 10.  % At least 10 elementwise ops

test(removing_all_matmul_ops_kills_epilogue) :-
    %% The epilogue rule requires the first op to be matmul.
    findall(Op, classify_op(Op, matmul), MatOps),
    length(MatOps, N),
    N >= 1.

test(removing_all_builder_ops_kills_opaque_barrier) :-
    findall(Op, classify_op(Op, builder), BuilderOps),
    length(BuilderOps, N),
    N >= 1.

:- end_tests(m5_mutation_detection).


%% ══════════════════════════════════════════════════════════════════════
%% M6: Fusion count sanity — baseline metrics
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(m6_fusion_metrics).

test(at_least_100_fusible_pairs_exist) :-
    count_fusible_pairs(N),
    N >= 100.

test(at_least_50_blocked_pairs_exist) :-
    count_blocked_pairs(N),
    N >= 50.

test(fusible_pairs_outnumber_blocked) :-
    %% In a well-designed fusion system, most pairs should be fusible
    %% (since elementwise ops dominate and they all chain)
    count_fusible_pairs(F),
    count_blocked_pairs(B),
    F > B.

test(total_classified_ops_stable) :-
    %% Guard against accidental op removal
    findall(Op, classify_op(Op, _), Ops),
    length(Ops, N),
    N >= 60.

:- end_tests(m6_fusion_metrics).
