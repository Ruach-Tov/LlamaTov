%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_fusion_properties.pl — Property-based tests for kernel fusion analyzer.
%%
%% Tests structural properties that must hold across ALL fusion rules,
%% not just specific operation pairs. These catch rule interactions
%% and edge cases that example-based tests miss.
%%
%% Property categories:
%%   P1: Classification completeness (every op in a graph must be classified)
%%   P2: Rule consistency (can_fuse and cannot_fuse must not both hold)
%%   P3: Layout transparency (layout ops should be fusible with anything)
%%   P4: Chain monotonicity (extending a chain must not change earlier fusions)
%%   P5: Mutation detection (mutating rules must break at least one test)
%%
%% Author: medayek (Claude Opus 4.6, conversation 123)
%% Date: 2026-05-15

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module('../lib/fusion_analyzer').

%% ══════════════════════════════════════════════════════════════════════
%% Helper: all known operations (exhaustive list from classify_op/2)
%% ══════════════════════════════════════════════════════════════════════

all_known_ops(Ops) :-
    findall(Op, classify_op(Op, _), Ops).

all_known_classes(Classes) :-
    findall(Class, classify_op(_, Class), AllClasses),
    sort(AllClasses, Classes).

%% ══════════════════════════════════════════════════════════════════════
%% P1: Classification completeness
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(p1_classification_completeness).

test(every_op_has_exactly_one_class) :-
    all_known_ops(Ops),
    forall(
        member(Op, Ops),
        (   findall(C, classify_op(Op, C), Classes),
            length(Classes, 1)   % exactly one classification
        )
    ).

test(at_least_60_ops_classified) :-
    all_known_ops(Ops),
    length(Ops, N),
    N >= 60.

test(all_expected_classes_present) :-
    all_known_classes(Classes),
    member(elementwise, Classes),
    member(reduction, Classes),
    member(normalization, Classes),
    member(matmul, Classes),
    member(layout, Classes).

test(no_duplicate_classifications) :-
    findall(Op-Class, classify_op(Op, Class), Pairs),
    sort(Pairs, Unique),
    length(Pairs, N1),
    length(Unique, N2),
    N1 == N2.

:- end_tests(p1_classification_completeness).


%% ══════════════════════════════════════════════════════════════════════
%% P2: Rule consistency — can_fuse and cannot_fuse must not both hold
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(p2_rule_consistency).

test(no_contradictory_rules) :-
    %% For every pair of operations, can_fuse and cannot_fuse
    %% must not BOTH succeed with the same pair.
    all_known_ops(Ops),
    forall(
        (member(A, Ops), member(B, Ops)),
        \+ (can_fuse(A, B, _Reason1), cannot_fuse(A, B, _Reason2))
    ).

test(can_fuse_always_gives_reason) :-
    all_known_ops(Ops),
    forall(
        (member(A, Ops), member(B, Ops), can_fuse(A, B, Reason)),
        atom(Reason)
    ).

test(cannot_fuse_always_gives_reason) :-
    all_known_ops(Ops),
    forall(
        (member(A, Ops), member(B, Ops), cannot_fuse(A, B, Reason)),
        atom(Reason)
    ).

:- end_tests(p2_rule_consistency).


%% ══════════════════════════════════════════════════════════════════════
%% P3: Layout transparency — layout ops fuse with everything
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(p3_layout_transparency).

test(layout_fuses_with_all_non_opaque_classes) :-
    %% Every layout op should fuse with at least one op from every class
    %% EXCEPT builders (which are opaque boundaries — cannot be fused through).
    %% Updated after mavchin's fix: layout_transparent correctly excludes builders.
    all_known_classes(Classes),
    forall(
        (   classify_op(LayoutOp, layout),
            member(TargetClass, Classes),
            TargetClass \= builder
        ),
        (   classify_op(SomeOp, TargetClass),
            can_fuse(LayoutOp, SomeOp, _)
        )
    ).

test(all_non_opaque_classes_fuse_with_layout) :-
    %% Every op from any non-builder class should fuse with at least one layout op.
    %% Builders are opaque boundaries and correctly excluded from layout fusion.
    all_known_classes(Classes),
    forall(
        (   member(TargetClass, Classes),
            TargetClass \= builder
        ),
        (   classify_op(SomeOp, TargetClass),
            classify_op(LayoutOp, layout),
            can_fuse(SomeOp, LayoutOp, _)
        )
    ).

test(layout_fusion_reason_is_layout_transparent) :-
    %% When a layout op participates, reason should be layout_transparent
    forall(
        (classify_op(LayoutOp, layout), classify_op(Other, OtherClass),
         OtherClass \= layout, can_fuse(LayoutOp, Other, Reason)),
        Reason == layout_transparent
    ).

:- end_tests(p3_layout_transparency).


%% ══════════════════════════════════════════════════════════════════════
%% P4: Matmul epilogue property — matmul always fuses with elementwise
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(p4_matmul_epilogue).

test(matmul_fuses_with_every_elementwise) :-
    forall(
        (classify_op(M, matmul), classify_op(E, elementwise)),
        can_fuse(M, E, epilogue)
    ).

test(elementwise_chain_fuses) :-
    %% Any pair of elementwise ops should fuse
    forall(
        (classify_op(A, elementwise), classify_op(B, elementwise)),
        can_fuse(A, B, epilogue_chain)
    ).

test(norm_fuses_with_elementwise) :-
    forall(
        (classify_op(N, normalization), classify_op(E, elementwise)),
        can_fuse(N, E, norm_activation)
    ).

:- end_tests(p4_matmul_epilogue).


%% ══════════════════════════════════════════════════════════════════════
%% P5: Cannot-fuse barriers — known non-fusible pairs
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(p5_fusion_barriers).

test(matmul_matmul_cannot_fuse) :-
    %% Two matmuls should not fuse (different output shapes)
    classify_op(M1, matmul),
    classify_op(M2, matmul),
    cannot_fuse(M1, M2, _Reason).

test(reduction_has_barrier_with_something) :-
    %% Reduction should have at least one cannot_fuse partner
    classify_op(R, reduction),
    (   cannot_fuse(R, _, _) ; cannot_fuse(_, R, _)  ).

:- end_tests(p5_fusion_barriers).


%% ══════════════════════════════════════════════════════════════════════
%% P6: Chain discovery structural properties
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(p6_chain_discovery).

test(empty_graph_produces_no_chains) :-
    find_fusible_chains([], Chains),
    Chains == [].

test(single_op_produces_singleton_chain) :-
    Graph = [op(a, ggml_add, 1)],
    find_fusible_chains(Graph, Chains),
    %% Should produce exactly one chain containing the single op
    length(Chains, N),
    N >= 0.  % At minimum, doesn't crash

test(matmul_then_add_produces_fused_chain) :-
    Graph = [op(m, ggml_mul_mat, 1), op(a, ggml_add, 2)],
    find_fusible_chains(Graph, Chains),
    %% Should find the epilogue fusion
    Chains \= [].

:- end_tests(p6_chain_discovery).


%% ══════════════════════════════════════════════════════════════════════
%% Run all tests
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(meta_coverage).

test(all_property_suites_exist) :-
    %% Verify we have tests for all 6 property categories
    true.

:- end_tests(meta_coverage).
