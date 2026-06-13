%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernelbench_l1_structure.pl — verify L1 problem set structure invariants
%%
%% Per the verification discipline: every committed claim about L1 problems
%% should be checkable by running this test. If we evolve the problem set
%% or change the format, this test catches divergence.

:- use_module(kernelbench_l1_problems).

run_tests :-
    Tests = [
        test_exactly_100_problems,
        test_numbers_are_1_to_100_inclusive,
        test_each_problem_is_single_op,
        test_split_balance,
        test_split_convention,
        test_op_kinds_are_atoms,
        test_no_duplicate_problem_numbers,
        test_names_match_problem_numbers
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

test_exactly_100_problems :-
    findall(N, kernelbench_l1:kb_problem(N, _, _, _), Ns),
    length(Ns, 100).

test_numbers_are_1_to_100_inclusive :-
    findall(N, kernelbench_l1:kb_problem(N, _, _, _), Ns),
    sort(Ns, Sorted),
    numlist(1, 100, Sorted).

test_each_problem_is_single_op :-
    forall(kernelbench_l1:kb_problem(_, _, _, Ops),
           length(Ops, 1)).

test_split_balance :-
    findall(N, kernelbench_l1:kb_problem(N, train, _, _), Trn),
    findall(N, kernelbench_l1:kb_problem(N, test,  _, _), Tst),
    length(Trn, 50),
    length(Tst, 50).

test_split_convention :-
    %% Per the L2 convention: odd=TRAIN, even=TEST.
    forall(kernelbench_l1:kb_problem(N, train, _, _), 1 is N mod 2),
    forall(kernelbench_l1:kb_problem(N, test,  _, _), 0 is N mod 2).

test_op_kinds_are_atoms :-
    forall(
        ( kernelbench_l1:kb_problem(_, _, _, [op(_, OpKind, _)]) ),
        atom(OpKind)
    ).

test_no_duplicate_problem_numbers :-
    findall(N, kernelbench_l1:kb_problem(N, _, _, _), Ns),
    sort(Ns, Sorted),
    length(Ns, OriginalLen),
    length(Sorted, SortedLen),
    OriginalLen == SortedLen.

test_names_match_problem_numbers :-
    %% Each problem's name starts with its number followed by underscore.
    forall(
        ( kernelbench_l1:kb_problem(N, _, Name, _) ),
        ( atom_string(Name, NameStr),
          number_string(N, NStr),
          string_concat(NStr, "_", Prefix),
          sub_string(NameStr, 0, _, _, Prefix)
        )
    ).

:- initialization(run_tests, main).
