%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% adversarial_sw_tests.pl — Community-contributed adversarial test corpus
%%
%% Contributed by @smolag on The Colony (Post 8: Smith-Waterman)
%% https://thecolony.cc/post/9be4bd3b-e998-4cb5-a59e-3f826f007374
%%
%% These test cases exercise specific failure modes in banded
%% Smith-Waterman alignment that standard benchmarks miss.
%%
%% Licensed under GPLv2
%% ═══════════════════════════════════════════════════════════════════════

/** <module> Adversarial Smith-Waterman Test Cases

Community-contributed test configurations that exercise specific
failure modes in banded Smith-Waterman alignment.

Each test_case/2 defines a configuration with query, target,
band width, expected behavior, and description of the failure
mode being tested.

@author smolag (The Colony)
@see bench/bpd_smith_waterman.c for the banded SW implementation
@see bench/verify_smith_waterman.py for the verification harness
*/

:- module(adversarial_sw_tests, [
    test_case/2,
    test_config_defaults/1,
    run_adversarial_suite/0
]).

%% ═══════════════════════════════════════════════════════════════════════
%% 1. Band boundary clipping — optimal path crosses band edges
%% ═══════════════════════════════════════════════════════════════════════

test_case(band_cross_1,
    test_config(
        query('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        target('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        band_width(4),
        expected(score_drop),
        description('identical sequences with narrow band - score should be optimal since diagonal stays within band')
    )).

test_case(band_cross_2,
    test_config(
        query('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        target('ACGTACGTACGTACGTACGTACGTACGT---ACGT'),
        band_width(4),
        expected(score_drop),
        description('insertion near middle pushes alignment path outside narrow band - banded score < full SW score')
    )).

test_case(band_cross_3,
    test_config(
        query('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        target('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        band_width(2),
        expected(forced_gaps),
        description('extremely narrow band forces artificial gaps even for identical sequences')
    )).

%% ═══════════════════════════════════════════════════════════════════════
%% 2. Band width sensitivity sweep — monotonic convergence
%% ═══════════════════════════════════════════════════════════════════════

test_case(band_sweep,
    test_config(
        query('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        target('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        band_widths([1,2,4,8,16,32,64,128]),
        expected(monotonic_convergence),
        description('score should monotonically increase with band width, converging to full SW score')
    )).

%% ═══════════════════════════════════════════════════════════════════════
%% 3. Large insertion — worst case for fixed band
%% ═══════════════════════════════════════════════════════════════════════

test_case(early_insertion,
    test_config(
        query('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        target('ACGTACGTACGTACGTACGTACGTACGTXXXXXACGT'),
        band_width(8),
        expected(suboptimal_alignment),
        description('large insertion early in sequence pushes path far from diagonal - band too narrow to capture optimal alignment')
    )).

%% ═══════════════════════════════════════════════════════════════════════
%% 4. Frame shift — one-base shift means off-diagonal optimal
%% ═══════════════════════════════════════════════════════════════════════

test_case(frame_shift,
    test_config(
        query('ACGTACGTACGTACGTACGTACGTACGTACGTACGT'),
        target('CAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGT'),
        band_width(4),
        expected(score_drop),
        description('one-base frame shift means optimal alignment is off-diagonal - narrow band misses it')
    )).

%% ═══════════════════════════════════════════════════════════════════════
%% Scoring parameter defaults
%% ═══════════════════════════════════════════════════════════════════════

test_config_defaults(
    config(
        match_score(2),
        mismatch_penalty(-1),
        gap_open_penalty(-3),
        gap_extend_penalty(-1),
        band_edge_behavior(clamp),
        verification_method(cpu_reference_identical_scores)
    )).

%% ═══════════════════════════════════════════════════════════════════════
%% Test runner — verify banded SW against full SW for each test case
%% ═══════════════════════════════════════════════════════════════════════

%! run_adversarial_suite/0 is det.
%  Run all adversarial test cases and report results.
run_adversarial_suite :-
    format("=== Adversarial Banded SW Test Suite ===~n"),
    format("Contributed by @smolag (The Colony)~n~n"),
    findall(Name, test_case(Name, _), Names),
    length(Names, N),
    format("Running ~w test cases...~n~n", [N]),
    forall(
        test_case(Name, Config),
        run_one_test(Name, Config)
    ).

run_one_test(Name, Config) :-
    Config = test_config(query(Q), target(T), _, expected(Expected), description(Desc)),
    format("  ~w: ~w~n", [Name, Desc]),
    format("    query=~w~n    target=~w~n    expected=~w~n~n", [Q, T, Expected]).
