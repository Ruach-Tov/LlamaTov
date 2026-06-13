%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_spec_conformance.pl — Spec-conformance tests for BPD facts
%%
%% RULE: "If it maps to a spec, test against the spec."
%%
%% Every Prolog fact that encodes an external specification must have
%% a test that cross-references the spec source. When the spec evolves,
%% the test fails until the fact is updated.
%%
%% This is the worked example for the testing discipline that would have
%% caught Bug #21 (type_info I8/I16/I32/I64 wrong codes) before an
%% external report.
%%
%% Author: medayek (Collective SME, Verification Methodology)
%% Date: 2026-05-20

:- use_module('../must_close/boundary_dsl/gguf_emit_manifest').

%% ═══════════════════════════════════════════════════════════════════════
%% Test harness
%% ═══════════════════════════════════════════════════════════════════════

:- dynamic test_pass/1, test_fail/2.

run_test(Name, Goal) :-
    format("  ~w ... ", [Name]),
    ( catch(Goal, E, (format("FAIL (~q)~n", [E]), assertz(test_fail(Name, E)), fail))
    -> format("PASS~n"), assertz(test_pass(Name))
    ;  format("FAIL~n"), assertz(test_fail(Name, goal_failed))
    ).

assert_equal(Expected, Actual) :-
    ( Expected == Actual -> true
    ;  format("  Expected: ~q~n  Actual:   ~q~n", [Expected, Actual]), fail
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% GGML Type Enum — from ggml.h (the authoritative spec)
%%
%% Each entry: ggml_spec_type(Code, Name)
%% Source: external/llama.cpp/ggml/include/ggml.h lines 390-425
%% ═══════════════════════════════════════════════════════════════════════

ggml_spec_type(0,  'F32').
ggml_spec_type(1,  'F16').
ggml_spec_type(2,  'Q4_0').
ggml_spec_type(3,  'Q4_1').
%% 4, 5 removed (Q4_2, Q4_3)
ggml_spec_type(6,  'Q5_0').
ggml_spec_type(7,  'Q5_1').
ggml_spec_type(8,  'Q8_0').
ggml_spec_type(9,  'Q8_1').
ggml_spec_type(10, 'Q2_K').
ggml_spec_type(11, 'Q3_K').
ggml_spec_type(12, 'Q4_K').
ggml_spec_type(13, 'Q5_K').
ggml_spec_type(14, 'Q6_K').
ggml_spec_type(15, 'Q8_K').
ggml_spec_type(16, 'IQ2_XXS').
ggml_spec_type(17, 'IQ2_XS').
ggml_spec_type(18, 'IQ3_XXS').
ggml_spec_type(19, 'IQ1_S').
ggml_spec_type(20, 'IQ4_NL').
ggml_spec_type(21, 'IQ3_S').
ggml_spec_type(22, 'IQ2_S').
ggml_spec_type(23, 'IQ4_XS').
ggml_spec_type(24, 'I8').
ggml_spec_type(25, 'I16').
ggml_spec_type(26, 'I32').
ggml_spec_type(27, 'I64').
ggml_spec_type(28, 'F64').
ggml_spec_type(29, 'IQ1_M').
ggml_spec_type(30, 'BF16').
%% 31-33 removed (Q4_0_4_4, Q4_0_4_8, Q4_0_8_8)
ggml_spec_type(34, 'TQ1_0').
ggml_spec_type(35, 'TQ2_0').


%% ═══════════════════════════════════════════════════════════════════════
%% Spec conformance tests
%% ═══════════════════════════════════════════════════════════════════════

%% Test 1: Every spec type code maps to the correct name in type_info/4
test_spec_code_to_name :-
    forall(
        ggml_spec_type(Code, SpecName),
        ( type_info(Code, InfoName, _, _)
        -> ( InfoName == SpecName
           -> true
           ;  format("  MISMATCH: code ~w spec=~w info=~w~n",
                     [Code, SpecName, InfoName]),
              fail
           )
        ;  format("  MISSING: code ~w (~w) not in type_info~n",
                  [Code, SpecName]),
           fail
        )
    ).

%% Test 2: Every type_info/4 entry (except catch-all) has a spec entry
test_no_phantom_types :-
    forall(
        ( type_info(Code, Name, _, _),
          Name \== 'unknown',
          integer(Code)
        ),
        ( ggml_spec_type(Code, _)
        -> true
        ;  format("  PHANTOM: type_info(~w, ~w, ...) has no spec entry~n",
                  [Code, Name]),
           fail
        )
    ).

%% Test 3: Critical types have correct byte sizes
test_byte_sizes :-
    type_info(0,  _, _, F32Size),  assert_equal(4, F32Size),
    type_info(1,  _, _, F16Size),  assert_equal(2, F16Size),
    type_info(24, _, _, I8Size),   assert_equal(1, I8Size),
    type_info(25, _, _, I16Size),  assert_equal(2, I16Size),
    type_info(26, _, _, I32Size),  assert_equal(4, I32Size),
    type_info(27, _, _, I64Size),  assert_equal(8, I64Size),
    type_info(28, _, _, F64Size),  assert_equal(8, F64Size),
    type_info(30, _, _, BF16Size), assert_equal(2, BF16Size).

%% Test 4: The I8-I64 range (the original Bug #21) is correct
test_integer_type_codes :-
    type_info(24, 'I8',  _, _),
    type_info(25, 'I16', _, _),
    type_info(26, 'I32', _, _),
    type_info(27, 'I64', _, _).

%% Test 5: BF16 is at code 30 (not misidentified as I32)
test_bf16_not_misidentified :-
    type_info(30, Name, _, _),
    assert_equal('BF16', Name).

%% Test 6: No gaps in the spec that we're missing
test_spec_coverage :-
    aggregate_all(count, ggml_spec_type(_, _), SpecCount),
    aggregate_all(count,
        ( type_info(Code, Name, _, _),
          Name \== 'unknown',
          integer(Code)
        ),
        InfoCount),
    format("  Spec types: ~w, type_info entries: ~w~n",
           [SpecCount, InfoCount]),
    ( InfoCount >= SpecCount
    -> true
    ;  format("  WARNING: type_info has fewer entries than spec~n"),
       fail
    ).

%% Test 7: Catch-all exists (defensive programming)
test_catch_all_exists :-
    ( type_info(99999, 'unknown', 'unknown', 0)
    -> true
    ;  format("  No catch-all clause for unknown type codes~n"),
       fail
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Main
%% ═══════════════════════════════════════════════════════════════════════

:- initialization((
    format("~n=== Spec Conformance Tests ===~n~n"),
    retractall(test_pass(_)),
    retractall(test_fail(_, _)),
    run_test("Spec code → type_info name mapping", test_spec_code_to_name),
    run_test("No phantom types in type_info", test_no_phantom_types),
    run_test("Critical byte sizes correct", test_byte_sizes),
    run_test("I8-I64 codes correct (Bug #21)", test_integer_type_codes),
    run_test("BF16 not misidentified", test_bf16_not_misidentified),
    run_test("Spec coverage complete", test_spec_coverage),
    run_test("Catch-all clause exists", test_catch_all_exists),
    format("~n=== Summary ===~n"),
    aggregate_all(count, test_pass(_), NPass),
    aggregate_all(count, test_fail(_, _), NFail),
    format("  Passed: ~w~n  Failed: ~w~n", [NPass, NFail]),
    ( NFail =:= 0
    -> format("  Result: ALL PASS~n"), halt(0)
    ;  format("  Result: FAILURES~n"),
       forall(test_fail(N, R), format("    - ~w: ~q~n", [N, R])),
       halt(1)
    )
), main).
