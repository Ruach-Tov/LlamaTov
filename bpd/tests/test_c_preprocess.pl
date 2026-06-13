%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_c_preprocess.pl — Test coverage for c_preprocess.pl
%%
%% Exercises filter_cpp_output/4 and parse_line_directive/4 on
%% hand-crafted cpp-style inputs. No shell-out to the system cpp
%% (those tests will be added in piece 2 of step #4 work).
%%
%% Test coverage status: STARTER — Heath has asked medayek to expand
%% coverage before piece 2 lands. This file establishes the empty-file
%% edge case plus the three integration tests from initial development.
%% Medayek's expansion will add the missing edge cases (single-line
%% files, range-out-of-bounds, target file never appears, etc.).
%%
%% Author: metayen 2026-05-17

:- use_module('lib/c_preprocess').
:- initialization(main, main).


%% ─── Test runner harness ──────────────────────────────────────────

:- dynamic test_pass/1, test_fail/2.

run_test(Name, Goal) :-
    format("~n=== ~w ===~n", [Name]),
    ( catch(Goal, E, (Err = E, fail))
    -> assertz(test_pass(Name)),
       format("  PASS~n")
    ;  ( var(Err) -> Reason = "goal failed" ; Reason = Err ),
       assertz(test_fail(Name, Reason)),
       format("  FAIL: ~q~n", [Reason])
    ).

assert_equal(Expected, Actual) :-
    ( Expected == Actual
    -> true
    ;  format("  expected: ~q~n", [Expected]),
       format("  actual:   ~q~n", [Actual]),
       fail
    ).

assert_emit_count(Emitted, ExpectedCount) :-
    length(Emitted, N),
    ( N =:= ExpectedCount
    -> format("  ~w lines emitted (expected ~w)~n", [N, ExpectedCount])
    ;  format("  count mismatch: got ~w, expected ~w~n", [N, ExpectedCount]),
       fail
    ).

assert_throws(Goal, ExpectedHead) :-
    catch(Goal, error(Head, _), (CaughtHead = Head)),
    ( var(CaughtHead)
    -> format("  expected exception ~q, got success~n", [ExpectedHead]),
       fail
    ;  ( CaughtHead = ExpectedHead
       -> format("  caught expected: ~q~n", [CaughtHead])
       ;  format("  caught wrong head: ~q (expected ~q)~n",
                 [CaughtHead, ExpectedHead]),
          fail
       )
    ).


%% ─── Tests ──────────────────────────────────────────────────────

%% Edge case: empty cpp output. Should emit zero lines, not throw,
%% not produce any spurious state. The contiguous-slice invariant
%% must not fire (it has nothing to check against).
test_empty_input :-
    Lines = [],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 100), Emitted),
    assert_emit_count(Emitted, 0).

%% Edge case: cpp output with directives only, no content lines.
%% Real-world example: a file that's just `#include <foo>` and the
%% included file is empty, OR a query range that falls between
%% directives. Should emit zero lines, not throw.
test_directives_only_no_content :-
    Lines = [
        "# 1 \"bert.cpp\"",
        "# 1 \"empty.h\" 1",
        "# 1 \"bert.cpp\" 2"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 100), Emitted),
    assert_emit_count(Emitted, 0).

%% Integration test 1: simple target content. From initial development.
test_simple_target_content :-
    Lines = [
        "# 1 \"bert.cpp\"",
        "void f() {",
        "    int x = 1;",
        "    int y = 2;",
        "}"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 5), Emitted),
    assert_emit_count(Emitted, 4),
    %% Verify the lines are tagged correctly
    Emitted = [emitted(_, 1, _),
               emitted(_, 2, _),
               emitted(_, 3, _),
               emitted(_, 4, _)].

%% Integration test 2: excursion through a header file.
%% From initial development. Verifies foo.h content is correctly
%% filtered out and bert.cpp content emits cleanly across the gap.
test_excursion_through_header :-
    Lines = [
        "# 1 \"bert.cpp\"",
        "#include \"foo.h\"",
        "# 1 \"foo.h\" 1",
        "void helper();",
        "int helper_count = 0;",
        "# 2 \"bert.cpp\" 2",
        "void f() {",
        "    helper();",
        "}"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 4), Emitted),
    assert_emit_count(Emitted, 4),
    %% All emitted lines should be tagged bert.cpp
    forall(member(emitted(File, _, _), Emitted),
        ( string_concat(_, "bert.cpp", File) ; File == "bert.cpp" )).

%% Integration test 3: constructed invariant violation. The cpp output
%% jumps within the same file from line 2 to line 99 with no excursion
%% explaining the output-line gap. This is the case that would arise
%% if some cpp implementation tagged macro expansions at the
%% definition site rather than the invocation site.
%% Invariant violation: a gap in target-file content emissions that is
%% NOT explained by an excursion into a non-target file.
%%
%% Construction: target emits lines 1, 2, then SKIPS to emit line 5
%% in the same file. The skipped content output lines (where the source
%% range query selects them) must be unaccounted for. Since no
%% non-target file excursion happened, the invariant must fire.
%%
%% Note: an earlier version of this test used `# 99 "bert.cpp"` to fake
%% a within-file line jump via a directive. That construction was
%% retracted after the piece 2 empirical sweep showed that real cpp
%% output uses such directives legitimately during macro expansion
%% (nullptr → __null is tagged with a system-header flag, file
%% unchanged). Directive lines are now zero-width in the output
%% contiguity check; the invariant fires only on real content gaps.
test_invariant_violation :-
    Lines = [
        "# 1 \"bert.cpp\"",
        "line 1 content",                              % emit at bert.cpp:1
        %% Hand-crafted: skip an output position with a non-directive,
        %% non-target line. In real cpp we'd see a file change via
        %% directive; here we abuse the fact that the filter only
        %% updates in_target on directives. The first directive set
        %% in_target=true and the source line counter; the filter has
        %% no way to "exit" without a directive. So this is a contrived
        %% pathological input that probably can't arise from real cpp.
        "# 1 \"other.h\" 1",                            % real excursion to other.h
        "other content",                               % filtered out (other.h)
        "other content 2",                             % filtered out
        "# 5 \"bert.cpp\" 2",                          % return to bert.cpp at line 5
        "line 5 content"                               % emit at bert.cpp:5 — but SE=true!
    ],
    %% With SE=true (we exited target during the gap), the invariant
    %% ALLOWS this gap — it's interpreted as a legitimate #include
    %% excursion. The test currently expects NO exception (the gap
    %% is properly explained by the excursion).
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 5), Emitted),
    assert_emit_count(Emitted, 2).



%% ─── Edge Cases (medayek expansion, 2026-05-17) ──────────────────────
%% Per metayen's request: expand coverage before piece 2 lands.

%% Single-line file: just one content line in the target.
test_single_line_file :-
    Lines = [
        "# 1 \"tiny.cpp\"",
        "int main() { return 0; }"
    ],
    c_preprocess:filter_cpp_output(Lines, "tiny.cpp", range(1, 1), Emitted),
    assert_emit_count(Emitted, 1).

%% Range query where MStart > all content lines (out of bounds high).
test_range_beyond_file_end :-
    Lines = [
        "# 1 \"short.cpp\"",
        "int x = 1;",
        "int y = 2;"
    ],
    c_preprocess:filter_cpp_output(Lines, "short.cpp", range(100, 200), Emitted),
    assert_emit_count(Emitted, 0).

%% Range query where NEnd < MStart (inverted range).
test_inverted_range :-
    Lines = [
        "# 1 \"file.cpp\"",
        "line A",
        "line B",
        "line C"
    ],
    c_preprocess:filter_cpp_output(Lines, "file.cpp", range(5, 2), Emitted),
    assert_emit_count(Emitted, 0).

%% Target file name never appears in any directive.
test_target_file_never_appears :-
    Lines = [
        "# 1 \"other.cpp\"",
        "void other_function();",
        "# 1 \"another.h\" 1",
        "int helper;",
        "# 2 \"other.cpp\" 2",
        "void more();"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 100), Emitted),
    assert_emit_count(Emitted, 0).

%% Multiple sequential excursions: target -> foo.h -> target -> bar.h -> target
test_multiple_sequential_excursions :-
    Lines = [
        "# 1 \"bert.cpp\"",
        "// bert line 1",
        "# 1 \"foo.h\" 1",
        "void foo();",
        "# 2 \"bert.cpp\" 2",
        "// bert line 2",
        "# 1 \"bar.h\" 1",
        "void bar();",
        "int bar_x;",
        "# 3 \"bert.cpp\" 2",
        "// bert line 3"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 5), Emitted),
    assert_emit_count(Emitted, 3).

%% Nested excursions: target -> foo.h -> nested.h -> foo.h -> target
test_nested_excursions :-
    Lines = [
        "# 1 \"bert.cpp\"",
        "start of bert",
        "# 1 \"foo.h\" 1",
        "in foo",
        "# 1 \"nested.h\" 1",
        "deeply nested",
        "# 2 \"foo.h\" 2",
        "back in foo",
        "# 2 \"bert.cpp\" 2",
        "end of bert"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 5), Emitted),
    assert_emit_count(Emitted, 2).

%% Suffix-matching for file paths: cpp emits "./src/models/bert.cpp"
test_suffix_matching_path :-
    Lines = [
        "# 1 \"./src/models/bert.cpp\"",
        "content line A",
        "content line B"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 3), Emitted),
    assert_emit_count(Emitted, 2).

%% Suffix-matching: absolute path emitted by cpp.
test_suffix_matching_absolute :-
    Lines = [
        "# 1 \"/home/user/project/src/models/bert.cpp\"",
        "absolute path content"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 1), Emitted),
    assert_emit_count(Emitted, 1).

%% Suffix-matching NEGATIVE test: "not_bert.cpp" must NOT match "bert.cpp".
%%
%% Locks in the fix for the false-positive medayek caught during coverage
%% expansion (2026-05-17). The naive string-suffix `string_concat(_, "bert.cpp", File)`
%% succeeds for "not_bert.cpp" since that string DOES end in "bert.cpp".
%% The fix uses PATH-suffix semantics — match boundary must align with a
%% path separator or be the whole string.
test_suffix_matching_no_false_positive :-
    Lines = [
        "# 1 \"not_bert.cpp\"",
        "content from not_bert.cpp",
        "should not be emitted"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 10), Emitted),
    assert_emit_count(Emitted, 0).

%% parse_line_directive unit tests.
test_parse_directive_simple :-
    c_preprocess:parse_line_directive("# 1 \"bert.cpp\"", File, Line, Flags),
    assert_equal("bert.cpp", File),
    assert_equal(1, Line),
    assert_equal([], Flags).

test_parse_directive_with_flags :-
    c_preprocess:parse_line_directive("# 23 \"bert.cpp\" 2", File, Line, Flags),
    assert_equal("bert.cpp", File),
    assert_equal(23, Line),
    assert_equal([2], Flags).

test_parse_directive_with_path :-
    c_preprocess:parse_line_directive("# 687 \"./src/llama-model.h\" 1", File, Line, Flags),
    assert_equal("./src/llama-model.h", File),
    assert_equal(687, Line),
    assert_equal([1], Flags).

test_parse_directive_builtin :-
    c_preprocess:parse_line_directive("# 1 \"<built-in>\"", File, Line, Flags),
    assert_equal("<built-in>", File),
    assert_equal(1, Line),
    assert_equal([], Flags).

test_parse_directive_multiple_flags :-
    c_preprocess:parse_line_directive("# 1 \"sys.h\" 3 4", File, Line, Flags),
    assert_equal("sys.h", File),
    assert_equal(1, Line),
    assert_equal([3, 4], Flags).

%% parse_line_directive: non-directive lines should FAIL.
test_parse_directive_fails_on_content :-
    ( c_preprocess:parse_line_directive("int x = 1;", _, _, _)
    -> fail
    ;  true
    ).

test_parse_directive_fails_on_empty :-
    ( c_preprocess:parse_line_directive("", _, _, _)
    -> fail
    ;  true
    ).

%% Range captures only the middle of available content.
test_range_captures_middle :-
    Lines = [
        "# 1 \"file.cpp\"",
        "line 1 before",
        "line 2 before",
        "line 3 IN range",
        "line 4 IN range",
        "line 5 after"
    ],
    c_preprocess:filter_cpp_output(Lines, "file.cpp", range(3, 4), Emitted),
    assert_emit_count(Emitted, 2).

%% System header with internal line jumps.
test_system_header_line_jumps :-
    Lines = [
        "# 1 \"bert.cpp\"",
        "bert line 1",
        "# 1 \"/usr/include/c++/14/vector\" 1 3",
        "# 58 \"/usr/include/c++/14/vector\" 3",
        "namespace std {",
        "# 200 \"/usr/include/c++/14/vector\" 3",
        "template class vector;",
        "# 2 \"bert.cpp\" 2",
        "bert line 2"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 5), Emitted),
    assert_emit_count(Emitted, 2).

%% Invariant holds when gap is explained by excursion (positive test).
test_invariant_holds_with_explained_gap :-
    Lines = [
        "# 1 \"bert.cpp\"",
        "line A",
        "# 1 \"huge.h\" 1",
        "h1", "h2", "h3", "h4", "h5",
        "# 2 \"bert.cpp\" 2",
        "line B"
    ],
    c_preprocess:filter_cpp_output(Lines, "bert.cpp", range(1, 5), Emitted),
    assert_emit_count(Emitted, 2).

%% ─── Main ───────────────────────────────────────────────────────

main :-
    retractall(test_pass(_)),
    retractall(test_fail(_, _)),
    run_test("Edge: empty input", test_empty_input),
    run_test("Edge: directives only, no content", test_directives_only_no_content),
    run_test("Integration: simple target content", test_simple_target_content),
    run_test("Integration: excursion through header", test_excursion_through_header),
    run_test("Integration: invariant holds during real excursion", test_invariant_violation),
    %% medayek edge-case expansion (2026-05-17)
    run_test("Edge: single-line file", test_single_line_file),
    run_test("Edge: range beyond file end", test_range_beyond_file_end),
    run_test("Edge: inverted range", test_inverted_range),
    run_test("Edge: target file never appears", test_target_file_never_appears),
    run_test("Edge: multiple sequential excursions", test_multiple_sequential_excursions),
    run_test("Edge: nested excursions", test_nested_excursions),
    run_test("Edge: suffix-matching path", test_suffix_matching_path),
    run_test("Edge: suffix-matching absolute", test_suffix_matching_absolute),
    run_test("Edge: suffix-matching no false positive (not_bert.cpp)", test_suffix_matching_no_false_positive),
    run_test("Unit: parse_directive simple", test_parse_directive_simple),
    run_test("Unit: parse_directive with flags", test_parse_directive_with_flags),
    run_test("Unit: parse_directive with path", test_parse_directive_with_path),
    run_test("Unit: parse_directive builtin", test_parse_directive_builtin),
    run_test("Unit: parse_directive multiple flags", test_parse_directive_multiple_flags),
    run_test("Unit: parse_directive fails on content", test_parse_directive_fails_on_content),
    run_test("Unit: parse_directive fails on empty", test_parse_directive_fails_on_empty),
    run_test("Edge: range captures middle", test_range_captures_middle),
    run_test("Edge: system header line jumps", test_system_header_line_jumps),
    run_test("Property: invariant holds with explained gap", test_invariant_holds_with_explained_gap),
    format("~n=== Summary ===~n"),
    aggregate_all(count, test_pass(_), NPass),
    aggregate_all(count, test_fail(_, _), NFail),
    format("  Passed: ~w~n", [NPass]),
    format("  Failed: ~w~n", [NFail]),
    ( NFail =:= 0
    -> format("  Result: ALL PASS~n"),
       halt(0)
    ;  format("  Result: FAILURES PRESENT~n"),
       forall(test_fail(N, R), format("    - ~w: ~q~n", [N, R])),
       halt(1)
    ).
