%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_c_preprocess_piece2.pl — Extended coverage for piece 2 (cpp shell-out).
%%
%% Exercises preprocess_file_segment/4,/5 on the full architecture corpus
%% and verifies edge cases specific to the cpp shell-out path.
%%
%% Author: medayek (Collective SME, Verification Methodology)
%% Date: 2026-05-17
%% Per metayen's interstage boundary consultation for piece 2.

:- use_module('lib/c_preprocess').
:- initialization(main, main).

%% ═══════════════════════════════════════════════════════════════════════
%% Configuration
%% ═══════════════════════════════════════════════════════════════════════

llama_cpp_src('./external/llama.cpp/src/models').

include_paths([
    "./external/llama.cpp/src",
    "./external/llama.cpp/include",
    "./external/llama.cpp/ggml/include"
]).

%% Full architecture list (from the dispatch table in llama-model.cpp).
%% This is the same denominator the lifter migration uses.
%% all_archs/1 — DYNAMIC architecture enumeration from upstream reality.
%%
%% Per Heath 2026-05-18 ~23:15 UTC: "(C) is what brings us to high-tech
%% work level." The substrate-honest discipline is to derive the test
%% surface from upstream filesystem state, NOT from a hand-curated
%% static list. The substrate stays automatically synchronized with
%% what llama.cpp actually ships — no manual maintenance, no staleness
%% drift.
%%
%% This predicate enumerates every .cpp file in
%% external/llama.cpp/src/models/, extracts the arch name from the
%% filename (basename without extension), and returns the sorted list.
%%
%% Substrate-design observations:
%%
%% 1. NOT every .cpp file is an arch (some are helpers like
%%    "delta-net-base.cpp" or "models.cpp" which define base classes
%%    rather than llama_model_<arch>::load_arch_tensors). The
%%    test_single_arch predicate uses pre-check_has_load_arch_tensors/1
%%    to classify such files as SKIP, not FAIL.
%%
%% 2. The substrate's broader arch_summary code already operates over
%%    128 .cpp files; this test now matches that surface.
%%
%% 3. When upstream adds new model architectures, this test
%%    AUTOMATICALLY covers them. No need to update a static list.
%%
%% This is the substantive "high-tech work level" Heath named: substrate
%% tests that track upstream reality continuously, not snapshots that
%% drift stale.
all_archs(Archs) :-
    llama_cpp_src(SrcDir),
    format(atom(Pattern), "~w/*.cpp", [SrcDir]),
    expand_file_name(Pattern, Files),
    maplist(arch_name_from_path, Files, Archs0),
    sort(Archs0, Archs).

%% arch_name_from_path(+CppPath, -ArchName) — basename minus ".cpp"
arch_name_from_path(CppPath, ArchName) :-
    file_base_name(CppPath, Base),
    atom_concat(ArchName, '.cpp', Base).

%% has_load_arch_tensors(+SourcePath) — true if source contains
%% the load_arch_tensors function definition. Used by sweep to
%% distinguish arch .cpp files (have load_arch_tensors) from helper
%% .cpp files (do not — e.g., delta-net-base.cpp defines llm_build_*
%% base classes, not llama_model_<arch>::load_arch_tensors).
has_load_arch_tensors(SourcePath) :-
    read_file_to_string(SourcePath, Source, []),
    sub_string(Source, _, _, _, "::load_arch_tensors(").


%% find_function_range(+SourcePath, +FuncName, -range(Start, End))
%%
%% Locate the source-line range of a function definition matching
%% `void llama_model_X::FuncName(...)` in the given source file.
%% Scans forward for the matching closing brace at column 0.
%%
%% Used by the sweep + line_map tests to locate load_arch_tensors.
find_function_range(SourcePath, FuncName, range(Start, End)) :-
    read_file_to_string(SourcePath, Source, []),
    split_string(Source, "\n", "", AllLines),
    length(AllLines, NLines),
    find_function_start(AllLines, FuncName, 1, Start),
    find_function_end(AllLines, Start, NLines, End).

find_function_start([Line | Rest], FuncName, N, Start) :-
    format(atom(Pattern), "::~w(", [FuncName]),
    ( sub_string(Line, _, _, _, Pattern),
      sub_string(Line, 0, _, _, "void ")
    -> Start = N
    ;  N1 is N + 1,
       find_function_start(Rest, FuncName, N1, Start)
    ).

find_function_end(Lines, Start, _NLines, End) :-
    StartIdx is Start - 1,
    length(Skip, StartIdx),
    append(Skip, Body, Lines),
    scan_for_close(Body, Start, End).

scan_for_close([Line | Rest], CurLine, End) :-
    ( Line == "}"
    -> End = CurLine
    ;  Next is CurLine + 1,
       scan_for_close(Rest, Next, End)
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Test 1: Full 95-arch sweep (preprocess_file_segment on each)
%% ═══════════════════════════════════════════════════════════════════════

test_full_arch_sweep :-
    all_archs(Archs),
    include_paths(Paths),
    llama_cpp_src(SrcDir),
    findall(Result,
        ( member(Arch, Archs),
          test_single_arch(Arch, SrcDir, Paths, Result)
        ),
        Results),
    aggregate_count(Results, ok, NOk),
    aggregate_count(Results, fail, NFail),
    aggregate_count(Results, skip, NSkip),
    aggregate_count(Results, warn, NWarn),
    format("~n  Sweep totals: OK=~w  FAIL=~w  WARN=~w  SKIP=~w~n",
           [NOk, NFail, NWarn, NSkip]),
    %% The sweep is informative — tolerate some FAIL/SKIP without
    %% failing the test overall. (Real corpus has missing files;
    %% some archs may not have load_arch_tensors at all.) But if
    %% MOST archs fail, that signals a regression.
    Total is NOk + NFail + NWarn + NSkip,
    Total > 0,
    OkRate is NOk / Total,
    ( OkRate >= 0.5
    -> format("  [SWEEP PASS] ~w/~w archs OK (~2f)~n", [NOk, Total, OkRate])
    ;  format("  [SWEEP FAIL] only ~w/~w archs OK (~2f) — regression?~n",
              [NOk, Total, OkRate]),
       fail
    ).

aggregate_count(List, Key, N) :-
    findall(1, member(Key, List), Ones),
    length(Ones, N).

test_single_arch(Arch, SrcDir, Paths, Result) :-
    format(atom(SourcePath), "~w/~w.cpp", [SrcDir, Arch]),
    ( exists_file(SourcePath)
    -> ( has_load_arch_tensors(SourcePath)
       -> catch(
              ( find_function_range(SourcePath, "load_arch_tensors", Range),
                preprocess_file_segment(SourcePath, Paths, Range, Text, _Map),
                string_length(Text, Len),
                ( Len > 0
                -> format("  [OK] ~w: ~w chars~n", [Arch, Len]), Result = ok
                ;  format("  [WARN] ~w: empty output~n", [Arch]), Result = warn
                )
              ),
              E,
              ( format("  [FAIL] ~w: ~q~n", [Arch, E]), Result = fail )
          )
       ;  %% File exists but doesn't define load_arch_tensors.
          %% This is either:
          %%   - A base class file (delta-net-base, mamba-base, *-base)
          %%   - An arch where load_arch_tensors lives in a sibling file
          %%     or has been consolidated (ernie4-5-moe, hunyuan-dense,
          %%     nemotron-h-moe, etc. define build_arch_graph only)
          %% In either case, c_preprocess has nothing to test on this
          %% specific function range — the substrate's preprocessing
          %% capability isn't relevant to this file. SKIP is the
          %% substrate-honest classification.
          format("  [SKIP] ~w: no load_arch_tensors in file~n", [Arch]),
          Result = skip
       )
    ;  format("  [SKIP] ~w: no source file~n", [Arch]), Result = skip
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Test 2: line_map correctness property
%% ═══════════════════════════════════════════════════════════════════════

%% For a known architecture (bert), verify that each line_map entry's
%% source_line corresponds to actual content from that line in the
%% source file.
test_line_map_correctness :-
    llama_cpp_src(SrcDir),
    include_paths(Paths),
    format(atom(SourcePath), "~w/bert.cpp", [SrcDir]),
    ( exists_file(SourcePath)
    -> find_function_range(SourcePath, "load_arch_tensors", Range),
       preprocess_file_segment(SourcePath, Paths, Range, _Text, Map),
       %% Read source lines for comparison
       read_file_to_string(SourcePath, Source, []),
       split_string(Source, "\n", "", SourceLines),
       length(SourceLines, NLines),
       %% The actual term shape is line_map(OutPos, SrcFile, SrcLine).
       %% Verify every source_line is within bounds of the source file.
       %% Path-suffix match for the file name (cpp may emit any prefix).
       forall(member(line_map(_OutPos, SrcFile, SrcLine), Map),
           ( ( string_concat(_, "/bert.cpp", SrcFile) ; SrcFile == "bert.cpp" )
           -> ( SrcLine > 0, SrcLine =< NLines
              -> true
              ;  format("  [WARN] line ~w out of bounds (max ~w)~n",
                        [SrcLine, NLines]),
                 fail
              )
           ;  true  % Non-target file entries are OK
           )),
       format("  [OK] line_map entries all within bounds~n")
    ;  format("  [SKIP] bert.cpp not found~n")
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Test 3: Edge cases for piece 2
%% ═══════════════════════════════════════════════════════════════════════

%% 3a: Very small function range (1-3 lines).
%% Some archs have tiny load_arch_tensors (just calling parent's).
test_small_function_range :-
    %% Use filter_cpp_output directly with a minimal input
    Lines = [
        "# 10 \"tiny_arch.cpp\"",
        "void f() { base::load_arch_tensors(); }"
    ],
    c_preprocess:filter_cpp_output(Lines, "tiny_arch.cpp", range(10, 10), Emitted),
    length(Emitted, N),
    format("  Small range: ~w lines emitted~n", [N]),
    N =:= 1.

%% 3b: cpp failure (missing -I flag) should throw cpp_failed.
%% This tests the error path when the shell-out to cpp returns non-zero.
test_cpp_failure_handling :-
    %% Use a non-existent file path; cpp returns non-zero and we throw.
    %% The actual throw shape is error(cpp_failed(ExitCode, Stderr), context(...)).
    catch(
        ( preprocess_file_segment("/nonexistent/path.cpp", [], range(1, 10), _, _),
          format("  [FAIL] expected cpp_failed exception but call succeeded~n"),
          fail
        ),
        error(cpp_failed(ExitCode, _Stderr), _),
        ( format("  [OK] cpp_failed thrown on bad path (exit code ~w)~n", [ExitCode]),
          ExitCode > 0
        )
    ).

%% 3c: Range where content is inside #if 0 block.
%% After preprocessing, the content should be absent (cpp strips it).
test_ifdef_zero_content :-
    Lines = [
        "# 1 \"file.cpp\"",
        "int before;",
        "# 5 \"file.cpp\"",
        "int after_ifdef;"
    ],
    %% Lines 2-4 were inside #if 0 and don't appear in cpp output.
    %% Only lines 1 and 5 have content. Range 1-5 should emit what's available.
    c_preprocess:filter_cpp_output(Lines, "file.cpp", range(1, 5), Emitted),
    length(Emitted, N),
    format("  #if 0 gap: ~w lines emitted (expected 2)~n", [N]),
    N =:= 2.


%% ═══════════════════════════════════════════════════════════════════════
%% Test 4: build_line_map unit test (off-by-one detection)
%% ═══════════════════════════════════════════════════════════════════════

%% Verify that the line counter in filter_cpp_output produces correct
%% source_line values in the emitted entries.
test_line_counter_accuracy :-
    Lines = [
        "# 1 \"test.cpp\"",
        "line at source 1",
        "line at source 2",
        "line at source 3",
        "# 10 \"test.cpp\"",
        "line at source 10",
        "line at source 11"
    ],
    c_preprocess:filter_cpp_output(Lines, "test.cpp", range(1, 20), Emitted),
    %% Verify each emitted line has the correct source_line tag
    Emitted = [
        emitted(_, 1, "line at source 1"),
        emitted(_, 2, "line at source 2"),
        emitted(_, 3, "line at source 3"),
        emitted(_, 10, "line at source 10"),
        emitted(_, 11, "line at source 11")
    ],
    format("  [OK] line counter matches source lines~n").

%% Off-by-one edge: directive resets to line N, next content is line N.
test_line_counter_after_directive :-
    Lines = [
        "# 5 \"test.cpp\"",
        "this is line 5",
        "this is line 6"
    ],
    c_preprocess:filter_cpp_output(Lines, "test.cpp", range(5, 6), Emitted),
    Emitted = [
        emitted(_, 5, "this is line 5"),
        emitted(_, 6, "this is line 6")
    ],
    format("  [OK] line counter correct after mid-file directive~n").


%% ═══════════════════════════════════════════════════════════════════════
%% Main
%% ═══════════════════════════════════════════════════════════════════════

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

main :-
    retractall(test_pass(_)),
    retractall(test_fail(_, _)),
    run_test("Edge: small function range", test_small_function_range),
    run_test("Edge: #if 0 content stripped", test_ifdef_zero_content),
    run_test("Unit: line counter accuracy", test_line_counter_accuracy),
    run_test("Unit: line counter after directive", test_line_counter_after_directive),
    %% Tests requiring filesystem access (may skip if files not present):
    run_test("Property: line_map correctness on bert", test_line_map_correctness),
    run_test("Error: cpp failure handling", test_cpp_failure_handling),
    %% Full sweep (slow, ~1-2 min):
    run_test("Sweep: all architectures", test_full_arch_sweep),
    format("~n=== Summary ===~n"),
    aggregate_all(count, test_pass(_), NPass),
    aggregate_all(count, test_fail(_, _), NFail),
    format("  Passed: ~w~n", [NPass]),
    format("  Failed: ~w~n", [NFail]),
    ( NFail =:= 0
    -> format("  Result: ALL PASS~n"), halt(0)
    ;  format("  Result: FAILURES~n"),
       forall(test_fail(N, R), format("    - ~w: ~q~n", [N, R])),
       halt(1)
    ).
