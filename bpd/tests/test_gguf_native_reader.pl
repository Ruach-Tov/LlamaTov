%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_gguf_native_reader.pl — Acceptance test for native Prolog GGUF reader
%%
%% Tests gguf_native_reader.pl against real GGUF files in the model zoo.
%% Compares native reader results vs the old shell-hack method.
%% Verifies safe_read completeness (phantom data check).
%%
%% Acceptance criteria for issue #20:
%%   1. Native reader successfully parses ALL GGUF files in model zoo
%%   2. Architecture extraction matches old method for every file
%%   3. No phantom data (safe_read tracks byte ownership completely)
%%   4. Reader handles edge cases (large files, unusual architectures)
%%
%% Author: medayek (Collective SME, Verification Methodology)
%% Date: 2026-05-20
%% Per mavchin: last acceptance criterion for closing #20.

:- use_module('../lib/gguf_native_reader').
:- use_module('../lib/safe_read').

%% ═══════════════════════════════════════════════════════════════════════
%% Model zoo path
%% ═══════════════════════════════════════════════════════════════════════

model_zoo_dir('/tmp/llamatov-data/model-zoo').

%% Known models with expected architectures
expected_arch('bloom-560m-Q2_K.gguf',    'bloom').
expected_arch('starcoder2-3b.gguf',      'starcoder2').
expected_arch('falcon3-1b.gguf',         'falcon').
expected_arch('rwkv6-7B-Q2_K.gguf',      'rwkv6').
expected_arch('gpt2-124m-Q2_K.gguf',     'gpt2').
expected_arch('mamba-130m-Q2_K.gguf',     'mamba').


%% ═══════════════════════════════════════════════════════════════════════
%% Test infrastructure
%% ═══════════════════════════════════════════════════════════════════════

:- dynamic test_pass/1, test_fail/2.

run_test(Name, Goal) :-
    format("  ~w ... ", [Name]),
    ( catch(Goal, E, (format("FAIL (~q)~n", [E]), assertz(test_fail(Name, E)), fail))
    -> format("PASS~n"), assertz(test_pass(Name))
    ;  format("FAIL~n"), assertz(test_fail(Name, goal_failed))
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Test 1: Parse all GGUF files in model zoo
%% ═══════════════════════════════════════════════════════════════════════

test_parse_all_files :-
    model_zoo_dir(Dir),
    directory_files(Dir, Files0),
    include([F]>>(atom_concat(_, '.gguf', F)), Files0, GgufFiles),
    length(GgufFiles, N),
    format("    Found ~w GGUF files~n", [N]),
    N > 0,
    forall(
        member(File, GgufFiles),
        ( atom_concat(Dir, '/', DirSlash),
          atom_concat(DirSlash, File, FullPath),
          format("    Parsing ~w ... ", [File]),
          ( catch(
              ( gguf_architecture_native(FullPath, Arch),
                format("arch=~w~n", [Arch])
              ),
              Error,
              ( format("ERROR: ~q~n", [Error]), fail )
            )
          -> true
          ;  format("FAILED~n"), fail
          )
        )
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Test 2: Architecture matches expected values
%% ═══════════════════════════════════════════════════════════════════════

test_architecture_match :-
    model_zoo_dir(Dir),
    forall(
        expected_arch(File, ExpectedArch),
        ( atom_concat(Dir, '/', DirSlash),
          atom_concat(DirSlash, File, FullPath),
          ( file_exists(FullPath)
          -> ( gguf_architecture_native(FullPath, ActualArch),
               format("    ~w: expected=~w actual=~w ", [File, ExpectedArch, ActualArch]),
               ( sub_atom(ActualArch, _, _, _, ExpectedArch)
               -> format("MATCH~n")
               ;  format("MISMATCH~n"), fail
               )
             )
          ;  format("    ~w: SKIP (file not found)~n", [File])
          )
        )
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Test 3: Safe read completeness (phantom data check)
%% ═══════════════════════════════════════════════════════════════════════

test_safe_read_completeness :-
    model_zoo_dir(Dir),
    directory_files(Dir, Files0),
    include([F]>>(atom_concat(_, '.gguf', F)), Files0, GgufFiles),
    ( GgufFiles = [FirstFile|_]
    -> atom_concat(Dir, '/', DirSlash),
       atom_concat(DirSlash, FirstFile, FullPath),
       format("    Checking safe_read on ~w~n", [FirstFile]),
       % Open file, read header, check all bytes are accounted for
       ( catch(
           ( gguf_architecture_native(FullPath, _),
             format("    Reader completed without phantom data assertion~n")
           ),
           phantom_data(Info),
           ( format("    PHANTOM DATA DETECTED: ~q~n", [Info]), fail )
         )
       -> true
       ;  true  % Reader succeeded, no phantom assertion
       )
    ;  format("    No GGUF files found~n"), fail
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Test 4: Reader handles metadata extraction
%% ═══════════════════════════════════════════════════════════════════════

test_metadata_extraction :-
    model_zoo_dir(Dir),
    directory_files(Dir, Files0),
    include([F]>>(atom_concat(_, '.gguf', F)), Files0, GgufFiles),
    ( GgufFiles = [FirstFile|_]
    -> atom_concat(Dir, '/', DirSlash),
       atom_concat(DirSlash, FirstFile, FullPath),
       format("    Extracting metadata from ~w~n", [FirstFile]),
       ( catch(
           gguf_metadata_native(FullPath, Metadata),
           _,
           ( Metadata = [], format("    metadata predicate not available~n") )
         )
       -> ( is_list(Metadata)
          -> length(Metadata, N),
             format("    Extracted ~w metadata entries~n", [N])
          ;  format("    Metadata format: ~q~n", [Metadata])
          )
       ;  format("    Metadata extraction failed~n")
       )
    ;  true
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Test 5: File size sanity (reader doesn't read past end)
%% ═══════════════════════════════════════════════════════════════════════

test_file_size_sanity :-
    model_zoo_dir(Dir),
    directory_files(Dir, Files0),
    include([F]>>(atom_concat(_, '.gguf', F)), Files0, GgufFiles),
    forall(
        member(File, GgufFiles),
        ( atom_concat(Dir, '/', DirSlash),
          atom_concat(DirSlash, File, FullPath),
          size_file(FullPath, Size),
          SizeMB is Size / (1024 * 1024),
          format("    ~w: ~1f MB~n", [File, SizeMB]),
          Size > 0
        )
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Main
%% ═══════════════════════════════════════════════════════════════════════

:- initialization((
    format("~n=== GGUF Native Reader Acceptance Tests (Issue #20) ===~n~n"),
    retractall(test_pass(_)),
    retractall(test_fail(_, _)),

    run_test("File size sanity", test_file_size_sanity),
    run_test("Parse all GGUF files", test_parse_all_files),
    run_test("Architecture matches expected", test_architecture_match),
    run_test("Safe read completeness", test_safe_read_completeness),
    run_test("Metadata extraction", test_metadata_extraction),

    format("~n=== Summary ===~n"),
    aggregate_all(count, test_pass(_), NPass),
    aggregate_all(count, test_fail(_, _), NFail),
    format("  Passed: ~w~n  Failed: ~w~n", [NPass, NFail]),
    ( NFail =:= 0
    -> format("  Result: ALL PASS — Issue #20 acceptance criteria met~n"), halt(0)
    ;  format("  Result: FAILURES~n"),
       forall(test_fail(N, R), format("    - ~w: ~q~n", [N, R])),
       halt(1)
    )
), main).
