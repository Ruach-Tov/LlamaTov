%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% c_preprocess.pl — C preprocessor layer for the BPD substrate.
%%
%% This module sits between raw .cpp source files and the c_ast.pl parser.
%% Its job: given a source file and a line range [M, N], produce the
%% post-preprocessor C++ text corresponding to those source lines, with
%% macros expanded and comments stripped, but without dragging in the
%% content of `#include`d headers.
%%
%% The strategy (Heath's framing, 2026-05-17):
%%
%%   "Show us how lines M..N of the file expand under cpp. An include
%%    file might expand to 100k lines, but if the include directives are
%%    all on lines < M, asking for the expansion of lines M..N
%%    automatically redacts all that crap that we don't need for now."
%%
%% We let the system cpp do its full work (including system header
%% expansion). The cpp output is interleaved with `# LINENO "FILE"`
%% directives that tell us the source-file-and-line origin of each
%% output line. We stream the output, track (source_file, source_line)
%% state from the directives, and emit only those lines whose tracked
%% origin matches our query.
%%
%% System header content (e.g., the 100k lines from <vector>) gets
%% tagged with its own `# N "/usr/include/c++/14/vector"` and filtered
%% out — we never even examine it past the line-counter increment.
%%
%% Macro expansions emit content tagged with the invocation site (per
%% GCC convention), so `LLAMA_LOAD_LOCALS;` at bert.cpp:23 expands into
%% output lines all tagged as bert.cpp content. The expansion passes
%% our filter naturally.
%%
%%
%% THE CONTIGUOUS-SLICE INVARIANT
%% ================================================================
%%
%% For a query "preprocess source lines M..N of file F", the cpp output
%% lines that pass our filter MUST form one contiguous slice (or a small
%% number of contiguous sequential slices, where the gaps correspond to
%% excursions into non-target files via #include).
%%
%% If we observe a gap between two emitted output lines where the
%% intervening output came from the TARGET FILE (not from an excursion
%% into a different file), the invariant has been violated. This would
%% indicate the cpp implementation tags macro expansions at the
%% DEFINITION SITE (e.g., llama-model.h:687) rather than the INVOCATION
%% SITE (bert.cpp:23). Our filter would then silently drop the expanded
%% content, producing parse-time errors with no diagnostic.
%%
%% By asserting the invariant and throwing on violation, we surface
%% cpp-semantics divergence as an early, loud diagnostic rather than as
%% downstream parse failure.
%%
%% Author: metayen 2026-05-17
%% Per Heath's plan to keep step #4 (lift_arch_tensors) on a clean
%% layering: preprocessor handles preprocessor things, parser handles
%% parser things.

:- module(c_preprocess, [
    filter_cpp_output/4,           % +CppOutputLines, +TargetFile, +Range, -EmittedLines
    parse_line_directive/4,        % +Line, -FileName, -LineNumber, -Flags
    preprocess_file_segment/4,     % +SourcePath, +IncludePaths, +Range, -ExpandedText
    preprocess_file_segment/5      % +SourcePath, +IncludePaths, +Range, -ExpandedText, -LineMap
]).

:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module(library(readutil)).


%% parse_line_directive(+Line, -FileName, -LineNumber, -Flags)
%%
%% Parses a `# LINENO "FILENAME" [FLAGS]` directive from cpp output.
%% Fails if Line is not such a directive.
%%
%% Examples (from GCC cpp -E output):
%%   "# 1 \"src/models/bert.cpp\""              → bert.cpp, 1, []
%%   "# 23 \"src/models/bert.cpp\" 2"           → bert.cpp, 23, [2]
%%   "# 1 \"<built-in>\""                       → <built-in>, 1, []
%%   "# 687 \"./src/llama-model.h\" 1"          → llama-model.h (path), 687, [1]
%%
%% GCC flag semantics:
%%   1 — start of a new file (e.g., entering an #include)
%%   2 — returning to a previous file (popping back out of an #include)
%%   3 — system header
%%   4 — should be wrapped in extern "C"
%% We capture them but don't currently act on them.

parse_line_directive(Line, FileName, LineNumber, Flags) :-
    string_codes(Line, [0'#, 0' | Rest]),     % must start with "# "
    string_codes(RestStr, Rest),
    split_string(RestStr, " ", "", Parts),
    Parts = [LineNumStr, QuotedFile | FlagStrs],
    number_string(LineNumber, LineNumStr),
    %% Strip the surrounding quotes from "filename"
    string_codes(QuotedFile, [0'", FileMid | RestCodes]),
    %% Find the closing quote
    append(FileCodes, [0'" | _], [FileMid | RestCodes]),
    string_codes(FileName, FileCodes),
    %% Parse the optional flags (each is a single digit 1..4)
    maplist([S, N]>>number_string(N, S), FlagStrs, Flags).


%% filter_cpp_output(+CppOutputLines, +TargetFile, +Range, -EmittedLines)
%%
%% Range = range(MStart, NEnd) — inclusive on both ends.
%% TargetFile is matched against the file name in each `# N "FILE"`
%% directive. Matching is done by suffix because cpp may emit
%% "./src/models/bert.cpp" or "src/models/bert.cpp" depending on the
%% invocation; we accept either.
%%
%% Maintains state:
%%   current_source_file  — most recent file from a `# N "FILE"` directive
%%   current_source_line  — incremented per non-directive line, reset by directives
%%   in_target            — boolean, true when current_source_file matches TargetFile
%%
%% INVARIANT (contiguous-slice): consecutive emitted lines must either
%%   (a) have output line numbers differing by exactly 1 (immediately
%%       consecutive in cpp output), OR
%%   (b) the gap between them must be entirely composed of lines whose
%%       current_source_file did NOT match TargetFile (i.e., excursions
%%       into other files).
%%
%% Violation → throws error(cpp_invariant_violated(...)).
%%
%% This skeleton implementation is intentionally minimal — it does
%% not yet invoke cpp; it consumes pre-existing cpp output as a list
%% of lines (suitable for unit testing on hand-crafted inputs).
%% The full preprocess_file_segment/4 that shells out to cpp will be
%% added in piece 3.

filter_cpp_output(Lines, TargetFile, range(MStart, NEnd), Emitted) :-
    filter_state(initial, TargetFile, MStart, NEnd, Lines, [], Emitted0),
    reverse(Emitted0, Emitted).

%% State carried through the fold:
%%   state(CurrentFile, CurrentLine, InTarget, LastEmittedOutputLine,
%%         SawTargetExitSinceLastEmit, OutputLineNum)
%%
%% Initially before any directive: file unknown, treat as 'unknown'.

filter_state(initial, TargetFile, MStart, NEnd, Lines, Acc0, AccOut) :-
    State0 = state('<unknown>', 0, false, none, false, 0),
    fold_lines(Lines, State0, TargetFile, MStart, NEnd, Acc0, AccOut).

fold_lines([], _State, _Tgt, _M, _N, Acc, Acc).
fold_lines([Line | Rest], State0, Tgt, M, N, Acc0, AccOut) :-
    State0 = state(_F0, _L0, _IT0, _LE0, _SE0, OutLine0),
    ( parse_line_directive(Line, NewFile, NewLine, _Flags)
    -> %% Directive: update file + line state. Don't emit. Don't advance
       %% OutLine — directives are zero-width metadata; they shouldn't
       %% disturb the contiguous-slice contiguity check on emitted lines.
       %% Empirical reason (qwen2 sweep, 2026-05-17): cpp emits a
       %% `# N "FILE" 3` (system-header flag) directive in the middle of
       %% expanding `nullptr` to `__null`. Without this fix, every
       %% nullptr in the target file would falsely fire the invariant.
       file_matches_target(NewFile, Tgt, NewInTarget),
       update_target_exit(State0, NewInTarget, SE1),
       State1 = state(NewFile, NewLine, NewInTarget, _, SE1, OutLine0),
       transfer_last_emit(State0, State1, State2),
       fold_lines(Rest, State2, Tgt, M, N, Acc0, AccOut)
    ; %% Content line: advance OutLine, maybe emit, increment current_source_line.
       OutLine1 is OutLine0 + 1,
       State0 = state(F, L, IT, LE, SE, _),
       L1 is L + 1,
       ( IT, L >= M, L =< N
       -> %% In range and in target — try to emit.
          check_invariant(LE, OutLine1, SE, F, L),
          NewAcc = [emitted(F, L, Line) | Acc0],
          State2 = state(F, L1, IT, OutLine1, false, OutLine1)
       ;  %% Not emitting.
          NewAcc = Acc0,
          State2 = state(F, L1, IT, LE, SE, OutLine1)
       ),
       fold_lines(Rest, State2, Tgt, M, N, NewAcc, AccOut)
    ).

%% Path-suffix match for file names. cpp may emit
%% "./src/foo.cpp", "src/foo.cpp", or "/abs/path/foo.cpp" — all should
%% match TargetFile == "foo.cpp" or TargetFile == "src/foo.cpp".
%%
%% IMPORTANT: this is a PATH-suffix match, not a string-suffix match.
%% "not_bert.cpp" does NOT match TargetFile == "bert.cpp" because the
%% match boundary must align with a path separator (or be the entire
%% string). Otherwise a file named "not_bert.cpp" would silently
%% emit content while the user queried "bert.cpp" — caught empirically
%% by medayek during coverage expansion (2026-05-17).
%%
%% Match cases:
%%   File == TargetFile                          → match (exact)
%%   File ends in "/" + TargetFile               → match (path suffix)
%% Anything else → no match.
file_matches_target(File, TargetFile, true) :-
    File == TargetFile, !.
file_matches_target(File, TargetFile, true) :-
    string_concat("/", TargetFile, SepSuffix),
    string_concat(_, SepSuffix, File), !.
file_matches_target(_, _, false).

%% When the file state changes, if we were in_target and now aren't,
%% mark "saw target exit." This is consumed by the invariant check.
update_target_exit(state(_, _, true, _, _, _), false, true) :- !.
update_target_exit(state(_, _, _, _, SE, _), _, SE).

%% Carry forward LastEmittedOutputLine through directive lines (which
%% don't emit and don't reset the LastEmit cursor).
transfer_last_emit(state(_, _, _, LE, _, _),
                   state(F, L, IT, _, SE, O),
                   state(F, L, IT, LE, SE, O)).

%% The contiguous-slice invariant check.
%%
%% LE = last emitted output line (or 'none' for first emission)
%% Cur = current output line
%% SE = saw_target_exit_since_last_emit
%% F, L = current source (for diagnostic)
%%
%% Allowed cases:
%%   1. LE == none                  — first emission, no prior to compare.
%%   2. Cur == LE + 1               — immediately consecutive, always fine.
%%   3. Cur > LE + 1 AND SE == true — gap explained by target-file excursion.
%%
%% Anything else: invariant violation.

check_invariant(none, _, _, _, _) :- !.
check_invariant(LE, Cur, _, _, _) :- Cur =:= LE + 1, !.
check_invariant(LE, Cur, true, _, _) :- Cur > LE + 1, !.
check_invariant(LE, Cur, false, F, L) :-
    throw(error(cpp_invariant_violated(contiguous_slice),
                context(gap_between_output_lines(LE, Cur),
                        in_target_file(F),
                        at_source_line(L),
                        diagnostic("cpp emitted lines from the target file with non-contiguous output line numbers, and we did NOT cross into a non-target file between emissions. This suggests the cpp implementation tags macro expansions at the definition site rather than the invocation site, or some other unhandled cpp-semantics divergence. Inspect cpp output around the gap."))
                )).


%% ─── Piece 2: shell out to system cpp ─────────────────────────────


%% preprocess_file_segment(+SourcePath, +IncludePaths, +Range, -ExpandedText)
%%
%% Convenience wrapper that discards the line_map.
preprocess_file_segment(SourcePath, IncludePaths, Range, ExpandedText) :-
    preprocess_file_segment(SourcePath, IncludePaths, Range, ExpandedText, _LineMap).


%% preprocess_file_segment(+SourcePath, +IncludePaths, +Range, -ExpandedText, -LineMap)
%%
%% Shell out to the system cpp, then filter for the target file +
%% line range. Returns:
%%
%%   ExpandedText — single string of the emitted lines joined with \n
%%   LineMap      — list of line_map(OutputPosition, SourceFile, SourceLine)
%%                  tuples for the emitted lines (1-indexed by position
%%                  in ExpandedText)
%%
%% Args:
%%   SourcePath   — path to the .cpp file to preprocess
%%   IncludePaths — list of strings, passed as -I flags to cpp
%%   Range        — range(MStart, NEnd), source-line range, 1-indexed inclusive
%%
%% TargetFile for filtering is derived from SourcePath via the
%% basename (file_base_name/2). The path-suffix match in
%% filter_cpp_output handles the case where cpp emits a different
%% prefix for the same file.
%%
%% Errors:
%%   - cpp_failed(ExitCode, Stderr) if cpp returns non-zero
%%   - cpp_invariant_violated(contiguous_slice) if the filter detects
%%     a discontinuity in target-file output (see check_invariant)

preprocess_file_segment(SourcePath, IncludePaths, range(MStart, NEnd), ExpandedText, LineMap) :-
    %% Derive target file basename for filtering
    file_base_name(SourcePath, TargetFileAtom),
    atom_string(TargetFileAtom, TargetFile),

    %% Build cpp argument list: -E (preprocess), -Idir for each include path,
    %% the source file last.
    findall(['-I', Dir], member(Dir, IncludePaths), IFlagsNested),
    flatten(IFlagsNested, IFlags),
    append(['-E' | IFlags], [SourcePath], CppArgs),

    %% Invoke cpp, capture stdout and stderr.
    run_cpp(CppArgs, CppStdout, _CppStderr, ExitCode),
    ( ExitCode =\= 0
    -> throw(error(cpp_failed(ExitCode, _CppStderr),
                   context(source(SourcePath),
                           include_paths(IncludePaths),
                           args(CppArgs))))
    ;  true
    ),

    %% Split into lines and filter for target file + range.
    split_string(CppStdout, "\n", "", Lines),
    filter_cpp_output(Lines, TargetFile, range(MStart, NEnd), Emitted),

    %% Assemble outputs.
    findall(Text, member(emitted(_, _, Text), Emitted), TextLines),
    atomics_to_string(TextLines, "\n", ExpandedText),

    %% Build line_map (1-indexed positions in ExpandedText).
    build_line_map(Emitted, 1, LineMap).

build_line_map([], _, []).
build_line_map([emitted(File, SrcLine, _Text) | Rest], Pos, [line_map(Pos, File, SrcLine) | RestMap]) :-
    Pos1 is Pos + 1,
    build_line_map(Rest, Pos1, RestMap).


%% run_cpp(+Args, -Stdout, -Stderr, -ExitCode)
%%
%% Invokes 'cpp' as a subprocess with the given arguments, capturing
%% stdout, stderr, and the exit code. Blocking — returns when cpp
%% finishes.
%%
%% Uses library(process) which is standard SWI-Prolog. The 'cpp' binary
%% must be on PATH; for the llama.cpp use case in our nix-shell
%% environment, GCC's cpp is what we want.
run_cpp(Args, Stdout, Stderr, ExitCode) :-
    process_create(path(cpp), Args,
                   [stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, Stdout),
    read_string(Err, _, Stderr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)).
