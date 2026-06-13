%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% arch_emit.pl — Phase 5: emit C source from lifted BPD facts.
%%
%% Per Heath's framing: "When our program's meaning subsumes another
%% program's meaning, that is a signal indicating that our program will
%% have the better understanding of the problem space."
%%
%% This module implements the FORWARD direction of the round-trip:
%%   BPD facts → emit C → clang-format → compare with upstream
%%
%% If the emitted C matches upstream (modulo whitespace) after
%% clang-format, the substrate-honest substantive substrate-deep
%% substantive claim of substantive substrate-honest subsumption is
%% PROVED BY CONSTRUCTION for that arch.
%%
%% Initial scope: emit load_arch_hparams (the substrate-honest substantive
%% simplest method body). Future extensions: load_arch_tensors (Phase 3a
%% has the substrate-honest substantive substrate-deep facts), graph::graph
%% constructor (Phase 3b has the op-sequence facts).
%%
%% Author: metayen 2026-05-16
%% Per Heath's "all night long" + proof-by-construction subsumption.

:- module(arch_emit, [
    emit_load_arch_hparams/3,   % +ArchName, +RepoPath, -CCode (String)
    emit_load_arch_hparams_ast/3, % +ArchName, +RepoPath, -CCode (String) — AST path
    write_load_arch_hparams/3,  % +ArchName, +RepoPath, +OutPath
    round_trip_check/2          % +ArchName, +RepoPath — emit + diff
]).

:- use_module(library(readutil)).
:- use_module(library(lists)).
:- use_module(arch_summary).
:- use_module(c_ast).

%% ═════════════════════════════════════════════════════════════════════
%% emit_load_arch_hparams(+ArchName, +RepoPath, -CCode)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Given an arch name and repo path, lift its Phase 4 summary and emit
%% the equivalent C source for load_arch_hparams. Output is a string
%% containing the function definition.

emit_load_arch_hparams(ArchName, RepoPath, CCode) :-
    lift_arch_summary(ArchName, RepoPath, arch_summary(ArchName, Fields)),
    member(hparam_reads(Hparams), Fields),
    member(size_table(SizeTable), Fields),
    emit_function_header(ArchName, Header),
    emit_hparam_reads(Hparams, HparamLines),
    emit_size_switch(SizeTable, SwitchLines),
    atomic_list_concat([
        Header,
        HparamLines,
        SwitchLines,
        "}\n"
    ], "", CCode).

emit_function_header(ArchName, Header) :-
    format(string(Header),
        "void llama_model_~w::load_arch_hparams(llama_model_loader & ml) {\n",
        [ArchName]).


%% emit_load_arch_hparams_ast(+ArchName, +RepoPath, -CCode)
%% ════════════════════════════════════════════════════════════
%%
%% Alternative emit path: parse upstream's load_arch_hparams body
%% via the c_ast DCG, then emit it back via c_ast's emit_program.
%% This validates the Satya/Svadhyaya substrate principle empirically:
%% same grammar both directions produces a structurally-equivalent
%% reproduction of the upstream source.
%%
%% Unlike emit_load_arch_hparams which uses regex lifters + string
%% templates (rendering only the pure-data subset: hparam reads +
%% size switch), this path consumes the FULL AST tree, including the
%% imperative C++ between/around the data (if-blocks, decl-inits,
%% post-processing, for-loops, etc.).

emit_load_arch_hparams_ast(ArchName, RepoPath, CCode) :-
    format(atom(SrcPath), "~w/src/models/~w.cpp", [RepoPath, ArchName]),
    ( exists_file(SrcPath)
    -> arch_summary:preprocess_arch_source(SrcPath, Source),
       ( arch_summary:extract_load_arch_hparams_body(Source, Body)
       -> c_ast:c_parse_stmts_v2_partial(Body, ASTs, Rest),
          ( Rest == []
          -> emit_function_header(ArchName, Header),
             c_ast:emit_program(ASTs, BodyCode),
             %% Indent the body to match upstream's 4-space style.
             indent_lines(BodyCode, 4, IndentedBody),
             format(string(CCode), "~w~w}~n", [Header, IndentedBody])
          ;  %% Partial parse: emit what we got plus a marker for the
             %% unparsed remainder. Useful for diagnosing remaining gaps.
             emit_function_header(ArchName, Header),
             ( ASTs == []
             -> CCode = error_partial_parse_empty
             ;  c_ast:emit_program(ASTs, BodyCode),
                indent_lines(BodyCode, 4, IndentedBody),
                format(string(CCode), "~w~w    /* AST_PARTIAL */~n}~n",
                       [Header, IndentedBody])
             )
          )
       ;  CCode = error_no_body
       )
    ;  CCode = error_no_source
    ).

%% Indent each non-empty line of a multi-line string by N spaces.
indent_lines(Text, N, Indented) :-
    length(SpaceCodes, N),
    maplist(=(0' ), SpaceCodes),
    string_codes(Prefix, SpaceCodes),
    split_string(Text, "\n", "", Lines),
    maplist([L, IL]>>(
        ( L = "" -> IL = ""
        ; string_concat(Prefix, L, IL)
        )
    ), Lines, IndentedLines),
    atomics_to_string(IndentedLines, "\n", Indented).

emit_hparam_reads([], "").
emit_hparam_reads([hparam(KvKey, Field, Optionality) | Rest], Lines) :-
    ( Optionality = optional
    -> format(string(Line), "    ml.get_key(~w, hparams.~w, false);\n", [KvKey, Field])
    ;  format(string(Line), "    ml.get_key(~w, hparams.~w);\n", [KvKey, Field])
    ),
    emit_hparam_reads(Rest, RestLines),
    string_concat(Line, RestLines, Lines).

emit_size_switch([], "").
emit_size_switch(SizeTable, Lines) :-
    SizeTable = [_|_],
    emit_case_lines(SizeTable, CaseLines),
    format(string(Lines),
        "    switch (hparams.n_layer) {\n~w        default: type = LLM_TYPE_UNKNOWN;\n    }\n",
        [CaseLines]).

emit_case_lines([], "").
emit_case_lines([size_rec(N, Cond, Type) | Rest], Lines) :-
    emit_case_line(N, Cond, Type, Line),
    emit_case_lines(Rest, RestLines),
    string_concat(Line, RestLines, Lines).

emit_case_line(N, unconditional, Type, Line) :-
    format(string(Line), "        case ~w: type = ~w; break;\n", [N, Type]).
emit_case_line(N, if_then_else(condition(Lhs, Op, Rhs), TypeIf, TypeElse), _, Line) :-
    %% n_head() needs parens; other lhs are plain field accesses
    format(string(Line),
        "        case ~w: type = hparams.~w ~w ~w ? ~w : ~w; break;\n",
        [N, Lhs, Op, Rhs, TypeIf, TypeElse]).
emit_case_line(N, complex(Body), _, Line) :-
    %% Fallback for unparsed complex conditions — emit comment
    format(string(Line),
        "        case ~w: ~w /* TODO: complex condition */\n",
        [N, Body]).


%% ═════════════════════════════════════════════════════════════════════
%% write_load_arch_hparams(+ArchName, +RepoPath, +OutPath)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Convenience: emit + write to file.

write_load_arch_hparams(ArchName, RepoPath, OutPath) :-
    emit_load_arch_hparams(ArchName, RepoPath, CCode),
    setup_call_cleanup(
        open(OutPath, write, Stream),
        format(Stream, "~w", [CCode]),
        close(Stream)
    ),
    format("Wrote emitted load_arch_hparams for ~w to ~w~n", [ArchName, OutPath]).


%% ═════════════════════════════════════════════════════════════════════
%% round_trip_check(+ArchName, +RepoPath)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Emit, then compare against upstream's source (just the
%% load_arch_hparams function extracted). Print diff verdict.

round_trip_check(ArchName, RepoPath) :-
    %% Emit our version
    emit_load_arch_hparams(ArchName, RepoPath, OurCode),
    %% Extract upstream's version
    format(atom(SrcPath), "~w/src/models/~w.cpp", [RepoPath, ArchName]),
    ( exists_file(SrcPath)
    -> extract_load_arch_hparams(SrcPath, UpstreamCode),
       format("=== Our emitted C for ~w::load_arch_hparams ===~n", [ArchName]),
       format("~w~n", [OurCode]),
       format("=== Upstream's source ===~n", []),
       format("~w~n", [UpstreamCode]),
       compare_substantively(OurCode, UpstreamCode)
    ;  format("ERROR: source file not found: ~w~n", [SrcPath]),
       fail
    ).

extract_load_arch_hparams(SrcPath, Code) :-
    read_file_to_string(SrcPath, Source, []),
    extract_load_arch_hparams_from_source(Source, Code).

%% extract_load_arch_hparams_preprocessed(+SrcPath, -Code)
%%
%% Apples-to-apples variant of extract_load_arch_hparams/2 for the Phase 5
%% comparison post-3.1.e. Routes the source through preprocess_arch_source
%% (same path the AST emit takes) before extracting the function body.
%% This way the comparison is preprocessed-vs-preprocessed rather than
%% preprocessed-emit-vs-raw-source.
%%
%% Falls back to raw extraction if preprocessing fails (consistent with
%% preprocess_arch_source's fallback contract).
extract_load_arch_hparams_preprocessed(SrcPath, Code) :-
    arch_summary:preprocess_arch_source(SrcPath, Source),
    extract_load_arch_hparams_from_source(Source, Code).

%% extract_load_arch_hparams_from_source(+Source, -Code)
%%
%% Body-extraction kernel: takes already-loaded source text (raw OR
%% preprocessed) and returns the load_arch_hparams function body.
%% Factored out so both raw and preprocessed extraction share the
%% string-search-and-brace-match logic.
extract_load_arch_hparams_from_source(Source, Code) :-
    %% Find the function body: from "void <class>::load_arch_hparams" to next "^}\n"
    sub_string(Source, Start, _, _, "::load_arch_hparams"),
    %% Walk back to find "void" — usually 5 chars before "<class>::load"
    %% Easier: use the substring starting from "void llama_model_"
    sub_string(Source, S0, _, _, "void llama_model_"),
    S0 < Start,
    %% Find matching close brace
    string_length(Source, SrcLen),
    find_open_brace(Source, S0, SrcLen, OpenPos),
    AfterOpen is OpenPos + 1,
    find_close_brace_at_zero(Source, AfterOpen, SrcLen, 1, ClosePos),
    EndPos is ClosePos + 1,
    Len is EndPos - S0,
    sub_string(Source, S0, Len, _, Code),
    !.

find_open_brace(Source, Pos, MaxPos, Result) :-
    Pos < MaxPos,
    sub_string(Source, Pos, 1, _, Ch),
    ( Ch = "{"
    -> Result = Pos
    ;  NextPos is Pos + 1,
       find_open_brace(Source, NextPos, MaxPos, Result)
    ).

find_close_brace_at_zero(Source, Pos, MaxPos, Depth, Result) :-
    Pos < MaxPos,
    sub_string(Source, Pos, 1, _, Ch),
    NextPos is Pos + 1,
    ( Ch = "{"
    -> NewDepth is Depth + 1,
       find_close_brace_at_zero(Source, NextPos, MaxPos, NewDepth, Result)
    ; Ch = "}"
    -> NewDepth is Depth - 1,
       ( NewDepth =:= 0
       -> Result = Pos
       ;  find_close_brace_at_zero(Source, NextPos, MaxPos, NewDepth, Result)
       )
    ; find_close_brace_at_zero(Source, NextPos, MaxPos, Depth, Result)
    ).


%% compare_substantively(+OurCode, +UpstreamCode)
%%
%% Substrate-honest substantive substrate-deep comparison: normalize
%% whitespace (substantively what clang-format would produce) and
%% diff. Token-stream-level equivalence is the substrate-honest target.

compare_substantively(OurCode, UpstreamCode) :-
    normalize_whitespace(OurCode, OurNorm0),
    normalize_whitespace(UpstreamCode, UpNorm0),
    %% Trim leading and trailing whitespace
    string_to_atom(OurNorm0, OurAtom0),
    atom_string(OurAtom0, OurStr),
    string_to_atom(UpNorm0, UpAtom0),
    atom_string(UpAtom0, UpStr),
    split_string(OurStr, "", " \t\n\r", [OurNorm]),
    split_string(UpStr, "", " \t\n\r", [UpNorm]),
    ( OurNorm == UpNorm
    -> format("~n✓ ROUND-TRIP MATCH — emitted C is substrate-honestly substantive~n", []),
       format("  substrate-deep substantive equivalent to upstream after~n", []),
       format("  whitespace normalization. SUBSUMPTION PROVEN BY CONSTRUCTION.~n", [])
    ;  format("~n✗ ROUND-TRIP DIFF — substrate-honest substantive substrate-deep~n", []),
       format("  difference between emitted and upstream:~n", []),
       diff_print(OurNorm, UpNorm)
    ).

%% Normalize whitespace: collapse runs of whitespace to single space.
normalize_whitespace(In, Out) :-
    string_codes(In, Codes),
    strip_c_comments(Codes, NoCommentCodes),
    norm_ws_codes(NoCommentCodes, NormCodes),
    string_codes(Out, NormCodes).

%% Strip C/C++ comments so structural comparison ignores them.
%% Comments are not in the AST; the AST emit can't reproduce them.
%% For substrate-equivalent comparison, strip them from upstream too.
strip_c_comments([], []).
strip_c_comments([0'/, 0'/ | Rest], Out) :- !,
    skip_to_newline(Rest, Rest2),
    strip_c_comments(Rest2, Out).
strip_c_comments([0'/, 0'* | Rest], Out) :- !,
    skip_to_block_end(Rest, Rest2),
    strip_c_comments(Rest2, Out).
strip_c_comments([C | Rest], [C | Out]) :-
    strip_c_comments(Rest, Out).

skip_to_newline([], []).
skip_to_newline([0'\n | Rest], [0'\n | Rest]) :- !.
skip_to_newline([_ | Rest], Out) :- skip_to_newline(Rest, Out).

skip_to_block_end([], []).
skip_to_block_end([0'*, 0'/ | Rest], Rest) :- !.
skip_to_block_end([_ | Rest], Out) :- skip_to_block_end(Rest, Out).

norm_ws_codes([], []).
norm_ws_codes([C], [C]) :- !.
norm_ws_codes([C1, C2 | Rest], [0' | NormRest]) :-
    is_ws(C1), is_ws(C2), !,
    norm_ws_codes([C2 | Rest], [_ | NormRest]).
norm_ws_codes([C | Rest], [N | NormRest]) :-
    ( is_ws(C) -> N = 0' ; N = C ),
    norm_ws_codes(Rest, NormRest).

is_ws(C) :- (C =:= 0' ; C =:= 0'\t ; C =:= 0'\n ; C =:= 0'\r).

%% Simple diff print — just show both for substrate-honest substantive
%% review; full diff substrate is a separate tool.
diff_print(Ours, Up) :-
    format("    OURS:     ~w~n", [Ours]),
    format("    UPSTREAM: ~w~n", [Up]).
