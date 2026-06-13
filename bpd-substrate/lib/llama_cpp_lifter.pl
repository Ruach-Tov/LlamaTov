%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% llama_cpp_lifter.pl — Lift BPD facts from llama.cpp source via c_ast.
%%
%% Per Heath's directive (2026-05-16 ~01:55 UTC, option 1 of 4 next-direction options):
%% "use Mavchin's C-AST to parse the llama.cpp, and ingest the switch-case and the
%%  compute graph builders, and convert all of that to BPD facts, so lifting all
%%  the facts for the per-architecture values in the BPD spec from the cpp source,
%%  then being able to generate that C code, modulo pretty-print for comparison."
%%
%% Two paths from BPD facts (Heath's two-path framing):
%%   Path 1: kernel fusion + warp optimization + profiling + tuning (REALIZED in
%%           prior commits: dp4a, BEATS cuBLAS by 1.12-1.87x on 3/3 shapes)
%%   Path 2: regenerate llama.cpp from concise BPD format, ease maintenance for
%%           upstream maintainers (THIS LIFTER + downstream emitter)
%%
%% Substrate state at this commit:
%%   - llama.cpp vendored at external/llama.cpp/ (depth=1 shallow)
%%   - 127 LLM_ARCH_* enum entries in src/llama-arch.h
%%   - 124 unique builder classes in src/llama-model.cpp dispatch switch
%%   - Existing QKV/FFN lifters in bpd/lib/qkv_lifter.pl + ffn_expander.pl
%%   - This file: the top-level lifter that ingests the dispatch + arch metadata
%%
%% Author: metayen 2026-05-16
%% Per Heath's option-1 directive. Cross-substrate-witness as design forks emerge.

:- module(llama_cpp_lifter, [
    lift_arch_enum/2,                 % +Path, -Archs  ; Path → [enum-name | ...]
    lift_dispatch_table/2,            % +Path, -Map    ; Path → [arch-class | ...]
    lift_arch_full/3,                 % +ArchName, +RepoPath, -BpdFacts  (Phase 3)
    lift_arch_tensors/2,              % +ArchSourcePath, -Tensors  (Phase 3a)
    lift_arch_graph_ops/2,            % +ArchSourcePath, -OpSequence  (Phase 3b)
    lift_graph_aliases/2,             % +ModelsHeaderPath, -Aliases  (Phase 3c)
    arch_count/1,                     % -N (number of archs in the enum)
    %% Shared structural parsers for models.h lines. Exported because
    %% arch_summary.pl's scan_template_params/4 also needs them.
    %% Per the 2026-05-18 regex convolution analysis (L6 ↔ A6).
    parse_using_graph_line/3,         % +Line, -Parent, -Specialization
    parse_struct_declaration_line/2,  % +Line, -ChildAtom
    %% Generic structural primitives — the substrate's accumulating
    %% toolkit for regex-free source scanning. Exported so other
    %% modules' migrations can compose them.
    match_prefix/3,                   % +Prefix, +Codes, -Rest
    find_substring_match/3,           % +Substr, +Codes, -RestAfter
    take_ident_chars/3,               % +Codes, -IdentCodes, -Rest (lowercase)
    take_uc_ident_chars/3,            % +Codes, -IdentCodes, -Rest (uppercase)
    skip_whitespace/2,                % +Codes, -RestAfterWS
    is_ident_char/1,                  % +Code
    is_uc_ident_char/1,               % +Code
    is_ident_boundary/1               % +Code
]).

:- use_module(c_ast).
:- use_module(library(readutil)).
:- use_module(library(lists)).
:- use_module(library(pcre)).


%% ═════════════════════════════════════════════════════════════════════
%% Phase 1: Lift the LLM_ARCH enum (the canonical architecture list)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Source: external/llama.cpp/src/llama-arch.h
%% Pattern: lines matching '    LLM_ARCH_<NAME>,' inside the enum
%%
%% Output: list of atoms, lowercased without the LLM_ARCH_ prefix
%%   e.g. ['llama', 'llama4', 'falcon', 'qwen2', ...]
%%
%% This is the canonical list — every other lifter operation iterates this.

%% Per 2026-05-18 regex-retirement work (continuation of Unit 4):
%% migrated from re_foldl to a structural per-line scanner. Empirically
%% bit-identical output (127 archs) on llama-arch.h. The cognitive
%% payoff is the same — one less PCRE grammar context-switch per
%% reader pass. The structural form composes the match_prefix/3,
%% is_uc_ident_char/1, and take_uc_ident_chars/3 helpers (the latter
%% two being uppercase siblings of the lowercase helpers introduced
%% with scan_op_calls/4 in commit 4ceecb97f).
lift_arch_enum(Path, Archs) :-
    read_file_to_string(Path, Source, []),
    split_string(Source, "\n", "", Lines),
    findall(A, (member(L, Lines), line_is_arch_enum_entry(L, A)), Archs).


%% line_is_arch_enum_entry(+Line, -ArchAtomLowercased)
%%
%% True iff Line has the exact shape "    LLM_ARCH_<UC_IDENT>,"
%% (4 leading spaces, the LLM_ARCH_ prefix, an uppercase identifier,
%% a comma). ArchAtomLowercased is the lowercase atom of the identifier.
%%
%% Equivalent to the prior regex "    LLM_ARCH_(?<name>[A-Z0-9_]+),"
%% with bit-identical empirical output on llama-arch.h.
line_is_arch_enum_entry(Line, ArchLc) :-
    string_codes(Line, Codes),
    %% Prefix: 4 spaces + "LLM_ARCH_" as a code list.
    %% Using string_codes here keeps the prefix in source as a natural
    %% string while feeding match_prefix its expected code-list form.
    string_codes("    LLM_ARCH_", PrefixCodes),
    match_prefix(PrefixCodes, Codes, AfterPrefix),
    take_uc_ident_chars(AfterPrefix, IdentCodes, [0', | _Trailing]),
    IdentCodes \= [],
    atom_codes(ArchAtom, IdentCodes),
    downcase_atom(ArchAtom, ArchLc).

%% Convenience wrapper.
arch_count(N) :-
    Path = '../external/llama.cpp/src/llama-arch.h',
    lift_arch_enum(Path, Archs),
    length(Archs, N).


%% ═════════════════════════════════════════════════════════════════════
%% Phase 2: Lift the dispatch table (enum → builder class)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Source: external/llama.cpp/src/llama-model.cpp
%% Pattern: switch-case mapping each LLM_ARCH_<NAME> to a class instantiation
%%
%%   case LLM_ARCH_QWEN2:
%%       return new llama_model_qwen2(params);
%%
%% Output: list of arch_class(EnumName, ClassName) pairs
%%   e.g. arch_class(qwen2, llama_model_qwen2)
%%
%% This is the FIRST artifact of the lifter — enables the
%% generate-the-dispatch-switch round-trip experiment.

%% Per 2026-05-18 regex-retirement work (L1 / α-simple subclass):
%% migrated from re_foldl to a per-line state machine.
%%
%% The pattern is a two-line block:
%%   case LLM_ARCH_X:
%%       return new llama_model_y(...);
%%
%% State machine tokens: case_only(E) | return_match(C) | inline(E,C) | other
%% State: PendingEnum = none | E (the enum from a previous case_only)
%%
%% Empirically bit-identical to the prior regex form on
%% external/llama.cpp/src/llama-model.cpp (124 pairs via either path).
lift_dispatch_table(Path, Pairs) :-
    read_file_to_string(Path, Source, []),
    split_string(Source, "\n", "", Lines),
    maplist(classify_dispatch_line, Lines, Tokens),
    scan_dispatch_tokens(Tokens, none, [], Pairs).


%% classify_dispatch_line(+Line, -Token)
%%
%% Classify a line into a dispatch-related token. Tries the most
%% specific shape first (inline = case+return), then case-only,
%% then return-match, finally 'other'.
classify_dispatch_line(Line, inline(E, C)) :-
    parse_dispatch_inline_line(Line, E, C), !.
classify_dispatch_line(Line, case_only(E)) :-
    parse_dispatch_case_line(Line, E), !.
classify_dispatch_line(Line, return_match(C)) :-
    parse_dispatch_return_line(Line, C), !.
classify_dispatch_line(_, other).


%% scan_dispatch_tokens(+Tokens, +PendingEnum, +Acc, -Pairs)
%%
%% State machine over the line token stream. PendingEnum holds the
%% enum from a preceding case_only(E); when followed by return_match(C)
%% we emit arch_class(E, C). An inline(E, C) line emits immediately.
scan_dispatch_tokens([], _, Acc, Pairs) :- reverse(Acc, Pairs).
scan_dispatch_tokens([inline(E, C) | Rest], _, Acc, Pairs) :- !,
    scan_dispatch_tokens(Rest, none, [arch_class(E, C) | Acc], Pairs).
scan_dispatch_tokens([case_only(E) | Rest], _, Acc, Pairs) :- !,
    scan_dispatch_tokens(Rest, E, Acc, Pairs).
scan_dispatch_tokens([return_match(C) | Rest], E, Acc, Pairs) :-
    E \= none, !,
    scan_dispatch_tokens(Rest, none, [arch_class(E, C) | Acc], Pairs).
scan_dispatch_tokens([_ | Rest], _, Acc, Pairs) :-
    %% Any other token clears the pending enum
    scan_dispatch_tokens(Rest, none, Acc, Pairs).


%% ═════════════════════════════════════════════════════════════════════
%% Phase 3 (pending): lift a single arch's build_graph() method
%% ═════════════════════════════════════════════════════════════════════
%%
%% lift_arch_full(+ArchName, +RepoPath, -BpdFacts)
%%
%% Lifts ALL substantive facts for an architecture:
%%   - Layer count, head count, head dim, FFN dim, RoPE settings
%%   - Op sequence in build_graph (QKV proj, attention, residuals, FFN, ...)
%%   - Special handling (GQA, MoE routing, RWKV time-mix, Mamba SSM, etc.)
%%
%% Compose this with existing lifters:
%%   - qkv_lifter.pl       — QKV section
%%   - ffn_expander.pl     — FFN section
%%   - (new) attn_lifter   — attention block
%%   - (new) layer_lifter  — full layer composition
%%
%% Status: NOT YET IMPLEMENTED.
%% Next step: pick Qwen2 as the proof-of-concept (existing QKV/FFN lifters
%% cover its sections), trace the call path from
%%   llama_model::build()  -->  llama_model_qwen2::build_graph()
%% and lift each method body.

%% lift_arch_full(+ArchName, +RepoPath, -BpdFacts)
%%
%% Composes Phase 3a (tensors) + Phase 3b (graph ops) into a complete
%% BPD-fact representation of an architecture.
%%
%% BpdFacts is a list of bpd_fact(Predicate, Args) terms:
%%   bpd_fact(arch_tensor, [qwen2, layer(i, attn_norm), [n_embd]])
%%   bpd_fact(arch_op,     [qwen2, layer(i, attention, build_norm), [inpL, attn_norm, rms]])
%%   ...
%%
%% Source path resolution: <RepoPath>/src/models/<arch>.cpp
%%   Some archs have hyphenated filenames (e.g. modern-bert.cpp); the
%%   lifter tries both arch and arch-with-dashes.

lift_arch_full(ArchName, RepoPath, BpdFacts) :-
    resolve_arch_source(ArchName, RepoPath, SourcePath),
    lift_arch_tensors(SourcePath, Tensors),
    lift_arch_graph_ops(SourcePath, Ops),
    maplist([T, bpd_fact(arch_tensor, [ArchName | T])]>>true, Tensors, TF),
    maplist([O, bpd_fact(arch_op, [ArchName | O])]>>true, Ops, OF),
    append(TF, OF, BpdFacts).

%% Try arch.cpp first, then arch-with-dashes.cpp (e.g. modern-bert)
resolve_arch_source(ArchName, RepoPath, SourcePath) :-
    format(atom(P1), "~w/src/models/~w.cpp", [RepoPath, ArchName]),
    ( exists_file(P1)
    -> SourcePath = P1
    ;  atom_chars(ArchName, Chars),
       maplist([C, D]>>(C = '_' -> D = '-' ; D = C), Chars, DashChars),
       atom_chars(DashedName, DashChars),
       format(atom(P2), "~w/src/models/~w.cpp", [RepoPath, DashedName]),
       ( exists_file(P2)
       -> SourcePath = P2
       ;  format("WARNING: source file not found for arch ~w (tried ~w and ~w)~n",
                 [ArchName, P1, P2]),
          fail
       )
    ).


%% ═════════════════════════════════════════════════════════════════════
%% Phase 3a: Lift tensor declarations from load_arch_tensors
%% ═════════════════════════════════════════════════════════════════════
%%
%% Pattern: lines like
%%   tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
%%   layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
%%
%% Output: list of [TensorRef, Shape] pairs where TensorRef captures
%% whether it's a top-level or per-layer tensor.

lift_arch_tensors(SourcePath, Tensors) :-
    read_file_to_string(SourcePath, Source, []),
    re_foldl(
        [Dict, A0, A]>>(
            get_dict(tn, Dict, TnStr),
            get_dict(shape, Dict, ShapeStr),
            atom_string(TnAtom, TnStr),
            atom_string(ShapeAtom, ShapeStr),
            parse_tensor_name(TnAtom, TensorRef),
            parse_shape(ShapeAtom, ShapeList),
            A = [[TensorRef, ShapeList] | A0]
        ),
        "create_tensor\\(tn\\((?<tn>LLM_TENSOR_[A-Z0-9_]+(?:,\\s*\"[a-z]+\")?(?:,\\s*i)?)\\),\\s*\\{(?<shape>[^}]+)\\}",
        Source,
        [],
        Reversed,
        []
    ),
    reverse(Reversed, Tensors).

%% Parse a tensor-name spec like 'LLM_TENSOR_ATTN_NORM, "weight", i' into
%% a structured term layer(i, attn_norm, weight) or tok_embd, etc.
parse_tensor_name(Atom, TensorRef) :-
    atom_codes(Atom, Codes),
    string_codes(Str, Codes),
    ( re_matchsub("LLM_TENSOR_(?<name>[A-Z0-9_]+),\\s*\"(?<part>[a-z]+)\",\\s*i", Str, Dict, [])
    -> get_dict(name, Dict, NameStr),
       get_dict(part, Dict, PartStr),
       atom_string(NameAtom, NameStr),
       downcase_atom(NameAtom, NameLc),
       atom_string(PartAtom, PartStr),
       TensorRef = layer(i, NameLc, PartAtom)
    ; re_matchsub("LLM_TENSOR_(?<name>[A-Z0-9_]+),\\s*\"(?<part>[a-z]+)\"", Str, Dict, [])
    -> get_dict(name, Dict, NameStr),
       get_dict(part, Dict, PartStr),
       atom_string(NameAtom, NameStr),
       downcase_atom(NameAtom, NameLc),
       atom_string(PartAtom, PartStr),
       TensorRef = global(NameLc, PartAtom)
    ; re_matchsub("LLM_TENSOR_(?<name>[A-Z0-9_]+)", Str, Dict, [])
    -> get_dict(name, Dict, NameStr),
       atom_string(NameAtom, NameStr),
       downcase_atom(NameAtom, NameLc),
       TensorRef = global(NameLc, weight)
    ; TensorRef = unparsed(Atom)
    ).

%% Parse a shape spec like 'n_embd, n_vocab' into [n_embd, n_vocab]
parse_shape(Atom, ShapeList) :-
    atom_string(Atom, Str),
    split_string(Str, ",", " \t", Parts),
    maplist([P, A]>>(string_to_atom(P, A)), Parts, ShapeList).


%% ═════════════════════════════════════════════════════════════════════
%% Phase 3b: Lift the per-layer op sequence from graph::graph constructor
%% ═════════════════════════════════════════════════════════════════════
%%
%% Substantive observation (qwen2 sample):
%% The per-layer body uses high-level helpers:
%%   build_norm, build_qkv, ggml_rope_ext, build_attn, ggml_add,
%%   build_ffn, build_cvec
%%
%% Plus model-global ops outside the layer loop:
%%   build_inp_embd, build_inp_pos, build_attn_inp_kv, build_inp_out_ids,
%%   build_lora_mm, ggml_build_forward_expand
%%
%% Phase 3b extracts the SEQUENCE of these call sites along with the
%% target variable (`cur`, `Qcur`, `Kcur`, `Vcur`, `ffn_inp`, ...).
%% This is enough fidelity to regenerate the build_graph body.
%%
%% Output: list of [Section, OpKind, TargetVar, Args] terms
%%   Section ∈ {preamble, layer, postamble}

lift_arch_graph_ops(SourcePath, Ops) :-
    read_file_to_string(SourcePath, Source, []),
    %% Substrate-honest approach: find the START of the graph::graph
    %% constructor and extract from there to balanced close-brace.
    %% Regex-based brace-counting is fragile for deeply-nested bodies;
    %% we use string-based scanning instead.
    ( extract_graph_constructor_body(Source, BodyStr)
    -> lift_op_sequence(BodyStr, Ops)
    ;  format("WARNING: could not find graph::graph constructor in ~w~n",
              [SourcePath]),
       Ops = []
    ).

%% Extract the body of the graph::graph constructor by finding the
%% opening line and counting braces to its matching close.
%%
%% Handles three patterns:
%%   1. `<class>::graph::graph(...) : ... {`       (most archs, e.g. qwen2)
%%   2. `<class>::graph<T>::graph(...) : ... {`    (templated archs, e.g. llama, phi3)
%%   3. `<class>::build(...) const {`              (older pattern, fallback)
extract_graph_constructor_body(Source, Body) :-
    string_length(Source, SrcLen),
    ( find_constructor_open_paren(Source, SrcLen, ParenPos)
    -> %% Skip past the constructor's parameter list (find matching close paren),
       %% then find the first `{` after that.
       NextAfter is ParenPos + 1,
       find_close_paren(Source, NextAfter, SrcLen, 1, ClosePPos),
       AfterParams is ClosePPos + 1,
       find_open_brace(Source, AfterParams, SrcLen, OpenBracePos),
       BodyStart is OpenBracePos + 1,
       find_close_brace(Source, BodyStart, SrcLen, 1, CloseBracePos),
       BodyLen is CloseBracePos - BodyStart,
       sub_string(Source, BodyStart, BodyLen, _, Body)
    ;  fail
    ).

%% Find the open-paren after `::graph::graph(` or `::graph<X>::graph(`.
find_constructor_open_paren(Source, _SrcLen, ParenPos) :-
    ( sub_string(Source, MatchStart, _, _, "::graph::graph(")
    -> ParenPos is MatchStart + 14
    ; %% Templated form: `graph<...>::graph(`
      sub_string(Source, MatchStart, _, _, ">::graph("),
      ParenPos is MatchStart + 8
    ).

%% Find matching close-paren given current depth = 1 at start.
find_close_paren(Source, Pos, MaxPos, Depth, Result) :-
    Pos < MaxPos,
    sub_string(Source, Pos, 1, _, Ch),
    NextPos is Pos + 1,
    ( Ch = "("
    -> NewDepth is Depth + 1,
       find_close_paren(Source, NextPos, MaxPos, NewDepth, Result)
    ; Ch = ")"
    -> NewDepth is Depth - 1,
       ( NewDepth =:= 0
       -> Result = Pos
       ;  find_close_paren(Source, NextPos, MaxPos, NewDepth, Result)
       )
    ; find_close_paren(Source, NextPos, MaxPos, Depth, Result)
    ).

%% Find the next '{' starting at Pos (skip past param list and base init)
find_open_brace(Source, Pos, MaxPos, Result) :-
    Pos < MaxPos,
    sub_string(Source, Pos, 1, _, Ch),
    ( Ch = "{"
    -> Result = Pos
    ;  NextPos is Pos + 1,
       find_open_brace(Source, NextPos, MaxPos, Result)
    ).

%% Find the position of the matching '}' starting after the opening '{'.
%% Pos = current scan position; Depth = current brace depth (1 = just past open).
find_close_brace(Source, Pos, MaxPos, Depth, Result) :-
    Pos < MaxPos,
    sub_string(Source, Pos, 1, _, Ch),
    NextPos is Pos + 1,
    ( Ch = "{"
    -> NewDepth is Depth + 1,
       find_close_brace(Source, NextPos, MaxPos, NewDepth, Result)
    ; Ch = "}"
    -> NewDepth is Depth - 1,
       ( NewDepth =:= 0
       -> Result = Pos
       ;  find_close_brace(Source, NextPos, MaxPos, NewDepth, Result)
       )
    ; find_close_brace(Source, NextPos, MaxPos, Depth, Result)
    ).

%% Identify substantive op calls in the constructor body.
%%
%% The 2026-05-17 audit (commit 841469458) flagged this and
%% classify_op_call/2 as 2 of 9 substantial regex sites worth
%% migrating away from. Per Heath's "F3 retires regex technology
%% for AST technology" directive, this is Migration Unit 4 of the
%% decomposition in docs/methodology/regex-lifter-decomposition.md.
%%
%% Substrate-honest replacement: walk the body's character codes
%% looking for build_X( or ggml_X( call sites at IDENTIFIER
%% BOUNDARIES. The previous regex form did not check boundaries,
%% silently producing false positives like classifying
%% "ggml_build_forward_expand" as op(build_helper, forward_expand)
%% (the substring "build_forward_expand(" matches the regex even
%% though it's mid-identifier). Empirically verified on real
%% llama.cpp sources where ggml_build_forward_expand and
%% ggml_build_forward_select exist.
%%
%% The new scanner respects identifier boundaries: a "build_" or
%% "ggml_" prefix only counts if preceded by a non-identifier
%% character. This catches the bug AND eliminates the regex.
lift_op_sequence(Body, Ops) :-
    atom_string(Body, BodyStr),
    string_codes(BodyStr, Codes),
    %% Pretend the body is preceded by a space (a boundary char) so
    %% calls at position 0 are recognized.
    scan_op_calls(Codes, 0' , [], Reversed),
    reverse(Reversed, Ops).


%% scan_op_calls(+Codes, +PrevChar, +AccRev, -OpsRev)
%%
%% Walk character codes, accumulating op(Kind, Name) terms when we
%% find build_X( or ggml_X( at an identifier boundary. PrevChar is
%% the character immediately before the current position; it
%% determines whether we're at a fresh identifier boundary.
scan_op_calls([], _, Acc, Acc).
scan_op_calls(Codes, PrevC, Acc, Result) :-
    is_ident_boundary(PrevC),
    try_op_at(Codes, build_helper, OpTerm, Rest),
    !,
    scan_op_calls(Rest, 0'(, [OpTerm | Acc], Result).
scan_op_calls(Codes, PrevC, Acc, Result) :-
    is_ident_boundary(PrevC),
    try_op_at(Codes, ggml_op, OpTerm, Rest),
    !,
    scan_op_calls(Rest, 0'(, [OpTerm | Acc], Result).
scan_op_calls([C|Rest], _, Acc, Result) :-
    scan_op_calls(Rest, C, Acc, Result).


%% try_op_at(+Codes, +OpKind, -OpTerm, -Rest)
%%
%% At the current position (Codes), try to match the prefix that
%% corresponds to OpKind (build_ or ggml_), followed by an identifier
%% suffix, followed by an open paren. On success, bind OpTerm =
%% op(OpKind, Name) and Rest to the codes after the open paren.
try_op_at(Codes, build_helper, op(build_helper, Name), Rest) :-
    match_prefix([0'b, 0'u, 0'i, 0'l, 0'd, 0'_], Codes, AfterPrefix),
    take_ident_chars(AfterPrefix, IdentCodes, [0'( | Rest]),
    IdentCodes \= [],
    atom_codes(Name, IdentCodes).
try_op_at(Codes, ggml_op, op(ggml_op, Name), Rest) :-
    match_prefix([0'g, 0'g, 0'm, 0'l, 0'_], Codes, AfterPrefix),
    take_ident_chars(AfterPrefix, IdentCodes, [0'( | Rest]),
    IdentCodes \= [],
    atom_codes(Name, IdentCodes).


%% match_prefix(+Prefix, +Codes, -Rest)
%%
%% Succeed iff Codes begins with the entire Prefix; bind Rest to
%% what follows. Pure structural pattern match — no regex.
match_prefix([], Codes, Codes).
match_prefix([P|Ps], [P|Cs], Rest) :-
    match_prefix(Ps, Cs, Rest).


%% find_substring_match(+Substr, +Codes, -RestAfter)
%%
%% Walk Codes looking for Substr as a substring; if found at some
%% position, succeed with RestAfter bound to what follows the
%% substring. Otherwise fail.
%%
%% Sibling of match_prefix/3: match_prefix anchors at the start;
%% find_substring_match scans anywhere. Used when a pattern need
%% not start at the beginning of the input.
find_substring_match(Substr, Codes, RestAfter) :-
    match_prefix(Substr, Codes, RestAfter), !.
find_substring_match(Substr, [_|Cs], RestAfter) :-
    find_substring_match(Substr, Cs, RestAfter).


%% take_ident_chars(+Codes, -IdentCodes, -Rest)
%%
%% Consume zero or more identifier characters from the front of Codes.
%% IdentCodes is the consumed prefix; Rest is what follows.
take_ident_chars([], [], []).
take_ident_chars([C|Cs], [C|IdentCodes], Rest) :-
    is_ident_char(C), !,
    take_ident_chars(Cs, IdentCodes, Rest).
take_ident_chars(Codes, [], Codes).


%% is_ident_char(+Code)
%%
%% True for ASCII chars that can appear inside a C identifier:
%% lowercase letter, digit, or underscore. Uppercase letters are
%% intentionally excluded because the op names we match are all
%% lowercase (build_X, ggml_X). However, uppercase still counts as
%% an identifier-boundary-inhibitor (see is_ident_boundary).
is_ident_char(C) :- C >= 0'a, C =< 0'z, !.
is_ident_char(C) :- C >= 0'0, C =< 0'9, !.
is_ident_char(0'_).


%% is_ident_boundary(+Code)
%%
%% True iff Code is NOT part of any C identifier. A "build_" or
%% "ggml_" prefix only counts as the start of a fresh call when
%% preceded by a boundary character. This is the substantive
%% improvement over the regex: respects identifier boundaries so
%% ggml_build_forward_expand is not mis-classified.
is_ident_boundary(C) :-
    \+ is_ident_char(C),
    \+ (C >= 0'A, C =< 0'Z).


%% is_uc_ident_char(+Code)
%%
%% True for chars that can appear inside an UPPERCASE C identifier
%% like LLM_ARCH_X or LLM_TENSOR_Y: uppercase letter, digit, or
%% underscore. Sibling of is_ident_char/1 for parsing the C++
%% enum-style constant names that recur in llama.cpp source.
is_uc_ident_char(C) :- C >= 0'A, C =< 0'Z, !.
is_uc_ident_char(C) :- C >= 0'0, C =< 0'9, !.
is_uc_ident_char(0'_).


%% take_uc_ident_chars(+Codes, -IdentCodes, -Rest)
%%
%% Sibling of take_ident_chars/3 for uppercase identifiers. Consume
%% zero or more uppercase-identifier characters from the front of
%% Codes; bind IdentCodes to the consumed prefix and Rest to the
%% remainder.
take_uc_ident_chars([], [], []).
take_uc_ident_chars([C|Cs], [C|IdentCodes], Rest) :-
    is_uc_ident_char(C), !,
    take_uc_ident_chars(Cs, IdentCodes, Rest).
take_uc_ident_chars(Codes, [], Codes).


%% skip_whitespace(+Codes, -Rest)
%%
%% Consume zero or more whitespace chars (space, tab) from the front
%% of Codes. Used by parsers that tolerate leading indent.
skip_whitespace([0' |Rest], Out) :- !, skip_whitespace(Rest, Out).
skip_whitespace([0'\t|Rest], Out) :- !, skip_whitespace(Rest, Out).
skip_whitespace(Codes, Codes).


%% parse_using_graph_line(+Line, -Parent, -Specialization)
%%
%% True iff Line contains a `using graph = llama_model_<parent>::graph<<spec>>?;`
%% C++ declaration. Tolerates leading whitespace (4-space indent in
%% models.h).
%%
%%   Parent: atom (the lowercase identifier after "llama_model_")
%%   Specialization: atom (the spec identifier) or 'none' (no spec)
%%
%% Shared between lift_graph_aliases/2 (this module) and
%% scan_template_params/4 (arch_summary.pl). Per the 2026-05-18 regex
%% convolution analysis, L6 and A6 had identical regex patterns; this
%% structural matcher serves both, eliminating the duplication.
%%
%% Empirically bit-identical to the prior regex form on models.h
%% (16 aliases via either path).
parse_using_graph_line(Line, Parent, Specialization) :-
    string_codes(Line, Codes),
    skip_whitespace(Codes, AfterWS),
    string_codes("using graph = llama_model_", UsingPrefix),
    match_prefix(UsingPrefix, AfterWS, AfterPrefix),
    take_ident_chars(AfterPrefix, ParentCodes, AfterParent),
    ParentCodes \= [],
    atom_codes(Parent, ParentCodes),
    string_codes("::graph", GraphSuffix),
    match_prefix(GraphSuffix, AfterParent, AfterGraph),
    ( AfterGraph = [0'< | InsideAngle]
    -> take_ident_chars(InsideAngle, SpecCodes, [0'> | AfterClose]),
       SpecCodes \= [],
       atom_codes(Specialization, SpecCodes),
       AfterClose = [0'; | _]
    ;  AfterGraph = [0'; | _],
       Specialization = none
    ).


%% parse_struct_declaration_line(+Line, -ChildAtom)
%%
%% True iff Line begins with `struct llama_model_<child>` followed by
%% an identifier boundary (whitespace, brace, colon, etc.). NO leading
%% whitespace is tolerated (matches the original regex anchor "^struct").
%%
%%   ChildAtom: the lowercase identifier after "llama_model_".
%%
%% Used by scan_template_params/4 in arch_summary.pl to track which
%% struct context we are inside when processing models.h lines.
parse_struct_declaration_line(Line, Child) :-
    string_codes(Line, Codes),
    string_codes("struct llama_model_", Prefix),
    match_prefix(Prefix, Codes, AfterPrefix),
    take_ident_chars(AfterPrefix, ChildCodes, AfterChild),
    ChildCodes \= [],
    %% Boundary check: end of line OR non-identifier char follows.
    ( AfterChild = [] ; AfterChild = [C|_], \+ is_ident_char(C) ),
    atom_codes(Child, ChildCodes).


%% parse_dispatch_case_line(+Line, -EnumLc)
%%
%% True iff Line has shape "<ws>case LLM_ARCH_<UC_IDENT>:" with
%% optional trailing whitespace only (no other content on this line).
%% EnumLc is the lowercase atom of the identifier after LLM_ARCH_.
%%
%% Used by lift_dispatch_table/2 (Phase 1) — recognizes the case
%% labels in the model-construction dispatch switch.
parse_dispatch_case_line(Line, EnumLc) :-
    string_codes(Line, Codes),
    skip_whitespace(Codes, AfterWS1),
    string_codes("case", CaseCodes),
    match_prefix(CaseCodes, AfterWS1, AfterCase),
    AfterCase = [WS | _],
    is_whitespace_code(WS),
    skip_whitespace(AfterCase, AfterWS2),
    string_codes("LLM_ARCH_", ArchPrefix),
    match_prefix(ArchPrefix, AfterWS2, AfterArchPrefix),
    take_uc_ident_chars(AfterArchPrefix, EnumCodes, AfterEnum),
    EnumCodes \= [],
    atom_codes(EnumAtom, EnumCodes),
    downcase_atom(EnumAtom, EnumLc),
    skip_whitespace(AfterEnum, [0': | AfterColon]),
    %% After the colon, only whitespace is allowed
    skip_whitespace(AfterColon, []).


%% parse_dispatch_return_line(+Line, -ClassAtom)
%%
%% True iff Line has shape "<ws>return new llama_model_<lc_ident>("
%% with optional content after the `(`. ClassAtom is the full
%% "llama_model_<suffix>" atom (e.g., llama_model_qwen2).
%%
%% Used by lift_dispatch_table/2 as the second half of the
%% pair-of-lines state machine.
parse_dispatch_return_line(Line, ClassAtom) :-
    string_codes(Line, Codes),
    skip_whitespace(Codes, AfterWS),
    string_codes("return new llama_model_", ReturnPrefix),
    match_prefix(ReturnPrefix, AfterWS, AfterPrefix),
    take_ident_chars(AfterPrefix, ClassCodes, AfterClass),
    ClassCodes \= [],
    AfterClass = [0'( | _],
    %% Reconstruct full class name including "llama_model_" prefix.
    append([0'l, 0'l, 0'a, 0'm, 0'a, 0'_, 0'm, 0'o, 0'd, 0'e, 0'l, 0'_], ClassCodes, FullCodes),
    atom_codes(ClassAtom, FullCodes).


%% parse_dispatch_inline_line(+Line, -EnumLc, -ClassAtom)
%%
%% True iff Line has shape "<ws>case LLM_ARCH_X: return new llama_model_Y(..."
%% (both case label and return statement on the same line).
%%
%% Defensive — real llama-model.cpp has these on separate lines,
%% but the regex's `\s*` between `:` and `return new` would tolerate
%% either form. Future maintainers may inline a case; the structural
%% form handles it correctly.
parse_dispatch_inline_line(Line, EnumLc, ClassAtom) :-
    string_codes(Line, Codes),
    skip_whitespace(Codes, AfterWS1),
    string_codes("case", CaseCodes),
    match_prefix(CaseCodes, AfterWS1, AfterCase),
    AfterCase = [WS | _],
    is_whitespace_code(WS),
    skip_whitespace(AfterCase, AfterWS2),
    string_codes("LLM_ARCH_", ArchPrefix),
    match_prefix(ArchPrefix, AfterWS2, AfterArchPrefix),
    take_uc_ident_chars(AfterArchPrefix, EnumCodes, AfterEnum),
    EnumCodes \= [],
    atom_codes(EnumAtom, EnumCodes),
    downcase_atom(EnumAtom, EnumLc),
    skip_whitespace(AfterEnum, [0': | AfterColon]),
    skip_whitespace(AfterColon, AfterColonWS),
    string_codes("return new llama_model_", ReturnPrefix),
    match_prefix(ReturnPrefix, AfterColonWS, AfterReturnPrefix),
    take_ident_chars(AfterReturnPrefix, ClassCodes, AfterClass),
    ClassCodes \= [],
    AfterClass = [0'( | _],
    append([0'l, 0'l, 0'a, 0'm, 0'a, 0'_, 0'm, 0'o, 0'd, 0'e, 0'l, 0'_], ClassCodes, FullCodes),
    atom_codes(ClassAtom, FullCodes).


%% is_whitespace_code(+Code)
%%
%% Whitespace per C source conventions: space, tab, newline, CR.
%% Stricter than skip_whitespace/2 which only handles space and tab;
%% used where we need to recognize end-of-line as whitespace too.
is_whitespace_code(0' ).
is_whitespace_code(0'\t).
is_whitespace_code(0'\n).
is_whitespace_code(0'\r).

%% ═════════════════════════════════════════════════════════════════════
%% Phase 3c: Lift graph-aliasing relationships
%% ═════════════════════════════════════════════════════════════════════
%%
%% Substantive substrate finding: many architectures reuse another arch's
%% graph builder via C++ `using` declarations in src/models/models.h.
%% Example: `using graph = llama_model_bert::graph;` means jina_bert_v2
%% delegates to bert.
%%
%% Output: list of graph_alias(Child, Parent, Specialization?) terms
%%   graph_alias(jina_bert_v2, bert, none)
%%   graph_alias(llama_embed, llama, embed)        % template specialization
%%   graph_alias(phimoe, phi3, iswa)

%% Per 2026-05-18 regex-retirement work: migrated from re_foldl to
%% per-line structural scan via parse_using_graph_line/3 (shared with
%% arch_summary's scan_template_params). Empirically bit-identical
%% output on models.h (16 aliases). See
%% docs/methodology/regex-inventory-uniform.md for the convolution
%% analysis that motivated unifying L6 and A6.
lift_graph_aliases(ModelsHeaderPath, Aliases) :-
    read_file_to_string(ModelsHeaderPath, Source, []),
    split_string(Source, "\n", "", Lines),
    findall(graph_alias(parent(P), spec(S)),
        ( member(Line, Lines),
          parse_using_graph_line(Line, P, S)
        ),
        Aliases).


%% classify_op_call/2 was the regex-based classifier for individual
%% call-site spec strings. It became dead code when lift_op_sequence
%% switched to inline boundary-respecting classification via
%% scan_op_calls/4 (Migration Unit 4 of the 2026-05-17 regex audit).
%% Removed in the same commit that shipped scan_op_calls.


%% ═════════════════════════════════════════════════════════════════════
%% Sanity check / smoke test
%% ═════════════════════════════════════════════════════════════════════

smoke_test :-
    format("=== llama_cpp_lifter smoke test ===~n", []),
    %% Test Phase 1
    Path1 = '../external/llama.cpp/src/llama-arch.h',
    lift_arch_enum(Path1, Archs),
    length(Archs, NArchs),
    format("Phase 1 (arch enum): ~d architectures lifted~n", [NArchs]),
    %% Test Phase 2
    Path2 = '../external/llama.cpp/src/llama-model.cpp',
    lift_dispatch_table(Path2, Pairs),
    length(Pairs, NPairs),
    format("Phase 2 (dispatch table): ~d arch-class pairs lifted~n", [NPairs]),
    %% Test Phase 3 — lift Qwen2 fully
    format("~nPhase 3 — lifting Qwen2 substantive content:~n", []),
    Qwen2Path = '../external/llama.cpp/src/models/qwen2.cpp',
    lift_arch_tensors(Qwen2Path, QwenTensors),
    length(QwenTensors, NQwenT),
    format("  Phase 3a (tensors): ~d tensor declarations lifted~n", [NQwenT]),
    ( NQwenT >= 3
    -> nth0(0, QwenTensors, T0), format("    Tensor 0: ~w~n", [T0]),
       nth0(1, QwenTensors, T1), format("    Tensor 1: ~w~n", [T1]),
       nth0(2, QwenTensors, T2), format("    Tensor 2: ~w~n", [T2])
    ; true
    ),
    lift_arch_graph_ops(Qwen2Path, QwenOps),
    length(QwenOps, NQwenOps),
    format("  Phase 3b (graph ops): ~d op call-sites lifted~n", [NQwenOps]),
    ( NQwenOps >= 5
    -> length(First5, 5), append(First5, _, QwenOps),
       format("    First 5: ~w~n", [First5])
    ; format("    All: ~w~n", [QwenOps])
    ),
    %% Test compose
    format("~nPhase 3 compose — lift_arch_full(qwen2, ...):~n", []),
    lift_arch_full(qwen2, '../external/llama.cpp', QwenFacts),
    length(QwenFacts, NQwenF),
    format("  Total BPD facts for qwen2: ~d~n", [NQwenF]),
    %% Cross-check: every enum entry should have a dispatch case?
    %% (Some enums like CLIP, T5 etc may NOT have dispatch entries —
    %% those are vision encoders or special cases. Count substantively.)
    findall(E, member(arch_class(E, _), Pairs), DispatchedEnums),
    sort(DispatchedEnums, DispatchedSet),
    sort(Archs, ArchsSet),
    intersection(DispatchedSet, ArchsSet, Both),
    length(Both, NBoth),
    subtract(ArchsSet, DispatchedSet, EnumOnly),
    length(EnumOnly, NEnumOnly),
    subtract(DispatchedSet, ArchsSet, DispatchOnly),
    length(DispatchOnly, NDispatchOnly),
    format("Cross-check:~n", []),
    format("  Enum AND dispatch: ~d~n", [NBoth]),
    format("  Enum-only (no dispatch — e.g., CLIP, T5): ~d~n", [NEnumOnly]),
    format("  Dispatch-only (subclasses?): ~d~n", [NDispatchOnly]),
    ( NEnumOnly =< 20
    -> format("  Enum-only list: ~w~n", [EnumOnly])
    ;  true
    ).
