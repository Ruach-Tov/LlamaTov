%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% arch_summary.pl — Lift per-arch summary forms (Phase 4 of llama.cpp ingestion).
%%
%% Per Heath's directive (2026-05-16 ~03:00 UTC): "lifting the per-arch
%% builders to a concise summary form. Essentially adding parameters and
%% template parameters, except in the BPD language."
%%
%% This module extracts the LOAD-BEARING DIFFERENCES between architectures
%% as compact BPD facts:
%%
%%   1. arch_hparam_read/3 — which GGUF metadata keys an arch reads
%%      from the loader (e.g., RMS_EPS, ROPE_FREQ_BASE, EXPERT_COUNT)
%%
%%   2. arch_size_recognition/5 — the (n_layer, n_embd, n_head) →
%%      LLM_TYPE_<SIZE> mapping table per arch (how the arch recognizes
%%      published model sizes like 7B, 13B, 70B)
%%
%%   3. arch_template_param/3 — template specializations used by
%%      derived archs (e.g., llama_embed = llama::graph<embed>,
%%      phimoe = phi3::graph<iswa>)
%%
%%   4. arch_summary/1 — a concise dict-like summary record per arch
%%
%% These together ARE the "parameters and template parameters in BPD
%% language" Heath named — the compact summary form that, with a small
%% emission step, regenerates the substantive content of load_arch_hparams.
%%
%% Author: metayen 2026-05-16
%% Per Heath's option-1 follow-on directive. Composes with:
%%   - llama_cpp_lifter.pl Phase 1+2+3 (enum, dispatch, tensors, ops, aliases)
%%   - arch_params.pl (hand-curated 22-dim table, to be cross-validated)

:- module(arch_summary, [
    lift_arch_hparams/2,         % +SourcePath, -Hparams (regex bootstrap)
    lift_arch_hparams_ast/2,     % +SourcePath, -Hparams (AST-based, Satya path)
    lift_arch_size_table/2,      % +SourcePath, -SizeRecs (regex bootstrap)
    lift_arch_size_table_ast/2,  % +SourcePath, -SizeRecs (AST-based, Satya path)
    lift_arch_template_params/2, % +ModelsHeaderPath, -TemplateParams
    lift_arch_summary/3,         % +ArchName, +RepoPath, -Summary
    summary_report/0,            % Print summary for all 124 archs
    preprocess_arch_source/2,    % +SourcePath, -Source (preprocessed via cpp)
    extract_load_arch_hparams_body/2  % +Source, -Body (brace-matched extraction)
]).

:- use_module(library(readutil)).
:- use_module(library(lists)).
:- use_module(library(pcre)).
:- use_module(llama_cpp_lifter).
:- use_module(c_ast).
:- use_module(c_preprocess).


%% ─── llama.cpp include-path resolver ────────────────────────────
%%
%% Given a source path like `<repo>/src/models/bert.cpp`, derive the
%% three standard llama.cpp include directories:
%%   <repo>/src
%%   <repo>/include
%%   <repo>/ggml/include
%%
%% Used by the AST lifters when preprocessing source through cpp.
%% This is llama.cpp-specific convenience; generalize when a second
%% project's convention appears.
llama_cpp_include_paths(SourcePath, IncludePaths) :-
    %% Walk up from `<repo>/src/models/<arch>.cpp` to `<repo>`.
    file_directory_name(SourcePath, ModelsDir),
    file_directory_name(ModelsDir, SrcDir),
    file_directory_name(SrcDir, RepoDir),
    atom_concat(RepoDir, '/src', SrcInc),
    atom_concat(RepoDir, '/include', InclInc),
    atom_concat(RepoDir, '/ggml/include', GgmlInc),
    IncludePaths = [SrcInc, InclInc, GgmlInc].

%% Preprocess a source file, returning its expanded content with
%% macros expanded and #include'd headers filtered out. Uses a wide
%% range (effectively whole-file) since we want the full function
%% bodies. Falls back to raw read_file_to_string if preprocessing
%% fails (e.g., no system cpp available, or the file is outside the
%% llama.cpp convention).
%%
%% Returns: the preprocessed text as a string, ready for
%% extract_load_arch_*_body to find the function and brace-match.
preprocess_arch_source(SourcePath, Source) :-
    catch(
        ( llama_cpp_include_paths(SourcePath, IncludePaths),
          c_preprocess:preprocess_file_segment(
              SourcePath, IncludePaths, range(1, 100000), Source, _LineMap)
        ),
        _Err,
        %% Fallback: read raw source if preprocessing fails.
        read_file_to_string(SourcePath, Source, [])
    ).


%% ═════════════════════════════════════════════════════════════════════
%% Lift hparam reads from load_arch_hparams
%% ═════════════════════════════════════════════════════════════════════
%%
%% Pattern: `ml.get_key(LLM_KV_<NAME>, hparams.<field>[, optional])`
%% or `ml.get_arr_n(LLM_KV_<NAME>, ...)` etc.
%%
%% Output: list of hparam(KvKey, HparamField, Optional?) terms
%%   hparam('LLM_KV_ATTENTION_LAYERNORM_RMS_EPS', f_norm_rms_eps, required)
%%   hparam('LLM_KV_EXPERT_COUNT', n_expert, optional)
%%
%% Bootstrap regex version; upgradable to c_ast parse for precision.

%% Per 2026-05-18 regex-retirement work (A1 of the regex inventory):
%% migrated from re_foldl to per-line structural scan via
%% parse_hparam_get_key_line/4. Empirically bit-identical output on
%% 128/128 arch source files in external/llama.cpp/src/models/.
lift_arch_hparams(SourcePath, Hparams) :-
    read_file_to_string(SourcePath, Source, []),
    split_string(Source, "\n", "", Lines),
    findall(hparam(K, F, O),
        ( member(Line, Lines),
          parse_hparam_get_key_line(Line, K, F, O)
        ),
        Hparams).


%% parse_hparam_get_key_line(+Line, -KvKey, -Field, -Optionality)
%%
%% True iff Line contains `ml.get_key(LLM_KV_X, hparams.field[(), false|true])`.
%%   KvKey: atom (e.g., 'LLM_KV_ATTENTION_LAYERNORM_RMS_EPS')
%%   Field: atom (e.g., 'f_norm_rms_eps')
%%   Optionality: 'required' (no opt arg) or 'optional' (false|true present)
%%
%% Tolerates arbitrary whitespace between tokens and an optional `()`
%% after the field name (method-call form). Empirically equivalent to
%% the prior regex:
%%   ml\\.get_key\\(\\s*(?<kv>LLM_KV_[A-Z0-9_]+)\\s*,\\s*hparams\\.(?<field>[a-z0-9_]+)(?:\\(\\))?(?:\\s*,\\s*(?<opt>false|true))?\\s*\\)
parse_hparam_get_key_line(Line, KvKey, Field, Optionality) :-
    string_codes(Line, Codes),
    string_codes("ml.get_key(", Prefix),
    llama_cpp_lifter:find_substring_match(Prefix, Codes, AfterPrefix),
    llama_cpp_lifter:skip_whitespace(AfterPrefix, AfterWS1),
    string_codes("LLM_KV_", KvPrefix),
    llama_cpp_lifter:match_prefix(KvPrefix, AfterWS1, AfterKvPrefix),
    llama_cpp_lifter:take_uc_ident_chars(AfterKvPrefix, KvCodes, AfterKv),
    KvCodes \= [],
    %% Reconstruct full KvKey including the LLM_KV_ prefix.
    atom_codes(KvSuffixAtom, KvCodes),
    atom_codes(KvPrefixAtom, KvPrefix),
    atom_concat(KvPrefixAtom, KvSuffixAtom, KvKey),
    %% Comma between kv and hparams.field
    llama_cpp_lifter:skip_whitespace(AfterKv, AfterWS2),
    AfterWS2 = [0', | AfterComma],
    llama_cpp_lifter:skip_whitespace(AfterComma, AfterWS3),
    string_codes("hparams.", HpPrefix),
    llama_cpp_lifter:match_prefix(HpPrefix, AfterWS3, AfterHpPrefix),
    llama_cpp_lifter:take_ident_chars(AfterHpPrefix, FieldCodes, AfterField),
    FieldCodes \= [],
    atom_codes(Field, FieldCodes),
    %% Optional () method-call suffix on the field name.
    ( AfterField = [0'(, 0') | AfterFieldRest]
    -> AfterFieldFinal = AfterFieldRest
    ;  AfterFieldFinal = AfterField
    ),
    %% Either ) (required) or , <opt> ) (optional).
    llama_cpp_lifter:skip_whitespace(AfterFieldFinal, AfterWS4),
    ( AfterWS4 = [0', | AfterComma2]
    -> llama_cpp_lifter:skip_whitespace(AfterComma2, AfterWS5),
       ( ( string_codes("false", FalseCodes),
           llama_cpp_lifter:match_prefix(FalseCodes, AfterWS5, AfterOpt) )
       ; ( string_codes("true", TrueCodes),
           llama_cpp_lifter:match_prefix(TrueCodes, AfterWS5, AfterOpt) )
       ),
       llama_cpp_lifter:skip_whitespace(AfterOpt, [0') | _]),
       Optionality = optional
    ;  AfterWS4 = [0') | _],
       Optionality = required
    ).


%% lift_arch_hparams_ast(+SourcePath, -Hparams)
%%
%% AST-based version of lift_arch_hparams. Uses c_ast's DCG to parse
%% the load_arch_hparams function body as a statement list, then walks
%% the parsed AST extracting hparam/3 facts from c_call expressions
%% matching ml.get_key(...) shapes.
%%
%% This is the substrate-honest replacement for the regex version,
%% in alignment with Heath's Satya/Svadhyaya principle: the AST is
%% the truth representation, not string patterns.
%%
%% Returns the same hparam/3 facts as the regex version for the
%% patterns both can handle. Future extensions on the AST side
%% (e.g., || fallback patterns, local-var targets) will only work
%% on this version.

lift_arch_hparams_ast(SourcePath, Hparams) :-
    preprocess_arch_source(SourcePath, Source),
    ( extract_load_arch_hparams_body(Source, Body)
    -> c_ast:c_parse_stmts_v2_partial(Body, ASTs, _RestTokens),
       findall(H, walk_for_hparam(ASTs, H), Hparams)
    ;  Hparams = []
    ).

%% Walk an arbitrary list of AST statements, descending into nested
%% constructs (if-then bodies, if-else bodies, decl-init initializers,
%% block bodies) producing hparam/3 facts at any depth where an
%% ml.get_key call appears.
%%
%% This matches the regex version's behavior of finding all hparam
%% reads anywhere in the function body, not just at top level.

walk_for_hparam(Stmts, H) :-
    is_list(Stmts), member(Stmt, Stmts),
    stmt_walk_hparam(Stmt, H).

%% Direct ml.get_key as an expression statement at any depth.
stmt_walk_hparam(Stmt, H) :- stmt_to_hparam(Stmt, H).

%% Descend into if-then-else, INCLUDING the condition expression
%% (deepseek2 has if(ml.get_key(...))).
stmt_walk_hparam(c_if(Cond, _Then), H) :-
    stmt_to_hparam(c_expr_stmt(Cond), H).
stmt_walk_hparam(c_if(_Cond, Then), H) :-
    walk_for_hparam(Then, H).
stmt_walk_hparam(c_if(Cond, _Then, _Else), H) :-
    stmt_to_hparam(c_expr_stmt(Cond), H).
stmt_walk_hparam(c_if(_Cond, Then, _Else), H) :-
    walk_for_hparam(Then, H).
stmt_walk_hparam(c_if(_Cond, _Then, Else), H) :-
    walk_for_hparam(Else, H).

%% C++17 if-with-init (lfm2): the init-statement may be a decl-init
%% whose initializer is an ml.get_key call. Descend into it.
stmt_walk_hparam(c_if_init(Init, _Cond, _Then), H) :-
    stmt_walk_hparam(Init, H).
stmt_walk_hparam(c_if_init(_Init, _Cond, Then), H) :-
    walk_for_hparam(Then, H).

%% Descend into bare block.
stmt_walk_hparam(c_block(Stmts), H) :-
    walk_for_hparam(Stmts, H).

%% Descend into decl-init: the initializer expression might be a
%% direct ml.get_key call. Wrap it as a synthetic expr-stmt so
%% stmt_to_hparam matches against it.
stmt_walk_hparam(c_decl_init(_Type, _Name, Init), H) :-
    stmt_to_hparam(c_expr_stmt(Init), H).

%% Descend into for-loop bodies.
stmt_walk_hparam(c_for(_Init, _Cond, _Step, Body), H) :-
    walk_for_hparam(Body, H).

%% Descend into switch case bodies.
stmt_walk_hparam(c_switch(_Discrim, Cases), H) :-
    member(c_case(_, CaseBody), Cases),
    walk_for_hparam(CaseBody, H).
stmt_walk_hparam(c_switch(_Discrim, Cases), H) :-
    member(c_default(DefaultBody), Cases),
    walk_for_hparam(DefaultBody, H).

%% Extract the body of load_arch_hparams from a source string.
%% Returns the contents between the function's outer { and matching }.
extract_load_arch_hparams_body(Source, Body) :-
    sub_string(Source, Start, _, _, "::load_arch_hparams"),
    %% Find the opening brace after the signature
    string_length(Source, SrcLen),
    find_open_brace(Source, Start, SrcLen, OpenPos),
    AfterOpen is OpenPos + 1,
    find_close_brace_at_zero(Source, AfterOpen, SrcLen, 1, ClosePos),
    BodyLen is ClosePos - AfterOpen,
    sub_string(Source, AfterOpen, BodyLen, _, Body),
    !.

%% Brace-matched extraction helpers (mirror of the ones in arch_emit.pl).
%% Walks the source forward looking for { and } to find a balanced range.
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

%% Recognize an ml.get_key statement and extract its hparam/3 fact.
%% Matches both required (2 args) and optional (3 args with false flag).
stmt_to_hparam(
    c_expr_stmt(c_call(c_member(c_var(ml), get_key), Args)),
    hparam(KvKey, Field, Optionality)
) :-
    args_to_hparam_parts(Args, KvKey, Field, Optionality).

%% Required: 2 args (KV, hparams.field)
args_to_hparam_parts(
    [c_var(KvKey), c_member(c_var(hparams), Field)],
    KvKey, Field, required
).
%% Required, method-call target: 2 args (KV, hparams.method())
args_to_hparam_parts(
    [c_var(KvKey), c_call(c_member(c_var(hparams), Field), [])],
    KvKey, Field, required
).
%% Optional: 3 args (KV, hparams.field, false)
args_to_hparam_parts(
    [c_var(KvKey), c_member(c_var(hparams), Field), c_var(false)],
    KvKey, Field, optional
).
%% Optional, method-call target: 3 args (KV, hparams.method(), false)
args_to_hparam_parts(
    [c_var(KvKey), c_call(c_member(c_var(hparams), Field), []), c_var(false)],
    KvKey, Field, optional
).


%% lift_arch_size_table_ast(+SourcePath, -SizeRecs)
%%
%% AST-based version of lift_arch_size_table. Parses the
%% load_arch_hparams body via the c_ast DCG, finds the c_switch AST
%% within the parsed statements, walks the cases extracting size_rec/3
%% facts.
%%
%% Produces the same size_rec(N, Cond, Type) shape as the regex
%% lift_arch_size_table so existing consumers (arch_emit.pl) keep
%% working without changes.
%%
%% Where the AST version differs from regex:
%%   - Nested switches (e.g., bert's case 12 inner switch on n_embd)
%%     are represented structurally rather than being flattened to a
%%     top-level list (which the regex did, losing the nesting info).
%%     The AST version emits nested_switch records that preserve the
%%     enclosing case label.

lift_arch_size_table_ast(SourcePath, SizeRecs) :-
    preprocess_arch_source(SourcePath, Source),
    ( extract_load_arch_hparams_body(Source, Body)
    -> c_ast:c_parse_stmts_v2_partial(Body, ASTs, _Rest),
       collect_size_recs(ASTs, SizeRecs)
    ;  SizeRecs = []
    ).

%% Deterministic recursive descent through the AST, accumulating
%% size_recs from any c_switch found at any depth (including inside
%% if-then-else bodies, blocks, for-bodies, etc.).
collect_size_recs([], []).
collect_size_recs([Stmt | Rest], AllRecs) :-
    stmt_size_recs(Stmt, Here),
    collect_size_recs(Rest, RestRecs),
    append(Here, RestRecs, AllRecs).

stmt_size_recs(c_switch(_Discrim, Cases), Recs) :- !,
    cases_to_size_recs(Cases, Recs).
stmt_size_recs(c_if(_Cond, Then), Recs) :- !,
    collect_size_recs(Then, Recs).
stmt_size_recs(c_if(_Cond, Then, Else), Recs) :- !,
    collect_size_recs(Then, ThenRecs),
    collect_size_recs(Else, ElseRecs),
    append(ThenRecs, ElseRecs, Recs).
stmt_size_recs(c_block(Stmts), Recs) :- !,
    collect_size_recs(Stmts, Recs).
stmt_size_recs(c_for(_, _, _, Body), Recs) :- !,
    collect_size_recs(Body, Recs).
stmt_size_recs(c_if_init(_, _, Then), Recs) :- !,
    collect_size_recs(Then, Recs).
stmt_size_recs(_Other, []).

%% Walk a list of c_case/c_default clauses producing size_rec/3 facts.
%% c_default is omitted (the regex version implicitly ignores it because
%% it doesn't match the "case N: type = ..." pattern).
%%
%% Nested switches: when a case body itself contains a c_switch (as in
%% bert and arwkv7), recurse into it producing additional size_rec
%% facts. This matches the regex version's flattening behavior — it
%% saw inner cases as top-level patterns. For the AST it's a deliberate
%% choice to preserve compatibility with downstream consumers; a
%% structurally-deeper version would emit nested_size_rec instead.
cases_to_size_recs([], []).
cases_to_size_recs([c_case(Value, Body) | Rest], AllRecs) :-
    case_body_to_size_recs(Value, Body, RecsHere),
    cases_to_size_recs(Rest, RecsRest),
    append(RecsHere, RecsRest, AllRecs).
cases_to_size_recs([c_default(_) | Rest], RecRest) :-
    cases_to_size_recs(Rest, RecRest).
cases_to_size_recs([_Unrecognized | Rest], RecRest) :-
    cases_to_size_recs(Rest, RecRest).

%% Extract zero, one, or many size_recs from a case body.
%%   - direct type assignment: one size_rec at this case level
%%   - nested switch in body: recurse into its cases (flattened output)
%%   - other body: zero size_recs (skip)
case_body_to_size_recs(c_int(N), Body, [size_rec(N, Cond, Type)]) :-
    member(c_assign(c_var(type), Rhs), Body), !,
    rhs_to_cond_type(Rhs, Cond, Type).
case_body_to_size_recs(_Value, Body, RecsFromNested) :-
    member(c_switch(_NestedDiscrim, NestedCases), Body), !,
    cases_to_size_recs(NestedCases, RecsFromNested).
%% Case body wraps the inner switch in a bare block:
%%   case 0: { switch (...) { ... } } break;
%% Descend through the block to find the inner switch.
case_body_to_size_recs(_Value, Body, RecsFromNested) :-
    member(c_block(BlockStmts), Body),
    member(c_switch(_, NestedCases), BlockStmts), !,
    cases_to_size_recs(NestedCases, RecsFromNested).
case_body_to_size_recs(_Value, _Body, []).

%% Convert a case-body RHS to (Cond, Type) pair matching the regex
%% lift's size_rec output shape.
%%
%% Simple type literal: type = LLM_TYPE_X;
rhs_to_cond_type(c_var(Type), unconditional, Type) :- !.
%% Parenthesized: type = (...)
rhs_to_cond_type(c_paren(Inner), Cond, Type) :- !,
    rhs_to_cond_type(Inner, Cond, Type).
%% Ternary: type = expr ? T1 : T2  (T2 may itself be a ternary)
rhs_to_cond_type(c_ternary(CondExpr, ThenExpr, ElseExpr),
                 if_then_else(condition(LhsAtom, OpAtom, RhsAtom), TypeIf, TypeElse),
                 TypeIf) :-
    cond_expr_parts(CondExpr, LhsAtom, OpAtom, RhsAtom),
    type_expr_atom(ThenExpr, TypeIf),
    type_expr_atom(ElseExpr, TypeElse).

%% Reduce a condition AST to (Lhs, Op, Rhs) atoms matching regex output.
cond_expr_parts(c_binop(Op, Lhs, Rhs), LhsAtom, Op, RhsAtom) :-
    expr_to_label_atom(Lhs, LhsAtom),
    expr_to_label_atom(Rhs, RhsAtom).

%% Convert an expression to its label atom for size_rec compatibility:
%%   c_member(c_var(hparams), n_embd)  ->  n_embd
%%   c_call(c_member(c_var(hparams), n_head), [])  ->  'n_head()'
%%   c_var(n_vocab)  ->  n_vocab  (local variable case)
%%   c_int(1024)  ->  '1024'
expr_to_label_atom(c_member(c_var(hparams), Field), Field) :- !.
expr_to_label_atom(c_call(c_member(c_var(hparams), Field), []), Atom) :- !,
    atom_concat(Field, '()', Atom).
expr_to_label_atom(c_var(Name), Name) :- !.
expr_to_label_atom(c_int(N), Atom) :- !,
    atom_number(Atom, N).
expr_to_label_atom(Expr, ExprAtom) :-
    %% Fallback: stringify the AST term itself
    format(atom(ExprAtom), "~q", [Expr]).

%% Convert a type-side expression to a flat type atom or nested cond.
type_expr_atom(c_var(Type), Type) :- !.
type_expr_atom(c_paren(Inner), Out) :- !,
    type_expr_atom(Inner, Out).
type_expr_atom(c_ternary(C, T, E), if_then_else(Cond, TIf, TElse)) :-
    cond_expr_parts(C, L, O, R),
    Cond = condition(L, O, R),
    type_expr_atom(T, TIf),
    type_expr_atom(E, TElse).


%% ═════════════════════════════════════════════════════════════════════
%% Lift the (n_layer, n_embd, n_head) → LLM_TYPE_X size recognition
%% ═════════════════════════════════════════════════════════════════════
%%
%% Pattern: `case <N>: type = LLM_TYPE_<SIZE>; break;`
%% Or for compound conditions: `case <N>: type = hparams.n_embd == <X> ? LLM_TYPE_<A> : LLM_TYPE_<B>; break;`
%%
%% Output: list of size_rec(LayerCount, Condition, TypeLabel) terms
%%   size_rec(32, unconditional,             'LLM_TYPE_7B')
%%   size_rec(24, n_embd_eq(1024),           'LLM_TYPE_0_5B')
%%   size_rec(24, n_embd_neq(1024),          'LLM_TYPE_1B')
%%
%% Substrate-honest scope: this captures the COMMON pattern. Complex
%% nested conditions (e.g. llama's granite vocab-size branch) parse
%% as 'complex(SourceText)' for now — upgradable later.

lift_arch_size_table(SourcePath, SizeRecs) :-
    read_file_to_string(SourcePath, Source, []),
    %% Find all `case <N>:` lines and the assignment that follows
    re_foldl(
        [Dict, A0, A]>>(
            get_dict(n, Dict, NStr),
            get_dict(body, Dict, BodyStr),
            atom_string(NAtom, NStr),
            atom_number(NAtom, NInt),
            classify_size_assignment(BodyStr, ClassifiedRec),
            ClassifiedRec = size_rec(_, Cond, Type),
            A = [size_rec(NInt, Cond, Type) | A0]
        ),
        "case (?<n>\\d+):\\s*(?<body>type = [^;]+;)",
        Source,
        [],
        Reversed,
        []
    ),
    reverse(Reversed, SizeRecs).

%% classify_size_assignment(+BodyStr, -Rec)
%% Pattern A: `type = LLM_TYPE_X;` → unconditional
%% Pattern B: `type = COND ? LLM_TYPE_A : LLM_TYPE_B;` → conditional
%% Otherwise: → complex(BodyStr)
%%
%% NOTE: this is the regex-based lifter. The substrate-honest path is
%% to replace it with an AST-based lifter using tree-sitter-cpp. See
%% TODO/lift-via-ast.md. Regex variants are NOT to be deepened further;
%% the right scope is the AST refactor, not new pattern strings.
classify_size_assignment(BodyStr, size_rec(_, Cond, Type)) :-
    ( re_matchsub("type = LLM_TYPE_(?<t>[A-Z0-9_]+);", BodyStr, Dict, [])
    -> get_dict(t, Dict, TStr),
       atom_string(TAtom, TStr),
       atom_concat('LLM_TYPE_', TAtom, Type),
       Cond = unconditional
    ; re_matchsub("type = hparams\\.(?<lhs>[a-z0-9_()]+)\\s*(?<op>==|!=|<|>|<=|>=)\\s*(?<rhs>[A-Za-z0-9_.()]+)\\s*\\?\\s*LLM_TYPE_(?<ta>[A-Z0-9_]+)\\s*:\\s*LLM_TYPE_(?<tb>[A-Z0-9_]+);",
                  BodyStr, Dict, [])
    -> get_dict(lhs, Dict, LhsStr),
       get_dict(op, Dict, OpStr),
       get_dict(rhs, Dict, RhsStr),
       get_dict(ta, Dict, TaStr),
       get_dict(tb, Dict, TbStr),
       atom_string(LhsAtom, LhsStr),
       atom_string(OpAtom, OpStr),
       atom_string(RhsAtom, RhsStr),
       atom_string(TaAtom, TaStr),
       atom_string(TbAtom, TbStr),
       atom_concat('LLM_TYPE_', TaAtom, TypeIf),
       atom_concat('LLM_TYPE_', TbAtom, TypeElse),
       Cond = if_then_else(condition(LhsAtom, OpAtom, RhsAtom), TypeIf, TypeElse),
       Type = TypeIf   % pick the if-branch as canonical; both are in Cond
    ; Cond = complex(BodyStr),
      Type = unknown
    ).


%% ═════════════════════════════════════════════════════════════════════
%% Lift template-parameter specializations from models.h
%% ═════════════════════════════════════════════════════════════════════
%%
%% Pattern: `using graph = llama_model_<parent>::graph<<spec>>;`
%% Reuses llama_cpp_lifter's Phase 3c, but extracts as template_param
%% facts oriented around the CHILD arch (the one that does the using).

lift_arch_template_params(ModelsHeaderPath, TemplateParams) :-
    %% Substrate-honest approach: scan line-by-line. When we see
    %% `struct llama_model_<X>` remember the name; when we then see
    %% `using graph = llama_model_<Y>::graph[<S>];` within that struct,
    %% emit template_param(X, Y, spec).
    %%
    %% Multi-line regex spanning struct→using doesn't reliably match
    %% across the many fields between them. Stateful scan is precise.
    read_file_to_string(ModelsHeaderPath, Source, []),
    split_string(Source, "\n", "", Lines),
    scan_template_params(Lines, none, [], Reversed),
    reverse(Reversed, TemplateParams).

%% scan_template_params(+Lines, +CurrentStruct, +Acc0, -Acc)
%%
%% Per-line state machine: tracks the currently-open struct context
%% and emits a template_param/3 fact when a using-graph declaration
%% is encountered inside a struct.
%%
%% Per 2026-05-18 regex-retirement work: migrated from re_matchsub
%% to llama_cpp_lifter's parse_struct_declaration_line/2 and
%% parse_using_graph_line/3 (shared structural parsers per the
%% convolution analysis — A5/A6 + L6 use one set of matchers).
%% Empirically bit-identical output on models.h (16 template_params).
scan_template_params([], _, Acc, Acc).
scan_template_params([Line|Rest], CurrentStruct, Acc0, Acc) :-
    ( llama_cpp_lifter:parse_struct_declaration_line(Line, ChildAtom)
    -> scan_template_params(Rest, ChildAtom, Acc0, Acc)
    ; llama_cpp_lifter:parse_using_graph_line(Line, ParentAtom, Spec),
      CurrentStruct \= none
    -> ( Spec == none
       -> Specialization = none
       ;  Specialization = template(Spec)
       ),
       Entry = template_param(CurrentStruct, ParentAtom, Specialization),
       scan_template_params(Rest, CurrentStruct, [Entry|Acc0], Acc)
    ;  scan_template_params(Rest, CurrentStruct, Acc0, Acc)
    ).


%% ═════════════════════════════════════════════════════════════════════
%% Compose into a per-arch summary record
%% ═════════════════════════════════════════════════════════════════════
%%
%% Output: arch_summary(Arch) record with fields:
%%   - hparam_reads:   list of hparam/3 terms
%%   - size_table:     list of size_rec/3 terms
%%   - template_spec:  none | template(Spec) | aliased(Parent, Spec)
%%   - n_tensors:      count of unique tensor declarations
%%   - n_ops:          count of substantive op call-sites
%%
%% This is the BPD-language summary form Heath named: parameters
%% (hparam_reads + size_table) + template parameters (template_spec).

lift_arch_summary(ArchName, RepoPath, Summary) :-
    %% Resolve source path (some archs use dashed names)
    format(atom(P1), "~w/src/models/~w.cpp", [RepoPath, ArchName]),
    ( exists_file(P1)
    -> SourcePath = P1
    ;  atom_chars(ArchName, Chars),
       maplist([C, D]>>(C = '_' -> D = '-' ; D = C), Chars, DashChars),
       atom_chars(DashedName, DashChars),
       format(atom(P2), "~w/src/models/~w.cpp", [RepoPath, DashedName]),
       ( exists_file(P2)
       -> SourcePath = P2
       ;  SourcePath = none
       )
    ),
    ( SourcePath \= none
    -> lift_arch_hparams(SourcePath, Hparams),
       lift_arch_size_table(SourcePath, SizeTable),
       lift_arch_tensors(SourcePath, Tensors),
       lift_arch_graph_ops(SourcePath, Ops),
       length(Tensors, NT),
       length(Ops, NO)
    ;  Hparams = [], SizeTable = [], NT = 0, NO = 0
    ),
    %% Template spec: look up in models.h
    format(atom(ModelsH), "~w/src/models/models.h", [RepoPath]),
    ( exists_file(ModelsH)
    -> lift_arch_template_params(ModelsH, AllTemplates),
       ( member(template_param(ArchName, Parent, Spec), AllTemplates)
       -> TemplateSpec = aliased(Parent, Spec)
       ;  TemplateSpec = none
       )
    ;  TemplateSpec = none
    ),
    Summary = arch_summary(ArchName, [
        hparam_reads(Hparams),
        size_table(SizeTable),
        template_spec(TemplateSpec),
        n_tensors(NT),
        n_ops(NO)
    ]).


%% ═════════════════════════════════════════════════════════════════════
%% summary_report — sweep all 124 archs and print compact summaries
%% ═════════════════════════════════════════════════════════════════════

summary_report :-
    RepoPath = '../external/llama.cpp',
    format(atom(ModelCpp), "~w/src/llama-model.cpp", [RepoPath]),
    lift_dispatch_table(ModelCpp, Pairs),
    findall(A, member(arch_class(A, _), Pairs), AllArchs),
    sort(AllArchs, SortedArchs),
    length(SortedArchs, NArchs),
    format("=== Per-arch summary report ===~n~n", []),
    format("Sweep across ~d architectures...~n~n", [NArchs]),
    %% Aggregate stats
    findall(arch_summary(A, F),
        ( member(A, SortedArchs),
          lift_arch_summary(A, RepoPath, arch_summary(A, F))
        ),
        Summaries),
    %% Print one-liner per arch (compact)
    forall(member(arch_summary(A, F), Summaries),
        ( member(hparam_reads(HP), F),
          member(size_table(ST), F),
          member(template_spec(TS), F),
          member(n_tensors(NT), F),
          member(n_ops(NO), F),
          length(HP, NHP),
          length(ST, NST),
          ( TS = aliased(Parent, Spec)
          -> ( Spec = template(S)
             -> format("  ~w: aliased to ~w::graph<~w>~n", [A, Parent, S])
             ;  format("  ~w: aliased to ~w (no template)~n", [A, Parent])
             )
          ;  format("  ~w: ~d hparams, ~d size-recs, ~d tensors, ~d ops~n",
                    [A, NHP, NST, NT, NO])
          )
        )),
    %% Aggregate counts
    aggregate_stats(Summaries).

aggregate_stats(Summaries) :-
    findall(NHP, ( member(arch_summary(_, F), Summaries),
                   member(hparam_reads(HP), F),
                   length(HP, NHP) ), AllNHP),
    findall(NST, ( member(arch_summary(_, F), Summaries),
                   member(size_table(ST), F),
                   length(ST, NST) ), AllNST),
    sum_list(AllNHP, TotalHP),
    sum_list(AllNST, TotalST),
    length(Summaries, NA),
    AvgHP is TotalHP / NA,
    AvgST is TotalST / NA,
    format("~n=== Aggregate stats ===~n", []),
    format("Architectures: ~d~n", [NA]),
    format("Total hparam reads: ~d (avg ~2f per arch)~n", [TotalHP, AvgHP]),
    format("Total size-recognition entries: ~d (avg ~2f per arch)~n", [TotalST, AvgST]),
    %% Count aliased archs (template_spec = aliased)
    findall(A, ( member(arch_summary(A, F), Summaries),
                 member(template_spec(aliased(_, _)), F) ), Aliased),
    length(Aliased, NAliased),
    NUnique is NA - NAliased,
    format("Aliased (delegated) archs: ~d~n", [NAliased]),
    format("Unique-graph archs: ~d~n", [NUnique]).
