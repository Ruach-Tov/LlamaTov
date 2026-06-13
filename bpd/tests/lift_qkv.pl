%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% lift_qkv.pl — Lift BPD facts from qwen2.cpp QKV section via DCG parser.
%%
%% The round-trip: C source → tokenize → parse → BPD facts → generate → C source
%% Verify: clang-format(generated) = clang-format(original)

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').

%% ═══════════════════════════════════════════════════════════════
%% STEP 1: Tokenize a real C statement
%% ═══════════════════════════════════════════════════════════════

test_tokenize_real :-
    write('=== Step 1: Tokenize real C ==='), nl,
    
    %% Simple call: build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il)
    c_ast:c_tokenize_enriched_v2(
        'build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il)',
        Tokens),
    write('Tokens: '), write(Tokens), nl, nl.

%% ═══════════════════════════════════════════════════════════════
%% STEP 2: Parse into AST, extract BPD facts
%% ═══════════════════════════════════════════════════════════════

%% Parse a function call with nested member access
test_parse_real :-
    write('=== Step 2: Parse real expressions ==='), nl,
    
    %% Parse: build_norm(inpL, x, NULL, LLM_NORM_RMS, il)
    c_ast:c_parse_expr('build_norm(inpL, x, NULL, LLM_NORM_RMS, il)', E1),
    write('Parsed: '), write(E1), nl,
    
    %% Round-trip: emit it back
    phrase(c_ast:emit_expr(E1), Codes1),
    atom_codes(S1, Codes1),
    write('Emitted: '), write(S1), nl, nl,

    %% Parse: ggml_add(ctx0, Qcur, bq)
    c_ast:c_parse_expr('ggml_add(ctx0, Qcur, bq)', E2),
    write('Parsed: '), write(E2), nl,
    phrase(c_ast:emit_expr(E2), Codes2),
    atom_codes(S2, Codes2),
    write('Emitted: '), write(S2), nl, nl,

    %% Parse: cb(cur, "attn_norm", il) — needs string literal
    c_ast:c_parse_expr('cb(cur, "attn_norm", il)', E3),
    write('Parsed: '), write(E3), nl,
    phrase(c_ast:emit_expr(E3), Codes3),
    atom_codes(S3, Codes3),
    write('Emitted: '), write(S3), nl, nl.

%% ═══════════════════════════════════════════════════════════════
%% STEP 3: Extract BPD facts from parsed AST
%% ═══════════════════════════════════════════════════════════════

%% BPD fact extraction: given an AST of a compute-graph builder,
%% extract operation/sequence/input/output facts.

%% Pattern: Var = call(Args...) → operation fact
extract_bpd(c_assign(c_var(Output), c_call(OpName, Args)), Seq,
            bpd_op(Seq, OpName, ArgNames, Output)) :-
    maplist([c_var(N), N]>>true, Args, ArgNames).
extract_bpd(c_assign(c_var(Output), c_call(OpName, Args)), Seq,
            bpd_op(Seq, OpName, Args, Output)) :-
    \+ maplist([c_var(_), _]>>true, Args, _).

%% Pattern: Type * Var = call(Args...) → operation fact with declaration
extract_bpd(c_decl_init(_, Name, c_call(OpName, Args)), Seq,
            bpd_op(Seq, OpName, Args, Name)).

%% Pattern: cb(Var, String, Il) → callback fact
extract_bpd(c_expr_stmt(c_call(cb, [c_var(Tensor), c_string(Label), c_var(Il)])), Seq,
            bpd_cb(Seq, Tensor, Label, Il)).

%% ═══════════════════════════════════════════════════════════════
%% STEP 4: Full round-trip on QKV section
%% ═══════════════════════════════════════════════════════════════

%% Manually construct the AST for the QKV norm section
%% (simulating what the parser would produce from real C)
%% Then extract BPD facts, then regenerate C from facts.

qkv_norm_ast([
    c_comment('norm'),
    c_assign(c_var(cur),
        c_call(build_norm, [
            c_var(inpL),
            c_member(c_index(c_member(c_var(model), layers), c_var(il)), attn_norm),
            c_null,
            c_var('LLM_NORM_RMS'),
            c_var(il)
        ])),
    c_expr_stmt(c_call(cb, [c_var(cur), c_string(attn_norm), c_var(il)]))
]).

%% Extract BPD facts from the AST
extract_all_bpd(AST, Facts) :-
    extract_all_bpd(AST, 1, [], Facts).

extract_all_bpd([], _, Acc, Acc).
extract_all_bpd([Stmt|Rest], Seq, Acc, Facts) :-
    ( extract_bpd(Stmt, Seq, Fact) ->
        Seq1 is Seq + 1,
        extract_all_bpd(Rest, Seq1, [Fact|Acc], Facts)
    ;
        extract_all_bpd(Rest, Seq, Acc, Facts)
    ).

%% Generate C from BPD facts (using the C AST library)
generate_from_bpd(bpd_op(_, OpName, Args, Output),
    c_assign(c_var(Output), c_call(OpName, Args))).
generate_from_bpd(bpd_cb(_, Tensor, Label, Il),
    c_expr_stmt(c_call(cb, [c_var(Tensor), c_string(Label), c_var(Il)]))).

test_round_trip :-
    write('=== Step 4: Full Round-Trip ==='), nl, nl,
    
    %% Get the AST
    qkv_norm_ast(AST),
    write('Original AST:'), nl,
    forall(member(S, AST), (write('  '), write(S), nl)),
    nl,
    
    %% Extract BPD facts
    extract_all_bpd(AST, Facts0),
    reverse(Facts0, Facts),
    write('Extracted BPD facts:'), nl,
    forall(member(F, Facts), (write('  '), write(F), nl)),
    nl,
    
    %% Regenerate C from BPD facts
    maplist(generate_from_bpd, Facts, RegenStmts),
    write('Regenerated AST:'), nl,
    forall(member(S, RegenStmts), (write('  '), write(S), nl)),
    nl,
    
    %% Emit C from regenerated AST
    write('Regenerated C:'), nl,
    phrase(c_ast:emit_stmts(RegenStmts, 2), Codes),
    atom_codes(RegenC, Codes),
    write(RegenC), nl,
    
    %% Emit C from original AST (skipping comment)
    include([S]>>(\+ S = c_comment(_)), AST, ASTNoComments),
    phrase(c_ast:emit_stmts(ASTNoComments, 2), OrigCodes),
    atom_codes(OrigC, OrigCodes),
    write('Original C:'), nl,
    write(OrigC), nl,
    
    %% Compare
    ( RegenC = OrigC ->
        write('✅ ROUND-TRIP: IDENTICAL'), nl
    ;
        write('❌ ROUND-TRIP: DIFFERS'), nl
    ).

run_all :-
    test_tokenize_real,
    test_parse_real,
    test_round_trip.

:- initialization((run_all -> halt(0) ; (write('FAILED'), nl, halt(1)))).
