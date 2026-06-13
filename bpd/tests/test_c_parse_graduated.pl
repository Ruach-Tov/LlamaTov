%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_c_parse_graduated.pl — graduated C-parser test suite (Iyun, 2026-05-29)
%% Design: Heath (atomic->composed measurable climb) + medayek (tiered-coverage methodology:
%% machine-readable counts, parse vs round-trip as first-class tiers, expected-fail annotations).
%%
%% TIERS:
%%   PARSE tier      — does the input parse?  (parse bugs)
%%   ROUND-TRIP tier — does parse->emit->reparse give the SAME AST?  (emit bugs — diff owner)
%% LEVELS (the climb): L0 literals -> L1 ops -> L2 stmts -> L3 composed exprs -> L4 compound
%%   -> L5 builder lines.  (L6 builder-blocks / L7 whole-file are separate heavier tests.)
%%
%% MACHINE-READABLE: each run asserts result(Tier,Level,Case,Status). Query level_summary/5
%%   and tier_summary/4. CI gate: \+ unexpected_fail(_,_,_,_).  (expected_fail = known gap;
%%   unexpected_fail = regression; unexpected_PASS = a gap got fixed -> promote it.)
%%
%% Invoke (CI):  swipl -q -g "use_module(lib/c_ast),consult(tests/test_c_parse_graduated),run_all,gate,halt"
%% Invoke (human): ...,run_all,report,halt

:- module(test_c_parse_graduated,
    [ run_all/0, report/0, gate/0,
      level_summary/5, tier_summary/4, unexpected_fail/4, unexpected_pass/3 ]).
:- dynamic result/4.   % result(Tier, Level, Case, Status)  Status in {pass,fail}

%% ---- KNOWN GAPS (expected failures): expected_fail(Tier, Case, Reason) ----
%% When a fix lands, the case PASSES -> unexpected_pass -> promote (remove the annotation).
expected_fail(parse, 'hex',           "tokenizer: 0xFF -> num(0) id(xFF)").
expected_fail(parse, 'bitand',        "expr: bitwise & has no production").
expected_fail(parse, 'bitor',         "tokenizer: | not lexed").
expected_fail(parse, 'bitxor',        "tokenizer: ^ not lexed").
expected_fail(parse, 'bitnot',        "tokenizer: ~ not lexed").
expected_fail(parse, 'shl',           "tokenizer+expr: << shift").
expected_fail(parse, 'shr',           "tokenizer+expr: >> shift").
expected_fail(parse, 'const-ptr',     "decl: const + named type (pointer)").
expected_fail(parse, 'const-named-noptr', "decl: const + named type (non-pointer)").
expected_fail(parse, 'return-val',    "stmt: return with value (no production)").
expected_fail(parse, 'while',         "stmt: while loop (no production)").

%% ---- harness ----
parse_expr_ok(Str) :- atom_string(A,Str), c_ast:c_tokenize(A,T),
    once(( phrase(c_ast:parse_expr_v2(_),T) ; phrase(c_ast:parse_expr(_),T) )).
parse_stmt_ok(Str) :- atom_string(A,Str), c_ast:c_tokenize(A,T),
    once(( phrase(c_ast:parse_stmt_v2(_),T) ; phrase(c_ast:parse_stmt(_),T) )).
parse_any(Str, AST) :- atom_string(A,Str), c_ast:c_tokenize(A,T),
    once(( phrase(c_ast:parse_stmt_v2(AST),T) ; phrase(c_ast:parse_stmt(AST),T)
         ; phrase(c_ast:parse_expr_v2(AST),T) ; phrase(c_ast:parse_expr(AST),T) )).

%% round-trip: parse -> emit -> reparse -> AST==AST  (statement or expression)
roundtrip_ok(Str) :-
    parse_any(Str, A1),
    ( catch(c_ast:emit_c(A1,R),_,fail) -> RA = R
    ; phrase(c_ast:emit_expr(A1),C), atom_codes(RA,C) ),
    parse_any(RA, A2), A1 == A2.

%% record a parse-tier case
pcase(Level, Case, Kind, Str) :-
    ( Kind == expr -> Goal = parse_expr_ok(Str) ; Goal = parse_stmt_ok(Str) ),
    ( catch(once(Goal),_,fail) -> S = pass ; S = fail ),
    assertz(result(parse, Level, Case, S)).
%% record a round-trip-tier case (only meaningful if it parses)
rcase(Level, Case, Str) :-
    ( catch(once(roundtrip_ok(Str)),_,fail) -> S = pass ; S = fail ),
    assertz(result(roundtrip, Level, Case, S)).

%% test both tiers for one case
both(Level, Case, Kind, Str) :- pcase(Level, Case, Kind, Str), rcase(Level, Case, Str).

%% ---- LEVELS (each case run on PARSE + ROUND-TRIP tiers) ----
level(0, [ ex('int',"5"), ex('float',"1.0"), ex('float-f',"1.0f"), ex('var',"x"),
           ex('nullptr',"nullptr"), ex('string',"\"hi\""),
           ex('neg-int',"-1"), ex('neg-float',"-1.0"), ex('hex',"0xFF") ]).
level(1, [ ex('add',"a + b"), ex('sub',"a - b"), ex('mul',"a * b"), ex('div',"a / b"),
           ex('mod',"a % b"), ex('eq',"a == b"), ex('lt',"a < b"), ex('gt',"a > b"),
           ex('logand',"a && b"), ex('logor',"a || b"), ex('ternary',"cond ? a : b"),
           ex('bitand',"a & b"), ex('bitor',"a | b"), ex('bitxor',"a ^ b"),
           ex('bitnot',"~a"), ex('shl',"a << 2"), ex('shr',"a >> 2"),
           ex('uminus-var',"-x"), ex('uminus-call',"-f(x)") ]).
level(2, [ st('decl',"int x;"), st('decl-init',"int x = 5;"),
           st('decl-ptr',"ggml_tensor * cur = f(ctx);"), st('assign',"cur = g(x);"),
           st('const-int',"const int x = 5;"), st('const-i64',"const int64_t n = h.v;"),
           st('const-ptr',"const ggml_tensor * p = q;"),
           st('const-named-noptr',"const ggml_tensor x = y;"),
           st('auto',"auto x = f();"), st('auto-ptr',"auto * p = build();") ]).
level(3, [ ex('nested-call',"f(g(x), h.y)"), ex('chain',"a.b[i].c"),
           ex('ggml',"ggml_mul_mat(ctx0, model.layers[il].wq, cur)"),
           ex('arrow-chain',"ctx->params.n_embd") ]).
level(4, [ st('if-block',"if (a) { x = 1; }"), st('if-else',"if (a) { x = 1; } else { x = 2; }"),
           st('if-cmp',"if (il == 0) { cur = inpL; }"),
           st('for-basic',"for (int i = 0; i < n; i++) { x = i; }"),
           st('for-preinc',"for (int il = 0; il < n_layer; ++il) { cur = f(il); }"),
           st('if-in-for',"for (int i = 0; i < n; i++) { if (i == 0) { x = 1; } }"),
           st('return-val',"return cur;"), st('while',"while (a) { x = 1; }") ]).
level(5, [ st('inp-embd',"ggml_tensor * inpL = build_inp_embd(model.tok_embd);"),
           st('rms-norm',"cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);"),
           st('qcur',"ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);"),
           st('residual',"cur = ggml_add(ctx0, cur, inpL);"),
           st('scale',"cur = ggml_scale(ctx0, cur, scale);"),
           st('reshape',"Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);"),
           st('const-decl',"const int64_t n_embd_head = hparams.n_embd_head_v;"),
           st('wo-proj',"cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);") ]).

run_level(L) :- level(L, Cases),
    forall(member(C, Cases),
        ( C = ex(Name,Str) -> both(L, Name, expr, Str)
        ; C = st(Name,Str) -> both(L, Name, stmt, Str) )).

run_all :- retractall(result(_,_,_,_)), forall(level(L,_), run_level(L)).

%% ---- QUERYABLE SUMMARIES ----
level_summary(Tier, Level, Total, Pass, Fail) :-
    aggregate_all(count, result(Tier,Level,_,_), Total),
    aggregate_all(count, result(Tier,Level,_,pass), Pass),
    Fail is Total - Pass.
tier_summary(Tier, Total, Pass, Fail) :-
    aggregate_all(count, result(Tier,_,_,_), Total),
    aggregate_all(count, result(Tier,_,_,pass), Pass),
    Fail is Total - Pass.

%% a FAIL that is NOT a known gap = regression.
%% For the ROUND-TRIP tier: a failure on input that DOESN'T PARSE is expected (you can't
%% round-trip what won't parse) — only a RT failure on PARSEABLE input is a real (emit) bug.
%% (medayek: "round-trip failures on parseable input reveal emit bugs.")
unexpected_fail(parse, Level, Case, "unexpected parse regression") :-
    result(parse, Level, Case, fail),
    \+ expected_fail(parse, Case, _).
unexpected_fail(roundtrip, Level, Case, "unexpected EMIT bug (parses but round-trip fails)") :-
    result(roundtrip, Level, Case, fail),
    result(parse, Level, Case, pass),        % it PARSED -> RT failure is a real emit bug
    \+ expected_fail(roundtrip, Case, _).
%% a known gap that now PASSES = fix landed, promote it
unexpected_pass(Tier, Level, Case) :-
    result(Tier, Level, Case, pass), expected_fail(Tier, Case, _).

%% ---- CI GATE ----
gate :-
    ( unexpected_fail(T,L,C,_)
      -> forall(unexpected_fail(T2,L2,C2,R2), format("REGRESSION [~w L~w] ~w: ~w~n",[T2,L2,C2,R2])),
         format("GATE FAIL~n",[]), halt(1)
      ;  ( unexpected_pass(T3,L3,C3)
           -> forall(unexpected_pass(T4,L4,C4), format("PROMOTE [~w L~w] ~w: known gap now PASSES — remove expected_fail~n",[T4,L4,C4])),
              format("GATE PASS (with promotions available)~n",[])
           ;  format("GATE PASS~n",[]) ) ).

%% ---- HUMAN REPORT ----
report :-
    format("=== PARSE tier ===~n",[]),
    forall(between(0,5,L), (level_summary(parse,L,T,P,F), format("  L~w: ~w/~w pass (~w fail)~n",[L,P,T,F]))),
    tier_summary(parse,PT,PP,PF), format("  PARSE TOTAL: ~w/~w (~w fail)~n",[PP,PT,PF]),
    format("=== ROUND-TRIP tier ===~n",[]),
    forall(between(0,5,L), (level_summary(roundtrip,L,T,P,F), format("  L~w: ~w/~w pass (~w fail)~n",[L,P,T,F]))),
    tier_summary(roundtrip,RT,RP,RF), format("  RT TOTAL: ~w/~w (~w fail)~n",[RP,RT,RF]),
    ( unexpected_pass(_,_,_)
      -> format("=== PROMOTIONS (known gaps now passing) ===~n",[]),
         forall(unexpected_pass(T2,L2,C2), format("  [~w L~w] ~w~n",[T2,L2,C2])) ; true ),
    ( unexpected_fail(_,_,_,_)
      -> format("=== REGRESSIONS ===~n",[]),
         forall(unexpected_fail(T3,L3,C3,_), format("  [~w L~w] ~w~n",[T3,L3,C3])) ; true ).
