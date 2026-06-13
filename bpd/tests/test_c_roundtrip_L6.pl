%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_c_roundtrip_L6.pl — LEVEL 6: full builder-body round-trip (Iyun, 2026-05-29)
%%
%% Separate from the fast self-contained L0-L5 suite (test_c_parse_graduated.pl) because
%% L6 shells out to g++ -E via metayen's c_preprocess:preprocess_file_segment (slow + needs
%% llama.cpp source present). Run on demand:  swipl test_c_roundtrip_L6.pl
%%
%% RESULT (verified 2026-05-29, post unary-sign fix): llm_build_llama round-trips 100% —
%% 21/21 statements parse -> emit -> reparse -> AST-identical, 1350/1350 tokens, 0 unparsed.
%% This is the round-trip-llama.cpp milestone for the llama builder body: lift to AST ->
%% regenerate (modulo pretty-printing) -> prove AST-identity, end to end.
%% Scope honesty: ONE builder (llama), AST<->source round-trip. NOT yet: all 56 builders,
%% nor regenerate-and-diff-against-the-source-FILE, nor mutation-completeness-proof.

%% load deps + call run/0 explicitly: swipl -q -g run -t halt -l this_file (after use_module of c_ast, c_preprocess)

bal(_,P,M,_,_):-P>=M,!,fail.
bal(T,P,M,D,R):-sub_string(T,P,1,_,C),N is P+1,
  (C=="{"->D1 is D+1,bal(T,N,M,D1,R)
  ;C=="}"->(D=:=1->R=P;D1 is D-1,bal(T,N,M,D1,R))
  ;bal(T,N,M,D,R)).
emit_one(S,Str):-( catch(c_ast:emit_c(S,R0),_,fail)->Str=R0 ; phrase(c_ast:emit_expr(S),C),atom_codes(Str,C) ).
rt_one(S):-emit_one(S,Str),atom_string(SA,Str),c_ast:c_tokenize(SA,T),
  once((phrase(c_ast:parse_stmt_v2(S2),T);phrase(c_ast:parse_stmt(S2),T))), S==S2.

run :-
    Src = "/tmp/llama_cpp_test/src/llama-model.cpp",
    ( exists_file(Src)
      -> Incs = ["/tmp/llama_cpp_test/src","/tmp/llama_cpp_test/include","/tmp/llama_cpp_test/ggml/include"],
         c_preprocess:preprocess_file_segment(Src, Incs, range(4475,4697), Text, _),
         sub_string(Text,P0,_,_,") : llm_graph_context"),
         sub_string(Text,P0,_,_,Tl), sub_string(Tl,BO,1,_,"{"), BodyOpen is P0+BO+1,
         string_length(Text,L), bal(Text,BodyOpen,L,1,Close), Len is Close-BodyOpen,
         sub_string(Text,BodyOpen,Len,_,Body), atom_string(BA,Body),
         c_ast:c_tokenize(BA,Toks), c_ast:parse_stmts_v2_greedy(Stmts,Toks,Rest),
         length(Stmts,N), length(Rest,NR),
         findall(i,(member(S,Stmts),catch(rt_one(S),_,fail)),OKs), length(OKs,NOK),
         format("=== LEVEL 6: llm_build_llama full-body round-trip ===~n",[]),
         format("  parsed: ~w statements, ~w tokens unparsed~n",[N,NR]),
         format("  round-trip: ~w/~w statements AST-identical (parse->emit->reparse)~n",[NOK,N]),
         ( NR=:=0, NOK=:=N
           -> format("  L6 PASS: full builder body round-trips 100%%~n",[])
           ;  format("  L6 PARTIAL~n",[]) )
      ;  format("=== LEVEL 6: SKIP (llama.cpp source not at ~w) ===~n",[Src]) ).
