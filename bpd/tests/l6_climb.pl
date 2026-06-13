%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% l6_climb.pl — LEVEL 6 measurable climb: round-trip M of N builder blocks.
%% Uses metayen preprocess_file_segment_cached/5 (fast) + per-block timeout (deci hangs).
:- use_module(library(time)).
incs(["/tmp/llama_cpp_test/src","/tmp/llama_cpp_test/include","/tmp/llama_cpp_test/ggml/include"]).
bal(_,P,M,_,_):-P>=M,!,fail.
bal(T,P,M,D,R):-sub_string(T,P,1,_,C),N is P+1,
  (C=="{"->D1 is D+1,bal(T,N,M,D1,R)
  ;C=="}"->(D=:=1->R=P;D1 is D-1,bal(T,N,M,D1,R)) ;bal(T,N,M,D,R)).
emit_one(S,Str):-( catch(c_ast:emit_c(S,R0),_,fail)->Str=R0 ; phrase(c_ast:emit_expr(S),C),atom_codes(Str,C) ).
rt_one(S):-emit_one(S,Str),atom_string(SA,Str),c_ast:c_tokenize(SA,T),
  once((phrase(c_ast:parse_stmt_v2(S2),T);phrase(c_ast:parse_stmt(S2),T))), S==S2.

builders([ llama-4475-4697, deci-4699-4868, baichuan-4869-4990, xverse-4991-5103,
           falcon-5104-5227, grok-5228-5389, dbrx-5390-5516, starcoder-5517-5625,
           bert-5725-5911, bloom-5912-6017, mpt-6018-6159, qwen2-6428-6545 ]).

%% returns rt(Arch, Verdict) where Verdict = full | part(N,NR,NOK) | timeout | error
block(Arch, S, E, rt(Arch, V)) :-
    incs(Incs),
    ( catch(call_with_time_limit(6, (
        ( current_predicate(c_preprocess:preprocess_file_segment_cached/5)
          -> c_preprocess:preprocess_file_segment_cached("/tmp/llama_cpp_test/src/llama-model.cpp", Incs, range(S,E), Text, _)
          ;  c_preprocess:preprocess_file_segment("/tmp/llama_cpp_test/src/llama-model.cpp", Incs, range(S,E), Text, _) ),
        sub_string(Text,P0,_,_,") : llm_graph_context"),
        sub_string(Text,P0,_,_,Tl), sub_string(Tl,BO,1,_,"{"), BodyOpen is P0+BO+1,
        string_length(Text,L), bal(Text,BodyOpen,L,1,Close), Len is Close-BodyOpen,
        sub_string(Text,BodyOpen,Len,_,Body), atom_string(BA,Body),
        c_ast:c_tokenize(BA,Toks), c_ast:parse_stmts_v2_greedy(Stmts,Toks,Rest),
        length(Stmts,N), length(Rest,NR),
        findall(i,(member(St,Stmts),catch(rt_one(St),_,fail)),OKs), length(OKs,NOK)
      )), time_limit_exceeded, V = timeout)
      -> ( var(V) -> ( N>0, NR=:=0, NOK=:=N -> V = full ; V = part(N,NR,NOK) ) ; true )
      ;  V = error ).

run :-
    builders(Bs), length(Bs, NB),
    format("=== L6 CLIMB: round-trip ~w builder blocks (cached preproc, 6s/block) ===~n",[NB]),
    findall(R, (member(A-S-E, Bs), block(A,S,E,R)), Rs),
    findall(A, member(rt(A,full),Rs), Full), length(Full, NFull),
    forall(member(rt(A,V),Rs), format("  ~w: ~w~n",[A,V])),
    format("=== SUMMARY: ~w / ~w builder blocks FULLY round-trip ===~n",[NFull,NB]),
    format("    full: ~w~n",[Full]),
    ( current_predicate(c_preprocess:preprocess_cache_stats/1)
      -> c_preprocess:preprocess_cache_stats(St), format("    cache: ~w~n",[St]) ; true ).
