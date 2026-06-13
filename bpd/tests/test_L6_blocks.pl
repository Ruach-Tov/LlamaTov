%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_L6_blocks.pl — LEVEL 6: round-trip MULTIPLE builder BLOCKS (Iyun + Heath, 2026-05-29)
%%
%% Ladder: L0 literals -> L1 ops -> L2 stmts -> L3 composed exprs -> L4 compound stmts
%%   -> L5 single builder LINES -> L6 multiple builder BLOCKS (this) -> L7 entire file.
%% "It makes it into a measurable climb. That is often all we need to summit." — Heath
%%
%% L6 round-trips each llm_build_<arch> body via metayen post-cpp segment (comments stripped —
%% raw bodies HANG the greedy parser on comment-glued tokens). Reports M-of-N + per-arch.
%% The per-arch breakdown directly answers "can we round-trip mistral/qwen/gemma?" — the gate
%% for running those Ollama models on our dispatch.
%%
%% Invoke: swipl -q -g "use_module(lib/c_ast),use_module(lib/c_preprocess),consult(tests/test_L6_blocks),run,halt"

incs(["/tmp/llama_cpp_test/src","/tmp/llama_cpp_test/include","/tmp/llama_cpp_test/ggml/include"]).
src("/tmp/llama_cpp_test/src/llama-model.cpp").

%% builder start lines (consecutive -> each range is [Start, NextStart-1])
builders([ llama-4475, deci-4699, baichuan-4869, xverse-4991, falcon-5104,
           grok-5228, dbrx-5390, starcoder-5517, refact-5626, bert-5725,
           bloom-5912, mpt-6018, qwen-6312, qwen2-6428, gemma-8164 ]).

bal(_,P,M,_,_):-P>=M,!,fail.
bal(T,P,M,D,R):-sub_string(T,P,1,_,C),N is P+1,
  (C=="{"->D1 is D+1,bal(T,N,M,D1,R)
  ;C=="}"->(D=:=1->R=P;D1 is D-1,bal(T,N,M,D1,R))
  ;bal(T,N,M,D,R)).
emit_one(S,Str):-( catch(c_ast:emit_c(S,R0),_,fail)->Str=R0 ; phrase(c_ast:emit_expr(S),C),atom_codes(Str,C) ).
rt_one(S):-emit_one(S,Str),atom_string(SA,Str),c_ast:c_tokenize(SA,T),
  once((phrase(c_ast:parse_stmt_v2(S2),T);phrase(c_ast:parse_stmt(S2),T))), S==S2.

:- use_module(library(time)).

%% per-block TIMEOUT (Heath/medayek): a hanging builder reports rt(Arch, timeout) — a
%% measured data point — instead of stalling the whole climb. 8s budget per block.
block_rt(Arch, Start, End, Result) :-
    src(Src), incs(Incs),
    ( catch(
        call_with_time_limit(8, (
          c_preprocess:preprocess_file_segment(Src, Incs, range(Start,End), Text, _),
          sub_string(Text,P0,_,_,") : llm_graph_context"),
          sub_string(Text,P0,_,_,Tl), sub_string(Tl,BO,1,_,"{"), BodyOpen is P0+BO+1,
          string_length(Text,L), bal(Text,BodyOpen,L,1,Close), Len is Close-BodyOpen,
          sub_string(Text,BodyOpen,Len,_,Body), atom_string(BA,Body),
          c_ast:c_tokenize(BA,Toks), c_ast:parse_stmts_v2_greedy(Stmts,Toks,Rest),
          length(Stmts,N), length(Rest,NR),
          findall(i,(member(S,Stmts),catch(rt_one(S),_,fail)),OKs), length(OKs,NOK)
        )),
        Err,
        ( Err == time_limit_exceeded -> Result = rt(Arch, timeout) ; Result = rt(Arch, error) ) )
      -> ( var(Result) -> Result = rt(Arch,N,NR,NOK) ; true )
      ;  Result = rt(Arch, fail) ).

ranges([], _, []).
ranges([A-S], _, [A-S-Last]) :- !, Last is S + 250.   % last builder: assume <=250 lines
ranges([A-S, B-S2 | T], _, [A-S-E | Rest]) :- E is S2 - 1, ranges([B-S2|T], _, Rest).

classify(rt(A,N,NR,NOK), A, Tag, Detail) :- integer(N), !,
    ( N>0, NR=:=0, NOK=:=N -> Tag = full ; Tag = part ),
    format(atom(Detail), "~w stmts, ~w toks left, rt ~w/~w", [N,NR,NOK,N]).
classify(rt(A, Status), A, Status, "") :- atom(Status).

run :-
    builders(Bs), ranges(Bs, _, R),
    length(R, NB), format("=== LEVEL 6: round-trip ~w builder blocks (post-cpp, 8s/block) ===~n",[NB]),
    findall(Res, (member(A-S-E, R), block_rt(A, S, E, Res)), Results),
    findall(A, (member(Res,Results), classify(Res,A,full,_)), FullArchs), length(FullArchs,NFull),
    forall(member(Res,Results),
        ( classify(Res, A, Tag, Detail),
          format("    [~w] ~w  ~w~n",[Tag,A,Detail]) )),
    format("=== L6 SUMMARY: ~w / ~w builder blocks FULLY round-trip ===~n",[NFull,NB]),
    format("    full: ~w~n",[FullArchs]).
