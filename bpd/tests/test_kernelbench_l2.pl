%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernelbench_l2.pl — Run fusion analyzer on all 100 KernelBench L2 problems.
%%
%% Split: odd problems = TRAIN (derive rules), even = TEST (validate generalization)
%% For each problem, report: which chains found, which rules fired.
%% Aggregate: rule coverage matrix (which rules impact which problems).
%%
%% GOAL: every fusion rule impacts multiple problems in BOTH splits.
%% No single-problem rules. Mutation analysis: removing any rule
%% should break multiple test cases.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/fusion_analyzer').
:- use_module('kernelbench_l2_problems').

:- dynamic rule_fires/3.  % rule_fires(Rule, Split, ProblemNum)
:- dynamic total_problems/1, problems_with_fusion/1.
total_problems(0). problems_with_fusion(0).

run_all :-
    write('═══════════════════════════════════════════════════════'), nl,
    write('KernelBench L2: Fusion Analysis on 100 Problems'), nl,
    write('═══════════════════════════════════════════════════════'), nl, nl,
    
    %% Run all problems
    forall(kb_problem(Num, Split, Name, Ops), (
        retract(total_problems(T)), T1 is T + 1, assert(total_problems(T1)),
        analyze_problem(Num, Split, Name, Ops)
    )),
    
    nl,
    write('═══════════════════════════════════════════════════════'), nl,
    write('RULE COVERAGE MATRIX'), nl,
    write('═══════════════════════════════════════════════════════'), nl, nl,
    
    %% Collect unique rules
    findall(R, rule_fires(R, _, _), AllRules),
    sort(AllRules, UniqueRules),
    
    forall(member(Rule, UniqueRules), (
        findall(N, rule_fires(Rule, train, N), TrainProbs),
        findall(N, rule_fires(Rule, test, N), TestProbs),
        length(TrainProbs, TrainCount),
        length(TestProbs, TestCount),
        Total is TrainCount + TestCount,
        ( TrainCount >= 2, TestCount >= 2 ->
            Mark = '✅ GENERAL'
        ; Total >= 2 ->
            Mark = '⚠️  SPARSE'
        ;
            Mark = '❌ SINGLE'
        ),
        format("  ~w ~w: TRAIN=~d TEST=~d (total=~d)~n", 
               [Mark, Rule, TrainCount, TestCount, Total])
    )),
    
    nl,
    total_problems(TP), problems_with_fusion(PF),
    format("Problems analyzed: ~d~n", [TP]),
    format("Problems with fusion opportunities: ~d (~d%)~n", 
           [PF, PF * 100 // TP]),
    nl,
    
    %% Check generalization: all rules should appear in BOTH splits
    findall(R, (rule_fires(R, _, _), 
                \+ rule_fires(R, train, _)), TestOnlyRules0),
    sort(TestOnlyRules0, TestOnlyRules),
    findall(R, (rule_fires(R, _, _),
                \+ rule_fires(R, test, _)), TrainOnlyRules0),
    sort(TrainOnlyRules0, TrainOnlyRules),
    
    ( TestOnlyRules = [], TrainOnlyRules = [] ->
        write('✅ ALL RULES GENERALIZE: every rule fires in both splits'), nl
    ;
        ( TestOnlyRules \= [] ->
            format("⚠️  Rules only in TEST (not learned from TRAIN): ~w~n", [TestOnlyRules])
        ; true ),
        ( TrainOnlyRules \= [] ->
            format("⚠️  Rules only in TRAIN (can't validate on TEST): ~w~n", [TrainOnlyRules])
        ; true )
    ).

analyze_problem(Num, Split, _Name, Ops) :-
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C, L), L > 1), Chains, MultiChains),
    length(MultiChains, ChainCount),
    
    ( ChainCount > 0 ->
        retract(problems_with_fusion(PF)), PF1 is PF + 1, 
        assert(problems_with_fusion(PF1)),
        
        %% Record which rules fired
        forall(member(Chain, MultiChains), (
            reverse(Chain, Fwd),
            record_chain_rules(Fwd, Split, Num)
        )),
        
        %% Print summary
        findall(Names, (
            member(Ch, MultiChains), 
            reverse(Ch, FwdCh),
            findall(N, member(op(N,_,_), FwdCh), Names)
        ), AllChainNames),
        format("  ~w #~d: ~d chains ~w~n", [Split, Num, ChainCount, AllChainNames])
    ;
        format("  ~w #~d: no fusion~n", [Split, Num])
    ).

record_chain_rules([], _, _).
record_chain_rules([_], _, _).
record_chain_rules([op(_,K1,_), op(_,K2,_) | Rest], Split, Num) :-
    ( can_fuse(K1, K2, Rule) ->
        ( rule_fires(Rule, Split, Num) -> true ; assert(rule_fires(Rule, Split, Num)) )
    ; true ),
    record_chain_rules([op(_,K2,_) | Rest], Split, Num).

:- initialization((run_all -> halt(0) ; (write('FAILED'), nl, halt(1)))).
