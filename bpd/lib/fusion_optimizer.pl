%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% fusion_optimizer.pl — Trivial kernel fusion by graph rewriting
%%
%% Given a sequence of elementwise operations on tensors,
%% detect and eliminate unnecessary DRAM round-trips.
%%
%% The core insight: if tensor T is written by kernel K1 and
%% read ONLY by kernel K2, and both are elementwise, they can
%% be fused into a single kernel that never materializes T.

:- module(fusion_optimizer, [
    fuse_chain/2,          % +OpChain, -FusedOps
    estimate_savings/2,    % +OpChain, -SavedBytes
    demonstrate/0
]).

%% An operation: op(Name, Inputs, Output, Expr)
%% Inputs and Output are tensor names.
%% Expr is the computation.

%% ═══════════════════════════════════════
%% Fusion rules
%% ═══════════════════════════════════════

%% Rule 1: Two consecutive elementwise ops where the first's output
%% is the second's only consumer → fuse into one kernel.
fuse_pair(op(_, Ins1, Mid, Expr1), op(_, [Mid|Rest], Out, Expr2),
          op(fused, Ins1Merged, Out, compose(Expr1, Expr2))) :-
    %% Merge input lists, removing the intermediate
    append(Ins1, Rest, Ins1Merged).

%% Rule 2: Identity (tmp = x) followed by use of tmp → eliminate tmp
fuse_pair(op(copy, [X], Tmp, identity), op(Name, Ins, Out, Expr),
          op(Name, InsFixed, Out, Expr)) :-
    select(Tmp, Ins, InsRest),
    InsFixed = [X|InsRest].

%% ═══════════════════════════════════════
%% Chain fusion: repeatedly apply fusion rules
%% ═══════════════════════════════════════

fuse_chain([], []).
fuse_chain([Op], [Op]).
fuse_chain([Op1, Op2 | Rest], Fused) :-
    (fuse_pair(Op1, Op2, Merged) ->
        fuse_chain([Merged | Rest], Fused)
    ;
        Fused = [Op1 | RestFused],
        fuse_chain([Op2 | Rest], RestFused)
    ).

%% ═══════════════════════════════════════
%% Cost model: how many bytes saved by fusion
%% ═══════════════════════════════════════

%% Each eliminated intermediate saves 2 × N × 4 bytes (write + read)
count_intermediates([], 0).
count_intermediates([_], 0).
count_intermediates([op(_,_,Out,_), Op2 | Rest], Count) :-
    Op2 = op(_,Ins,_,_),
    (member(Out, Ins) ->
        count_intermediates([Op2|Rest], SubCount),
        Count is SubCount + 1
    ;
        count_intermediates([Op2|Rest], Count)
    ).

estimate_savings(Chain, SavedBytes) :-
    count_intermediates(Chain, Intermediates),
    %% Assume N=1048576 (1M elements), 4 bytes each, read+write = 8 bytes
    SavedBytes is Intermediates * 1048576 * 8.

%% ═══════════════════════════════════════
%% Demonstration: pessimize then optimize
%% ═══════════════════════════════════════

demonstrate :-
    format("═══ FUSION OPTIMIZER DEMO ═══~n~n"),
    
    %% Level 0: Already optimal
    _Optimal = [op(add_relu, [a, b], y, relu(add(a, b)))],
    format("Level 0 (optimal):    ~w ops~n", [1]),
    
    %% Level 1: Split into add + relu
    Level1 = [
        op(add,  [a, b], tmp1, add(a, b)),
        op(relu, [tmp1], y,    relu(tmp1))
    ],
    fuse_chain(Level1, Fused1),
    length(Level1, L1Before), length(Fused1, L1After),
    format("Level 1 (add+relu):   ~w ops → ~w ops (fused)~n", [L1Before, L1After]),
    
    %% Level 2: Add useless copies
    Level2 = [
        op(copy, [a],    tmp_a, identity),
        op(copy, [b],    tmp_b, identity),
        op(add,  [tmp_a, tmp_b], tmp1, add(tmp_a, tmp_b)),
        op(relu, [tmp1], y,      relu(tmp1))
    ],
    fuse_chain(Level2, Fused2),
    length(Level2, L2Before), length(Fused2, L2After),
    format("Level 2 (+copies):    ~w ops → ~w ops (fused)~n", [L2Before, L2After]),
    
    %% Level 3: Chain of 5 pointless ops
    Level3 = [
        op(copy, [a],    t1, identity),
        op(neg,  [t1],   t2, neg(t1)),
        op(neg,  [t2],   t3, neg(t2)),
        op(add,  [t3, b], t4, add(t3, b)),
        op(relu, [t4],   y,  relu(t4))
    ],
    fuse_chain(Level3, Fused3),
    length(Level3, L3Before), length(Fused3, L3After),
    format("Level 3 (+neg+neg):   ~w ops → ~w ops (fused)~n", [L3Before, L3After]),
    
    %% Savings — estimate_savings/2 returns bytes; divide once to get MB.
    estimate_savings(Level1, SB1), S1 is SB1 / (1024*1024),
    estimate_savings(Level2, SB2), S2 is SB2 / (1024*1024),
    estimate_savings(Level3, SB3), S3 is SB3 / (1024*1024),
    format("~nDRAM savings (N=1M):~n"),
    format("  Level 1: ~1f MB saved~n", [S1]),
    format("  Level 2: ~1f MB saved~n", [S2]),
    format("  Level 3: ~1f MB saved~n", [S3]),
    
    format("~nThe optimizer should recover Level 0 from ALL levels.~n").
