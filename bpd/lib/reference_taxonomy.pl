%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% reference_taxonomy.pl — the DECISION TAXONOMY of reference implementations + Pareto-substitution.
%% (Iyun, 2026-05-29, Heath.) Each reference implementation is a SET OF DECISIONS at coordinates.
%% We can PRINT the catalog/decision-tree of what each reference chose ("they chose butterfly
%% accumulation", "they chose weave hd=64"), QUANTIFY each choice on (ULP-from-best, performance),
%% and SELECT from our library the semantic-equivalents that are AT-LEAST-AS-GOOD numerically AND
%% AT-LEAST-AS-FAST (Pareto-dominating substitutions).

:- module(reference_taxonomy,
    [ reference_decision/3, decision_cost/3, semantic_equivalent/2,
      pareto_substitution/4, tradeoff_substitution/6, offer_for/2, catalog/1, decision_tree/2 ]).
:- use_module(library(lists)).

%% ── 1. THE CATALOG: what each reference CHOSE at each coordinate (decision-tree) ──
%% reference_decision(Reference, Coordinate, Choice). Reference in {ollama_ggml, huggingface, ...}.
reference_decision(ollama_ggml,  [attn, score],        accumulation(sequential)).
reference_decision(ollama_ggml,  [attn, qkv],          rope_weave(interleave, hd(64))).
reference_decision(ollama_ggml,  [attn, kv_cache],     kv_dtype(f16)).
reference_decision(ollama_ggml,  [attn, kv_cache],     roundtrip(f32, f16, f32)).
reference_decision(huggingface,  [attn, score],        accumulation(sequential)).
reference_decision(huggingface,  [attn, qkv],          rope_weave(split_half, hd(64))).
reference_decision(huggingface,  [attn, kv_cache],     kv_dtype(f16)).

%% ── 2. QUANTIFY each choice: (ULP-from-best-precision, performance) ──
%% decision_cost(Choice, ULPFromBest, Performance). ULPFromBest = ULP deviation from the numerically
%% optimal choice (0 = best precision). Performance = relative throughput (1.0 = baseline; >1 faster).
%% Grounded where MEASURED; predicted (labeled) otherwise.
decision_cost(accumulation(sequential),      ulp_from_best(32),  perf(1.00)).   % MEASURED 32 ULP @ score
decision_cost(accumulation(butterfly),       ulp_from_best(0),   perf(0.95)).   % predicted: best precision, ~5% slower
decision_cost(accumulation(kahan),           ulp_from_best(1),   perf(0.85)).   % predicted: near-best, slower
decision_cost(kv_dtype(f16),                 ulp_from_best(high),perf(1.00)).   % f16 round-trip = lossy
decision_cost(kv_dtype(f32),                 ulp_from_best(0),   perf(0.98)).   % lossless, 2x memory, slightly slower
decision_cost(roundtrip(f32,f16,f32),        ulp_from_best(high),perf(1.00)).   % the lossy round-trip itself
decision_cost(rope_weave(interleave, hd(64)),ulp_from_best(0),   perf(1.00)).   % correct-for-ggml (0 ULP vs target)
decision_cost(rope_weave(split_half, hd(64)),ulp_from_best(0),   perf(1.00)).   % correct-for-hf (0 ULP vs HF)

%% ── 3. OUR LIBRARY: semantic equivalents (same math, different implementation) ──
%% semantic_equivalent(Choice, Alternative). The substitution preserves SEMANTICS (same function).
semantic_equivalent(accumulation(sequential), accumulation(butterfly)).
semantic_equivalent(accumulation(sequential), accumulation(kahan)).
semantic_equivalent(kv_dtype(f16),            kv_dtype(f32)).
semantic_equivalent(roundtrip(f32,f16,f32),   kv_dtype(f32)).   % skip the round-trip = keep f32

%% ── 4. PARETO-SUBSTITUTION: at-least-as-good numerically AND at-least-as-fast ──
%% pareto_substitution(Reference, Coordinate, Original, Better) — Better dominates Original:
%% ULPFromBest(Better) =< ULPFromBest(Original)  AND  perf(Better) >= perf(Original).
%% (strict domination on at least one axis; here >= on both = at-least-as-good both ways.)
pareto_substitution(Ref, Coord, Orig, Better) :-
    reference_decision(Ref, Coord, Orig),
    semantic_equivalent(Orig, Better),
    decision_cost(Orig,   ulp_from_best(U0), perf(P0)),
    decision_cost(Better, ulp_from_best(U1), perf(P1)),
    ulp_le(U1, U0),            % at-least-as-good numerically
    P1 >= P0.                  % at-least-as-fast
%% ULP comparison (numbers, plus the symbolic 'high')
ulp_le(_, high) :- !.                       % anything <= high
ulp_le(high, _) :- !, fail.                 % high <= number only if number is high (handled above)
ulp_le(A, B) :- number(A), number(B), A =< B.

%% ── TRADE-OFF substitutions: better on ONE axis, with the cost on the other NAMED ──
%% Most real improvements are trade-offs, not Pareto wins. tradeoff_substitution(Ref, Coord, Orig,
%% Better, Gain, Cost) — honestly offers the improvement WITH its cost (precision-for-speed or vice versa).
tradeoff_substitution(Ref, Coord, Orig, Better, Gain, Cost) :-
    reference_decision(Ref, Coord, Orig),
    semantic_equivalent(Orig, Better),
    decision_cost(Orig,   ulp_from_best(U0), perf(P0)),
    decision_cost(Better, ulp_from_best(U1), perf(P1)),
    \+ pareto_substitution(Ref, Coord, Orig, Better),   % not a free win
    ( ( ulp_le(U1,U0), U1 \== U0 ) ; P1 > P0 ),         % strictly better on SOME axis
    Gain = gain(precision(U0->U1), speed(P0->P1)),
    Cost = cost(precision(U0->U1), speed(P0->P1)).

%% offer_for(Coordinate, Offers) — the full offer at a coordinate: pareto wins + trade-offs, honestly.
offer_for(Coord, Offers) :-
    findall(pareto(Ref,O,B), pareto_substitution(Ref,Coord,O,B), Pareto),
    findall(tradeoff(Ref,O,B,G,C), tradeoff_substitution(Ref,Coord,O,B,G,C), Trade),
    append(Pareto, Trade, Offers).

%% ── RENDERING: the catalog + the decision-tree per reference ──
catalog(Catalog) :-
    findall(decision(Ref, Coord, Choice, ulp(U), perf(P)),
            ( reference_decision(Ref, Coord, Choice),
              ( decision_cost(Choice, ulp_from_best(U), perf(P)) -> true ; U = unknown, P = unknown ) ),
            Catalog).
decision_tree(Ref, Tree) :-
    findall(Coord-Choice, reference_decision(Ref, Coord, Choice), Tree).
