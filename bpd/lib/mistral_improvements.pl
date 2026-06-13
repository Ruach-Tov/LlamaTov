%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% mistral_improvements.pl — quantified, provable IMPROVEMENTS offered at measured coordinates.
%% (Iyun, 2026-05-29, Heath.) The divergence map is a provable ontogeny of reference implementations;
%% at each measured coordinate we can OFFER an improvement = an addressed delta with a CONTRACT
%% (stability | precision | speed), gated by acceptance. contribution(Id, Coordinate, Delta, Contract).

:- module(mistral_improvements, [ improvement/4, improvement_for/2, offer/3 ]).
:- use_module(library(lists)).

%% improvement(Id, Coordinate, Delta, Contract). Coordinate = coordinate(Path, Kind, Tags).
%% Delta = the transform offered. Contract = contract(Kind, Claim, Proof) where Kind in
%% stability|precision|speed, Claim is the quantified improvement, Proof is how it is verified.

%% ── IMPROVEMENT 1: lossless KV-cache (Heath's example) ──
%% Reference does {f32->f16 ; kv-cache(f16) ; f16->f32} — a LOSSY round-trip (every cached k/v is
%% f16-quantized). Offer {kv-cache(f32)} — keep the cache in f32. Trades memory for precision.
improvement(kv_cache_f32,
    coordinate([mistral, attn, kv_cache], op, [precision]),
    delta(replace('{f32->f16; kv_cache(f16); f16->f32}', '{kv_cache(f32)}')),
    contract(precision,
             claim('eliminates f16 round-trip quantization of every cached key/value; ULP error of the f16 round-trip removed'),
             proof('measure ULP(f16-roundtrip kv) vs f32 kv across a prompt -> the eliminated error is the quantified delta; cost = 2x kv-cache memory'))).

%% ── IMPROVEMENT 2: butterfly accumulation (Heath's example) ──
%% Sequential accumulation grows rounding error O(n) and DENORMALIZES small terms against a large
%% running sum. Butterfly (pairwise/tree) accumulation is O(log n) error and keeps magnitudes
%% comparable -> minimum denormalization energy. Offer at every REDUCTION coordinate.
improvement(butterfly_accumulation,
    coordinate([mistral, layer(l), attn, score], reduction, [stability, precision]),
    delta(replace(sequential_accumulation, butterfly_accumulation)),
    contract(stability,
             claim('softmax/attention reduction: O(n) sequential error -> O(log n) tree error; minimizes denormalization (small terms not lost against large running sum)'),
             proof('the score cell is MEASURED at 32 ULP (sequential softmax accumulation, ref=pytorch); butterfly accumulation predicted to reduce it -> re-measure with butterfly reduction; the ULP reduction is the quantified delta'))).

%% butterfly also applies to the other reductions (the same delta, different coordinate):
improvement(butterfly_attn_accumulate,
    coordinate([mistral, layer(l), attn, attn_v], reduction, [stability, precision]),
    delta(replace(sequential_accumulation, butterfly_accumulation)),
    contract(stability,
             claim('attention value-accumulation (softmax @ v): O(n)->O(log n) error, minimum denormalization energy'),
             proof('cascade test showed o_proj residual ~1000 ULP (accumulation-order); butterfly attn-accumulate predicted to reduce it -> re-measure'))).

%% ── the OFFER interface: what improvement(s) can we offer at a coordinate, and the contract ──
improvement_for(Path, Improvements) :-
    findall(imp(Id, Delta, Contract),
            improvement(Id, coordinate(Path, _, _), Delta, Contract), Improvements).

%% offer(Coordinate, Contract, Claim) — render an offer for a coordinate.
offer(Path, ContractKind, Claim) :-
    improvement(_, coordinate(Path, _, _), _, contract(ContractKind, Claim, _)).
