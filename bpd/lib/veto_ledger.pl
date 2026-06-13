%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% veto_ledger.pl — the PROFITABILITY ledger the scanner cites (validity != profitability).
%%
%% WHY (unified review rec E, Bocher): the auto-fuser scanner reports VALID fusion opportunities, but
%% some are known LOSSES — store-side silu is 0.55x when it forces re-quant; rms->quant is e2e-neutral
%% on the graph path. A reader without ledger knowledge would wire a loser. This module is the curated
%% record of MEASURED fusion verdicts (the project's hard-won experience, sourced from ir_param_axes.pl
%% axis_metric facts), so the scanner can ANNOTATE each find with profitable / neutral / anti-fusion.
%%
%% "Validity is what CAN be fused; profitability is what SHOULD be. The diff is the ledger." (Heath)
%%
%% fusion_verdict(+Pattern, -Verdict, -Note):
%%   Verdict = profitable(GainPct) | neutral(GainPct) | anti(Factor) | unknown
%%   Pattern is an op-pattern atom or chain the scanner can match against its finds.
:- module(veto_ledger, [
    fusion_verdict/3,        % +Pattern, -Verdict, -Note
    cite/2,                  % +Pattern, -CitationAtom  (human-readable one-liner)
    anti_fusions/1,          % -ListOfAntiPatterns
    ledger_summary/0
]).

%% ── THE LEDGER (measured, from ir_param_axes.pl + this project's experiments) ──────────────────
%% SHIPPED WINNERS (profitable, in PRODUCTION_PROFILE):
fusion_verdict(add_resid,      profitable(5.9),  'o/down-proj GEMV with fused residual add; eliminates k_add_resid launch; shipped as addres').
fusion_verdict(fixed_carry,    profitable(0.63), 'down-proj fixed output carry; eliminates per-layer memcpy; 24 graph nodes/token; shipped').
fusion_verdict(qk_joined,      profitable(1.13), 'rope over joined Q,K; one rope launch/layer; needs qk contiguous; shipped').
fusion_verdict(silu_quant,     profitable(0.87), 'fold silu(g)*u into the quant (gate/up-compatible); the door the store-side silu veto mapped to; shipped').
fusion_verdict(epilogue_matmul_elementwise, profitable(always), 'matmul + elementwise epilogue: always profitable (the canonical fusion)').

%% NEUTRAL (valid, bit-identical, but kept OFF — no e2e gain on the graph path):
fusion_verdict(rms_quant,      neutral(-0.05),   'fold rms_norm into the quant; born polyglot, 0-ULP, but the fused kernel does the SAME compute as the pair (launch overhead already banked by graph capture); VALID not PROFITABLE; default off').

%% ANTI-FUSIONS (measured LOSSES — do NOT wire; the veto ledger proper):
fusion_verdict(store_side_silu, anti(0.55),      'silu fused on the STORE side forces re-quant -> 0.55x. The veto that MAPPED to the profitable quant-side silu_quant fold. Wire silu_quant, NOT this.').
fusion_verdict(reciprocal_mul_softmax, anti(0.58), 'softmax via shared 1/sum reciprocal: precision-diluted 0.58 (double-rounding), max_ulp 1. Not bit-exact; nearly trivial gain.').

%% Default: unknown -> the scanner says so explicitly (honest absence, not silent green).
fusion_verdict(_, unknown, 'no measured verdict in the ledger — validity only; MEASURE before wiring') :- !.

%% ── citation: a one-line human-readable tag the scanner prints next to a find ──────────────────
cite(Pattern, Cite) :-
    fusion_verdict(Pattern, V, Note), !,
    ( V = profitable(always) -> format(atom(Cite), 'PROFITABLE(always) — ~w', [Note])
    ; V = profitable(G)      -> format(atom(Cite), 'PROFITABLE(+~w%) — ~w', [G, Note])
    ; V = neutral(G)         -> format(atom(Cite), 'NEUTRAL(~w%, kept OFF) — ~w', [G, Note])
    ; V = anti(F)            -> format(atom(Cite), '⚠ ANTI-FUSION(~wx — DO NOT WIRE) — ~w', [F, Note])
    ;                           format(atom(Cite), 'UNKNOWN — ~w', [Note]) ).

anti_fusions(L) :- findall(P, fusion_verdict(P, anti(_), _), L).

ledger_summary :-
    format("═══ FUSION VETO LEDGER (measured verdicts) ═══~n"),
    forall((fusion_verdict(P, V, _), V \= unknown, P \= '_'),
           ( cite(P, C), format("  ~w~t~20|: ~w~n", [P, C]) )).
