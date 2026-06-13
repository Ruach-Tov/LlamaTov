%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_fusion_reduction_gate.pl — proves the reduction-order gate catches the rms->quant bug class.
:- use_module('lib/numerical_stability.pl').
:- initialization(main).

%% declare the contract the gate checks (mirrors norm_softmax_from_facts.pl)
:- assertz(numerical_stability:reduction_order(rms_ss, lanes(256), strided, tree(pairwise, 8))).

main :-
    style_check(-discontiguous),
    format("=== fusion reduction-order gate tests ===~n", []),

    %% T1: bit_exact rms fold WITHOUT attestation -> must VETO (the bug we hit).
    ( numerical_stability:fusion_reduction_gate(fusion(rms_quant, [rms_norm, quant], bit_exact), bit_exact)
    ->  format("T1 FAIL: gate let an un-attested bit_exact rms fold through~n", []), R1 = fail
    ;   format("T1 PASS: un-attested bit_exact rms fold VETOED (caught the serial-vs-block_row class)~n", []), R1 = ok ),

    %% T2: same fold WITH the order-preservation attestation -> must PASS.
    assertz(numerical_stability:reduction_order_preserved(rms_quant, rms_norm,
              reduction_order(rms_ss, lanes(256), strided, tree(pairwise, 8)))),
    ( numerical_stability:fusion_reduction_gate(fusion(rms_quant, [rms_norm, quant], bit_exact), bit_exact)
    ->  format("T2 PASS: attested fold (preserves lanes(256) pairwise tree) ALLOWED~n", []), R2 = ok
    ;   format("T2 FAIL: gate vetoed a properly-attested fold~n", []), R2 = fail ),

    %% T3: a fusion folding NO reduction-bearing op -> PASS (nothing to preserve).
    ( numerical_stability:fusion_reduction_gate(fusion(silu_mul, [silu, mul], bit_exact), bit_exact)
    ->  format("T3 PASS: non-reduction fold (silu*mul) ALLOWED~n", []), R3 = ok
    ;   format("T3 FAIL: gate vetoed a non-reduction fold~n", []), R3 = fail ),

    %% T4: a NON-bit_exact rms fold (reordering DECLARED) -> PASS (the reorder is honest).
    ( numerical_stability:fusion_reduction_gate(fusion(rms_approx, [rms_norm, quant], approx), approx)
    ->  format("T4 PASS: non-bit_exact (declared-reorder) fold ALLOWED~n", []), R4 = ok
    ;   format("T4 FAIL: gate vetoed a fold that openly declares reordering~n", []), R4 = fail ),

    ( (R1==ok, R2==ok, R3==ok, R4==ok)
    ->  format("~nRESULTS: 4/4 PASS — the gate catches the bug class and allows honest folds~n", [])
    ;   format("~nRESULTS: FAILURES present~n", []) ),
    halt.
