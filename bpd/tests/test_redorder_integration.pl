%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_module('lib/numerical_stability.pl').
:- use_module('lib/apply_fusion.pl').
:- use_module('kernelgen/emitters/fused_rms_quant.pl').
:- assertz(numerical_stability:reduction_order(rms_ss, lanes(256), strided, tree(pairwise, 8))).
:- initialization(main).
main :-
    style_check(-discontiguous),
    format("=== reduction-order gate integration ===~n", []),
    %% Minimal facts for a rms_quant fusion (op_kind etc. — apply may need them, but the PRECONDITION
    %% runs first and is what we test). We only check the precondition gating here.
    F = fusion(rms_quant, [rms_norm, quant], bit_exact),
    %% (1) BEFORE attestation: precondition must REFUSE.
    ( apply_fusion:fusion_reduction_precondition(F)
    -> format("T1 FAIL: un-attested rms_quant fusion was allowed~n", []), R1 = fail
    ;  format("T1 PASS: un-attested rms_quant fusion REFUSED by precondition~n", []), R1 = ok ),
    %% (2) Emit the fused kernel -> auto-attests (discharges the obligation).
    fused_rms_quant:attest_rms_quant_reduction_order,
    %% (3) AFTER attestation: precondition must PASS.
    ( apply_fusion:fusion_reduction_precondition(F)
    -> format("T2 PASS: after attestation, rms_quant fusion ALLOWED~n", []), R2 = ok
    ;  format("T2 FAIL: attested fusion still refused~n", []), R2 = fail ),
    ( (R1==ok, R2==ok) -> format("~nRESULTS: 2/2 PASS — gate folded into apply_fusion + emitter attests~n", [])
    ; format("~nRESULTS: FAILURES~n", []) ),
    halt.
