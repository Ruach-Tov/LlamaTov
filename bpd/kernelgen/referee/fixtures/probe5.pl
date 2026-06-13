%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- consult("bpd/lib/numerical_stability.pl").
:- initialization(main).
main :-
    % declare the reduction-order contract for rms_ss (the gate looks this up)
    assertz(numerical_stability:reduction_order(rms_ss, lanes(256), strided, tree(pairwise,8))),
    % 1. fusion folding rms_norm, claims bit_exact, NO attestation -> must VETO
    Bad = fusion(bad_rms_fold, [rms_norm], quant),
    ( numerical_stability:fusion_reduction_gate(Bad, bit_exact)
      -> format("violating fold ACCEPTED -> GATE-BROKEN~n")
      ;  format("violating fold VETOED -> GATE-OK~n") ),
    % 2. with the attestation present -> must ACCEPT
    numerical_stability:current_reduction_order(rms_ss, Order),
    assertz(numerical_stability:reduction_order_preserved(bad_rms_fold, rms_norm, Order)),
    ( numerical_stability:fusion_reduction_gate(Bad, bit_exact)
      -> format("attested fold ACCEPTED -> GATE-OK~n")
      ;  format("attested fold VETOED -> GATE-BROKEN~n") ),
    % 3. tolerance-class (reordering declared) -> must ACCEPT without attestation
    Tol = fusion(tol_fold, [rms_norm], quant),
    ( numerical_stability:fusion_reduction_gate(Tol, tolerance(0.000001))
      -> format("tolerance fold ACCEPTED -> GATE-OK~n")
      ;  format("tolerance fold VETOED -> GATE-BROKEN~n") ),
    % 4. fusion folding NO reduction op -> must ACCEPT (nothing to preserve)
    None = fusion(elementwise_fold, [silu_mul], quant),
    ( numerical_stability:fusion_reduction_gate(None, bit_exact)
      -> format("no-reduction fold ACCEPTED -> GATE-OK~n")
      ;  format("no-reduction fold VETOED -> GATE-BROKEN~n") ),
    halt.
