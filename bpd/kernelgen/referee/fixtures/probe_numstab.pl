%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% FAILED-TEST INJECTION: every numerical_stability detector. Uses the module's own
%% public API (assert_value_range, dynamic op/4) — not raw assertz on static preds.
:- consult("bpd/lib/numerical_stability.pl").
:- initialization(main).

check(Label, Detector, Op, Expect) :-
    G =.. [Detector, Op, Result],
    ( catch(numerical_stability:G, E, (format("~w: EXCEPTION ~w -> DETECTOR-BROKEN~n",[Label,E]), fail))
      -> ( Expect == warn, Result = warning(_,_,_) -> format("~w: fired -> DETECTOR-OK~n",[Label])
         ; Expect == ok,   Result == ok            -> format("~w: clean -> DETECTOR-OK~n",[Label])
         ; format("~w: got ~w expected ~w -> DETECTOR-DEAD~n",[Label,Result,Expect]) )
      ; format("~w: detector failed outright -> DETECTOR-BROKEN~n",[Label]) ).

main :-
    %% Pattern 1: catastrophic cancellation (op type 'add', operands a≈-b)
    assertz(numerical_stability:op(cc_bad, add, [t_a, t_b], [t_out])),
    numerical_stability:assert_value_range(t_a, 0.99, 1.0),
    numerical_stability:assert_value_range(t_b, -1.0, -0.99),
    check("cancellation/bad", check_catastrophic_cancellation, cc_bad, warn),
    assertz(numerical_stability:op(cc_ok, add, [t_c, t_d], [t_out2])),
    numerical_stability:assert_value_range(t_c, 1.0, 2.0),
    numerical_stability:assert_value_range(t_d, 3.0, 4.0),
    check("cancellation/benign", check_catastrophic_cancellation, cc_ok, ok),
    %% Pattern 2: div by near zero (op type 'div')
    assertz(numerical_stability:op(div_bad, div, [t_n, t_z], [t_q])),
    numerical_stability:assert_value_range(t_n, 1.0, 10.0),
    numerical_stability:assert_value_range(t_z, -0.001, 0.001),
    check("div-near-zero/bad", check_div_by_near_zero, div_bad, warn),
    assertz(numerical_stability:op(div_ok, div, [t_n2, t_z2], [t_q2])),
    numerical_stability:assert_value_range(t_n2, 1.0, 10.0),
    numerical_stability:assert_value_range(t_z2, 0.5, 2.0),
    check("div-near-zero/benign", check_div_by_near_zero, div_ok, ok),
    %% Pattern 3: overflow risk
    assertz(numerical_stability:op(ov_bad, exp, [t_big], [t_e])),
    numerical_stability:assert_value_range(t_big, 0.0, 200.0),
    check("overflow/exp(200)", check_overflow_risk, ov_bad, warn),
    assertz(numerical_stability:op(ov_ok, exp, [t_sm], [t_e2])),
    numerical_stability:assert_value_range(t_sm, -1.0, 1.0),
    check("overflow/exp(1)", check_overflow_risk, ov_ok, ok),
    %% Pattern 4: reduction sensitivity (needs reduce op + dim size)
    assertz(numerical_stability:op(red_bad, reduce_sum, [t_long], [t_s])),
    assertz(numerical_stability:reduce_dim_size(red_bad, 1000000)),
    numerical_stability:assert_value_range(t_long, -1000.0, 1000.0),
    check("reduction/1M", check_reduction_sensitivity, red_bad, warn),
    assertz(numerical_stability:op(red_ok, reduce_sum, [t_short], [t_s2])),
    assertz(numerical_stability:reduce_dim_size(red_ok, 8)),
    numerical_stability:assert_value_range(t_short, -1.0, 1.0),
    check("reduction/8", check_reduction_sensitivity, red_ok, ok),
    %% Pattern 5: exp underflow
    assertz(numerical_stability:op(eu_bad, exp, [t_neg], [t_u])),
    numerical_stability:assert_value_range(t_neg, -200.0, -100.0),
    check("exp-underflow/bad", check_exp_underflow, eu_bad, warn),
    %% Pattern 6: fma sensitivity
    assertz(numerical_stability:op(fma1, fma, [t_x,t_y,t_z3], [t_f])),
    check("fma/flagged", check_fma_sensitivity, fma1, warn),
    halt.
