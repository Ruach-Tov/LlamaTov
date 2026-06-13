%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% Round 2: probe-shape corrections — overflow wants MUL (not exp); fma wants add(mul_out,c)
%% with a producing mul op; cancellation probe was exactly ON the 0.01 boundary (strict <).
:- consult("bpd/lib/numerical_stability.pl").
:- initialization(main).
check(Label, Detector, Op, Expect) :-
    G =.. [Detector, Op, Result],
    ( catch(numerical_stability:G, E, (format("~w: EXCEPTION ~w -> BROKEN~n",[Label,E]), fail))
      -> ( Expect == warn, Result = warning(_,_,_) -> format("~w: fired -> DETECTOR-OK~n",[Label])
         ; Expect == ok,   Result == ok            -> format("~w: clean -> DETECTOR-OK~n",[Label])
         ; format("~w: got ~w expected ~w -> DETECTOR-DEAD~n",[Label,Result,Expect]) )
      ; format("~w: failed outright -> BROKEN~n",[Label]) ).
main :-
    %% cancellation: tighter than the boundary (ratio 0.001 < 0.01)
    assertz(numerical_stability:op(cc2, add, [u_a, u_b], [u_o])),
    numerical_stability:assert_value_range(u_a, 0.9995, 1.0),
    numerical_stability:assert_value_range(u_b, -1.0, -0.9995),
    check("cancellation/strict-inside", check_catastrophic_cancellation, cc2, warn),
    %% overflow: MUL of two huge operands (log sum > 80)
    assertz(numerical_stability:op(ov2, mul, [u_h1, u_h2], [u_p])),
    numerical_stability:assert_value_range(u_h1, 0.0, 1.0e20),
    numerical_stability:assert_value_range(u_h2, 0.0, 1.0e20),
    check("overflow/mul(1e20,1e20)", check_overflow_risk, ov2, warn),
    assertz(numerical_stability:op(ov3, mul, [u_s1, u_s2], [u_p2])),
    numerical_stability:assert_value_range(u_s1, 0.0, 10.0),
    numerical_stability:assert_value_range(u_s2, 0.0, 10.0),
    check("overflow/mul(10,10)", check_overflow_risk, ov3, ok),
    %% fma: add whose first input IS a mul output
    assertz(numerical_stability:op(m1, mul, [u_x, u_y], [mul_out_t])),
    assertz(numerical_stability:op(fma2, add, [mul_out_t, u_c], [u_f])),
    check("fma/add(mul_out,c)", check_fma_sensitivity, fma2, warn),
    halt.
