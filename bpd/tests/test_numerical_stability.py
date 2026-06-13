#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Test the numerical stability checker against known instabilities.

Verifies that the Prolog static analyzer correctly detects each
instability pattern we've discovered empirically.

Author: medayek
"""
import subprocess, sys, os, tempfile

PROLOG_TEST = r"""
:- use_module('../lib/numerical_stability').

%% Test 1: GELU catastrophic cancellation
%% 1 + erf(x) where erf range is [-1, 1]
test_gelu_cancellation :-
    clear_stability_facts,
    assert_value_range(erf_out, -1.0, 1.0),
    assert_value_range(const_one, 1.0, 1.0),
    assert(op(gelu_add, add, [const_one, erf_out], [gelu_sum])),
    check_catastrophic_cancellation(gelu_add, Result),
    (Result = warning(catastrophic_cancellation, _, _) ->
        write('PASS  test_gelu_cancellation: detected'), nl
    ;
        write('FAIL  test_gelu_cancellation: not detected'), nl
    ).

%% Test 2: Division by near-zero
%% softmax: exp(x) / sum where sum can be near zero
test_div_near_zero :-
    clear_stability_facts,
    assert_value_range(numerator, 0.0, 1.0),
    assert_value_range(denominator, -0.001, 1.0),
    assert(op(sm_div, div, [numerator, denominator], [sm_out])),
    check_div_by_near_zero(sm_div, Result),
    (Result = warning(div_by_near_zero, _, _) ->
        write('PASS  test_div_near_zero: detected'), nl
    ;
        write('FAIL  test_div_near_zero: not detected'), nl
    ).

%% Test 3: Safe division (should NOT trigger)
test_safe_div :-
    clear_stability_facts,
    assert_value_range(num2, 0.0, 1.0),
    assert_value_range(denom2, 0.5, 2.0),
    assert(op(safe_div, div, [num2, denom2], [safe_out])),
    check_div_by_near_zero(safe_div, Result),
    (Result = ok ->
        write('PASS  test_safe_div: correctly not triggered'), nl
    ;
        write('FAIL  test_safe_div: false positive'), nl
    ).

%% Test 4: Exp underflow
test_exp_underflow :-
    clear_stability_facts,
    assert_value_range(large_neg, -200.0, 10.0),
    assert(op(exp_op, exp, [large_neg], [exp_out])),
    check_exp_underflow(exp_op, Result),
    (Result = warning(exp_underflow, _, _) ->
        write('PASS  test_exp_underflow: detected'), nl
    ;
        write('FAIL  test_exp_underflow: not detected'), nl
    ).

%% Test 5: Reduction sensitivity
test_reduction_sensitivity :-
    clear_stability_facts,
    assert(op(big_sum, reduce_sum, [big_input], [sum_out])),
    assert(reduce_dim_size(big_sum, 4096)),
    check_reduction_sensitivity(big_sum, Result),
    (Result = warning(reduction_sensitive, _, _) ->
        write('PASS  test_reduction_sensitivity: detected'), nl
    ;
        write('FAIL  test_reduction_sensitivity: not detected'), nl
    ).

%% Test 6: Small reduction (should NOT trigger)
test_small_reduction :-
    clear_stability_facts,
    assert(op(small_sum, reduce_sum, [small_input], [small_out])),
    assert(reduce_dim_size(small_sum, 32)),
    check_reduction_sensitivity(small_sum, Result),
    (Result = ok ->
        write('PASS  test_small_reduction: correctly not triggered'), nl
    ;
        write('FAIL  test_small_reduction: false positive'), nl
    ).

%% Test 7: FMA sensitivity
test_fma_sensitivity :-
    clear_stability_facts,
    assert(op(inner_mul, mul, [a_vec, b_vec], [mul_out])),
    assert(op(accum_add, add, [mul_out, running_sum], [new_sum])),
    check_fma_sensitivity(accum_add, Result),
    (Result = warning(fma_sensitive, _, _) ->
        write('PASS  test_fma_sensitivity: detected'), nl
    ;
        write('FAIL  test_fma_sensitivity: not detected'), nl
    ).

%% Test 8: Full stability check
test_full_check :-
    clear_stability_facts,
    assert_value_range(erf2, -1.0, 1.0),
    assert_value_range(one2, 1.0, 1.0),
    assert_value_range(neg_input, -200.0, 10.0),
    assert(op(cancel_add, add, [one2, erf2], [cancel_out])),
    assert(op(bad_exp, exp, [neg_input], [exp_out2])),
    assert(op(big_reduce, reduce_sum, [data], [reduced])),
    assert(reduce_dim_size(big_reduce, 8192)),
    check_numerical_stability(Warnings),
    length(Warnings, N),
    format('PASS  test_full_check: ~d warnings detected~n', [N]),
    forall(member(W, Warnings),
        (W = warning(Class, Op, _Msg),
         format('       ~w: ~w~n', [Class, Op]))).

:- initialization((
    test_gelu_cancellation,
    test_div_near_zero,
    test_safe_div,
    test_exp_underflow,
    test_reduction_sensitivity,
    test_small_reduction,
    test_fma_sensitivity,
    test_full_check,
    halt
)).
"""

def main():
    # Write test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(test_dir, '_test_stability.pl')
    
    with open(test_path, 'w') as f:
        f.write(PROLOG_TEST)
    
    try:
        result = subprocess.run(
            ['swipl', '-g', 'true', test_path],
            capture_output=True, text=True, timeout=30,
            cwd=test_dir)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:500])
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("SKIP: swipl not available")
    except subprocess.TimeoutExpired:
        print("FAIL: timeout")
        sys.exit(1)
    finally:
        if os.path.exists(test_path):
            os.unlink(test_path)


if __name__ == "__main__":
    main()
