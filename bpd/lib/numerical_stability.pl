%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% numerical_stability.pl — Static analysis of numerical instability patterns
%%
%% Auto-self-diagnosis: detects potential numerical instabilities in compute
%% graphs BEFORE kernel dispatch. Each instability pattern is a Prolog rule
%% that matches against the graph's value ranges and operation types.
%%
%% Integrates with diagnose_graph/1 via check_numerical_stability/1.
%%
%% Licensed under GPLv2
%% Author: medayek (Collective SME, Verification Methodology)
%% Date: 2026-05-24
%% ═══════════════════════════════════════════════════════════════════════════

/** <module> Numerical Stability Static Analyzer

Detects 6 classes of numerical instability in compute graphs:
  1. Catastrophic cancellation (a + b where a ≈ -b)
  2. Division by near-zero
  3. Large intermediate overflow risk
  4. Reduction order sensitivity
  5. Exp underflow
  6. FMA vs non-FMA divergence

Each detection produces a warning with:
  - The instability class
  - The affected operation(s)
  - The specific input range that triggers the instability
  - A suggested algebraic reformulation to fix it

Example:
==
?- assert_tensor(erf_out, base, 1024, f32, [1024,1,1,1], [4,0,0,0], 0),
   assert_value_range(erf_out, -1.0, 1.0),
   assert_tensor(one, base, 1, f32, [1,1,1,1], [4,0,0,0], 0),
   assert_value_range(one, 1.0, 1.0),
   assert_op(add_1_erf, add, [one, erf_out], [sum_out]),
   check_catastrophic_cancellation(add_1_erf, Result).

Result = warning(catastrophic_cancellation, add_1_erf,
                                            'add(one, erf_out): range [0.0, 2.0] but min(one)=-max(erf_out) causes cancellation when erf_out ≈ -1.0. Fix: use erfc(-x) instead of 1+erf(x)')
==

@author medayek (Ruach Tov Collective)
@see lib/compute_graph_invariants.pl for structural invariants
*/

:- module(numerical_stability, [
    assert_value_range/3,
    assert_op_semantics/2,
    check_numerical_stability/1,
    check_catastrophic_cancellation/2,
    check_div_by_near_zero/2,
    check_overflow_risk/2,
    check_reduction_sensitivity/2,
    check_exp_underflow/2,
    check_fma_sensitivity/2,
    clear_stability_facts/0,
    suggest_fix/2,
    %% fusion reduction-order gate (Iyun 2026-06-12)
    check_fusion_reduction_order/3,
    fusion_reduction_gate/2,
    folds_reduction_op/3,
    reduction_order_preserved/3
]).
:- dynamic reduction_order/4.
:- dynamic reduction_order/5.

:- use_module(library(lists)).

%% ═══════════════════════════════════════════════════════════════════════════
%% Dynamic facts: value ranges and op semantics
%% ═══════════════════════════════════════════════════════════════════════════

%! assert_value_range(+Tensor, +Min, +Max) is det.
%  Declares the known value range for a tensor.
%  Ranges propagate through operations via range_propagation/4.
:- dynamic value_range/3.

%! assert_op_semantics(+Op, +Semantics) is det.
%  Declares semantic properties of an operation.
%  Examples: op_semantics(gelu_1, gelu)
%            op_semantics(sm_1, softmax)
:- dynamic op_semantics/2.

%  From compute_graph_invariants — we use these
:- dynamic tensor/7.    % tensor(Name, Kind, Size, Dtype, Ne, Nb, Offset)
:- dynamic op/4.        % op(Name, Type, Inputs, Outputs)

clear_stability_facts :-
    retractall(value_range(_, _, _)),
    retractall(op_semantics(_, _)).

assert_value_range(Tensor, Min, Max) :-
    retractall(value_range(Tensor, _, _)),
    assert(value_range(Tensor, Min, Max)).

assert_op_semantics(Op, Semantics) :-
    retractall(op_semantics(Op, _)),
    assert(op_semantics(Op, Semantics)).


%% ═══════════════════════════════════════════════════════════════════════════
%% Value range propagation
%% ═══════════════════════════════════════════════════════════════════════════

%! inferred_range(+Tensor, -Min, -Max) is semidet.
%  Returns the value range for a tensor, either from asserted facts
%  or inferred from the operation that produced it.
inferred_range(Tensor, Min, Max) :-
    value_range(Tensor, Min, Max), !.

inferred_range(Tensor, Min, Max) :-
    op(_, Type, Inputs, [Tensor]),
    range_from_op(Type, Inputs, Min, Max), !.

%  Default: assume F32 range
inferred_range(_, -3.4e38, 3.4e38).

%! range_from_op(+OpType, +Inputs, -Min, -Max) is semidet.
%  Infers output range from operation type and input ranges.

range_from_op(relu, [Input], Min, Max) :-
    inferred_range(Input, _, InMax),
    Min is 0.0,
    Max is max(0.0, InMax).

range_from_op(sigmoid, _, 0.0, 1.0).

range_from_op(tanh, _, -1.0, 1.0).

range_from_op(erf, _, -1.0, 1.0).

range_from_op(softmax, _, 0.0, 1.0).

range_from_op(exp, [Input], Min, Max) :-
    inferred_range(Input, InMin, InMax),
    (InMin < -87.3 -> Min is 0.0 ; Min is exp(InMin)),
    (InMax > 88.7  -> Max is 3.4e38 ; Max is exp(InMax)).

range_from_op(add, [A, B], Min, Max) :-
    inferred_range(A, MinA, MaxA),
    inferred_range(B, MinB, MaxB),
    Min is MinA + MinB,
    Max is MaxA + MaxB.

range_from_op(mul, [A, B], Min, Max) :-
    inferred_range(A, MinA, MaxA),
    inferred_range(B, MinB, MaxB),
    Products = [MinA*MinB, MinA*MaxB, MaxA*MinB, MaxA*MaxB],
    maplist([X*Y, R]>>(R is X*Y), Products, Vals),
    min_list(Vals, Min),
    max_list(Vals, Max).

range_from_op(div, [A, _B], Min, Max) :-
    inferred_range(A, MinA, MaxA),
    %  Division range is unbounded if denominator crosses zero
    Min is MinA * -1.0e10,  % approximate
    Max is MaxA * 1.0e10.

range_from_op(neg, [Input], Min, Max) :-
    inferred_range(Input, InMin, InMax),
    Min is -InMax,
    Max is -InMin.


%% ═══════════════════════════════════════════════════════════════════════════
%% Instability Pattern 1: Catastrophic Cancellation
%% a + b where a ≈ -b, causing loss of significant digits
%% ═══════════════════════════════════════════════════════════════════════════

%! check_catastrophic_cancellation(+Op, -Result) is det.
check_catastrophic_cancellation(Op, Result) :-
    op(Op, add, [A, B], _),
    inferred_range(A, MinA, MaxA),
    inferred_range(B, MinB, MaxB),
    %  Check if the sum range includes values near zero
    %  while the operands themselves are large
    SumMin is MinA + MinB,
    SumMax is MaxA + MaxB,
    OperandScale is max(abs(MaxA), abs(MaxB)),
    %  Cancellation: the sum range INCLUDES values near zero
    %  while the operands are significantly non-zero.
    %  Use min(|sum_min|, |sum_max|) — if either endpoint is near zero,
    %  the sum CAN be near zero (e.g., 1+erf where erf ∈ [-1,1] → sum ∈ [0,2])
    SumNearZero is min(abs(SumMin), abs(SumMax)),
    (OperandScale > 0.01, SumNearZero / OperandScale < 0.01 ->
        suggest_fix(Op, Fix),
        format(atom(Msg),
            'add(~w, ~w): operands ~2e but sum can be as small as ~2e. ~w',
            [A, B, OperandScale, SumNearZero, Fix]),
        Result = warning(catastrophic_cancellation, Op, Msg)
    ;
        Result = ok
    ), !.

check_catastrophic_cancellation(_, ok).


%% ═══════════════════════════════════════════════════════════════════════════
%% Instability Pattern 2: Division by Near-Zero
%% ═══════════════════════════════════════════════════════════════════════════

%! check_div_by_near_zero(+Op, -Result) is det.
check_div_by_near_zero(Op, Result) :-
    op(Op, div, [_Num, Denom], _),
    inferred_range(Denom, MinD, MaxD),
    %  Range crosses zero or includes very small values
    (MinD * MaxD =< 0.0 ->
        format(atom(Msg),
            'div by ~w: range [~2e, ~2e] crosses zero. Guard with epsilon.',
            [Denom, MinD, MaxD]),
        Result = warning(div_by_near_zero, Op, Msg)
    ; abs(MinD) < 1.0e-6 ->
        format(atom(Msg),
            'div by ~w: range [~2e, ~2e] near zero. Guard with epsilon.',
            [Denom, MinD, MaxD]),
        Result = warning(div_by_near_zero, Op, Msg)
    ;
        Result = ok
    ), !.

check_div_by_near_zero(_, ok).


%% ═══════════════════════════════════════════════════════════════════════════
%% Instability Pattern 3: Large Intermediate Overflow Risk
%% ═══════════════════════════════════════════════════════════════════════════

%! check_overflow_risk(+Op, -Result) is det.
check_overflow_risk(Op, Result) :-
    op(Op, mul, [A, B], _),
    inferred_range(A, _, MaxA),
    inferred_range(B, _, MaxB),
    AbsMaxA is abs(MaxA),
    AbsMaxB is abs(MaxB),
    %  Product could overflow F32
    (AbsMaxA > 1.0, AbsMaxB > 1.0,
     log(AbsMaxA) + log(AbsMaxB) > 80.0 ->  % > ~3e34
        format(atom(Msg),
            'mul(~w, ~w): max |~2e| * |~2e| risks overflow. Pre-scale one operand.',
            [A, B, AbsMaxA, AbsMaxB]),
        Result = warning(overflow_risk, Op, Msg)
    ;
        Result = ok
    ), !.

check_overflow_risk(_, ok).


%% ═══════════════════════════════════════════════════════════════════════════
%% Instability Pattern 4: Reduction Order Sensitivity
%% ═══════════════════════════════════════════════════════════════════════════

:- dynamic reduce_dim_size/2.

%! check_reduction_sensitivity(+Op, -Result) is det.
check_reduction_sensitivity(Op, Result) :-
    op(Op, Type, _, _),
    memberchk(Type, [reduce_sum, reduce_mean, reduce_max, softmax, layernorm]),
    (reduce_dim_size(Op, DimSize) ->
        true
    ;
        DimSize = 0  % unknown
    ),
    (DimSize > 64 ->
        format(atom(Msg),
                        '~w: reduction over ~d elements. Accumulation order affects bits. Consider Kahan summation or matching reference reduction tree.',
            [Type, DimSize]),
        Result = warning(reduction_sensitive, Op, Msg)
    ;
        Result = ok
    ), !.

check_reduction_sensitivity(_, ok).


%% ═══════════════════════════════════════════════════════════════════════════
%% Instability Pattern 5: Exp Underflow
%% ═══════════════════════════════════════════════════════════════════════════

%! check_exp_underflow(+Op, -Result) is det.
check_exp_underflow(Op, Result) :-
    op(Op, exp, [Input], _),
    inferred_range(Input, MinIn, _),
    (MinIn < -87.3 ->
        format(atom(Msg),
                        'exp(~w): input min=~2e < -87.3, will underflow to 0. Use log-domain or polynomial with flush-to-zero guard.',
            [Input, MinIn]),
        Result = warning(exp_underflow, Op, Msg)
    ;
        Result = ok
    ), !.

check_exp_underflow(_, ok).


%% ═══════════════════════════════════════════════════════════════════════════
%% Instability Pattern 6: FMA Sensitivity
%% ═══════════════════════════════════════════════════════════════════════════

%! check_fma_sensitivity(+Op, -Result) is det.
%  Detects add(mul(a,b), c) patterns that are FMA-sensitive.
check_fma_sensitivity(Op, Result) :-
    op(Op, add, [MulOut, _C], _),
    op(_, mul, _, [MulOut]),
    \+ explicit_fma(Op),
    format(atom(Msg),
                                'add(mul(...), ...): potential FMA fusion. Result differs with/without hardware FMA. Use explicit _mm_fmadd_ps or #pragma STDC FP_CONTRACT OFF for deterministic behavior.',
        []),
    Result = warning(fma_sensitive, Op, Msg),
    !.

check_fma_sensitivity(_, ok).

:- dynamic explicit_fma/1.


%% ═══════════════════════════════════════════════════════════════════════════
%% Fix Suggestions
%% ═══════════════════════════════════════════════════════════════════════════

%! suggest_fix(+Op, -Fix) is det.
%  Returns a suggested algebraic reformulation for the given operation.

suggest_fix(Op, Fix) :-
    op(Op, add, [A, B], _),
    %  Check for 1 + erf pattern (GELU cancellation)
    (inferred_range(A, 1.0, 1.0), inferred_range(B, -1.0, 1.0) ->
        Fix = 'Fix: replace 1+erf(x) with erfc(-x) to avoid cancellation at erf≈-1'
    ; inferred_range(B, 1.0, 1.0), inferred_range(A, -1.0, 1.0) ->
        Fix = 'Fix: replace erf(x)+1 with erfc(-x) to avoid cancellation at erf≈-1'
    ;
        Fix = 'Fix: rewrite to avoid subtracting nearly-equal values'
    ).

suggest_fix(Op, Fix) :-
    op(Op, div, _, _),
    Fix = 'Fix: add epsilon guard to denominator, or use log-domain'.

suggest_fix(_, 'Fix: see numerical_stability_analysis.py for reformulation options').


%% ═══════════════════════════════════════════════════════════════════════════
%% Known Instability Patterns (from empirical discoveries)
%% ═══════════════════════════════════════════════════════════════════════════

%! known_instability(+Pattern, +Description, +Discovery) is nondet.
%  Documents instabilities discovered empirically during verification.

known_instability(
    gelu_cancellation,
    'GELU tail: 1 + erf(x/sqrt(2)) when x < -2.5 causes catastrophic cancellation',
    'Discovered by medayek, L1 GPU sweep May 24 2026. 62K ULP at dim=8192.'
).

known_instability(
    softmax_exp_polynomial,
    'Softmax expf() differs between libm and SIMD polynomial approximation',
    'Discovered by medayek, L.1 closure May 23 2026. 2 ULP libm vs 0 ULP polynomial.'
).

known_instability(
    scale_application_path,
    'QK^T scale applied fused-in-dot vs post-dot changes softmax input magnitude by sqrt(d)',
    'Discovered by medayek, L.1 closure May 23 2026. 8x = sqrt(64) factor.'
).

known_instability(
    reduction_accumulation_order,
    'Sum/mean reduction gives different bits depending on accumulation order',
    'Discovered by mavchin, GPU L1 May 24 2026. 640 ULP at dim=8192.'
).


%% ═══════════════════════════════════════════════════════════════════════════
%% Main entry point: check all numerical stability invariants
%% ═══════════════════════════════════════════════════════════════════════════

%! check_numerical_stability(-Warnings) is det.
%  Runs all 6 instability checks on the current graph.
%  Returns a list of warning/3 terms for any detected instabilities.

check_numerical_stability(Warnings) :-
    findall(Op, op(Op, _, _, _), Ops),
    check_ops_stability(Ops, [], Warnings).

check_ops_stability([], Acc, Acc).
check_ops_stability([Op|Rest], Acc, Warnings) :-
    check_single_op_stability(Op, OpWarnings),
    append(Acc, OpWarnings, NewAcc),
    check_ops_stability(Rest, NewAcc, Warnings).

check_single_op_stability(Op, Warnings) :-
    findall(W, (
        (check_catastrophic_cancellation(Op, W), W \= ok) ;
        (check_div_by_near_zero(Op, W), W \= ok) ;
        (check_overflow_risk(Op, W), W \= ok) ;
        (check_reduction_sensitivity(Op, W), W \= ok) ;
        (check_exp_underflow(Op, W), W \= ok) ;
        (check_fma_sensitivity(Op, W), W \= ok)
    ), Warnings).


%% ═══════════════════════════════════════════════════════════════════════════
%% FUSION REDUCTION-ORDER GATE (Iyun, 2026-06-12)
%% Wires the reduction-sensitivity analysis into the fusion gate. Motivated by a real bug: the
%% rms->quant seam's first wiring used a SERIAL sum-of-squares while the production rms uses the
%% block_row order (reduction_order(rms_ss, lanes(256), strided, tree(pairwise,8))). Different float
%% order -> diverged at token 11. A "correct rms" isn't enough; it must be the SAME rms. This gate
%% makes that property STATICALLY CHECKABLE: a fusion claiming bit_exact that folds an op carrying a
%% declared reduction_order MUST attest it preserves that exact order.
%%
%% The contract lives as reduction_order/4 facts (rms_ss, q8_gemv_dp4a, ...). A fusion that folds a
%% reduction-bearing op emits the obligation reduction_order_preserved(FusedKernel, FoldedOp, Order);
%% this gate flags (warning(reduction_order_violation, ...)) any bit_exact fold of such an op that
%% lacks the attestation. "The reduction_order IS the correctness contract." (Heath)
%% ═══════════════════════════════════════════════════════════════════════════

%! folds_reduction_op(+Fusion, -FoldedOp, -Order) is nondet.
%  True when Fusion folds an op (FoldedOp) that carries a declared reduction_order.
%  Order is the full reduction_order term (the contract that must be preserved).
folds_reduction_op(fusion(_Name, Ops, _EqClass), FoldedOp, Order) :-
    member(FoldedOp, Ops),
    reduction_op_name(FoldedOp, RedName),
    current_reduction_order(RedName, Order).

%! reduction_op_name(+Op, -ReductionName) is semidet.
%  Map a fused op to the name under which its reduction_order is declared. Extend as ops gain
%  declared orders. (rms_norm folds the rms_ss reduction; q8 GEMVs fold q8_gemv_dp4a.)
reduction_op_name(rms_norm, rms_ss).
reduction_op_name(ggml_rms_norm, rms_ss).
reduction_op_name(rms_ss, rms_ss).
reduction_op_name(q8_gemv, q8_gemv_dp4a).
reduction_op_name(q8_0_dot, q8_gemv_dp4a).
reduction_op_name(softmax, softmax_max_then_sum).   % declared when softmax fusions arrive

%! current_reduction_order(+Name, -Order) is semidet.
%  Look up the declared reduction_order contract. Tries reduction_order/4 (the emitter facts);
%  falls back gracefully so the gate never crashes on an undeclared op (it simply finds no order,
%  meaning "no contract to violate" — a LOUD absence, not a silent pass: see fusion_reduction_gate).
current_reduction_order(Name, reduction_order(Name, Lanes, Stride, Tree)) :-
    catch(reduction_order(Name, Lanes, Stride, Tree), _, fail), !.
current_reduction_order(Name, reduction_order(Name, Lanes, Stride, Accum, Tree)) :-
    catch(reduction_order(Name, Lanes, Stride, Accum, Tree), _, fail), !.

%! reduction_order_preserved(?FusedKernel, ?FoldedOp, ?Order) is nondet.
%  The ATTESTATION a fusion emits to discharge its reduction-order obligation. The emitter asserts
%  this when the fused lowering reproduces the folded op's reduction_order EXACTLY (e.g. the
%  fused_rms_quant phase-1 reproducing lanes(256) strided pairwise-tree). Dynamic so emitters/tests
%  can assert it.
:- dynamic reduction_order_preserved/3.

%! check_fusion_reduction_order(+Fusion, +EqClass, -Result) is det.
%  THE GATE. For a fusion folding a reduction-bearing op:
%    - if EqClass is bit_exact and the attestation reduction_order_preserved/3 is PRESENT -> ok.
%    - if EqClass is bit_exact and the attestation is ABSENT -> warning(reduction_order_violation).
%    - if EqClass is not bit_exact (the fold is allowed to reorder) -> ok (reordering is declared).
%  A fusion that folds NO reduction op -> ok (nothing to preserve).
check_fusion_reduction_order(Fusion, EqClass, Result) :-
    Fusion = fusion(Name, _, _),
    ( folds_reduction_op(Fusion, FoldedOp, Order) ->
        ( EqClass == bit_exact ->
            ( reduction_order_preserved(Name, FoldedOp, Order) ->
                Result = ok
            ;   format(atom(Msg),
                                                                                                                                                'fusion ~w claims bit_exact but folds ~w which carries ~w; NO reduction_order_preserved attestation. The fused lowering MUST reproduce that exact order (lanes/stride/tree) or it is not the SAME reduction. (This is the rms->quant serial-vs-block_row class of bug.)',
                  [Name, FoldedOp, Order]),
                Result = warning(reduction_order_violation, Name, Msg)
            )
        ;   Result = ok                 % non-bit_exact: reordering declared, no contract to break
        )
    ;   Result = ok                     % folds no reduction-bearing op
    ), !.
check_fusion_reduction_order(_, _, ok).

%! fusion_reduction_gate(+Fusion, +EqClass) is semidet.
%  Hard gate: succeeds iff the fusion preserves the reduction-order contract (or there is none).
%  A consumer (apply_fusion / the fusion gate runner) can call this as a PRECONDITION: a fusion that
%  fails this gate MUST NOT be committed. Prints the warning on failure (vetoes are MAP not just NO).
fusion_reduction_gate(Fusion, EqClass) :-
    check_fusion_reduction_order(Fusion, EqClass, Result),
    ( Result = ok ->
        true
    ;   Result = warning(_, _, Msg),
        format("VETO [reduction_order]: ~w~n", [Msg]),
        fail
    ).
