# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Numerical instability static analysis — design for auto-self-diagnosis.

Detectable patterns (all discoverable via static analysis of the compute graph):

1. CATASTROPHIC CANCELLATION: a + b where a ≈ -b
   Pattern: add(X, Y) where range(X) ∩ range(-Y) ≠ ∅
   Examples:
     - 1 + erf(x) when x < -2.5 (GELU tail)
     - a - a + epsilon (residual connections with small corrections)
     - softmax normalization: exp(x-max)/sum where most terms ≈ 0
   Fix: algebraic reformulation (erfc, log-sum-exp, etc.)
   
   Prolog rule:
     catastrophic_cancel(Op) :-
         op_type(Op, add),
         source(Op, 0, A), source(Op, 1, B),
         value_range(A, MinA, MaxA),
         value_range(B, MinB, MaxB),
         MinA + MaxB < epsilon,  % a + b can be near zero
         MaxA + MinB > -epsilon, % while |a|, |b| are large
         abs(MaxA) > 100 * epsilon.

2. DIVISION BY NEAR-ZERO: a / b where b → 0
   Pattern: div(X, Y) where 0 ∈ range(Y)
   Examples:
     - softmax: exp(x) / sum(exp(x)) when sum → 0 (all masked)
     - layer norm: x / sqrt(var + eps) when var → 0
     - 1/sum_exp in our softmax kernel
   Fix: guard with epsilon, use log-domain computation
   
   Prolog rule:
     div_by_near_zero(Op) :-
         op_type(Op, div),
         source(Op, 1, Denom),
         value_range(Denom, Min, Max),
         Min * Max =< 0.  % range crosses zero

3. LARGE INTERMEDIATE VALUES: a * b where |result| >> max_float32
   Pattern: mul(X, Y) where max(|X|)*max(|Y|) > 3.4e38
   Examples:
     - unscaled QK^T with large head_dim (our scale_application_path!)
     - gradient scaling in backprop
   Fix: pre-scale one operand
   
   Prolog rule:
     overflow_risk(Op) :-
         op_type(Op, mul),
         source(Op, 0, A), source(Op, 1, B),
         value_range(A, _, MaxA),
         value_range(B, _, MaxB),
         MaxA * MaxB > 1.0e30.

4. REDUCTION ORDER SENSITIVITY: sum of many terms
   Pattern: reduce_sum(X, dim) where dim_size > threshold
   Examples:
     - matmul inner dimension (K > 1024)
     - softmax denominator sum
     - layer norm mean/variance
   Fix: Kahan summation, pairwise reduction, or match reference order
   
   Prolog rule:
     reduction_sensitive(Op) :-
         op_type(Op, reduce_sum),
         reduce_dim_size(Op, DimSize),
         DimSize > 64.  % threshold for accumulation order to matter

5. LOSS OF SIGNIFICANCE IN EXP: exp(x) where x << -88
   Pattern: exp(X) where min(X) < -88 (underflow to 0)
   Examples:
     - softmax with large logit differences
     - GELU tail exp(-x²) for |x| > 9
   Fix: guard, use log-domain, or polynomial approximation with flush-to-zero
   
   Prolog rule:
     exp_underflow(Op) :-
         op_type(Op, exp),
         source(Op, 0, X),
         value_range(X, Min, _),
         Min < -87.3.  % expf underflows below this

6. FMA vs NON-FMA DIVERGENCE: a*b + c
   Pattern: add(mul(A,B), C) — compiler may or may not fuse to FMA
   Examples:
     - dot products (each a += x[i]*y[i] is a potential FMA)
     - polynomial evaluation (Horner form)
     - our erf polynomial
   Fix: explicit FMA intrinsics or explicit non-FMA (#pragma STDC FP_CONTRACT OFF)
   
   Prolog rule:
     fma_sensitive(Op) :-
         op_type(Op, add),
         source(Op, 0, MulOp),
         op_type(MulOp, mul),
         % The add and mul could be fused — platform-dependent behavior
         not(explicit_fma(Op)).

IMPLEMENTATION PLAN:
====================

Phase 1: Value range propagation
  - Each tensor gets a (min, max) range annotation
  - Ranges propagate through the graph: add ranges, mul ranges, etc.
  - Leaf tensors: range from data statistics or dtype bounds

Phase 2: Pattern matching
  - Run each instability pattern against the annotated graph
  - Report warnings with severity levels:
    WARNING: potential instability (detected by range analysis)
    ERROR: confirmed instability (detected by empirical divergence)

Phase 3: Auto-fix suggestions
  - For each detected pattern, suggest the algebraic reformulation
  - e.g., add(1, erf(x)) → erfc(-x) when range(erf(x)) includes -1

Phase 4: Integration with dispatch/3
  - diagnose_graph/1 already runs before dispatch
  - Add numerical_stability check alongside dtype_coherence, scale_coherence

PROLOG INTEGRATION:
===================

check_numerical_stability(Graph, Report) :-
    findall(W, catastrophic_cancel_warning(Graph, W), CancelWarnings),
    findall(W, div_by_near_zero_warning(Graph, W), DivWarnings),
    findall(W, overflow_risk_warning(Graph, W), OverflowWarnings),
    findall(W, reduction_sensitive_warning(Graph, W), ReductionWarnings),
    append([CancelWarnings, DivWarnings, OverflowWarnings, ReductionWarnings], AllWarnings),
    (AllWarnings == [] -> 
        Report = stable
    ;
        Report = warnings(AllWarnings)
    ).
"""
