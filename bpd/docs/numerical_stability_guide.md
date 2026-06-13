# Numerical Stability Static Analyzer

## What It Is

`check_numerical_stability/1` is a Prolog-based static analyzer that detects
numerical instabilities in compute graphs **before any kernel runs**. It
examines the structure and value ranges of operations in a compute graph and
reports potential precision hazards — places where floating-point rounding,
cancellation, or overflow could produce different results on different
hardware, compilers, or execution orders.

It is part of the BPD substrate's self-diagnosis infrastructure, alongside
`check_dtype_coherence/3` (type safety), `check_scale_coherence/3` (scale
conventions), and CUPTI profiling (performance). Together, these give the
substrate the ability to reason about its own correctness, precision, and
performance.

## Why It Exists

During the L.1 end-to-end closure investigation (May 2026), we traced a
wrong token prediction through six root causes. Each root cause was a
numerical precision issue that could have been detected statically:

| Root Cause | ULP Impact | Stability Pattern |
|-----------|------------|-------------------|
| Unscaled QK^T (8× factor) | total | `overflow_risk` |
| GELU tail (1 + erf ≈ 0) | 62,277 | `catastrophic_cancellation` |
| Reduction accumulation order | 640 | `reduction_sensitive` |
| libm vs polynomial expf | 2 | `fma_sensitive` |
| Softmax masked positions | 0 | `exp_underflow` |
| F16 conversion algorithm | 4,094 | (dtype, not stability) |

The analyzer was built so that future compute graphs would be diagnosed
automatically — no 38-hour bisection needed.

## How To Use It

### Basic Usage (Prolog)

```prolog
:- use_module('lib/numerical_stability').

% 1. Clear any previous graph facts
clear_stability_facts.

% 2. Declare value ranges for input tensors
assert_value_range(my_input, -10.0, 10.0).
assert_value_range(const_one, 1.0, 1.0).

% 3. Assert the operations in your graph
assert(op(erf_op, erf, [my_input], [erf_out])).
assert(op(add_op, add, [const_one, erf_out], [sum_out])).
assert(op(exp_op, exp, [shifted_scores], [exp_out])).
assert_value_range(shifted_scores, -200.0, 0.0).

% 4. For reductions, declare the dimension size
assert(reduce_dim_size(sum_op, 4096)).

% 5. Run the stability check
check_numerical_stability(Warnings).

% 6. Inspect the results
% Warnings is a list of warning(Class, Op, Message) terms
```

### Integration with diagnose_graph/1

The stability checker integrates with the existing graph diagnosis pipeline:

```prolog
diagnose_graph_full(Report) :-
    check_all_invariants(StructuralDiags),    % dtype, views, strides
    check_numerical_stability(StabilityWarns), % precision hazards
    append(StructuralDiags, StabilityWarns, AllDiags),
    format_report(AllDiags, Report).
```

### From Python

```python
import subprocess

result = subprocess.run(
    ['swipl', '-g',
     'use_module("lib/numerical_stability"), '
     'assert_value_range(x, -10, 10), '
     'assert(op(relu_op, relu, [x], [y])), '
     'check_numerical_stability(W), '
     'write(W), halt'],
    capture_output=True, text=True)
print(result.stdout)  # [] if stable, [warning(...), ...] if not
```

## Understanding the Output

Each warning is a `warning(Class, Op, Message)` term:

```
warning(catastrophic_cancellation, add_op,
  'add(const_one, erf_out): operands 1.0e+00 but sum can be as small
   as 0.0e+00. Fix: replace 1+erf(x) with erfc(-x) to avoid
   cancellation at erf≈-1')
```

### The Six Warning Classes

#### 1. `catastrophic_cancellation`

**What:** Adding two values that nearly cancel (a + b where a ≈ -b).

**Why it matters:** When the result is near zero but the operands are
large, most significant bits are lost. A 1 ULP error in either operand
becomes a huge relative error in the result.

**Real example:** GELU computes `1 + erf(x/√2)`. When x < -2.5,
erf ≈ -1, so the sum ≈ 0. A 1 ULP error in erf produces 62,277 ULP
in the final GELU output.

**How to fix:** Algebraic reformulation.
- `1 + erf(x)` → `erfc(-x)` (computed directly, no subtraction)
- `a - b` where a ≈ b → rearrange to compute the difference directly

#### 2. `div_by_near_zero`

**What:** Dividing by a value whose range includes zero or near-zero.

**Why it matters:** Division by near-zero amplifies any error in the
numerator to infinity. Results become hardware-dependent.

**Real example:** Softmax divides by `sum(exp(x))`. If all logits are
very negative (masked), the sum can be near zero.

**How to fix:**
- Add epsilon guard: `a / (b + 1e-7)`
- Use log-domain: `log_softmax = x - log(sum(exp(x)))` via log-sum-exp
- Clamp denominator: `a / max(b, epsilon)`

#### 3. `overflow_risk`

**What:** Multiplying values whose product could exceed F32 range.

**Why it matters:** F32 max is ~3.4e38. If intermediate products
overflow, the result is ±inf, and all downstream computation is garbage.

**Real example:** Our QK^T computation without the 1/√d scale produced
scores 8× too large. While not overflowing F32, the magnified values
changed softmax behavior significantly.

**How to fix:**
- Pre-scale one operand: apply `1/√d` to Q or K before the matmul
- Use mixed precision: compute in F64 then downcast
- Split the computation into smaller steps

#### 4. `reduction_sensitive`

**What:** Summing many floating-point values where the accumulation
order affects the result.

**Why it matters:** Floating-point addition is not associative.
`(a + b) + c ≠ a + (b + c)` in general. Different hardware (CPU vs GPU),
different compilers, and different parallelization strategies produce
different accumulation orders, giving different results.

**Real example:** GPU sum_reduce with dim=8192 diverges by 640 ULP from
CPU because PyTorch uses a 2D block reduction tree on GPU (splitting
across warps) while CPU accumulates sequentially.

**How to fix:**
- Match the reference implementation's exact reduction order
- Use Kahan compensated summation (error ≤ 2 ULP regardless of order)
- Use pairwise reduction (O(log n) error growth instead of O(n))
- Accept as a characterized substrate-design parameter

#### 5. `exp_underflow`

**What:** Computing exp(x) where x < -87.3 (F32 underflow to zero).

**Why it matters:** Values that should be "very small but nonzero"
become exactly zero, which can cause division-by-zero downstream or
change the semantics of masking operations.

**Real example:** Causal softmax masks future positions with -inf.
exp(-inf) = 0 is correct, but exp(-88) also produces 0, which may
not be the intended behavior for "very unlikely but not impossible."

**How to fix:**
- Guard with a threshold: `exp(max(x, -87.0))`
- Use log-domain computation to avoid materializing tiny values
- Use a polynomial approximation with explicit flush-to-zero semantics

#### 6. `fma_sensitive`

**What:** An `add(mul(a,b), c)` pattern where the compiler may or may
not fuse into a single FMA (fused multiply-add) instruction.

**Why it matters:** FMA computes `a*b + c` with a single rounding step
(one rounding at the end), while separate mul+add has two rounding steps
(one after mul, one after add). The results differ by up to 1 ULP.

**Real example:** Our softmax polynomial evaluation uses Horner form
(`((c5*x + c4)*x + c3)*x + ...`). Each step is a potential FMA. The
ggml binary was compiled with different FMA behavior than ours,
producing 2 ULP on specific values.

**How to fix:**
- Use explicit FMA intrinsics: `_mm_fmadd_ps(a, b, c)` (guarantees FMA)
- Use `#pragma STDC FP_CONTRACT OFF` (guarantees NO FMA)
- Match the reference compiler's FMA behavior

## How To Extend It

### Adding a New Pattern

1. **Identify the pattern empirically.** Run your kernel, compare against
   a reference, find the divergence, trace the root cause to a specific
   floating-point operation.

2. **Write the Prolog detection rule.** Each rule takes an Op identifier
   and returns either `ok` or `warning(Class, Op, Message)`:

```prolog
%! check_my_new_pattern(+Op, -Result) is det.
check_my_new_pattern(Op, Result) :-
    op(Op, OpType, Inputs, _Outputs),
    %% Your detection logic here:
    %% - Check op type
    %% - Check input value ranges via inferred_range/3
    %% - Check structural patterns (e.g., op feeding into another op)
    (detected_condition ->
        format(atom(Msg), 'Description of the instability. Fix: ...', []),
        Result = warning(my_new_class, Op, Msg)
    ;
        Result = ok
    ).
```

3. **Add it to `check_single_op_stability/2`:**

```prolog
check_single_op_stability(Op, Warnings) :-
    findall(W, (
        (check_catastrophic_cancellation(Op, W), W \= ok) ;
        (check_div_by_near_zero(Op, W), W \= ok) ;
        %% ... existing checks ...
        (check_my_new_pattern(Op, W), W \= ok)     %% ADD HERE
    ), Warnings).
```

4. **Add a `known_instability/3` fact documenting the discovery:**

```prolog
known_instability(
    my_new_class,
    'Description of the instability pattern',
    'Discovered by AGENT, DATE. EMPIRICAL_EVIDENCE.'
).
```

5. **Add a fix suggestion to `suggest_fix/2`** if applicable.

6. **Write a test case in `test_numerical_stability.py`** that verifies
   your pattern is correctly detected.

### Adding Value Range Propagation for New Op Types

The `range_from_op/4` predicate infers output ranges from input ranges.
To support a new operation:

```prolog
range_from_op(my_op, [Input], Min, Max) :-
    inferred_range(Input, InMin, InMax),
    %% Compute output range from input range
    Min is f(InMin),
    Max is f(InMax).
```

Common patterns:
- Clipping ops: `range_from_op(clamp, _, ClampMin, ClampMax).`
- Monotonic ops: propagate min/max directly
- Non-monotonic ops: compute range from critical points

### Adding Semantic Annotations

For patterns that depend on the mathematical MEANING of an operation
(not just its type), use `assert_op_semantics/2`:

```prolog
assert_op_semantics(my_gelu_add, gelu_one_plus_erf).

check_gelu_cancellation(Op, Result) :-
    op_semantics(Op, gelu_one_plus_erf),
    Result = warning(catastrophic_cancellation, Op,
        'GELU 1+erf pattern: use erfc(-x) reformulation').
```

## File Locations

| File | Purpose |
|------|---------|
| `lib/numerical_stability.pl` | The analyzer (Prolog module) |
| `lib/compute_graph_invariants.pl` | Structural invariants (dtype, views, strides) |
| `lib/dispatch.pl` | Kernel dispatch with mandatory invariant checking |
| `tests/test_numerical_stability.py` | Test suite (8 test cases) |
| `docs/numerical_stability_analysis.py` | Design document |

## Relationship to Other Substrate Components

```
                    ┌─────────────────────────┐
                    │   Scientist's Intent     │
                    │   "Run GELU on this"     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Compute Graph          │
                    │   (Prolog facts)         │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼────────┐ ┌──────▼───────┐ ┌───────▼────────┐
    │ Structural        │ │ Numerical    │ │ Performance    │
    │ Invariants        │ │ Stability    │ │ Profiling      │
    │ (dtype, views,    │ │ (this module)│ │ (CUPTI from    │
    │  strides, scale)  │ │              │ │  Prolog)       │
    └─────────┬────────┘ └──────┬───────┘ └───────┬────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   dispatch/3            │
                    │   (blocks on errors,    │
                    │    warns on hazards)    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Kernel Execution      │
                    │   (verified correct,    │
                    │    precision-aware,     │
                    │    performance-tuned)   │
                    └─────────────────────────┘
```

The substrate doesn't just run kernels. It understands them.
