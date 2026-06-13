# The Inline-Literal Canon

**Date discovered**: 2026-05-18 ~18:43 UTC
**Discovered by**: mavchin (CUPTI deep profiling)
**Captured by**: metayen
**Status**: substrate invariant — preserve, do not refactor away

## The principle

The c_ast substrate emits float constants as inline literals at every
use site:

```prolog
c_binop('*', c_float_f(0.5), c_var(rho))
```

emits as:

```c
0.5f * rho
```

This is the **optimal form** for the better backend (LLVM-based
compilation paths). It is **not** to be "cleaned up" by binding
constants to named locals.

## The mechanism

mavchin's CUPTI deep profiling on Tesla P4 showed that LLVM (used by
CUDA-oxide and similar paths) folds inline constants directly into
fused-multiply-add (FMA) immediate operands:

```
LLVM emit:  fma.f32 r0, r1, 0.5f, r2     (one instruction)
nvcc emit:  mov.f32 r3, 0.5f
            fma.f32 r0, r1, r3, r2       (two instructions)
```

For each constant used in an FMA-eligible context:

- LLVM: zero extra instructions (constant lives in the FMA opcode itself)
- nvcc: one `mov.f32` instruction per use site, requiring a register

Across all 8 transcendental kernels measured, this difference produced
**+23% instructions but 25% better IPC** on the LLVM path. The IPC win
comes from eliminating unnecessary register moves and the pipeline
stalls they cause.

For `k_exp` specifically: LLVM emit had 6 fewer `mov.f32` instructions
than nvcc emit, despite computing the same arithmetic.

## What this means for the substrate

The substrate's natural emit style is **already optimal**. This was
discovered, not designed. But now that we know, we keep it.

**Substrate invariant**: `c_float_f(K)` must stay inline at the use
site. Do not refactor toward "DRY constants" by introducing:

```prolog
c_decl_init(c_type(float), half, c_float_f(0.5)),
c_assign(c_var(x), c_binop('*', c_var(half), c_var(rho)))
```

That form produces:

```c
float half = 0.5f;
x = half * rho;
```

which forces nvcc to materialize `half` in a register via `mov.f32`,
and likely defeats LLVM's FMA folding too (since `half` is now a
named local with arbitrary lifetime, not a literal the compiler can
inline at the use site).

The "DRY constants" instinct is a software-engineering reflex from
languages where constant materialization is free. On GPU silicon
where register pressure and pipeline stalls dominate, **constants are
not free** — and inlining them at every use is the right form.

## Detection: how to spot a regression

If a future commit introduces named constant locals for floats that
appear in arithmetic expressions, it is an IPC regression. Watch for:

- `c_decl_init(c_type(float), <name>, c_float_f(K))` followed by
  `c_var(<name>)` appearing inside any `c_binop` or `c_call` to FMA-
  eligible math functions
- Refactor commits with messages like "deduplicate constants" or
  "extract magic numbers" applied to kernel bodies
- Code review comments suggesting "make the constants symbolic for
  readability"

The substrate-honest response to all of these: **no.** The constants
already have a name (their numeric value); they don't need a Prolog
variable to be readable. The emit form `0.5f * rho` is already as
readable as `half * rho` while being substantively faster on the
silicon.

## Where this might legitimately not apply

There are circumstances where named constants ARE the right form:

1. **Truly variable parameters** (passed as kernel arguments). These
   are `c_var(<name>)` already, not constants.

2. **Compile-time constants used in non-arithmetic contexts** (e.g.,
   array sizes for `c_shared_decl(..., c_int(8))`). These don't enter
   FMA pipelines.

3. **Compile-time expressions involving multiple constants** that the
   compiler will fold anyway (e.g., `c_float_f(0.5)` and
   `c_float_f(1.4)` combined via `c_binop`). The folding happens
   regardless of substrate-side binding.

The invariant applies specifically to **single float constants
appearing in arithmetic expressions inside kernel bodies**. That's
where the FMA immediate operand optimization lives.

## Connection to other substrate-honesty principles

This joins the established methodology accumulation:

- **Comprehension over verbatim** (c_raw is debt, structural c_ast is
  comprehension)
- **Intent vs approximation** (regex captures approximation, AST
  captures intent)
- **Bug-for-bug as comprehension proof** (when subsuming software)
- **Physics-for-physics as correctness proof** (when subsuming a
  domain)
- **Measure, don't assume; align with the subsumption target**
  (medayek's principle from CFD C2.1)
- **Inline-literal canon** (this principle — the substrate's natural
  emit is the optimal emit, do not refactor away)

The pattern across all of them: the substrate's representations should
match the structural and operational nature of what they represent.
The substrate should be able to reason about what it produces, and
when it produces something accidentally optimal, it should be able to
recognize that and preserve it.

## Empirical citation

This principle is grounded in CUPTI hardware event measurements on
Tesla P4 (compute capability 6.1), across 8 transcendental kernels
(k_exp, k_log, k_sin, k_cos, k_tan, k_sinh, k_cosh, k_tanh) compared
across three compilation paths: nvcc, NVRTC, and CUDA-oxide.

The +23% instruction count / -1.6% active cycles / 25% better IPC
delta was consistent across all 8 kernels. The mechanism (FMA
immediate operand folding) was confirmed by instruction-level PTX
inspection.

Future profiling on other architectures (Ampere, Hopper, RDNA, etc.)
may reveal different magnitudes but the principle should hold: any
ISA with fused arithmetic and immediate operands benefits from inline
literals over named constant locals.
