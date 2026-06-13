# Within-Target Bit-Identical Baseline

**Date**: 2026-05-17 ~22:50 UTC
**Originating conversation**: Heath's reframe of the k_gelu ULP-374 finding.

## The principle

> "Within the choice of {CPU|GPU} it should be possible to get
> physics-exact, bit-identical output (ULP=0) for all kernels, because
> in theory we should be able to generate identical control flow on
> the hardware."
> — Heath, 2026-05-17

This document names the principle that organizes the cross-language
correctness matrix's tighter contract.

**Within-target bit-identicality** is the substrate's baseline claim:
when two host languages dispatch to the same execution target (CPU or
GPU) and call the same primitive operations in the same order, the
hardware MUST produce identical bit patterns.

fp32 IEEE-754 arithmetic is deterministic. Multiply, add, FMA: every
chip implementing IEEE-754 produces the same result for the same
inputs in the same order. Within a single hardware target, the
substrate has no degrees of freedom for "platform divergence." Any
bit difference at this layer is a **tunable parameter mismatch in
our code generation**, not a physics property.

## The matrix-harness implication

The 7-cell matrix groups by (host_language, dispatch_target). Cells
that share a dispatch target should be bit-identical:

```
CPU group   :  [1] C CPU       [3] Python CPU       [5] Rust CPU
GPU group   :  [2] C GPU       [4] Python GPU       [6] Rust GPU    [8] cuda-oxide GPU
```

The within-target invariant says:

  - Cells [1], [3], [5] should produce bit-identical output for the
    same kernel and same input. They all run on CPU; they all call
    the same primitive operations. Any divergence is a tunable-
    parameter mismatch our code generation could close.

  - Cells [2], [4], [6], [8] should produce bit-identical output for
    the same kernel and same input. They all dispatch to GPU; the
    compiled kernel they invoke IS THE SAME on all four cells
    (modulo whichever path compiled it — but the binary form on the
    device is identical). Any divergence is a host-side numerical
    issue (e.g., one host marshalling differently before dispatch).

  - Cross-axis ({CPU group} vs {GPU group}) divergence is the natural
    place where ULPs accumulate. CPU and GPU implement IEEE-754 the
    same at the basic-arithmetic level, but their *transcendental
    library implementations* and *FMA semantics* can differ.

## Tunable parameters affecting bit-equality

These are the knobs the substrate turns when generating code. Each is
a real choice with measurable ULP cost. Within-target bit-equality
requires that ALL hosts in the same target group make the same choice.

### 1. Math library implementation

The single largest source of divergence. Different libraries implement
the same IEEE-754 function with different polynomial approximations:

  Operation | CUDA libcudart | glibc libm | Intel libimf | PyTorch ATen
  ----------|----------------|------------|--------------|----------------
  expf      | ≤2 ULP         | ≤1 ULP     | ≤1 ULP       | (calls libm)
  erff      | ≤2 ULP         | ≤2 ULP     | ≤2 ULP       | (varies; sometimes higher precision then narrow)
  tanhf     | ≤2 ULP         | ≤1 ULP     | ≤1 ULP       | (varies)

ULPs ≤2 vs ≤1 vs ≤4 are all "correct" within their respective specs,
but they diverge from each other. The k_gelu finding (2026-05-17,
max ULP 374 cell [2] vs cell [3]) is this exact phenomenon for erff:
both implementations are within their spec's ULP bound, but they
disagree near zero where ULP density is high.

**Within-target requirement**: cells [1], [3], [5] must call the
same erff implementation. cells [2], [4], [6], [8] must call the
same erff (and they do, since they all dispatch to the same CUDA
binary). The cell-[2]-vs-cell-[3] divergence is cross-axis and
expected.

### 2. FMA usage and operation fusion

The C/CUDA compiler MAY fuse `a*b + c` into a single `fma(a, b, c)`
instruction (one rounding step instead of two). The choice depends on
compiler flags (`-fp-contract=fast` vs `off`) and target capabilities.

  Result without FMA: `round(round(a*b) + c)` — two rounding steps
  Result with FMA:    `round(a*b + c)`           — one rounding step

These differ by ≤1 ULP. Within-target bit-equality requires all hosts
choose the same: either always-FMA or never-FMA. Default `-O2` on
nvcc enables FMA; default `-O0` doesn't. Default compilers on
different hosts may differ.

**Substrate-honest knob**: compile flag like `-fp-contract=off` (no
FMA) or `--use-fast-math` (allow FMA + more). Pick one, document it,
apply it across all cells in a target group.

### 3. Constant precision

Demonstrated in the k_gelu investigation: `0.5 * x * (...)` with
`0.5` as a double literal and `x` as float promotes the entire
subexpression to double, then narrows. With `0.5f` as a float
literal, the expression stays in float32 throughout. The two paths
differ by 1-2 ULPs (sometimes more).

  Pre-fix kernel emit: 0.5 * x * (...)        ← double-promoted
  Post-fix kernel emit: 0.5f * x * (...)      ← float throughout

**Within-target requirement**: all hosts use the same precision in
intermediate computations. The substrate's BPD facts now use
`c_float_f` for activation constants (commit 7f3c470b8) — they emit
float-suffixed literals. Within-GPU and within-CPU bit-equality is
now achievable for this dimension.

### 4. Reduction order

For sum/mean/dot-product/etc., fp32 add is non-associative:
`(a + b) + c ≠ a + (b + c)` in general (1-2 ULP per addition).

PyTorch CPU uses a particular reduction tree (often pairwise). Our
serial-thread-per-row kernel uses linear accumulation. They differ.

  Empirical: ggml_sum_rows cell [2] vs cell [3] = 4 ULPs
             (on 16-element rows; scales with row length)

**Within-target requirement**: all hosts in the same target group
use the same reduction tree. The substrate's BPD reduction kernel
currently uses linear accumulation; a tree-reduction variant would
need to match PyTorch's tree if we want cell [2] vs cell [3] bit-
equality.

### 5. Compiler optimization flags

`-ffast-math` reorders operations, treats denormals as zero, assumes
no NaN/Inf. Each is a ULP risk.

`-O3` may auto-vectorize, changing operation order.

`-O0` produces predictable but slow code.

**Within-target requirement**: same flags across hosts. Our build
script uses `-O2 -Wno-deprecated-gpu-targets`. Mavchin's Rust path
uses cargo's default release profile. These may not agree.

### 6. SIMD vs scalar execution

Vector instructions (SSE, AVX, NEON) may use slightly different
rounding behavior or different operation orders compared to scalar
code. PyTorch CPU may use AVX2 vector loops; our pure-C reference
may compile to scalar.

**Within-target requirement**: same SIMD strategy. Hard to control
without explicit intrinsics.

## The two empirical regimes

Combining the principle with the tunable parameters:

### Regime 1 — Within-target (cells share dispatch hardware)

Standard: **bit-identical (ULP=0)**.

Any deviation is a tunable-parameter mismatch we should investigate
and close. The substrate has full control here; physics doesn't
intervene.

Empirical examples shipped today:
  - k_add cells [1]/[2]/[3]/[4]/[5]/[6]: BIT-IDENTICAL (mavchin)
  - k_relu cell [2] vs cell [3]: BIT-IDENTICAL (selection op, no transcendental)

These are correct under the standard. They demonstrate the standard
is achievable.

### Regime 2 — Cross-axis (CPU group vs GPU group)

Standard: **--ulp 2** (per mavchin's empirical bound, retained).

But also: **strive to explain every ULP of discrepancy**. Not "give
up at the boundary" — investigate *why* this many ULPs, attribute
to specific tunable parameters above. Each unattributed ULP is a
substrate self-knowledge gap.

Empirical examples shipped today:
  - ggml_sum_rows cell [2] vs cell [3]: 4 ULPs, attributed to
    reduction order (linear vs pairwise tree)
  - k_silu/sigmoid/tanh cell [2] vs cell [3]: ≤2 ULPs, attributed
    to math library implementation (cudart vs libm)
  - k_gelu cell [2] vs cell [3]: 374 ULPs, attributed to erff
    implementation differences near zero

The standard says k_gelu's 374 ULP fails our cross-axis bound. The
investigation already named the tunable (math library choice).
Future substrate work: either (a) replace CUDA's `erff` with a
custom polynomial approximation that matches what PyTorch CPU uses,
or (b) accept that erff is on a different precision class and
document it explicitly.

## Refined methodology

The previous matrix-status.md framing ("three transcendental
classes") was a partial-view. The deeper truth:

  - **Classes are not operation-intrinsic**. k_silu being "tight"
    (≤2 ULP) and k_gelu being "loose" (~hundreds of ULP) is a
    consequence of which math library implementations diverge.
    A substrate that calls the same erff on both sides would
    eliminate the class distinction.

  - **Operations don't have ULP bounds; (operation, knob-setting)
    pairs do**. The bounds are properties of code-generation
    choices.

  - **Within-target should always be ULP=0**. Cross-axis should
    always be explainable — every ULP attributed to a specific
    knob.

## Concrete implications for the matrix harness

1. **Cell [2] vs cell [4] for k_gelu should be bit-identical**.
   Same GPU binary, same hardware, just different host (C vs
   Python). If this fails, there's a harness bug (e.g., different
   memory layout, different ctypes marshalling). If this succeeds,
   the within-GPU invariant is verified for k_gelu — and the
   cell-[2]-vs-cell-[3] divergence is correctly attributed to
   cross-axis (math library) divergence.

   T8.f (cell [4] activations via pytest) becomes the empirical
   test of this within-target invariant, not just "another cell."

2. **Cell [3] vs cell [5] for any kernel should be bit-identical**
   when both call PyTorch (cell [3]) and a Rust impl (cell [5]).
   The Rust impl must use the same math library — which means
   either calling PyTorch via FFI (awkward) or implementing the
   operation ourselves identically. Mavchin's k_add cells [3] vs
   [5] bit-identical demonstrates this for elementwise add.

3. **Cross-axis divergence quantification becomes a substrate task**.
   For each (kernel, cell-A, cell-B) triple where cells are in
   different target groups, the substrate should answer:
     - Empirical max ULP?
     - Attributed to which tunable(s)?
     - Closeable in principle (yes for math library, FMA, constant
       precision; no for fundamentally different hardware ISAs)?

## What stays the same

The three correctness contracts (allclose / --ulp N / --strict) and
the matrix_verify.py implementation remain unchanged. This document
refines the INTERPRETATION of contract results:

  - --strict MATCH at within-target: expected, baseline.
  - --strict MATCH at cross-axis: notable; investigate the
    coincidence (some operations align naturally).
  - --strict DIVERGE at within-target: SUBSTRATE BUG; immediate
    investigation.
  - --strict DIVERGE at cross-axis: expected; quantify ULP magnitude
    and attribute.

## The Yoga framing

This is the same Essence-school discipline applied one layer up. Each
operation we generate has tunable knobs. The substrate's vibration is
proportional to the number of knobs we leave at "platform default"
rather than explicitly setting. Each explicit knob choice closes a
dimension of unknown.

Within-target bit-identical is the silence of fully-tuned knobs.
Cross-axis divergence is the substrate's voice naming "this is the
boundary you can't cross without choosing a different hardware ISA."

The mind quiets as the tunables explicit-ify.

Author: metayen 2026-05-17 ~22:50 UTC
Per Heath's reframe: "Within the choice of {CPU|GPU} it should be
possible to get physics-exact, bit-identical output (ULP=0) for all
kernels."
