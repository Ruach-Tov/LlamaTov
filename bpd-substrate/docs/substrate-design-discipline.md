# Substrate-Design Discipline — the parameter-as-family principle

**Author**: metayen, with Heath's substantive direction  
**Date**: 2026-05-21  
**Status**: Foundational substrate-design discipline statement

## The principle in one sentence

> **A substrate-design parameter is a NAMED FAMILY OF IEEE-CORRECT CHOICES that applies wherever the substantive shape applies. Naming it where PyTorch makes a specific choice is the START of the work, not the end. The substrate's job is to expose the parameter at every op where the family applies, then SWEEP it.**

## The discipline this replaces

The naïve approach to building a kernel substrate that matches PyTorch is **PyTorch-mimicry discipline**:

1. Read PyTorch's source.
2. Implement what PyTorch does.
3. Stop when bits match.

This produces a substrate that **subsumes PyTorch's specific choices** but doesn't make the choice-space visible. The substrate is then bounded by PyTorch's design decisions.

**Substrate-design discipline** is structurally different:

1. Read PyTorch's source.
2. Identify the **family of IEEE-correct choices** that PyTorch's specific choice is one instance of.
3. Name the family as a substrate-design parameter.
4. Implement PyTorch's specific choice as the default instantiation.
5. **Continue**: expose the same parameter at every other op where the substantive shape applies, even if PyTorch makes a different choice there.
6. Sweep the parameter space — find configurations PyTorch never explored.

The substrate that emerges has **substantively more degrees of freedom** than PyTorch's design space alone.

## Where this came from (empirical history)

This principle was crystallized in conversation between Heath and metayen on 2026-05-21 during the Stanford KernelBench L1 push. The substrate had reached 93/100 BIT_IDENTICAL by naming 8 substrate-design parameters:

```prolog
accumulation_precision(fp32)
opmath_precision(fp32)
cpu_fp_mode(strict)
bn_mode(precomputed_scale_offset)
reduction_strategy(cascade(8, 4, 4, 16))
rsqrt_variant(reciprocal_sqrt)
gemm_tile_strategy(goto_sandy(768, 384, 16, 4))
norm_division_strategy(direct_division)
cumulative_acc_type(double)
```

The 9th parameter (`cumulative_acc_type(double)`) was added when the cumsum family of kernels (#89, #90, #91, #92, #93) all needed PyTorch's "accumulate in f64, store as f32" pattern.

Heath then observed substantively: **"even if PyTorch uses f32 and we match, we could still extend that parameter to those cases, and then when we sweep the parameter for performance searches, we have that degree of freedom everywhere that we might benefit from it"**.

This was the substantive shift. The parameter `cumulative_acc_type` had been treated as a constraint to match PyTorch's specific choice on cumsum-family ops. Substrate-design discipline says it's a **family of choices** that applies wherever sequential dependent accumulation is the substantive shape — even for ops where PyTorch chose `float` and we already matched.

## Why this matters for the substrate's value

### 1. Performance sweep freedom

When `cumulative_acc_type` is exposed at every applicable op:

```prolog
% Currently:
platform_param(pytorch_cpu_default, cumulative_acc_type(double)).

% After parameter-as-family expansion, applicable to:
%   bpd_cumsum_cpu, bpd_cumprod_cpu, bpd_masked_cumsum_cpu  (currently use it)
%   bpd_layernorm_cpu (running stats — currently uses Welford f32, could use f64 acc)
%   bpd_kl_div_loss_cpu (sum reduction — currently f32, could use f64)
%   bpd_mse_loss_cpu, bpd_huber_loss_cpu, bpd_hinge_loss_cpu (mean — could use f64)
%   ... and many more
```

Each op gets a `cumulative_acc_type` choice. The sweep tests all (op × choice) pairs and finds:
- Which configurations preserve bit-identity with PyTorch
- Which preserve bit-identity AND are faster than PyTorch's choice
- Which trade off bit-identity for substantial speedup (under user-declared tolerance)

### 2. Discovering new IEEE-correct configurations

PyTorch's design space is one path through the substrate-design family. The substrate can find:

- `cumulative_acc_type(kahan_f32)` — Kahan-corrected float accumulation
- `cumulative_acc_type(neumaier_f32)` — Improved Kahan
- `cumulative_acc_type(double_pairwise)` — Cascade then double sum
- `cumulative_acc_type(extended)` — 80-bit x87 extended precision (where available)

For each op, the substrate empirically determines: does this configuration preserve bit-identity? Match PyTorch? Beat PyTorch's performance? Match it bit-for-bit but on different hardware?

### 3. Target-platform substrate variants

When porting to NEW hardware (NEON, AVX-512, future accelerators), the substrate-design parameter exposure makes the porting work substantively easier:

```prolog
platform_param(pytorch_cpu_neon,    cumulative_acc_type(?)).  % Discover empirically
platform_param(pytorch_cpu_avx512,  cumulative_acc_type(?)).
platform_param(pytorch_cuda_a100,   cumulative_acc_type(?)).
```

Sweep the parameter space on each target. The substrate discovers the right choice per platform without re-reading PyTorch's source for each one.

### 4. Public contribution becomes substantively easier

A community contributor with knowledge of one parameter family (e.g., ARM NEON SIMD scheduling) can drop in and contribute a new instantiation of `gemm_tile_strategy` for NEON hardware. They don't need to understand the whole substrate. They just need to:

1. Add their instantiation: `gemm_tile_strategy(goto_neon(P, Q, UM, UN))`
2. Verify it produces some target's bits (or characterized error bound)
3. Submit a PR

The substrate's contract is: **bit-identity is the merge bar**. The parameter family makes that contract composable with new contributions.

## How to apply this discipline practically

When you encounter a place in the substrate where:

1. You're matching PyTorch's specific choice for op X
2. The match works (BIT_IDENTICAL)

Ask: **what is the family of IEEE-correct choices that PyTorch's specific choice is one instance of?**

For each member of that family, ask:

3. **At which other ops does this family apply substantively?** (Same shape)
4. **At each such op, what is PyTorch's specific choice there?**
5. **Is PyTorch's choice the same as for op X, or different?**

The substrate-design move: declare the parameter family. Make all the relevant ops parameterized over it. Default each op to PyTorch's specific choice for that op. **Then the parameter-as-family is exposed substrate-wide for sweeping.**

## Practical example: cumulative_acc_type

### PyTorch's specific choices (empirically observed)

| Op | PyTorch's `cumulative_acc_type` choice |
|---|---|
| torch.cumsum | double |
| torch.cumprod | double |
| torch.cumsum (reverse) | double |
| torch.cumsum (exclusive) | double |
| torch.cumsum × mask | double |
| torch.mean (cascade reduction) | float (uses cascade(8,4,4,16) instead) |
| torch.sum (cascade reduction) | float (cascade) |
| F.layer_norm (Welford rowwise) | float (welford_simd8_cascade_chunk16) |
| F.batch_norm (precomputed_scale_offset) | float (naive sequential two-pass) |

### The substantive shape question

The substantive shape that determines whether `cumulative_acc_type(double)` is the **right** choice is: **does the accumulator carry a substantial running value that can lose precision at f32?**

- Cumsum: YES — running sum can grow large. → double.
- torch.sum via cascade: NO — accumulator is reduced in tree, no single running value. → float + cascade.
- Welford: NO — uses (mean, M2) state where M2 is bounded by the population variance scale. → float.
- Batchnorm two-pass: NO at small spatial size, but YES at large. → float currently. **Substrate-design opportunity**: sweep this parameter for batchnorm at large shapes.

### Substrate-design implementation

Each kernel exposes `cumulative_acc_type` as a parameter, defaulting to PyTorch's specific choice:

```c
// bpd_layernorm_cpu — currently uses Welford f32 (matches PyTorch). 
// Expose parameter for future sweep:
typedef enum { CUMACC_FLOAT, CUMACC_DOUBLE } cumacc_t;
void bpd_layernorm_cpu_param(..., cumacc_t cumacc) {
    if (cumacc == CUMACC_FLOAT) { /* Welford f32 path — bit-identical PyTorch */ }
    else if (cumacc == CUMACC_DOUBLE) { /* Welford f64 → cast to f32 */ }
}
void bpd_layernorm_cpu(...) {
    bpd_layernorm_cpu_param(..., CUMACC_FLOAT);  // default = PyTorch's choice
}
```

Then a sweep harness can test both configurations across many shapes and discover whether `CUMACC_DOUBLE` ever produces a substantively different (perhaps slightly more accurate, perhaps faster on some hardware) result while remaining IEEE-correct.

## Substrate-design discipline statement

The substrate is **not** "implement PyTorch's choices in Prolog + C." The substrate is **"name the substrate-design choice families, sweep them, and find the substantive optima."**

PyTorch is one path. We sweep the families. **The substrate's value is the parameter space we expose, not the specific instantiation we match.**

Bit-identity is the **contract**. Parameter-family is the **substrate**. PyTorch's choice is the **default**. Other valid choices are the **substrate's degrees of freedom**.

🕯️⚒️
