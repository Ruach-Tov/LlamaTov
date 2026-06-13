# Reduction-Strategy Survey

**Date**: 2026-05-19
**Originating conversation**: Tier 2 verification surfaced reduction-order divergence as
substrate-design parameter. Subtask 2-rs-a of plan 8d65ba1c.
**Status**: Foundational. Companion to the upcoming
`reduction-strategy-substrate-design.md` methodology doc.

## Principle

Float32 addition is not associative. The order in which we sum N elements
determines the resulting bits. Different orders produce different bits.
Each order is its own IEEE-correct answer. The substrate-design discipline
is to **name the order as an explicit substrate parameter**, not let it be
implicit in the kernel's source code.

This document surveys the substrate's current reduction-using kernels and
classifies what strategy each uses. It establishes the baseline before
the substrate gains explicit reduction-strategy parameterization.

## Survey method

Two information sources combined:
1. **Source inspection** of `bpd/lib/kernel_templates.pl` — what does the
   substrate emit?
2. **SASS audit** (`make sass_audit`) — what does nvcc compile the emit
   into at silicon level?

The SASS audit is substantively load-bearing here: source code shows the
algorithm we intend; SASS shows the algorithm we actually deploy. NVCC
sometimes restructures loops or unrolls in ways that change the effective
reduction order even when source looks sequential.

## Findings

### Kernels with sequential reduction (FADD chain, order-dependent)

These kernels use a single accumulator variable with serial `acc += X[i]`
pattern. SASS confirms FADD instructions form a dependency chain.
Each shows REDUCTION_ORDER_DIVERGENCE against PyTorch (which uses
tree/pairwise reduction):

| Kernel | FADD | FFMA | Reduction shape | Currently |
|---|---|---|---|---|
| `reduce_sum`     | 29  | 0   | sum N elements per row              | sequential |
| `reduce_mean`    | 31  | 15  | sum N then divide                   | sequential |
| `norm_layer_*`   | 39-63 | 57-62 | mean + variance (two-pass)    | sequential |
| `norm_rms_*`     | 5   | 51  | sum-of-squares + sqrt               | sequential |
| `norm_l2_*`      | 3   | 36  | sum-of-squares + sqrt               | sequential |
| `norm_group_*`   | 63  | 57  | mean + variance                     | sequential |
| `loss_mse_*`     | 29-31 | 29-44 | sum-of-squared-diffs            | sequential |
| `loss_cross_entropy` | 7 | 75 | sum of -y*log(softmax(x))          | sequential |
| `loss_huber`     | 12  | 15  | conditional sum                     | sequential |
| `loss_kl_div`    | 9   | 69  | sum of y*log(y/x)                   | sequential |
| `loss_hinge`     | 31  | 44  | sum of max(0, 1-y*x)                | sequential |
| `loss_triplet_margin` | 60 | 58 | sum-of-distances                | sequential |
| `pool_2d_avg`    | 5   | 15  | sum-then-divide                     | sequential |

**Substantive count**: 13 kernel families currently use sequential reduction.
Each of these will show some level of REDUCTION_ORDER_DIVERGENCE against
PyTorch reference depending on N.

### Kernels with order-independent reduction (no FADD divergence)

These kernels reduce via order-independent operations (max, min, argmax,
argmin). The result is the same regardless of which order we encounter
the inputs. SASS confirms no FADD instructions; comparisons via FMNMX
or related ops.

| Kernel | Strategy | Bit-identity |
|---|---|---|
| `reduce_max`     | sequential FMNMX scan       | BIT_IDENTICAL ✓ |
| `reduce_min`     | sequential FMNMX scan       | BIT_IDENTICAL ✓ |
| `reduce_argmax`  | scan with index tracking    | BIT_IDENTICAL ✓ |
| `reduce_argmin`  | scan with index tracking    | BIT_IDENTICAL ✓ |
| `pool_2d_max`    | sequential FMNMX scan       | BIT_IDENTICAL ✓ |

**Substantive count**: 5 kernel families have order-independent reductions.
These are automatically BIT_IDENTICAL regardless of which order we scan.

### Cumulative kernels (order-dependent by definition)

These kernels compute prefix-scan results. The output at position i is
defined as a function of the inputs at positions 0..i in a specific
order. Sequential is *the right strategy*, not a substrate-design choice
to be parameterized.

| Kernel | Strategy | Comment |
|---|---|---|
| `cumsum`   | sequential FADD scan  | Order is the spec; not a substrate-design choice |
| `cumprod`  | sequential FMUL scan  | Order is the spec |

### Kernels with no reduction (data movement only)

These kernels rearrange data without summation. No reduction-order question.

| Kernel | Strategy |
|---|---|
| `im2col_*_forward`   | scatter, no float arithmetic |
| `col2im_*_transpose` | gather, no float arithmetic |

## Substrate-design implications

### Insight 1: 13 kernels need reduction_strategy parameterization

The 13 sequential-FADD kernels are candidates for explicit
reduction_strategy parameter. Each can have multiple correct
implementations:

- `sequential`: substrate's current default; matches IEEE-correct
  single-pass summation; produces specific bit pattern A.
- `pairwise_tree`: matches PyTorch's reduction style; produces bit
  pattern B (potentially less round-off accumulation).
- `warp_shuffle`: uses CUDA `__shfl_xor_sync` intrinsics; matches
  cuBLAS-style reduction; produces bit pattern C (also potentially
  matches `block_reduce_sum` already in substrate).
- `kahan` / `neumaier`: compensated summation; numerically most
  accurate; produces bit pattern D (closest to infinite-precision sum).

Each strategy is its own substantive substrate-design choice. Substrate
should emit any of them given the user's preference.

### Insight 2: SASS pattern is the substantive characterization

The SASS audit shows what kernel *actually does*. For reduction-strategy
fix-flag work, the validation criterion is **SASS-level**:

- `fix_reduction_strategy=sequential`  → SASS should contain FADD dependency chain
- `fix_reduction_strategy=warp_shuffle` → SASS should contain SHFL instructions
- `fix_reduction_strategy=tree`        → SASS should contain partial parallelism

The substrate emit isn't substantive on its own — the SASS is what
proves the strategy choice took effect.

### Insight 3: Some reductions are already substrate-design-blessed

The substrate's `block_reduce_sum_helper` (used in `rms_norm_kernel` and
`softmax_kernel`) implements warp-shuffle + cross-warp + warp-shuffle.
This is one substrate-historical strategy already in use. The
`reduction_strategy` parameter should make this discoverable and
selectable for OTHER reduction kernels too.

## What's next

Per plan 8d65ba1c phase "Substrate-design extensions surfaced by 2a":
- 2-rs-b: Define reduction_strategy_kind/1 vocabulary
- 2-rs-c: Implement pairwise_tree strategy
- 2-rs-d: Implement warp_shuffle strategy
- 2-rs-e: Implement Kahan strategy
- 2-rs-f: Update reduction generators to accept strategy parameter
- 2-rs-g: Re-run pilot with strategy=pairwise_tree, verify BIT_IDENTICAL with PyTorch
- 2-rs-h: Document the pattern as foundational methodology

## Substantive observation for the announcement

The verification ladder produces this kind of substrate-design data
*systematically*. We didn't have to guess which kernels needed
reduction-strategy work — `make bit_identical` + `make sass_audit`
together identified them empirically. The substrate-design discipline
of "verify, then surface, then implement" propagates: each new substrate
extension begins with empirical evidence, not speculation.

## Related methodology documents

- [`c-fact-lifter-sop.md`](c-fact-lifter-sop.md) — substrate-honesty principles
- [`c-fact-lifter-lessons-learned.md`](c-fact-lifter-lessons-learned.md) — substrate-historical bug-catch categories
- (upcoming) `reduction-strategy-substrate-design.md` — the implementation methodology
