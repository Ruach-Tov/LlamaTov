# Hypothesis-Testing Trajectory: cuBLAS sgemv Subsumption

**Date**: 2026-05-18 (afternoon UTC)
**Participants**: mavchin (empirical experiments on Tesla P4), metayen
(structural reasoning + literature/source research), Heath (direction)
**Status**: curriculum example — the journey of hypothesis-testing
toward Tech-Level subsumption

This document captures the *trajectory* of hypothesis-testing in a
specific cuBLAS sgemv subsumption attempt, preserved as curriculum
distinct from the eventual answer. The substantive value is in the
*method* of moving through hypotheses, not in which hypothesis
ultimately landed.

Future wizards encountering similar subsumption work can study this
trajectory to understand:

- How structural reasoning and empirical measurement interact
- When hypotheses are productively generated vs when they exhaust
- How the substrate-honest principle "measure, don't assume" plays
  out in practice
- What "the wall" looks like when hypothesis-testing converges

## Context

The collective was working on subsuming `cuBLAS sgemv` (single-
precision general matrix-vector multiply) at bit-identical fidelity.
For input `y = A * x` with A of size M=64, K=256:

- mavchin had built a kernel structurally similar to cuBLAS's
  expected approach: 32 threads per row, single accumulator,
  `__shfl_xor_sync` warp reduction
- Output gap from cuBLAS: **12 ULP, 49/64 rows differing**
- Five candidate causes (a)–(e) named at the outset

The 12 ULP was "tantalizingly close" — small enough to suggest a
single structural difference, large enough to be substantively
non-zero.

## The five initial hypotheses

mavchin enumerated five candidate causes:

  (a) `__shfl_xor` vs `__shfl_down` produces different reduction trees
  (b) Multiple accumulator registers per thread (cuBLAS ILP)
  (c) `+=` vs `__fadd_rn` in cross-warp combine
  (d) Multi-row blocks (different launch geometry)
  (e) Column-major memory layout / `OP_T` transpose

This was the *expanded* hypothesis space. Each candidate had a
substantive reason to be considered.

## Hypothesis testing in order

### Round 1 — metayen's prioritization

After receiving the five candidates, metayen analyzed them structurally
without running experiments:

- **(a) Worked out analytically** that XOR and DOWN reductions
  produce *identical* lane-0 results despite different intermediate
  states. Predicted: not the cause.

- **(e) Hypothesized as highest priority**: 49/64 row mismatch pattern
  (NOT all 64) suggested almost-aligned layout where some rows happen
  to coincide and others diverge — classic signature of layout
  mismatch.

- **(b) Hypothesized as second priority**: modern GPU codes use 2-4
  accumulators for ILP. cuBLAS would likely do this.

- **(c) Depends on (b)** — only relevant if multi-accumulator combine
  is the structure.

- **(d) Structural variation** — doesn't change per-thread arithmetic.

Sent recommended experiment order to mavchin: Experiment 3 (layout
test), Experiment 2 (two-accumulator), Experiment 1 (XOR vs DOWN
microbench), Experiment 4 (`__fadd_rn` final).

### Round 2 — mavchin's empirical results

mavchin ran experiments and reported:

  (a) XOR vs DOWN — **equivalent at lane 0** (metayen's analysis correct)
  (b) Multi-accumulator — **WORSE (48 ULP vs 12)**, cuBLAS uses single
  (e) Layout — **NOT the cause**, both row/col major give same 12 ULP

Two of metayen's two highest-priority hypotheses were **empirically
disproved**. The XOR/DOWN analysis was correct. Five hypotheses
remained:

- Vectorized loads (float2/float4)
- Loop unrolling factor
- Some kernel structure metayen hadn't anticipated
- 64 threads (2 warps) per row with shared-memory cross-warp reduce
- A different K-dimension iteration order

### Round 3 — mavchin requests CUTLASS source check

mavchin asked metayen to check NVIDIA's open-source CUTLASS library
for its gemv implementation, since CUTLASS is "NVIDIA's open-source
reference and likely mirrors cuBLAS's approach."

metayen, with Heath's "feel free to web search" permission, fetched
`include/cutlass/gemm/kernel/gemv.h` from
`github.com/NVIDIA/cutlass`. Substantive findings:

CUTLASS has TWO specializations:

1. **Column-major A**: 1 thread per row (`kThreadsPerRow = 1`),
   no warp reduction needed
2. **Row-major A**: `kThreadsPerRow = min(kThreadCount / (sizeof(A) *
   kElementsPerAccess), 16)` — for scalar sgemv, that's **16 threads
   per row, NOT 32**

The row-major specialization uses *tiled iteration* with
`tileA_k = kThreadsPerRow * kElementsPerAccess` and explicit
`arch::global_load` with cache hints.

This was a *new* substrate-design insight: cuBLAS's pattern might be
substantively different from mavchin's 32-threads-per-row assumption.

metayen sent a concrete experiment: try 16 threads per row.

### Round 4 — empirical CUTLASS-pattern test

mavchin ran the CUTLASS-pattern experiments:

  32 threads/row (current):       12 ULP ← STILL CLOSEST
  16 threads/row (CUTLASS rowmaj): 48 ULP ← WORSE
  1 thread/row (CUTLASS colmaj):  179 ULP ← MUCH WORSE

**CUTLASS is NOT what cuBLAS uses on sm_61.**

This was the second time metayen's hypothesis (this time grounded in
CUTLASS source) was empirically disproved. The empirical data kept
winning over structural reasoning.

### Round 5 — recognition of the wall

By Round 4, the team had moved through 7 hypotheses, each tested
empirically. All but one (vectorized float2 loads, queued) had been
disproved. The remaining hypothesis space was:

  1. Vectorized float2/float4 loads grouping FMAs differently
  2. NVCC loop unrolling structure
  3. SASS-level instruction ordering / register allocation

Heath surfaced the substantive shift: continuing to generate
hypotheses was producing marginal information per experiment. The
methodology had moved from *discovering structure* to *eliminating
candidates*. Different cost-benefit tradeoff.

Heath asked mavchin to pursue **SASS comparison** — disassembling
cuBLAS's actual binary and reading the instruction sequence directly.
This shifts from inspection-based hypothesis-testing to empirical
ground truth.

### Where the trajectory is at this writing

mavchin is preparing the SASS comparison. The 12 ULP remains
unsubsumed but **localized**. The remaining space is narrow enough
that SASS comparison will close it.

The methodology shift is captured in the sibling doc
`sass-comparison-as-substrate-honesty.md`.

## What this trajectory teaches

### 1. Inspection-based reasoning is bounded

metayen's two highest-priority hypotheses (layout and multi-
accumulator) were both empirically disproved. The CUTLASS-source-
grounded hypothesis was also disproved. **Structural reasoning,
even when well-supported, can be wrong.**

This isn't a failure of the wizard; it's a property of the problem.
Subsumption against proprietary binaries operates against *NVIDIA's
choices*, not *the substrate's reasoning about what NVIDIA might
have chosen*. Empirical measurement is the arbiter.

### 2. Each ruled-out hypothesis is substantive progress

The trajectory shows: even when no hypothesis lands, the space of
possible causes shrinks. After 7 ruled-out hypotheses, the remaining
space is:

  - FMA-chain order within a single warp's per-thread accumulation
  - Plus a few specific possibilities (float2 loads, loop unrolling)

That's substantively narrower than the starting space of "anywhere
in the cuBLAS kernel structure." The collective has *learned*
something across the trajectory, even before the answer lands.

### 3. The wall has a recognizable shape

The wall — when hypothesis-testing exhausts — shows three patterns:

  - Several hypotheses in a row ruled out without closing the gap
  - New hypotheses are increasingly speculative
  - Remaining space is narrow enough that each test answers a small
    question

When the wall is recognized, the substrate-honest move is to change
methodology. SASS comparison is the substrate's tool for
*after-the-wall* subsumption work.

### 4. Empirical experiments compound in value

mavchin's experiments produced more than yes/no answers. Each
experiment generated *substrate-design knowledge* about what cuBLAS
on sm_61 with K=256 does NOT do:

  - Does NOT use multi-accumulator (single accumulator confirmed)
  - Does NOT depend on memory layout (row vs col give same result)
  - Does NOT use CUTLASS row-major (16 threads/row makes it worse)
  - Does NOT use CUTLASS column-major (1 thread/row much worse)
  - Does NOT have non-trivial cross-warp logic (32 threads/row is
    structurally right)

This is durable knowledge about cuBLAS's specific kernel selection
for this case. Future subsumption work for nearby kernels (sgemvT,
sgemm with small M, dgemv) can start from this knowledge instead
of re-deriving it.

### 5. Web research has bounded value

metayen's research into CUTLASS source produced a substantive
hypothesis (kThreadsPerRow=16) that turned out to be empirically
wrong. This is methodology-significant: **open-source references
are not always representative of proprietary implementations.**

CUTLASS is "NVIDIA's open-source library" but it does not dictate
cuBLAS's internals. cuBLAS may have predated CUTLASS, may have
sm_61-specific paths CUTLASS doesn't expose, or may use different
strategies entirely.

The lesson is calibration: open-source references provide
*plausible* hypotheses, not *certain* answers. They're worth
consulting (cheap, often informative) but require the same empirical
verification as any other hypothesis.

### 6. Collaboration patterns matter

The trajectory shows a productive division of labor:

  - **mavchin** owns the empirical layer: actual GPU runs, ULP
    measurements, cuBLAS comparisons. Has the silicon.
  - **metayen** owns the reasoning layer: structural analysis,
    literature/source research, hypothesis generation. Has the
    abstract substrate.
  - **Heath** owns the framing layer: when to shift methodology
    (hypothesis-testing → SASS comparison), what success looks
    like at each stage, what the educational gain is.

Each wizard's contribution was necessary. metayen's hypotheses,
even when wrong, *narrowed the space* by providing testable
candidates. mavchin's experiments empirically arbitrated. Heath's
framing recognized when the methodology should shift.

A single-wizard version of this trajectory would have been less
substantive. The structural reasoning AND the empirical
measurements were both load-bearing, and they needed different
specialties to produce.

## What this trajectory does NOT teach

The trajectory does NOT teach which hypothesis was *right* — that
answer is forthcoming from mavchin's SASS work. The trajectory is
about the *method*, not the *answer*.

The trajectory does NOT teach that hypothesis-testing was wasted.
The 7 ruled-out hypotheses are now durable knowledge about cuBLAS's
behavior. Future subsumption work compounds on this.

The trajectory does NOT teach a specific decision rule for "when
to shift methodology." Heath's recognition of the wall was
substantively right but methodologically intuitive. A future
substrate-design exercise could formalize the wall-recognition
criteria (e.g., "after 5 ruled-out hypotheses with narrow
remaining space, shift to empirical-ground-truth methodology").

## Connection to other methodology

This trajectory operates under several established principles:

- **Measure, don't assume; align with the subsumption target**
  (medayek's principle from CFD C2.1): the trajectory's repeated
  pattern of "metayen hypothesizes → mavchin empirically arbitrates"
  IS this principle in action

- **Comprehension over verbatim**: the substrate is trying to
  *comprehend* what cuBLAS does, not just produce matching bytes.
  Each ruled-out hypothesis improves comprehension even when it
  doesn't produce a match.

- **SASS comparison as substrate-honest reverse-engineering**
  (sibling doc): the methodology shift this trajectory leads to

## Future curriculum value

This trajectory becomes a template for similar subsumption work
across LAPACK / cuDNN / cuFFT. The four-phase pattern from the
SASS-comparison doc maps onto the trajectory's structure:

  Phase 1 (hypothesis-testing until narrowing): Rounds 1-4 here
  Phase 2 (recognition of the wall): Round 5 here
  Phase 3 (SASS disassembly): forthcoming
  Phase 4 (substrate emit toward known SASS): forthcoming

Future subsumption work can use this trajectory as:

  - A *checklist* of common hypotheses to test (reduction tree,
    multi-accumulator, layout, threads-per-row, CUTLASS-pattern)
  - A *recognition guide* for when the wall has been hit
  - A *collaboration template* for the structural-reasoning +
    empirical-measurement + framing-direction division of labor

## State at this writing

  - 7 hypotheses tested, 6 ruled out, 1 (vectorized float2) queued
    but likely to also be ruled out
  - 12 ULP gap remains, localized to FMA-chain order within
    per-thread accumulation
  - SASS comparison begun by mavchin per Heath's direction
  - Methodology shift documented in sibling doc

The trajectory will receive an addendum or closing section when
the SASS comparison resolves the gap and the substantive answer
is known. The current trajectory documentation is intentionally
written before the answer is in hand, preserving the curriculum
value of the hypothesis-testing journey distinct from the answer.

---

*Authored 2026-05-18 ~17:35 UTC by metayen, per Heath's "(a) then
(c)" direction. Captures the trajectory as curriculum before the
answer is known, so the method is discoverable independent of the
specific sgemv outcome.*
