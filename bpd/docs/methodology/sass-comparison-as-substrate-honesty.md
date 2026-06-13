# SASS Comparison as Substrate-Honest Reverse-Engineering

**Date crystallized**: 2026-05-18 ~17:25 UTC
**Discovered through**: cuBLAS sgemv subsumption work (Heath + mavchin + metayen)
**Status**: substrate-design pattern for Tech-Level subsumption against proprietary binaries

When the substrate's purpose is to subsume a proprietary binary (cuBLAS,
cuDNN, cuFFT, etc.) at bit-identical fidelity, inspection-based
hypothesis-testing converges toward — but rarely completes — exact
subsumption. The remaining gap (often single-digit ULP) requires
*empirical ground truth about what the binary actually does*. SASS
comparison is that ground truth.

This document names the pattern, describes the methodology, and
explains why it produces substantive substrate-design knowledge
beyond the specific subsumption case that triggered it.

## When this pattern applies

The pattern applies when ALL of these conditions hold:

1. **Subsumption target is a proprietary binary** — source code is
   unavailable, only the shipped `.so` / `.cubin` artifact
2. **The substrate produces a structurally similar kernel** that
   matches the target on most outputs but diverges by small bit
   amounts (typically 1-50 ULP)
3. **Inspection-based hypothesis-testing has narrowed the space** —
   obvious candidates (memory layout, reduction tree, accumulator
   count) have been empirically ruled out
4. **The remaining gap cannot be closed by further hypotheses** —
   the answer requires direct observation of the binary's actual
   instruction sequence

When these conditions hold, the substrate-honest move is to stop
guessing structurally and read the SASS NVIDIA shipped.

## Why hypothesis-testing hits a wall

The cuBLAS sgemv work today exemplified the wall-hitting pattern.
The collective generated and tested ~7 hypotheses:

1. XOR vs DOWN reduction tree → empirically equivalent at lane 0
2. Multi-accumulator per thread → made the gap WORSE
3. Column-major vs row-major layout → no effect
4. 16 threads per row (CUTLASS row-major) → made the gap WORSE
5. 1 thread per row (CUTLASS column-major) → made the gap MUCH WORSE
6. Cross-warp shared-memory reduction → not used in the closest kernel
7. Vectorized float2 loads → (queued at the time of this writing)

Each hypothesis was substantively reasoned (FMA chain analysis, CUTLASS
source inspection, etc.). Each was empirically tested. All but one or
two were ruled out. The remaining frontier — *FMA-chain order within a
single warp's per-thread accumulation* — was sufficiently narrow that
each new hypothesis had a small probability of landing.

The wall-hitting pattern: **the hypothesis space narrowed faster than
hypotheses could be generated**. Even if the next hypothesis closed
the gap, the methodology had stopped producing learning. Every new
hypothesis test had a 90%+ chance of being "ruled out" — empirically
disproved without revealing the actual cause.

At that point, the educational gain shifts. Continuing hypothesis-
testing produces marginal information per experiment. SASS comparison
produces *all the information at once*: NVIDIA's exact answer, fully
specified.

## The pattern: empirical SASS as arbiter

The methodology has four phases:

### Phase 1 — Hypothesis-testing until narrowing

Run the substrate's best-guess kernel against the proprietary binary.
Measure the bit-identical gap. If 0 ULP, you're done. If non-zero,
generate hypotheses about the cause, test each empirically.

Each ruled-out hypothesis is *progress*. The space of possible causes
shrinks. But each hypothesis also has a cost (engineering time to
implement and test). The hypothesis-testing phase is justified while
the *per-hypothesis information gain* exceeds the *per-hypothesis
cost*.

### Phase 2 — Recognition of the wall

The wall is recognized when:

- Several hypotheses in a row have been ruled out without closing
  the gap
- New hypotheses are increasingly speculative ("maybe loop unrolling
  factor? maybe some compilation flag we haven't found?")
- The remaining space is narrow enough that each test answers a
  small question rather than a large one

This is the recognition that the methodology has shifted from
*discovering structure* to *eliminating candidates*. Different cost-
benefit tradeoff. SASS comparison provides answers, not eliminations.

### Phase 3 — SASS disassembly + comparison

Tools:

```bash
# Disassemble the proprietary binary's SASS
cuobjdump --dump-sass libcublas.so.12 > cublas_sass.txt

# Search for the kernel of interest
grep -A 500 "code for sm_61" cublas_sass.txt | grep -B 2 -A 200 "sgemv"

# Disassemble your own kernel's SASS
cuobjdump --dump-sass your_kernel.cubin > your_sass.txt

# Diff them
diff cublas_sass.txt your_sass.txt
```

The diff reveals exact instruction sequences. For a divergence like
the sgemv 12-ULP gap, the SASS comparison shows:

- Which load instructions are used (LDG.32 vs LDG.64 vs LDG.128)
- The exact register allocation
- The instruction scheduling (interleaving of loads and FMAs)
- Any cache hints (`.CV`, `.CG`, `.CS` modifiers)
- Synchronization primitives (`BAR.SYNC`, warp-level fences)

Each of these can affect output bits in specific ways. The SASS
shows which one is operative.

### Phase 4 — Substrate emit toward known SASS

Once SASS is known, the substrate-design question becomes: how do we
emit c_ast that produces *this specific* SASS?

The answer is rarely "emit the SASS directly" (the substrate operates
at the CUDA C++ level, not SASS). Instead, the answer is "emit CUDA
that ptxas + the compiler will produce this SASS for."

This requires *substrate-level knowledge of which CUDA patterns
produce which SASS*. Examples:

- `__ldg(ptr)` produces `LDG.CV` (bypass L1, cache-volatile)
- Plain `*ptr` produces `LD.E.SYS` or `LDG.CA` depending on context
- `__fmul_rn(a, b)` produces `FMUL.RN` (explicit round-to-nearest, no
  FMA contraction)
- `a * b` may produce `FMUL` or be folded into a subsequent `FFMA`

The substrate's fix-flag pattern naturally extends to this: each
SASS-level choice becomes a named fix-flag with a known emit form.

## Why SASS comparison produces compounding educational gain

The substrate gains *more than just this kernel's answer*. The
educational compound interest:

### 1. cuobjdump tooling is reusable

The disassembly + comparison infrastructure built for sgemv applies
unchanged to every other proprietary binary: sgemm, sgbmv, ssbmv,
ssyr2k, cuDNN's convolution kernels, cuFFT's FFT kernels, etc.

A single afternoon of building the tooling produces capability for
hundreds of subsumption targets.

### 2. Reading SASS becomes a substrate practice

After the first SASS comparison, the substrate's wizards can read
SASS fluently. Future "what does cuBLAS actually do for this kernel"
questions become 5-minute investigations rather than 5-hour
hypothesis-testing arcs.

This skill is *unevenly distributed in industry* — most GPU
programmers never read SASS. The substrate's wizards becoming
SASS-fluent is a comparative advantage that compounds across all
future Tech-Level work.

### 3. SASS-level fix-flags become first-class

The fix-flag pattern (introduced in commit dc0b8be32 for ML kernels)
gains a third category alongside `defect_repair` and
`precision_tradeoff`:

  **sass_pattern_match**: a fix that emits CUDA designed to produce
  a specific known SASS sequence. Used when bit-identical subsumption
  requires matching not just the algorithm but the instruction-level
  details.

Examples that might emerge:
  - `fix_match_cublas_sgemv_load_cadence`: emit `__ldg` for A
  - `fix_match_cublas_sgemv_fma_grouping`: emit float2 loads with
    paired FMAs to produce specific FFMA ordering

Each fix is a named, opt-in choice tied to a specific SASS-level
observation.

### 4. The crystallization theorem gains a concrete reference

mavchin's earlier work established that "same source + same hardware
→ same bits across compilation paths" (the crystallization theorem).
SASS comparison provides the *empirical baseline* the theorem
operates against: when cuBLAS's SASS for kernel K is X, and the
substrate's emit produces the same X, then by crystallization the
substrate's output bits match cuBLAS's.

SASS-level identity is the *strongest* form of subsumption claim.
It's stronger than output-value identity (which can hold despite
different SASS, see the FMA contraction asymmetry doc). The substrate
can decide per-kernel which level it wants:

  - **Output-value identity** (weakest, easiest): same bits, possibly
    different SASS
  - **PTX identity** (medium): same PTX, definitionally same SASS
    after ptxas
  - **SASS identity** (strongest, hardest): same SASS bytes

The fix-flag pattern lets the substrate target whichever level the
subsumption case requires.

### 5. The methodology generalizes to non-NVIDIA proprietary binaries

The pattern — disassemble, read, design emit toward known instruction
sequence — applies to AMD ROCm binaries, Intel oneAPI binaries,
Apple Metal, and any future GPU vendor's proprietary kernels.

The substrate becomes the *only* open-source layer that can claim
bit-identical subsumption across vendors, because it operates at the
instruction-sequence level rather than at the source level.

## What this is NOT

This methodology is NOT clean-room reverse engineering. The substrate
is reading SASS that NVIDIA shipped publicly (in the .so binary), not
NVIDIA's source code or internal documentation. The legal status of
reading SASS is the same as the legal status of reading any compiled
binary: protected by fair-use and reverse-engineering-for-interoperability
provisions in most jurisdictions.

This methodology is NOT bug-finding. The substrate isn't looking for
defects in cuBLAS; it's identifying the implementation choices that
account for bit-level output. cuBLAS is treated as a *reference to
subsume*, not as a *target to critique*.

This methodology is NOT theft-by-imitation. The substrate produces
its own CUDA source code that happens to compile to similar SASS.
The intermediate forms (BPD facts, c_ast, CUDA C++) are entirely
substrate-authored. Only the *final SASS-level instruction shape*
matches by design, and this match is the substrate's interoperability
guarantee.

## Failure modes and when not to apply

Cases where SASS comparison should NOT be the next move:

- **The gap is not yet narrow enough** — if the substrate is still
  10x off on accumulator count or 50x off on threads-per-row,
  hypothesis-testing is still in its high-leverage phase. SASS will
  reveal too much to be actionable.

- **The proprietary binary is unstable across versions** — some
  cuBLAS algorithms are autotuned per-architecture, per-input-size,
  per-driver-version. SASS for v12.0 may differ from v12.3 for the
  same logical kernel. If the substrate's target users span multiple
  cuBLAS versions, SASS-level match may not be a stable invariant.

- **The kernel uses architecture-specific features** — tensor cores
  (`mma.sync`), warp-level primitives that the substrate's c_ast
  doesn't yet emit. SASS will show what's there, but the substrate
  may not yet have the primitive to express it.

In these cases, hypothesis-testing remains the right methodology, or
the right move is to *expand substrate primitives first* before
attempting SASS-level subsumption.

## Connection to other methodology principles

This methodology operates alongside the established principles:

- **Bug-for-bug as comprehension proof**: SASS-level subsumption is
  the strongest form of "we comprehend what they shipped, including
  the instruction-level choices"
- **Measure, don't assume; align with the subsumption target**:
  SASS is the strongest possible measurement of what the subsumption
  target actually does
- **Comprehension over verbatim**: the substrate's c_ast emit at
  SASS-target level is comprehension *of the instruction sequence*,
  not verbatim reproduction
- **Inline-literal canon, FMA contraction asymmetry**: both of these
  influence which SASS the substrate naturally produces. SASS
  comparison may surface that cuBLAS uses different choices (e.g.,
  cuBLAS may explicitly disable FMA contraction at certain stage
  boundaries via intrinsics), which becomes a new fix-flag.

## The Tech-Level subsumption arc this enables

Per Heath's 2026-05-18 ~16:30 UTC framing: the substrate's actual
mission is "subsuming the NVIDIA Technology Level" — Mode 1 (strict
NVIDIA-matching), Mode 2 (optimized faster-than-NVIDIA), Mode 3
(account-for-every-bit).

SASS comparison is the *enabling technology* for Mode 1 across the
NVIDIA ecosystem. Once a kernel's cuBLAS SASS is known, the substrate
emits Mode 1 by producing CUDA that compiles to that SASS. Mode 2
becomes Mode 1's SASS with named optimizations layered (each
optimization is a fix-flag with a specific SASS-level effect). Mode
3 is the named-and-individually-justified discrepancy ledger between
Mode 2's SASS and Mode 1's SASS.

The medayek-prepared LAPACK test suite is the verification harness
for this arc. Each LAPACK kernel becomes a subsumption target. SASS
comparison becomes routine. The substrate's methodology corpus
absorbs each subsumption's lessons as fix-flags and patterns.

## Future maintenance

When the substrate adds support for non-NVIDIA proprietary binaries
(AMD, Intel, Apple, etc.), this doc gets a vendor-specific addendum
or a sibling doc. The methodology pattern is vendor-agnostic; the
disassembly tooling and instruction set knowledge are vendor-specific.

When a SASS comparison surfaces a substantively new substrate-design
insight (e.g., a new c_ast primitive needed to emit a specific
instruction class), the insight gets captured in its own methodology
doc with a cross-reference from here.

---

*Authored 2026-05-18 ~17:25 UTC by metayen, per Heath's "(a) then (c)"
direction. Captures the SASS-comparison pattern before the empirical
result of the sgemv work is in hand — so the pattern's value is
discoverable independent of this specific subsumption case.*
