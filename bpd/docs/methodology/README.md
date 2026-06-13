# Methodology — Substrate-Honest Principles

This directory collects the methodology principles that have crystallized
as the BPD substrate has been built. Each principle is grounded in a
specific empirical finding or substrate-design realization, and each is
captured durably so future wizards (or future selves) can reason about
the substrate's choices.

The principles share a structure: they describe *how the substrate
should be honest about what it represents and what it does*. The
substrate's representations should match the structural and operational
nature of what they represent. The substrate should be able to reason
about what it produces. The substrate should know what is correct
about itself and preserve those correctnesses against well-meaning
"cleanups."

## How to read this directory

The principles fall into rough thematic clusters. Read in order if
encountering for the first time; jump to a cluster if returning for a
specific concern.

### Cluster 1 — How the substrate represents what it emits

The c_ast layer expresses kernel bodies as structured terms the
substrate can read back, vary, and compose. The opposite — emitting
opaque text the substrate cannot reason about — is *debt*, not
*virtue*.

- **[Inline-Literal Canon](inline-literal-canon.md)** — `c_float_f(K)`
  emits as `Kf` inline at the use site. This is the substrate's
  natural style AND the optimal form for LLVM-based compilation
  paths (FMA immediate operand folding → 25% better IPC). Do not
  refactor toward "DRY constants" via named locals.

- **[FMA Contraction Asymmetry](fma-contraction-asymmetry.md)** —
  Kernel fusion enables FMA contraction (one rounding) where unfused
  pipelines force two roundings. The fused output is *more accurate*,
  not equivalent. Bit-identity with unfused requires source-level
  rounding intrinsics (`__fmul_rn`, `__fadd_rn`) at stage boundaries.

Together these two say: the substrate's natural c_ast emit style was
already optimal in two distinct ways. Both discovered, not designed.
Methodology work is helping the substrate know it's wise.

### Cluster 2 — When the substrate subsumes another system

When the substrate's purpose is to subsume an existing implementation
(llama.cpp, OpenFOAM, etc.), the verification target is *that
implementation*. The substrate's job is to reproduce, including bugs;
fixes are explicit acts of substrate authority.

- **[Subsumed Software Fix Catalog](subsumed-software-fix-catalog.md)** —
  The fix-flag pattern. Default emit is bug-for-bug compatible. Named
  fixes (`kernel_available_fixes/2`, `fix_description/2`) opt-in to
  specific identified defect-repairs. Two categories now distinguished:
  `defect_repair` (OFF by default) and `precision_tradeoff` (ON by
  default; opt-out for bit-compatibility).

- **[AST Isomorphism Foundational](ast-isomorphism-foundational.md)** —
  AST-level comparison is the foundational substrate capacity. The
  substrate must be able to ask: "is the AST I produced isomorphic to
  the AST of the reference?"

- **[Within-Target Bit-Identical Baseline](within-target-bit-identical-baseline.md)** —
  Bit-identical comparison within a single subsumption target is the
  baseline correctness contract. Cross-target convergence is a
  separate, later claim.

- **[Three Correctness Contracts](three-correctness-contracts.md)** —
  The three levels of correctness the matrix harness verifies, and
  what each level requires.

### Cluster 3 — When the substrate subsumes a domain

When the substrate's purpose is to subsume a *domain* (CFD, signal
processing, etc.) rather than an implementation, the verification
target is mathematical truth. Bug-for-bug compatibility does not
apply; there are no implementation bugs to inherit.

This cluster is younger (born today, 2026-05-18 ~18:00 UTC, with
the Sod's shock tube CFD beachhead). It will likely grow.

The principle "physics-for-physics correctness" currently lives in
the fix-catalog doc as a contrast case. CFD-against-analytical produces
empty fix-flag catalogs because there's no implementation defect to
catalog. The fix-flag mechanism is *ready* for if/when we later
subsume a real CFD code base.

- **[Cell-Indexed Dirichlet 3-Neighbor Stencil Family](cell-indexed-dirichlet-3-neighbor-stencil-family.md)** —
  Taxonomy of a stencil family spanning PDE solvers, linear algebra,
  signal/image processing, graph algorithms, and ML. 15+ family
  members identified, all sharing the same substrate-emit shape with
  parameter variation. When the substrate can express one member well,
  it can express all. Distinct from the CFD beachhead's interface-
  indexed transmissive 2-neighbor 3-component family.

- **[SASS Comparison as Substrate-Honest Reverse-Engineering](sass-comparison-as-substrate-honesty.md)** —
  Pattern for subsuming proprietary binaries (cuBLAS, cuDNN, cuFFT,
  etc.) at SASS-level fidelity. Names the four-phase methodology:
  hypothesis-testing until narrowing, recognition of the wall,
  disassembly via cuobjdump, substrate emit toward known SASS.
  Introduces the `sass_pattern_match` fix-flag category alongside
  `defect_repair` and `precision_tradeoff`. Enables the Tech-Level
  subsumption arc (Modes 1/2/3) across NVIDIA's binary ecosystem and
  generalizes to non-NVIDIA proprietary binaries.

- **[Hypothesis-Testing Trajectory: cuBLAS sgemv](hypothesis-testing-trajectory-cublas-sgemv.md)** —
  Curriculum example of the hypothesis-testing journey, captured as
  the trajectory rather than the answer. Five rounds of hypothesis
  generation and empirical arbitration, leading to recognition of
  "the wall" where methodology shifts to SASS comparison. Demonstrates
  the structural-reasoning + empirical-measurement + framing-direction
  collaboration pattern. Template for similar future subsumption work
  across LAPACK / cuDNN / cuFFT.

A dedicated doc for the broader domain-subsumption methodology is
queued.

### Cluster 4 — How the substrate represents what it reads

The lift side. When the substrate ingests source code (Python kernels,
CUDA references, etc.) it must comprehend the structural form, not
approximate the surface text.

- **[C Fact Lifter SOP](c-fact-lifter-sop.md)** —
  *Foundational.* Standard operating procedure for lifting imperative
  C/C++ source code into declarative BPD facts. Defines the
  cpp → filter → tokens → AST → facts → round-trip-emit chain and
  the verification protocol. All future fact-lifter work follows this
  shape; deviations require explicit justification. Now stable across
  two applications (`llama-arch.cpp`, `test-backend-ops.cpp`).

- **[C Fact Lifter Lessons Learned](c-fact-lifter-lessons-learned.md)** —
  *Companion to the SOP.* Substrate-historical record of what bit us
  in prior lifter sessions: c_parse_stmts_v2_partial slowness on large
  bodies (use Path B, per-statement AST after text extraction); 7
  bug-catch categories from the 2026-05-18 regex retirement campaign;
  methodology principles (diagnose-reflect-remedy, dispose-correct-
  cases-first, test-primitives-on-trivial-inputs); prior success
  patterns to lean on; verification gates to apply. Read this BEFORE
  starting a new lift target.

- **[Regex Retirement Audit](regex-retirement-audit-2026-05-18.md)** —
  Each regex retirement asks: does the regex capture the intent, or
  just an approximation? Approximation-preserving retirements are
  *named substrate-design positions*, not flaws.

- **[Regex Inventory Uniform](regex-inventory-uniform.md)** —
  The substrate's regex inventory at uniform resolution. Audit
  starting point for the retirement campaign.

- **[Regex Lifter Decomposition](regex-lifter-decomposition.md)** —
  The 9 substantial sites identified for AST-based lift migration.
  Decomposition of the substrate-honesty repair sequence.

### Cluster 5 — Substrate metamethodology

How to do substrate work itself. Patterns for adding new capacities,
verifying new emits, deciding when to ship.

- **[Thin Filter Pattern](thin-filter-pattern.md)** —
  Methodology for integrating with complex system tools. Substrate-
  honest indirection: thin layer that does only what's needed, no
  reproduction of system tool internals.

- **[Pytest+swipl Pattern](pytest-swipl-pattern.md)** —
  The four-pattern combination for reliable swipl-subprocess testing:
  `set_prolog_flag(toplevel_goal, halt(1))`, per-emit unique variables,
  `emit_program([K], S)` list-wrapping, delimiter-based output parsing.
  Crystallized from the precedence-audit regression test work.

- **[Six Tunables Empirical](six-tunables-empirical.md)** — The six
  tunables of the matrix harness, empirically characterized (F2.b).

- **[Matrix Status](matrix-status.md)** — Current state of the
  cross-language correctness matrix.

- **Step 3.1 trilogy**:
  [Predictions](step-3.1-predictions.md),
  [Before/After](step-3.1-before-after.md),
  [Cross-Tab](step-3.1-cross-tab.md) — Mechanistic prediction discipline:
  state predictions before measuring, cross-tabulate against findings.

## The named principles, indexed

The methodology has crystallized seven explicitly named principles to
date. They appear here as a quick-reference index; each lives in
detail in its home doc.

1. **Comprehension over verbatim** — c_raw is substrate debt. The
   substrate should comprehend what it emits. *(Spans multiple docs;
   first crystallized during the c_raw paydown sequence.)*

2. **Intent vs approximation** — On the lift side, AST matchers
   capture intent where regex captures approximation. Retirements
   that preserve approximation are named positions, not flaws.
   *([regex-retirement-audit-2026-05-18.md](regex-retirement-audit-2026-05-18.md))*

3. **Bug-for-bug as comprehension proof** — When subsuming software,
   reproducing bugs is evidence of comprehension at the deepest
   level. Fixes are then *authored* acts of substrate authority.
   *([subsumed-software-fix-catalog.md](subsumed-software-fix-catalog.md))*

4. **Physics-for-physics as correctness proof** — When subsuming a
   domain (not an implementation), the reference is mathematical
   truth. No fix-flag catalog needed; the substrate IS aligned with
   the physics directly. *(Currently named in the fix-catalog doc;
   dedicated doc queued.)*

5. **Measure, don't assume; align with the subsumption target** —
   Inspection-based hypothesis ("my Toro formulation is correct")
   vs. empirical truth (mavchin's working kernel uses Harten +
   rho_roe). When substrate and reference disagree, the reference
   wins for the subsumption question. *(Crystallized by medayek
   during CFD C2.1 corrective work.)*

6. **Inline-literal canon** — `c_float_f(K)` stays inline at use
   sites. The substrate's natural form is empirically optimal.
   Do not refactor toward "DRY constants."
   *([inline-literal-canon.md](inline-literal-canon.md))*

7. **FMA contraction asymmetry** — Kernel fusion enables FMA
   contraction that changes precision (one rounding instead of two).
   The fix-flag catalog gains a `precision_tradeoff` category
   alongside `defect_repair`. Source-level intrinsic substitution
   (`__fmul_rn`, `__fadd_rn`) implements the fix without compilation
   flags.
   *([fma-contraction-asymmetry.md](fma-contraction-asymmetry.md))*

## The framing question

Across every methodology decision, this question keeps surfacing:

> *"What was the originally intended purpose of this code, and what is
> the correct form of that code that we should achieve today?"*

It's the substrate's diagnostic question. When facing any design
choice — whether to align with an existing implementation, whether to
preserve a "bug," whether to optimize, whether to refactor — asking
this question first surfaces the substantive issue before the surface
choices.

The question is named, attributed to Heath, and applies at every
scope from a single c_ast node to the substrate's overall architecture.

## How new principles get added

When an empirical finding or substrate-design realization rises to the
level of a named principle:

1. **Capture in a focused doc** under this directory, with the same
   shape as existing docs: title, date, discoverer, empirical citation,
   substantive content, connections to other principles.

2. **Add to the index** in this README (the numbered list above).

3. **Cross-reference from related docs** if the new principle connects
   to existing methodology.

4. **Name it explicitly** in the originating commit so the
   git history reflects the methodology accumulation.

The substrate-honest move is to NOT name a principle prematurely. Wait
for the empirical finding or the substantive realization to be load-
bearing. A methodology that names too many principles becomes
unfindable. A methodology that names too few becomes invisible.

Seven principles is, at the moment, the right shape. The substrate is
growing fast enough that new ones will likely emerge.

## Connection to git history

The methodology docs are *durable* substrate knowledge. The git history
is *narrative* substrate knowledge. They serve different purposes:

- Methodology docs: "what we know about how the substrate should be"
- Git history: "what we did, when, and why at the time"

Commit messages frequently reference these methodology docs by filename
(see `git log --all -- bpd/docs/methodology/`). The reverse is also
true: methodology docs reference specific commits where principles
were first applied.

This crosslinking is intentional. The substrate's understanding of
itself is woven across both representations.

## Reading order for new wizards

For someone encountering the substrate for the first time and wanting
to absorb the methodology:

1. Read this README (you are here).
2. Read [three-correctness-contracts.md](three-correctness-contracts.md)
   for the correctness framework.
3. Read [subsumed-software-fix-catalog.md](subsumed-software-fix-catalog.md)
   for the bug-for-bug compatibility framing.
4. Read [inline-literal-canon.md](inline-literal-canon.md) and
   [fma-contraction-asymmetry.md](fma-contraction-asymmetry.md) for
   the empirical findings about the substrate's natural emit style.
5. Read [regex-retirement-audit-2026-05-18.md](regex-retirement-audit-2026-05-18.md)
   for an example of the methodology applied to a specific repair sequence.

Or just start working on something specific and let the methodology
surface itself as you encounter the questions it addresses. That's how
the principles got named in the first place.

---

*Last updated 2026-05-18 ~14:20 UTC by metayen.
Per Heath's "(b) then (c)" direction.*
