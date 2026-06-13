# Subsumed Software Fix Catalog

**Per Heath's 2026-05-18 ~07:55 UTC framing**: "We want to have bug-for-bug
compatibility with any software we subsume — and we will have flags to
disable the bugs we match."

This document catalogs the named, individually-toggleable defect-repairs
in the substrate's reproduction of subsumed software. Each fix is an
*authored act of substrate authority* — a specific identified defect
with a specific identified repair.

## The substrate-honest framing

The substrate has two responsibilities the prior work had silently conflated:

  1. **Reproducing** an existing implementation (for verification, compatibility,
     and proof-of-subsumption)
  2. **Producing** the correct implementation (the substrate's own analytical
     authority over what the kernel should be)

Conflating them means either:
  - Shipping inherited bugs as "correct" (silent substrate failure)
  - Breaking bit-identical compatibility (silent substrate divergence)

The fix-flag mechanism separates these responsibilities:

  - **Default emit** (empty Fixes list): bug-for-bug compatible with the
    subsumed software. The substrate faithfully reproduces, including
    known defects. This is *evidence of comprehension* — being able to
    reproduce the system exactly (bugs and all) demonstrates we understand
    it at a level deeper than just "approximately correct."

  - **Fixes opted-in via Fixes list**: each named fix disables a specific
    identified defect. The fix is an explicit act of substrate authority:
    "this is a defect in the subsumed software; we have a repair; the repair
    is named and inspectable."

## Substrate API

Three predicates form the substrate's fix-flag layer:

```prolog
%% Discovery: harness asks "what fixes does this kernel know about?"
kernel_available_fixes(+KernelPred, -FixList).

%% Emit: caller asks for a specific fix-list to be applied.
%% kernel_predicate(+Fixes, -Kernel) where Fixes ⊆ available_fixes.
%% kernel_predicate(-Kernel) is sugar for kernel_predicate([], -Kernel).

%% Description: harness reports include human-readable fix descriptions.
fix_description(+FixAtom, -Description).
```

The bit-identical test harness enumerates each kernel's available fixes,
constructs the powerset of fix combinations, and tests each combination's
bit-identical match against multiple references (Ollama, PyTorch CPU,
the substrate's own prior emit, etc.). Results form a cross-tab indexed
by (kernel, fix-combination, reference).

## Naming convention

`fix_<kernel>_<defect_descriptor>` where:
  - `<kernel>` identifies which kernel the fix applies to
  - `<defect_descriptor>` succinctly names what's being repaired

The fix names are *not* the names of the repairs — they're the names of
the *defects* being disabled. Reading the substrate, "with fix X applied"
means "with defect X disabled."

## Catalog

### fix_softmax_phase_inter_race

**Origin**: substrate analysis 2026-05-18 ~07:35 UTC, by metayen during
subtask 4B of the c_raw debt paydown sequence.

**Diagnosis**: llama.cpp's `soft_max_f32` (ggml-cuda/softmax.cu, line 102-120)
calls `block_reduce<MAX>` followed by the strided exp+sum loop, followed
by `block_reduce<SUM>` with no `__syncthreads()` between the two
`block_reduce` invocations.

Each `block_reduce` call writes to and reads from the shared buffer
`buf_iw`. The final operation of `block_reduce` is a warp-synchronous
reduction (`warp_reduce_<op>` via `__shfl_xor_sync`) — this synchronizes
threads within a warp but does NOT synchronize across warps. After
`block_reduce` returns, different warps may be at different points in
the second warp_reduce.

When Phase 2's `block_reduce` begins, its first write to `buf_iw[warp_id]`
could potentially overlap with Phase 1's gather read of `buf_iw[lane_id]`
in a slow warp that hasn't yet exited Phase 1's block_reduce.

**Severity**: theoretical only. No empirical race observed. The strided
exp+sum loop between the two `block_reduce` calls takes substantially
longer than the warp-synchronous tail of Phase 1's reduction, so in
practice all warps complete Phase 1's gather before any warp starts
Phase 2's write. llama.cpp relies on this de facto barrier.

**Fix mechanism**: insert `__syncthreads()` between the strided exp+sum
loop and the call to `block_reduce_sum(tmp, buf_iw)`. This is an explicit
inter-phase barrier, removing the dependency on warp-scheduling assumptions.

**Status**: substrate-authored fix. Not present in llama.cpp upstream.
Available as opt-in for bit-identical-with-llama.cpp-NOT-required scenarios.

**Empirical impact**: TBD per harness measurement. Predicted: no impact
on bit-identical match against llama.cpp (the sync would be the only
source-level divergence); no impact on bit-identical match against
PyTorch CPU softmax; theoretical safety improvement on novel hardware
or compilers that schedule warps more aggressively than current NVIDIA
GPUs.

## How to add a new fix

When a new defect is identified in subsumed software:

  1. **Name it** following the `fix_<kernel>_<defect>` convention
  2. **Add to `kernel_available_fixes/2`** for the affected kernel
  3. **Implement the conditional emit** in the kernel's arity-2 predicate,
     using `member(fix_name, Fixes)` to gate the repair
  4. **Add `fix_description/2`** with full diagnosis matching the format
     in this catalog
  5. **Add a section here** documenting:
     - Origin (who/when/how identified)
     - Diagnosis (precise technical description)
     - Severity (theoretical/empirical/load-bearing)
     - Fix mechanism (what changes in the emit)
     - Status (substrate-authored vs upstream-pending vs disputed)
     - Empirical impact (filled in after harness measurement)

## Connection to other substrate-honesty methodology

The fix-flag pattern parallels the regex audit's intent-vs-approximation
principle (per `bpd/docs/methodology/regex-retirement-audit-2026-05-18.md`):

  - **Regex audit**: each retirement asks "does the regex capture the intent,
    or just an approximation?" Approximation-preserving retirements are
    *named substrate-design positions*, not bugs.

  - **Fix catalog**: each fix asks "is this defect in the subsumed software
    something we should reproduce or repair?" Reproductions are
    *bug-for-bug compatibility positions*; repairs are
    *authored substrate authority*.

Both methodologies treat substrate behavior as a *cataloged claim*, not
opaque code. The substrate becomes the source of long-lived knowledge
about its own design decisions and its understanding of subsumed systems.

---

Authored: metayen 2026-05-18 ~08:05 UTC
Per Heath's bug-for-bug compatibility framing.
