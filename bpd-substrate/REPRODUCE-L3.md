# Reproducing the L3 Census — KernelBench Level-3, Whole-Model Bit-Exact

> **This document is a DRAFT.** The L3 arc is in progress; sections marked
> `<!-- TBD -->` will land as the corresponding rungs complete. The frame
> is ready; the numbers land as they are measured.

## The claim (quote it whole or not at all)

**Of the 30 units emitted into the store, 24 reproduce the benchmark's
own `Model.forward` bit for bit — at the benchmark's own inputs, on
the configuration named below. The store's full accounting:
`24 BIT_EXACT · 2 DIFFERS · 4 SKIPPED · of 30 in the emitted store, of
29 producible` (one unit orphaned: #25).**

**Of the 50 KernelBench Level-3 problems**: `29 producible · 7 in-reach
· 14 gaps · = 50 total`. The ceiling is 50 by construction — every
non-gated problem carries a named capability that would move it into
scope.

*Two sentences, each with its own denominator, neither able to borrow
the other's. The 24 is a fact about the store; the 50 is a fact about
the problem set. A reader who stops at the first sentence has "24 of
30"; a reader who reads only the second has the pipeline's three-way
over the whole set. Neither collapses into "24 of 50."*

*(Numbers pinned to census-eleven, commit `7714824a`. Tool provenance:
`lift_chain b76dbfc0d036 · emit_wrapper 442887c8e672 · auto_pipeline
dec39e108d8c`. Guards: orphan-1-not-producible (#25) · quiescent per
Mavdil's census. Summaries rot; a census commit carries its own guards
that a composed summary silently drops. **A verdict without its tool
hash is a verdict about an unknown artefact** (Mavdil, `18a2df5`):
when a later census fires, update the pin AND the tool-hash triple,
not the numbers alone. See "Reading the result" in
`verification/RUNBOOK-L3.md` for the expected output shape.)*

Five caveats are part of the claim, not fine print:

1. **Bit-exact at the benchmark's inputs does not mean the kernels are
   correct.** A clamp with wrong bounds passes bit-exact when no benchmark
   input reaches the clamp (this class of hazard was caught in L2 by
   static audit, not the bitwise gate). The gate answers a specific
   question; do not read a broader one from it.
2. **The configuration is part of the number.** Bitwise equality is
   substrate-specific: during L2 the same `S/n` vs `S*(1/n)` expression
   differed on 63% of values in strict numpy fp32, 65% in a raw sm_61
   CUDA kernel, and 0% under torch-CUDA. A different torch/CUDA/GPU may
   legitimately give a different number — that is a property of the
   target, not a refutation. Reproduce on this configuration, or measure
   and name yours.
3. **The ceiling is part of the number.** L3's 50 is not a denominator
   over which the ratio is uncapped; it is a ceiling *by construction* —
   every problem outside the current census carries a specific named
   capability the pipeline does not yet possess (e.g. cuDNN-RNN 14-operand
   ULP dossier, channel-shuffle boundary derivation, hardware-OOM under
   the reference implementation). Reading "N of 50" without the ceiling
   frame overstates what the number claims. The ceiling is measured, not
   assumed.
4. **The denominators are separate lines, not one chain.** The claim
   above splits them explicitly. **What we emitted** (`22 bit-exact of
   28 in store · 1 DIFFERS · 5 SKIPPED · of 27 producible`) is a fact
   about the *store*. **The problem set** (`27 producible · 9 in-reach
   · 14 gaps · = 50 total`) is a fact about the *pipeline over all 50
   KernelBench Level-3 problems*. The store may exceed the currently-
   producible count when an earlier emission has since been orphaned
   by pipeline drift — census-ten names `1 unit not producible: 25`,
   which is why `in-store` is 28 while `producible` is 27. `9
   in-reach` names problems the pipeline reads as having no fusable
   epilogue in the walker's own reach-verdict (an honest capability
   limit, not a failure); `14 gaps` names problems the pipeline
   refuses. Every number in both lines is a measured denominator.
   **Merging the two lines into `22 of 50` collapses two different
   questions into one, and reads as a stronger claim than either
   supports.** The doc refuses that merge.
5. **Four positions in the frame: measured · named-boundary · in-reach
   · ceiling.** Not every problem outside BIT_EXACT lives in the same
   class; the frame carries four distinct dispositions and refuses to
   merge them:

   - **Measured** — the gate produced a compare-able number: BIT_EXACT
     (n_diff = 0), DIFFERS (n_diff > 0 with characterization), SKIPPED
     (measured non-emission with named cause such as hardware OOM).
   - **Named-boundary** — the pipeline sees the shape and refuses
     cleanly at a specific named stage, before producing a compare-able
     number. This is measured (the refusal fired) but distinct from
     DIFFERS/SKIPPED because the gate never produced a number to
     compare. The boundary is stronger than a gap precisely because
     it NAMES what would move it INTO SCOPE: a specific substrate arm.
     Live example at census-eleven: **#6 — `folded 0` (zero chains,
     zero kernels emitted, measured absence)**; the pipeline sees the
     shape and produces nothing to gate, and the zero-emission is
     itself the measurement AT the boundary. *Moving into scope is
     what the substrate does; whether it then gates clean is the
     gate's verdict, not the substrate's promise.*
     **A boundary is only honest if something was measured at it**
     (Mavdil's doctrine, refined at census-eleven per the #14 live-fire:
     a "named-boundary" that has no measurement AT the boundary is
     a deferral, not a disposition. #14 DenseNet121 was carried as a
     named-boundary until Mavdil's own six-segment measurement showed
     it BIT_EXACT; the boundary was a defect, not a disposition. The
     doctrine tightens the fourth position: a name without a measurement
     is not a boundary — it is a hypothesis about what the boundary
     might be). *Corollary: the census-ten exemplar (#30 as parameter-
     stage boundary) resolved into DIFFERS at census-eleven when the
     Parameter-naming arm landed — the boundary's own hypothesis got
     measured and moved into the measured column.*
   - **In-reach** — a specific substrate build (container-reach,
     branch-cat, wrapper-env, etc.) would move the problem if it
     lands. Neither measured (not yet emitted or gated) nor part of
     the ceiling-by-construction (the machinery is being actively
     built or scoped). In-reach numbers appear in "What we expect
     next" — a forecast section, not a results section.
   - **Ceiling** — 50 by construction (every non-gated problem carries
     a named capability that would move it into scope). The ceiling
     names the outer bound; specific problems within it live in one of
     the three positions above.

   **Merging positions collapses distinctions the frame depends on.**
   Reading in-reach as measured overstates completion. Reading named-
   boundary as SKIPPED implies the box was the constraint when it
   wasn't. Reading a forecast in a results-shaped sentence is a
   specific class of overclaim; the doc refuses each of these
   collapses.

## The measured configuration

| component | value |
|---|---|
| GPU | NVIDIA Tesla P4 (sm_61, `TORCH_CUDA_ARCH_LIST=6.1`, ~7.4 GB usable) |
| CUDA | 12.8 (`nvcc --fmad=false` — load-bearing, do not drop) |
| torch | 2.7.0 (CUDA 12.8 build) |
| KernelBench | commit `423217d9` (pin your clone to this) |
| python | 3.12 · SWI-Prolog on PATH · gcc 14.3 (also for the `g++` link step) |
| einops | vendored pure-python whl in `lib/` (needed for Mamba2 problems: #48, #49) |

**The enclave path is not needed for reproducibility. The provenance of
these numbers is the configuration** — sm_61, CUDA 12.8, torch 2.7.0,
KernelBench `423217d9`, `--fmad=false`. A different path on the same
configuration should give the same bits. A different configuration may
not, and that is what the caveats are for.

The L3 measured configuration is the same as L2's except for the
einops vendoring: L3's #48/#49 (Mamba2) require einops that older
container images did not ship; the vendored pure-python whl in
`lib/` closes the env-gap without altering the arithmetic.

## The two halves

The claim has two independent parts — reproduce both or you have
reproduced half of it:

- **the pipeline** (this repo: `lib/lift_chain.py`,
  `lib/auto_pipeline.py`, `tools/emit_wrapper.py`,
  `tools/kernel_runner.py`, plus the L3-specific extensions for
  container-driven forwards, recurrence-loops via hook lists, and
  multi-flow replay) emits every kernel from problem source and gates
  each whole model against torch, bitwise;
- **the census** (`verification/RUNBOOK-L3.md`) re-emits the store,
  runs a three-way producibility classification (`producible ·
  in-reach · gaps`, of 50 total), refuses unproducible units, checks
  provenance windows, and prints `BIT_EXACT n · DIFFERS n · SKIPPED n
  · of <in-store>`. The `in-reach` category is the walker's own
  reach-verdict on epilogue fusability (formal name:
  `no-epilogue-in-reach`, per Heath's ruling: the `-YET` in the label
  matters — nothing in L3 has been found genuinely fusion-free) —
  not a claim about the model's shape, but a claim about what the
  walker sees. L3 requires **three steps in order** (re-emit →
  producibility → gate), because `mkproducible3.py` does not rewrite
  existing units and `auto_pipeline.py` emits to stdout — see the
  RUNBOOK for the exact commands. Twice the producibility pass has
  found units the store could no longer justify, and in both cases
  the per-unit gates would have said nothing at all. **The refused
  must not outlive their refusal.**

## Environment and commands

The full recipe lives in `verification/RUNBOOK-L3.md`. In summary:

- **Three steps in order** (not two like L2): re-emit → producibility
  → gate. The re-emit loop is L3-specific; do not skip it.
- **Wait ~2 minutes between producibility and gate.** The gate
  refuses if the manifest is too fresh relative to the store; the
  abort message is the guard working, not a failure.
- **Write the re-emit as a script file, not an inline shell command.**
  Nested quoting has silently eaten the loop before.
- **Dump the verdict to a file and read the file.** Never pipe the
  dict through `cut -c` or similar; truncation is indistinguishable
  from a shorter number.
- **Unit prefix is `l3imp<N>`**, not `l3_<N>` or `imp<N>`.

**Re-emit before you gate, and gate what you emitted, not what you
found on disk** — a stale artefact is indistinguishable from a current
one at gate time; L3 has caught two orphans this way (units present in
the store but refused by the pipeline; both had earlier appeared to
gate clean on stale artefacts). The refused must not outlive their
refusal.

A problem failing any step counts against the total — a finding, not a
footnote.

## The characterized rows

Every not-BIT_EXACT row in census-eleven is named, not deferred. Each
carries the specific capability that would move it into the BIT_EXACT
column, or the specific boundary that prevents it.

**BIT_EXACT (24 of 30), notable rows added at census-eleven:**

- **#14 DenseNet121DenseBlock — 6 segments, 337M elements, `n_diff = 0`
  on every one.** *Previously carried as a named-boundary and it was a
  defect.* The pipeline was swallowing six activation islands into torch
  stages; one arm in the walker later, six kernels + zero gate. Mavdil's
  doctrine from the census: **"A boundary is only honest if something
  was measured at it."** He had `folded 6 · BIT_EXACT · n_diff 0` in his
  own file two days earlier, filed as feasibility and deferred to the
  pipeline's refusal — self-caught retrospectively when the mission-frame
  correction ("never rest at torch-does-it-all") named the failure mode.
- **#28 VisionTransformer — 1 segment, 201k elements, `n_diff = 0`.**
  Second transformer row.

**Prior-census notable rows (still valid):**

- **#29 SwinMLP — the first transformer row** (census-ten headline).
  36 segments, 85,800,960 elements, `n_diff = 0` on every one. Four
  levels of custom-module recursion: `Model → BasicLayer →
  SwinMLPBlock → Mlp`, with modulelist loops, free functions
  (`window_partition` / `window_reverse`), depth-3 Mlp, residual adds,
  and view gymnastics. The gate produced the result; Bocher diagnosed
  the recursion structure; Mavdil independently verified at 36/36;
  Medayek's mutation suite sealed 4/4 across rung-1-BasicLayer /
  rung-2-Block / rung-3-Mlp / gate+mutation. **One independent
  verification**, three roles, all named — not "thrice-confirmed."
  (See Provenance below for the honest record of a superseded artefact
  gated an hour earlier and reported wrong.)
- **#18 SqueezeNet — the branch-cat's census row** (census-nine). 26
  segments (8 FireModules × 3 relu-islands + stem + classifier), 1.3B+
  elements zero on every one; the custom-recursion capability's first
  landing.

**DIFFERS (2 of 30):**

- **#43** — `max_abs = 3.13e-07 against a reference of 0.806`. A named
  precision-shaped tail, not structural. `<!-- verify: mechanism-shape
  of #43's residual against Mavdil's next characterization -->`
- **#30 SwinV2** — moved from named-boundary at census-ten to DIFFERS
  at census-eleven when the Parameter-naming arm landed. Measured:
  `18 segments · 7 clean · 11 differing · max_abs 1.130e+01 vs
  ref_absmax 9.669e+00`. The absolute difference exceeds the reference's
  own magnitude — structural, not rounding. **All eleven differing
  segments are save-scoped residual adds; the six clean ones are
  `gelu`.** The boundary's own hypothesis got measured and moved into
  the measured column (Mavdil, census-eleven).

**SKIPPED (4 of 30), each with a named cause:**

- **#2 — hardware OOM** (unfused reference; 7.4 GB card cannot fit the
  reference model, before comparison is possible). Hardware ceiling,
  not a pipeline gap.
- **#17 — hardware OOM** (7.4 GB card, per Mavdil's census-ten row).
  Same class as #2. *(Note: earlier arcs framed #17 as a return-cat
  refusal; the pipeline now lifts that shape but hits the OOM ceiling
  before gate.)*
- **#25 — orphaned on module_shortcut_residual**. Present in the
  emitted store from an earlier session but the current pipeline
  cannot reproduce it; the orphan guard names it (`1 unit not
  producible: 25`). Store may hold what the pipeline can no longer
  justify — refusal is honest.
- **#8 — launch `rc=700`** (CUDA runtime error). Between census-seven
  (`rc=-3` no-geometry) and census-eight (`rc=700`) the error class
  moved; the row stays honestly refused. Sub-rung: the segment-saved
  launch machinery.

## Named boundary (the fourth position of caveat 5)

**#6 — `folded 0`**. Zero chains, zero kernels emitted. The pipeline
sees the shape and produces nothing to gate; the zero-emission is
itself the measurement AT the boundary. This is a measured absence,
not an assumed one.

**A named boundary is a row, not an absence** (Mavdil's framing per
`48689a76d`, refined at census-eleven per the boundary-honesty
doctrine); a **measured refusal**, not a gap. The gate names where
the pipeline stopped, so the row prints. #6 is:

- **Not DIFFERS** — the gate never produced a mismatched number
  (nothing was emitted to compare)
- **Not SKIPPED-on-hardware** — the box is not the constraint
- **Not a gap in walker's reach-verdict** — the walker sees the shape
- **Not in-reach** — no forecast is required; the refusal is measured
  (`folded 0` IS the measurement)

It is the **fourth position** of caveat 5 (`measured · named-boundary
· in-reach · ceiling`): a MEASURED refusal (measured emptiness) with
a specific named substrate that would move it into scope — a chain-
lifting arm that produces non-empty folded output for #6's shape.
*Moving into scope is what the substrate does; whether it gates clean
is the gate's verdict, not the substrate's promise.* Distinct from
BIT_EXACT/DIFFERS/SKIPPED because the gate never produced a
compare-able number (there was nothing to compare); distinct
from in-reach because the refusal fired (in-reach is not-yet-measured;
named-boundary is measured-and-refused).

*(Historical note: at census-ten, #30 SwinV2 was the named-boundary
exemplar — parameter-stage refusal at `logit_scale is not an
nn.Module`. At census-eleven the Parameter-naming arm landed and #30
moved to DIFFERS (see the DIFFERS section above). The boundary's own
hypothesis got measured, and the hypothesis resolved. That is the
doctrine working exactly as written — a name was a hypothesis; someone
measured; the disposition moved into the measured column. #6 is the
current live exemplar because its emptiness was measured rather than
assumed.)*

## What we expect next (forecast, not results)

*This section is a forecast, not a measurement. Numbers here have not
been gated. When a machinery build lands and its problems gate clean,
they move from here into the measured claim above and the pin updates
to the new census commit.* **A forecast in a results-shaped sentence
is a specific class of overclaim** — the fifth caveat exists to
prevent this section's numbers from being read as measured.

Named machinery in build or scope, per problem class:

- **Custom-module-recursion — LANDED, opens the next tier.** The
  named critical path landed at census-nine (#18 SqueezeNet, the
  branch-cat's census row) and its deepest instance landed at
  census-ten (#29 SwinMLP, 4-level recursion: `Model → BasicLayer →
  SwinMLPBlock → Mlp`). Now in-reach for the next tier: #14 and #28
  (custom-module bodies the pipeline can now walk). Neither is yet
  emitted or gated. `<!-- verify: #14 and #28's exact positions
  against Mavdil's rung notes -->`
- **Branch-cat — LANDED, informs #17/#18/#6.** First form (Return-arm)
  landed for #18's Fire-modules; second form (list-var arm) landed for
  #6 with the CENSUS-INELIGIBLE disposition below. #17 currently
  refuses on hardware-OOM (the pipeline lifts the shape; the 7.4 GB
  card does not fit the reference model).
- **Wrapper-env / multi-flow replay — LANDED for #50, still open for
  #43.** #50 sealed at census-five (12.5M elements, deterministic-
  twice) via the M_FLOW machinery. #43 is now the DIFFERS row at
  census-ten (`max_abs = 3.13e-07 against a reference of 0.806`);
  the residual precision-tail is what remains to characterize.
- **RNN / cuDNN-RNN — verdict-class question genuinely open.** Seven
  problems' zero-ULP path awaits a `T>1` second-mechanism (Mavdil's
  thread narrowing to kernel-selection dispatch: the launcher can be
  made to match torch, but the fully-zero path needs a second-
  mechanism not yet named). Whether the 7 land as BIT_EXACT or as a
  new WITHIN_TOLERANCE verdict class is open. *This entry stands by;
  the doc's caveat structure does not yet name a second verdict
  class. A sixth caveat is pre-drafted and will land IF the verdict
  class does.*
- **SwinV2 residual-adds resolution (informs #30).** At census-eleven,
  the Parameter-naming arm moved #30 from named-boundary to DIFFERS
  (18 segments · 7 clean · 11 differing · save-scoped residual adds
  fail structurally). The gate produced a number; the number differs.
  The next arm would resolve the save-scoped residual-add accumulation
  order so that the 11 differing segments match. Not currently under
  active build; named as substrate when the transformer sub-ladder
  returns.

**CENSUS-INELIGIBLE — reconciliation with census-ten's taxonomy** *(the
structural forecast from earlier arcs; check against Mavdil's cut):*

An earlier forecast entry (based on `77eebc7a2` from the branch-cat
second-form ruling) held that CENSUS-INELIGIBLE should be a fourth
denominator slot. In census-ten's actual output, no separate
CENSUS-INELIGIBLE line appears; the `producible · in-reach · gaps · =
50` three-way still holds. #6 (the specific CENSUS-INELIGIBLE candidate)
is in the `gaps 14` bucket at the of-50 level — not because the
CENSUS-INELIGIBLE ruling was retracted, but because the census tool's
output shape hasn't been extended to print the fourth slot.

**Status of the CENSUS-INELIGIBLE distinction**:
- Ruled distinct from `in-reach` (Iyun `9c49b19a`) because a
  CENSUS-INELIGIBLE problem doesn't become a store unit at all
- Not yet emitted as a separate line in the census tool
- #6 currently counted in `gaps 14` per the tool's classification

The fourth-slot addition is a future taxonomy update, not a current
result. Left in forecast until the census tool prints it.

*(Totals in this section are not summed into a "how many in reach"
figure. A cross-category total would read as a results claim; keeping
each class named with its machinery honors the fifth caveat.)*

## Provenance

**Stranger-clone verification** (Doresh, 2026-09-11): a fresh
`git clone --branch main --single-branch` of this repository, tarred,
shipped to a working directory (`/tmp/stranger` on the enclave),
unpacked, and run through Steps 2 & 3 of `verification/RUNBOOK-L3.md`
reproduced census-four digit-for-digit. Log path on the enclave:
`/tmp/stranger_gate.log`. Result: `BIT_EXACT 16 · DIFFERS 1 · SKIPPED 2
· of 19`, same three not-clean items (#25 DIFFERS with
`n_diff=240,844,785`; #2 SKIP OOM; #8 SKIP producibility mismatch), same
`n_diff` to the digit as `697f2d364`. *(This proves the recipe and
tooling travel; it does not prove a stranger with a bare machine can
reproduce the number — the substrate [kb_level3, P4-class card, CUDA
12.8, torch 2.7.0] must match, per the second caveat. "'I checked the
log' ≠ 'I was told it passed' — the log is citable" — Doresh's
discipline; the log is the artifact, the message about the log is
not.)*

L3-specific transcriptions and their provenance:

- **RNN classes (#33 VanillaRNN, #34 RNN-with-sequence)**. The
  recurrence-term-first-censused emission (fires:256 for #34) verifies
  every timestep independently — the hooks capture lists, one input/
  output pair per fire, all 256 measured rather than induced from one.
  `<!-- TBD: which auto_pipeline.py path handles the fby (recurrence-term)
  emission; verify against the emit source before publication. -->`
- **`fby` term emission**. Named in the census-four report as the
  recurrence operator's first landed pattern. `<!-- TBD: transcription
  source and provenance line -->`
- **QKV rung / wrapper-env / multi-flow replay** (#50 sealed at
  census-five). Lift complete both problems (#43, #50); #50 emits and
  gates via M_FLOW machinery (the linker's symbol-resolution: q/k/v
  across the split + the island rebind). #43 remains as the DIFFERS
  row at census-ten with a residual precision-tail.
- **Branch-cat rung** (#18 SqueezeNet at census-nine, #6 CENSUS-INELIGIBLE
  ruling per `77eebc7a2`). The first form (Return-arm) landed for #18's
  Fire-modules; the second form (list-var arm) landed for #6.
- **#29 SwinMLP — the first transformer row** (census-ten). Four levels
  of custom-module recursion: `Model → BasicLayer → SwinMLPBlock →
  Mlp`. Free functions (`window_partition` / `window_reverse`) walked
  through the reporting bridge. The gate produced 36/36 zero at
  85,800,960 elements. Medayek's mutation suite sealed 4/4 across the
  recursion depth (rung-1 BasicLayer / rung-2 Block / rung-3 Mlp /
  gate + mutation). One independent verification (Mavdil's bench),
  three roles (produce / diagnose / verify), all named.

**One honest record from census-ten** (Mavdil's discipline, per
`48689a76d`): #29 was gated an hour before the census-ten cut against
a superseded artefact and reported STRUCTURALLY WRONG (twelve of
thirty-six clean, max_abs exceeding the reference's own magnitude).
The artefact was rebuilt minutes later and now gates 36/36. Both
readings were correct — the defect was real (a residual-add whose
saved operand resolved to the same tensor as its primary), and the
twelve-clean/twenty-four-differing split decomposed exactly into five
named roots when someone read it. The over-retraction ("I called my
own measurement wrong when only its subject had moved") is the
specific class the "a verdict without its tool hash is a verdict about
an unknown artefact" doctrine (Mavdil `18a2df5`) exists to prevent:
staleness invalidates the verdict's SUBJECT, not its ARITHMETIC. Fixed
on both sides: tool-hashes now recorded with every gate; the builder
sends a one-line notice when a unit moves under a report.

- Every number in the campaign record was produced by a measurement
  someone ran; the doctrine that kept it that way is in `docs/` — start
  with the frame above, it is the load-bearing part.

## The ladder (for context, not part of the claim)

The census progression measured while this document is drafted:

    11 of 17  ·  14 of 20  ·  15 of 19  ·  16 of 19  ·
    17 of 20  ·  18 of 21  ·  19 of 23  ·  20 of 24  ·
    21 of 26  ·  22 of 28  ·  24 of 30

Written as pairs, not bare numbers, because the denominator moved
between rungs: the store shrinks when orphans are deleted, grows when
new problems become producible. A bare `11 → ... → 22` would imply a
fixed denominator and a monotone climb; the climb is real, the
denominator is not fixed. **Each step is a set of specific capabilities
that landed, not a general trend: name the capability that moved the
count, or the count is a slogan.**

Capabilities that moved specific rungs (partial, not exhaustive):
QKV wrapper-env / multi-flow replay (#50); container-reach family
(MobileNetV1 #19, EfficientNetMBConv #21, MobileNetV2 #20 across
27+2+35 = 64 segments, all zero on every one — the family that
reported "nothing to fuse" three days before was reachable once the
walker entered what it had refused to read); custom-module-recursion's
first landing (#18 SqueezeNet, census-nine); custom-module-recursion's
deepest landing (#29 SwinMLP, census-ten, 4-level recursion into the
first transformer row).

## Why this document is a draft

The L3 arc is in progress. Sections marked `<!-- TBD -->` will land as
the corresponding rungs complete. The claim's shape, the config, the
two-halves discipline, the caveats-in-claim, and the ceiling frame are
ready now; the numbers that fill the frame land as they are measured.
Publishing an in-progress claim would violate the discipline the L2
document embodies. Landing the frame in advance ensures that when the
numbers arrive, the document that carries them is honest by
construction.

**Updating the numbers:** the count numbers above are current as of a
specific census commit, cited inline in the claim paragraph. Fresh
census runs will produce different numbers, and each rung has its own
denominator (see the ladder). When updating the numbers, update the
provenance-cite too — the numbers without their commit are a number
without its frame.
