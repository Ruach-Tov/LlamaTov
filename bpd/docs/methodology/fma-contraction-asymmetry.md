# FMA Contraction Asymmetry in Kernel Fusion

**Date discovered**: 2026-05-18 ~18:55 UTC
**Discovered by**: mavchin (root-cause analysis of fusion bit-diffs)
**Captured by**: metayen
**Status**: substrate-design principle — extends the fix-flag taxonomy
**Solution found**: 2026-05-18 ~19:01 UTC — see addendum at end of doc

## The principle

When two operations sit in different kernels (unfused), the compiler
must materialize the intermediate result and apply two separate
roundings. When the same operations sit in one kernel (fused), the
compiler is permitted by IEEE 754-2008 to contract them into a single
fused multiply-add (FMA), producing **one** rounding instead of two.

The fused output is therefore **more accurate**, not equivalent. The
bit difference between fused and unfused emits is the fused kernel
producing less rounding error.

This means kernel fusion creates an asymmetry: bit-identical with
unfused requires disabling the FMA contraction the compiler would
otherwise apply. The substrate must be able to express both choices.

## The empirical mechanism

mavchin's measurement on the `norm → silu → mul → add` fused kernel:

```
WITH FMA contraction (--fmad=true, default):  436 bit-diffs / 2048 elements
WITHOUT FMA contraction (--fmad=false):       0 diffs (bit-identical)
```

The 436 differing positions in the fused output are exactly the
positions where the compiler contracted `residual + silu_val * gate`
into `fma(silu_val, gate, residual)`. Each contraction replaces:

```
tmp = silu_val * gate       (rounded to float32)
out = tmp + residual         (rounded to float32)
```

with:

```
out = fma(silu_val, gate, residual)   (single rounding at the end)
```

Both forms compute the same mathematical expression. The fused form
preserves an extra ~24 bits of intermediate precision before the
final rounding. On average, the rounding error is smaller. The bits
differ, but the fused output is closer to the true real-valued
result.

## What this says about the substrate

Kernel fusion is not a pure performance optimization. It changes
output precision. Any fusion strategy the substrate adopts must make
this explicit.

Two design consequences:

### 1. The fix-flag catalog needs a second category

The fix-flag pattern in `kernel_available_fixes/2` was originally
designed for **defect repairs**: a named bug in subsumed software, off
by default for bug-for-bug compatibility, opt-in to apply the fix.

FMA contraction fits a different category: **precision tradeoffs**.
The "fix" is to disable a precision-improvement to restore bit-
compatibility with a less-precise reference. Default-on for accuracy,
opt-in-off for compatibility.

Both categories use the same `kernel_available_fixes/2` substrate
API. But they have inverted polarity in the default position:

| Category | Default | Opt-in means |
|---|---|---|
| Defect repair | OFF | Apply the fix (improve correctness) |
| Precision tradeoff | ON (or compiler default) | Disable the improvement (match older bits) |

The substrate should record which category each fix belongs to. A
harness running cross-tabulated fix combinations needs to know
whether to expect the default position or the opt-in position is
"better" for each fix.

Proposed `fix_kind/2` predicate (or extend `fix_description` with a
metadata field):

```prolog
fix_kind(fix_softmax_phase_inter_race, defect_repair).
fix_kind(fix_disable_fma_contraction, precision_tradeoff).
```

### 2. Fixes can live below the source layer

Most fixes registered in the substrate so far change source text. The
Harten entropy fix adds c_ast statements. The softmax pre-syncthreads
fix adds a `c_syncthreads` node.

`fix_disable_fma_contraction` is different. It does not change source.
It changes a compilation flag (`--fmad=false`). The same emitted CUDA
produces different SASS based on this flag.

This means the substrate needs a way to attach **compilation hints**
to a kernel, not just source-level modifications. A first sketch:

```prolog
kernel_compilation_hint(KernelPred, Fixes, nvcc_flag, '--fmad=false') :-
    member(fix_disable_fma_contraction, Fixes).
```

The harness's compile fixture would read these hints and pass them to
nvcc/NVRTC/CUDA-oxide. Different backends might need different flag
spellings (`-ffp-contract=off` for LLVM-based paths, `--fmad=false`
for nvcc). The substrate would need per-backend flag mappings.

This is substrate-design work of substantive scope. For now, the fix
exists as a name; the compilation-hint plumbing is queued.

## When to apply which mode

**Default (FMA on)**: kernels that are end-points in the computation
graph and whose output's precision is the actual value of interest.
Inference output tokens, simulation results at t=T_FINAL, anything
the application consumes. FMA's precision improvement makes these
more accurate.

**Compatibility mode (FMA off)**: kernels whose output is being
verified against a known-precision reference. Bit-identical testing
against a reference that itself doesn't have FMA (e.g., a Python
NumPy reference, or an unfused CUDA kernel from a prior version of
the substrate). Here the goal isn't accuracy — it's matching a
specific bit-pattern that the reference produced.

**Mixed**: during development of a new substrate emit, run with FMA
off to verify bit-identity with the source it's translating. Once
matched, switch FMA on for production.

## When NOT to apply

FMA contraction only applies to expressions of the form
`a * b + c` (or commutative variants). Pure multiplication
(`a * b`) is unaffected. Pure addition (`a + b`) is unaffected.
Division, sqrt, transcendentals — unaffected by FMA flag.

For kernels with no FMA-eligible expressions, the flag is moot.
The substrate's fix-flag should not register `fix_disable_fma_
contraction` for kernels that don't actually contain such patterns,
because the harness would waste cycles testing a flag that has no
effect.

A future substrate enhancement could scan a kernel's c_ast for
FMA-eligible patterns and only register the fix if patterns exist.
For now: register manually based on analysis.

## Connection to other methodology principles

This joins:

- **Bug-for-bug as comprehension proof** (when subsuming software with
  known defects). Same substrate machinery, different semantic
  category.

- **Inline-literal canon** (constants inline for LLVM FMA folding).
  Note the connection: the inline-literal canon makes FMA folding
  *possible* (LLVM can fold the constant into the FMA immediate);
  this principle says FMA contraction *also changes precision*.
  Together they constrain how the substrate emits arithmetic:
  constants inline, contraction enabled by default, both for
  precision.

- **Measure, don't assume** (medayek's CFD principle). mavchin's
  finding here is the same pattern: the fusion bit-diffs weren't a
  bug. They were the substrate being more accurate. Measuring (CUPTI
  profiling + `--fmad=false` comparison) revealed what assuming
  ("must be a bug because bits differ") would have led us to
  pathologically reproduce the unfused output.

## Empirical citation

`norm → silu → mul → add` fused kernel, Tesla P4 (sm_61):

- Default compilation: 436 bit-diffs out of 2048 elements vs unfused
- `--fmad=false`: 0 diffs (bit-identical)

The 436 differing positions are exactly the positions where
`residual + silu_val * gate` is FMA-eligible. The substrate-honest
interpretation: the fused kernel is **more accurate** by an average
of one rounding ULP at those positions.

## Future substrate work this opens

1. Add `fix_kind/2` (or equivalent metadata) to distinguish defect
   repairs from precision tradeoffs.

2. Build the compilation-hint plumbing: per-backend flag mappings,
   harness-side flag passing.

3. Register `fix_disable_fma_contraction` for any substrate kernel
   that contains FMA-eligible patterns and is being verified against
   a non-FMA reference.

4. Consider whether `fix_disable_fma_contraction` should be the
   harness's default during development (until bit-identity with
   reference is established) and `default` during production (FMA on
   for precision).

These are substrate-design directions opened by this finding,
appropriate to future sessions.


## Addendum (2026-05-18 ~19:01 UTC) — source-level solution found

Within an hour of writing the body of this doc, mavchin discovered that
the entire compilation-hint detour is unnecessary. The fix can be
expressed at the source layer using CUDA's explicit-rounding intrinsics
`__fmul_rn` and `__fadd_rn` (round-to-nearest, the IEEE 754 default).

### The mechanism

When emitted at stage boundaries within a fused kernel, these intrinsics
force the rounding that would otherwise happen at the DRAM round-trip
of the unfused pipeline:

```c
// Fused kernel with explicit rounding at stage boundaries:
mul_result = __fmul_rn(silu_val, gate[c]);   // matches what unfused
                                              // kernel_B would write to DRAM
out = __fadd_rn(residual[c], mul_result);    // matches what unfused
                                              // kernel_C would write to DRAM
```

Result on the `norm → silu → mul → add` fused kernel:

```
WITH FMA contraction (natural):           436 diffs / 2048
WITH __fmul_rn + __fadd_rn at boundaries: 0 diffs (bit-identical)
WITH --fmad=false:                        0 diffs (also bit-identical)
```

The intrinsics achieve the same bit-identity as the compilation flag
but at the source layer, where the substrate controls c_ast directly.

### Why this is better than --fmad=false

mavchin named three substantive advantages:

1. **Per-stage granularity**, not per-kernel. The substrate can allow
   FMA contraction WITHIN a logical stage (e.g., the polynomial
   evaluation inside `silu` itself, where contraction is fine) but
   prevent it ACROSS stage boundaries (where it would diverge from
   the unfused reference).

2. **Source-level**, not compilation-hint. The substrate expresses the
   choice in c_ast: `c_call('__fmul_rn', [A, B])` instead of
   `c_binop('*', A, B)`. No backend-specific flag plumbing, no
   compile-time fragility.

3. **Portable across backends**. `__fmul_rn` and `__fadd_rn` are
   defined in CUDA's math API (`__device__` intrinsics in
   `cuda_runtime.h`). They work identically under nvcc, NVRTC, and
   CUDA-oxide. The compilation path is irrelevant to their behavior.

### What this revises in the doc above

**The claim "fixes can live below the source layer" is overstated.**
The compilation-hint plumbing sketched earlier (per-backend flag
mappings, harness-side flag passing) was the right answer to a
question that turned out to have a better source-level answer.

Fixes can live at the source layer using rounding intrinsics. The
substrate-honest path for FMA-contraction control is:

```prolog
%% Default emit: natural binop, compiler picks FMA if beneficial
c_binop('*', A, B)        %% emits: A * B  (may contract with adjacent +)

%% Strict-bit-identity emit: explicit round-to-nearest
c_call('__fmul_rn', [A, B])   %% emits: __fmul_rn(A, B)  (no FMA possible)
```

The fix-flag mechanism is unchanged from the original doc: register
`fix_disable_fma_contraction` per kernel that needs it, with
`fix_kind(_, precision_tradeoff)`. But the IMPLEMENTATION of the fix
in the kernel emit is c_ast-level intrinsic substitution, not a
compilation-hint.

### The three modes mavchin named

The substrate now supports three fusion-precision modes, all
expressible at the source layer, all single-kernel-launch, all
DRAM-traffic-free:

| Mode | Emit | Result |
|---|---|---|
| `--strict` | `__fmul_rn` / `__fadd_rn` at stage boundaries | 0 ULP vs unfused reference |
| `--fast` | natural `c_binop('*', ...)` and `c_binop('+', ...)`; compiler contracts to FMA | More accurate than unfused; 436 diffs vs unfused |
| `--allclose` | compiler default behavior | Whatever the compiler does (may differ across backends) |

These three modes correspond cleanly to the fix-flag pattern:

```prolog
%% Mode = --strict
softmax_kernel([fix_disable_fma_contraction], K).

%% Mode = --fast (default)
softmax_kernel([], K).

%% Mode = --allclose (no flag; whatever default behavior produces)
softmax_kernel([], K).   %% (same as --fast in practice)
```

`--strict` and `--fast` are substrate-controlled. `--allclose` is the
"don't care, accept compiler defaults" mode that matches `--fast` in
the current compilers but might diverge in future ones.

### Substantive substrate-honesty observation

The original doc was written based on the empirical finding (436 diffs
disappear with `--fmad=false`) and a plausible-but-not-yet-validated
substrate path (compilation-hint plumbing). Mavchin's source-level
solution arrived an hour later and revealed a cleaner path.

Both findings stay in the doc. The compilation-hint path is preserved
as a fallback for hypothetical situations where source-level intrinsics
aren't available (e.g., a future backend that doesn't expose
`__fmul_rn` equivalents). But the substrate's default path is now the
source-level intrinsic emit.

This is the kind of methodology evolution that the framing question
keeps surfacing: *"What was the originally intended purpose, and what
is the correct form today?"* The compilation-hint approach was the
correct form for ~50 minutes. Then a better form was discovered. The
doc preserves the trajectory so future wizards see how the
understanding evolved.

### Connection to inline-literal canon

These two findings together produce a substantive substrate-design
observation: **the substrate's natural c_ast emit style was already
optimal in two distinct ways**.

- Inline literals (`c_float_f(0.5)` rather than named locals) enable
  LLVM FMA-immediate folding → 25% better IPC
- Natural arithmetic (`c_binop('*', A, B)`) enables FMA contraction
  → more accurate output

Both were discovered, not designed. Together they say: the substrate
should emit arithmetic in its most natural per-operation form and let
the compiler optimize. Optimization-blocking maneuvers (named constant
locals, explicit round-to-nearest) should be **opt-in fixes** for
specific bit-compatibility needs, not default behavior.

The substrate is, in some sense, already wise. Tonight's methodology
work is helping it know that it's wise, so future "cleanups" don't
make it unwise.
