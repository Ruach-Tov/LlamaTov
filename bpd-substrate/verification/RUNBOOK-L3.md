# Reproducing the L3 number

**The claim being reproduced:** of 50 KernelBench Level-3 problems,
16 bit-exact of 19 in the emitted store, of 20 producible, with the
remainder characterized (see REPRODUCE-L3.md). The ceiling is 50 by
construction. *(Numbers current as of census-four, `697f2d364` with
the `ecc0e0230` prose correction.)*

> **This is not a claim that the kernels are correct.** Bit-exactness at
> the benchmark's inputs is silent about inputs the benchmark does not
> supply — the same class of hazard as L2, and the same discipline. **The
> number with its frame is knowledge; without it, marketing.**
>
> **The census counts store units, not problems.** `16 of 19` names what
> we emitted and how many gate; the honest denominator chain is
> `bit-exact · in-store · producible · nothing-to-fuse · gaps · total`
> and every step is a measured number. See REPRODUCE-L3.md for the frame.

## Configuration

The published number was measured on one configuration. **A different
substrate may give a different number**; on the same two fp32
expressions L2 measured `numpy` disagreeing with itself 63% of the time,
a raw CUDA kernel 65%, and torch-on-CUDA 0%. Record yours.

```
GPU arch      sm_61          (TORCH_CUDA_ARCH_LIST=6.1)
GPU memory    ~7.4 GB usable (Tesla P4 — #2 SKIP fires here)
CUDA          12.8
nvcc flags    --fmad=false   ← the gate compiles with FMA contraction OFF
```

**The enclave path is not needed for reproducibility. The provenance of
these numbers is the configuration** — sm_61, CUDA 12.8, torch 2.7.0,
KernelBench `423217d9`, `--fmad=false`. A different path on the same
configuration should give the same bits. A different configuration may
not, and that is what the caveats are for.

## The problem set

Same commit as L2. **A different commit is a different set of problems,
and the count would not be comparable.**

```
https://github.com/ScalingIntelligence/KernelBench
commit 423217d9fda91e0c2d67e4a43bf62f96f6d104f1
```

Level 3 is the 50 problems in `KernelBench/level3/`. **Verify you have
50 files before running anything** — a partial checkout produces a
smaller denominator and a number that looks like a result.

## Layout

Everything is driven by one variable. Defaults match the campaign's
enclave.

```
BPD_ROOT     the checkout root       (default /home/dibbur-patch)
BPD_STORE    emitted kernels         (default $BPD_ROOT/emitted3)
BPD_KB       the problem set         (default $BPD_ROOT/kb_level3)
BPD_PREFIX   unit naming             (default l3imp — units are l3imp<N>.cu)
CUDA_HOME    the CUDA toolchain
TORCH_CUDA_ARCH_LIST=6.1
```

## What you need installed

```
python      3.12          (3.10+ should work; 3.12.11 is what was measured)
torch       2.7.0         built against your CUDA
CUDA        12.8          nvcc must be on PATH or CUDA_HOME must point at it
SWI-Prolog  swipl on PATH — the pipeline's chain solver requires it
gcc         14.3.0        for nvcc's host compilation, and for the g++ link step
einops      pure-python whl vendored in lib/ — closes the Mamba2 env-gap
```

## The three commands (L3 requires three, not two)

**L3 has a third step that L2 does not:** `mkproducible3.py` does *not*
rewrite existing units, and `auto_pipeline.py` emits to *stdout*.
Neither collapses the store into a uniform emission window on its own.
Run the re-emit loop first.

**Run all three from this directory** (`verification/`). Paths are
discovered relative to `BPD_ROOT`.

### Step 1 — Re-emit the store (collapses the store's mtime spread)

**Write this as a script file, not an inline `ssh` command.** Nested
quoting through `ssh → nohup → bash -c` has silently eaten the loop
before, reported success, and rewritten zero units. **A store that is
CURRENT is not the same as a store that is UNIFORM.**

```sh
#!/usr/bin/env bash
for f in $BPD_KB/*.py; do
  p=$(basename "$f" | cut -d_ -f1)
  out=$(timeout 90 python3 lib/auto_pipeline.py "$f" "l3imp$p" 2>/dev/null)
  case "$out" in *__global__*) printf "%s" "$out" > "$BPD_STORE/l3imp$p.cu" ;; esac
done
```

### Step 2 — The producibility pass (writes `producible3.txt`)

```
BPD_KB=$BPD_KB python3 mkproducible3.py    # ~4 min — the precondition
                                            # ★ THEN WAIT ~2 MINUTES ★
```

Output (three-way as of `c1451a4b3`):

```
producible: 20   nothing-to-fuse: 13   gaps: 17        (= 50)
```

### Step 3 — The gate

**⚠ The wait between (2) and (3) is not optional and it is not a bug.**
The gate refuses if the manifest is too fresh relative to the store:

```
ABORT: store written recently.
```

**That refusal is correct** — a verdict must not race an emission. Wait
until the store has been quiet for roughly two minutes, then run the
gate. **An abort here means the guard is working, not that you have
failed.**

```
BPD_PREFIX=l3imp BPD_STORE=$BPD_STORE BPD_KB=$BPD_KB \
  CUDA_HOME=... TORCH_CUDA_ARCH_LIST=6.1 python3 wrapgate3.py
```

**The compile route:** the gate uses `nvcc -c` to compile the kernel
object, then `g++` to link against the versioned `libcudart.so.12`.
Both stages are load-bearing — `nvcc` alone will not produce a runnable
wrapper. *(This compile route cost Bocher a debugging session to find
while the three-step order was being worked out; six failed attempts
preceded it.)*

## Reading the result

Expected output shape:

```
orphan guard: producible-manifest N s old, 0 unit(s) not producible
provenance: quiescent N s, spread N s
BIT_EXACT 16   DIFFERS 1   SKIPPED 2   of 19
```

Every number in the shape is a measured denominator, not a slogan.
`of 19` names the size of the store the census actually ran against;
it is not `of 50`. **See REPRODUCE-L3.md for the full denominator
chain.**

## Three gotchas a newcomer will hit (each cost one cycle)

1. **Nested quoting eats the re-emit loop.** Write Step 1 as a script
   file, not an inline `ssh → nohup → bash -c`. If it reports success
   but the store's mtime spread has not moved, the loop did not run.
2. **Dump the verdict to a file and read the file.** Never pipe the
   verdict dict through `cut -c` or similar — it truncates mid-record
   and you will read a number that was never printed. *(Same class of
   hazard as the "display-limit-is-not-a-measurement-boundary" keeper
   from L2, wearing work clothes: a truncated read of a data structure
   is not a partial result; it's a hidden result.)*
3. **Check the unit naming before blaming the kernels.** The store
   uses `l3imp<N>.cu`, not `l3_<N>.cu`. Guessing the wrong prefix has
   produced false reports of "nine kernels broken" that were actually
   zero kernels found by the gate.

## If your number differs

Report the configuration first, not the count. The most likely causes,
in order:

1. **A different arch or CUDA version** — reduction order is hardware-
   and compiler-dependent
2. **FMA contraction left on** — `--fmad=false` is not optional here
3. **A different KernelBench commit** — the problems must be the ones
   the number was measured against
4. **A stale store** — run Step 1 (re-emit) then Step 2 (producibility)
   before Step 3. If Step 2 reports gaps, the store and the pipeline
   disagree and Step 3's number would be meaningless
5. **einops missing** — if you see #48/#49 failing with import errors,
   check that the vendored whl is on `PYTHONPATH`
6. **The wrong unit prefix** — `l3imp<N>`, not `l3_<N>`

## What has changed since L2's RUNBOOK

- **Three steps instead of two**: L3's `mkproducible3.py` does not
  rewrite existing units and `auto_pipeline.py` emits to stdout;
  the Step 1 re-emit loop collapses the store's mtime spread.
- **Three-way producibility classification** (as of `c1451a4b3`):
  producible / nothing-to-fuse / gaps, not just producible / not.
- **`einops` env-gap closed** by vendored pure-python whl in `lib/`,
  for Mamba2 problems (#48, #49).
- **Unit prefix** is `l3imp<N>` (not `l3_<N>` or `imp<N>`); the gate
  requires this and will refuse to find kernels named otherwise.
- **`nvcc -c` then `g++` link route** against versioned
  `libcudart.so.12` (Bocher's finding — the debugging cost preceded
  the working recipe).

## Related instruments

Also shipped in `verification/` for whole-model verification beyond the
per-segment gate:

- `wholegate.py` — the whole-model gate (the invariant for the
  transform era). Segment verdicts become per-rewrite diagnostics;
  this gate's verdict is what says whether semantics held.
- `wholegate_controls.py` — a positive control that perturbs one
  weight by one ULP and requires the gate to SEE it. A gate that
  only returns BIT_EXACT on a self-check has proved nothing.
- `mutsuite.py` — Medayek's mutation-suite discriminator: whether
  a survival is an equivalent mutant (a fact about the model) or
  a blind gate (a fact about the instrument). Emits the full matrix;
  the equivalent set is an explicit output, never folded silently
  into a denominator.

These are shipped alongside the L3 gate because **a gate published
without its controls asks a reader to trust it** — the whole point of
the public export is that a reader verifies rather than trusts.

## Provenance

The three-step order (re-emit → producibility → gate) was verified by
grepping the scripts, not recalled. The compile route through
`nvcc -c` + `g++` link against versioned `libcudart.so.12` is Bocher's
finding — the debugging session that produced it named the specific
symbol-resolution failure that six earlier attempts had missed. See
REPRODUCE-L3.md for the broader publishability frame.
