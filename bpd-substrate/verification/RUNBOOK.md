# Reproducing the number

**The claim being reproduced:** all 100 KernelBench Level-2 problems emit a fused CUDA kernel whose
whole-model output is bit-identical to the benchmark's own `Model`, **at the benchmark's own
inputs**.

> **This is not a claim that the kernels are correct.** Bit-exactness at the benchmark's inputs is
> silent about inputs the benchmark does not supply. One kernel in this set passed while clamping to
> the wrong bounds, because 0.0000% of the benchmark's values reach the clamp — a *static* constant
> audit found it, and nothing in the gate would have. **The number with its frame is knowledge;
> without it, marketing.**

## Configuration

The published number was measured on one configuration. **A different substrate may give a different
number, and this is not hypothetical:** on the same two fp32 expressions, `numpy` disagrees with
itself 63% of the time, a raw CUDA kernel 65%, and torch-on-CUDA 0% — because each implements the
operation differently. Record yours.

```
GPU arch      sm_61          (TORCH_CUDA_ARCH_LIST=6.1)
CUDA          12.8
nvcc flags    --fmad=false   ← the gate compiles with FMA contraction OFF
```

## The problem set

The number is measured against a **specific commit** of KernelBench. A different commit is a
different set of problems, and the count would not be comparable.

```
https://github.com/ScalingIntelligence/KernelBench
commit 423217d9fda91e0c2d67e4a43bf62f96f6d104f1
```

Clone and check out that commit exactly:

```
git clone https://github.com/ScalingIntelligence/KernelBench
git -C KernelBench checkout 423217d9fda91e0c2d67e4a43bf62f96f6d104f1
```

Level 2 is the 100 problems in `KernelBench/level2/`. **Verify you have 100 files before running
anything** — a partial checkout produces a smaller denominator and a number that looks like a
result.

## Layout

Everything is driven by one variable. Defaults match the campaign's enclave.

```
BPD_ROOT     the checkout root       (default /home/dibbur-patch)
BPD_STORE    emitted kernels         (default $BPD_ROOT/emitted)
BPD_KB       the problem set         (default $BPD_ROOT/kb_level2)
CUDA_HOME    the CUDA toolchain
```

## What you need installed

The configuration table above names what the number was measured *on*. This is what you must
**install** to run anything at all:

```
python      3.12          (3.10+ should work; 3.12.11 is what was measured)
torch       2.7.0         built against your CUDA — a CPU-only torch will not run this
CUDA        12.8          nvcc must be on PATH or CUDA_HOME must point at it
SWI-Prolog  swipl on PATH — the pipeline's chain solver requires it
gcc         14.3.0        for nvcc's host compilation
```

**A missing `torch` is the first thing a fresh machine hits.** There is no fallback path; the gate
compares against torch's own kernels by construction.

## The two commands

**Run both from this directory** (`verification/`). Paths are discovered relative to `BPD_ROOT`, so
the working directory only needs to contain the scripts.

```
BPD_ROOT=/your/checkout python3 mkproducible.py    # ~4 min — the precondition
                                                    # ★ THEN WAIT ~2 MINUTES ★
BPD_ROOT=/your/checkout python3 wrapgate.py        # ~15 min — the batch
```

**⚠ The wait is not optional and it is not a bug.** `mkproducible.py` re-emits the whole store, so
the store has *just been written* when it finishes. The gate then refuses:

```
ABORT: store written recently.
```

**That refusal is correct** — a verdict must not race an emission — but running the two commands
back to back triggers it every time. Wait until the store has been quiet for roughly two minutes,
then run the gate. **An abort here means the guard is working, not that you have failed.**


```
python3 mkproducible.py      # 1. writes producible.txt; ~4 min
python3 wrapgate.py          # 2. the batch;             ~15 min
```

**Run them in that order and do not skip the first.** `mkproducible.py` re-emits every problem
through the current pipeline and records which ones it can still produce. It has three jobs:

1. it is the **producibility precondition** — a store may hold kernels that gate bit-exact and can
   no longer be generated from source. *A hundred containing orphans is not a hundred.*
2. it is a **regression test for the generator** — twice in one morning a build left previously
   sealed problems unproducible, which per-unit gating cannot surface because the artefact is
   sitting right there.
3. it leaves the store in **one emission window**, which the gate requires.

## Reading the result

```
BIT_EXACT 100   DIFFERS 0   SKIPPED 0   of 100
```

The gate refuses to print rather than print something unattributable. It aborts if:

- **the store was written recently** — a verdict must not race an emission
- **the mtime spread exceeds 900s** — a mixed store cannot say which census a verdict belongs to
- **the producibility manifest is missing** — the orphan guard then says so and *declines to judge*
  rather than silently passing

Any of these is a real refusal. **Do not tune the thresholds until the answer looks right.**

## If your number differs

Report the configuration first, not the count. The most likely causes, in order:

1. **a different arch or CUDA version** — reduction order is hardware- and compiler-dependent
2. **FMA contraction left on** — `--fmad=false` is not optional here
3. **a different KernelBench commit** — the problems must be the ones the number was measured against
4. **a stale store** — run `mkproducible.py` first; if it reports gaps, the store and the pipeline
   disagree and the gate's number would be meaningless
