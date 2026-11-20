# Independent verification

This directory holds the **gate** — the instruments that check the emitted kernels — kept separate
from the pipeline that produces them. That separation is the point: **no bench sealed its own work.**
Every result in the published number was produced by one party and gated by another.

## The claim, stated exactly

> **100 of 100 KernelBench Level-2 problems emit a fused CUDA kernel whose whole-model output is
> bit-identical to the benchmark's own `Model`, at the benchmark's own inputs, on the configuration
> named below.**

### What it does not say

**It does not say the kernels are correct.** Bit-exactness at the benchmark's inputs is silent about
inputs the benchmark does not supply.

One kernel in this set clamped to the wrong bounds — `(-1,1)` where its source says `(-2,2)` — and
gated BIT_EXACT anyway, because **0.0000% of the benchmark's values reach the clamp**. A *static*,
input-independent constant audit found it (`constaudit.py`); nothing in the gate would have, ever.

> **The number with its frame is knowledge. Without its frame it is marketing.**

## The configuration the number was measured on

Bit-exactness is **substrate-specific**, and this is not a theoretical caution. The same two `fp32`
expressions — `S / n` and `S * (1/n)` — disagree on:

```
numpy fp32 arrays            63% of values
a raw CUDA kernel            65%
torch on CUDA                 0%      (torch never true-divides)
```

Same values, same expressions, three answers. **The substrate decides.** So the number names its own:

```
GPU            Tesla P4, compute capability sm_61
CUDA           12.8   (nvcc release 12.8)
torch          2.7.0  (built against CUDA 12.8)
python         3.12.11
gcc            14.3.0
nvcc flags     --fmad=false      ← FMA contraction OFF; not optional
KernelBench    423217d9fda91e0c2d67e4a43bf62f96f6d104f1
```

**If you reproduce this on different hardware and get a different number, that is not a failed
reproduction — it is a measurement of substrate dependence, and a more interesting result than a
confirmation.** Please report it as such, with your configuration named.

## Running it

See `RUNBOOK.md`. Two commands, in order:

```
BPD_ROOT=/your/checkout python3 mkproducible.py    # ~4 min — the precondition
BPD_ROOT=/your/checkout python3 wrapgate.py        # ~15 min — the batch
```

## Why `mkproducible.py` runs first

A store can hold kernels that gate BIT_EXACT and **can no longer be generated from source**. Twice in
one morning a build left previously-sealed problems unproducible; per-unit gating cannot surface
that, because the artefact is sitting right there, compiling and passing.

```
A STALE ARTEFACT IS INDISTINGUISHABLE FROM A CURRENT ONE AT GATE TIME.
```

`mkproducible.py` re-emits every problem through the current pipeline and writes the manifest the
gate reads. **A hundred containing orphans is not a hundred** — it is ninety-seven results and three
artefacts.

## The gate refuses rather than guesses

It aborts on a recently-written store, on an mtime spread over 900s, and — when the producibility
manifest is absent — **it says so and declines to judge, rather than silently passing.**

Each of those blocked the published number at least once.

> **Do not tune the thresholds until the answer looks right.**
