# Reproducing the Hundred — KernelBench Level-2, Whole-Model Bit-Exact

## The claim (quote it whole or not at all)

**100 of 100 KernelBench Level-2 problems reproduce the
benchmark's own `Model.forward` bit for bit — at the benchmark's
own inputs, on the configuration named below.**

Two caveats are part of the claim, not fine print:

1. **Bit-exact at the benchmark's inputs does not mean the kernels
   are correct.** A clamp with wrong bounds passes bit-exact when
   no benchmark input reaches the clamp (this happened — a static
   audit caught it; the bitwise gate could not).
2. **The configuration is part of the number.** Bitwise equality
   is substrate-specific: during this campaign the same `S/n` vs
   `S*(1/n)` expression differed on 63% of values in strict numpy
   fp32, 65% in a raw sm_61 CUDA kernel, and 0% under torch-CUDA.
   A different torch/CUDA/GPU may legitimately give a different
   number — that is a property of the target, not a refutation.
   Reproduce on this configuration, or measure and name yours.

## The measured configuration

| component | value |
|---|---|
| GPU | NVIDIA Tesla P4 (sm_61) |
| CUDA | 12.8 (`nvcc --fmad=false` — load-bearing, do not drop) |
| torch | 2.7.0 (CUDA 12.8 build) |
| KernelBench | commit `423217d` (pin your clone to this) |
| python | 3.12 · SWI-Prolog on PATH · gcc |

## The two halves

The claim has two independent parts — reproduce both or you have
reproduced half of it:

- **the pipeline** (this repo: `lib/lift_chain.py`,
  `lib/auto_pipeline.py`, `tools/emit_wrapper.py`,
  `tools/kernel_runner.py`) emits every kernel from problem
  source and gates each whole model against torch, bitwise;
- **the census** (`verification/RUNBOOK.md` — the independent
  tally) re-emits the store from scratch, refuses unproducible
  units, checks provenance windows, and prints
  `BIT_EXACT n · DIFFERS n · SKIPPED n · of 100`. Run
  `mkproducible.py` FIRST, and do not tune its thresholds until
  the answer looks right — the refusals are features.

## Environment

```sh
export CUDA_HOME=/path/to/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export BPD_KB=/path/to/KernelBench/KernelBench/level2
```

## One problem, end to end

```sh
python3 lib/auto_pipeline.py $BPD_KB/14_*.py imp14 > imp14.cu
python3 tools/emit_wrapper.py $BPD_KB/14_*.py imp14 imp14.cu w14.py
python3 w14.py   # → {'seg0': {'N': 1024, 'n_diff': 0, 'max_ulp': 0, ...}}
```

`n_diff: 0` on every segment = the whole model is bit-exact.

## All hundred

Loop the three commands over `$BPD_KB/*.py` with pid `imp<N>`:
every problem must emit (no `[GAP]`), every wrapper must build,
every verdict must read `n_diff: 0`. **Re-emit before you gate,
and gate what you emitted, not what you found on disk** — a stale
artefact is indistinguishable from a current one at gate time.
Then run the census for the independent tally. A problem failing
any step counts against the total — a finding, not a footnote.

## Provenance

- The reduction orders in `auto_pipeline.py` are transcriptions
  of torch's own CUDA kernels (fp32, fmad off, reciprocal-multiply
  mean, the vectorized reduce with its per-row shift head);
  `docs/reference/Reduce_cuh_v2.7.0_reference.cuh` is the vendored
  source they were read from, and the file's comments carry the
  per-order provenance.
- Every number in the campaign record was produced by a
  measurement someone ran; the doctrine that kept it that way is
  in `docs/` — start with the frame above, it is the load-bearing
  part.
