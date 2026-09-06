# Reproducing LlamaTov Results

This document describes how to **verify** the 0-ULP bit-identity claims
made in this repository. You do not need to understand the generation
pipeline — only the verification path.

## Prerequisites

- Python 3.10+
- PyTorch 2.6+ (CPU is sufficient for verification)
- CUDA toolkit (for GPU kernel verification, optional)
- SWI-Prolog (for the substrate)

```bash
pip install torch  # CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick verification (< 5 minutes)

```bash
cd bpd-substrate
make verify FOCUS=cpu          # CPU bit-identity: all kernels should be 0 ULP
make verify FOCUS=cublas       # cuBLAS bit-identity (requires CUDA)
```

The `verify` target runs differential comparison against reference
implementations. Each kernel is tested with deterministic inputs and
checked for **exact bit-identity** (0 ULP).

> ⚠️ cuBLAS results may vary by device (SM count, L2 size, driver version).
> See [issue #7](https://github.com/Ruach-Tov/LlamaTov/issues/7) for details.

## KernelBench L1 verification

The auto-lift pipeline verifies that all 65 KernelBench L1 kernels can
be lifted from PyTorch source to BPD Logtalk facts:

```bash
cd bpd
PYTHONPATH=tests:lib python3 lib/auto_lift_registry.py \
    --registry pytorch_kernel_library --summary
```

Expected: `65/65 lifted (0 empty)` — one program, no agent in the loop.

## What "0 ULP" means

A result is **0 ULP** (zero units in the last place) when the emitted
kernel's output is **bit-identical** to the reference implementation's
output for every test input. Not "close" — identical. The same bytes.

## Performance claims

The headline performance (+16.7%, 168 tok/s on Tesla P4) was measured
in June 2026. Performance numbers are **measured once, not continuously
re-validated**. The 0-ULP correctness claims ARE continuously re-gated
by the validation pipeline.

## Honest negatives

- cuBLAS bit-identity depends on device characteristics, not just
  compute capability (`sm_XX`). Two cards of the same `sm_86` can
  disagree deterministically. See issue #7.
- The Prefix-VBR paper was retracted due to a bit-counting error.
  See commit history.

---

*Ruach Tov Collective, 2026.*
