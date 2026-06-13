#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""regenerate.py — Deterministically regenerate cross-backend fixtures.

Run from this directory: `python3 regenerate.py`. Bit-identical given
the same Python + numpy + torch versions and the seed used per fixture.

For each fixture:
  1. Generate input with a documented seed
  2. Compute the PyTorch CPU reference (this IS cell [3] of the
     7-cell matrix per mavchin's framing: Python host, CPU dispatch)
  3. Save both as .npy files in this directory

The matrix harness's per-backend tests read the input fixture, dispatch
the kernel, and compare the result against the corresponding cell file
(typically the cell-[3] PyTorch reference, but may be cross-compared
against other cells as the matrix fills out).

## Fixture naming convention (post 21:46 UTC reframe)

  <op>_<shape>.npy                    — shared input vector
  <op>_<shape>_cell3_python_cpu.npy   — cell [3]: Python host, CPU
                                         dispatch (PyTorch reference)
  <op>_<shape>_cell<N>_<lang>_<dev>.npy — other cells as they land:
                                         cell2_c_gpu, cell4_python_gpu,
                                         cell5_rust_cpu, cell6_rust_gpu,
                                         cell8_cudaoxide_gpu

Each name reveals (a) which kernel computation, (b) shape, (c) which
cell of the 7-cell matrix the output represents. The name IS the
structural identity, not a convenience label.

Do NOT regenerate without intent. The fixtures are pinned in git so
every commit's tests run on bit-identical inputs.

Per Heath's cross-language correctness matrix vision (2026-05-17),
this is part of 1.c.i: design + ship the per-backend output format
and write the deterministic input fixtures shared across all
host-language backends.

Author: metayen 2026-05-17
"""

import os
import sys
import numpy as np
import torch

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add bpd/lib to path so we can import cpu_references for the expected outputs.
sys.path.insert(0, os.path.join(FIXTURE_DIR, '..', '..', 'lib'))
import cpu_references as cr


def save_fixture(name: str, array: np.ndarray) -> None:
    """Save a numpy array as a versioned fixture file."""
    path = os.path.join(FIXTURE_DIR, name)
    np.save(path, array)
    print(f"  wrote {name}  shape={array.shape}  dtype={array.dtype}")


def regenerate_reduction_fixtures():
    """Reduction-family fixtures.

    All reduction ops share the input fixture (same [8, 16] random matrix).
    Each op gets its own cell-[3] output from the corresponding
    cpu_reference (Python host, CPU dispatch, PyTorch implementation).

    Per mavchin's 7-cell-per-kernel framing (2026-05-17 intercom 21:42 UTC),
    the matrix harness explicitly names which cell each output represents:
      _8x16.npy                    — shared input vector
      _8x16_cell3_python_cpu.npy   — cell [3]: Python host, CPU dispatch
                                      (PyTorch implementation via cpu_reference_*)

    Cell [2] (C host, GPU dispatch) outputs are produced by the C harness
    binary on enclave and saved as <op>_c_output.npy, then compared to
    cell [3] via matrix_verify.py. Other cells (Python GPU, Rust CPU/GPU,
    cuda-oxide GPU) will be added as their substrate lands.
    """
    print("Reduction family:")
    torch.manual_seed(42)
    x = torch.randn(8, 16, dtype=torch.float32)
    save_fixture('ggml_sum_rows_8x16.npy', x.numpy())

    # Reuse the same input across all reduction op_kinds. Substrate-honest:
    # every op gets compared on identical data. Per-op cell-[3] files
    # differentiate by the reduction performed.
    #
    # Ops with output shape [outer] (1D, captured by harness_reduction.cu):
    #   ggml_sum_rows, ggml_mean, ggml_max, ggml_min, ggml_argmax, ggml_argmin
    # Ops with output shape [outer, N] (2D, deferred — needs harness extension):
    #   ggml_cumsum, ggml_cumprod
    save_fixture('ggml_mean_8x16.npy', x.numpy())
    save_fixture('ggml_max_8x16.npy', x.numpy())
    save_fixture('ggml_min_8x16.npy', x.numpy())
    save_fixture('ggml_argmax_8x16.npy', x.numpy())
    save_fixture('ggml_argmin_8x16.npy', x.numpy())

    # Cell [3] outputs — Python host, CPU dispatch, PyTorch reference.
    y_sum = cr.cpu_reference_sum_rows(x)
    save_fixture('ggml_sum_rows_8x16_cell3_python_cpu.npy', y_sum.numpy())

    y_mean = cr.cpu_reference_mean(x)
    save_fixture('ggml_mean_8x16_cell3_python_cpu.npy', y_mean.numpy())

    y_max = cr.cpu_reference_max(x)
    save_fixture('ggml_max_8x16_cell3_python_cpu.npy', y_max.numpy())

    y_min = cr.cpu_reference_min(x)
    save_fixture('ggml_min_8x16_cell3_python_cpu.npy', y_min.numpy())

    y_argmax = cr.cpu_reference_argmax(x)
    save_fixture('ggml_argmax_8x16_cell3_python_cpu.npy', y_argmax.numpy())

    y_argmin = cr.cpu_reference_argmin(x)
    save_fixture('ggml_argmin_8x16_cell3_python_cpu.npy', y_argmin.numpy())


def regenerate_activation_fixtures():
    """Activation-family fixtures (elementwise unary).

    Per mavchin's 7-cell-per-kernel framing, activations have all 7
    executable cells in scope (no cell-[1] CPU-substrate problem like
    reductions, because per-element ops are trivial in every host
    language).

    Per Heath's A1 (2026-05-17 22:00 UTC): use what's natural for
    elementwise (1D shapes), BUT include esoteric/oddball sizes for
    excellent test coverage. Boundary conditions to probe:
      128   — exactly half a thread block
      256   — exactly one thread block
      257   — one element over a block (tests the if(i>=n) guard
              AND the (N+255)/256 ceil-division)
      1000  — prime-ish, non-aligned size
      1023  — one under a block-aligned boundary
      1024  — exactly 4 thread blocks

    The 5 activations all share the same input vector at each size,
    so the cell-[3] outputs differentiate by the activation function.

    Per Heath's A3: --strict-maxxing where the physics allows.
      relu:   pure conditional, no transcendental, expect BIT-IDENTICAL
              between cell [2] and cell [3] (--strict).
      silu, sigmoid, tanh, gelu: transcendental functions, ≤2 ULP
              between CPU and GPU (per mavchin's empirical boundary).
    """
    print("Activation family:")
    torch.manual_seed(42)

    sizes = [128, 256, 257, 1000, 1023, 1024]
    # Per the 2026-05-17 gelu terminology investigation, "k_gelu" is
    # split into two canonical forms with distinct cell-[3] references
    # (see lib/terminology.pl). Each canonical form gets its own
    # fixtures for cross-axis verification.
    activations = [
        ('k_silu',      cr.cpu_reference_silu),
        ('k_sigmoid',   cr.cpu_reference_sigmoid),
        ('k_relu',      cr.cpu_reference_relu),
        ('k_gelu_tanh', cr.cpu_reference_gelu_tanh),  # ggml/Ollama form
        ('k_gelu_erf',  cr.cpu_reference_gelu_erf),   # exact form
        ('k_tanh',      cr.cpu_reference_tanh),
    ]

    for n in sizes:
        # Shared input vector at size n. New seed per size to keep each
        # size's data distinct (otherwise size-1024 and size-1023 would
        # share prefix data; non-issue for correctness, but cleaner
        # tests if each size has independent random data).
        torch.manual_seed(42 + n)
        x = torch.randn(n, dtype=torch.float32)
        for op_name, _ in activations:
            save_fixture(f'{op_name}_{n}.npy', x.numpy())
        # Per-activation cell [3] (Python host CPU, PyTorch reference).
        for op_name, ref_fn in activations:
            y = ref_fn(x)
            save_fixture(f'{op_name}_{n}_cell3_python_cpu.npy', y.numpy())


def main():
    print(f"Regenerating fixtures in {FIXTURE_DIR}")
    print()
    regenerate_reduction_fixtures()
    print()
    regenerate_activation_fixtures()
    print()
    print("Done. Fixtures are deterministic given seeded RNGs.")


if __name__ == '__main__':
    main()
