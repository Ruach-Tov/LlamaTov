#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_layer2_primitives.py — Bit-identity verification for bpd_residual_add_cpu
and bpd_concat_channel_cpu.

Per Heath's discipline 2026-05-20 ~05:30 UTC: 0 ULP per stage before composing
further. Before the orchestrator can dispatch Layer 2 (C3 module) and the FPN
concat layers, the new primitives must be bit-identical with PyTorch CPU.

Tests:
  1. bpd_residual_add_cpu vs torch.add — random tensors, YOLO-typical shapes
  2. bpd_concat_channel_cpu vs torch.cat — 2-input (C3) and 4-input (SPPF) cases

Both kernels should be trivially BIT_IDENTICAL with PyTorch CPU since:
  - residual_add: one FADD per element, no FMA possible; IEEE 754 a + b is
    deterministic
  - concat: pure memcpy, no arithmetic

If any divergence appears, the β-discipline applies: shrink, disassemble,
narrow, fix, verify.
"""
import ctypes
import os
import sys
import numpy as np

try:
    import torch
except ImportError:
    sys.exit("torch required")

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)


def ulp(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), int(d.size)


def report(label, ref, sub):
    mu, nd, nt = ulp(ref, sub)
    abs_diff = float(np.abs(ref.astype(np.float32) - sub.astype(np.float32)).max())
    if mu == 0:
        print(f"  {label:<55} BIT_IDENTICAL  ({nt} elements, 0 ULP)")
        return True
    else:
        print(f"  {label:<55} DIVERGENT  max {mu} ULP, {nd}/{nt} diffs, abs err {abs_diff:.3e}")
        return False


def load_lib():
    so_path = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
    if not os.path.exists(so_path):
        sys.exit(f"{so_path} not found — build it first with `gcc -O2 -shared -fPIC -o build/bpd_cpu.so bench/bpd_cpu.c -lm`")
    lib = ctypes.CDLL(so_path)
    # bpd_residual_add_cpu(a, b, output, n)
    lib.bpd_residual_add_cpu.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.bpd_residual_add_cpu.restype = None
    # bpd_concat_channel_cpu(inputs**, c_each*, n_inputs, N, H, W, output)
    lib.bpd_concat_channel_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # inputs (array of pointers)
        ctypes.POINTER(ctypes.c_int),      # c_each
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p]
    lib.bpd_concat_channel_cpu.restype = None
    return lib


def test_residual_add(lib):
    """Test bpd_residual_add_cpu against torch.add."""
    print("\n=== bpd_residual_add_cpu ===")
    print("(elementwise add: out[i] = a[i] + b[i])")

    test_shapes = [
        (16,),                       # trivial
        (1, 16, 320, 320),           # YOLOv5n Layer 0 output
        (1, 32, 160, 160),           # YOLOv5n Layer 1 output
        (1, 64, 80, 80),             # C3/4 backbone
        (1, 128, 40, 40),            # C3/6 backbone
        (1, 256, 20, 20),            # C3/8 + SPPF
    ]
    rng = np.random.default_rng(42)
    all_pass = True
    for shape in test_shapes:
        a = (rng.standard_normal(shape) * 2.0).astype(np.float32)
        b = (rng.standard_normal(shape) * 2.0).astype(np.float32)

        # Substrate
        a_c = np.ascontiguousarray(a)
        b_c = np.ascontiguousarray(b)
        out_sub = np.zeros_like(a_c)
        lib.bpd_residual_add_cpu(a_c.ctypes.data, b_c.ctypes.data,
                                  out_sub.ctypes.data, a.size)

        # PyTorch reference
        out_pt = (torch.from_numpy(a) + torch.from_numpy(b)).numpy()

        shape_str = "x".join(str(s) for s in shape)
        ok = report(f"shape {shape_str}", out_pt, out_sub)
        all_pass = all_pass and ok
    return all_pass


def test_concat_channel(lib):
    """Test bpd_concat_channel_cpu against torch.cat(dim=1)."""
    print("\n=== bpd_concat_channel_cpu ===")
    print("(channel-axis concat: NCHW, axis=1)")

    rng = np.random.default_rng(137)
    all_pass = True

    # YOLOv5n concat patterns:
    #   C3 internal:   2-input  (bottleneck path + shortcut path)
    #   FPN concat:    2-input  (upsample + skip)
    #   SPPF:          4-input  (input + maxpool*3)
    test_cases = [
        # (case_label, list of channel counts, N_batch, H, W)
        ("C3 internal — 16 + 16 ch",      [16, 16],     1, 320, 320),
        ("FPN concat P4 — 128 + 128 ch", [128, 128],   1, 40, 40),
        ("SPPF — 256 ch ×4",              [256]*4,      1, 20, 20),
        ("trivial 4-input small",         [3, 5, 7, 11], 1, 4, 4),
    ]

    for label, c_each, N, H, W in test_cases:
        n_inputs = len(c_each)
        inputs_np = [
            (rng.standard_normal((N, c, H, W)) * 1.5).astype(np.float32)
            for c in c_each
        ]

        # Substrate
        inputs_c = [np.ascontiguousarray(t) for t in inputs_np]
        input_ptrs = (ctypes.c_void_p * n_inputs)(
            *[t.ctypes.data for t in inputs_c]
        )
        c_each_arr = (ctypes.c_int * n_inputs)(*c_each)
        C_total = sum(c_each)
        out_sub = np.zeros((N, C_total, H, W), dtype=np.float32)
        lib.bpd_concat_channel_cpu(input_ptrs, c_each_arr, n_inputs,
                                     N, H, W, out_sub.ctypes.data)

        # PyTorch reference
        out_pt = torch.cat([torch.from_numpy(t) for t in inputs_c], dim=1).numpy()

        ok = report(label, out_pt, out_sub)
        all_pass = all_pass and ok
    return all_pass


def main():
    print("=" * 78)
    print("verify_layer2_primitives.py — bit-identity for residual_add + concat_channel")
    print("PyTorch path: MKL-DNN disabled, single-threaded (controllable reference)")
    print("=" * 78)
    lib = load_lib()

    add_pass = test_residual_add(lib)
    concat_pass = test_concat_channel(lib)

    print()
    print("=" * 78)
    if add_pass and concat_pass:
        print("ALL PASS: Layer 2 primitives are BIT_IDENTICAL with PyTorch CPU.")
        print("Ready for orchestrator dispatch (C3 module + FPN concat).")
        return 0
    else:
        print("FAIL: divergence surfaced. Apply β-discipline: shrink, disassemble, narrow, fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
