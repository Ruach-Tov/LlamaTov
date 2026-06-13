#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_fusion_F7.py — verify bpd_conv2d_bias_sigmoid_fused_cpu BIT_IDENTICAL.

F7 fuses Conv + Bias + Sigmoid in a single kernel. Used in the YOLOv5 Detect
head where the 3 detection convs have bias and no BN.

Reference (unfused chain, already verified BIT_IDENTICAL with PyTorch CPU):
  bpd_conv2d_full_cpu(in, w, bias, out, ...)  -> conv_out with bias added
  bpd_sigmoid_cpu(conv_out, sigmoid_out, n)    -> sigmoid applied elementwise

Fused (F7):
  bpd_conv2d_bias_sigmoid_fused_cpu(in, w, bias, out, ...)
    -> sigmoid(GEMM(w, im2col(in)) + bias[co]) all in one kernel

Test shapes match YOLOv5n Detect head:
  P3 detect: M=255, K=64,  N=80*80   (1x1, bias)
  P4 detect: M=255, K=128, N=40*40
  P5 detect: M=255, K=256, N=20*20
"""
import ctypes
import os
import sys

import numpy as np

SO = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")


def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


def main():
    lib = ctypes.CDLL(SO)
    lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
    lib.bpd_conv2d_full_cpu.restype = None
    lib.bpd_sigmoid_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
    lib.bpd_sigmoid_cpu.restype = None
    has_f7 = hasattr(lib, 'bpd_conv2d_bias_sigmoid_fused_cpu')
    if has_f7:
        lib.bpd_conv2d_bias_sigmoid_fused_cpu.argtypes = (
            [ctypes.c_void_p]*4 + [ctypes.c_int]*11)
        lib.bpd_conv2d_bias_sigmoid_fused_cpu.restype = None

    print(f"Substrate: {SO}")
    print(f"F7 kernel available: {has_f7}")
    print()

    if not has_f7:
        print("⚠️  bpd_conv2d_bias_sigmoid_fused_cpu not yet built. Build first.")
        sys.exit(2)

    # YOLOv5n Detect-head shapes + a few generic test cases
    shapes = [
        # (label, N, Cin, H, W, Cout, kH, kW, stride, pad)
        ("P3 detect 1x1",      1,  64, 80, 80, 255, 1, 1, 1, 0),
        ("P4 detect 1x1",      1, 128, 40, 40, 255, 1, 1, 1, 0),
        ("P5 detect 1x1",      1, 256, 20, 20, 255, 1, 1, 1, 0),
        ("Generic 3x3 s=1",    1,  16, 32, 32,  32, 3, 3, 1, 1),
        ("Generic 1x1",        1,  64, 16, 16, 128, 1, 1, 1, 0),
    ]

    print(f"{'Shape':<22} {'Input shape':<22} {'Output shape':<22} {'Status':<25}")
    print("-" * 92)

    rng = np.random.default_rng(2026)
    all_pass = True
    for label, N, Cin, H, W, Cout, kH, kW, stride, pad in shapes:
        x = (rng.standard_normal((N, Cin, H, W)) * 0.3).astype(np.float32)
        w = (rng.standard_normal((Cout, Cin, kH, kW)) * (1.0/np.sqrt(Cin*kH*kW))).astype(np.float32)
        bias = (rng.standard_normal(Cout) * 0.1).astype(np.float32)

        # Reference: bpd_conv2d_full_cpu (with bias) -> bpd_sigmoid_cpu
        H_out = (H + 2*pad - kH) // stride + 1
        W_out = (W + 2*pad - kW) // stride + 1
        conv_out = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
        lib.bpd_conv2d_full_cpu(
            x.ctypes.data, w.ctypes.data, bias.ctypes.data,
            conv_out.ctypes.data,
            N, Cin, H, W, Cout, kH, kW,
            stride, stride, pad, pad, 1, 1, 1)
        ref = np.zeros_like(conv_out)
        lib.bpd_sigmoid_cpu(conv_out.ctypes.data, ref.ctypes.data, conv_out.size)

        # Fused F7
        fused = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
        lib.bpd_conv2d_bias_sigmoid_fused_cpu(
            x.ctypes.data, w.ctypes.data, bias.ctypes.data,
            fused.ctypes.data,
            N, Cin, H, W, Cout, kH, kW,
            stride, stride, pad, pad)

        max_ulp, n_diff, n_total = ulp_distance(ref, fused)
        in_s = f"({N},{Cin},{H},{W})"
        out_s = f"({N},{Cout},{H_out},{W_out})"
        if max_ulp == 0:
            status = "BIT_IDENTICAL"
        else:
            status = f"DIVERGENT max={max_ulp} n={n_diff}/{n_total}"
            all_pass = False
        print(f"{label:<22} {in_s:<22} {out_s:<22} {status:<25}")

    print()
    if all_pass:
        print("✅ F7 fused kernel is BIT_IDENTICAL with the unfused chain across all shapes.")
        sys.exit(0)
    else:
        print("❌ F7 fused kernel DIVERGES on at least one shape. gdb time.")
        sys.exit(1)


if __name__ == "__main__":
    main()
