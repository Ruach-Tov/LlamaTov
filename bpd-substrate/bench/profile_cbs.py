# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Profile run_cbs: GEMM time vs epilogue time on real YOLOv5n forward pass.

Empirical confirmation of Medayek's 84%/2% diagnosis. Adds time.perf_counter
around each substrate kernel call inside run_cbs (in a profiling fork of
yolo_forward.py) and reports per-layer breakdown.
"""
import ctypes
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, "/tmp/bpd_test/bench")
from yolo_forward import load_yolov5n_weights, precompute_bn, get_layer_weights, get_cN_weights

SO = "/tmp/bpd_test/build/bpd_cpu.so"
lib = ctypes.CDLL(SO)
lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
lib.bpd_conv2d_full_cpu.restype = None
lib.bpd_conv2d_bn_silu_fused_cpu.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
lib.bpd_conv2d_bn_silu_fused_cpu.restype = None
lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3 + [ctypes.c_float]
lib.bpd_batchnorm_cpu_affine_fused.restype = None
lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
lib.bpd_silu_cpu.restype = None


def run_cbs_timed(x, weight, bn_gamma, bn_beta, bn_mean, bn_var, stride, pad):
    """Run unfused CBS chain with per-step timing."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    bn_gamma = np.asarray(bn_gamma, dtype=np.float32)
    bn_beta = np.asarray(bn_beta, dtype=np.float32)
    bn_mean = np.asarray(bn_mean, dtype=np.float32)
    bn_var = np.asarray(bn_var, dtype=np.float32)

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    kH, kW = weight.shape[2], weight.shape[3]
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1

    # 1. Conv (im2col + GEMM)
    t0 = time.perf_counter()
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_full_cpu(
        x.ctypes.data, weight.ctypes.data, 0,
        out.ctypes.data,
        N, C_in, H, W, C_out, kH, kW,
        stride, stride, pad, pad, 1, 1, 1)
    t_conv = time.perf_counter() - t0

    # 2. BN
    t0 = time.perf_counter()
    bn_out = np.zeros_like(out)
    scale_buf = np.zeros(C_out, dtype=np.float32)
    offset_buf = np.zeros(C_out, dtype=np.float32)
    out_c = np.ascontiguousarray(out, dtype=np.float32)
    lib.bpd_batchnorm_cpu_affine_fused(
        out_c.ctypes.data, bn_gamma.ctypes.data, bn_beta.ctypes.data,
        bn_mean.ctypes.data, bn_var.ctypes.data, bn_out.ctypes.data,
        scale_buf.ctypes.data, offset_buf.ctypes.data,
        N, C_out, H_out * W_out, 1e-5)
    t_bn = time.perf_counter() - t0

    # 3. SiLU
    t0 = time.perf_counter()
    silu_out = np.zeros_like(bn_out)
    lib.bpd_silu_cpu(bn_out.ctypes.data, silu_out.ctypes.data, bn_out.size)
    t_silu = time.perf_counter() - t0

    return silu_out, t_conv, t_bn, t_silu


def main():
    weights = load_yolov5n_weights("/tmp/yolo_canonical/yolov5n.pt")

    # 640x640 RGB input
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)

    # All CBS layers as (layer_idx, input_shape, stride, pad, label)
    cbs_layers = [
        (0,  16, 6, 2, 2, "L0 focus 6x6 s=2"),
        (1,  32, 3, 2, 1, "L1 cbs 3x3 s=2"),
        (3,  64, 3, 2, 1, "L3 cbs 3x3 s=2"),
        (5, 128, 3, 2, 1, "L5 cbs 3x3 s=2"),
        (7, 256, 3, 2, 1, "L7 cbs 3x3 s=2"),
        (10,128, 1, 1, 0, "L10 head cbs 1x1"),
        (14, 64, 1, 1, 0, "L14 head cbs 1x1"),
        (18, 64, 3, 2, 1, "L18 head cbs 3x3 s=2"),
        (21,128, 3, 2, 1, "L21 head cbs 3x3 s=2"),
    ]

    print("=" * 96)
    print("Per-CBS-layer timing breakdown (substrate scalar GEMM, AVX1 host)")
    print("=" * 96)
    print(f"{'Layer':<28} {'Input shape':<20} {'Conv ms':<12} {'BN ms':<10} {'SiLU ms':<10} {'GEMM %':<10}")
    print("-" * 96)

    # We need to run the forward pass to get the right input shape for each layer.
    # Simpler: just run each CBS in isolation at its expected input shape, with random data.
    layer_inputs = {
        0: (1, 3, 640, 640),  1: (1, 16, 320, 320), 3: (1, 32, 160, 160),
        5: (1, 64,  80,  80), 7: (1, 128, 40, 40), 10: (1, 256, 20, 20),
        14: (1, 128, 40, 40), 18: (1, 64,  80, 80), 21: (1, 128, 40, 40),
    }

    total_conv = 0.0
    total_bn = 0.0
    total_silu = 0.0
    for li, c_out, k, s, p, label in cbs_layers:
        w = get_layer_weights(weights, li)
        in_shape = layer_inputs[li]
        x_l = rng.standard_normal(in_shape).astype(np.float32)
        # warm-up
        out, _, _, _ = run_cbs_timed(x_l, w['conv_weight'], w['bn_weight'], w['bn_bias'],
                                      w['bn_mean'], w['bn_var'], s, p)
        # measure
        out, t_conv, t_bn, t_silu = run_cbs_timed(
            x_l, w['conv_weight'], w['bn_weight'], w['bn_bias'],
            w['bn_mean'], w['bn_var'], s, p)
        total_conv += t_conv
        total_bn += t_bn
        total_silu += t_silu
        total = t_conv + t_bn + t_silu
        pct_conv = 100 * t_conv / total if total > 0 else 0
        print(f"{label:<28} {str(in_shape):<20} {t_conv*1000:>8.2f}    {t_bn*1000:>6.2f}    {t_silu*1000:>6.2f}    {pct_conv:>6.1f}%")

    grand_total = total_conv + total_bn + total_silu
    print("-" * 96)
    print(f"{'TOTAL across 9 top-level CBS':<28} {' ':<20} {total_conv*1000:>8.2f}    {total_bn*1000:>6.2f}    {total_silu*1000:>6.2f}")
    print(f"  Conv (GEMM): {100*total_conv/grand_total:.1f}%")
    print(f"  BN epilogue: {100*total_bn/grand_total:.1f}%")
    print(f"  SiLU:        {100*total_silu/grand_total:.1f}%")
    print()
    print(f"Medayek predicted: ~84% Conv, ~2% BN+SiLU")
    print(f"Empirical:         {100*total_conv/grand_total:.1f}% Conv, {100*(total_bn+total_silu)/grand_total:.1f}% BN+SiLU")


if __name__ == "__main__":
    main()
