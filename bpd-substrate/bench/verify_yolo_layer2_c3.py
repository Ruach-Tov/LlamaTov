#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_yolo_layer2_c3.py — Layer 2 (first C3) end-to-end bit-identity vs PyTorch CPU.

Per Heath's discipline:
  "we want to get up to 24 layers before this is done"
  "0 ULP per stage as we compose"

This harness extends the per-stage verification to Layer 2 — the first C3
module in YOLOv5n. C3 composes:

  CBS(cv1) → Bottleneck × n → ↘
                                Concat → CBS(cv3)
  CBS(cv2) ──────────────────→ ↗

Every primitive used by run_cN is independently BIT_IDENTICAL with PyTorch:
  - bpd_conv2d_cpu        (verified Layer 0+1)
  - bpd_batchnorm via affine numpy form  (verified Layer 0+1, post-opmath-fix)
  - bpd_silu_cpu          (verified Layer 0+1)
  - bpd_residual_add_cpu  (verified today)
  - bpd_concat_channel_cpu (verified today)

This harness tests the COMPOSITION at Layer 2. If 0 ULP holds, we have
empirical confirmation that Layer 0+1+2 stays bit-identical end-to-end.
If divergence surfaces, β-discipline applies.

Uses real YOLOv5n.pt weights (loaded by yolo_forward.py's existing path).
PyTorch reference is implemented inline using PyTorch primitives matching
ultralytics' C3 definition (no ultralytics dependency).
"""
import ctypes
import os
import sys
from pathlib import Path

# Make bench/ importable
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("torch required")

torch.backends.mkldnn.enabled = False
torch.backends.cudnn.enabled = False
torch.set_num_threads(1)

from yolo_forward import (
    load_yolov5n_weights, run_cbs, run_cN, get_layer_weights, get_cN_weights,
    precompute_bn,
)


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


# ── PyTorch reference implementations ──

def pt_cbs(x_t, conv_w, bn_g, bn_b, bn_m, bn_v, stride, pad):
    """PyTorch CBS = Conv + BN(eval) + SiLU."""
    conv_w_f32 = np.ascontiguousarray(conv_w, dtype=np.float32)
    c_in = conv_w_f32.shape[1]
    c_out = conv_w_f32.shape[0]
    kH = conv_w_f32.shape[2]
    with torch.no_grad():
        conv = nn.Conv2d(c_in, c_out, kH, stride=stride, padding=pad, bias=False)
        conv.weight.data = torch.from_numpy(conv_w_f32)
        bn = nn.BatchNorm2d(c_out)
        bn.weight.data = torch.from_numpy(np.asarray(bn_g, dtype=np.float32))
        bn.bias.data = torch.from_numpy(np.asarray(bn_b, dtype=np.float32))
        bn.running_mean.data = torch.from_numpy(np.asarray(bn_m, dtype=np.float32))
        bn.running_var.data = torch.from_numpy(np.asarray(bn_v, dtype=np.float32))
        bn.eps = 1e-5
        bn.eval()
        return F.silu(bn(conv(x_t))).numpy()


def pt_bottleneck(x_t, weights, shortcut=True):
    """PyTorch Bottleneck = CBS(1x1) + CBS(3x3) + optional residual."""
    y = pt_cbs(x_t, weights['cv1_conv'], weights['cv1_bn_gamma'],
               weights['cv1_bn_beta'], weights['cv1_bn_mean'],
               weights['cv1_bn_var'], stride=1, pad=0)
    y = pt_cbs(torch.from_numpy(y), weights['cv2_conv'], weights['cv2_bn_gamma'],
               weights['cv2_bn_beta'], weights['cv2_bn_mean'],
               weights['cv2_bn_var'], stride=1, pad=1)
    if shortcut:
        return x_t.numpy() + y
    return y


def pt_c3(x_t, weights, n, shortcut=True):
    """PyTorch C3: cv1 -> n×Bottleneck -> concat with cv2(x) -> cv3."""
    y1 = pt_cbs(x_t, weights['cv1_conv'], weights['cv1_bn_gamma'],
                weights['cv1_bn_beta'], weights['cv1_bn_mean'],
                weights['cv1_bn_var'], stride=1, pad=0)
    for i in range(n):
        y1 = pt_bottleneck(torch.from_numpy(y1), weights[f'm{i}'], shortcut=shortcut)
    y2 = pt_cbs(x_t, weights['cv2_conv'], weights['cv2_bn_gamma'],
                weights['cv2_bn_beta'], weights['cv2_bn_mean'],
                weights['cv2_bn_var'], stride=1, pad=0)
    y3 = np.concatenate([y1, y2], axis=1)
    return pt_cbs(torch.from_numpy(y3), weights['cv3_conv'],
                  weights['cv3_bn_gamma'], weights['cv3_bn_beta'],
                  weights['cv3_bn_mean'], weights['cv3_bn_var'],
                  stride=1, pad=0)


def main():
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/yolov5n.pt"
    print("=" * 78)
    print("verify_yolo_layer2_c3.py — Layer 2 (first C3) bit-identity vs PyTorch CPU")
    print(f"Weights: {pt_path}")
    print("PyTorch path: MKL-DNN disabled, single-threaded")
    print("=" * 78)

    # Load BPD library
    cpu_so = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
    if not os.path.exists(cpu_so):
        sys.exit(f"{cpu_so} not found")
    lib = ctypes.CDLL(cpu_so)
    lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
    lib.bpd_conv2d_cpu.restype = None
    lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.bpd_silu_cpu.restype = None
    lib.bpd_residual_add_cpu.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.bpd_residual_add_cpu.restype = None
    lib.bpd_concat_channel_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p]
    lib.bpd_concat_channel_cpu.restype = None
    # bpd_batchnorm_cpu_affine_fused — needed by run_cbs after the bug fix
    lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.bpd_batchnorm_cpu_affine_fused.restype = None

    # Load real YOLOv5n weights
    all_weights = load_yolov5n_weights(pt_path)
    print(f"\nLoaded {len(all_weights)} tensors")

    # Construct Layer 0+1 output as input to Layer 2 (so we test under the
    # actual data distribution Layer 2 would see in the orchestrator).
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)

    # Layer 0 (CBS 3 → 16, k=6, s=2, p=2)
    w0 = get_layer_weights(all_weights, 0)
    x = run_cbs(x, w0['conv_weight'], w0['bn_weight'], w0['bn_bias'],
                w0['bn_mean'], w0['bn_var'], stride=2, pad=2, lib=lib)
    print(f"  Layer 0 output: {x.shape}")

    # Layer 1 (CBS 16 → 32, k=3, s=2, p=1)
    w1 = get_layer_weights(all_weights, 1)
    x = run_cbs(x, w1['conv_weight'], w1['bn_weight'], w1['bn_bias'],
                w1['bn_mean'], w1['bn_var'], stride=2, pad=1, lib=lib)
    print(f"  Layer 1 output: {x.shape}")

    # Layer 2 (C3, c_in=32, c_out=32, n=1, shortcut=True per yolov5n.yaml)
    n_bottleneck = 1
    shortcut = True
    w2 = get_cN_weights(all_weights, 2, n_bottleneck)
    if w2 is None:
        sys.exit("Layer 2 weights extraction returned None")
    print(f"  Layer 2 C3 weights loaded: n={n_bottleneck} shortcut={shortcut}")

    # Run substrate path
    sub_out = run_cN(x, w2, n=n_bottleneck, shortcut=shortcut, lib=lib)
    print(f"  Layer 2 output (substrate): {sub_out.shape} range=[{sub_out.min():.4f}, {sub_out.max():.4f}]")

    # Run PyTorch reference
    pt_out = pt_c3(torch.from_numpy(x), w2, n=n_bottleneck, shortcut=shortcut)
    print(f"  Layer 2 output (PyTorch):   {pt_out.shape} range=[{pt_out.min():.4f}, {pt_out.max():.4f}]")

    print()
    print("=== Bit-identity verdict ===")
    ok = report("Layer 2 C3 (substrate vs PyTorch)", pt_out, sub_out)

    print()
    print("=" * 78)
    if ok:
        print("BIT_IDENTICAL: Layer 2 composes to 0 ULP. Ready for Layers 3+.")
        return 0
    else:
        print("DIVERGENT: a new substrate-design parameter has surfaced at C3.")
        print("Apply β-discipline: shrink, disassemble, narrow, fix, verify.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
