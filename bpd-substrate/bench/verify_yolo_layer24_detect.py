#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_yolo_layer24_detect.py — Verify Layer 24 Detect head BIT_IDENTICAL.

The final Essence of YOLOv5n: the Detect head takes P3, P4, P5 feature maps
and produces detection outputs through three 1x1 conv heads, reshape+permute,
sigmoid, and anchor-grid scaling.

This verifier:
  1. Generates synthetic P3/P4/P5 feature maps (same shape PyTorch sees after
     full backbone+FPN)
  2. Runs substrate run_detect on them
  3. Runs PyTorch's Detect.forward on the same inputs with same weights
  4. Compares both the inference output (cat(z, 1)) and the per-level raw
     outputs (x[i] permuted shape) for bit-identity.
"""
import ctypes, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

from yolo_forward import load_yolov5n_weights, run_detect


def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64); bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai); bi = np.where(bi < 0, BASE - bi, bi)
    return int(np.abs(ai - bi).max()), int((np.abs(ai - bi) > 0).sum()), int(ai.size)


# Reference Detect module (inline — matches ultralytics/yolov5)
class _Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.stride = None  # set externally

    def _make_grid(self, nx, ny, i):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y = torch.arange(ny, device=d, dtype=t)
        x = torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.na, 1, 1, 2).expand(shape)
        return grid, anchor_grid

    def forward(self, x):
        # x is list of [P3, P4, P5]
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]
            wh = (wh * 2) ** 2 * self.anchor_grid[i]
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))
        return torch.cat(z, 1), x


def main():
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/yolov5n.pt"
    weights = load_yolov5n_weights(pt_path)

    so = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")
    lib = ctypes.CDLL(so)
    lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
    lib.bpd_conv2d_full_cpu.restype = None
    lib.bpd_sigmoid_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
    lib.bpd_sigmoid_cpu.restype = None
    if hasattr(lib, 'bpd_detect_postprocess_cpu'):
        lib.bpd_detect_postprocess_cpu.argtypes = (
            [ctypes.c_void_p]*3 + [ctypes.c_float] + [ctypes.c_void_p] + [ctypes.c_int]*5)
        lib.bpd_detect_postprocess_cpu.restype = None

    # Extract Detect weights (Layer 24)
    base = "_modules.model._modules.24._modules.m._modules."
    detect_weights = {
        'm0_weight': weights[f"{base}0._parameters.weight"],  # (255, 64, 1, 1)
        'm0_bias':   weights[f"{base}0._parameters.bias"],
        'm1_weight': weights[f"{base}1._parameters.weight"],  # (255, 128, 1, 1)
        'm1_bias':   weights[f"{base}1._parameters.bias"],
        'm2_weight': weights[f"{base}2._parameters.weight"],  # (255, 256, 1, 1)
        'm2_bias':   weights[f"{base}2._parameters.bias"],
    }
    # YOLOv5n anchors (standard from yaml)
    anchors_yaml = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    anchors = np.array(anchors_yaml, dtype=np.float32).reshape(3, 3, 2)
    strides = np.array([8.0, 16.0, 32.0], dtype=np.float32)
    nc = 80

    # Synthetic P3, P4, P5 feature maps matching the shapes from D.3 sweep
    # For 640x640 input: P3=(1,64,80,80), P4=(1,128,40,40), P5=(1,256,20,20)
    rng = np.random.default_rng(42)
    p3 = (rng.standard_normal((1, 64, 80, 80)) * 0.3).astype(np.float32)
    p4 = (rng.standard_normal((1, 128, 40, 40)) * 0.3).astype(np.float32)
    p5 = (rng.standard_normal((1, 256, 20, 20)) * 0.3).astype(np.float32)

    # ── Substrate detect ──
    feature_maps_sub = [p3.copy(), p4.copy(), p5.copy()]
    inf_sub, raw_sub = run_detect(feature_maps_sub, detect_weights, anchors, strides, nc, lib=lib)

    # ── Reference detect ──
    detect_ref = _Detect(nc=nc, anchors=anchors_yaml, ch=[64, 128, 256]).eval()
    detect_ref.stride = torch.tensor(strides)
    # Load conv weights
    with torch.no_grad():
        detect_ref.m[0].weight.copy_(torch.from_numpy(detect_weights['m0_weight']))
        detect_ref.m[0].bias.copy_(torch.from_numpy(detect_weights['m0_bias']))
        detect_ref.m[1].weight.copy_(torch.from_numpy(detect_weights['m1_weight']))
        detect_ref.m[1].bias.copy_(torch.from_numpy(detect_weights['m1_bias']))
        detect_ref.m[2].weight.copy_(torch.from_numpy(detect_weights['m2_weight']))
        detect_ref.m[2].bias.copy_(torch.from_numpy(detect_weights['m2_bias']))
    feature_maps_ref = [torch.from_numpy(p3.copy()), torch.from_numpy(p4.copy()), torch.from_numpy(p5.copy())]
    with torch.no_grad():
        inf_ref, raw_ref = detect_ref(feature_maps_ref)
    inf_ref_np = inf_ref.numpy()
    raw_ref_np = [r.numpy() for r in raw_ref]

    # ── Compare ──
    print()
    print("Detect head verification (synthetic P3/P4/P5)")
    print(f"{'Output':<25} {'Shape':<25} {'Max ULP':<10} {'Diff':<18} {'Status':<15}")
    print("-" * 95)

    # Per-level raw outputs
    for i, (rs, rr) in enumerate(zip(raw_sub, raw_ref_np)):
        max_ulp, n_diff, n_total = ulp_distance(rr, rs)
        status = "BIT_IDENTICAL" if max_ulp == 0 else "DIVERGENT"
        print(f"raw[{i}] (P{i+3})              {str(list(rs.shape)):<25} {max_ulp:<10} {n_diff:<6}/{n_total:<10} {status}")

    # Inference output
    max_ulp, n_diff, n_total = ulp_distance(inf_ref_np, inf_sub)
    status = "BIT_IDENTICAL" if max_ulp == 0 else "DIVERGENT"
    print(f"inference (cat z)         {str(list(inf_sub.shape)):<25} {max_ulp:<10} {n_diff:<6}/{n_total:<10} {status}")


if __name__ == "__main__":
    main()
