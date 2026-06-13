#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_yolo_layer9_sppf.py — Verify Layer 9 SPPF BIT_IDENTICAL.

YOLOv5n's SPPF (Spatial Pyramid Pooling Fast):
    cv1: Conv 256->128 (k=1, s=1, p=0) + BN + SiLU
    3x maxpool(k=5, s=1, p=2) producing a pyramid
    cv2: Conv 512->256 (k=1, s=1, p=0) + BN + SiLU

Input: (1, 256, 20, 20)
Output: (1, 256, 20, 20)

The substantive substrate-design composition: SPPF reuses already-verified
primitives (cbs, maxpool2d, concat). If each primitive is BIT_IDENTICAL,
the orchestrated whole should be BIT_IDENTICAL.
"""
import ctypes, os, sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

from yolo_forward import load_yolov5n_weights, run_sppf

SO = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")
lib = ctypes.CDLL(SO)

# Set up ctypes
# run_cbs uses bpd_conv2d_full_cpu (im2col+GEMM): 4 ptrs + 15 ints
lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
lib.bpd_conv2d_full_cpu.restype = None
lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
lib.bpd_conv2d_cpu.restype = None
lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3 + [ctypes.c_float]
lib.bpd_batchnorm_cpu_affine_fused.restype = None
lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
lib.bpd_silu_cpu.restype = None
# bpd_maxpool2d_cpu: 8 ints (N, C, H, W, kH, kW, stride, pad)
lib.bpd_maxpool2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*8
lib.bpd_maxpool2d_cpu.restype = None
lib.bpd_upsample_nearest2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*4
lib.bpd_upsample_nearest2d_cpu.restype = None
lib.bpd_concat_channel_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_void_p]
lib.bpd_concat_channel_cpu.restype = None


def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64); bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai); bi = np.where(bi < 0, BASE - bi, bi)
    return int(np.abs(ai - bi).max())


def main():
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/yolov5n.pt"
    weights = load_yolov5n_weights(pt_path)
    
    # Find Layer 9 weights
    prefix = "_modules.model._modules.9._modules."
    sppf_w = {
        'cv1_weight': weights[f"{prefix}cv1._modules.conv._parameters.weight"],
        'cv1_bn_gamma': weights[f"{prefix}cv1._modules.bn._parameters.weight"],
        'cv1_bn_beta': weights[f"{prefix}cv1._modules.bn._parameters.bias"],
        'cv1_bn_mean': weights[f"{prefix}cv1._modules.bn._buffers.running_mean"],
        'cv1_bn_var': weights[f"{prefix}cv1._modules.bn._buffers.running_var"],
        'cv2_weight': weights[f"{prefix}cv2._modules.conv._parameters.weight"],
        'cv2_bn_gamma': weights[f"{prefix}cv2._modules.bn._parameters.weight"],
        'cv2_bn_beta': weights[f"{prefix}cv2._modules.bn._parameters.bias"],
        'cv2_bn_mean': weights[f"{prefix}cv2._modules.bn._buffers.running_mean"],
        'cv2_bn_var': weights[f"{prefix}cv2._modules.bn._buffers.running_var"],
    }
    
    # Input: simulate Layer 8 output shape (1, 256, 20, 20)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 256, 20, 20)).astype(np.float32) * 0.5
    
    # Substrate SPPF
    out_sub = run_sppf(x, sppf_w, k=5, lib=lib)
    
    # Reference PyTorch SPPF
    class SPPF(nn.Module):
        def __init__(self, c1, c2, k):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(c_), nn.SiLU())
            self.cv2 = nn.Sequential(nn.Conv2d(c_ * 4, c2, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(c2), nn.SiLU())
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
            return self.cv2(torch.cat([x, y1, y2, y3], 1))
    
    sppf = SPPF(256, 256, k=5).eval()
    # Load Layer 9 weights into the module
    with torch.no_grad():
        sppf.cv1[0].weight.copy_(torch.from_numpy(sppf_w['cv1_weight']))
        sppf.cv1[1].weight.copy_(torch.from_numpy(sppf_w['cv1_bn_gamma']))
        sppf.cv1[1].bias.copy_(torch.from_numpy(sppf_w['cv1_bn_beta']))
        sppf.cv1[1].running_mean.copy_(torch.from_numpy(sppf_w['cv1_bn_mean']))
        sppf.cv1[1].running_var.copy_(torch.from_numpy(sppf_w['cv1_bn_var']))
        sppf.cv2[0].weight.copy_(torch.from_numpy(sppf_w['cv2_weight']))
        sppf.cv2[1].weight.copy_(torch.from_numpy(sppf_w['cv2_bn_gamma']))
        sppf.cv2[1].bias.copy_(torch.from_numpy(sppf_w['cv2_bn_beta']))
        sppf.cv2[1].running_mean.copy_(torch.from_numpy(sppf_w['cv2_bn_mean']))
        sppf.cv2[1].running_var.copy_(torch.from_numpy(sppf_w['cv2_bn_var']))
    
    with torch.no_grad():
        out_ref = sppf(torch.from_numpy(x)).numpy()
    
    u = ulp_distance(out_ref, out_sub)
    print(f"SPPF (1, 256, 20, 20): {'BIT_IDENTICAL' if u == 0 else f'DIVERGENT {u} ULP'}")
    print(f"  Output shape: {out_sub.shape}")
    print(f"  Elements: {out_sub.size}")


if __name__ == "__main__":
    main()
