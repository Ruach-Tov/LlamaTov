#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_yolo_composition_sweep.py — Greedy composition sweep for YOLOv5n.

Per Heath 2026-05-21 ~02:45 UTC: 'we can try combinations of composing the
YOLO kernels, sort of like a greedy matmul optimizer, and locate the layers
or layer:layer interfaces where we are not BIT_IDENTICAL, and quantify the
number, type, and classes of the remaining divergencies.'

The substantive substrate-design substantive plan:
  1. Run substrate forward through each layer i=0..24, keeping all
     intermediates (sub_intermediates[i])
  2. Run PyTorch reference forward through the same prefix, keeping
     intermediates (ref_intermediates[i])
  3. At each layer boundary, compute:
       - max ULP between sub_intermediates[i] and ref_intermediates[i]
       - num positions diverging
       - layer type (cbs/c3/sppf/concat/upsample/detect)
       - input shape, output shape
  4. Surface the composition surface: a table of where divergence enters
     and how it propagates.
  
The substantive output: a structured divergence map that names exactly
which layers introduce divergence and which propagate it from upstream.
"""
import ctypes, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

from yolo_forward import (load_yolov5n_weights, yolov5n_architecture,
                            run_cbs, run_cN, run_sppf, run_upsample,
                            run_concat, run_maxpool2d, get_cN_weights)


def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64); bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai); bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


def setup_lib():
    so = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")
    lib = ctypes.CDLL(so)
    lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
    lib.bpd_conv2d_full_cpu.restype = None
    lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
    lib.bpd_conv2d_cpu.restype = None
    lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3 + [ctypes.c_float]
    lib.bpd_batchnorm_cpu_affine_fused.restype = None
    lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
    lib.bpd_silu_cpu.restype = None
    lib.bpd_residual_add_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]
    lib.bpd_residual_add_cpu.restype = None
    lib.bpd_concat_channel_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p]
    lib.bpd_concat_channel_cpu.restype = None
    lib.bpd_maxpool2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*8
    lib.bpd_maxpool2d_cpu.restype = None
    lib.bpd_upsample_nearest2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*4
    lib.bpd_upsample_nearest2d_cpu.restype = None
    # Needed for run_detect (Layer 24)
    lib.bpd_sigmoid_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
    lib.bpd_sigmoid_cpu.restype = None
    # Phase 3.1 F3 fused kernel (optional — present in builds with F3 landed)
    if hasattr(lib, 'bpd_conv2d_bn_silu_fused_cpu'):
        lib.bpd_conv2d_bn_silu_fused_cpu.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
        lib.bpd_conv2d_bn_silu_fused_cpu.restype = None
    # Phase 3.CAT.TDD F3-v2 fused kernel
    if hasattr(lib, 'bpd_conv2d_bn_silu_fused_cpu_v2'):
        lib.bpd_conv2d_bn_silu_fused_cpu_v2.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
        lib.bpd_conv2d_bn_silu_fused_cpu_v2.restype = None
    # Phase 3.2 F8 Detect post-sigmoid fused kernel (optional)
    if hasattr(lib, 'bpd_detect_postprocess_cpu'):
        # (in, grid, anchor_grid, stride float, out, bs, na, ny, nx, no)
        lib.bpd_detect_postprocess_cpu.argtypes = (
            [ctypes.c_void_p]*3 + [ctypes.c_float] + [ctypes.c_void_p] + [ctypes.c_int]*5)
        lib.bpd_detect_postprocess_cpu.restype = None
    # Phase 3.4 F4 CBS + Residual Add fused kernel (optional)
    if hasattr(lib, 'bpd_conv2d_bn_silu_add_fused_cpu'):
        # (in, weight, alpha, beta, residual, out, N, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW)
        lib.bpd_conv2d_bn_silu_add_fused_cpu.argtypes = (
            [ctypes.c_void_p]*6 + [ctypes.c_int]*11)
        lib.bpd_conv2d_bn_silu_add_fused_cpu.restype = None
    return lib


# ── PyTorch reference YOLOv5n (inline, no ultralytics) ──

class _CBS(nn.Module):
    """Conv + BN + SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        c_ = c2  # hidden
        self.cv1 = _CBS(c1, c_, 1, 1, 0)
        self.cv2 = _CBS(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class _C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = _CBS(c1, c_, 1, 1, 0)
        self.cv2 = _CBS(c1, c_, 1, 1, 0)
        self.cv3 = _CBS(2 * c_, c2, 1, 1, 0)
        self.m = nn.Sequential(*[_Bottleneck(c_, c_, shortcut) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class _SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = _CBS(c1, c_, 1, 1, 0)
        self.cv2 = _CBS(c_ * 4, c2, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


def load_cbs_into_module(mod, w):
    """Load a CBS dict into _CBS module."""
    with torch.no_grad():
        mod.conv.weight.copy_(torch.from_numpy(np.ascontiguousarray(w['conv_weight'], dtype=np.float32)))
        mod.bn.weight.copy_(torch.from_numpy(np.asarray(w['bn_weight'], dtype=np.float32)))
        mod.bn.bias.copy_(torch.from_numpy(np.asarray(w['bn_bias'], dtype=np.float32)))
        mod.bn.running_mean.copy_(torch.from_numpy(np.asarray(w['bn_mean'], dtype=np.float32)))
        mod.bn.running_var.copy_(torch.from_numpy(np.asarray(w['bn_var'], dtype=np.float32)))


def load_cn_into_module(mod, all_weights, layer_idx, n):
    """Load C3 weights into _C3 module.
    
    get_cN_weights returns a flat dict with keys like cv1_conv, cv1_bn_gamma,...
    and m{i} sub-dicts.
    """
    w = get_cN_weights(all_weights, layer_idx, n)
    for cv_name, mod_cv in [('cv1', mod.cv1), ('cv2', mod.cv2), ('cv3', mod.cv3)]:
        load_cbs_into_module(mod_cv, {
            'conv_weight': w[f'{cv_name}_conv'],
            'bn_weight': w[f'{cv_name}_bn_gamma'],
            'bn_bias': w[f'{cv_name}_bn_beta'],
            'bn_mean': w[f'{cv_name}_bn_mean'],
            'bn_var': w[f'{cv_name}_bn_var'],
        })
    for i in range(n):
        bn_w = w[f'm{i}']
        bn_cv1 = {'conv_weight': bn_w['cv1_conv'], 'bn_weight': bn_w['cv1_bn_gamma'],
                   'bn_bias': bn_w['cv1_bn_beta'], 'bn_mean': bn_w['cv1_bn_mean'],
                   'bn_var': bn_w['cv1_bn_var']}
        bn_cv2 = {'conv_weight': bn_w['cv2_conv'], 'bn_weight': bn_w['cv2_bn_gamma'],
                   'bn_bias': bn_w['cv2_bn_beta'], 'bn_mean': bn_w['cv2_bn_mean'],
                   'bn_var': bn_w['cv2_bn_var']}
        load_cbs_into_module(mod.m[i].cv1, bn_cv1)
        load_cbs_into_module(mod.m[i].cv2, bn_cv2)


def get_sppf_weights(all_weights, layer_idx):
    prefix = f"_modules.model._modules.{layer_idx}._modules."
    return {
        'cv1_weight': all_weights[f"{prefix}cv1._modules.conv._parameters.weight"],
        'cv1_bn_gamma': all_weights[f"{prefix}cv1._modules.bn._parameters.weight"],
        'cv1_bn_beta': all_weights[f"{prefix}cv1._modules.bn._parameters.bias"],
        'cv1_bn_mean': all_weights[f"{prefix}cv1._modules.bn._buffers.running_mean"],
        'cv1_bn_var': all_weights[f"{prefix}cv1._modules.bn._buffers.running_var"],
        'cv2_weight': all_weights[f"{prefix}cv2._modules.conv._parameters.weight"],
        'cv2_bn_gamma': all_weights[f"{prefix}cv2._modules.bn._parameters.weight"],
        'cv2_bn_beta': all_weights[f"{prefix}cv2._modules.bn._parameters.bias"],
        'cv2_bn_mean': all_weights[f"{prefix}cv2._modules.bn._buffers.running_mean"],
        'cv2_bn_var': all_weights[f"{prefix}cv2._modules.bn._buffers.running_var"],
    }


def load_sppf_module(mod, sppf_w):
    with torch.no_grad():
        mod.cv1.conv.weight.copy_(torch.from_numpy(np.ascontiguousarray(sppf_w['cv1_weight'], dtype=np.float32)))
        mod.cv1.bn.weight.copy_(torch.from_numpy(np.asarray(sppf_w['cv1_bn_gamma'], dtype=np.float32)))
        mod.cv1.bn.bias.copy_(torch.from_numpy(np.asarray(sppf_w['cv1_bn_beta'], dtype=np.float32)))
        mod.cv1.bn.running_mean.copy_(torch.from_numpy(np.asarray(sppf_w['cv1_bn_mean'], dtype=np.float32)))
        mod.cv1.bn.running_var.copy_(torch.from_numpy(np.asarray(sppf_w['cv1_bn_var'], dtype=np.float32)))
        mod.cv2.conv.weight.copy_(torch.from_numpy(np.ascontiguousarray(sppf_w['cv2_weight'], dtype=np.float32)))
        mod.cv2.bn.weight.copy_(torch.from_numpy(np.asarray(sppf_w['cv2_bn_gamma'], dtype=np.float32)))
        mod.cv2.bn.bias.copy_(torch.from_numpy(np.asarray(sppf_w['cv2_bn_beta'], dtype=np.float32)))
        mod.cv2.bn.running_mean.copy_(torch.from_numpy(np.asarray(sppf_w['cv2_bn_mean'], dtype=np.float32)))
        mod.cv2.bn.running_var.copy_(torch.from_numpy(np.asarray(sppf_w['cv2_bn_var'], dtype=np.float32)))


def main():
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/yolov5n.pt"
    weights = load_yolov5n_weights(pt_path)
    lib = setup_lib()
    
    # Test input: same shape YOLOv5n expects
    rng = np.random.default_rng(42)
    x_input = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)
    
    # Architecture: all 24 layers excluding the final Detect head (D.4 territory).
    # Inter-layer dataflow caching threads outputs of layers 4, 6, 10, 14
    # forward to their downstream concat consumers at 12, 16, 19, 22.
    layers = [
        # Backbone
        (0, 'cbs', {'c1':3, 'c2':16, 'k':6, 's':2, 'p':2}),
        (1, 'cbs', {'c1':16, 'c2':32, 'k':3, 's':2, 'p':1}),
        (2, 'c3', {'c1':32, 'c2':32, 'n':1, 'shortcut':True}),
        (3, 'cbs', {'c1':32, 'c2':64, 'k':3, 's':2, 'p':1}),
        (4, 'c3', {'c1':64, 'c2':64, 'n':2, 'shortcut':True}),          # P3 cache
        (5, 'cbs', {'c1':64, 'c2':128, 'k':3, 's':2, 'p':1}),
        (6, 'c3', {'c1':128, 'c2':128, 'n':3, 'shortcut':True}),        # P4 cache
        (7, 'cbs', {'c1':128, 'c2':256, 'k':3, 's':2, 'p':1}),
        (8, 'c3', {'c1':256, 'c2':256, 'n':1, 'shortcut':True}),
        (9, 'sppf', {'c1':256, 'c2':256, 'k':5}),
        # Head with FPN
        (10, 'cbs', {'c1':256, 'c2':128, 'k':1, 's':1, 'p':0}),         # P5-cv1 cache
        (11, 'upsample', {}),
        (12, 'concat', {'from': [6]}),                                    # cat with P4
        (13, 'c3', {'c1':256, 'c2':128, 'n':1, 'shortcut':False}),
        (14, 'cbs', {'c1':128, 'c2':64, 'k':1, 's':1, 'p':0}),          # P4-cv1 cache
        (15, 'upsample', {}),
        (16, 'concat', {'from': [4]}),                                    # cat with P3
        (17, 'c3', {'c1':128, 'c2':64, 'n':1, 'shortcut':False}),       # P3 head output
        (18, 'cbs', {'c1':64, 'c2':64, 'k':3, 's':2, 'p':1}),
        (19, 'concat', {'from': [14]}),                                   # cat with P4-cv1
        (20, 'c3', {'c1':128, 'c2':128, 'n':1, 'shortcut':False}),      # P4 head output
        (21, 'cbs', {'c1':128, 'c2':128, 'k':3, 's':2, 'p':1}),
        (22, 'concat', {'from': [10]}),                                   # cat with P5-cv1
        (23, 'c3', {'c1':256, 'c2':256, 'n':1, 'shortcut':False}),      # P5 head output
    ]
    
    # ── Run substrate forward, capture intermediates ──
    # cache[i] stores the output of layer i so downstream FPN concats can reach back.
    cache = {}
    sub_intermediates = []
    x_sub = x_input.copy()
    from yolo_forward import get_layer_weights
    for layer_idx, kind, cfg in layers:
        if kind == 'cbs':
            w = get_layer_weights(weights, layer_idx)
            x_sub = run_cbs(x_sub, w['conv_weight'], w['bn_weight'], w['bn_bias'],
                             w['bn_mean'], w['bn_var'],
                             stride=cfg['s'], pad=cfg['p'], lib=lib)
        elif kind == 'c3':
            cn_w = get_cN_weights(weights, layer_idx, cfg['n'])
            x_sub = run_cN(x_sub, cn_w, n=cfg['n'],
                             shortcut=cfg.get('shortcut', True), lib=lib)
        elif kind == 'sppf':
            sppf_w = get_sppf_weights(weights, layer_idx)
            x_sub = run_sppf(x_sub, sppf_w, k=cfg['k'], lib=lib)
        elif kind == 'upsample':
            x_sub = run_upsample(x_sub, lib=lib)
        elif kind == 'concat':
            # Concat current tensor with cached outputs of layers in cfg['from']
            tensors = [x_sub] + [cache[i] for i in cfg['from']]
            x_sub = run_concat(tensors, lib=lib)
        else:
            raise ValueError(f"unknown kind {kind} at layer {layer_idx}")
        cache[layer_idx] = x_sub.copy()
        sub_intermediates.append((layer_idx, kind, x_sub.copy()))
    
    # ── Run PyTorch reference forward, capture intermediates ──
    ref_cache = {}
    ref_intermediates = []
    x_ref_t = torch.from_numpy(x_input.copy())
    for layer_idx, kind, cfg in layers:
        if kind == 'cbs':
            mod = _CBS(cfg['c1'], cfg['c2'], cfg['k'], cfg['s'], cfg['p']).eval()
            w = get_layer_weights(weights, layer_idx)
            load_cbs_into_module(mod, w)
            with torch.no_grad():
                x_ref_t = mod(x_ref_t)
        elif kind == 'c3':
            mod = _C3(cfg['c1'], cfg['c2'], n=cfg['n'],
                       shortcut=cfg.get('shortcut', True)).eval()
            load_cn_into_module(mod, weights, layer_idx, cfg['n'])
            with torch.no_grad():
                x_ref_t = mod(x_ref_t)
        elif kind == 'sppf':
            mod = _SPPF(cfg['c1'], cfg['c2'], k=cfg['k']).eval()
            sppf_w = get_sppf_weights(weights, layer_idx)
            load_sppf_module(mod, sppf_w)
            with torch.no_grad():
                x_ref_t = mod(x_ref_t)
        elif kind == 'upsample':
            with torch.no_grad():
                x_ref_t = F.interpolate(x_ref_t, scale_factor=2, mode='nearest')
        elif kind == 'concat':
            with torch.no_grad():
                x_ref_t = torch.cat([x_ref_t] + [ref_cache[i] for i in cfg['from']], dim=1)
        else:
            raise ValueError(f"unknown kind {kind} at layer {layer_idx}")
        ref_cache[layer_idx] = x_ref_t.clone()
        ref_intermediates.append((layer_idx, kind, x_ref_t.numpy().copy()))
    
    # ── Compare per layer ──
    print()
    print("Greedy composition sweep — substrate vs PyTorch reference")
    print(f"Input shape: {x_input.shape}")
    print()
    print(f"{'Layer':<6} {'Kind':<8} {'Output shape':<22} {'Max ULP':<10} {'Diff positions':<18} {'% diverged':<12} {'Status':<15}")
    print("-" * 100)
    
    for (li, k, sub), (_, _, ref) in zip(sub_intermediates, ref_intermediates):
        max_ulp, n_diff, n_total = ulp_distance(sub, ref)
        pct = (n_diff / n_total) * 100
        status = "BIT_IDENTICAL" if max_ulp == 0 else f"DIVERGENT"
        shape = str(list(sub.shape))
        print(f"{li:<6} {k:<8} {shape:<22} {max_ulp:<10} {n_diff:<6}/{n_total:<10} {pct:<6.2f}%      {status}")

    # ── Layer 24: Detect head — the final Essence ──
    print()
    print("Layer 24 Detect head — substrate vs PyTorch reference")
    print(f"{'Output':<22} {'Shape':<25} {'Max ULP':<10} {'Diff':<18} {'Status':<15}")
    print("-" * 90, flush=True)

    from yolo_forward import run_detect

    # Layer 24 weights
    base = "_modules.model._modules.24._modules.m._modules."
    detect_weights = {
        'm0_weight': weights[f"{base}0._parameters.weight"],
        'm0_bias':   weights[f"{base}0._parameters.bias"],
        'm1_weight': weights[f"{base}1._parameters.weight"],
        'm1_bias':   weights[f"{base}1._parameters.bias"],
        'm2_weight': weights[f"{base}2._parameters.weight"],
        'm2_bias':   weights[f"{base}2._parameters.bias"],
    }
    anchors_yaml = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    anchors = np.array(anchors_yaml, dtype=np.float32).reshape(3, 3, 2)
    strides = np.array([8.0, 16.0, 32.0], dtype=np.float32)
    nc = 80

    # Substrate Detect: use cached outputs of layers 17 (P3), 20 (P4), 23 (P5)
    feat_sub = [cache[17].copy(), cache[20].copy(), cache[23].copy()]
    inf_sub_24, raw_sub_24 = run_detect(feat_sub, detect_weights, anchors, strides, nc, lib=lib)

    # Reference Detect: matching inline module
    class _DetectRef(nn.Module):
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
            self.stride = None

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

    detect_ref = _DetectRef(nc=nc, anchors=anchors_yaml, ch=[64, 128, 256]).eval()
    detect_ref.stride = torch.tensor(strides)
    with torch.no_grad():
        detect_ref.m[0].weight.copy_(torch.from_numpy(np.ascontiguousarray(detect_weights['m0_weight'], dtype=np.float32)))
        detect_ref.m[0].bias.copy_(torch.from_numpy(np.asarray(detect_weights['m0_bias'], dtype=np.float32)))
        detect_ref.m[1].weight.copy_(torch.from_numpy(np.ascontiguousarray(detect_weights['m1_weight'], dtype=np.float32)))
        detect_ref.m[1].bias.copy_(torch.from_numpy(np.asarray(detect_weights['m1_bias'], dtype=np.float32)))
        detect_ref.m[2].weight.copy_(torch.from_numpy(np.ascontiguousarray(detect_weights['m2_weight'], dtype=np.float32)))
        detect_ref.m[2].bias.copy_(torch.from_numpy(np.asarray(detect_weights['m2_bias'], dtype=np.float32)))

    # Use PyTorch's cached layer-17/20/23 outputs (ref_cache holds tensors)
    feat_ref = [ref_cache[17].clone(), ref_cache[20].clone(), ref_cache[23].clone()]
    with torch.no_grad():
        inf_ref_24, raw_ref_24 = detect_ref(feat_ref)
    inf_ref_24_np = inf_ref_24.numpy()
    raw_ref_24_np = [r.numpy() for r in raw_ref_24]

    # Per-level raw outputs
    total_elements_yolo_e2e = 0
    all_bit_identical = True
    for i, (rs, rr) in enumerate(zip(raw_sub_24, raw_ref_24_np)):
        max_ulp, n_diff, n_total = ulp_distance(rr, rs)
        status = "BIT_IDENTICAL" if max_ulp == 0 else "DIVERGENT"
        if max_ulp != 0:
            all_bit_identical = False
        total_elements_yolo_e2e += n_total
        print(f"raw[{i}] (P{i+3})         {str(list(rs.shape)):<25} {max_ulp:<10} {n_diff:<6}/{n_total:<10} {status}")

    max_ulp, n_diff, n_total = ulp_distance(inf_ref_24_np, inf_sub_24)
    status = "BIT_IDENTICAL" if max_ulp == 0 else "DIVERGENT"
    if max_ulp != 0:
        all_bit_identical = False
    total_elements_yolo_e2e += n_total
    print(f"inference (cat z)     {str(list(inf_sub_24.shape)):<25} {max_ulp:<10} {n_diff:<6}/{n_total:<10} {status}")

    # ── Final verdict ──
    print()
    if all_bit_identical:
        print("🕯️⛵  YOLOv5n END-TO-END BIT_IDENTICAL with PyTorch CPU  ⛵🕯️")
        print(f"     All 24 layers + Detect head: {total_elements_yolo_e2e:,} detection-output floats verified.")
    else:
        print("DIVERGENT somewhere in the end-to-end forward pass.")


if __name__ == "__main__":
    main()
