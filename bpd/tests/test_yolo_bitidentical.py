#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_yolo_bitidentical.py — Verify BPD bit-identity against YOLOv5n.

Layer-by-layer comparison of Conv2d+BatchNorm+SiLU (CBS) blocks
through our substrate vs PyTorch with trained YOLOv5n weights.

This catches:
  - Conv2d im2col GEMM bugs (including M-tail, N-tail)
  - BatchNorm fusion errors
  - SiLU precision issues
  - Any composition-level wiring bugs

Requires: YOLOv5 repo at /tmp/yolov5, weights at /tmp/yolo_canonical/yolov5n.pt

Extracted from /tmp/verify_yolo_bitidentical.py and formalized for
regression suite.

Author: medayek (formalized from collective work)
"""
#!/usr/bin/env python3
"""Verify BPD bit-identity against YOLOv5n with trained weights, layer by layer."""
import sys, os, types
sys.path.insert(0, "/tmp/yolov5")

import torch, torch.nn as nn, numpy as np, ctypes
torch.set_num_threads(1); torch.backends.mkldnn.enabled = False

lib = ctypes.CDLL(os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so"))
c_void = ctypes.c_void_p; c_int = ctypes.c_int; c_float = ctypes.c_float
lib.bpd_conv2d_full_cpu.argtypes = [c_void]*4 + [c_int]*14; lib.bpd_conv2d_full_cpu.restype = None
lib.bpd_silu_cpu.argtypes = [c_void]*2 + [c_int]; lib.bpd_silu_cpu.restype = None
lib.bpd_batchnorm_cpu_affine_fused.argtypes = [c_void]*8 + [c_int]*3 + [c_float]
lib.bpd_batchnorm_cpu_affine_fused.restype = None
lib.bpd_maxpool2d_cpu.argtypes = [c_void]*2 + [c_int]*8; lib.bpd_maxpool2d_cpu.restype = None

def ulp(a, b):
    af = a.flatten().astype(np.float32)
    bf = b.flatten().astype(np.float32)
    d = np.abs(af.view(np.int32).astype(np.int64) - bf.view(np.int32).astype(np.int64))
    return int(d.max()), int((d > 0).sum()), len(af)

def bpd_conv_bn_silu(x_np, conv_mod, bn_mod):
    """Run Conv2d + BatchNorm2d + SiLU through our substrate."""
    m = conv_mod
    w = np.ascontiguousarray(m.weight.data.numpy(), dtype=np.float32)
    b = np.zeros(m.out_channels, dtype=np.float32)
    if m.bias is not None:
        b = m.bias.data.numpy().astype(np.float32)

    N, Cin, H, W = x_np.shape
    kH, kW = m.kernel_size; sh, sw = m.stride; ph, pw = m.padding
    dh, dw = m.dilation; groups = m.groups
    Ho = (H + 2*ph - dh*(kH-1) - 1) // sh + 1
    Wo = (W + 2*pw - dw*(kW-1) - 1) // sw + 1

    conv_out = np.zeros((N, m.out_channels, Ho, Wo), dtype=np.float32)
    xc = np.ascontiguousarray(x_np, dtype=np.float32)
    lib.bpd_conv2d_full_cpu(xc.ctypes.data, w.ctypes.data, b.ctypes.data, conv_out.ctypes.data,
                             c_int(N), c_int(Cin), c_int(H), c_int(W),
                             c_int(m.out_channels), c_int(kH), c_int(kW),
                             c_int(sh), c_int(sw), c_int(ph), c_int(pw),
                             c_int(dh), c_int(dw), c_int(groups))

    # BatchNorm
    bm = bn_mod
    g = bm.weight.data.numpy().astype(np.float32)
    beta = bm.bias.data.numpy().astype(np.float32)
    mean = bm.running_mean.data.numpy().astype(np.float32)
    var = bm.running_var.data.numpy().astype(np.float32)
    C = bm.num_features
    HW = Ho * Wo
    bn_out = np.zeros_like(conv_out)
    sb = np.zeros(C, dtype=np.float32); ob = np.zeros(C, dtype=np.float32)
    lib.bpd_batchnorm_cpu_affine_fused(conv_out.ctypes.data, g.ctypes.data, beta.ctypes.data,
                                        mean.ctypes.data, var.ctypes.data, bn_out.ctypes.data,
                                        sb.ctypes.data, ob.ctypes.data,
                                        c_int(N), c_int(C), c_int(HW), c_float(bm.eps))

    # SiLU
    silu_out = np.zeros_like(bn_out)
    lib.bpd_silu_cpu(bn_out.ctypes.data, silu_out.ctypes.data, c_int(bn_out.size))

    return silu_out

# Load the model
ckpt = torch.load("/tmp/yolo_canonical/yolov5n.pt", map_location="cpu", weights_only=False)
model = ckpt["model"].float().eval()
print("YOLOv5n loaded (trained weights)")
print()

# Test input
torch.manual_seed(42)
inp = torch.randn(1, 3, 640, 640) * 0.1

print(f"{'Layer':<8} {'Type':<8} {'Shape':<25} {'max_ULP':>8} {'n_diffs':>8} {'Status'}")
print("=" * 70)

x_pt = inp
x_bpd = inp.numpy().astype(np.float32).copy()

all_pass = True
for i, layer in enumerate(model.model):
    layer_type = type(layer).__name__

    # Run PyTorch
    with torch.no_grad():
        x_pt_next = layer(x_pt)

    if layer_type == "Conv":
        # YOLOv5 Conv = Conv2d + BN + SiLU
        conv_mod = layer.conv
        bn_mod = layer.bn
        # Run BPD
        x_bpd_next = bpd_conv_bn_silu(x_bpd, conv_mod, bn_mod)

        pt_np = x_pt_next.numpy().astype(np.float32)
        mu, nd, nt = ulp(x_bpd_next, pt_np)
        status = "BIT_IDENTICAL" if mu == 0 else f"{mu} ULP"
        if mu > 0: all_pass = False

        print(f"{i:<8} {'CBS':<8} {str(list(x_pt_next.shape)):<25} {mu:>8} {nd:>8} {status}")

        x_bpd = x_bpd_next
        x_pt = x_pt_next
    else:
        # For C3, SPPF, etc. — use PyTorch output as BPD input
        # (we haven't routed these compound modules yet)
        x_pt = x_pt_next
        x_bpd = x_pt.numpy().astype(np.float32).copy()
        print(f"{i:<8} {layer_type:<8} {str(list(x_pt.shape)):<25} {'--':>8} {'--':>8} (PyTorch)")

    if i >= 9:
        break  # backbone only

print()
print(f"All CBS layers BIT_IDENTICAL: {'YES ✅' if all_pass else 'NO ❌'}")
