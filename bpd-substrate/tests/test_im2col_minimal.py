#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Find the simplest conv2d case where our im2col diverges from PyTorch."""
import numpy as np, ctypes, torch
torch.set_num_threads(1)

lib = ctypes.CDLL("build/bpd_cpu.so")
c_void = ctypes.c_void_p; c_int = ctypes.c_int
lib.bpd_conv2d_full_cpu.argtypes = [c_void]*4 + [c_int]*14
lib.bpd_conv2d_full_cpu.restype = None

def test_conv(Cin, H, W, Cout, kH, stride, pad, label):
    x = np.arange(Cin*H*W, dtype=np.float32).reshape(1, Cin, H, W) + 1.0
    w = np.ones((Cout, Cin, kH, kH), dtype=np.float32)
    b = np.zeros(Cout, dtype=np.float32)
    Ho = (H + 2*pad - kH) // stride + 1
    Wo = (W + 2*pad - kH) // stride + 1

    bpd_out = np.zeros((1, Cout, Ho, Wo), dtype=np.float32)
    lib.bpd_conv2d_full_cpu(x.ctypes.data, w.ctypes.data, b.ctypes.data, bpd_out.ctypes.data,
        c_int(1), c_int(Cin), c_int(H), c_int(W), c_int(Cout), c_int(kH), c_int(kH),
        c_int(stride), c_int(stride), c_int(pad), c_int(pad), c_int(1), c_int(1), c_int(1))

    xt = torch.from_numpy(x)
    conv = torch.nn.Conv2d(Cin, Cout, kH, stride, pad, bias=False)
    conv.weight.data = torch.from_numpy(w)
    with torch.no_grad():
        pt_out = conv(xt).numpy()

    match = np.array_equal(bpd_out, pt_out)
    max_diff = np.abs(bpd_out - pt_out).max()
    n_diffs = int((bpd_out != pt_out).sum())

    if match:
        print(f"  {label:<40s} 0 ULP")
    else:
        print(f"  {label:<40s} DIFF max={max_diff:.4f} n={n_diffs}")
        # Show first difference location
        for h in range(Ho):
            for ww in range(Wo):
                if bpd_out[0, 0, h, ww] != pt_out[0, 0, h, ww]:
                    print(f"    first diff at [0,0,{h},{ww}]: bpd={bpd_out[0,0,h,ww]:.4f} pt={pt_out[0,0,h,ww]:.4f}")
                    return False
    return match

print("=== Minimal im2col divergence finder ===\n")

# No padding, no stride (simplest)
test_conv(1, 4, 4, 1, 3, 1, 0, "1ch 4x4 k=3 s=1 p=0")
test_conv(1, 8, 8, 1, 3, 1, 0, "1ch 8x8 k=3 s=1 p=0")
test_conv(3, 8, 8, 1, 3, 1, 0, "3ch 8x8 k=3 s=1 p=0")

# With stride (YOLO pattern)
test_conv(1, 4, 4, 1, 3, 2, 0, "1ch 4x4 k=3 s=2 p=0")
test_conv(1, 8, 8, 1, 3, 2, 1, "1ch 8x8 k=3 s=2 p=1")
test_conv(3, 8, 8, 1, 3, 2, 1, "3ch 8x8 k=3 s=2 p=1")
test_conv(16, 8, 8, 1, 3, 2, 1, "16ch 8x8 k=3 s=2 p=1")
test_conv(32, 8, 8, 1, 3, 2, 1, "32ch 8x8 k=3 s=2 p=1")

# With padding
test_conv(1, 4, 4, 1, 3, 1, 1, "1ch 4x4 k=3 s=1 p=1")
test_conv(3, 4, 4, 1, 3, 1, 1, "3ch 4x4 k=3 s=1 p=1")

# 1x1 conv
test_conv(32, 8, 8, 16, 1, 1, 0, "32ch 8x8 k=1 s=1 p=0")
test_conv(64, 4, 4, 32, 1, 1, 0, "64ch 4x4 k=1 s=1 p=0")

# YOLO-like
test_conv(32, 160, 160, 64, 3, 2, 1, "32ch 160x160 k=3 s=2 p=1 (YOLO L3)")
test_conv(64, 80, 80, 128, 3, 2, 1, "64ch 80x80 k=3 s=2 p=1 (YOLO L5)")

# Bigger Cin sweep to find threshold
print("\n--- Cin sweep (8x8, k=3, s=2, p=1) ---")
for cin in [1, 2, 4, 8, 16, 24, 32, 48, 64]:
    test_conv(cin, 8, 8, 1, 3, 2, 1, f"{cin}ch 8x8 k=3 s=2 p=1")
