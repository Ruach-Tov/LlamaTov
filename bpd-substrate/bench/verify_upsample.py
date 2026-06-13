#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify upsample_nearest2d is bit-identical with PyTorch.

Tests:
  1. Hand-written 2×2 → 4×4 (verifiable by inspection)
  2. Random inputs at multiple shapes
  3. YOLO-typical shapes (e.g., 1×256×13×13 → 1×256×26×26)

All comparisons are bit-exact (0 ULP) because nearest-neighbor
upsample is a pure data movement op — no floating-point arithmetic.
"""
import ctypes, os, sys, numpy as np

try:
    import torch
    import torch.nn.functional as F
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    sys.exit("error: torch with CUDA required")

SO_PATH = os.environ.get("BPD_UPSAMPLE_SO", "/tmp/bpd_upsample.so")

def load_lib():
    if not os.path.exists(SO_PATH):
        sys.exit(f"error: {SO_PATH} not found. Compile bench/upsample.cu first.")
    lib = ctypes.CDLL(SO_PATH)
    lib.gpu_alloc.restype = ctypes.c_void_p
    lib.gpu_alloc.argtypes = [ctypes.c_int]
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_sync.argtypes = []
    lib.bpd_upsample_nearest2d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    return lib

def bpd_upsample(lib, x_np):
    N, C, H, W = x_np.shape
    out_size = N * C * (2*H) * (2*W)
    dI = lib.gpu_alloc(x_np.size * 4)
    dO = lib.gpu_alloc(out_size * 4)
    lib.gpu_h2d(dI, x_np.ctypes.data, x_np.size * 4)
    lib.bpd_upsample_nearest2d(dI, dO, N, C, H, W)
    lib.gpu_sync()
    out = np.zeros((N, C, 2*H, 2*W), dtype=np.float32)
    lib.gpu_d2h(out.ctypes.data, dO, out_size * 4)
    lib.gpu_free(dI); lib.gpu_free(dO)
    return out

def main():
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    lib = load_lib()
    all_pass = True

    # ── Test 1: Hand-written 1×1×2×2 → 1×1×4×4 ──
    print("\n── Test 1: Hand-written 2×2 → 4×4 ──")
    x = np.array([[[[1.0, 2.0],
                     [3.0, 4.0]]]], dtype=np.float32)
    expected = np.array([[[[1.0, 1.0, 2.0, 2.0],
                            [1.0, 1.0, 2.0, 2.0],
                            [3.0, 3.0, 4.0, 4.0],
                            [3.0, 3.0, 4.0, 4.0]]]], dtype=np.float32)
    got = bpd_upsample(lib, x)
    match_expected = np.array_equal(got, expected)
    pt_ref = F.interpolate(torch.from_numpy(x).cuda(), scale_factor=2, mode='nearest').cpu().numpy()
    match_pytorch = np.array_equal(got, pt_ref)
    print(f"  Expected match: {match_expected}  PyTorch match: {match_pytorch}")
    if not match_expected:
        print(f"  Expected:\n{expected}")
        print(f"  Got:\n{got}")
        all_pass = False

    # ── Test 2: Hand-written 1×2×3×3 → 1×2×6×6 ──
    print("\n── Test 2: Multi-channel 2×3×3 → 2×6×6 ──")
    x2 = np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)
    got2 = bpd_upsample(lib, x2)
    pt_ref2 = F.interpolate(torch.from_numpy(x2).cuda(), scale_factor=2, mode='nearest').cpu().numpy()
    match2 = np.array_equal(got2, pt_ref2)
    # Verify by hand: element [0,0,0,0] = x[0,0,0,0] = 0.0
    #                  element [0,0,1,0] = x[0,0,0,0] = 0.0 (oh=1, ih=0)
    #                  element [0,0,2,0] = x[0,0,1,0] = 3.0 (oh=2, ih=1)
    hand_check = (got2[0,0,0,0] == 0.0 and got2[0,0,1,0] == 0.0 and
                  got2[0,0,2,0] == 3.0 and got2[0,1,0,0] == 9.0)
    print(f"  PyTorch match: {match2}  Hand-check: {hand_check}")
    if not match2: all_pass = False

    # ── Test 3: Random inputs at multiple shapes ──
    print("\n── Test 3: Random shapes ──")
    rng = np.random.default_rng(42)
    shapes = [
        (1, 1, 4, 4),        # minimal
        (1, 3, 8, 8),        # small RGB
        (2, 64, 16, 16),     # typical feature map
        (1, 256, 13, 13),    # YOLO backbone (before upsample)
        (4, 128, 32, 32),    # larger batch
    ]
    for shape in shapes:
        x_np = rng.standard_normal(shape).astype(np.float32)
        got = bpd_upsample(lib, x_np)
        ref = F.interpolate(torch.from_numpy(x_np).cuda(), scale_factor=2, mode='nearest').cpu().numpy()
        match = np.array_equal(got, ref)
        N, C, H, W = shape
        tag = "0 ULP ✓" if match else "MISMATCH"
        if not match: all_pass = False
        print(f"  {N}×{C}×{H}×{W} → {N}×{C}×{2*H}×{2*W}  {got.size:>8} elements  {tag}")

    print()
    if all_pass:
        print("PASS: upsample_nearest2d is BIT-IDENTICAL with PyTorch F.interpolate.")
        print("      Pure data movement — no floating-point arithmetic — every bit matches.")
    else:
        print("FAIL: mismatch detected.")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
