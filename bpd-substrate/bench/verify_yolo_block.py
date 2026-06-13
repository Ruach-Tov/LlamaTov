#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify one YOLO backbone block: Conv2D + BatchNorm + Mish.

Defines the block in PyTorch, extracts weights, runs the same
computation through our BPD kernels, compares bit-for-bit.

This is the repeating unit in YOLOv4-CSPDarknet. If one block
matches, the full backbone is N repetitions of verified blocks.
"""
import ctypes, os, sys, numpy as np

try:
    import torch
    import torch.nn as nn
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    sys.exit("error: torch with CUDA required")

SO_PATH = os.environ.get("BPD_YOLO_SO", "/tmp/bpd_yolo_block.so")

def load_lib():
    if not os.path.exists(SO_PATH):
        sys.exit(f"error: {SO_PATH} not found")
    lib = ctypes.CDLL(SO_PATH)
    lib.gpu_alloc.restype = ctypes.c_void_p
    lib.gpu_alloc.argtypes = [ctypes.c_int]
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_sync.argtypes = []
    lib.bpd_conv2d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int]
    lib.bpd_batchnorm.argtypes = [ctypes.c_void_p]*6 + [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.bpd_mish.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    return lib

def to_gpu(lib, arr):
    d = lib.gpu_alloc(arr.nbytes)
    lib.gpu_h2d(d, arr.ctypes.data, arr.nbytes)
    return d

def from_gpu(lib, d, shape, dtype=np.float32):
    out = np.zeros(shape, dtype=dtype)
    lib.gpu_d2h(out.ctypes.data, d, out.nbytes)
    return out

def ulp(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum())

def mish_np(x):
    return x * np.tanh(np.log1p(np.exp(x)))

def main():
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.enabled = False  # Pascal sm_61 cuDNN compat
    lib = load_lib()
    torch.manual_seed(42)

    # ── Define the YOLO block in PyTorch ──
    C_in, C_out = 32, 64
    kH, kW = 3, 3
    stride, pad = 1, 1
    N, H, W = 1, 16, 16
    eps = 1e-5

    conv = nn.Conv2d(C_in, C_out, (kH, kW), stride=stride, padding=pad, bias=False).cuda()
    bn = nn.BatchNorm2d(C_out, eps=eps).cuda()
    bn.eval()  # inference mode — use running stats

    # Force some running stats
    with torch.no_grad():
        for _ in range(10):
            bn(conv(torch.randn(N, C_in, H, W, device='cuda')))
    bn.eval()

    # ── Run PyTorch reference ──
    x = torch.randn(N, C_in, H, W, device='cuda')
    with torch.no_grad():
        conv_out = conv(x)
        bn_out = bn(conv_out)
        # Mish: PyTorch uses torch.nn.functional.mish
        import torch.nn.functional as F
        ref = F.mish(bn_out)

    ref_np = ref.cpu().numpy()

    # ── Extract weights ──
    w_conv = conv.weight.data.cpu().numpy().astype(np.float32)       # [C_out, C_in, kH, kW]
    w_gamma = bn.weight.data.cpu().numpy().astype(np.float32)        # [C_out]
    w_beta = bn.bias.data.cpu().numpy().astype(np.float32)           # [C_out]
    w_mean = bn.running_mean.cpu().numpy().astype(np.float32)        # [C_out]
    w_var = bn.running_var.cpu().numpy().astype(np.float32)          # [C_out]
    x_np = x.cpu().numpy().astype(np.float32)

    # ── Run BPD kernels ──
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1

    dX = to_gpu(lib, x_np)
    dW = to_gpu(lib, w_conv)
    dConvOut = lib.gpu_alloc(N * C_out * H_out * W_out * 4)

    # Step 1: Conv2D
    lib.bpd_conv2d(dX, dW, dConvOut, N, C_in, H, W, C_out, kH, kW, stride, pad)
    lib.gpu_sync()

    # Verify conv alone
    conv_bpd = from_gpu(lib, dConvOut, (N, C_out, H_out, W_out))
    conv_ref = conv_out.cpu().numpy()
    conv_ulp, conv_diffs = ulp(conv_ref, conv_bpd)

    # Step 2: BatchNorm
    dGamma = to_gpu(lib, w_gamma)
    dBeta = to_gpu(lib, w_beta)
    dMean = to_gpu(lib, w_mean)
    dVar = to_gpu(lib, w_var)
    dBnOut = lib.gpu_alloc(N * C_out * H_out * W_out * 4)

    lib.bpd_batchnorm(dConvOut, dGamma, dBeta, dMean, dVar, dBnOut,
                       N, C_out, H_out * W_out, ctypes.c_float(eps))
    lib.gpu_sync()

    bn_bpd = from_gpu(lib, dBnOut, (N, C_out, H_out, W_out))
    bn_ref = bn_out.cpu().numpy()
    bn_ulp, bn_diffs = ulp(bn_ref, bn_bpd)

    # Step 3: Mish
    total_elem = N * C_out * H_out * W_out
    dMishOut = lib.gpu_alloc(total_elem * 4)
    lib.bpd_mish(dBnOut, dMishOut, total_elem)
    lib.gpu_sync()

    mish_bpd = from_gpu(lib, dMishOut, (N, C_out, H_out, W_out))
    mish_ulp, mish_diffs = ulp(ref_np, mish_bpd)

    # ── Report ──
    print(f"\nYOLO Block: Conv2D({C_in}→{C_out}, {kH}×{kW}) + BatchNorm + Mish")
    print(f"Input: {N}×{C_in}×{H}×{W} → Output: {N}×{C_out}×{H_out}×{W_out}")
    print(f"Total elements per stage: {total_elem}")
    print()
    print(f"  Stage 1 (Conv2D):    max {conv_ulp:>6} ULP  ({conv_diffs}/{total_elem} diffs)")
    print(f"  Stage 2 (BatchNorm): max {bn_ulp:>6} ULP  ({bn_diffs}/{total_elem} diffs)")
    print(f"  Stage 3 (Mish):      max {mish_ulp:>6} ULP  ({mish_diffs}/{total_elem} diffs)")
    print()

    if mish_ulp == 0:
        print("  *** FULL BLOCK BIT-IDENTICAL WITH PyTorch ***")
    elif mish_ulp <= 4:
        print(f"  PASS: {mish_ulp} ULP (within reduction-order tolerance)")
    else:
        print(f"  Max absolute error: {np.abs(ref_np - mish_bpd).max():.8f}")
        # Find worst element
        diff = np.abs(ref_np - mish_bpd)
        worst = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  Worst at {worst}: ref={ref_np[worst]:.8f} bpd={mish_bpd[worst]:.8f}")

    # Cleanup
    for d in [dX, dW, dConvOut, dGamma, dBeta, dMean, dVar, dBnOut, dMishOut]:
        lib.gpu_free(d)

    return 0 if mish_ulp <= 4 else 1

if __name__ == "__main__":
    sys.exit(main())
