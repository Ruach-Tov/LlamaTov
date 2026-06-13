#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_yolo_layer0_per_stage.py — Per-stage bit-identity for YOLOv5n Layer 0.

Per Heath's substantive direction 2026-05-20 ~05:30 UTC:
'verify that we are measured and bit-identical for our intermediate computations,
 already even on layers 0+1. as we compose each layer, we will remain at 0ULP
 bit-identical, and try for sass-identical output. we need to know how to do
 everything the stock emitter can do, and then do it better than stock can do it.'

The substantive substrate-design plan:
  1. Empirically observe PyTorch's exact intermediate values at each stage of
     Layer 0 (Conv 3->16, BN, SiLU)
  2. Compare BPD's intermediates at each stage
  3. Surface the precise divergence: which stage diverges, by how many ULP,
     in what direction (over/under PyTorch's value, or sequence-different)
  4. Per medayek's framework: classify per-stage as
     BIT_IDENTICAL / WITHIN_ERROR_BOUND / EXCEEDS_ERROR_BOUND

The goal is NOT to pass; it's to KNOW. Each divergence is a substrate-design
fact: which strategy choice (FMA vs separate ops, accumulation order, rsqrt
variant, SiLU formulation) does PyTorch make? That tells us what to match.
"""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("error: torch required.")

# Disable MKL-DNN so PyTorch uses ATen's naive CPU path (more controllable reference)
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)  # Deterministic single-threaded reference


def ulp_distance(a, b):
    """IEEE 754 sign-magnitude ULP distance."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), int(d.size), float(np.abs(a - b).max())


def classify(ref, sub, label, K_eff=None, max_input=None):
    """Per-stage classification per medayek's framework."""
    max_ulp, n_diffs, n_total, max_abs = ulp_distance(ref, sub)
    if max_ulp == 0:
        return f"  {label:<20}  BIT_IDENTICAL  ({n_total} elements, 0 ULP)"
    if K_eff and max_input:
        # Tier 2 error bound: 6 * sqrt(K) * eps * max|input|
        eps = float(np.finfo(np.float32).eps)
        bound = 6.0 * float(np.sqrt(K_eff)) * eps * max_input
        if max_abs < bound:
            return (f"  {label:<20}  WITHIN_ERROR_BOUND  "
                    f"(max {max_ulp} ULP, {n_diffs}/{n_total} diffs, "
                    f"abs err {max_abs:.2e} < bound {bound:.2e})")
        else:
            return (f"  {label:<20}  EXCEEDS_BOUND  "
                    f"(max {max_ulp} ULP, abs err {max_abs:.2e} > bound {bound:.2e})")
    return (f"  {label:<20}  DIVERGENT  "
            f"(max {max_ulp} ULP, {n_diffs}/{n_total} diffs, abs err {max_abs:.2e})")


def load_bpd_cpu_lib():
    """Load bpd_cpu.so and bind argtypes."""
    so_path = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
    if not os.path.exists(so_path):
        sys.exit(f"error: {so_path} not found")
    lib = ctypes.CDLL(so_path)
    # bpd_conv2d_cpu(input, weight, output, N, C_in, H, W, C_out, kH, kW, stride, pad)
    lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int]
    lib.bpd_conv2d_cpu.restype = None
    # bpd_batchnorm_cpu(input, gamma, beta, mean, var, output, N, C, HW, eps)
    lib.bpd_batchnorm_cpu.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.bpd_batchnorm_cpu.restype = None
    # bpd_batchnorm_cpu_affine_fused(input, gamma, beta, mean, var, output,
    #                                  scale_buf, offset_buf, N, C, HW, eps)
    lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.bpd_batchnorm_cpu_affine_fused.restype = None
    # bpd_silu_cpu(input, output, n)
    lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.bpd_silu_cpu.restype = None
    return lib


def bpd_conv2d(lib, x, w, stride, pad):
    """Run BPD CPU conv2d."""
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    x = np.ascontiguousarray(x, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_cpu(x.ctypes.data, w.ctypes.data, out.ctypes.data,
                       N, C_in, H, W, C_out, kH, kW, stride, pad)
    return out


def bpd_batchnorm(lib, x, gamma, beta, mean, var, eps=1e-5):
    """Run BPD CPU batchnorm (substrate-historical 4-op form)."""
    N, C, H, W = x.shape
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    lib.bpd_batchnorm_cpu(x.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                          mean.ctypes.data, var.ctypes.data, out.ctypes.data,
                          N, C, H*W, eps)
    return out


def bpd_batchnorm_affine_fused(lib, x, gamma, beta, mean, var, eps=1e-5):
    """Run BPD CPU batchnorm — affine-fused form (matches PyTorch bit-for-bit)."""
    N, C, H, W = x.shape
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    scale_buf = np.zeros(C, dtype=np.float32)
    offset_buf = np.zeros(C, dtype=np.float32)
    lib.bpd_batchnorm_cpu_affine_fused(
        x.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
        mean.ctypes.data, var.ctypes.data, out.ctypes.data,
        scale_buf.ctypes.data, offset_buf.ctypes.data,
        N, C, H*W, eps)
    return out


def bpd_silu(lib, x):
    """Run BPD CPU SiLU."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    lib.bpd_silu_cpu(x.ctypes.data, out.ctypes.data, x.size)
    return out


def main():
    print("=" * 80)
    print("verify_yolo_layer0_per_stage.py — per-stage bit-identity for YOLOv5n Layer 0")
    print("Substrate-design discipline: KNOW the divergence, then close it.")
    print("PyTorch path: MKL-DNN disabled, single-threaded (controllable reference)")
    print("=" * 80)

    # Load BPD CPU lib
    lib = load_bpd_cpu_lib()

    # ---- Reproducible input ----
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 3, 320, 320)).astype(np.float32)  # smaller than 640 for faster test
    # ---- Layer 0 weights (small synthetic for empirical clarity) ----
    weight = rng.standard_normal((16, 3, 6, 6)).astype(np.float32) * 0.05
    gamma = rng.standard_normal((16,)).astype(np.float32) * 0.5 + 1.0
    beta = rng.standard_normal((16,)).astype(np.float32) * 0.1
    mean = rng.standard_normal((16,)).astype(np.float32) * 0.2
    var = (rng.standard_normal((16,)).astype(np.float32) * 0.1 + 1.0) ** 2  # positive
    eps = 1e-5

    K_conv = 3 * 6 * 6  # accumulation depth for conv
    K_bn = 1  # batchnorm: per-element no accumulation (just affine)
    K_silu = 1  # silu: per-element

    # ============================================================
    # Stage 0a: Conv2D
    # ============================================================
    print("\n--- Stage 0a: Conv2D (3->16, 6x6, stride=2, pad=2) ---")
    print(f"  Input: {x.shape}, K_eff (accumulation depth) = {K_conv}")

    # BPD CPU
    bpd_conv_out = bpd_conv2d(lib, x, weight, stride=2, pad=2)

    # PyTorch reference (MKL-DNN disabled above)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        wt = torch.from_numpy(weight)
        torch_conv_out = F.conv2d(xt, wt, bias=None, stride=2, padding=2).numpy()

    print(classify(torch_conv_out, bpd_conv_out, "BPD vs PyTorch",
                   K_eff=K_conv, max_input=max(abs(x).max(), abs(weight).max())))

    # Also: f64 truth oracle for the conv
    with torch.no_grad():
        x64 = torch.from_numpy(x).double()
        w64 = torch.from_numpy(weight).double()
        truth_conv = F.conv2d(x64, w64, bias=None, stride=2, padding=2).float().numpy()
    print(classify(truth_conv, bpd_conv_out, "BPD vs f64-truth",
                   K_eff=K_conv, max_input=max(abs(x).max(), abs(weight).max())))
    print(classify(truth_conv, torch_conv_out, "PyTorch vs f64-truth",
                   K_eff=K_conv, max_input=max(abs(x).max(), abs(weight).max())))

    # Stage 0a output to feed Stage 0b
    bpd_after_conv = bpd_conv_out
    torch_after_conv = torch_conv_out

    # ============================================================
    # Stage 0b: BatchNorm
    # ============================================================
    print("\n--- Stage 0b: BatchNorm (16 channels, eval mode) ---")
    print("  Comparing TWO substrate forms:")
    print("    bpd_batchnorm_cpu          (substrate-historical, 4 ops)")
    print("    bpd_batchnorm_cpu_affine_fused  (PyTorch-aligned, 2 ops)")

    # BPD CPU 4-op form (using BPD's own conv output as input)
    bpd_bn_4op_from_bpd = bpd_batchnorm(lib, bpd_after_conv, gamma, beta, mean, var, eps)
    # BPD CPU 4-op form on PyTorch's conv output (isolate BN's divergence from conv's)
    bpd_bn_4op_from_torch_conv = bpd_batchnorm(lib, torch_after_conv, gamma, beta, mean, var, eps)

    # BPD CPU affine-fused form (the substantive fix)
    bpd_bn_fused_from_bpd = bpd_batchnorm_affine_fused(lib, bpd_after_conv, gamma, beta, mean, var, eps)
    bpd_bn_fused_from_torch_conv = bpd_batchnorm_affine_fused(lib, torch_after_conv, gamma, beta, mean, var, eps)

    # PyTorch reference
    with torch.no_grad():
        bn = nn.BatchNorm2d(16)
        bn.weight.data = torch.from_numpy(gamma)
        bn.bias.data = torch.from_numpy(beta)
        bn.running_mean.data = torch.from_numpy(mean)
        bn.running_var.data = torch.from_numpy(var)
        bn.eps = eps
        bn.eval()
        torch_bn_out = bn(torch.from_numpy(torch_after_conv)).numpy()

    # 4-op form: substrate-historical baseline (expect ~32768 ULP divergence)
    print("  --- bpd_batchnorm_cpu (4-op substrate-historical) ---")
    print(classify(torch_bn_out, bpd_bn_4op_from_torch_conv, "4op BPD-BN vs PT-BN (same input)",
                   K_eff=K_bn, max_input=abs(torch_after_conv).max()))
    print(classify(torch_bn_out, bpd_bn_4op_from_bpd, "4op BPD chain vs PT chain",
                   K_eff=K_conv, max_input=max(abs(x).max(), abs(weight).max())))

    # Affine-fused form: substantive substrate-design fix (target: 0 ULP)
    print("  --- bpd_batchnorm_cpu_affine_fused (2-op PyTorch-aligned) ---")
    print(classify(torch_bn_out, bpd_bn_fused_from_torch_conv, "fused BPD-BN vs PT-BN (same input)",
                   K_eff=K_bn, max_input=abs(torch_after_conv).max()))
    print(classify(torch_bn_out, bpd_bn_fused_from_bpd, "fused BPD chain vs PT chain",
                   K_eff=K_conv, max_input=max(abs(x).max(), abs(weight).max())))

    # Use the affine-fused form for the SiLU stage (the substantive forward path)
    bpd_bn_out_from_bpd = bpd_bn_fused_from_bpd

    # ============================================================
    # Stage 0c: SiLU
    # ============================================================
    print("\n--- Stage 0c: SiLU (x * sigmoid(x)) ---")

    # BPD CPU silu on BPD chain
    bpd_silu_out_from_bpd = bpd_silu(lib, bpd_bn_out_from_bpd)

    # BPD CPU silu on PyTorch BN output (isolate silu's divergence)
    bpd_silu_out_from_torch_bn = bpd_silu(lib, torch_bn_out)

    # PyTorch reference
    with torch.no_grad():
        torch_silu_out = F.silu(torch.from_numpy(torch_bn_out)).numpy()

    print(classify(torch_silu_out, bpd_silu_out_from_torch_bn, "BPD-SiLU vs PT-SiLU (same input)",
                   K_eff=K_silu, max_input=abs(torch_bn_out).max()))
    print(classify(torch_silu_out, bpd_silu_out_from_bpd, "BPD chain vs PT chain (full layer 0)",
                   K_eff=K_conv, max_input=max(abs(x).max(), abs(weight).max())))

    # The full-layer-0 substrate output to feed Layer 1
    bpd_layer0_out = bpd_silu_out_from_bpd
    torch_layer0_out = torch_silu_out

    # ============================================================
    # Layer 1: Conv2D (16->32, 3x3, stride=2, pad=1) + BN + SiLU
    # ============================================================
    print("\n" + "=" * 80)
    print("LAYER 1: Conv2D (16->32, 3x3, stride=2, pad=1) + BN + SiLU")
    print("Substrate-design discipline: 0 ULP at every stage, on Layer 0's output")
    print("=" * 80)

    # Layer 1 weights (synthetic, same seed for reproducibility)
    rng_l1 = np.random.default_rng(137)
    weight1 = (rng_l1.standard_normal((32, 16, 3, 3)) * 0.05).astype(np.float32)
    gamma1 = (rng_l1.standard_normal((32,)) * 0.5 + 1.0).astype(np.float32)
    beta1 = (rng_l1.standard_normal((32,)) * 0.1).astype(np.float32)
    mean1 = (rng_l1.standard_normal((32,)) * 0.2).astype(np.float32)
    var1 = ((rng_l1.standard_normal((32,)) * 0.1 + 1.0) ** 2).astype(np.float32)
    K_conv_l1 = 16 * 3 * 3  # accumulation depth for layer 1 conv

    # --- Stage 1a: Conv2D ---
    print("\n--- Stage 1a: Conv2D (16->32, 3x3, stride=2, pad=1) ---")
    print(f"  Input shape: {bpd_layer0_out.shape}, K_eff = {K_conv_l1}")

    # Conv on BPD's layer-0 output (the chain we substantively care about)
    bpd_conv1_from_bpd = bpd_conv2d(lib, bpd_layer0_out, weight1, stride=2, pad=1)

    # Conv on PyTorch's layer-0 output (isolates Layer 1 conv's own divergence)
    bpd_conv1_from_torch = bpd_conv2d(lib, torch_layer0_out, weight1, stride=2, pad=1)

    # PyTorch reference (single-step from torch_layer0_out)
    with torch.no_grad():
        wt1 = torch.from_numpy(weight1)
        torch_conv1_out = F.conv2d(torch.from_numpy(torch_layer0_out), wt1,
                                    bias=None, stride=2, padding=1).numpy()

    max_input_l1 = max(abs(torch_layer0_out).max(), abs(weight1).max())
    print(classify(torch_conv1_out, bpd_conv1_from_torch, "BPD-Conv1 vs PT-Conv1 (same input)",
                   K_eff=K_conv_l1, max_input=max_input_l1))
    print(classify(torch_conv1_out, bpd_conv1_from_bpd, "BPD chain vs PT chain (Layer 0+1a)",
                   K_eff=K_conv_l1, max_input=max_input_l1))

    # --- Stage 1b: BatchNorm ---
    print("\n--- Stage 1b: BatchNorm (32 channels, affine-fused) ---")

    # BN on BPD's L1-conv output (the substantive chain)
    bpd_bn1_from_bpd = bpd_batchnorm_affine_fused(lib, bpd_conv1_from_bpd,
                                                    gamma1, beta1, mean1, var1, eps)
    # BN on PyTorch's L1-conv output (isolates BN's own divergence at L1)
    bpd_bn1_from_torch = bpd_batchnorm_affine_fused(lib, torch_conv1_out,
                                                     gamma1, beta1, mean1, var1, eps)

    # PyTorch reference
    with torch.no_grad():
        bn1_pt = nn.BatchNorm2d(32)
        bn1_pt.weight.data = torch.from_numpy(gamma1)
        bn1_pt.bias.data = torch.from_numpy(beta1)
        bn1_pt.running_mean.data = torch.from_numpy(mean1)
        bn1_pt.running_var.data = torch.from_numpy(var1)
        bn1_pt.eps = eps
        bn1_pt.eval()
        torch_bn1_out = bn1_pt(torch.from_numpy(torch_conv1_out)).numpy()

    print(classify(torch_bn1_out, bpd_bn1_from_torch, "BPD-BN1 vs PT-BN1 (same input)",
                   K_eff=K_bn, max_input=abs(torch_conv1_out).max()))
    print(classify(torch_bn1_out, bpd_bn1_from_bpd, "BPD chain vs PT chain (Layer 0+1a+1b)",
                   K_eff=K_conv_l1, max_input=max_input_l1))

    # --- Stage 1c: SiLU ---
    print("\n--- Stage 1c: SiLU ---")

    bpd_silu1_from_bpd = bpd_silu(lib, bpd_bn1_from_bpd)
    bpd_silu1_from_torch = bpd_silu(lib, torch_bn1_out)

    with torch.no_grad():
        torch_silu1_out = F.silu(torch.from_numpy(torch_bn1_out)).numpy()

    print(classify(torch_silu1_out, bpd_silu1_from_torch, "BPD-SiLU1 vs PT-SiLU1 (same input)",
                   K_eff=K_silu, max_input=abs(torch_bn1_out).max()))
    print(classify(torch_silu1_out, bpd_silu1_from_bpd, "BPD chain vs PT chain (full layer 0+1)",
                   K_eff=K_conv_l1, max_input=max_input_l1))

    # ============================================================
    # Substantive substrate-design summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUBSTANTIVE OBSERVATIONS:")
    print("=" * 80)
    print(
        "1. Each stage compared in two modes:\n"
        "   - SAME INPUT (isolates the stage's own divergence)\n"
        "   - COMPOSED (shows how earlier-stage divergence propagates)\n"
        "\n"
        "2. Truth-oracle (f64) comparisons show the substantive substrate-design\n"
        "   property: both BPD and PyTorch may differ from f64 truth, by\n"
        "   different amounts. Bit-identity with PyTorch ≠ correctness vs truth.\n"
        "\n"
        "3. The substrate-honest substrate-design move per Heath 2026-05-20:\n"
        "   'do everything the stock emitter can do, and then do it better than stock.'\n"
        "   First: KNOW where stock differs from us, and where both differ from truth.\n"
        "   Then: name the substrate-design parameter (reduction_strategy,\n"
        "   rsqrt_variant, etc.) and emit the matching variant.\n"
        "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
