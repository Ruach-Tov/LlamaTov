# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""ADDITIONAL test cases identified during codebase survey (task 1f).

These are in bench/ or /tmp/ and should be formalized into the regression suite.

PRIORITY 1 — High-value, catches real bug classes:

  verify_kernelbench_l1_cpu.py (1235L)
    94/100 Stanford KernelBench L1 problems BIT_IDENTICAL with PyTorch.
    6 remaining: SELU, ELU (expf), InstanceNorm, RMSNorm (reduction),
    CrossEntropy (logf), ScaledDotProductAttention (scale_application_path).
    → Should be in make verify_l1_full

  verify_cascade_sweep.py (160L)
    Sweeps cascade-reduction orderings for GEMM accumulation.
    Catches reduction-order bugs (same class as softmax_reduction_order).
    → Should be in make verify_l1_medium

  verify_gemm_sweep.py (178L)
    Sweeps all gemm_pattern instantiations for bit-identity.
    Catches tile dispatch bugs (same class as MUL_MAT fix).
    → Should be in make verify_l1_medium

  verify_quant.py (137L)
    Q4_K dequantization vs llama.cpp reference.
    Catches quantizer bugs (same class as matmul_quantizer_path).
    → Should be in make verify_l1_fast

PRIORITY 2 — Domain-specific, catches composition bugs:

  verify_fusion_F3.py (200L) + verify_fusion_F3_v2.py (146L)
    Conv+BN+SiLU fused kernel bit-identity vs unfused.
    → YOLO domain regression

  verify_fusion_F7.py (121L)
    Conv+Bias+Sigmoid fused kernel.
    → YOLO domain regression

  verify_yolo_layer2_c3.py (217L)
    C3 bottleneck block bit-identity.
    → YOLO domain regression

  verify_yolo_layer9_sppf.py (131L)
    SPPF (Spatial Pyramid Pooling Fast) bit-identity.
    → YOLO domain regression

  verify_yolo_layer24_detect.py (161L)
    Detect head bit-identity.
    → YOLO domain regression

PRIORITY 3 — GPU-specific:

  verify_quant_gpu.py (208L)
    Q4_K GPU dequant + tokens/sec measurement.
    → GPU regression when we have GPU CI

  verify_sw_gpu.py (119L)
    Smith-Waterman GPU vs CPU reference.
    → GPU domain regression

  tier2/sass_audit.py (113L)
    Disassembles emitted kernels and characterizes instruction mix.
    → Code generation quality gate

TOTAL: 19 bench verifiers + 59 /tmp scripts available for extraction.
Recommended: formalize the 4 Priority 1 tests first.
"""
