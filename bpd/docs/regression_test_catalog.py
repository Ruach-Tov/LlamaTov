# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""LlamaTov Regression Test Catalog — comprehensive bit-identity test coverage.

Catalogs ALL existing test cases and identifies gaps for regression testing.
Goal: catch any divergence from bit-identical operations (vs ggml or vs torch)
before it reaches production.

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-24

EXISTING TEST LEVELS:
=====================

Level 0: UNIT TESTS (synthetic inputs, verify math properties)
  tests/test_llama_l1_kernels.py — 10 tests, ALL 0 ULP
    L.1.4a-c: RMSNorm (no-weight, with-weight, double-accum stress)
    L.1.5a-c: RoPE NEOX (Llama3, Llama2, decode pos=512)
    L.1.6a-b: KV cache write (sequential, scattered positions)
    L.1.7a-b: Causal softmax (prefill, decode)
    
Level 1: PER-KERNEL FIXTURE TESTS (fixture inputs, compare to ggml output)
  tests/correctness/per_op_gates.py — 15 op types, 1137 ops
    GET_ROWS, RMS_NORM, MUL, MUL_MAT, ROPE, ADD, SILU, SOFT_MAX, CPY
    RESHAPE, VIEW, PERMUTE, TRANSPOSE, CONT, NONE
    
  tests/test_embed_entry_point.py — embedding from raw GGUF (0 ULP) ✅
    Catches dequant errors that per_op_gates can't (tests GGUF→embedding path)

Level 2: COMPOSITION TESTS (wiring between kernels)
  tests/test_block_layer0.py — single block composition
  tests/correctness/per_layer_replay.py — per-layer with fixture inputs
  tests/correctness/replay_gate.py — sequential graph replay (has fixture reshape bug)
  
Level 3: END-TO-END (full forward pass)
  tests/correctness/end_to_end_gate.py — token prediction comparison
    Argmax match + correlation + stale-data triple-check

EXISTING DOMAIN-SPECIFIC TESTS:
================================

YOLO:
  bench/verify_yolo_block.py — YOLO block composition
  bench/verify_yolo_per_stage.py — per-stage verification
  bench/verify_yolo_composition_sweep.py — fusion sweep
  bench/verify_yolo_layer*.py — specific layer tests

BLAS:
  bench/verify_blas.py — basic linear algebra
  bench/verify_gemm_sweep.py — GEMM parameter sweep
  bench/verify_mm_avx1.py — AVX1 matmul
  bench/test_gemm_q8_0_cpu.py — Q8_0 quantized matmul
  bench/test_gemm_q8_0_tiles.py — tile dispatch

Smith-Waterman:
  bench/test_smith_waterman.py — SW alignment
  bench/test_sw_batch.py — batched SW
  bench/verify_smith_waterman.py — vs Python reference

Stanford L1:
  bench/verify_kernelbench_l1_cpu.py — 94/100 BIT_IDENTICAL
  bench/verify_kernelbench_l2_cpu.py — L2 kernel verification

CFD (torch-cfd lifted):
  tests/test_torch_cfd_stencils.py (in Ruach-Tov repo) — 7/8 BIT_IDENTICAL
  tests/test_torch_cfd_spectral.py (in Ruach-Tov repo) — 8/11 BIT_IDENTICAL
  tests/test_gpu_kernel_compare.py (in Ruach-Tov repo) — 13 kernels GPU vs CPU

GAPS — TESTS THAT SHOULD EXIST:
================================

1. test_softmax_polynomial.py
   Verify bpd_softmax_causal_ggml_cpu (polynomial exp) against fixture
   Currently proved 0 ULP by standalone test but NOT in regression suite
   
2. test_f16_roundtrip_ggml.py
   Verify f32→f16→f32 using Maratyszcza algorithm matches ggml
   Currently has .astype() vs .view() bug in test_kv_cache_f16_roundtrip.py
   
3. test_attention_separated.py
   Verify bpd_separated_attn_cpu with post-scaled batch softmax
   Currently proved 1859 ULP vs Flash, needs fixture comparison

4. test_scale_application_path.py  
   Verify unscaled QK^T matches fixture idx 41 (raw dot products)
   This caught the 8× factor — should be a permanent regression test

5. test_per_op_gates_c_kernels.py
   verify_soft_max using C kernel (not Python numpy.exp)
   Currently reports 4 ULP artifact from Python; C kernel is 0 ULP

6. test_multi_token_generation.py
   Autoregressive chain: generate N tokens and compare each against ggml
   Catches accumulation bugs across multiple inference steps

7. test_llama_all_layers.py
   Per-op gates across ALL 16 layers (not just layer 0-1)
   Currently max-ops limits coverage

RECOMMENDED make verify_l1 TARGET:
====================================

make verify_l1_fast:    # <5 seconds
    test_llama_l1_kernels.py (10 synthetic tests)
    test_embed_entry_point.py (embedding dequant)

make verify_l1_medium:  # <30 seconds
    per_op_gates.py --max-ops 200 (layer 0 coverage)
    test_block_layer0.py (wiring)

make verify_l1_full:    # <5 minutes
    per_op_gates.py --max-ops 9999 (all 1137 ops)
    end_to_end_gate.py (token match)
    test_softmax_polynomial.py (0 ULP exp)
    test_scale_application_path.py (QK^T scale)
"""
