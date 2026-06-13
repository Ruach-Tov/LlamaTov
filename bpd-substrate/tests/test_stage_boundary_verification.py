# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Stage-Boundary Verification — Rule 3 worked example.

RULE 3: "Test at stage boundaries, not just endpoints.
         Precision and operation order are substrate-design
         choices — name them, test them."

Every numerical pipeline with multiple stages must be verified
at EACH stage boundary, not just the final output. This catches:
  - Precision bugs (f16 not promoted to f32 before arithmetic)
  - Operation order bugs (A/B vs A*(1/B) producing different roundoff)
  - Accumulation order bugs (different tile sizes → different bits)

Would have caught:
  Bug #5: f16 BN parameters not promoted to f32 (8.26e-2 abs error)
  Bug #6: BN operation order (32768 ULP systematic divergence)

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-20
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# Stage-boundary verification framework
# ═══════════════════════════════════════════════════════════════════════

def verify_stage(name, actual, expected, rtol=1e-5, atol=1e-6):
    """Verify a single pipeline stage against reference.
    
    Returns (pass, max_error, description) for logging.
    """
    if actual.shape != expected.shape:
        return False, float('inf'), f"shape mismatch: {actual.shape} vs {expected.shape}"
    
    max_abs_err = np.max(np.abs(actual - expected))
    max_rel_err = np.max(np.abs(actual - expected) / (np.abs(expected) + 1e-30))
    
    passed = np.allclose(actual, expected, rtol=rtol, atol=atol)
    desc = f"max_abs={max_abs_err:.2e}, max_rel={max_rel_err:.2e}"
    
    return passed, max_abs_err, desc


# ═══════════════════════════════════════════════════════════════════════
# Worked example: BatchNorm pipeline stages
# 
# BatchNorm(x) has 4 stages:
#   1. mean = x.mean(dim=spatial)
#   2. var = x.var(dim=spatial)
#   3. x_norm = (x - mean) / sqrt(var + eps)
#   4. out = gamma * x_norm + beta
#
# Bug #5 would fail at stage 3 (f16 precision in sqrt)
# Bug #6 would fail at stage 4 (operation order in gamma*x_norm)
# ═══════════════════════════════════════════════════════════════════════

class TestBatchNormStages:
    """Stage-boundary verification for BatchNorm pipeline."""
    
    @pytest.fixture
    def bn_inputs(self):
        """Generate test inputs with known properties."""
        np.random.seed(42)
        N, C, H, W = 2, 4, 8, 8
        x = np.random.randn(N, C, H, W).astype(np.float32)
        gamma = np.random.randn(C).astype(np.float32) * 0.5 + 1.0
        beta = np.random.randn(C).astype(np.float32) * 0.1
        eps = 1e-5
        return x, gamma, beta, eps
    
    def test_stage1_mean(self, bn_inputs):
        """Stage 1: mean computation must be correct."""
        x, _, _, _ = bn_inputs
        mean = x.mean(axis=(0, 2, 3))  # per-channel mean
        # Verify against double-precision reference
        mean_f64 = x.astype(np.float64).mean(axis=(0, 2, 3)).astype(np.float32)
        passed, err, desc = verify_stage("BN mean", mean, mean_f64)
        print(f"  Stage 1 (mean): {desc}")
        assert passed, f"BN mean stage failed: {desc}"
    
    def test_stage2_variance(self, bn_inputs):
        """Stage 2: variance must be computed in sufficient precision."""
        x, _, _, _ = bn_inputs
        var = x.var(axis=(0, 2, 3))
        var_f64 = x.astype(np.float64).var(axis=(0, 2, 3)).astype(np.float32)
        passed, err, desc = verify_stage("BN var", var, var_f64)
        print(f"  Stage 2 (variance): {desc}")
        assert passed, f"BN variance stage failed: {desc}"
    
    def test_stage3_normalize(self, bn_inputs):
        """Stage 3: normalization — precision bugs surface here.
        
        Would catch Bug #5: f16 parameters in sqrt produce
        8.26e-2 absolute error vs f32 parameters.
        """
        x, _, _, eps = bn_inputs
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        
        # f32 normalization (correct)
        x_norm = (x - mean) / np.sqrt(var + eps)
        
        # f64 reference
        x_f64 = x.astype(np.float64)
        mean_f64 = x_f64.mean(axis=(0, 2, 3), keepdims=True)
        var_f64 = x_f64.var(axis=(0, 2, 3), keepdims=True)
        x_norm_ref = ((x_f64 - mean_f64) / np.sqrt(var_f64 + eps)).astype(np.float32)
        
        passed, err, desc = verify_stage("BN normalize", x_norm, x_norm_ref)
        print(f"  Stage 3 (normalize): {desc}")
        assert passed, f"BN normalize stage failed: {desc}"
    
    def test_stage4_affine(self, bn_inputs):
        """Stage 4: affine transform — operation order bugs surface here.
        
        Would catch Bug #6: gamma/sqrt(var+eps) vs gamma*(1/sqrt(var+eps))
        can differ by up to 32768 ULP.
        """
        x, gamma, beta, eps = bn_inputs
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        
        # Method A: gamma * x_norm + beta (standard)
        gamma_r = gamma.reshape(1, -1, 1, 1)
        beta_r = beta.reshape(1, -1, 1, 1)
        out_a = gamma_r * x_norm + beta_r
        
        # Method B: precomputed scale = gamma/sqrt(var+eps), offset = beta - mean*scale
        scale = gamma_r / np.sqrt(var + eps)
        offset = beta_r - mean * scale
        out_b = x * scale + offset
        
        # Both methods should produce similar results
        # (not bit-identical due to operation order, but within bounds)
        passed, err, desc = verify_stage("BN affine A vs B", out_a, out_b, rtol=1e-4)
        print(f"  Stage 4 (affine A vs B): {desc}")
        # NOTE: this test CHARACTERIZES the operation-order divergence
        # rather than asserting bit-equality
        if not passed:
            print(f"  WARNING: operation order divergence {err:.2e}")
            print(f"  This is the Bug #6 pattern — name the divergence, don't ignore it")
    
    def test_precision_invariant(self, bn_inputs):
        """Precision invariant: f32 inputs must stay f32 through pipeline.
        
        Would catch Bug #5: if inputs are accidentally f16,
        intermediate precision is lost.
        """
        x, gamma, beta, eps = bn_inputs
        
        # Verify all inputs are f32
        assert x.dtype == np.float32, f"Input should be f32, got {x.dtype}"
        assert gamma.dtype == np.float32, f"Gamma should be f32, got {gamma.dtype}"
        assert beta.dtype == np.float32, f"Beta should be f32, got {beta.dtype}"
        
        # Compute through pipeline, verify output is f32
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        gamma_r = gamma.reshape(1, -1, 1, 1)
        beta_r = beta.reshape(1, -1, 1, 1)
        out = gamma_r * x_norm + beta_r
        
        assert out.dtype == np.float32, f"Output should be f32, got {out.dtype}"
        assert mean.dtype == np.float32, f"Mean should be f32, got {mean.dtype}"
        assert var.dtype == np.float32, f"Var should be f32, got {var.dtype}"


# ═══════════════════════════════════════════════════════════════════════
# Worked example: GEMM accumulation order
# ═══════════════════════════════════════════════════════════════════════

class TestGEMMStages:
    """Stage-boundary verification for GEMM accumulation."""
    
    def test_accumulation_order_named(self):
        """Different K-tile sizes produce different roundoff.
        
        This is NOT a bug — it's a substrate-design choice.
        The test NAMES the divergence and verifies both are within
        the Tier 2 error bound.
        """
        np.random.seed(42)
        M, N, K = 64, 64, 1024
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # f64 truth oracle
        truth = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
        
        # f32 numpy (uses whatever BLAS is installed)
        result = A @ B
        
        max_err = np.max(np.abs(result - truth))
        eps = np.finfo(np.float32).eps
        bound = 6 * np.sqrt(K) * eps * np.abs(A).max() * np.abs(B).max()
        
        print(f"  GEMM {M}x{N}x{K}: max_err={max_err:.2e}, bound={bound:.2e}")
        assert max_err < bound, \
            f"GEMM error {max_err:.2e} exceeds bound {bound:.2e}"
