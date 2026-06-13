# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_cpu_references.py — verify cpu_references.py implementations.

Per Scope C-Extended-A: verify the PyTorch reference implementations
in bpd/lib/cpu_references.py satisfy their mathematical invariants AND
match PyTorch's builtin equivalents where applicable.

These references become the GROUND TRUTH for Scope C-Extended-B
(when GPU is available: compile emitted CUDA → execute → compare to
these references). Until then, validating the references is what we
can do CPU-only.

Author: metayen 2026-05-15
Per mavchin's directive (inbox 19:30): "correctness before coverage."
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import pytest
import torch
import torch.nn.functional as F

import cpu_references as cr


# ════════════════════════════════════════════════════════════════════════
# Test fixtures (random tensors with fixed seed for reproducibility)
# ════════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_2d():
    torch.manual_seed(42)
    return torch.randn(8, 16)


@pytest.fixture
def small_image():
    torch.manual_seed(42)
    return torch.randn(2, 3, 8, 8)


@pytest.fixture
def pos_probs():
    """Positive probability-like tensor (rows sum to 1) — for KL/CE refs."""
    torch.manual_seed(42)
    raw = torch.rand(4, 16)
    return raw / raw.sum(dim=-1, keepdim=True)


# ════════════════════════════════════════════════════════════════════════
# Family 1: Reductions — verify against torch builtins
# ════════════════════════════════════════════════════════════════════════

class TestReductionReferences:

    def test_sum_rows_matches_torch_sum(self, small_2d):
        ours = cr.cpu_reference_sum_rows(small_2d)
        builtin = small_2d.sum(dim=-1)
        assert torch.allclose(ours, builtin)

    def test_mean_matches_torch_mean(self, small_2d):
        ours = cr.cpu_reference_mean(small_2d)
        builtin = small_2d.mean(dim=-1)
        assert torch.allclose(ours, builtin)

    def test_max_matches_torch_max(self, small_2d):
        ours = cr.cpu_reference_max(small_2d)
        builtin = small_2d.max(dim=-1).values
        assert torch.allclose(ours, builtin)

    def test_min_matches_torch_min(self, small_2d):
        ours = cr.cpu_reference_min(small_2d)
        builtin = small_2d.min(dim=-1).values
        assert torch.allclose(ours, builtin)

    def test_argmax_returns_float(self, small_2d):
        ours = cr.cpu_reference_argmax(small_2d)
        assert ours.dtype == torch.float32
        # Cast to int and verify it matches torch.argmax
        builtin = small_2d.argmax(dim=-1)
        assert torch.equal(ours.long(), builtin)

    def test_argmin_returns_float(self, small_2d):
        ours = cr.cpu_reference_argmin(small_2d)
        builtin = small_2d.argmin(dim=-1)
        assert torch.equal(ours.long(), builtin)

    def test_cumsum_matches_torch_cumsum(self, small_2d):
        ours = cr.cpu_reference_cumsum(small_2d)
        builtin = small_2d.cumsum(dim=-1)
        assert torch.allclose(ours, builtin)

    def test_cumprod_matches_torch_cumprod(self, small_2d):
        ours = cr.cpu_reference_cumprod(small_2d)
        builtin = small_2d.cumprod(dim=-1)
        assert torch.allclose(ours, builtin)


# ════════════════════════════════════════════════════════════════════════
# Family 2: Normalizations — verify mathematical invariants
# ════════════════════════════════════════════════════════════════════════

class TestNormalizationReferences:

    def test_layer_norm_matches_F_layer_norm(self, small_2d):
        """Without affine, our reference should match F.layer_norm."""
        ours = cr.cpu_reference_layer_norm(small_2d, eps=1e-5)
        # F.layer_norm normalizes over the last D dims given normalized_shape
        builtin = F.layer_norm(small_2d, [small_2d.shape[-1]],
                                eps=1e-5)
        assert torch.allclose(ours, builtin, atol=1e-5)

    def test_layer_norm_with_affine_applies_weight_bias(self, small_2d):
        weight = torch.ones(small_2d.shape[-1]) * 2.0
        bias = torch.ones(small_2d.shape[-1])
        ours = cr.cpu_reference_layer_norm(small_2d, eps=1e-5,
                                            weight=weight, bias=bias)
        # Mean across last dim should be ~1.0 (after the bias)
        # Standard deviation should be ~2.0 (scale weight)
        plain = cr.cpu_reference_layer_norm(small_2d, eps=1e-5)
        expected = plain * weight + bias
        assert torch.allclose(ours, expected)

    def test_rms_norm_makes_mean_squares_close_to_1(self, small_2d):
        """RMS-normalized output should have mean(x²) ≈ 1 along inner dim."""
        ours = cr.cpu_reference_rms_norm(small_2d, eps=1e-8)
        mean_sq = (ours * ours).mean(dim=-1)
        assert torch.allclose(mean_sq, torch.ones_like(mean_sq), atol=1e-4)

    def test_rms_norm_matches_manual_formula(self, small_2d):
        """RMS = sqrt(mean(x²) + eps); Y = X / RMS."""
        eps = 1e-5
        ours = cr.cpu_reference_rms_norm(small_2d, eps=eps)
        rms = (small_2d * small_2d).mean(dim=-1, keepdim=True).add(eps).sqrt()
        manual = small_2d / rms
        assert torch.allclose(ours, manual, atol=1e-6)

    def test_l2_norm_makes_sum_squares_close_to_1(self, small_2d):
        """L2-normalized output should have sum(x²) ≈ 1 along inner dim."""
        ours = cr.cpu_reference_l2_norm(small_2d, eps=1e-8)
        sum_sq = (ours * ours).sum(dim=-1)
        assert torch.allclose(sum_sq, torch.ones_like(sum_sq), atol=1e-4)

    def test_l2_norm_distinct_from_rms_norm(self, small_2d):
        """L2 norm uses SUM of squares, RMS uses MEAN — should differ."""
        l2 = cr.cpu_reference_l2_norm(small_2d, eps=1e-8)
        rms = cr.cpu_reference_rms_norm(small_2d, eps=1e-8)
        assert not torch.allclose(l2, rms, atol=1e-4)

    def test_group_norm_delegates_to_layer_norm(self, small_2d):
        """Group norm currently delegates; verify same output."""
        g = cr.cpu_reference_group_norm(small_2d, eps=1e-5)
        l = cr.cpu_reference_layer_norm(small_2d, eps=1e-5)
        assert torch.allclose(g, l)


# ════════════════════════════════════════════════════════════════════════
# Family 3: Pooling — verify against F.max_pool2d / F.avg_pool2d
# ════════════════════════════════════════════════════════════════════════

class TestPoolingReferences:

    def test_pool_2d_max_matches_F_max_pool2d(self, small_image):
        ours = cr.cpu_reference_pool_2d_max(small_image, kH=2, kW=2,
                                              stride_h=2, stride_w=2)
        builtin = F.max_pool2d(small_image, kernel_size=2, stride=2)
        assert torch.allclose(ours, builtin)

    def test_pool_2d_avg_matches_F_avg_pool2d_exclude_pad(self, small_image):
        ours = cr.cpu_reference_pool_2d_avg(small_image, kH=2, kW=2,
                                              stride_h=2, stride_w=2)
        builtin = F.avg_pool2d(small_image, kernel_size=2, stride=2,
                                count_include_pad=False)
        assert torch.allclose(ours, builtin)

    def test_pool_2d_output_shape_correct(self, small_image):
        """Input [2,3,8,8] with k=2, s=2 → [2,3,4,4]."""
        ours = cr.cpu_reference_pool_2d_max(small_image, kH=2, kW=2,
                                              stride_h=2, stride_w=2)
        assert ours.shape == (2, 3, 4, 4)


# ════════════════════════════════════════════════════════════════════════
# Family 4: Convolutions — verify im2col + matmul = conv2d
# ════════════════════════════════════════════════════════════════════════

class TestConvolutionReferences:

    def test_im2col_2d_output_shape(self, small_image):
        """im2col 2D output: [B*outH*outW, C*kH*kW]."""
        B, C, H, W = small_image.shape  # 2, 3, 8, 8
        kH, kW = 3, 3
        unfolded = cr.cpu_reference_im2col_2d(small_image, kH, kW,
                                                stride_h=1, stride_w=1,
                                                pad_h=0, pad_w=0)
        # outH = 8 - 3 + 1 = 6, outW = 6
        expected_rows = B * 6 * 6
        expected_cols = C * kH * kW
        assert unfolded.shape == (expected_rows, expected_cols)

    def test_im2col_plus_matmul_equals_conv2d(self, small_image):
        """im2col(X) @ W.T (reshaped) = conv2d(X, W) — the composition."""
        kH, kW = 3, 3
        Cout = 4
        # Weight: [Cout, Cin, kH, kW]
        weight = torch.randn(Cout, 3, kH, kW)
        # Conv reference
        conv_out = cr.cpu_reference_conv_2d(small_image, weight,
                                              stride_h=1, stride_w=1,
                                              pad_h=0, pad_w=0)
        # im2col + matmul reconstruction
        B = small_image.shape[0]
        unfolded = cr.cpu_reference_im2col_2d(small_image, kH, kW)
        # Weight reshape: [Cout, Cin*kH*kW]
        w_flat = weight.view(Cout, -1)
        # matmul: [B*outH*outW, Cin*kH*kW] × [Cin*kH*kW, Cout] = [B*outH*outW, Cout]
        matmul_out = unfolded @ w_flat.T
        # Reshape: [B*outH*outW, Cout] → [B, Cout, outH, outW]
        outH, outW = conv_out.shape[2], conv_out.shape[3]
        reconstructed = matmul_out.view(B, outH, outW, Cout).permute(0, 3, 1, 2)
        assert torch.allclose(reconstructed, conv_out, atol=1e-5)


# ════════════════════════════════════════════════════════════════════════
# Family 5: Losses — verify against F builtins
# ════════════════════════════════════════════════════════════════════════

class TestLossReferences:

    def test_mse_matches_F_mse_loss(self):
        torch.manual_seed(42)
        x = torch.randn(4, 16)
        y = torch.randn(4, 16)
        ours = cr.cpu_reference_mse_loss(x, y, reduction='mean')
        builtin = F.mse_loss(x, y, reduction='mean')
        assert torch.allclose(ours, builtin)

    def test_huber_smooth_at_origin(self):
        """Huber loss at x=y=0 should be exactly 0."""
        x = torch.zeros(4, 16)
        y = torch.zeros(4, 16)
        loss = cr.cpu_reference_huber_loss(x, y, delta=1.0, reduction='mean')
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_huber_quadratic_region(self):
        """When |x-y| < delta, Huber = 0.5*(x-y)² (matches MSE/2)."""
        torch.manual_seed(42)
        x = torch.randn(4, 16) * 0.1  # small values
        y = torch.zeros(4, 16)
        huber = cr.cpu_reference_huber_loss(x, y, delta=1.0, reduction='mean')
        # In quadratic region, expected = mean(0.5 * x²)
        expected = (0.5 * x * x).mean()
        assert torch.allclose(huber, expected, atol=1e-6)

    def test_huber_linear_region(self):
        """When |x-y| >> delta, Huber becomes linear in |x-y|."""
        delta = 0.5
        x = torch.tensor([5.0])  # well above delta
        y = torch.tensor([0.0])
        huber = cr.cpu_reference_huber_loss(x, y, delta=delta, reduction='mean')
        # Expected: delta * (|x-y| - 0.5*delta) = 0.5 * (5 - 0.25) = 2.375
        expected = torch.tensor(0.5 * (5.0 - 0.5 * 0.5))
        assert torch.allclose(huber, expected, atol=1e-6)

    def test_kl_div_zero_when_distributions_equal(self, pos_probs):
        """KL(P || P) = 0."""
        kl = cr.cpu_reference_kl_div_loss(pos_probs, pos_probs, reduction='sum')
        assert torch.allclose(kl, torch.tensor(0.0), atol=1e-5)

    def test_hinge_zero_when_correctly_classified_with_margin(self):
        """When y*x > 1, hinge = 0."""
        x = torch.tensor([2.0, -2.0, 3.0])
        y = torch.tensor([1.0, -1.0, 1.0])
        loss = cr.cpu_reference_hinge_loss(x, y, reduction='sum')
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_hinge_positive_when_margin_violated(self):
        """When y*x < 1, hinge = 1 - y*x > 0."""
        x = torch.tensor([0.5])
        y = torch.tensor([1.0])
        loss = cr.cpu_reference_hinge_loss(x, y, reduction='sum')
        # Expected: 1 - 0.5 = 0.5
        assert torch.allclose(loss, torch.tensor(0.5))

    def test_triplet_zero_when_correctly_ordered_with_margin(self):
        """When d_an > d_ap + margin, triplet loss = 0."""
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[0.1, 0.1]])  # close to anchor
        negative = torch.tensor([[5.0, 5.0]])  # far from anchor
        loss = cr.cpu_reference_triplet_margin_loss(
            anchor, positive, negative, margin=0.5
        )
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_triplet_positive_when_negative_too_close(self):
        """When negative is closer than positive, triplet loss > 0."""
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[5.0, 5.0]])  # far
        negative = torch.tensor([[0.1, 0.1]])  # close
        loss = cr.cpu_reference_triplet_margin_loss(
            anchor, positive, negative, margin=1.0
        )
        # d_ap=50, d_an=0.02, margin=1 → loss = max(0, 50 - 0.02 + 1) = 50.98
        assert loss > 50.0


# ════════════════════════════════════════════════════════════════════════
# Family B1: Activations — verify against F.* builtins
# ════════════════════════════════════════════════════════════════════════

class TestActivationReferences:

    def test_leaky_relu_matches_F(self):
        torch.manual_seed(42)
        x = torch.randn(8)
        ours = cr.cpu_reference_leaky_relu(x, alpha=0.01)
        builtin = F.leaky_relu(x, negative_slope=0.01)
        assert torch.allclose(ours, builtin)

    def test_gelu_exact_matches_F_gelu_none(self):
        """Exact GELU (no tanh approximation)."""
        torch.manual_seed(42)
        x = torch.randn(8)
        ours = cr.cpu_reference_gelu_exact(x)
        builtin = F.gelu(x, approximate='none')
        assert torch.allclose(ours, builtin)

    def test_selu_matches_F(self):
        torch.manual_seed(42)
        x = torch.randn(8)
        ours = cr.cpu_reference_selu(x)
        builtin = F.selu(x)
        assert torch.allclose(ours, builtin)

    def test_elu_matches_F(self):
        torch.manual_seed(42)
        x = torch.randn(8)
        ours = cr.cpu_reference_elu(x, alpha=1.0)
        builtin = F.elu(x, alpha=1.0)
        assert torch.allclose(ours, builtin)

    def test_hardsigmoid_matches_F(self):
        torch.manual_seed(42)
        x = torch.randn(8)
        ours = cr.cpu_reference_hardsigmoid(x)
        builtin = F.hardsigmoid(x)
        assert torch.allclose(ours, builtin)

    def test_softplus_matches_F(self):
        torch.manual_seed(42)
        x = torch.randn(8)
        ours = cr.cpu_reference_softplus(x)
        builtin = F.softplus(x)
        assert torch.allclose(ours, builtin)

    def test_softsign_matches_F(self):
        torch.manual_seed(42)
        x = torch.randn(8)
        ours = cr.cpu_reference_softsign(x)
        builtin = F.softsign(x)
        assert torch.allclose(ours, builtin)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
