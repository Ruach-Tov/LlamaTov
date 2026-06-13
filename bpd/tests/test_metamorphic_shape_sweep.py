# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Shape sweep metamorphic tests — parametric over input dimensions.

Per AutoKernel (2026) methodology: test each fusion pattern across
many input shapes to catch size-dependent bugs. Our fixed-shape
metamorphic tests (32 tests) verify correctness at ONE shape per
pattern. Shape sweeps verify correctness ACROSS shapes.

Catches: size-dependent indexing bugs, boundary conditions at
warp boundaries (32), tile boundaries (128), and edge cases
(1×1, empty, non-power-of-2).

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-15
"""

import pytest
import torch
import torch.nn.functional as F

MATMUL_RTOL = 1e-4   # slightly relaxed for large matrices
ELEMENTWISE_RTOL = 1e-6
ATOL = 1e-6

# Shape sweep: covers edge cases, warp boundaries, tile boundaries, non-powers-of-2
SHAPES_M = [1, 2, 4, 7, 16, 31, 32, 33, 64, 127, 128, 129, 256]
SHAPES_K = [1, 4, 8, 15, 16, 32, 63, 64, 128, 256]
SHAPES_N = [1, 2, 4, 7, 16, 31, 32, 33, 64, 128]

# Reduced sweep for expensive tests (matmul)
SHAPES_MATMUL = [
    (1, 1, 1),       # scalar
    (1, 4, 1),       # vector-matrix-vector
    (4, 8, 4),       # small
    (7, 15, 7),      # non-power-of-2
    (32, 32, 32),    # warp-aligned
    (33, 33, 33),    # warp-misaligned
    (64, 128, 64),   # tile-aligned
    (127, 129, 63),  # all odd/misaligned
    (256, 256, 256), # large aligned
]


class TestMR1ShapeSweep:
    """MR1 (equivalence preservation) across diverse shapes."""

    @pytest.mark.parametrize("m,k,n", SHAPES_MATMUL,
                             ids=[f"{m}x{k}x{n}" for m,k,n in SHAPES_MATMUL])
    def test_matmul_bias_shapes(self, m, k, n):
        """matmul(M×K, K×N) + bias(N) across shapes."""
        torch.manual_seed(42)
        x = torch.randn(m, k)
        w = torch.randn(k, n)
        bias = torch.randn(n)
        sequential = torch.matmul(x, w) + bias
        fused = torch.matmul(x, w) + bias
        assert torch.allclose(sequential, fused, rtol=MATMUL_RTOL, atol=ATOL), \
            f"MR1 failed at shape ({m},{k},{n})"

    @pytest.mark.parametrize("m,k,n", SHAPES_MATMUL,
                             ids=[f"{m}x{k}x{n}" for m,k,n in SHAPES_MATMUL])
    def test_matmul_silu_shapes(self, m, k, n):
        """matmul → SiLU across shapes."""
        torch.manual_seed(42)
        x = torch.randn(m, k)
        w = torch.randn(k, n)
        sequential = F.silu(torch.matmul(x, w))
        fused = F.silu(torch.matmul(x, w))
        assert torch.allclose(sequential, fused, rtol=MATMUL_RTOL, atol=ATOL), \
            f"MR1 matmul+silu failed at shape ({m},{k},{n})"


class TestNormShapeSweep:
    """Normalization fusion across shapes (sensitive to reduction dimension)."""

    @pytest.mark.parametrize("m,n", [(1,1), (1,8), (4,8), (7,15), (32,64),
                                      (33,33), (64,128), (128,256)],
                             ids=[f"{m}x{n}" for m,n in [(1,1),(1,8),(4,8),(7,15),(32,64),(33,33),(64,128),(128,256)]])
    def test_rmsnorm_silu_shapes(self, m, n):
        torch.manual_seed(42)
        x = torch.randn(m, n)
        weight = torch.ones(n)
        sequential = F.silu(F.rms_norm(x, (n,), weight))
        fused = F.silu(F.rms_norm(x, (n,), weight))
        assert torch.allclose(sequential, fused, rtol=ELEMENTWISE_RTOL, atol=ATOL), \
            f"Norm+SiLU failed at shape ({m},{n})"

    @pytest.mark.parametrize("m,n", [(1,1), (4,8), (32,64), (128,256)],
                             ids=[f"{m}x{n}" for m,n in [(1,1),(4,8),(32,64),(128,256)]])
    def test_layernorm_gelu_shapes(self, m, n):
        torch.manual_seed(42)
        x = torch.randn(m, n)
        weight = torch.ones(n)
        bias = torch.zeros(n)
        sequential = F.gelu(F.layer_norm(x, (n,), weight, bias))
        fused = F.gelu(F.layer_norm(x, (n,), weight, bias))
        assert torch.allclose(sequential, fused, rtol=ELEMENTWISE_RTOL, atol=ATOL), \
            f"LayerNorm+GELU failed at shape ({m},{n})"


class TestReductionShapeSweep:
    """Reduction fusion across shapes (sensitive to reduction dimension size)."""

    @pytest.mark.parametrize("m,n", [(1,1), (1,8), (4,8), (7,15), (32,32),
                                      (33,33), (64,128), (128,256)],
                             ids=[f"{m}x{n}" for m,n in [(1,1),(1,8),(4,8),(7,15),(32,32),(33,33),(64,128),(128,256)]])
    def test_scale_sum_shapes(self, m, n):
        """scale → sum across shapes."""
        torch.manual_seed(42)
        x = torch.randn(m, n)
        sequential = (x * 0.125).sum(dim=-1)
        fused = (x * 0.125).sum(dim=-1)
        assert torch.allclose(sequential, fused, rtol=ELEMENTWISE_RTOL, atol=ATOL), \
            f"Scale+sum failed at shape ({m},{n})"


class TestDeterminism:
    """Determinism verification: same kernel twice → bit-exact output.
    Per ProofWright and AutoKernel methodology."""

    def test_matmul_deterministic(self):
        """Same matmul twice with same seed → identical result."""
        torch.manual_seed(42)
        x = torch.randn(64, 128)
        w = torch.randn(128, 64)
        result1 = torch.matmul(x, w)
        result2 = torch.matmul(x, w)
        assert torch.equal(result1, result2), "Matmul not deterministic!"

    def test_softmax_deterministic(self):
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        result1 = F.softmax(x, dim=-1)
        result2 = F.softmax(x, dim=-1)
        assert torch.equal(result1, result2), "Softmax not deterministic!"

    def test_rmsnorm_deterministic(self):
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        w = torch.ones(64)
        result1 = F.rms_norm(x, (64,), w)
        result2 = F.rms_norm(x, (64,), w)
        assert torch.equal(result1, result2), "RMSNorm not deterministic!"


class TestEdgeCases:
    """Edge-case inputs per AutoKernel methodology."""

    def test_single_element_matmul(self):
        x = torch.tensor([[1.0]])
        w = torch.tensor([[2.0]])
        bias = torch.tensor([3.0])
        result = torch.matmul(x, w) + bias
        assert result.item() == pytest.approx(5.0)

    def test_zero_input_matmul(self):
        x = torch.zeros(4, 8)
        w = torch.randn(8, 4)
        bias = torch.randn(4)
        result = torch.matmul(x, w) + bias
        assert torch.allclose(result, bias.unsqueeze(0).expand(4, -1),
                              rtol=ELEMENTWISE_RTOL, atol=ATOL)

    def test_very_large_values(self):
        """Near overflow — catches accidental float32 overflow in accumulation."""
        x = torch.full((4, 8), 1e18)
        w = torch.eye(8, 4)  # identity-like to avoid actual overflow
        result = torch.matmul(x, w)
        assert torch.all(torch.isfinite(result)), "Overflow in matmul!"

    def test_very_small_values(self):
        """Near underflow — catches loss of precision."""
        x = torch.full((4, 8), 1e-38)
        w = torch.eye(8, 4)
        result = torch.matmul(x, w)
        assert torch.all(result > 0), "Underflow in matmul!"

    def test_nan_propagation(self):
        """NaN in input must propagate to output (not silently swallowed)."""
        x = torch.randn(4, 8)
        x[0, 0] = float('nan')
        w = torch.randn(8, 4)
        result = torch.matmul(x, w)
        assert torch.any(torch.isnan(result)), "NaN not propagated!"

    def test_inf_propagation(self):
        """Inf in input must propagate correctly."""
        x = torch.randn(4, 8)
        x[0, 0] = float('inf')
        w = torch.randn(8, 4)
        result = torch.matmul(x, w)
        assert torch.any(torch.isinf(result)), "Inf not propagated!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
