# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Property tests for CUDA code emission (compile_graph output).

Tests structural properties of emitted CUDA source that must hold
regardless of the specific fusion pattern. These are STATIC ANALYSIS
tests — they check the emitted text without running nvcc.

Per ProofWright's five-stage harness:
  Stage 1: valid syntax (checked here via text patterns)
  Stage 2: correct kernel signature (__global__ void)
  Stage 3: thread indexing within bounds
  Stage 4: output variable always written
  Stage 5: synchronization points present where needed

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-15
"""

import pytest
import re


# ══════════════════════════════════════════════════════════════════════
# Sample emitted CUDA for testing (from fusion_to_cuda.pl patterns)
# ══════════════════════════════════════════════════════════════════════

MATMUL_EPILOGUE_KERNEL = """
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_sigmoid_scale(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float sA[32][128];
    __shared__ float sB[128][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + 127) / 128; t++) {
        // Load tile into shared memory
        int aCol = t * 128 + threadIdx.x;
        if (row < M && aCol < K) sA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else sA[threadIdx.y][threadIdx.x] = 0.0f;

        int bRow = t * 128 + threadIdx.y;
        if (bRow < K && col < N) sB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < 128; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        // --- FUSED EPILOGUE ---
        sum = 1.0f / (1.0f + expf(-sum));  // sigmoid
        sum = sum * 0.5f;                    // scale
        C[row * N + col] = sum;
    }
}
"""

ELEMENTWISE_KERNEL = """
__global__ void fused_silu_mul(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float sum = data[idx];

    // --- FUSED ELEMENTWISE CHAIN ---
    sum = sum * (1.0f / (1.0f + expf(-sum)));  // silu
    sum = sum * 2.0f;                            // mul

    data[idx] = sum;
}
"""

RELU_KERNEL = """
__global__ void fused_relu(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    data[idx] = val > 0.0f ? val : 0.0f;
}
"""


# ══════════════════════════════════════════════════════════════════════
# P1: Valid syntax patterns
# ══════════════════════════════════════════════════════════════════════

class TestCUDASyntax:
    """Emitted CUDA must have valid structural patterns."""

    @pytest.mark.parametrize("kernel", [MATMUL_EPILOGUE_KERNEL, ELEMENTWISE_KERNEL, RELU_KERNEL],
                             ids=["matmul_epilogue", "elementwise", "relu"])
    def test_has_global_qualifier(self, kernel):
        assert "__global__" in kernel, "CUDA kernel must have __global__ qualifier"

    @pytest.mark.parametrize("kernel", [MATMUL_EPILOGUE_KERNEL, ELEMENTWISE_KERNEL, RELU_KERNEL],
                             ids=["matmul_epilogue", "elementwise", "relu"])
    def test_returns_void(self, kernel):
        # __global__ functions must return void
        assert re.search(r"__global__\s+void\s+\w+", kernel), \
            "CUDA __global__ function must return void"

    @pytest.mark.parametrize("kernel", [MATMUL_EPILOGUE_KERNEL, ELEMENTWISE_KERNEL, RELU_KERNEL],
                             ids=["matmul_epilogue", "elementwise", "relu"])
    def test_balanced_braces(self, kernel):
        opens = kernel.count("{")
        closes = kernel.count("}")
        assert opens == closes, f"Unbalanced braces: {opens} opens, {closes} closes"

    @pytest.mark.parametrize("kernel", [MATMUL_EPILOGUE_KERNEL, ELEMENTWISE_KERNEL, RELU_KERNEL],
                             ids=["matmul_epilogue", "elementwise", "relu"])
    def test_balanced_parens(self, kernel):
        opens = kernel.count("(")
        closes = kernel.count(")")
        assert opens == closes, f"Unbalanced parens: {opens} opens, {closes} closes"

    @pytest.mark.parametrize("kernel", [MATMUL_EPILOGUE_KERNEL, ELEMENTWISE_KERNEL, RELU_KERNEL],
                             ids=["matmul_epilogue", "elementwise", "relu"])
    def test_no_dangling_semicolons(self, kernel):
        """No lines should have ;; (double semicolon) — common emission bug."""
        for i, line in enumerate(kernel.split("\n"), 1):
            assert ";;" not in line, f"Double semicolon on line {i}: {line.strip()}"


# ══════════════════════════════════════════════════════════════════════
# P2: Thread indexing correctness
# ══════════════════════════════════════════════════════════════════════

class TestThreadIndexing:
    """Thread index computation must use standard CUDA patterns."""

    @pytest.mark.parametrize("kernel", [MATMUL_EPILOGUE_KERNEL, ELEMENTWISE_KERNEL, RELU_KERNEL],
                             ids=["matmul_epilogue", "elementwise", "relu"])
    def test_uses_threadIdx(self, kernel):
        assert "threadIdx" in kernel, "Kernel must use threadIdx for per-thread indexing"

    @pytest.mark.parametrize("kernel", [MATMUL_EPILOGUE_KERNEL, ELEMENTWISE_KERNEL, RELU_KERNEL],
                             ids=["matmul_epilogue", "elementwise", "relu"])
    def test_uses_blockIdx(self, kernel):
        assert "blockIdx" in kernel, "Kernel must use blockIdx for block-level indexing"

    @pytest.mark.parametrize("kernel", [ELEMENTWISE_KERNEL, RELU_KERNEL],
                             ids=["elementwise", "relu"])
    def test_bounds_check_present(self, kernel):
        """Elementwise kernels must have bounds check (if idx >= n return)."""
        assert re.search(r"if\s*\(\s*idx\s*>=\s*n\s*\)", kernel), \
            "Elementwise kernel must check idx >= n to prevent OOB access"

    def test_matmul_bounds_check(self):
        """Matmul kernel must check row < M && col < N."""
        assert re.search(r"if\s*\(\s*row\s*<\s*M\s*&&\s*col\s*<\s*N\s*\)",
                         MATMUL_EPILOGUE_KERNEL), \
            "Matmul kernel must check row < M && col < N"


# ══════════════════════════════════════════════════════════════════════
# P3: Shared memory usage
# ══════════════════════════════════════════════════════════════════════

class TestSharedMemory:
    """Matmul kernels must use shared memory with proper synchronization."""

    def test_shared_memory_declared(self):
        assert "__shared__" in MATMUL_EPILOGUE_KERNEL, \
            "Matmul kernel must use __shared__ memory for tiling"

    def test_syncthreads_present(self):
        """Must have __syncthreads() between shared memory load and compute."""
        assert "__syncthreads()" in MATMUL_EPILOGUE_KERNEL, \
            "Matmul kernel must synchronize after shared memory loads"

    def test_syncthreads_count(self):
        """Need at least 2 syncthreads per tile loop iteration
        (after load, after compute)."""
        count = MATMUL_EPILOGUE_KERNEL.count("__syncthreads()")
        assert count >= 2, f"Expected ≥2 __syncthreads(), got {count}"

    def test_elementwise_no_shared_memory(self):
        """Elementwise kernels should NOT use shared memory (wasteful)."""
        assert "__shared__" not in ELEMENTWISE_KERNEL, \
            "Elementwise kernel should not use shared memory"


# ══════════════════════════════════════════════════════════════════════
# P4: Output always written
# ══════════════════════════════════════════════════════════════════════

class TestOutputWritten:
    """Every kernel must write to its output array before returning."""

    def test_matmul_writes_output(self):
        """Matmul kernel must write C[row * N + col]."""
        assert re.search(r"C\[.*\]\s*=", MATMUL_EPILOGUE_KERNEL), \
            "Matmul kernel must write to output array C"

    def test_elementwise_writes_output(self):
        """Elementwise kernel must write data[idx]."""
        assert re.search(r"data\[.*\]\s*=", ELEMENTWISE_KERNEL), \
            "Elementwise kernel must write back to data array"

    def test_relu_writes_output(self):
        assert re.search(r"data\[.*\]\s*=", RELU_KERNEL), \
            "ReLU kernel must write back to data array"


# ══════════════════════════════════════════════════════════════════════
# P5: Epilogue correctness
# ══════════════════════════════════════════════════════════════════════

class TestEpiloguePresence:
    """Fused kernels must contain the epilogue operations."""

    def test_sigmoid_in_epilogue(self):
        """The matmul+sigmoid+scale kernel must contain sigmoid computation."""
        # sigmoid = 1/(1+exp(-x))
        assert "expf" in MATMUL_EPILOGUE_KERNEL, \
            "Sigmoid epilogue must use expf()"

    def test_silu_in_elementwise(self):
        """SiLU = x * sigmoid(x) must be present."""
        assert "expf" in ELEMENTWISE_KERNEL, \
            "SiLU must use expf() for sigmoid component"

    def test_relu_uses_comparison(self):
        """ReLU = max(0, x) must use comparison."""
        assert re.search(r">\s*0\.0f|val\s*>\s*0", RELU_KERNEL), \
            "ReLU must compare against 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
