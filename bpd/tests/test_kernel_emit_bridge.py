# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_kernel_emit_bridge.py — verify kernel_emit_bridge.py.

Per mavchin's directive (inbox 19:42:46): option (b) — write the swipl
subprocess bridge as a separate commit; they run it on P4.

This test verifies the bridge produces correct emissions for every
family. The CUDA source is checked structurally; actual GPU execution
is Scope C-Extended-B (mavchin's territory).

Author: metayen 2026-05-15
"""

import sys
import os
import shutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import kernel_emit_bridge as keb


# ════════════════════════════════════════════════════════════════════════
# Skip all tests if swipl isn't available
# ════════════════════════════════════════════════════════════════════════

SWIPL_REQUIRED = pytest.mark.skipif(
    shutil.which('swipl') is None,
    reason="kernel_emit_bridge requires swipl (SWI-Prolog) in PATH"
)


# ════════════════════════════════════════════════════════════════════════
# Family 1: Reductions
# ════════════════════════════════════════════════════════════════════════

@SWIPL_REQUIRED
class TestReductionBridge:

    def test_emit_sum_rows(self):
        src = keb.emit_kernel_reduction('ggml_sum_rows')
        assert '#include <cuda_runtime.h>' in src
        assert '__global__ void reduce_sum(' in src
        assert 'acc + v' in src   # accumulator updates correctly

    def test_emit_mean(self):
        src = keb.emit_kernel_reduction('ggml_mean')
        assert '__global__ void reduce_mean(' in src
        # divides by N — accept either (float)N or (float)(N) form,
        # since the c_ast emit can produce either depending on cast
        # operand parenthesization conventions
        assert '(float)N' in src or '(float)(N)' in src

    def test_emit_argmax_returns_float(self):
        src = keb.emit_kernel_reduction('ggml_argmax')
        assert '__global__ void reduce_argmax(' in src
        # casts argmax to float — accept either form
        assert '(float)arg' in src or '(float)(arg)' in src

    def test_emit_cumsum_writes_inline(self):
        src = keb.emit_kernel_reduction('ggml_cumsum')
        assert '__global__ void cumsum(' in src
        # Inline write inside the loop, not after
        assert 'Y[idx] = acc' in src


# ════════════════════════════════════════════════════════════════════════
# Family 2: Normalizations
# ════════════════════════════════════════════════════════════════════════

@SWIPL_REQUIRED
class TestNormBridge:

    def test_emit_rms_norm_plain(self):
        src = keb.emit_kernel_norm('ggml_rms_norm', affine=False)
        assert '__global__ void norm_rms(' in src
        assert 'inv_rms' in src
        assert 'sqrtf' in src
        assert 'W[i]' not in src   # no affine

    def test_emit_rms_norm_affine(self):
        src = keb.emit_kernel_norm('ggml_rms_norm', affine=True)
        assert '__global__ void norm_rms(' in src
        assert 'W[i]' in src   # affine W applied

    def test_emit_layer_norm_uses_mean(self):
        src = keb.emit_kernel_norm('ggml_norm', affine=False)
        assert '__global__ void norm_layer(' in src
        assert 'mean' in src
        assert 'inv_std' in src

    def test_emit_l2_norm_uses_inv_norm(self):
        src = keb.emit_kernel_norm('ggml_l2_norm', affine=False)
        assert '__global__ void norm_l2(' in src
        assert 'inv_norm' in src


# ════════════════════════════════════════════════════════════════════════
# Family 3: Pooling
# ════════════════════════════════════════════════════════════════════════

@SWIPL_REQUIRED
class TestPoolBridge:

    def test_emit_pool_2d_max(self):
        src = keb.emit_kernel_pool('ggml_pool_2d', 2, 'max')
        assert '__global__ void pool_2d_max(' in src
        # max uses ternary
        assert 'v > acc' in src

    def test_emit_pool_2d_avg(self):
        src = keb.emit_kernel_pool('ggml_pool_2d', 2, 'avg')
        assert '__global__ void pool_2d_avg(' in src
        # avg divides by count — accept either (float)count or (float)(count)
        assert '(float)count' in src or '(float)(count)' in src


# ════════════════════════════════════════════════════════════════════════
# Family 4: Convolutions
# ════════════════════════════════════════════════════════════════════════

@SWIPL_REQUIRED
class TestConvBridge:

    def test_emit_im2col_2d_forward(self):
        src = keb.emit_kernel_im2col('ggml_conv_2d', 2, 'forward')
        assert '__global__ void im2col_2d_forward(' in src
        # 2D im2col has the triple-nested loop over (c, kh, kw)
        assert 'kH' in src
        assert 'kW' in src
        # Proper row-major indexing
        assert '((b * C + c) * H + ih) * W + iw' in src


# ════════════════════════════════════════════════════════════════════════
# Family 5: Losses
# ════════════════════════════════════════════════════════════════════════

@SWIPL_REQUIRED
class TestLossBridge:

    def test_emit_mse_loss_mean(self):
        src = keb.emit_kernel_loss('ggml_mse_loss', 'mean')
        assert '__global__ void loss_mse(' in src
        assert '(x_i - y_i)' in src   # squared diff

    def test_emit_huber_loss(self):
        src = keb.emit_kernel_loss('ggml_huber_loss', 'mean')
        assert '__global__ void loss_huber(' in src
        assert 'fabsf' in src   # uses |x-y|

    def test_emit_triplet_loss(self):
        src = keb.emit_kernel_loss('ggml_triplet_margin_loss', 'mean')
        assert '__global__ void loss_triplet_margin(' in src
        assert 'anchor' in src
        assert 'positive' in src
        assert 'negative' in src


# ════════════════════════════════════════════════════════════════════════
# Bundle: emit_all_l1_kernels
# ════════════════════════════════════════════════════════════════════════

@SWIPL_REQUIRED
class TestEmitAllKernels:

    def test_bundle_contains_all_families(self):
        bundle = keb.emit_all_l1_kernels()
        # Reductions
        assert 'reduce_sum' in bundle
        assert 'reduce_argmax' in bundle
        assert 'cumsum' in bundle
        # Norms (both affine modes)
        assert 'norm_rms_plain' in bundle
        assert 'norm_rms_affine' in bundle
        # Pool
        assert 'pool_2d_max' in bundle
        assert 'pool_2d_avg' in bundle
        # Conv
        assert 'im2col_2d_forward' in bundle
        # Loss
        assert 'loss_mse' in bundle
        assert 'loss_triplet_margin' in bundle

    def test_bundle_size_is_25_kernels(self):
        bundle = keb.emit_all_l1_kernels()
        assert len(bundle) == 25

    def test_every_kernel_starts_with_cuda_include(self):
        bundle = keb.emit_all_l1_kernels()
        for name, src in bundle.items():
            assert src.startswith('#include <cuda_runtime.h>'), \
                f"Kernel {name} doesn't start with cuda_runtime.h include"

    def test_every_kernel_has_global_void(self):
        bundle = keb.emit_all_l1_kernels()
        for name, src in bundle.items():
            assert '__global__ void' in src, \
                f"Kernel {name} missing __global__ void"


# ════════════════════════════════════════════════════════════════════════
# Name extraction
# ════════════════════════════════════════════════════════════════════════

class TestKernelNameExtraction:
    """These tests don't require swipl; they test the regex helper."""

    def test_extracts_simple_name(self):
        src = '#include <cuda_runtime.h>\n\n__global__ void my_kernel(int x) { }'
        assert keb.extract_kernel_name(src) == 'my_kernel'

    def test_returns_none_when_no_kernel(self):
        src = '#include <cuda_runtime.h>\n// no kernel here'
        assert keb.extract_kernel_name(src) is None

    def test_extracts_kernel_with_complex_signature(self):
        src = ('__global__ void norm_rms(const float * X, float * Y, '
               'int N, int outer, float eps, const float * W) { }')
        assert keb.extract_kernel_name(src) == 'norm_rms'


# ════════════════════════════════════════════════════════════════════════
# Error handling
# ════════════════════════════════════════════════════════════════════════

@SWIPL_REQUIRED
class TestErrorHandling:

    def test_invalid_op_kind_raises_error(self):
        """Invalid op_kind should cause swipl to fail; bridge should raise."""
        with pytest.raises(keb.KernelEmitError):
            keb.emit_kernel_reduction('ggml_not_a_real_op')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
