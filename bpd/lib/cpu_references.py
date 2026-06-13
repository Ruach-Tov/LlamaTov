# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""cpu_references.py — Canonical CPU PyTorch references for B2 kernel templates.

Per mavchin's directive (inbox 19:30:18): "Scope C-Extended (semantic
correctness vs PyTorch reference). This is what closes the 'do our emitted
kernels produce CORRECT output' question."

Each function here is the GROUND TRUTH for what its corresponding kernel
template in bpd/lib/kernel_templates.pl SHOULD compute. When mavchin's
ctypes bridge eventually compiles and executes an emitted kernel on GPU,
its output should match `cpu_reference_<family>(*args)` within numerical
tolerance.

Per medayek's robust-kbench finding: this is the foundation for catching
"exploitable loopholes" — by comparing against an independent
implementation (PyTorch), we catch any bug where our kernel template
emits CUDA that compiles + has the right shape + computes the WRONG math.

Design principles:
1. Each reference uses PyTorch operations only (no custom CUDA)
2. Runs on CPU (no GPU dependency for reference computation)
3. Single canonical implementation per ggml op kind
4. Matches the kernel template's mathematical operation exactly
5. Tolerance for numerical comparison: rtol=1e-5, atol=1e-6 (standard fp32)

Author: metayen 2026-05-15 ~21:35 UTC
Per mavchin's Scope C-Extended priority directive.
"""

import torch
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════
# Family 1: Reductions
# ════════════════════════════════════════════════════════════════════════
#
# Kernel template: bpd/lib/kernel_templates.pl::generate_kernel_reduction/4
# Input shape:  [outer, N] (or any shape flattened to outer × N)
# Output shape: [outer]   (single value per outer "row")
# Axis: inner dim (axis_inner mode in the template)


def cpu_reference_sum_rows(x: torch.Tensor) -> torch.Tensor:
    """ggml_sum_rows: sum along the innermost dim.

    Matches Y[o] = sum_i X[o*N + i]
    """
    return x.sum(dim=-1)


def cpu_reference_mean(x: torch.Tensor) -> torch.Tensor:
    """ggml_mean: mean along the innermost dim.

    Matches Y[o] = (sum_i X[o*N + i]) / N
    """
    return x.mean(dim=-1)


def cpu_reference_max(x: torch.Tensor) -> torch.Tensor:
    """ggml_max: max along innermost dim. Returns float-cast values."""
    return x.max(dim=-1).values


def cpu_reference_min(x: torch.Tensor) -> torch.Tensor:
    """ggml_min: min along innermost dim."""
    return x.min(dim=-1).values


def cpu_reference_argmax(x: torch.Tensor) -> torch.Tensor:
    """ggml_argmax: index of max along innermost dim, cast to float.

    Matches Y[o] = (float)arg, where arg is the position of the max.
    """
    return x.argmax(dim=-1).float()


def cpu_reference_argmin(x: torch.Tensor) -> torch.Tensor:
    """ggml_argmin: index of min, cast to float."""
    return x.argmin(dim=-1).float()


def cpu_reference_cumsum(x: torch.Tensor) -> torch.Tensor:
    """ggml_cumsum: cumulative sum along innermost dim. Output same shape as input."""
    return x.cumsum(dim=-1)


def cpu_reference_cumprod(x: torch.Tensor) -> torch.Tensor:
    """ggml_cumprod: cumulative product along innermost dim."""
    return x.cumprod(dim=-1)


# ════════════════════════════════════════════════════════════════════════
# Family 2: Normalizations
# ════════════════════════════════════════════════════════════════════════
#
# Kernel template: bpd/lib/kernel_templates.pl::generate_kernel_norm/4
# Input shape:  [outer, N]
# Output shape: [outer, N] (same as input)


def cpu_reference_layer_norm(x: torch.Tensor, eps: float = 1e-5,
                              weight: torch.Tensor = None,
                              bias: torch.Tensor = None) -> torch.Tensor:
    """ggml_norm (LayerNorm): normalize over inner dim.

    Matches:
      mean = X.mean(dim=-1)
      var  = X.var(dim=-1, unbiased=False)
      Y    = (X - mean) / sqrt(var + eps) * weight + bias

    The template currently uses sum-then-mean which equals torch.mean,
    and sumsq/N - mean² which equals torch.var(unbiased=False).
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    inv_std = (var + eps).rsqrt()
    y = (x - mean) * inv_std
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def cpu_reference_rms_norm(x: torch.Tensor, eps: float = 1e-5,
                            weight: torch.Tensor = None) -> torch.Tensor:
    """ggml_rms_norm: RMS normalize over inner dim.

    Matches:
      inv_rms = 1 / sqrt(mean(X²) + eps)
      Y = X * inv_rms * weight

    Per the kernel template: inv_rms = 1/sqrt(sumsq/N + eps).
    """
    sumsq = (x * x).mean(dim=-1, keepdim=True)
    inv_rms = (sumsq + eps).rsqrt()
    y = x * inv_rms
    if weight is not None:
        y = y * weight
    return y


def cpu_reference_l2_norm(x: torch.Tensor, eps: float = 1e-5,
                           weight: torch.Tensor = None) -> torch.Tensor:
    """ggml_l2_norm: L2 normalize over inner dim.

    Matches:
      inv_norm = 1 / sqrt(sum(X²) + eps)   (NOT mean — full sum)
      Y = X * inv_norm * weight

    Per the kernel template: inv_norm = 1/sqrt(sumsq + eps), where
    sumsq is the SUM of squares (not mean).
    """
    sumsq = (x * x).sum(dim=-1, keepdim=True)
    inv_norm = (sumsq + eps).rsqrt()
    y = x * inv_norm
    if weight is not None:
        y = y * weight
    return y


def cpu_reference_group_norm(x: torch.Tensor, eps: float = 1e-5,
                              weight: torch.Tensor = None,
                              bias: torch.Tensor = None) -> torch.Tensor:
    """ggml_group_norm: same as layer norm in current template (single group).

    The template's group_norm clause delegates to layer norm. When
    B2-Norm-Extended adds proper per-group statistics, this reference
    will need updating.
    """
    return cpu_reference_layer_norm(x, eps, weight, bias)


# ════════════════════════════════════════════════════════════════════════
# Family 3: Pooling (2D max/avg)
# ════════════════════════════════════════════════════════════════════════
#
# Kernel template: bpd/lib/kernel_templates.pl::generate_kernel_pool/5
# Input shape:  [B, C, inH, inW]
# Output shape: [B, C, outH, outW]


def cpu_reference_pool_2d_max(x: torch.Tensor, kH: int, kW: int,
                               stride_h: int = 1, stride_w: int = 1,
                               pad_h: int = 0, pad_w: int = 0) -> torch.Tensor:
    """ggml_pool_2d max: 2D max pooling. Matches torch.nn.functional.max_pool2d."""
    return F.max_pool2d(x, kernel_size=(kH, kW),
                        stride=(stride_h, stride_w),
                        padding=(pad_h, pad_w))


def cpu_reference_pool_2d_avg(x: torch.Tensor, kH: int, kW: int,
                               stride_h: int = 1, stride_w: int = 1,
                               pad_h: int = 0, pad_w: int = 0) -> torch.Tensor:
    """ggml_pool_2d avg: 2D average pooling.

    Note: The kernel template divides by `count` (the number of in-bounds
    elements per window) — this matches `count_include_pad=False`. PyTorch
    default is `count_include_pad=True`. Specify explicitly to match.
    """
    return F.avg_pool2d(x, kernel_size=(kH, kW),
                        stride=(stride_h, stride_w),
                        padding=(pad_h, pad_w),
                        count_include_pad=False)


# ════════════════════════════════════════════════════════════════════════
# Family 4: Convolutions (via im2col + matmul)
# ════════════════════════════════════════════════════════════════════════
#
# Kernel template: bpd/lib/kernel_templates.pl::generate_kernel_im2col/4
# The im2col kernel only does the UNFOLD; the matmul stage is mavchin's
# existing tiled matmul kernel. Reference here is for the im2col output.


def cpu_reference_im2col_2d(x: torch.Tensor, kH: int, kW: int,
                             stride_h: int = 1, stride_w: int = 1,
                             pad_h: int = 0, pad_w: int = 0,
                             dilation_h: int = 1, dilation_w: int = 1) -> torch.Tensor:
    """im2col 2D: unfold patches into [B*outH*outW, C*kH*kW].

    Matches torch.nn.functional.unfold but reshaped to the kernel template's
    output layout: row-major over (b, oh, ow), with columns over (c, kh, kw).
    """
    B, C, H, W = x.shape
    # F.unfold produces [B, C*kH*kW, outH*outW] in (kH, kW) layout per channel
    unfolded = F.unfold(x, kernel_size=(kH, kW),
                        stride=(stride_h, stride_w),
                        padding=(pad_h, pad_w),
                        dilation=(dilation_h, dilation_w))
    # outH, outW from F.unfold:
    outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1
    # Reshape to [B, C*kH*kW, outH*outW] → [B, outH*outW, C*kH*kW]
    # Then flatten B and outH*outW into a single row dim.
    out = unfolded.transpose(1, 2).reshape(B * outH * outW, C * kH * kW)
    return out


def cpu_reference_conv_2d(x: torch.Tensor, weight: torch.Tensor,
                           stride_h: int = 1, stride_w: int = 1,
                           pad_h: int = 0, pad_w: int = 0,
                           dilation_h: int = 1, dilation_w: int = 1) -> torch.Tensor:
    """Full 2D conv = im2col + matmul + reshape.

    Matches F.conv2d. This is the FULL kernel-template-pipeline output
    (when im2col + mavchin's matmul are composed).
    """
    return F.conv2d(x, weight,
                    stride=(stride_h, stride_w),
                    padding=(pad_h, pad_w),
                    dilation=(dilation_h, dilation_w))


# ════════════════════════════════════════════════════════════════════════
# Family 5: Losses
# ════════════════════════════════════════════════════════════════════════
#
# Kernel template: bpd/lib/kernel_templates.pl::generate_kernel_loss/4


def cpu_reference_mse_loss(x: torch.Tensor, y: torch.Tensor,
                            reduction: str = 'mean') -> torch.Tensor:
    """ggml_mse_loss: (x - y)² element-wise, then mean or sum."""
    return F.mse_loss(x, y, reduction=reduction)


def cpu_reference_cross_entropy_loss(x: torch.Tensor, y: torch.Tensor,
                                      reduction: str = 'mean') -> torch.Tensor:
    """ggml_cross_entropy_loss: -y * log(x + 1e-12), then mean or sum.

    The kernel template uses x as softmax-output probabilities AND y as
    one-hot or soft-label targets. This is the soft-label CE form, not
    F.cross_entropy (which expects logits + class indices).
    """
    eps = 1e-12
    loss_elements = -y * torch.log(x + eps)
    if reduction == 'mean':
        return loss_elements.mean()
    elif reduction == 'sum':
        return loss_elements.sum()
    else:
        return loss_elements


def cpu_reference_huber_loss(x: torch.Tensor, y: torch.Tensor,
                              delta: float = 1.0,
                              reduction: str = 'mean') -> torch.Tensor:
    """ggml_huber_loss: |x-y|<δ ? 0.5*(x-y)² : δ*(|x-y| - 0.5*δ)."""
    diff = x - y
    abs_diff = diff.abs()
    quadratic = 0.5 * diff * diff
    linear = delta * (abs_diff - 0.5 * delta)
    loss_elements = torch.where(abs_diff < delta, quadratic, linear)
    if reduction == 'mean':
        return loss_elements.mean()
    elif reduction == 'sum':
        return loss_elements.sum()
    else:
        return loss_elements


def cpu_reference_kl_div_loss(x: torch.Tensor, y: torch.Tensor,
                               reduction: str = 'sum') -> torch.Tensor:
    """ggml_kl_div_loss: y * (log(y+ε) - log(x+ε)).

    The kernel template assumes x and y are probability distributions.
    """
    eps = 1e-12
    loss_elements = y * (torch.log(y + eps) - torch.log(x + eps))
    if reduction == 'mean':
        return loss_elements.mean()
    elif reduction == 'sum':
        return loss_elements.sum()
    else:
        return loss_elements


def cpu_reference_hinge_loss(x: torch.Tensor, y: torch.Tensor,
                              reduction: str = 'mean') -> torch.Tensor:
    """ggml_hinge_loss: max(0, 1 - y * x).

    Assumes y ∈ {-1, +1}.
    """
    loss_elements = torch.clamp(1.0 - y * x, min=0.0)
    if reduction == 'mean':
        return loss_elements.mean()
    elif reduction == 'sum':
        return loss_elements.sum()
    else:
        return loss_elements


def cpu_reference_triplet_margin_loss(anchor: torch.Tensor,
                                       positive: torch.Tensor,
                                       negative: torch.Tensor,
                                       margin: float = 1.0,
                                       reduction: str = 'mean') -> torch.Tensor:
    """ggml_triplet_margin_loss: max(0, ||a-p||² - ||a-n||² + margin).

    Per-batch reduction over the inner dim, then mean/sum over batch.
    The kernel template uses SQUARED distance (not the standard L2 norm
    distance used by torch.nn.TripletMarginLoss).
    """
    d_ap = ((anchor - positive) ** 2).sum(dim=-1)
    d_an = ((anchor - negative) ** 2).sum(dim=-1)
    loss_per_batch = torch.clamp(d_ap - d_an + margin, min=0.0)
    if reduction == 'mean':
        return loss_per_batch.mean()
    elif reduction == 'sum':
        return loss_per_batch.sum()
    else:
        return loss_per_batch


# ════════════════════════════════════════════════════════════════════════
# Family B1: Elementwise epilogue activations (the simple ones)
# ════════════════════════════════════════════════════════════════════════
#
# These are fused INTO matmul kernels; the CPU reference here computes
# just the activation, applied after a matmul (which is the standard
# F.linear or @ operation).


# ─────────────────────────────────────────────────────────────────
# CANONICAL ACTIVATION FAMILY (matches activation_expr/3 facts in
# lib/kernel_templates_llama.pl, BPD-emitted via unary_activation_kernel/2).
# Each function IS cell [3] (Python host, CPU dispatch) for the matrix
# harness's elementwise column.
# ─────────────────────────────────────────────────────────────────

def cpu_reference_silu(x: torch.Tensor) -> torch.Tensor:
    """k_silu: x / (1 + exp(-x)).

    Matches the BPD kernel's per-element expression:
      activation_expr(k_silu, X, c_binop(/, X, c_paren(...))).
    Equivalent to F.silu(x) = x * sigmoid(x), but the kernel writes
    the division form. PyTorch's F.silu may compute it differently
    internally; the matrix-verify --strict contract will tell us
    whether cell [2] (C-host GPU) matches cell [3] bit-for-bit.
    """
    return F.silu(x)


def cpu_reference_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """k_sigmoid: 1 / (1 + exp(-x)).

    Matches the BPD kernel's per-element expression:
      activation_expr(k_sigmoid, X, c_binop(/, c_float(1.0), c_paren(...))).
    """
    return torch.sigmoid(x)


def cpu_reference_relu(x: torch.Tensor) -> torch.Tensor:
    """k_relu: fmaxf(0, x).

    Matches the BPD kernel's per-element expression:
      activation_expr(k_relu, X, c_call(fmaxf, [c_float(0.0), X])).
    Pure conditional, no transcendental — expected bit-identical
    between cell [2] (C-host GPU) and cell [3] (Python CPU) under
    --strict-maxxing.
    """
    return F.relu(x)


def cpu_reference_tanh(x: torch.Tensor) -> torch.Tensor:
    """k_tanh: tanhf(x).

    Matches the BPD kernel's per-element expression:
      activation_expr(k_tanh, X, c_call(tanhf, [X])).
    """
    return torch.tanh(x)


def cpu_reference_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """k_gelu_tanh: 0.5x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715*x^2))).

    Matches BPD activation_expr(k_gelu_tanh, ...) and ggml_gelu_f32
    in external/llama.cpp/ggml/src/ggml-cpu/vec.h. This is the form
    llama.cpp uses at all three LLM_FFN_GELU dispatch sites — what
    Ollama actually runs at inference time.

    Empirically bit-identical to F.gelu(approximate='tanh').
    Per the 2026-05-17 gelu investigation: PyTorch's tanh-form output
    IS our cell-[3] reference for this canonical variant.
    """
    return F.gelu(x, approximate='tanh')


def cpu_reference_gelu_erf(x: torch.Tensor) -> torch.Tensor:
    """k_gelu_erf: 0.5x * (1 + erf(x / sqrt(2))).

    Matches BPD activation_expr(k_gelu_erf, ...) and ggml_gelu_erf_f32
    in external/llama.cpp/ggml/src/ggml-cpu/vec.h. llama.cpp does NOT
    use this form in its model dispatch; it's exposed as an alternative
    for use cases that need the exact erf form.

    F.gelu(approximate='none') is the same mathematical function but
    PyTorch's implementation may differ in precision from the BPD-
    emitted erff path (CUDA libcudart's erff has up to ≤2 ULP per its
    spec, while PyTorch CPU may use higher-precision then narrow).
    Empirically diverges by up to thousands of ULPs at near-zero
    outputs — see docs/methodology/matrix-status.md for full picture.
    """
    return F.gelu(x, approximate='none')


def cpu_reference_gelu(x: torch.Tensor) -> torch.Tensor:
    """DEPRECATED alias for cpu_reference_gelu_erf.

    Kept for one-cycle backward compatibility. New callers should use
    cpu_reference_gelu_tanh (for matching ggml/llama.cpp/Ollama) or
    cpu_reference_gelu_erf (for matching pytorch_default / HF "gelu").

    The single-name "gelu" carries a hidden assumption about which
    mathematical form is meant. Use the disambiguated names.
    """
    return cpu_reference_gelu_erf(x)


def cpu_reference_leaky_relu(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """ggml_leaky_relu: x > 0 ? x : alpha * x."""
    return F.leaky_relu(x, negative_slope=alpha)


def cpu_reference_gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """ggml_gelu (exact form, using erff): 0.5 * x * (1 + erf(x / sqrt(2))).

    The kernel template uses the EXACT GELU (via erff), not the tanh
    approximation. Matches F.gelu(x, approximate='none').

    NOTE: cpu_reference_gelu is the new canonical name (added for
    naming symmetry across the activation family). This function is
    retained for backward compatibility but cpu_reference_gelu is the
    preferred reference for new callers.
    """
    return F.gelu(x, approximate='none')


def cpu_reference_selu(x: torch.Tensor) -> torch.Tensor:
    """ggml_selu: scale * (x > 0 ? x : alpha * (exp(x) - 1)).

    scale = 1.0507009873554804934
    alpha = 1.6732632423543772848
    """
    return F.selu(x)


def cpu_reference_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ggml_elu: x > 0 ? x : alpha * (exp(x) - 1)."""
    return F.elu(x, alpha=alpha)


def cpu_reference_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """ggml_hardsigmoid: max(0, min(1, x/6 + 0.5)). Matches F.hardsigmoid."""
    return F.hardsigmoid(x)


def cpu_reference_softplus(x: torch.Tensor) -> torch.Tensor:
    """ggml_softplus: log(1 + exp(x)). Matches F.softplus."""
    return F.softplus(x)


def cpu_reference_softsign(x: torch.Tensor) -> torch.Tensor:
    """ggml_softsign: x / (1 + |x|). Matches F.softsign."""
    return F.softsign(x)
