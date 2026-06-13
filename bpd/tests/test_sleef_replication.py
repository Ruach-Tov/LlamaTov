#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Replicate PyTorch's exact erf polynomial for bit-identical GELU.

PyTorch AVX2 vec256_float.h erf() uses Abramowitz & Stegun 7.1.26
with FMA instructions + SLEEF exp. We replicate the EXACT computation
in scalar C with matching FMA behavior.

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-20
"""
import torch, numpy as np, ctypes, tempfile, subprocess, os

# PyTorch's EXACT erf polynomial from vec256_float.h:
#   p  = 0.3275911f
#   p1 = 0.254829592f
#   p2 = -0.284496736f
#   p3 = 1.421413741f
#   p4 = -1.453152027f
#   p5 = 1.061405429f
#
# Algorithm:
#   sign = sign(x)
#   abs_x = |x|
#   t = 1 / (p * abs_x + 1)
#   r = fma(p5, t, p4)
#   r = fma(r, t, p3)
#   r = fma(r, t, p2)
#   r = fma(r, t, p1)
#   pow2 = x * x
#   tmp4 = exp(-pow2)      // SLEEF exp
#   tmp5 = -tmp4
#   tmp6 = tmp5 * t
#   tmp7 = fma(tmp6, r, 1.0)
#   erf = sign * tmp7

C_SOURCE = r"""
#include <math.h>

// Replicate PyTorch's exact erf computation (A&S 7.1.26 + FMA)
static float pytorch_erff(float x) {
    const float p  = 0.3275911f;
    const float p1 = 0.254829592f;
    const float p2 = -0.284496736f;
    const float p3 = 1.421413741f;
    const float p4 = -1.453152027f;
    const float p5 = 1.061405429f;

    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float abs_x = fabsf(x);

    // t = 1 / (p * abs_x + 1)
    float t = 1.0f / fmaf(p, abs_x, 1.0f);

    // Horner polynomial with FMA: r = ((((p5*t + p4)*t + p3)*t + p2)*t + p1)
    float r = fmaf(p5, t, p4);
    r = fmaf(r, t, p3);
    r = fmaf(r, t, p2);
    r = fmaf(r, t, p1);

    // exp(-x*x)
    float pow2 = x * x;
    float neg_pow2 = -pow2;
    float exp_val = expf(neg_pow2);

    // erf = sign * (1 - r * t * exp(-x*x))
    //     = sign * fma(-exp_val * t, r, 1.0)
    float tmp6 = (-exp_val) * t;
    float result = fmaf(tmp6, r, 1.0f);
    return sign * result;
}

void pytorch_gelu_erf(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + pytorch_erff(x * 0.7071067811865476f));
    }
}

void pytorch_gelu_tanh(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    }
}

void pytorch_softmax(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row_in = in + r * cols;
        float *row_out = out + r * cols;

        // Pass 1: max
        float max_val = row_in[0];
        for (int c = 1; c < cols; c++) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        // Pass 2: exp + sum
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row_out[c] = expf(row_in[c] - max_val);
            sum += row_out[c];
        }

        // Pass 3: normalize
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row_out[c] *= inv_sum;
        }
    }
}

void pytorch_logsoftmax(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row_in = in + r * cols;
        float *row_out = out + r * cols;

        float max_val = row_in[0];
        for (int c = 1; c < cols; c++) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += expf(row_in[c] - max_val);
        }
        float log_sum = logf(sum);

        for (int c = 0; c < cols; c++) {
            row_out[c] = (row_in[c] - max_val) - log_sum;
        }
    }
}
"""

def compile_and_test():
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(C_SOURCE)
        c_path = f.name
    so_path = c_path.replace(".c", ".so")

    # Compile with FMA enabled (critical for matching PyTorch's _mm256_fmadd_ps)
    result = subprocess.run(
        ["gcc", "-O2", "-mfma", "-shared", "-fPIC", "-o", so_path, c_path, "-lm"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        print(f"Compile failed: {result.stderr}")
        return
    os.unlink(c_path)

    lib = ctypes.CDLL(so_path)

    # Test GELU erf
    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(16, 16384)
    x_np = x.numpy().copy()

    # PyTorch reference
    ref_gelu = torch.nn.functional.gelu(x).numpy()
    ref_gelu_tanh = torch.nn.functional.gelu(x, approximate='tanh').numpy()
    ref_softmax = torch.nn.functional.softmax(x[:, :4096], dim=-1).numpy()
    ref_logsoftmax = torch.nn.functional.log_softmax(x[:, :4096], dim=-1).numpy()

    # Our replicated GELU erf
    lib.pytorch_gelu_erf.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    out_gelu = np.zeros_like(x_np)
    lib.pytorch_gelu_erf(x_np.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          out_gelu.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          x_np.size)
    out_gelu = out_gelu.reshape(x_np.shape)

    # Our replicated GELU tanh
    lib.pytorch_gelu_tanh.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    out_gelu_tanh = np.zeros_like(x_np)
    lib.pytorch_gelu_tanh(x_np.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           out_gelu_tanh.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           x_np.size)
    out_gelu_tanh = out_gelu_tanh.reshape(x_np.shape)

    # Softmax
    lib.pytorch_softmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    x_sm = x_np[:, :4096].copy()
    out_sm = np.zeros_like(x_sm)
    lib.pytorch_softmax(x_sm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         out_sm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         16, 4096)

    # LogSoftmax
    lib.pytorch_logsoftmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    out_lsm = np.zeros_like(x_sm)
    lib.pytorch_logsoftmax(x_sm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            out_lsm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            16, 4096)

    # Results
    def classify(name, ref, our):
        ref_bits = ref.flatten().view(np.uint32)
        our_bits = our.flatten().view(np.uint32)
        mismatches = np.sum(ref_bits != our_bits)
        if mismatches == 0:
            print(f"  ✅ {name:25s} BIT_IDENTICAL  (0 ULP, {len(ref_bits)} elements)")
        else:
            max_ulp = int(np.max(np.abs(ref_bits.astype(np.int64) - our_bits.astype(np.int64))))
            pct = mismatches / len(ref_bits) * 100
            print(f"  ❌ {name:25s} MISMATCH       max_ULP={max_ulp:>8d}  ({mismatches}/{len(ref_bits)} = {pct:.1f}%)")

    print("=" * 70)
    print("SLEEF Polynomial Replication — Bit-Identical vs PyTorch")
    print("=" * 70)
    classify("GELU (erf, A&S+FMA)", ref_gelu, out_gelu)
    classify("GELU (tanh)", ref_gelu_tanh, out_gelu_tanh)
    classify("Softmax", ref_softmax, out_sm)
    classify("LogSoftmax", ref_logsoftmax, out_lsm)

    os.unlink(so_path)

if __name__ == '__main__':
    compile_and_test()
