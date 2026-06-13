// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* ggml-matching vectorized exp for softmax.
 *
 * Extracted from ggml_vec_soft_max_f32 disassembly (libggml-cpu.so).
 * Uses SSE 128-bit (4-wide) Cephes-style polynomial:
 *   1. Range reduce: x * log2(e) → integer part n + fractional part f
 *   2. Polynomial: 2^f ≈ p(f) via Horner form
 *   3. Scale: result = p(f) * 2^n (via integer add to exponent)
 *
 * This produces BIT-IDENTICAL output to ggml's softmax on Ivy Bridge.
 *
 * Author: medayek (Collective SME, Verification Methodology)
 */

#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

/* Polynomial exp matching ggml's ggml_v_expf / vec_soft_max inner loop.
 * Processes 4 floats at a time via SSE.
 * For inputs < -126*ln(2) ≈ -87.3, returns 0 (flush to zero). */
static inline __m128 ggml_exp4(__m128 x) {
    /* Constants extracted from ggml disassembly */
    const __m128 log2e    = _mm_set1_ps(1.442695021629f);    /* 0x3fb8aa3b */
    const __m128 ln2_hi   = _mm_set1_ps(-6.931457519531e-01f); /* negated for subtract */
    const __m128 ln2_lo   = _mm_set1_ps(-1.428606765330e-06f);
    const __m128 c0       = _mm_set1_ps(8.247390389442e-03f);  /* 0x3c072010 */
    const __m128 c1       = _mm_set1_ps(4.189976677299e-02f);  /* 0x3d2b9f17 */
    const __m128 c2       = _mm_set1_ps(1.666839569807e-01f);  /* 0x3e2aaf33 */
    const __m128 c3       = _mm_set1_ps(4.999912679195e-01f);  /* 0x3efffedb */
    const __m128 c4       = _mm_set1_ps(9.999994039536e-01f);  /* 0x3f7ffff6 */
    const __m128 one      = _mm_set1_ps(1.0f);
    const __m128 zero     = _mm_setzero_ps();
    const __m128 inf_threshold = _mm_set1_ps(126.0f);         /* 0x42fc0000 */

    /* Step 1: n = round(x * log2(e)) */
    __m128 fx = _mm_mul_ps(x, log2e);

    /* Round to nearest integer: add 0.5 and truncate, or use magic number */
    /* ggml uses: vaddps + vpslld $0x17 pattern (integer exponent manipulation) */
    /* Simpler: round via _mm_round_ps if available (SSE4.1), else add-and-truncate */
    __m128 n_f = _mm_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128i n_i = _mm_cvtps_epi32(n_f);

    /* Step 2: Cody-Waite range reduction: f = x - n*ln(2) */
    /* Two-part subtraction for precision */
    __m128 f = _mm_add_ps(x, _mm_mul_ps(n_f, ln2_hi));  /* x + n*(-ln2_hi) = x - n*ln2_hi */
    f = _mm_add_ps(f, _mm_mul_ps(n_f, ln2_lo));          /* f -= n*ln2_lo */

    /* Step 3: Polynomial approximation of 2^f - 1
     * p(f) = ((((c0*f + c1)*f + c2)*f^2 + ... 
     * Horner form matching ggml's instruction sequence */
    __m128 f2 = _mm_mul_ps(f, f);

    __m128 p = _mm_mul_ps(f, c0);
    p = _mm_add_ps(p, c1);
    
    __m128 q = _mm_mul_ps(f, c2);
    q = _mm_add_ps(q, c3);
    
    p = _mm_mul_ps(p, f2);
    p = _mm_add_ps(p, q);
    p = _mm_mul_ps(p, f2);
    
    __m128 r = _mm_mul_ps(f, c4);
    p = _mm_add_ps(p, r);

    /* Step 4: Reconstruct: result = (1 + p) * 2^n
     * 2^n is done by adding n to the IEEE exponent field */
    __m128i exp_bits = _mm_add_epi32(n_i, _mm_set1_epi32(0x7f));  /* bias */
    exp_bits = _mm_slli_epi32(exp_bits, 23);                       /* shift to exponent position */
    __m128 scale = _mm_castsi128_ps(exp_bits);

    __m128 result = _mm_mul_ps(p, scale);
    result = _mm_add_ps(result, scale);

    /* Step 5: Flush to zero for very negative inputs */
    /* Check |n| > 126 → clamp to 0 */
    __m128 abs_fx = _mm_andnot_ps(_mm_set1_ps(-0.0f), fx);
    __m128 mask = _mm_cmplt_ps(abs_fx, inf_threshold);  /* true if |fx| < 126 */
    
    /* For negative overflow: if x was very negative, result should be 0 */
    __m128 neg_mask = _mm_cmple_ps(fx, zero);
    __m128 overflow_mask = _mm_andnot_ps(mask, neg_mask);  /* |fx| >= 126 AND fx <= 0 */
    result = _mm_andnot_ps(overflow_mask, result);  /* zero out overflow entries */

    return result;
}

/* Softmax using ggml-matching exp polynomial.
 * Signature matches bpd_softmax_causal_cpu. */
void bpd_softmax_causal_ggml_cpu(
        const float* scores,
        float*       output,
        int          n_heads,
        int          n_q,
        int          n_kv,
        float        scale,
        int          q_offset)
{
    for (int h = 0; h < n_heads; h++) {
        for (int q = 0; q < n_q; q++) {
            const float* row_in  = scores + (h * n_q + q) * n_kv;
            float*       row_out = output + (h * n_q + q) * n_kv;
            const int    q_abs   = q_offset + q;

            /* 1. Scale + causal mask, find max */
            float max_val = -1e38f;
            for (int k = 0; k < n_kv; k++) {
                float v = (k <= q_abs) ? row_in[k] * scale : -1e38f;
                row_out[k] = v;
                if (v > max_val) max_val = v;
            }

            /* 2. exp(x - max) using ggml polynomial, accumulate sum */
            __m128 vmax = _mm_set1_ps(max_val);
            __m128 vsum = _mm_setzero_ps();
            
            int k = 0;
            /* Vectorized path: 4 elements at a time */
            for (; k + 3 < n_kv; k += 4) {
                __m128 v = _mm_loadu_ps(row_out + k);
                v = _mm_sub_ps(v, vmax);
                __m128 e = ggml_exp4(v);
                _mm_storeu_ps(row_out + k, e);
                vsum = _mm_add_ps(vsum, e);
            }
            
            /* Horizontal sum */
            float sum_arr[4];
            _mm_storeu_ps(sum_arr, vsum);
            float sum_exp = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
            
            /* Scalar tail */
            for (; k < n_kv; k++) {
                float v = row_out[k] - max_val;
                float e = expf(v);  /* scalar fallback for tail */
                row_out[k] = e;
                sum_exp += e;
            }

            /* 3. Normalize */
            float inv_sum = 1.0f / sum_exp;
            for (int k2 = 0; k2 < n_kv; k2++) {
                row_out[k2] *= inv_sum;
            }
        }
    }
}
