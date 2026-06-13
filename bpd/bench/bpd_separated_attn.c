// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_separated_attn_cpu: ggml-matching separated attention.
 *
 * Implements QK^T → scale → causal mask → softmax → attn×V
 * in SEPARATE steps, matching ggml's exact computation order.
 *
 * This is attention_math_strategy(separated_sequential) —
 * the 0-ULP-vs-ggml value. The fused Flash Attention
 * (bpd_gqa_attn_cpu) is the O(1)-memory value.
 *
 * Author: medayek (Collective SME, Verification Methodology)
 * Plan: c13d771b Phase 2d
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* Separated attention: QK^T -> scale -> causal mask -> softmax -> attn*V
 *
 * Args:
 *   q:          [n_tokens, n_heads * head_dim]  F32
 *   k:          [n_kv, n_kv_heads * head_dim]   F32
 *   v:          [n_kv, n_kv_heads * head_dim]   F32
 *   out:        [n_tokens, n_heads * head_dim]  F32
 *   n_tokens:   number of query tokens
 *   n_kv:       number of key/value tokens (= n_tokens for prefill)
 *   n_heads:    number of attention heads (32 for llama3.2-1b)
 *   n_kv_heads: number of KV heads (8 for GQA)
 *   head_dim:   dimension per head (64)
 *   scale:      1/sqrt(head_dim)
 *   kv_pos:     KV cache start position (0 for prefill)
 */
void bpd_separated_attn_cpu(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int n_tokens,
    int n_kv,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    int kv_pos)
{
    int heads_per_kv = n_heads / n_kv_heads;

    /* Allocate intermediates */
    float* qk = (float*)malloc(n_kv * sizeof(float));       /* one row of QK^T scores */
    float* attn_weights = (float*)malloc(n_kv * sizeof(float)); /* softmax output */

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;  /* GQA: multiple Q heads share one KV head */

        for (int t = 0; t < n_tokens; t++) {
            /* Step 1: QK^T for this (head, token) against all KV positions */
            const float* q_vec = q + t * (n_heads * head_dim) + h * head_dim;

            for (int s = 0; s < n_kv; s++) {
                const float* k_vec = k + s * (n_kv_heads * head_dim) + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
                qk[s] = dot * scale;
            }

            /* Step 2: Causal mask — positions after current token get -inf */
            int causal_limit = kv_pos + t + 1;
            for (int s = causal_limit; s < n_kv; s++) {
                qk[s] = -INFINITY;
            }

            /* Step 3: Softmax (subtract max for numerical stability, then exp+normalize) */
            float max_val = -INFINITY;
            for (int s = 0; s < n_kv; s++) {
                if (qk[s] > max_val) max_val = qk[s];
            }

            float sum_exp = 0.0f;
            for (int s = 0; s < n_kv; s++) {
                attn_weights[s] = expf(qk[s] - max_val);
                sum_exp += attn_weights[s];
            }

            float inv_sum = 1.0f / sum_exp;
            for (int s = 0; s < n_kv; s++) {
                attn_weights[s] *= inv_sum;
            }

            /* Step 4: Weighted sum of V */
            float* out_vec = out + t * (n_heads * head_dim) + h * head_dim;
            memset(out_vec, 0, head_dim * sizeof(float));

            for (int s = 0; s < n_kv; s++) {
                const float* v_vec = v + s * (n_kv_heads * head_dim) + kv_h * head_dim;
                float w = attn_weights[s];
                for (int d = 0; d < head_dim; d++) {
                    out_vec[d] += w * v_vec[d];
                }
            }
        }
    }

    free(qk);
    free(attn_weights);
}
