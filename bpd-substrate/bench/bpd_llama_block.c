// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* ─────────────────────────────────────────────────────────────────────────
 * L.1.10  bpd_llama_block_cpu / bpd_llama_forward_cpu
 *
 * Full transformer block composition and complete forward pass.
 * Composes the individual L.1.4–L.1.9 kernels into a single-function
 * transformer block, then iterates n_layers blocks for the full forward.
 *
 * This file is compiled separately and linked with bpd_cpu.so, or
 * #included at the end of bpd_cpu.c.
 *
 * Memory strategy:
 *   - Two scratch buffers (ping-pong) for intermediate activations
 *   - KV cache is pre-allocated externally (max_seq_len * n_kv_heads * head_dim)
 *   - All weight pointers are passed via a model descriptor struct
 *
 * ggml reference: llama_decode_internal → llm_build_llama
 * ───────────────────────────────────────────────────────────────────────── */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── Forward declarations of the composed kernels ─────────────────────── */
extern void bpd_rmsnorm_llama_cpu(const float* input, const float* weight,
                                   float* output, int n_rows, int row_len, float eps);
extern void bpd_rope_neox_cpu(const float* input, float* output,
                               const int32_t* pos_ids, int n_tokens, int n_heads,
                               int head_dim, int n_dims, float freq_base);
extern void bpd_rope_norm_freqs_cpu(const float* input, float* output,
                                     const int32_t* pos_ids,
                                     const float* freq_factors,
                                     int n_tokens, int n_heads,
                                     int head_dim, int n_dims, float freq_base);
extern void bpd_kv_cache_write_cpu(float* cache, const float* src,
                                    const int32_t* pos_ids, int n_tokens,
                                    int n_kv_heads, int head_dim, int max_seq_len);
extern void bpd_gqa_attn_cpu(const float* q, const float* k, const float* v,
                              float* dst, int n_q_tokens, int n_kv,
                              int n_q_heads, int n_kv_heads, int head_dim,
                              float scale, int kv_offset);
extern void bpd_add_f32_cpu(const float* a, const float* b, float* dst, int n);
extern void bpd_silu_f32_cpu(const float* x, float* dst, int n);
extern void bpd_mul_f32_cpu(const float* a, const float* b, float* dst, int n);
extern void bpd_swiglu_fuse_cpu(const float* gate, const float* up, float* dst, int n);
extern void bpd_qmatmul_q8_0_llamafile_cpu(const uint8_t* W_q8_0, const float* X_f32,
                                             float* out, int m_weight, int m_tokens, int K);
extern void bpd_embed_lookup_q8_0_cpu(const uint8_t* table, const int32_t* token_ids,
                                       float* out, int n_tokens, int embed_dim);
extern void bpd_argmax_dim_cpu(const float* x, long* out,
                                int outer, int dim_size, int inner);

/* ── Model configuration struct ──────────────────────────────────────── */
typedef struct {
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int embed_dim;       /* = n_heads * head_dim */
    int ffn_dim;         /* intermediate size (e.g., 14336 for Llama3-8B) */
    int vocab_size;
    int max_seq_len;
    float rms_eps;
    float rope_base;
    int rope_dim;        /* usually = head_dim */
} bpd_llama_config;

/* ── Per-layer weight pointers ───────────────────────────────────────── */
typedef struct {
    const float*   attn_norm_w;   /* [embed_dim] */
    const uint8_t* w_q;           /* Q8_0: [n_heads*head_dim, embed_dim] */
    const uint8_t* w_k;           /* Q8_0: [n_kv_heads*head_dim, embed_dim] */
    const uint8_t* w_v;           /* Q8_0: [n_kv_heads*head_dim, embed_dim] */
    const uint8_t* w_o;           /* Q8_0: [embed_dim, n_heads*head_dim] */
    const float*   ffn_norm_w;    /* [embed_dim] */
    const uint8_t* w_gate;        /* Q8_0: [ffn_dim, embed_dim] */
    const uint8_t* w_up;          /* Q8_0: [ffn_dim, embed_dim] */
    const uint8_t* w_down;        /* Q8_0: [embed_dim, ffn_dim] */
} bpd_llama_layer_weights;

/* ── Full model weight pointers ──────────────────────────────────────── */
typedef struct {
    const uint8_t*            token_embd;    /* Q8_0: [vocab_size, embed_dim] */
    const bpd_llama_layer_weights* layers;   /* array of n_layers */
    const float*              output_norm_w; /* [embed_dim] */
    const uint8_t*            output_w;      /* Q8_0: [vocab_size, embed_dim] */
    const float*              rope_freqs;    /* [n_dims/2] or NULL for no NTK-aware freq scaling */
} bpd_llama_weights;

/* ── Single transformer block ────────────────────────────────────────── */
void bpd_llama_block_cpu(
        float*                       x,          /* [n_tokens, embed_dim] in/out */
        const bpd_llama_layer_weights* lw,
        const bpd_llama_config*      cfg,
        const int32_t*               pos_ids,    /* [n_tokens] */
        int                          n_tokens,
        int                          kv_pos,     /* current KV cache position */
        float*                       k_cache,    /* [max_seq_len, n_kv_heads*head_dim] */
        float*                       v_cache,    /* [max_seq_len, n_kv_heads*head_dim] */
        float*                       scratch1,   /* >= n_tokens * max(embed_dim, ffn_dim) */
        float*                       scratch2,   /* >= n_tokens * max(embed_dim, ffn_dim) */
        float*                       scratch3,   /* >= n_tokens * max(n_heads*head_dim, ffn_dim) */
        const float*                 rope_freqs) /* [n_dims/2] or NULL */
{
    const int E = cfg->embed_dim;
    const int H = cfg->n_heads;
    const int HKV = cfg->n_kv_heads;
    const int D = cfg->head_dim;
    const int F = cfg->ffn_dim;
    const int n_kv = kv_pos + n_tokens;  /* total KV length after this step */

    /* ── Attention sub-block ─────────────────────────────────────────── */

    /* 1. RMSNorm (attention) */
    bpd_rmsnorm_llama_cpu(x, lw->attn_norm_w, scratch1, n_tokens, E, cfg->rms_eps);

    /* 2. Q/K/V projections (Q8_0 matmul) */
    /* Q: [n_tokens, E] @ W_q^T → [n_tokens, H*D] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_q, scratch1, scratch2, H * D, n_tokens, E);
    /* K: [n_tokens, E] @ W_k^T → [n_tokens, HKV*D] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_k, scratch1, scratch3, HKV * D, n_tokens, E);

    /* 3. RoPE on Q (in-place in scratch2). NORM-style with optional NTK-aware
     *    freq_factors (Llama 3 long-context extension). */
    bpd_rope_norm_freqs_cpu(scratch2, scratch2, pos_ids, rope_freqs,
                            n_tokens, H, D, cfg->rope_dim, cfg->rope_base);

    /* 4. RoPE on K (in-place in scratch3). NORM-style with freq_factors. */
    bpd_rope_norm_freqs_cpu(scratch3, scratch3, pos_ids, rope_freqs,
                            n_tokens, HKV, D, cfg->rope_dim, cfg->rope_base);

    /* 5. Write K to KV cache */
    bpd_kv_cache_write_cpu(k_cache, scratch3, pos_ids, n_tokens,
                           HKV, D, cfg->max_seq_len);

    /* 6. V projection: [n_tokens, E] @ W_v^T → [n_tokens, HKV*D] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_v, scratch1, scratch3, HKV * D, n_tokens, E);

    /* 7. Write V to KV cache */
    bpd_kv_cache_write_cpu(v_cache, scratch3, pos_ids, n_tokens,
                           HKV, D, cfg->max_seq_len);

    /* 8. GQA attention: Q against full KV cache */
    float scale = 1.0f / sqrtf((float)D);
    bpd_gqa_attn_cpu(scratch2, k_cache, v_cache,
                     scratch1,  /* output: [n_tokens, H*D] */
                     n_tokens, n_kv, H, HKV, D, scale, kv_pos);

    /* 9. Output projection: [n_tokens, H*D] @ W_o^T → [n_tokens, E] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_o, scratch1, scratch2, E, n_tokens, H * D);

    /* 10. Residual add: x = x + attn_out */
    bpd_add_f32_cpu(x, scratch2, x, n_tokens * E);

    /* ── FFN sub-block ───────────────────────────────────────────────── */

    /* 11. RMSNorm (FFN) */
    bpd_rmsnorm_llama_cpu(x, lw->ffn_norm_w, scratch1, n_tokens, E, cfg->rms_eps);

    /* 12. Gate projection: [n_tokens, E] @ W_gate^T → [n_tokens, F] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_gate, scratch1, scratch2, F, n_tokens, E);

    /* 13. Up projection: [n_tokens, E] @ W_up^T → [n_tokens, F] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_up, scratch1, scratch3, F, n_tokens, E);

    /* 14. SwiGLU: silu(gate) * up → scratch2 */
    bpd_swiglu_fuse_cpu(scratch2, scratch3, scratch2, n_tokens * F);

    /* 15. Down projection: [n_tokens, F] @ W_down^T → [n_tokens, E] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_down, scratch2, scratch1, E, n_tokens, F);

    /* 16. Residual add: x = x + ffn_out */
    bpd_add_f32_cpu(x, scratch1, x, n_tokens * E);
}

/* ── Full forward pass ───────────────────────────────────────────────── */
void bpd_llama_forward_cpu(
        const int32_t*             token_ids,   /* [n_tokens] */
        int                        n_tokens,
        const bpd_llama_weights*   weights,
        const bpd_llama_config*    cfg,
        const int32_t*             pos_ids,     /* [n_tokens] absolute positions */
        int                        kv_pos,      /* first position in KV cache to write */
        float*                     k_cache,     /* [n_layers, max_seq_len, n_kv_heads*head_dim] */
        float*                     v_cache,     /* [n_layers, max_seq_len, n_kv_heads*head_dim] */
        float*                     logits_out,  /* [n_tokens, vocab_size] */
        long*                      token_out)   /* [n_tokens] argmax result */
{
    const int E = cfg->embed_dim;
    const int F = cfg->ffn_dim;
    const int H = cfg->n_heads;
    const int D = cfg->head_dim;
    const int HKV = cfg->n_kv_heads;
    const int max_dim = (E > F) ? E : F;
    const int max_proj = (H * D > F) ? H * D : F;

    /* Allocate working buffers */
    float* x        = (float*)malloc((size_t)n_tokens * E * sizeof(float));
    float* scratch1 = (float*)malloc((size_t)n_tokens * max_dim * sizeof(float));
    float* scratch2 = (float*)malloc((size_t)n_tokens * max_proj * sizeof(float));
    float* scratch3 = (float*)malloc((size_t)n_tokens * max_proj * sizeof(float));
    if (!x || !scratch1 || !scratch2 || !scratch3) goto cleanup;

    /* 1. Token embedding lookup */
    bpd_embed_lookup_q8_0_cpu(weights->token_embd, token_ids, x, n_tokens, E);

    /* 2. Iterate through all transformer layers */
    const size_t kv_layer_stride = (size_t)cfg->max_seq_len * HKV * D;
    for (int layer = 0; layer < cfg->n_layers; layer++) {
        float* layer_k_cache = k_cache + layer * kv_layer_stride;
        float* layer_v_cache = v_cache + layer * kv_layer_stride;

        bpd_llama_block_cpu(x, &weights->layers[layer], cfg,
                            pos_ids, n_tokens, kv_pos,
                            layer_k_cache, layer_v_cache,
                            scratch1, scratch2, scratch3,
                            weights->rope_freqs);
    }

    /* 3. Final RMSNorm */
    bpd_rmsnorm_llama_cpu(x, weights->output_norm_w, scratch1, n_tokens, E, cfg->rms_eps);

    /* 4. Output projection (logits): [n_tokens, E] @ W_output^T → [n_tokens, vocab_size] */
    bpd_qmatmul_q8_0_llamafile_cpu(weights->output_w, scratch1, logits_out,
                                   cfg->vocab_size, n_tokens, E);

    /* 5. Argmax over vocabulary dimension */
    bpd_argmax_dim_cpu(logits_out, token_out, n_tokens, cfg->vocab_size, 1);

cleanup:
    free(x);
    free(scratch1);
    free(scratch2);
    free(scratch3);
}
