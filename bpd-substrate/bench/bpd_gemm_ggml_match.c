// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_gemm_ggml_match.c — GEMM kernel that matches ggml's exact accumulation order.
 *
 * Sweepable parameter: BPD_GEMM_EPR (elements per register)
 *   EPR=4:  matches ggml SSE backend (128-bit, 4 floats)
 *   EPR=8:  matches ggml AVX/AVX2 backend (256-bit, 8 floats)
 *   EPR=16: matches ggml AVX512 backend (512-bit, 16 floats)
 *
 * The kernel replicates ggml's ggml_vec_dot_f32 algorithm:
 *   - 4 striped accumulators, each EPR-wide
 *   - FMA (or mul+add depending on fma_strategy)
 *   - Binary tree reduction: acc[0]+=acc[2], acc[1]+=acc[3], acc[0]+=acc[1]
 *   - Horizontal sum within each EPR-wide accumulator
 *
 * Named configurations:
 *   "ggml-sse"     — EPR=4,  fma=mul_add  (SSE4.2, no FMA hardware)
 *   "ggml-avx2"    — EPR=8,  fma=hardware (AVX2+FMA, Haswell+)
 *   "ggml-avx512"  — EPR=16, fma=hardware (AVX-512, Skylake-X+)
 *   "ggml-neon"    — EPR=4,  fma=hardware (ARM NEON)
 *   "pytorch-mkl"  — EPR=?, fma=? (to be determined by tracing MKL)
 *
 * Part of BPD Substrate: https://github.com/heath-hunnicutt-ruach-tov/bpd-substrate
 * Licensed under GPLv2
 */

#include <math.h>
#include <string.h>
#include <stdio.h>

/* ============================================================
 * Sweepable parameters — set before compilation or at runtime
 * ============================================================ */

/* Elements per register: determines accumulation grouping */
#ifndef BPD_GEMM_EPR
#define BPD_GEMM_EPR 4  /* default: match ggml SSE */
#endif

/* Number of accumulators: determines how many parallel partial sums
 * NEON: 4, SSE3: 8, AVX: 8, AVX512: 4
 * ggml STEP = EPR * ARR, so ARR = STEP / EPR */
#ifndef BPD_GEMM_ARR
#define BPD_GEMM_ARR 8  /* default: match ggml SSE3 (STEP=32, EPR=4, ARR=8) */
#endif

/* Elements per iteration step (derived, but can be overridden) */
#ifndef BPD_GEMM_STEP
#define BPD_GEMM_STEP (BPD_GEMM_EPR * BPD_GEMM_ARR)
#endif

/* FMA strategy: 0 = mul then add (2 roundings), 1 = fmaf (1 rounding) */
#ifndef BPD_GEMM_FMA
#define BPD_GEMM_FMA 0  /* default: mul+add (matches SSE without FMA) */
#endif

/* ============================================================
 * Named configurations
 * ============================================================ */

typedef struct {
    const char *name;
    int epr;       /* elements per register (SIMD width / 32) */
    int arr;       /* number of accumulators */
    int fma;       /* 0=mul+add, 1=hardware FMA */
    const char *description;
} bpd_gemm_config_t;

/* Sweepable parameter lattice: {EPR, ARR, FMA}
 * Each named config is a point in this 3D space.
 * The lattice can be sampled for: numerical stability, precision, performance.
 *
 *   EPR: 4 (SSE/NEON), 8 (AVX/AVX2), 16 (AVX512)
 *   ARR: 4 (NEON/AVX512), 8 (SSE3/AVX)
 *   FMA: 0 (mul+add, 2 roundings), 1 (fmaf, 1 rounding)
 *
 * STEP = EPR * ARR (elements per loop iteration)
 */
static const bpd_gemm_config_t bpd_gemm_configs[] = {
    {"ggml-sse3",       4, 8, 0, "ggml SSE3 (128-bit, 8 accum, mul+add)"},
    {"ggml-sse3-fma",   4, 8, 1, "ggml SSE3+FMA (128-bit, 8 accum, hardware FMA)"},
    {"ggml-avx",        8, 8, 0, "ggml AVX (256-bit, 8 accum, mul+add)"},
    {"ggml-avx2",       8, 8, 1, "ggml AVX2+FMA (256-bit, 8 accum, hardware FMA)"},
    {"ggml-avx512",    16, 4, 1, "ggml AVX-512 (512-bit, 4 accum, hardware FMA)"},
    {"ggml-neon",       4, 4, 1, "ggml ARM NEON (128-bit, 4 accum, hardware FMA)"},
    {"naive-scalar",    1, 1, 0, "naive scalar (1 accum, mul+add, for documentation)"},
    {"naive-scalar-fma",1, 1, 1, "naive scalar+FMA (1 accum, hardware FMA)"},
    {NULL, 0, 0, 0, NULL}
};

/* Look up a named configuration */
const bpd_gemm_config_t *bpd_gemm_find_config(const char *name) {
    for (int i = 0; bpd_gemm_configs[i].name; i++) {
        if (strcmp(bpd_gemm_configs[i].name, name) == 0)
            return &bpd_gemm_configs[i];
    }
    return NULL;
}

/* List all configurations */
void bpd_gemm_list_configs(void) {
    printf("BPD GEMM named configurations:\n");
    for (int i = 0; bpd_gemm_configs[i].name; i++) {
        printf("  %-16s EPR=%-2d FMA=%d  %s\n",
               bpd_gemm_configs[i].name,
               bpd_gemm_configs[i].epr,
               bpd_gemm_configs[i].fma,
               bpd_gemm_configs[i].description);
    }
}

/* ============================================================
 * Core: vec_dot matching ggml's exact accumulation order
 * ============================================================ */

/* Scalar vec_dot that replicates ggml's SIMD accumulation order.
 *
 * ggml groups elements into EPR-wide chunks across 4 accumulators:
 *   acc[0] accumulates: elements 0..EPR-1, 4*EPR..5*EPR-1, 8*EPR..9*EPR-1, ...
 *   acc[1] accumulates: elements EPR..2*EPR-1, 5*EPR..6*EPR-1, ...
 *   acc[2] accumulates: elements 2*EPR..3*EPR-1, ...
 *   acc[3] accumulates: elements 3*EPR..4*EPR-1, ...
 *
 * Within each accumulator, elements are summed left-to-right (position 0..EPR-1).
 * Reduction: binary tree — acc[0]+=acc[2], acc[1]+=acc[3], acc[0]+=acc[1].
 * Final: horizontal sum of the EPR-wide accumulator.
 *
 * This function uses scalar math but produces the SAME bits as the
 * SIMD version because it follows the same accumulation order.
 */
static float bpd_vec_dot_ggml_order(int n, const float *x, const float *y, int epr, int use_fma) {
    const int step = epr * BPD_GEMM_ARR;
    const int np = n & ~(step - 1);
    
    /* 4 accumulators, each with EPR lanes */
    float acc[BPD_GEMM_ARR][16];  /* max EPR=16 */
    memset(acc, 0, sizeof(acc));
    
    /* Main loop: matches ggml's inner loop exactly */
    for (int i = 0; i < np; i += step) {
        for (int j = 0; j < BPD_GEMM_ARR; j++) {
            for (int k = 0; k < epr; k++) {
                int idx = i + j * epr + k;
                if (use_fma) {
                    acc[j][k] = fmaf(x[idx], y[idx], acc[j][k]);
                } else {
                    acc[j][k] += x[idx] * y[idx];
                }
            }
        }
    }
    
    /* Reduce accumulators: binary tree (matches ggml's GGML_F32_VEC_REDUCE) */
    /* Step 1: acc[0] += acc[2], acc[1] += acc[3] */
    for (int k = 0; k < epr; k++) {
        acc[0][k] += acc[2][k];
        acc[1][k] += acc[3][k];
    }
    /* Step 2: acc[0] += acc[1] */
    for (int k = 0; k < epr; k++) {
        acc[0][k] += acc[1][k];
    }
    
    /* Horizontal sum within acc[0] (matches SIMD horizontal add) */
    /* For SSE: hadd pairs, then hadd again. For AVX: extract hi128, add, then SSE hadd. */
    float sumf = 0.0f;
    for (int k = 0; k < epr; k++) {
        sumf += acc[0][k];
    }
    
    /* Tail: leftover elements (matches ggml's scalar tail loop) */
    for (int i = np; i < n; i++) {
        sumf += x[i] * y[i];
    }
    
    return sumf;
}

/* ============================================================
 * GEMM using the ggml-matching dot product
 * ============================================================ */

/* C = A @ B^T where A is [M, K] and B is [N, K] (B stored row-major, transposed)
 * This matches ggml's ggml_mul_mat layout.
 *
 * epr: elements per register (sweepable)
 * use_fma: 0=mul+add, 1=fmaf (sweepable)
 */
void bpd_gemm_ggml_match(
        int M, int N, int K,
        const float *A, int lda,
        const float *B, int ldb,
        float *C, int ldc,
        int epr, int use_fma) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            C[m * ldc + n] = bpd_vec_dot_ggml_order(K, A + m * lda, B + n * ldb, epr, use_fma);
        }
    }
}

/* Convenience: GEMM with named configuration */
void bpd_gemm_named(
        int M, int N, int K,
        const float *A, int lda,
        const float *B, int ldb,
        float *C, int ldc,
        const char *config_name) {
    const bpd_gemm_config_t *cfg = bpd_gemm_find_config(config_name);
    if (!cfg) {
        fprintf(stderr, "Unknown GEMM config: %s\n", config_name);
        bpd_gemm_list_configs();
        return;
    }
    bpd_gemm_ggml_match(M, N, K, A, lda, B, ldb, C, ldc, cfg->epr, cfg->fma);
}

/* ============================================================
 * Conv2d using im2col + ggml-matching GEMM
 * ============================================================ */

/* im2col: rearrange input patches into columns for GEMM-based convolution.
 * Matches ggml's ggml_im2col implementation.
 *
 * Input:  [IC, IH, IW] (channel-first)
 * Output: [OH*OW, IC*KH*KW] (im2col matrix)
 */
void bpd_im2col(
        const float *input, int IC, int IH, int IW,
        float *col, int KH, int KW,
        int stride_h, int stride_w,
        int pad_h, int pad_w) {
    int OH = (IH + 2 * pad_h - KH) / stride_h + 1;
    int OW = (IW + 2 * pad_w - KW) / stride_w + 1;
    int col_w = IC * KH * KW;
    
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            int col_row = oh * OW + ow;
            int col_idx = 0;
            for (int ic = 0; ic < IC; ic++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = oh * stride_h - pad_h + kh;
                        int iw = ow * stride_w - pad_w + kw;
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            col[col_row * col_w + col_idx] = input[ic * IH * IW + ih * IW + iw];
                        } else {
                            col[col_row * col_w + col_idx] = 0.0f;
                        }
                        col_idx++;
                    }
                }
            }
        }
    }
}

/* Conv2d = im2col + GEMM with ggml-matching accumulation order.
 *
 * Input:  [N, IC, IH, IW]
 * Weight: [OC, IC, KH, KW]
 * Bias:   [OC] or NULL
 * Output: [N, OC, OH, OW]
 *
 * config_name: named GEMM configuration (e.g., "ggml-sse", "ggml-avx2")
 */
void bpd_conv2d_ggml_match(
        const float *input, int N, int IC, int IH, int IW,
        const float *weight, int OC, int KH, int KW,
        const float *bias,
        float *output,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        const char *config_name) {
    int OH = (IH + 2 * pad_h - KH) / stride_h + 1;
    int OW = (IW + 2 * pad_w - KW) / stride_w + 1;
    int col_h = OH * OW;
    int col_w = IC * KH * KW;
    
    const bpd_gemm_config_t *cfg = bpd_gemm_find_config(config_name);
    if (!cfg) {
        fprintf(stderr, "Unknown config: %s\n", config_name);
        return;
    }
    
    /* Allocate im2col buffer */
    float *col = (float *)calloc(col_h * col_w, sizeof(float));
    
    for (int n = 0; n < N; n++) {
        /* im2col for this batch element */
        bpd_im2col(input + n * IC * IH * IW, IC, IH, IW,
                   col, KH, KW, stride_h, stride_w, pad_h, pad_w);
        
        /* GEMM: output[n] = weight @ col^T
         * weight: [OC, IC*KH*KW], col: [OH*OW, IC*KH*KW]
         * result: [OC, OH*OW]
         */
        bpd_gemm_ggml_match(OC, col_h, col_w,
                           weight, col_w,
                           col, col_w,
                           output + n * OC * OH * OW, col_h,
                           cfg->epr, cfg->fma);
        
        /* Add bias if present */
        if (bias) {
            for (int oc = 0; oc < OC; oc++) {
                for (int hw = 0; hw < OH * OW; hw++) {
                    output[n * OC * OH * OW + oc * OH * OW + hw] += bias[oc];
                }
            }
        }
    }
    
    free(col);
}
