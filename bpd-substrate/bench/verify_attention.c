// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// verify_attention.c — standalone 0-ULP conformance test for bpd_gqa_attn_cpu vs ggml.
// Builds the ggml reference attention (scores -> ggml softmax -> ggml_vec_dot_f32 V-sum)
// on IDENTICAL inputs and compares bit-for-bit. Fills the attention coverage hole.
//
// This is the test that SHOULD have caught the V-sum scalar-vs-SIMD-tree divergence.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

extern void bpd_gqa_attn_cpu(const float* q, const float* k, const float* v, float* dst,
                             int n_q_tokens, int n_kv, int n_q_heads, int n_kv_heads,
                             int head_dim, float scale, int kv_offset);
// ggml reference primitives
extern void  ggml_vec_dot_f32(int n, float* s, size_t bs, const float* x, size_t bx,
                              const float* y, size_t by, int nrc);
// ggml softmax: y = exp(x-max)/sum, using ggml poly-exp + double sum (we replicate its exact form)
// ggml_vec_soft_max_f32 is in vec.cpp; we link it.
extern double ggml_vec_soft_max_f32(int n, float* y, const float* x, float max);

#define NQ 6      // query positions (prefill)
#define NKV 6     // keys
#define HQ 32     // query heads
#define HKV 8     // kv heads
#define D 64      // head dim

static float q[NQ*HQ*D], k[NKV*HKV*D], v[NKV*HKV*D];
static float ours[NQ*HQ*D], ref[NQ*HQ*D];

// ggml reference attention: for each (head, qpos): scores=scale*Q.K, softmax, kqv=softmax.V (vec_dot_f32)
static void ggml_ref_attention(float scale) {
    int gqa = HQ / HKV;
    for (int h = 0; h < HQ; h++) {
        int kvh = h / gqa;
        for (int qp = 0; qp < NQ; qp++) {
            const float* qv = q + (qp*HQ + h)*D;
            float scores[NKV];
            // causal: key kp <= qp
            int nk = qp + 1;
            float maxv = -1e30f;
            for (int kp = 0; kp < nk; kp++) {
                const float* kv_ = k + (kp*HKV + kvh)*D;
                float s; ggml_vec_dot_f32(D, &s, 0, qv, 0, kv_, 0, 1);
                scores[kp] = s * scale;
                if (scores[kp] > maxv) maxv = scores[kp];
            }
            ggml_vec_soft_max_f32(nk, scores, scores, maxv);  // ggml's exact softmax
            // normalize
            double sum = 0; for (int kp=0;kp<nk;kp++) sum += scores[kp];
            float inv = (float)(1.0/sum);  // ggml soft_max returns sum; here recompute consistently
            for (int kp=0;kp<nk;kp++) scores[kp] *= inv;
            // kqv[d] = vec_dot(softmax, V[:,d]) — ggml's V-sum order (gather V column)
            float* out = ref + (qp*HQ + h)*D;
            float vcol[NKV];
            for (int d = 0; d < D; d++) {
                for (int kp=0;kp<nk;kp++) vcol[kp] = v[(kp*HKV+kvh)*D + d];
                float s; ggml_vec_dot_f32(nk, &s, 0, scores, 0, vcol, 0, 1);
                out[d] = s;
            }
        }
    }
}

int main() {
    srand(42);
    for (int i=0;i<NQ*HQ*D;i++) q[i] = ((float)rand()/RAND_MAX - 0.5f);
    for (int i=0;i<NKV*HKV*D;i++) { k[i] = ((float)rand()/RAND_MAX - 0.5f); v[i] = ((float)rand()/RAND_MAX - 0.5f); }
    float scale = 1.0f / sqrtf((float)D);

    bpd_gqa_attn_cpu(q, k, v, ours, NQ, NKV, HQ, HKV, D, scale, 0);
    ggml_ref_attention(scale);

    int maxulp=0, ndiff=0, n=NQ*HQ*D; double maxabs=0;
    for (int i=0;i<n;i++) {
        int32_t a,b; memcpy(&a,&ours[i],4); memcpy(&b,&ref[i],4);
        int u=abs(a-b); if(u){ndiff++; if(u>maxulp)maxulp=u;}
        double d=fabs((double)ours[i]-ref[i]); if(d>maxabs)maxabs=d;
    }
    printf("ATTENTION vs ggml: maxULP=%d ndiff=%d/%d maxabs=%.3e  %s\n",
        maxulp, ndiff, n, maxabs, maxulp==0 ? "BIT-IDENTICAL (PASS)" : "DIVERGENT (FAIL)");
    return maxulp==0 ? 0 : 1;
}
