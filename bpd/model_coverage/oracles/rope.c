// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// rope_oracle.c — bit-exact ROPE oracle for Llama-3 layer-0 Qcur.
// Loads pre-rope 0007_MUL_MAT_Qcur [64,32,2], applies ggml_rope_ext (NEOX), compares to 0011.
// Build: clang -mavx -mf16c -mno-avx2 -mno-fma -I$REF/ggml/include -L$REF/build/bin \
//        rope_oracle.c -o rope_oracle -lggml-cpu -lggml-base -lggml -lm
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HEAD_DIM 64
#define N_HEADS  32
#define N_TOK    2
#define HDR 80

static float* load_dump(const char* path, int nfloats) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, HDR, SEEK_SET);
    float* buf = malloc(nfloats * sizeof(float));
    size_t got = fread(buf, sizeof(float), nfloats, f);
    fclose(f);
    if ((int)got != nfloats) { fprintf(stderr, "short read %zu/%d\n", got, nfloats); exit(1); }
    return buf;
}

int main(int argc, char** argv) {
    ggml_cpu_init();
    const char* pre  = argv[1];
    const char* post = argv[2];
    int n = HEAD_DIM * N_HEADS * N_TOK;  // 4096
    float* xin  = load_dump(pre,  n);
    float* xref = load_dump(post, n);

    struct ggml_init_params p = { 64*1024*1024, NULL, false };
    struct ggml_context* ctx = ggml_init(p);

    // input a: [HEAD_DIM, N_HEADS, N_TOK]
    struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEADS, N_TOK);
    memcpy(a->data, xin, n * sizeof(float));

    // positions b: I32 [N_TOK] = (0, 1)
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_TOK);
    ((int32_t*)pos->data)[0] = 0;
    ((int32_t*)pos->data)[1] = 1;

    // rope_factors c: [n_rot/2]=32 floats from 0014_rope_freqs.weight
    float* rf = load_dump(argv[3], 32);
    struct ggml_tensor* rfac = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    memcpy(rfac->data, rf, 32*sizeof(float));

    // ggml_rope_ext: NEOX mode=2, n_dims=64, freq_base=500000, freq_scale=1, n_ctx_orig=8192
    struct ggml_tensor* r = ggml_rope_ext(ctx, a, pos, rfac,
        /*n_dims*/64, /*mode STANDARD*/0, /*n_ctx_orig*/4096,
        /*freq_base*/500000.0f, /*freq_scale*/1.0f,
        /*ext_factor*/0.0f, /*attn_factor*/1.0f, /*beta_fast*/32.0f, /*beta_slow*/1.0f);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    float* out = (float*)r->data;
    int max_ulp = 0, nz = 0;
    for (int i = 0; i < n; i++) {
        int32_t ai, bi;
        memcpy(&ai, &out[i], 4); memcpy(&bi, &xref[i], 4);
        int d = abs(ai - bi);
        if (d > 0) { nz++; if (d > max_ulp) max_ulp = d; }
    }
    printf("ROPE Qcur [64,32,2]: maxULP=%d nonzero=%d/%d  out[0]=%.9g ref[0]=%.9g\n",
           max_ulp, nz, n, out[0], xref[0]);
    FILE* of=fopen("/tmp/my_rope_out.bin","wb"); fwrite(out,4,n,of); fclose(of);
    printf(max_ulp == 0 ? "GREEN (bit-exact)\n" : "DIVERGENT\n");
    return 0;
}
