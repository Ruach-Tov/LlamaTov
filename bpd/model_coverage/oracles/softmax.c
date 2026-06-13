// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// softmax_oracle.c — bit-exact ggml_soft_max_ext oracle for Llama-3 layer-0 attention scores.
// input = node_20 Q·K^T scores [32,2,32], mask = 0045 [32,2,32], scale=0.125 (1/sqrt(64)).
// Build: clang -mavx -mf16c -mno-avx2 -mno-fma -I$REF/ggml/include -L$REF/build/bin
//        softmax_oracle.c -o softmax_oracle -lggml-cpu -lggml-base -lggml -lm
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NE0 32
#define NE1 2
#define NE2 32
#define HDR 80

static float* load_dump(const char* path, int nfloats) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, HDR, SEEK_SET);
    float* buf = malloc(nfloats * sizeof(float));
    size_t got = fread(buf, sizeof(float), nfloats, f);
    fclose(f);
    if ((int)got != nfloats) { fprintf(stderr, "short read %zu/%d in %s\n", got, nfloats, path); exit(1); }
    return buf;
}

int main(int argc, char** argv) {
    ggml_cpu_init();
    int n = NE0 * NE1 * NE2;  // 2048
    float* xin  = load_dump(argv[1], n);   // 0038 scores
    float* mask = load_dump(argv[2], n);   // 0045 mask
    float* xref = load_dump(argv[3], n);   // 0041 softmax output

    struct ggml_init_params p = { 64*1024*1024, NULL, false };
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NE0, NE1, NE2);
    memcpy(a->data, xin, n * sizeof(float));
    // construct causal mask: 2D [ne0=keys=32, ne1=queries=2]. q attends keys 0..q AND key<2.
    struct ggml_tensor* m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, NE1);
    float* md = (float*)m->data;
    for (int q = 0; q < NE1; q++)
        for (int k = 0; k < NE0; k++)
            md[q*NE0 + k] = (k <= q && k < NE1) ? 0.0f : -INFINITY;

    // ggml_soft_max_ext(a, mask, scale=0.125, max_bias=0)
    struct ggml_tensor* r = ggml_soft_max_ext(ctx, a, m, 0.125f, 0.0f);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    float* out = (float*)r->data;
    int max_ulp = 0, nz = 0;
    for (int i = 0; i < n; i++) {
        int32_t ai, bi; memcpy(&ai, &out[i], 4); memcpy(&bi, &xref[i], 4);
        int d = abs(ai - bi);
        if (d > 0) { nz++; if (d > max_ulp) max_ulp = d; }
    }
    printf("SOFTMAX node_21 [32,2,32]: maxULP=%d nonzero=%d/%d  out[0]=%.9g ref[0]=%.9g\n",
           max_ulp, nz, n, out[0], xref[0]);
    printf(max_ulp == 0 ? "GREEN (bit-exact)\n" : "DIVERGENT\n");
    return 0;
}
