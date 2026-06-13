// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// node22.c — scores·V, EXACT source replica (llama-graph.cpp: kqv = ggml_mul_mat(v, kq)).
// v = TRANSPOSED V-cache F16, kq = softmax output (0041 [32,2,32], verified green). out 0044 [64,2,32].
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HDR 80
static float* ldf(const char* p, int n){ FILE* f=fopen(p,"rb"); fseek(f,HDR,SEEK_SET); float* b=malloc(n*4); fread(b,4,n,f); fclose(f); return b; }

int main(int c, char** v) {
    ggml_cpu_init();
    const int hd=64, nkvh=8, nkv_real=2, n_kv=32, nh=32, nt=2;
    float* vcur = ldf(v[1], hd*nkvh*nt);   // Vcur post (0018-analog for V) [64,8,2] = [hd, n_kv_heads, n_tok]
    float* kq   = ldf(v[2], n_kv*nt*nh);   // 0041 softmax output [32,2,32] = [n_kv, n_tok, n_head]
    float* xref = ldf(v[3], hd*nt*nh);     // 0044 [64,2,32]

    struct ggml_init_params p={512*1024*1024,NULL,false}; struct ggml_context* ctx=ggml_init(p);

    // Vcur [64,8,2] = [hd, n_kv_heads, n_tok]
    struct ggml_tensor* Vcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nkvh, nt);
    memcpy(Vcur->data, vcur, ggml_nbytes(Vcur));

    // V cache: source does v = ggml_cont(ggml_transpose(v)) → V stored TRANSPOSED.
    // The V cache layout for mul_mat(v, kq): v must be [n_kv, head_dim, n_kv_heads] (transposed so n_kv contracts with kq's n_kv).
    // Build cache as [head_dim, n_kv, n_kv_heads] then the read is transposed. Replay: cpy Vcur into cache, transpose.
    struct ggml_tensor* cache_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hd, n_kv, nkvh);
    memset(cache_v->data, 0, ggml_nbytes(cache_v));
    // write Vcur [hd, nkvh, nt] -> permute [hd, nt, nkvh] -> cpy into first nkv_real of cache
    struct ggml_tensor* Vcur_p = ggml_cont(ctx, ggml_permute(ctx, Vcur, 0, 2, 1, 3)); // [hd, nt, nkvh]
    struct ggml_tensor* v_wr = ggml_view_3d(ctx, cache_v, hd, nkv_real, nkvh, cache_v->nb[1], cache_v->nb[2], 0);
    struct ggml_tensor* cpy = ggml_cpy(ctx, Vcur_p, v_wr);

    // read view full padded [hd, n_kv=32, nkvh], then TRANSPOSE to [n_kv, hd, nkvh] (v_trans path)
    struct ggml_tensor* v_rd = ggml_view_3d(ctx, cache_v, hd, n_kv, nkvh, cache_v->nb[1], cache_v->nb[2], 0);
    struct ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, v_rd));  // [n_kv, hd, nkvh]

    // kq softmax output [n_kv=32, n_tok=2, n_head=32]
    struct ggml_tensor* KQ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_kv, nt, nh);
    memcpy(KQ->data, kq, ggml_nbytes(KQ));

    // kqv = ggml_mul_mat(v, kq): v=[n_kv,hd,nkvh] contracts n_kv with kq=[n_kv,nt,nh] -> [hd, nt, nh]
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, vt, KQ);
    ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cpy);
    ggml_build_forward_expand(gf, kqv);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    float* o=(float*)kqv->data; int n=ggml_nelements(kqv);
    printf("kqv ne[%lld,%lld,%lld] nelem=%d\n",(long long)kqv->ne[0],(long long)kqv->ne[1],(long long)kqv->ne[2],n);
    int mx=0,nz=0;
    for(int i=0;i<n && i<hd*nt*nh;i++){int32_t a,b;memcpy(&a,&o[i],4);memcpy(&b,&xref[i],4);int d=abs(a-b);if(d){nz++;if(d>mx)mx=d;}}
    printf("scores.V node_22: maxULP=%d nz=%d/%d out[0]=%.9g ref[0]=%.9g\n",mx,nz,n,o[0],xref[0]);
    printf(mx==0?"GREEN (bit-exact)\n":"DIVERGENT\n");
    return 0;
}
