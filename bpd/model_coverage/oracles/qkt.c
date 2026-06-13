// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// qkt10.c — EXACT replica of llama-graph.cpp attention (read from source, not guessed).
// kq = ggml_mul_mat(k, q); ggml_mul_mat_set_prec(kq, GGML_PREC_F32).
// k->ne1 = n_kv (padded), q->ne2 = n_head. result [n_kv, n_tokens, n_head] = [32,2,32].
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HDR 80
static float* ldf(const char* p, int n){ FILE* f=fopen(p,"rb"); fseek(f,HDR,SEEK_SET); float* b=malloc(n*4); fread(b,4,n,f); fclose(f); return b; }

int main(int c, char** v) {
    ggml_cpu_init();
    const int hd=64, nkvh=8, nkv_real=2, n_kv=32, nh=32, nt=2;  // n_kv padded to 32 (= ref ne0)
    float* qcur = ldf(v[1], hd*nh*nt);   // 0011 Qcur post-rope [64,32,2] = [hd, n_head, n_tok]
    float* kcur = ldf(v[2], hd*nkvh*nt); // 0018 Kcur post-rope [64,8,2] = [hd, n_kv_heads, n_tok]
    float* xref = ldf(v[3], 32*2*32);

    struct ggml_init_params p={512*1024*1024,NULL,false}; struct ggml_context* ctx=ggml_init(p);

    struct ggml_tensor* Qcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, nt);
    memcpy(Qcur->data, qcur, ggml_nbytes(Qcur));
    struct ggml_tensor* Kcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nkvh, nt);
    memcpy(Kcur->data, kcur, ggml_nbytes(Kcur));

    // cache: [head_dim, n_kv(padded 32), n_kv_heads]  (the standard llama kv layout, n_ctx slot=32 here for the view)
    struct ggml_tensor* cache_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hd, n_kv, nkvh);
    memset(cache_k->data, 0, ggml_nbytes(cache_k));

    // write Kcur (n_kv_real=2 tokens) into the cache. Kcur [hd, nkvh, nt] -> permute to [hd, nt, nkvh] (tok->kv pos)
    struct ggml_tensor* Kcur_p = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3)); // [hd, nt, nkvh]
    struct ggml_tensor* k_wr = ggml_view_3d(ctx, cache_k, hd, nkv_real, nkvh, cache_k->nb[1], cache_k->nb[2], 0);
    struct ggml_tensor* cpy = ggml_cpy(ctx, Kcur_p, k_wr);

    // k = the FULL padded cache view [hd, n_kv=32, nkvh] (n_kv = k->ne[1] per source)
    struct ggml_tensor* k = ggml_view_3d(ctx, cache_k, hd, n_kv, nkvh, cache_k->nb[1], cache_k->nb[2], 0);
    // q = Qcur permuted (0,2,1,3): [hd, n_head, n_tok] -> [hd, n_tok, n_head]
    struct ggml_tensor* q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);  // [hd, n_tok, n_head]

    // kq = ggml_mul_mat(k, q) -> [k->ne1=n_kv=32, q->ne1=n_tok=2, q->ne2=n_head=32]
    struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);  // THE KEY: force F32 precision

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cpy);
    ggml_build_forward_expand(gf, kq);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    float* o=(float*)kq->data; int n=ggml_nelements(kq);
    printf("kq ne[%lld,%lld,%lld] nelem=%d\n",(long long)kq->ne[0],(long long)kq->ne[1],(long long)kq->ne[2],n);
    int mx=0,nz=0,realbad=0;
    for(int i=0;i<n && i<32*2*32;i++){
        int32_t a,b; memcpy(&a,&o[i],4); memcpy(&b,&xref[i],4); int d=abs(a-b);
        if(d){nz++; if(d>mx)mx=d; if((i%32)<2)realbad++;}
    }
    printf("QK^T source-replica: maxULP=%d nz=%d/%d realbad=%d out[0]=%.9g ref[0]=%.9g\n",mx,nz,n,realbad,o[0],xref[0]);
    printf(mx==0?"GREEN (bit-exact)\n":"DIVERGENT\n");
    return 0;
}
