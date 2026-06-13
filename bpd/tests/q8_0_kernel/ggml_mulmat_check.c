// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// ggml_mulmat_check.c — run the ACTUAL ggml_mul_mat (full dispatch incl. llamafile)
// from /tmp/llama_cpp_test's libggml on the Qcur matmul, compare to spec_dump_v2.
// This is the authoritative provenance test: if 0 ULP, spec_dump_v2 came from this build.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"

#define K 2048
#define NROW 2048   // attn_q output rows (ne01)
#define NTOK 2      // tokens (ne11)

int main(int argc, char** argv){
    // argv[1]=W q8_0 bytes (NROW rows * 2176), argv[2]=X f32 (NTOK*K), argv[3]=dump
    ggml_cpu_init();
    long wbytes = (long)NROW*(K/32)*34;
    uint8_t* Wbytes = malloc(wbytes);
    FILE* wf=fopen(argv[1],"rb"); fread(Wbytes,1,wbytes,wf); fclose(wf);
    float* Xf = malloc(sizeof(float)*NTOK*K);
    FILE* xf=fopen(argv[2],"rb"); fread(Xf,sizeof(float),NTOK*K,xf); fclose(xf);

    size_t mem = 256*1024*1024;
    struct ggml_init_params p = { mem, NULL, false };
    struct ggml_context* ctx = ggml_init(p);

    // src0 = weight: q8_0, ne = [K, NROW] (ne0=K cols, ne1=NROW rows)
    struct ggml_tensor* W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, NROW);
    memcpy(W->data, Wbytes, wbytes);
    // src1 = activation: f32, ne = [K, NTOK]
    struct ggml_tensor* X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NTOK);
    memcpy(X->data, Xf, sizeof(float)*NTOK*K);

    struct ggml_tensor* OUT = ggml_mul_mat(ctx, W, X);  // -> ne = [NROW, NTOK]
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, OUT);
    ggml_graph_compute_with_ctx(ctx, gf, 1);  // 1 thread for determinism

    float* out = (float*)OUT->data;  // out[token*NROW + row]
    // dump 0007: [ne0=NROW, ne1=NTOK], flat = row + token*NROW. Same layout as out.
    FILE* df=fopen(argv[3],"rb"); fseek(df,80,SEEK_SET);
    int total = NROW*NTOK;
    float* dump = malloc(sizeof(float)*total);
    fread(dump,sizeof(float),total,df); fclose(df);

    int mism=0,max_ulp=0; double maxabs=0;
    for(int i=0;i<total;i++){
        float g=out[i], e=dump[i];
        if(g!=e){ mism++; double a=fabs((double)g-(double)e); if(a>maxabs)maxabs=a;
            int gi,ei; memcpy(&gi,&g,4); memcpy(&ei,&e,4); int u=abs(gi-ei); if(u>max_ulp)max_ulp=u;
            if(mism<=4) printf("  idx %d: got %.9g exp %.9g ulp %d\n", i, g, e, u); }
    }
    printf("ggml_mul_mat (this build) vs spec_dump_v2: %d/%d mism, max abs %.3e, max ULP %d\n", mism, total, maxabs, max_ulp);
    printf("out[0]=%.17g dump[0]=%.17g\n", out[0], dump[0]);
    ggml_free(ctx);
    return 0;
}
