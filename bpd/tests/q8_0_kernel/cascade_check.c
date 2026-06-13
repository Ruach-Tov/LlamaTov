// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// cascade_check.c — generalized ggml_mul_mat oracle for any layer-0 matmul.
// args: W_bytes_file X_f32_file dump_out_file NROW NTOK K
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"

int main(int argc, char** argv){
    if(argc<7){ fprintf(stderr,"usage: %s W X dump NROW NTOK K\n",argv[0]); return 1; }
    int NROW=atoi(argv[4]), NTOK=atoi(argv[5]), K=atoi(argv[6]);
    ggml_cpu_init();
    long wbytes = (long)NROW*(K/32)*34;
    uint8_t* Wb=malloc(wbytes);
    FILE* wf=fopen(argv[1],"rb"); if(fread(Wb,1,wbytes,wf)!=(size_t)wbytes){fprintf(stderr,"W fail (want %ld)\n",wbytes);return 1;} fclose(wf);
    float* Xf=malloc(sizeof(float)*NTOK*K);
    FILE* xf=fopen(argv[2],"rb"); if(fread(Xf,sizeof(float),NTOK*K,xf)!=(size_t)NTOK*K){fprintf(stderr,"X fail\n");return 1;} fclose(xf);

    struct ggml_init_params p={ (size_t)512*1024*1024, NULL, false };
    struct ggml_context* ctx=ggml_init(p);
    struct ggml_tensor* W=ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, NROW);
    memcpy(W->data, Wb, wbytes);
    struct ggml_tensor* X=ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NTOK);
    memcpy(X->data, Xf, sizeof(float)*NTOK*K);
    struct ggml_tensor* OUT=ggml_mul_mat(ctx, W, X);
    struct ggml_cgraph* gf=ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, OUT);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    float* out=(float*)OUT->data;

    FILE* df=fopen(argv[3],"rb"); fseek(df,80,SEEK_SET);
    int total=NROW*NTOK;
    float* dump=malloc(sizeof(float)*total);
    if(fread(dump,sizeof(float),total,df)!=(size_t)total){fprintf(stderr,"dump fail\n");return 1;} fclose(df);
    int mism=0,mu=0; double ma=0;
    for(int i=0;i<total;i++){ if(out[i]!=dump[i]){mism++; double a=fabs((double)out[i]-(double)dump[i]); if(a>ma)ma=a; int gi,ei;memcpy(&gi,&out[i],4);memcpy(&ei,&dump[i],4);int u=abs(gi-ei);if(u>mu)mu=u;} }
    printf("%s: %d/%d mism, maxabs %.3e, maxULP %d (out[0]=%.9g dump[0]=%.9g)\n", argv[3], mism, total, ma, mu, out[0], dump[0]);
    ggml_free(ctx); return 0;
}
