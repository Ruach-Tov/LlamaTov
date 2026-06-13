// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// elementwise_check.c — ggml unary/binary op oracle. Verifies silu/add/mul/rms_norm
// vs captured dumps using ggml's own ops. args: OP inA [inB|rmsw] dump N (total elements)
// OP in {silu, add, mul}. For silu: only inA. For add/mul: inA, inB. dump compared.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"

static float* readf(const char* path, int n){ float* b=malloc(sizeof(float)*n); FILE*f=fopen(path,"rb"); if(fread(b,sizeof(float),n,f)!=(size_t)n){fprintf(stderr,"read fail %s\n",path);exit(1);} fclose(f); return b; }
static float* readdump(const char* path, int n){ float* b=malloc(sizeof(float)*n); FILE*f=fopen(path,"rb"); fseek(f,80,SEEK_SET); if(fread(b,sizeof(float),n,f)!=(size_t)n){fprintf(stderr,"dump fail\n");exit(1);} fclose(f); return b; }

int main(int argc, char** argv){
    // argv: OP inA inB_or_dump dump_or_N ...
    const char* op=argv[1];
    ggml_cpu_init();
    struct ggml_init_params p={(size_t)256*1024*1024,NULL,false};
    struct ggml_context* ctx=ggml_init(p);
    int N; float* A; float* B=NULL; const char* dumpf;
    if(strcmp(op,"silu")==0){ N=atoi(argv[4]); A=readf(argv[2],N); dumpf=argv[3]; }
    else { N=atoi(argv[5]); A=readf(argv[2],N); B=readf(argv[3],N); dumpf=argv[4]; }
    struct ggml_tensor* ta=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,N); memcpy(ta->data,A,sizeof(float)*N);
    struct ggml_tensor* out;
    if(strcmp(op,"silu")==0) out=ggml_silu(ctx,ta);
    else { struct ggml_tensor* tb=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,N); memcpy(tb->data,B,sizeof(float)*N);
           out = (strcmp(op,"add")==0)? ggml_add(ctx,ta,tb) : ggml_mul(ctx,ta,tb); }
    struct ggml_cgraph* gf=ggml_new_graph(ctx); ggml_build_forward_expand(gf,out); ggml_graph_compute_with_ctx(ctx,gf,1);
    float* o=(float*)out->data;
    float* dump=readdump(dumpf,N);
    int mism=0,mu=0; double ma=0;
    for(int i=0;i<N;i++){ if(o[i]!=dump[i]){mism++; double a=fabs((double)o[i]-(double)dump[i]); if(a>ma)ma=a; int gi,ei;memcpy(&gi,&o[i],4);memcpy(&ei,&dump[i],4);int u=abs(gi-ei);if(u>mu)mu=u;} }
    printf("%s %s: %d/%d mism, maxabs %.3e, maxULP %d (o[0]=%.9g dump[0]=%.9g)\n", op, dumpf, mism, N, ma, mu, o[0], dump[0]);
    ggml_free(ctx); return 0;
}
