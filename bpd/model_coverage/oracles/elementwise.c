// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HDR 80
static float* ldf(const char*p,int n){FILE*f=fopen(p,"rb");fseek(f,HDR,SEEK_SET);float*b=malloc(n*4);fread(b,4,n,f);fclose(f);return b;}
int main(int c,char**v){
  // argv: src0 src1 ref op(add|mul) ne0 ne1 [w_ne0 for broadcast mul]
  ggml_cpu_init();
  int ne0=atoi(v[5]), ne1=atoi(v[6]); int n=ne0*ne1;
  const char* op=v[4];
  int w1d = (c>7 && atoi(v[7])>0) ? atoi(v[7]) : 0;  // if set, src1 is 1D [w1d] broadcast
  float* a=ldf(v[1],n); float* b=ldf(v[2], w1d?w1d:n); float* xr=ldf(v[3],n);
  struct ggml_init_params p={128*1024*1024,NULL,false}; struct ggml_context* ctx=ggml_init(p);
  struct ggml_tensor* A=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ne0,ne1); memcpy(A->data,a,n*4);
  struct ggml_tensor* B;
  if(w1d){ B=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,w1d); memcpy(B->data,b,w1d*4); }
  else   { B=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ne0,ne1); memcpy(B->data,b,n*4); }
  struct ggml_tensor* r = (op[0]==(char)97)? ggml_add(ctx,A,B) : ggml_mul(ctx,A,B);
  struct ggml_cgraph* gf=ggml_new_graph(ctx); ggml_build_forward_expand(gf,r); ggml_graph_compute_with_ctx(ctx,gf,1);
  float* o=(float*)r->data; int mx=0,nz=0;
  for(int i=0;i<n;i++){int32_t x,y;memcpy(&x,&o[i],4);memcpy(&y,&xr[i],4);int d=abs(x-y);if(d){nz++;if(d>mx)mx=d;}}
  printf("%s [%d,%d]: maxULP=%d nz=%d/%d %s\n",v[4],ne0,ne1,mx,nz,n,mx==0?"GREEN":"DIV");return 0;
}
