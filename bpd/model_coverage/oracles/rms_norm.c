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
  ggml_cpu_init();
  int ne0=2048,ne1=2,n=ne0*ne1; float eps=atof(v[3]);
  float* in=ldf(v[1],n); float* xr=ldf(v[2],n);
  struct ggml_init_params p={128*1024*1024,NULL,false}; struct ggml_context* ctx=ggml_init(p);
  struct ggml_tensor* a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ne0,ne1); memcpy(a->data,in,n*4);
  struct ggml_tensor* r=ggml_rms_norm(ctx,a,eps);
  struct ggml_cgraph* gf=ggml_new_graph(ctx); ggml_build_forward_expand(gf,r); ggml_graph_compute_with_ctx(ctx,gf,1);
  float* o=(float*)r->data; int mx=0,nz=0;
  for(int i=0;i<n;i++){int32_t x,y;memcpy(&x,&o[i],4);memcpy(&y,&xr[i],4);int d=abs(x-y);if(d){nz++;if(d>mx)mx=d;}}
  printf("RMS_NORM eps=%g: maxULP=%d nz=%d/%d out[0]=%.9g ref[0]=%.9g\n",eps,mx,nz,n,o[0],xr[0]);
  printf(mx==0?"GREEN\n":"DIVERGENT\n"); return 0;
}
