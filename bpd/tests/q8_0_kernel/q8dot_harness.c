// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
// the Prolog-generated kernel (linked from q8dot.o)
extern float bpd_q8_0_dot(const int8_t* wq, const float* wd, const int8_t* aq, const float* ad, int nb);
static float h2f(uint16_t h){ return _cvtsh_ss(h); }
static uint16_t f2h(float f){ return _cvtss_sh(f, 0); }
int main(int argc,char**argv){
  long off=atol(argv[1]); int N=2048,QK=32,bpr=N/QK;
  FILE* bf=fopen(argv[2],"rb"); fseek(bf,off,SEEK_SET);
  uint8_t* W=malloc((size_t)N*bpr*34); fread(W,1,(size_t)N*bpr*34,bf); fclose(bf);
  FILE* kf=fopen(argv[3],"rb"); uint32_t dt,nd;long ne[4];uint64_t nb_[4],nbytes;
  fread(&dt,4,1,kf);fread(&nd,4,1,kf);fread(ne,8,4,kf);fread(nb_,8,4,kf);fread(&nbytes,8,1,kf);
  float* kqv=malloc(nbytes);fread(kqv,1,nbytes,kf);fclose(kf);int ntok=nbytes/4/N;
  FILE* sf=fopen(argv[4],"rb");fread(&dt,4,1,sf);fread(&nd,4,1,sf);fread(ne,8,4,sf);fread(nb_,8,4,sf);fread(&nbytes,8,1,sf);
  float* spec=malloc(nbytes);fread(spec,1,nbytes,sf);fclose(sf);
  // separate weight quants + scales into the kernel's layout
  int8_t* wq=malloc((size_t)N*bpr*32); float* wd=malloc((size_t)N*bpr*4);
  for(int r=0;r<N;r++)for(int b=0;b<bpr;b++){
    wd[r*bpr+b]=h2f(*(uint16_t*)(W+((size_t)r*bpr+b)*34));
    for(int j=0;j<32;j++) wq[((size_t)r*bpr+b)*32+j]=*(int8_t*)(W+((size_t)r*bpr+b)*34+2+j);
  }
  int8_t* aq=malloc(bpr*32); float* ad=malloc(bpr*4);
  float maxabs=0;
  for(int t=0;t<ntok;t++){
    // quantize activation row to q8_0 (ggml EXACT: d=amax/127 fp32, id=1/d_fp32, store fp16(d))
    for(int b=0;b<bpr;b++){
      float amax=0; for(int j=0;j<32;j++){float a=fabsf(kqv[(size_t)t*N+b*32+j]);if(a>amax)amax=a;}
      float d=amax/127.0f; float id=d?1.0f/d:0.0f;
      ad[b]=h2f(f2h(d));  // stored scale = fp16(d)
      for(int j=0;j<32;j++) aq[b*32+j]=(int8_t)roundf(kqv[(size_t)t*N+b*32+j]*id);
    }
    for(int r=0;r<N;r++){
      float s=bpd_q8_0_dot(wq+(size_t)r*bpr*32, wd+(size_t)r*bpr, aq, ad, bpr);
      float dd=fabsf(s-spec[(size_t)t*N+r]); if(dd>maxabs)maxabs=dd;
    }
  }
  printf("o_proj via PROLOG-GENERATED q8_0_dot (AVX path) vs ggml dump: max_abs=%.6e %s\n",maxabs,maxabs==0?"*** 0 ULP ***":"");
  return 0;
}
