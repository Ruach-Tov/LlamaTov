// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
extern void quantize_row_q8_0(const float* x, void* y, int64_t k);  // ggml REAL quant (from .so)
// my harness quant (the one in q8dot_harness.c)
static uint16_t f2h(float f){return _cvtss_sh(f,0);}
int main(int argc,char**argv){
  FILE* kf=fopen(argv[1],"rb"); uint32_t dt,nd;long ne[4];uint64_t nb_[4],nbytes;
  fread(&dt,4,1,kf);fread(&nd,4,1,kf);fread(ne,8,4,kf);fread(nb_,8,4,kf);fread(&nbytes,8,1,kf);
  float* kqv=malloc(nbytes);fread(kqv,1,nbytes,kf);fclose(kf);
  int N=2048,QK=32,bpr=N/QK;
  // ggml real quant of token-0 row
  uint8_t* gq=malloc(bpr*34); quantize_row_q8_0(kqv, gq, N);
  // my harness quant
  int mism_q=0, mism_d=0;
  for(int b=0;b<bpr;b++){
    float amax=0; for(int j=0;j<32;j++){float a=fabsf(kqv[b*32+j]);if(a>amax)amax=a;}
    float d=amax/127.0f; float id=d?1.0f/d:0.0f; uint16_t mydh=f2h(d);
    uint16_t gdh=*(uint16_t*)(gq+b*34);
    if(mydh!=gdh){mism_d++; if(mism_d<=3) printf("  block %d: my d_h=0x%04x ggml d_h=0x%04x\n",b,mydh,gdh);}
    int8_t* gqs=(int8_t*)(gq+b*34+2);
    for(int j=0;j<32;j++){ int8_t myq=(int8_t)roundf(kqv[b*32+j]*id); if(myq!=gqs[j]){mism_q++; if(mism_q<=5)printf("    blk%d[%d]: my q=%d ggml q=%d (x=%.6f)\n",b,j,myq,gqs[j],kqv[b*32+j]);} }
  }
  printf("ACTIVATION QUANT: scale mismatches=%d/%d  quant mismatches=%d/%d\n", mism_d,bpr, mism_q,bpr*32);
  printf("  -> %s\n", (mism_d==0&&mism_q==0)?"IDENTICAL (residual is NOT input quant)":"DIFFERS (THIS is the residual source)");
  return 0;
}
