// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// Replicate ggml quantize_row_q8_0_ref EXACTLY for one row, dump qs + d bits.
#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#define QK 32
int main(void){
  float x[2048];
  for(int i=0;i<2048;i++){ if(scanf("%f",&x[i])!=1) return 1; }
  int nb=2048/QK;
  for(int i=0;i<nb;i++){
    float amax=0.0f;
    for(int j=0;j<QK;j++){ float v=fabsf(x[i*QK+j]); if(v>amax)amax=v; }
    float d=amax/127.0f;
    float id = d ? 1.0f/d : 0.0f;
    unsigned short dh=_cvtss_sh(d,0);
    printf("%04x",dh);
    for(int j=0;j<QK;j++){ float x0=x[i*QK+j]*id; int q=(int)roundf(x0); printf(" %d",q); }
    printf("\n");
  }
  return 0;
}
