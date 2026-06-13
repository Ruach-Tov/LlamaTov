// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
extern float bpd_q8_0_dot(const int8_t* wq, const float* wd, const int8_t* aq, const float* ad, int nb);
int main(){
  // nb=2 (one two-block iteration). Known values, trace vs hand-computed.
  int8_t wq[64], aq[64]; for(int j=0;j<64;j++){wq[j]=(int8_t)((j%7)-3); aq[j]=(int8_t)((j%5)-2);}
  float wd[2]={0.0125f,0.02f}, ad[2]={0.03f,0.015f};
  // hand ref: per block, isum = dot(32); acc += (wd*ad)*isum
  float acc=0;
  for(int b=0;b<2;b++){
    int isum=0; for(int j=0;j<32;j++) isum+=(int)wq[b*32+j]*(int)aq[b*32+j];
    float prod=(wd[b]*ad[b])*(float)isum; acc+=prod;
    printf("  block %d: isum=%d scale=%.6e prod=%.6e\n", b, isum, wd[b]*ad[b], prod);
  }
  printf("REFERENCE acc (nb=2) = %.10e\n", acc);
  float k=bpd_q8_0_dot(wq,wd,aq,ad,2);
  printf("KERNEL (vec) = %.10e   diff=%.3e\n", k, fabsf(k-acc));
  return 0;
}
