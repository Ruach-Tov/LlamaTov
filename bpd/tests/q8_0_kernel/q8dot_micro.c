// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
extern float bpd_q8_0_dot(const int8_t* wq, const float* wd, const int8_t* aq, const float* ad, int nb);
int main(){
  // ONE block of 32, tiny known int8 values. Trace every intermediate.
  int8_t wq[32], aq[32];
  for(int j=0;j<32;j++){ wq[j]=(int8_t)(j-16); aq[j]=(int8_t)((j%5)-2); } // small known
  float wd[1]={0.0125f}, ad[1]={0.03f};   // simple scales (1 block)
  // hand reference (the AVX path arithmetic, traced):
  int isum=0; for(int j=0;j<32;j++) isum += (int)wq[j]*(int)aq[j];
  float scale = wd[0]*ad[0];           // fmul
  float dotf  = (float)isum;           // sitofp
  float prod  = scale*dotf;            // fmul
  float acc   = 0.0f + prod;           // fadd
  printf("REFERENCE trace (1 block):\n");
  printf("  isum (int dot)   = %d\n", isum);
  printf("  scale=wd*ad      = %.10e\n", scale);
  printf("  dotf=(float)isum = %.10e\n", dotf);
  printf("  prod=scale*dotf  = %.10e\n", prod);
  printf("  acc=0+prod       = %.10e\n", acc);
  float k = bpd_q8_0_dot(wq, wd, aq, ad, 1);
  printf("KERNEL bpd_q8_0_dot = %.10e\n", k);
  printf("  MATCH=%d  diff=%.3e\n", (k==acc), fabsf(k-acc));
  // bit-level
  uint32_t kb,ab; __builtin_memcpy(&kb,&k,4); __builtin_memcpy(&ab,&acc,4);
  printf("  kernel bits=0x%08x  ref bits=0x%08x  ULP=%d\n", kb, ab, (int)(kb>ab?kb-ab:ab-kb));
  return 0;
}
