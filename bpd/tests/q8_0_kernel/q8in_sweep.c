// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <math.h>
extern float bpd_q8_0_dot(const int8_t* wq, const float* wd, const int8_t* aq, const float* ad, int nb);
// ggml-EXACT reference: replicate the AVX two-block/8-lane/hsum accumulation in C (same instructions).
static inline __m128i mae(const __m128i x,const __m128i y){return _mm_maddubs_epi16(_mm_sign_epi8(x,x),_mm_sign_epi8(y,x));}
static inline float hsum8(__m256 x){__m128 r=_mm_add_ps(_mm256_extractf128_ps(x,1),_mm256_castps256_ps128(x));r=_mm_add_ps(r,_mm_movehl_ps(r,r));r=_mm_add_ss(r,_mm_movehdup_ps(r));return _mm_cvtss_f32(r);}
static float ggml_ref(const int8_t* wq,const float* wd,const int8_t* aq,const float* ad,int nb){
  __m256 acc=_mm256_setzero_ps(); __m128i ones=_mm_set1_epi16(1);
  for(int b=0;b+1<nb;b+=2){
    // block b -> p_lo (4 i32), block b+1 -> p_hi
    __m128i xb0=_mm_loadu_si128((const __m128i*)(wq+b*32)), xb1=_mm_loadu_si128((const __m128i*)(wq+b*32+16));
    __m128i yb0=_mm_loadu_si128((const __m128i*)(aq+b*32)), yb1=_mm_loadu_si128((const __m128i*)(aq+b*32+16));
    __m128i plo=_mm_add_epi32(_mm_madd_epi16(mae(xb0,yb0),ones),_mm_madd_epi16(mae(xb1,yb1),ones));
    __m128i Xb0=_mm_loadu_si128((const __m128i*)(wq+(b+1)*32)), Xb1=_mm_loadu_si128((const __m128i*)(wq+(b+1)*32+16));
    __m128i Yb0=_mm_loadu_si128((const __m128i*)(aq+(b+1)*32)), Yb1=_mm_loadu_si128((const __m128i*)(aq+(b+1)*32+16));
    __m128i phi=_mm_add_epi32(_mm_madd_epi16(mae(Xb0,Yb0),ones),_mm_madd_epi16(mae(Xb1,Yb1),ones));
    __m256 pf=_mm256_cvtepi32_ps(_mm256_set_m128i(phi,plo));
    float s0=wd[b]*ad[b], s1=wd[b+1]*ad[b+1];
    __m256 d=_mm256_set_m128(_mm_set1_ps(s1),_mm_set1_ps(s0));
    acc=_mm256_add_ps(_mm256_mul_ps(d,pf),acc);
  }
  return hsum8(acc);
}
int main(){
  for(int nb=2;nb<=64;nb*=2){
    int8_t wq[64*32], aq[64*32]; float wd[64], ad[64];
    for(int b=0;b<nb;b++){wd[b]=0.01f+0.001f*b; ad[b]=0.02f-0.0001f*b;
      for(int j=0;j<32;j++){wq[b*32+j]=(int8_t)((b*7+j)%13-6); aq[b*32+j]=(int8_t)((b*5+j)%11-5);}}
    float k=bpd_q8_0_dot(wq,wd,aq,ad,nb);
    float g=ggml_ref(wq,wd,aq,ad,nb);
    uint32_t kb,gb; __builtin_memcpy(&kb,&k,4);__builtin_memcpy(&gb,&g,4);
    printf("nb=%2d: kernel=%.8e ggml-exact=%.8e diff=%.3e ULP=%d\n", nb, k, g, fabsf(k-g), (int)(kb>gb?kb-gb:gb-kb));
  }
  return 0;
}
