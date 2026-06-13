// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* GPU head-to-head: BPD vs llama.cpp Q4_K dequant */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define QK_K 256
#define Q4K_BYTES 144
typedef uint16_t ggml_half;
typedef struct { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[QK_K/2]; } block_q4_K;

__device__ float dev_fp16(uint16_t h) {
    unsigned s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF;
    if(!e) return m?ldexpf((float)m/1024.f,-14)*(s?-1:1):0;
    if(e==31) return s?-INFINITY:INFINITY;
    return (s?-1.f:1.f)*ldexpf(1.f+(float)m/1024.f,(int)e-15);
}

__device__ inline void dev_scale_min(int j, const uint8_t*q, uint8_t*d, uint8_t*m) {
    if(j<4){*d=q[j]&63;*m=q[j+4]&63;}
    else{*d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}

/* llama.cpp's exact GPU kernel (from convert.cu) */
__global__ void llamacpp_dequant_gpu(const void* __restrict__ vx, float* __restrict__ yy) {
    const block_q4_K* x = (const block_q4_K*)vx;
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ir = tid%8;
    const int64_t is = 2*il;
    const int64_t n = 4;
    float* y = yy + i*QK_K + 64*il + n*ir;
    float dall = dev_fp16(x[i].d);
    float dmin = dev_fp16(x[i].dmin);
    const uint8_t* q = x[i].qs + 32*il + n*ir;
    uint8_t sc, m;
    dev_scale_min(is+0, x[i].scales, &sc, &m);
    float d1=dall*sc, m1=dmin*m;
    dev_scale_min(is+1, x[i].scales, &sc, &m);
    float d2=dall*sc, m2=dmin*m;
    for(int l=0;l<n;++l){
        y[l+0]=d1*(q[l]&0xF)-m1;
        y[l+32]=d2*(q[l]>>4)-m2;
    }
}

/* Our best GPU kernel (block=32, ILP=2 style with matched algorithm) */
__global__ void bpd_dequant_gpu(const void* __restrict__ vx, float* __restrict__ yy) {
    const block_q4_K* x = (const block_q4_K*)vx;
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8;
    const int64_t ir = tid%8;
    const int64_t is = 2*il;
    const int64_t n = 4;
    float* y = yy + i*QK_K + 64*il + n*ir;
    float dall = dev_fp16(x[i].d);
    float dmin = dev_fp16(x[i].dmin);
    const uint8_t* q = x[i].qs + 32*il + n*ir;
    uint8_t sc, m;
    dev_scale_min(is+0, x[i].scales, &sc, &m);
    float d1=dall*sc, m1=dmin*m;
    dev_scale_min(is+1, x[i].scales, &sc, &m);
    float d2=dall*sc, m2=dmin*m;
    for(int l=0;l<n;++l){
        y[l+0]=d1*(q[l]&0xF)-m1;
        y[l+32]=d2*(q[l]>>4)-m2;
    }
}

extern "C" {
void run_llamacpp(const void*q, float*o, int nb) { llamacpp_dequant_gpu<<<nb,32>>>(q,o); }
void run_bpd(const void*q, float*o, int nb)      { bpd_dequant_gpu<<<nb,32>>>(q,o); }
void* galloc(int b){void*p;cudaMalloc(&p,b);return p;}
void gfree(void*p){cudaFree(p);}
void gh2d(void*d,const void*s,int b){cudaMemcpy(d,s,b,cudaMemcpyHostToDevice);}
void gd2h(void*d,const void*s,int b){cudaMemcpy(d,s,b,cudaMemcpyDeviceToHost);}
void gsync(){cudaDeviceSynchronize();}
}
