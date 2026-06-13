// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* soa_chain_gate.c — RUNG-3 0-ULP gate: dst = W2 · quantize(W1 · x).
 * Verifies the IR chain (@bpd_soa_chain_q8_0) is bit-identical to a CPU
 * reference doing the IDENTICAL gemv1 → quantize → gemv2 with the SAME atoms.
 * 0-ULP here proves bit-identity SURVIVES composition across matmuls — the
 * property the full-model claim rests on.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

extern float bpd_q8_0_dot(const int8_t*, const float*, const int8_t*, const float*, int);
extern void  bpd_soa_gemv_q8_0(const int8_t*, const float*, const int8_t*, const float*, float*, int, int);
extern void  bpd_q8_0_quantize(const float*, int8_t*, float*, int);
extern void  bpd_soa_chain_q8_0(const int8_t* w1q, const float* w1d, const int8_t* xq, const float* xd,
                                float* y1, int8_t* aq2, float* ad2,
                                const int8_t* w2q, const float* w2d, float* dst,
                                int nrows1, int nb1, int nrows2, int nb2);

static void fillq(int8_t* q, size_t n){ for(size_t i=0;i<n;i++) q[i]=(int8_t)((i*7919)%255-127); }
static void fills(float* s, size_t n){ for(size_t i=0;i<n;i++) s[i]=0.01f+0.001f*(float)(i%17); }

int main(int argc,char**argv){
    /* W1: nrows1 × (nb1*32);  x: 1 col of nb1 blocks.  y1: nrows1 floats.
       quantize y1 → nb2 = nrows1/32 blocks.  W2: nrows2 × (nb2*32). */
    int nb1 = argc>1?atoi(argv[1]):64;     /* x has nb1 blocks (ncols1 = nb1*32) */
    int nrows1 = argc>2?atoi(argv[2]):2048;
    int nrows2 = argc>3?atoi(argv[3]):2048;
    int QK=32;
    int ncols1 = nb1*QK;
    int nb2 = nrows1/QK;                    /* y1 (nrows1) re-quantized into nb2 blocks */
    int ncols2 = nb2*QK;                    /* = nrows1 */

    /* W1 */
    int8_t* w1q=malloc((size_t)nrows1*ncols1); float* w1d=malloc((size_t)nrows1*nb1*4);
    fillq(w1q,(size_t)nrows1*ncols1); fills(w1d,(size_t)nrows1*nb1);
    /* x (1 column) */
    int8_t* xq=malloc(ncols1); float* xd=malloc(nb1*4);
    fillq(xq,ncols1); fills(xd,nb1);
    /* W2 */
    int8_t* w2q=malloc((size_t)nrows2*ncols2); float* w2d=malloc((size_t)nrows2*nb2*4);
    fillq(w2q,(size_t)nrows2*ncols2); fills(w2d,(size_t)nrows2*nb2);

    /* IR chain */
    float* y1=calloc(nrows1,4); int8_t* aq2=calloc(ncols2,1); float* ad2=calloc(nb2,4);
    float* dst=calloc(nrows2,4);
    bpd_soa_chain_q8_0(w1q,w1d,xq,xd, y1,aq2,ad2, w2q,w2d,dst, nrows1,nb1,nrows2,nb2);

    /* CPU reference: identical gemv1 → quantize → gemv2 with the SAME atoms */
    float* r_y1=calloc(nrows1,4); int8_t* r_aq2=calloc(ncols2,1); float* r_ad2=calloc(nb2,4);
    float* r_dst=calloc(nrows2,4);
    for(int r=0;r<nrows1;r++)
        r_y1[r]=bpd_q8_0_dot(w1q+(size_t)r*ncols1, w1d+(size_t)r*nb1, xq, xd, nb1);
    bpd_q8_0_quantize(r_y1, r_aq2, r_ad2, nb2);
    for(int r=0;r<nrows2;r++)
        r_dst[r]=bpd_q8_0_dot(w2q+(size_t)r*ncols2, w2d+(size_t)r*nb2, r_aq2, r_ad2, nb2);

    /* 0-ULP comparison of final dst (and the intermediate, to localize) */
    int idiff=0,ddiff=0; float dmax=0; int worst=-1;
    for(int r=0;r<nrows1;r++) if(y1[r]!=r_y1[r]) idiff++;
    for(int r=0;r<nrows2;r++){ if(dst[r]!=r_dst[r]){ ddiff++; float e=fabsf(dst[r]-r_dst[r]); if(e>dmax){dmax=e;worst=r;} } }

    printf("RUNG-3 chain gate: nb1=%d nrows1=%d nrows2=%d (dst = W2 · quantize(W1 · x))\n", nb1,nrows1,nrows2);
    printf("  intermediate y1: %d/%d differ %s\n", idiff, nrows1, idiff==0?"*** 0 ULP ***":"DIVERGENT");
    printf("  final dst:       %d/%d differ (max_abs=%.3e worst %d) %s\n",
           ddiff, nrows2, dmax, worst, ddiff==0?"*** 0 ULP ***":"DIVERGENT");
    return (idiff==0 && ddiff==0) ? 0 : 1;
}
