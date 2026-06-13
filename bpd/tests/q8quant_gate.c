// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* q8quant_gate.c — 0-ULP gate for the q8_0 quantize atom (rung-3 crux).
 * Verifies @bpd_q8_0_quantize matches the ggml-EXACT activation quantization
 * byte-for-byte (quants) and bit-for-bit (scales). This is the FP-sensitive
 * step where chain composition bit-identity is non-trivial.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

extern void bpd_q8_0_quantize(const float* x, int8_t* q, float* d, int nb);

/* fp16 round-trip via the same path the IR uses (fptrunc/fpext to half).
 * Use compiler _Float16 if available; else emulate via uint16. Here we rely on
 * the host supporting __fp16 round; fall back to a manual round. */
static float fp16_round(float f){
    /* IEEE half round-to-nearest-even, matching llvm fptrunc+fpext */
    union { float f; uint32_t u; } in = { f };
    uint32_t x = in.u, sign = (x>>16)&0x8000;
    int32_t exp = ((x>>23)&0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    uint16_t h;
    if (exp <= 0){ h = sign; }
    else if (exp >= 0x1f){ h = sign | 0x7c00; }
    else {
        uint32_t m = man >> 13, r = man & 0x1fff;
        h = sign | (exp<<10) | m;
        if (r > 0x1000 || (r==0x1000 && (m&1))) h++;   /* round-half-even */
    }
    /* expand back to float */
    uint32_t hs=(h>>15)&1, he=(h>>10)&0x1f, hm=h&0x3ff, o;
    if (he==0){ if(!hm) o=hs<<31; else { he=113; while(!(hm&0x400)){hm<<=1;he--;} hm&=0x3ff; o=(hs<<31)|(he<<23)|(hm<<13);} }
    else if (he==0x1f) o=(hs<<31)|(0xff<<23)|(hm<<13);
    else o=(hs<<31)|((he+112)<<23)|(hm<<13);
    union { uint32_t u; float f; } out = { o };
    return out.f;
}

int main(int argc, char** argv){
    int nb = argc>1?atoi(argv[1]):64;
    int QK=32, n=nb*QK;
    float* x = malloc(n*4);
    for(int i=0;i<n;i++) x[i] = 0.5f*sinf(0.01f*i) + 0.001f*(i%97) - 0.3f;  /* varied magnitudes */

    int8_t* q = calloc(n,1);   float* d = calloc(nb,4);
    int8_t* rq= calloc(n,1);   float* rd= calloc(nb,4);

    /* IR quantize */
    bpd_q8_0_quantize(x, q, d, nb);

    /* CPU ggml-EXACT reference */
    for(int b=0;b<nb;b++){
        float amax=0; for(int j=0;j<QK;j++){ float a=fabsf(x[b*QK+j]); if(a>amax)amax=a; }
        float dd = amax/127.0f;
        float dh = fp16_round(dd);          /* stored scale = fp16(d) */
        rd[b] = dh;
        float id = (dd!=0.0f) ? 1.0f/dd : 0.0f;   /* id from fp32 d (ggml-exact) */
        for(int j=0;j<QK;j++) rq[b*QK+j] = (int8_t)roundf(x[b*QK+j]*id);
    }

    int qdiff=0, ddiff=0; float dmax=0;
    for(int i=0;i<n;i++) if(q[i]!=rq[i]) qdiff++;
    for(int b=0;b<nb;b++){ if(d[b]!=rd[b]){ ddiff++; float e=fabsf(d[b]-rd[b]); if(e>dmax)dmax=e; } }

    printf("q8_0 quantize gate (nb=%d, n=%d):\n", nb, n);
    printf("  quants: %d/%d differ %s\n", qdiff, n, qdiff==0?"*** 0 (byte-exact) ***":"DIVERGENT");
    printf("  scales: %d/%d differ (max_abs=%.3e) %s\n", ddiff, nb, dmax, ddiff==0?"*** 0 ULP ***":"DIVERGENT");
    free(x);free(q);free(d);free(rq);free(rd);
    return (qdiff==0 && ddiff==0) ? 0 : 1;
}
