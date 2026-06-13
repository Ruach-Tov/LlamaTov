// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* soa_ffn_gate.c — RUNG-4 0-ULP gate: a llama FFN block with SwiGLU fusion.
 * Verifies the IR FFN (@bpd_soa_ffn_q8_0) is bit-identical to a CPU reference
 * doing the IDENTICAL gate-gemv + up-gemv + SwiGLU + quantize + down-gemv with
 * the SAME atoms. 0-ULP here proves the SwiGLU fusion (dropped by tonight's
 * hand-kernel) composes bit-identically BY CONSTRUCTION.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

extern float bpd_q8_0_dot(const int8_t*, const float*, const int8_t*, const float*, int);
extern void  bpd_swiglu_fused_cpu(const float* gate, const float* up, float* out, int n);
extern void  bpd_q8_0_quantize(const float*, int8_t*, float*, int);
extern void  bpd_soa_ffn_q8_0(
    const int8_t* xq, const float* xd,
    const int8_t* wgq, const float* wgd, const int8_t* wuq, const float* wud,
    const int8_t* wdq, const float* wdd,
    float* gate, float* up, float* act, int8_t* aqd, float* add, float* out,
    int n_ff, int nb_in, int nb_ff, int n_embd);

static void fillq(int8_t* q, size_t n){ for(size_t i=0;i<n;i++) q[i]=(int8_t)((i*7919)%255-127); }
static void fills(float* s, size_t n){ for(size_t i=0;i<n;i++) s[i]=0.0008f+0.00002f*(float)(i%17); }

int main(int argc,char**argv){
    int n_embd = argc>1?atoi(argv[1]):2048;
    int n_ff   = argc>2?atoi(argv[2]):8192;
    int QK=32;
    int nb_in = n_embd/QK;          /* input blocks */
    int nb_ff = n_ff/QK;            /* act blocks */
    int ncols_in = n_embd;          /* gate/up have n_embd-wide rows */
    int ncols_ff = n_ff;            /* down has n_ff-wide rows */

    /* quantized normalized input (1 col) */
    int8_t* xq=malloc(ncols_in); float* xd=malloc(nb_in*4);
    fillq(xq,ncols_in); fills(xd,nb_in);
    /* weights */
    int8_t* wgq=malloc((size_t)n_ff*ncols_in); float* wgd=malloc((size_t)n_ff*nb_in*4);
    int8_t* wuq=malloc((size_t)n_ff*ncols_in); float* wud=malloc((size_t)n_ff*nb_in*4);
    int8_t* wdq=malloc((size_t)n_embd*ncols_ff); float* wdd=malloc((size_t)n_embd*nb_ff*4);
    fillq(wgq,(size_t)n_ff*ncols_in); fills(wgd,(size_t)n_ff*nb_in);
    fillq(wuq,(size_t)n_ff*ncols_in); fills(wud,(size_t)n_ff*nb_in);
    fillq(wdq,(size_t)n_embd*ncols_ff); fills(wdd,(size_t)n_embd*nb_ff);

    /* IR FFN */
    float* gate=calloc(n_ff,4); float* up=calloc(n_ff,4); float* act=calloc(n_ff,4);
    int8_t* aqd=calloc(ncols_ff,1); float* add=calloc(nb_ff,4); float* out=calloc(n_embd,4);
    bpd_soa_ffn_q8_0(xq,xd, wgq,wgd,wuq,wud, wdq,wdd,
                     gate,up,act,aqd,add,out, n_ff,nb_in,nb_ff,n_embd);

    /* CPU reference: identical sequence with same atoms */
    float* r_gate=calloc(n_ff,4); float* r_up=calloc(n_ff,4); float* r_act=calloc(n_ff,4);
    int8_t* r_aqd=calloc(ncols_ff,1); float* r_add=calloc(nb_ff,4); float* r_out=calloc(n_embd,4);
    for(int r=0;r<n_ff;r++) r_gate[r]=bpd_q8_0_dot(wgq+(size_t)r*ncols_in, wgd+(size_t)r*nb_in, xq, xd, nb_in);
    for(int r=0;r<n_ff;r++) r_up[r]  =bpd_q8_0_dot(wuq+(size_t)r*ncols_in, wud+(size_t)r*nb_in, xq, xd, nb_in);
    bpd_swiglu_fused_cpu(r_gate, r_up, r_act, n_ff);
    bpd_q8_0_quantize(r_act, r_aqd, r_add, nb_ff);
    for(int r=0;r<n_embd;r++) r_out[r]=bpd_q8_0_dot(wdq+(size_t)r*ncols_ff, wdd+(size_t)r*nb_ff, r_aqd, r_add, nb_ff);

    /* 0-ULP at each stage (localize any divergence) */
    int gd=0,ad=0,od=0; float omax=0; int worst=-1;
    for(int r=0;r<n_ff;r++){ if(gate[r]!=r_gate[r]) gd++; if(act[r]!=r_act[r]) ad++; }
    for(int r=0;r<n_embd;r++){ if(out[r]!=r_out[r]){ od++; float e=fabsf(out[r]-r_out[r]); if(e>omax){omax=e;worst=r;} } }

    printf("RUNG-4 FFN gate: n_embd=%d n_ff=%d  (out = Wd·quant(silu(Wg·x)·(Wu·x)))\n", n_embd, n_ff);
    printf("  gate (Wg·x):          %d/%d differ %s\n", gd, n_ff, gd==0?"0 ULP":"DIVERGENT");
    printf("  act  (SwiGLU fusion): %d/%d differ %s\n", ad, n_ff, ad==0?"*** 0 ULP ***":"DIVERGENT");
    printf("  out  (Wd·quant(act)): %d/%d differ (max_abs=%.3e worst %d) %s\n",
           od, n_embd, omax, worst, od==0?"*** 0 ULP ***":"DIVERGENT");
    return (gd==0 && ad==0 && od==0) ? 0 : 1;
}
