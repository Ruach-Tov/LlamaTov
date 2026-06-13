// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* soa_gemv_gate.c — RUNG-2 0-ULP gate for the SoA Q8_0 gemv emitter.
 *
 * Verifies that the EMITTED gemv IR (@bpd_soa_gemv_q8_0) produces output
 * BIT-IDENTICAL (0 ULP) to a CPU reference that loops the SAME verified
 * block-dot (@bpd_q8_0_dot) over rows × columns.
 *
 * The discipline (Heath's "0 ULP for each rung"): a rung is verified by
 * COMPOSITION over the already-verified atom. Since both the IR gemv and the
 * CPU reference call the IDENTICAL @bpd_q8_0_dot, any difference is a
 * COMPOSITION error (wrong stride / index / dst placement), not an FP-order
 * difference. So 0-ULP here proves the gemv composes the dot correctly.
 *
 * Tests BOTH shapes (the trap that hid the decode bug):
 *   ncols_dst=1 (decode)   ncols_dst=2 (prefill)
 *
 * Linked with: q8dot.o (from emit_q8_0_dot) + soa_gemv_d{1,2}.o (from emit_soa_gemv).
 *
 * Author: Iyun, 2026-06-06 (rung-2 of the bottom-up SoA chain)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* the Prolog-generated block-dot (the verified atom) */
extern float bpd_q8_0_dot(const int8_t* wq, const float* wd,
                          const int8_t* aq, const float* ad, int nb);

/* the emitted SoA gemv (two shape-specialized objects share the same symbol;
 * we build/link them separately and call via a function pointer per shape) */
extern void bpd_soa_gemv_q8_0(const int8_t* wq, const float* wd,
                              const int8_t* aq, const float* ad,
                              float* dst, int nrows, int nb);

/* deterministic synthetic data (same generator the shape-sweep used) */
static void fill_quants(int8_t* q, size_t n){ for(size_t i=0;i<n;i++) q[i]=(int8_t)((i*7919)%255-127); }
static void fill_scales(float* s, size_t n){ for(size_t i=0;i<n;i++) s[i]=0.01f+0.001f*(float)(i%17); }

int main(int argc, char** argv){
    int nrows = argc>1?atoi(argv[1]):2048;
    int ncols = argc>2?atoi(argv[2]):2048;          /* quant lanes per row */
    int ncols_dst = argc>3?atoi(argv[3]):1;          /* 1=decode, 2=prefill */
    int QK=32, nb=ncols/QK;

    size_t wq_n=(size_t)nrows*ncols, wd_n=(size_t)nrows*nb;
    size_t aq_n=(size_t)ncols_dst*ncols, ad_n=(size_t)ncols_dst*nb;
    int8_t *wq=malloc(wq_n), *aq=malloc(aq_n);
    float  *wd=malloc(wd_n*4), *ad=malloc(ad_n*4);
    fill_quants(wq,wq_n); fill_quants(aq,aq_n);
    fill_scales(wd,wd_n);  fill_scales(ad,ad_n);

    int N=nrows*ncols_dst;
    float *gpu=calloc(N,4), *ref=calloc(N,4);

    /* CPU reference: loop the SAME verified dot over rows × columns.
     * dst[j*nrows + r] = bpd_q8_0_dot(W_row_r, A_col_j)  — SoA strides. */
    for(int j=0;j<ncols_dst;j++)
        for(int r=0;r<nrows;r++)
            ref[(size_t)j*nrows+r] = bpd_q8_0_dot(
                wq+(size_t)r*ncols, wd+(size_t)r*nb,
                aq+(size_t)j*ncols, ad+(size_t)j*nb, nb);

    /* the emitted IR gemv: it does the column loop internally for ncols_dst.
     * Our two objects are specialized per ncols_dst; here we call the linked one. */
    bpd_soa_gemv_q8_0(wq, wd, aq, ad, gpu, nrows, nb);

    /* 0-ULP comparison, element-wise */
    int ndiff=0; float maxabs=0; int worst=-1;
    for(int i=0;i<N;i++){
        if(ref[i]!=gpu[i]){ ndiff++; float d=fabsf(ref[i]-gpu[i]); if(d>maxabs){maxabs=d;worst=i;} }
    }
    printf("RUNG-2 gemv gate: nrows=%d ncols=%d nb=%d ncols_dst=%d (%s)\n",
           nrows,ncols,nb,ncols_dst, ncols_dst==1?"DECODE":"PREFILL");
    printf("  IR-gemv vs CPU-loop-of-dot: %d/%d differ, max_abs=%.3e (worst idx %d)  -> %s\n",
           ndiff, N, maxabs, worst, ndiff==0 ? "*** 0 ULP ***" : "DIVERGENT");
    free(wq);free(wd);free(aq);free(ad);free(gpu);free(ref);
    return ndiff==0 ? 0 : 1;
}
