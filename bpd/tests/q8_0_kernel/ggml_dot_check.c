// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// ggml_dot_check.c — call ggml's ACTUAL q8_0 dot on weight-row0 . activation-tok0.
// Reads: 2048 weight q8_0 bytes? No — simpler: read raw weight blocks (64*34 bytes) from a file,
// and 2048 activation floats from stdin. Quantize activation with ggml's quantize_row_q8_0,
// then call ggml_vec_dot_q8_0_q8_0. Print result with full precision.
//
// Build: clang -mavx -mf16c -mno-avx2 -mno-fma -O2 ggml_dot_check.c \
//   -I/tmp/llama_cpp_test/ggml/include -I/tmp/llama_cpp_test/ggml/src \
//   -I/tmp/llama_cpp_test/ggml/src/ggml-cpu \
//   -L/tmp/llama_cpp_test/build/bin -lggml-cpu -lggml-base -o ggml_dot_check
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-quants.h"

#define QK8_0 32
#define K 2048
#define NB 64

typedef struct { uint16_t d; int8_t qs[QK8_0]; } blk_q8_0;

int main(int argc, char** argv){
    if(argc<2){ fprintf(stderr,"usage: %s weight_blocks.bin < activation.txt\n",argv[0]); return 1; }
    // read 64*34 = 2176 weight bytes (already q8_0)
    FILE* wf=fopen(argv[1],"rb");
    static blk_q8_0 wblk[NB];
    if(fread(wblk,34,NB,wf)!=NB){ fprintf(stderr,"weight read fail\n"); return 1; }
    fclose(wf);
    // read 2048 activation floats from stdin
    static float act[K];
    for(int i=0;i<K;i++){ if(scanf("%f",&act[i])!=1){ fprintf(stderr,"act read fail @%d\n",i); return 1; } }
    // quantize activation with ggml's reference quantizer
    static blk_q8_0 ablk[NB];
    quantize_row_q8_0(act, (void*)ablk, K);
    // call ggml's q8_0 dot
    ggml_cpu_init();
    fprintf(stderr,"avx=%d f16c=%d avx2=%d fma=%d\n", ggml_cpu_has_avx(), ggml_cpu_has_f16c(), ggml_cpu_has_avx2(), ggml_cpu_has_fma());
    // DUMP all quants/scales for comparison
    FILE* df=fopen("/tmp/ggml_aq.txt","w");
    for(int i=0;i<NB;i++){ fprintf(df,"%04x",ablk[i].d); for(int j=0;j<QK8_0;j++) fprintf(df," %d",ablk[i].qs[j]); fprintf(df,"\n"); }
    fclose(df);
    float result=0.0f;
    ggml_vec_dot_q8_0_q8_0(K, &result, 0, (void*)wblk, 0, (void*)ablk, 0, 1);
    printf("ggml_vec_dot_q8_0_q8_0 = %.17g\n", result);
    // also print first few activation quants for cross-check
    printf("act blk0 d=0x%04x qs[:8]=", ablk[0].d);
    for(int j=0;j<8;j++) printf("%d ", ablk[0].qs[j]);
    printf("\n");
    return 0;
}
