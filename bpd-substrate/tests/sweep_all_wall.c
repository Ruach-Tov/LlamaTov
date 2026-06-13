// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef void (*unary_fn)(const float*, float*, int);
typedef void (*gemm_fn)(const float*, const float*, float*, int, int, int);

double bench(void (*run)(void), int reps) {
    run(); /* warmup */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < reps; i++) run();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return ((t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec)) / 1e6 / reps;
}

/* All substrate kernels */
extern void bpd_relu_cpu(const float*, float*, int);
extern void bpd_silu_cpu(const float*, float*, int);
extern void bpd_sigmoid_cpu(const float*, float*, int);
extern void bpd_tanh_cpu(const float*, float*, int);
extern void bpd_gelu_cpu(const float*, float*, int);
extern void bpd_mish_cpu(const float*, float*, int);
extern void bpd_leaky_relu_cpu(const float*, float*, int);
extern void bpd_elu_cpu(const float*, float*, int);
extern void bpd_selu_cpu(const float*, float*, int);
extern void bpd_neg_cpu(const float*, float*, int);
extern void bpd_abs_cpu(const float*, float*, int);
extern void bpd_hardsigmoid_cpu(const float*, float*, int);
extern void bpd_softplus_cpu(const float*, float*, int);
extern void bpd_add_f32_cpu(const float*, const float*, float*, int);
extern void bpd_mul_f32_cpu(const float*, const float*, float*, int);
extern void bpd_softmax_cpu(const float*, float*, int, int);
extern void bpd_gemm_v2_full(const float*, const float*, float*, int, int, int);
extern void bpd_mm_cpu_avx1(const float*, const float*, float*, int, int, int);
extern void bpd_mm_cpu_avx1_v2(const float*, const float*, float*, int, int, int);
extern void bpd_mm_cpu(const float*, const float*, float*, int, int, int);
extern void bpd_layernorm_cpu(const float*, const float*, const float*, float*, int, int, float);
extern void bpd_batchnorm_cpu_affine_fused(const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*, int, int, int, float);
extern void bpd_maxpool2d_cpu(const float*, float*, int, int, int, int, int, int, int, int);
extern void bpd_conv2d_full_cpu(const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int);
extern void bpd_linear_cpu(const float*, const float*, const float*, float*, int, int, int);

/* Globals for closures */
static const float *gx, *gx2, *gw, *gb;
static float *gy, *gc;
static int gn, gM, gN, gK;
static float *g_bn_g, *g_bn_b, *g_bn_m, *g_bn_v, *g_bn_s, *g_bn_o;
static float *g_ln_g, *g_ln_b;

#define MAKE_UNARY(NAME, FN) static void run_##NAME(void) { FN(gx, gy, gn); }
MAKE_UNARY(relu, bpd_relu_cpu)
MAKE_UNARY(silu, bpd_silu_cpu)
MAKE_UNARY(sigmoid, bpd_sigmoid_cpu)
MAKE_UNARY(tanh_, bpd_tanh_cpu)
MAKE_UNARY(gelu, bpd_gelu_cpu)
MAKE_UNARY(mish, bpd_mish_cpu)
MAKE_UNARY(leaky_relu, bpd_leaky_relu_cpu)
MAKE_UNARY(elu, bpd_elu_cpu)
MAKE_UNARY(selu, bpd_selu_cpu)
MAKE_UNARY(neg, bpd_neg_cpu)
MAKE_UNARY(abs_, bpd_abs_cpu)
MAKE_UNARY(hardsigmoid, bpd_hardsigmoid_cpu)
MAKE_UNARY(softplus, bpd_softplus_cpu)

static void run_add(void) { bpd_add_f32_cpu(gx, gx2, gy, gn); }
static void run_mul(void) { bpd_mul_f32_cpu(gx, gx2, gy, gn); }
static void run_softmax(void) { bpd_softmax_cpu(gx, gy, 1024, 1024); }
static void run_gemm_v2(void) { bpd_gemm_v2_full(gx, gx2, gc, gM, gN, gK); }
static void run_gemm_avx1(void) { bpd_mm_cpu_avx1(gx, gx2, gc, gM, gN, gK); }
static void run_gemm_v1(void) { bpd_mm_cpu_avx1_v2(gx, gx2, gc, gM, gN, gK); }
static void run_layernorm(void) { bpd_layernorm_cpu(gx, g_ln_g, g_ln_b, gy, 32, 512, 1e-5f); }
static void run_batchnorm(void) {
    bpd_batchnorm_cpu_affine_fused(gx, g_bn_g, g_bn_b, g_bn_m, g_bn_v,
        gy, g_bn_s, g_bn_o, 1, 64, 1024, 1e-5f);
}
static void run_maxpool(void) {
    bpd_maxpool2d_cpu(gx, gy, 1, 64, 64, 64, 2, 2, 2, 0);
}
static void run_conv2d(void) {
    bpd_conv2d_full_cpu(gx, gw, gb, gc, 1, 64, 64, 64, 128, 3, 3, 1, 1, 0, 0, 1, 1, 1);
}
static void run_linear(void) {
    bpd_linear_cpu(gx, gw, gb, gc, 32, 256, 512);
}

int main() {
    int n = 1000000;
    gn = n;
    gx = malloc(n*4); gx2 = malloc(n*4); gy = malloc(n*4);
    srand(42);
    for (int i = 0; i < n; i++) { ((float*)gx)[i] = (float)rand()/RAND_MAX*10-5; ((float*)gx2)[i] = (float)rand()/RAND_MAX*10-5; }

    /* Alloc for norms */
    g_bn_g = calloc(64,4); g_bn_b = calloc(64,4); g_bn_m = calloc(64,4);
    g_bn_v = malloc(64*4); g_bn_s = calloc(64,4); g_bn_o = calloc(64,4);
    for (int i=0;i<64;i++) g_bn_v[i]=1.0f;
    for (int i=0;i<64;i++) g_bn_g[i]=1.0f;
    g_ln_g = malloc(512*4); g_ln_b = calloc(512,4);
    for (int i=0;i<512;i++) g_ln_g[i]=1.0f;

    /* Alloc for GEMM/conv/linear */
    gM=1024; gN=1024; gK=1024;
    gw = malloc(128*64*3*3*4); gb = calloc(128,4);
    gc = calloc(gM*gN,4);
    for (int i=0;i<128*64*9;i++) ((float*)gw)[i]=(float)rand()/RAND_MAX*0.01f;

    printf("%%  =================================================================\n");
    printf("%%  cpu_sweep_facts.pl - Full parameter space profiling\n");
    printf("%%  Platform: Ivy Bridge, AVX1, no FMA, gcc -O2 -mavx\n");
    printf("%%  =================================================================\n\n");
    printf("%%  sweep_fact(Op, Strategy, Bottleneck, N, WallMs, NsPerElem).\n\n");

    struct { const char* name; void (*fn)(void); int reps; const char* strategy; const char* bottleneck; int elems; } all[] = {
        {"relu",        run_relu,        30, "avx1",   "memory",        n},
        {"silu",        run_silu,        20, "scalar", "transcendental",n},
        {"sigmoid",     run_sigmoid,     20, "scalar", "transcendental",n},
        {"tanh",        run_tanh_,       20, "scalar", "transcendental",n},
        {"gelu",        run_gelu,        20, "scalar", "transcendental",n},
        {"mish",        run_mish,        10, "scalar", "transcendental",n},
        {"leaky_relu",  run_leaky_relu,  30, "avx1",   "memory",        n},
        {"elu",         run_elu,         20, "scalar", "transcendental",n},
        {"selu",        run_selu,        20, "scalar", "transcendental",n},
        {"neg",         run_neg,         30, "avx1",   "memory",        n},
        {"abs",         run_abs_,        30, "avx1",   "memory",        n},
        {"hardsigmoid", run_hardsigmoid, 30, "avx1",   "memory",        n},
        {"softplus",    run_softplus,    10, "scalar", "transcendental",n},
        {"add",         run_add,         30, "avx1",   "memory",        n},
        {"mul",         run_mul,         30, "avx1",   "memory",        n},
        {"softmax",     run_softmax,     20, "scalar", "transcendental",1024*1024},
        {"layernorm",   run_layernorm,   30, "scalar", "memory",        32*512},
        {"batchnorm",   run_batchnorm,   30, "scalar", "memory",        64*1024},
        {"maxpool2d",   run_maxpool,     20, "scalar", "memory",        64*64*64},
        {"gemm",        run_gemm_avx1,    3, "avx1_mr16_nr4",    "compute", gM*gN},
        {"gemm",        run_gemm_v1,      3, "avx1_mr4_nr16",    "compute", gM*gN},
        {"gemm",        run_gemm_v2,      3, "tiled_v2_kblocked","compute", gM*gN},
        {"conv2d",      run_conv2d,       3, "tiled_v2",         "compute", 128*62*62},
        {"linear",      run_linear,      10, "tiled_v2_transB",  "compute", 32*256},
    };
    int N_ALL = sizeof(all)/sizeof(all[0]);

    for (int i = 0; i < N_ALL; i++) {
        double ms = bench(all[i].fn, all[i].reps);
        double ns_per_elem = ms * 1e6 / all[i].elems;
        printf("sweep_fact(%s, %s, %s, %d, %.3f, %.2f).\n",
               all[i].name, all[i].strategy, all[i].bottleneck,
               all[i].elems, ms, ns_per_elem);
    }

    printf("\n%%  === Summary by bottleneck class ===\n");
    double mem_total=0, trans_total=0, comp_total=0;
    int mem_n=0, trans_n=0, comp_n=0;
    for (int i = 0; i < N_ALL; i++) {
        double ms = bench(all[i].fn, all[i].reps);
        double ns = ms*1e6/all[i].elems;
        if (strcmp(all[i].bottleneck,"memory")==0) { mem_total+=ns; mem_n++; }
        else if (strcmp(all[i].bottleneck,"transcendental")==0) { trans_total+=ns; trans_n++; }
        else { comp_total+=ms; comp_n++; }
    }
    printf("%%  memory-bound:        avg %.1f ns/elem (%d ops)\n", mem_total/mem_n, mem_n);
    printf("%%  transcendental-bound: avg %.1f ns/elem (%d ops)\n", trans_total/trans_n, trans_n);
    printf("%%  compute-bound:        avg %.1f ms (%d configs)\n", comp_total/comp_n, comp_n);

    return 0;
}
