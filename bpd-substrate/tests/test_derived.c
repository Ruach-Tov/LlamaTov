// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Prolog-derived functions */
#include "/tmp/derived_activations.c"

/* Hand-written substrate kernels */
extern void bpd_relu_cpu(const float*, float*, int);
extern void bpd_silu_cpu(const float*, float*, int);
extern void bpd_sigmoid_cpu(const float*, float*, int);
extern void bpd_tanh_cpu(const float*, float*, int);
extern void bpd_gelu_cpu(const float*, float*, int);
extern void bpd_mish_cpu(const float*, float*, int);
extern void bpd_leaky_relu_cpu(const float*, float*, int);
extern void bpd_elu_cpu(const float*, float*, int);
extern void bpd_selu_cpu(const float*, float*, int);
extern void bpd_hardsigmoid_cpu(const float*, float*, int);
extern void bpd_softplus_cpu(const float*, float*, int);
extern void bpd_abs_cpu(const float*, float*, int);
extern void bpd_neg_cpu(const float*, float*, int);

int ulp_diff(float a, float b) {
    int32_t ai, bi;
    memcpy(&ai, &a, 4);
    memcpy(&bi, &b, 4);
    int64_t d = (int64_t)ai - (int64_t)bi;
    return (int)(d < 0 ? -d : d);
}

typedef void (*kernel_fn)(const float*, float*, int);
typedef float (*derived_fn)(float);

void test_one(const char* name, kernel_fn kern, derived_fn derived,
              const float* inputs, int n) {
    float* kern_out = calloc(n, sizeof(float));
    kern(inputs, kern_out, n);

    int max_ulp = 0, n_diffs = 0;
    for (int i = 0; i < n; i++) {
        float d = derived(inputs[i]);
        int u = ulp_diff(kern_out[i], d);
        if (u > max_ulp) max_ulp = u;
        if (u > 0) n_diffs++;
    }

    printf("  %-15s max_ulp=%-6d n_diffs=%d/%d  %s\n",
           name, max_ulp, n_diffs, n,
           max_ulp == 0 ? "BIT_IDENTICAL" : "DIVERGENT");
    free(kern_out);
}

int main() {
    int n = 10000;
    float* inputs = malloc(n * sizeof(float));
    srand(42);
    for (int i = 0; i < n; i++)
        inputs[i] = ((float)rand() / RAND_MAX - 0.5f) * 20.0f;

    printf("=== Prolog-Derived vs Hand-Written: %d test values ===\n\n", n);

    test_one("relu",        bpd_relu_cpu,        derived_relu,        inputs, n);
    test_one("silu",        bpd_silu_cpu,        derived_silu,        inputs, n);
    test_one("sigmoid",     bpd_sigmoid_cpu,     derived_sigmoid,     inputs, n);
    test_one("tanh",        bpd_tanh_cpu,        derived_tanh,        inputs, n);
    test_one("gelu",        bpd_gelu_cpu,        derived_gelu,        inputs, n);
    test_one("mish",        bpd_mish_cpu,        derived_mish,        inputs, n);
    test_one("leaky_relu",  bpd_leaky_relu_cpu,  derived_leaky_relu,  inputs, n);
    test_one("elu",         bpd_elu_cpu,         derived_elu,         inputs, n);
    test_one("selu",        bpd_selu_cpu,        derived_selu,        inputs, n);
    test_one("hardsigmoid", bpd_hardsigmoid_cpu, derived_hardsigmoid, inputs, n);
    test_one("abs",         bpd_abs_cpu,         derived_abs,         inputs, n);
    test_one("neg",         bpd_neg_cpu,         derived_neg,         inputs, n);
    test_one("softplus",    bpd_softplus_cpu,    derived_softplus,    inputs, n);

    free(inputs);
    return 0;
}
