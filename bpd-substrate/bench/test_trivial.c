// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* test_trivial.c
 * Hand-verifiable correctness tests for bpd_cpu.c matmul kernels.
 * Every expected value is computed by hand or is trivially exact in float32.
 *
 * Build:
 *   gcc -O0 -o test_trivial test_trivial.c ../build/bpd_cpu.so -lm -Wl,-rpath,../build
 *   (or link bpd_cpu.c directly)
 *
 * All tests print PASS or FAIL with the actual vs expected values.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ── Declarations matching bpd_cpu.c exports ── */
extern void bpd_mm_cpu(const float *A, const float *B, float *C,
                        int M, int N, int K);
extern void bpd_mm_bias_relu_cpu(const float *A, const float *B,
                                  const float *bias, float *C,
                                  int M, int N, int K);

/* ── Helpers ── */
static int pass_count = 0, fail_count = 0;

static void check_f32(const char *name, float got, float expected) {
    if (got == expected) {
        printf("  PASS  %-40s  got=%.8g\n", name, got);
        pass_count++;
    } else {
        printf("  FAIL  %-40s  got=%.8g  expected=%.8g  diff=%.3g\n",
               name, got, expected, got - expected);
        fail_count++;
    }
}

static void check_array(const char *name, const float *got,
                         const float *expected, int n) {
    int all_ok = 1;
    for (int i = 0; i < n; i++) {
        if (got[i] != expected[i]) { all_ok = 0; break; }
    }
    if (all_ok) {
        printf("  PASS  %s\n", name);
        pass_count++;
    } else {
        printf("  FAIL  %s\n", name);
        for (int i = 0; i < n; i++) {
            if (got[i] != expected[i])
                printf("        [%d] got=%.8g  expected=%.8g\n",
                       i, got[i], expected[i]);
        }
        fail_count++;
    }
}

/* ── Test cases ── */

/* Test 1: 1x1 matmul — trivially C[0,0] = A[0,0] * B[0,0] */
static void test_1x1(void) {
    printf("\n--- Test 1: 1x1 matmul ---\n");
    float A[1] = {3.0f};
    float B[1] = {4.0f};
    float C[1] = {0.0f};
    bpd_mm_cpu(A, B, C, 1, 1, 1);
    check_f32("1x1: 3*4=12", C[0], 12.0f);
}

/* Test 2: 1x1 with negative */
static void test_1x1_neg(void) {
    printf("\n--- Test 2: 1x1 negative ---\n");
    float A[1] = {-2.0f};
    float B[1] = {5.0f};
    float C[1] = {0.0f};
    bpd_mm_cpu(A, B, C, 1, 1, 1);
    check_f32("1x1: -2*5=-10", C[0], -10.0f);
}

/* Test 3: 2x2 identity matrix — A @ I = A */
static void test_2x2_identity(void) {
    printf("\n--- Test 3: 2x2 A @ I = A ---\n");
    float A[4] = {1.0f, 2.0f,
                  3.0f, 4.0f};
    float I[4] = {1.0f, 0.0f,
                  0.0f, 1.0f};
    float C[4] = {0};
    bpd_mm_cpu(A, I, C, 2, 2, 2);
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    check_array("2x2 A@I = A", C, expected, 4);
}

/* Test 4: 2x2 I @ A = A */
static void test_2x2_identity_left(void) {
    printf("\n--- Test 4: 2x2 I @ A = A ---\n");
    float I[4] = {1.0f, 0.0f,
                  0.0f, 1.0f};
    float A[4] = {5.0f, 6.0f,
                  7.0f, 8.0f};
    float C[4] = {0};
    bpd_mm_cpu(I, A, C, 2, 2, 2);
    check_array("2x2 I@A = A", C, A, 4);
}

/* Test 5: 2x2 zero matrix — A @ 0 = 0 */
static void test_2x2_zero(void) {
    printf("\n--- Test 5: 2x2 A @ 0 = 0 ---\n");
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float Z[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float C[4] = {0};
    bpd_mm_cpu(A, Z, C, 2, 2, 2);
    float expected[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    check_array("2x2 A@0 = 0", C, expected, 4);
}

/* Test 6: 2x2 known product
 * A = [[1,2],[3,4]], B = [[5,6],[7,8]]
 * C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
 * C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
 * C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
 * C[1,1] = 3*6 + 4*8 = 18 + 32 = 50
 */
static void test_2x2_known(void) {
    printf("\n--- Test 6: 2x2 known product ---\n");
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[4] = {0};
    bpd_mm_cpu(A, B, C, 2, 2, 2);
    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    check_array("2x2 [[1,2],[3,4]] @ [[5,6],[7,8]]", C, expected, 4);
    check_f32("  C[0,0]=19", C[0], 19.0f);
    check_f32("  C[0,1]=22", C[1], 22.0f);
    check_f32("  C[1,0]=43", C[2], 43.0f);
    check_f32("  C[1,1]=50", C[3], 50.0f);
}

/* Test 7: 3x3 known product (all ones)
 * A = B = ones(3,3)
 * C[i,j] = sum_k 1*1 = 3 for all i,j
 */
static void test_3x3_ones(void) {
    printf("\n--- Test 7: 3x3 all-ones product ---\n");
    float A[9], B[9], C[9] = {0};
    for (int i = 0; i < 9; i++) A[i] = B[i] = 1.0f;
    bpd_mm_cpu(A, B, C, 3, 3, 3);
    float expected[9];
    for (int i = 0; i < 9; i++) expected[i] = 3.0f;
    check_array("3x3 ones @ ones = 3*ones", C, expected, 9);
}

/* Test 8: 4x4 all-ones — C[i,j] = 4 */
static void test_4x4_ones(void) {
    printf("\n--- Test 8: 4x4 all-ones product ---\n");
    float A[16], B[16], C[16] = {0};
    for (int i = 0; i < 16; i++) A[i] = B[i] = 1.0f;
    bpd_mm_cpu(A, B, C, 4, 4, 4);
    float expected[16];
    for (int i = 0; i < 16; i++) expected[i] = 4.0f;
    check_array("4x4 ones @ ones = 4*ones", C, expected, 16);
}

/* Test 9: Non-square — 2x3 @ 3x2
 * A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
 * C[0,0] = 1*7+2*9+3*11 = 7+18+33 = 58
 * C[0,1] = 1*8+2*10+3*12 = 8+20+36 = 64
 * C[1,0] = 4*7+5*9+6*11 = 28+45+66 = 139
 * C[1,1] = 4*8+5*10+6*12 = 32+50+72 = 154
 */
static void test_nonsquare_2x3x2(void) {
    printf("\n--- Test 9: 2x3 @ 3x2 non-square ---\n");
    float A[6] = {1,2,3,4,5,6};
    float B[6] = {7,8,9,10,11,12};
    float C[4] = {0};
    bpd_mm_cpu(A, B, C, 2, 2, 3);
    float expected[4] = {58.0f, 64.0f, 139.0f, 154.0f};
    check_array("2x3 @ 3x2", C, expected, 4);
    check_f32("  C[0,0]=58",  C[0], 58.0f);
    check_f32("  C[0,1]=64",  C[1], 64.0f);
    check_f32("  C[1,0]=139", C[2], 139.0f);
    check_f32("  C[1,1]=154", C[3], 154.0f);
}

/* Test 10: 1xN dot product — row @ column = scalar
 * A = [1,2,3,4], B = [[1],[2],[3],[4]]
 * result = 1+4+9+16 = 30
 */
static void test_dot_product(void) {
    printf("\n--- Test 10: 1x4 dot product ---\n");
    float A[4] = {1,2,3,4};
    float B[4] = {1,2,3,4};  /* column vector as 4x1 */
    float C[1] = {0};
    bpd_mm_cpu(A, B, C, 1, 1, 4);
    check_f32("1x4 @ 4x1 = 30", C[0], 30.0f);
}

/* Test 11: Powers of 2 — exact in float32
 * A = diag(2,4,8), B = diag(2,4,8)
 * C = diag(4,16,64)
 */
static void test_powers_of_2(void) {
    printf("\n--- Test 11: diagonal powers of 2 ---\n");
    float A[9] = {2,0,0, 0,4,0, 0,0,8};
    float B[9] = {2,0,0, 0,4,0, 0,0,8};
    float C[9] = {0};
    bpd_mm_cpu(A, B, C, 3, 3, 3);
    float expected[9] = {4,0,0, 0,16,0, 0,0,64};
    check_array("diag(2,4,8)^2 = diag(4,16,64)", C, expected, 9);
}

/* Test 12: Large all-ones — N=32, C[i,j] = 32 (exact in float32) */
static void test_32x32_ones(void) {
    printf("\n--- Test 12: 32x32 all-ones ---\n");
    float A[1024], B[1024], C[1024] = {0};
    for (int i = 0; i < 1024; i++) A[i] = B[i] = 1.0f;
    bpd_mm_cpu(A, B, C, 32, 32, 32);
    float expected[1024];
    for (int i = 0; i < 1024; i++) expected[i] = 32.0f;
    check_array("32x32 ones @ ones = 32*ones", C, expected, 1024);
}

/* Test 13: Large all-ones — N=64, C[i,j] = 64 (exact in float32) */
static void test_64x64_ones(void) {
    printf("\n--- Test 13: 64x64 all-ones ---\n");
    float A[4096], B[4096], C[4096] = {0};
    for (int i = 0; i < 4096; i++) A[i] = B[i] = 1.0f;
    bpd_mm_cpu(A, B, C, 64, 64, 64);
    float expected[4096];
    for (int i = 0; i < 4096; i++) expected[i] = 64.0f;
    check_array("64x64 ones @ ones = 64*ones", C, expected, 4096);
}

/* Test 14: 512x512 all-ones — C[i,j] = 512 (exact in float32) */
static void test_512x512_ones(void) {
    printf("\n--- Test 14: 512x512 all-ones ---\n");
    static float A[512*512], B[512*512], C[512*512];
    for (int i = 0; i < 512*512; i++) A[i] = B[i] = 1.0f;
    memset(C, 0, sizeof(C));
    bpd_mm_cpu(A, B, C, 512, 512, 512);
    int ok = 1;
    for (int i = 0; i < 512*512; i++) {
        if (C[i] != 512.0f) { ok = 0; break; }
    }
    if (ok) { printf("  PASS  512x512 ones @ ones = 512*ones\n"); pass_count++; }
    else {
        printf("  FAIL  512x512 ones @ ones: first bad element: ");
        for (int i = 0; i < 512*512; i++)
            if (C[i] != 512.0f) { printf("C[%d]=%.8g\n", i, C[i]); break; }
        fail_count++;
    }
}

/* Test 15: mm_bias_relu — trivial 1x1
 * C = relu(A@B + bias) = relu(3*4 + 2) = relu(14) = 14
 */
static void test_bias_relu_1x1(void) {
    printf("\n--- Test 15: 1x1 mm_bias_relu ---\n");
    float A[1] = {3.0f};
    float B[1] = {4.0f};
    float bias[1] = {2.0f};
    float C[1] = {0};
    bpd_mm_bias_relu_cpu(A, B, bias, C, 1, 1, 1);
    check_f32("relu(3*4+2)=14", C[0], 14.0f);
}

/* Test 16: mm_bias_relu — negative result clamped to 0
 * C = relu((-5)*3 + 1) = relu(-14) = 0
 */
static void test_bias_relu_clamp(void) {
    printf("\n--- Test 16: mm_bias_relu clamp to 0 ---\n");
    float A[1] = {-5.0f};
    float B[1] = {3.0f};
    float bias[1] = {1.0f};
    float C[1] = {0};
    bpd_mm_bias_relu_cpu(A, B, bias, C, 1, 1, 1);
    check_f32("relu(-5*3+1)=0", C[0], 0.0f);
}

/* Test 17: 2x2 mm_bias_relu
 * A = [[1,0],[0,1]], B = [[2,3],[4,5]], bias = [-10, 0]
 * A@B = [[2,3],[4,5]]
 * +bias row0: [2-10, 3+0] = [-8, 3]
 * +bias row1: [4-10, 5+0] = [-6, 5]
 * relu: [0, 3, 0, 5]
 */
static void test_bias_relu_2x2(void) {
    printf("\n--- Test 17: 2x2 mm_bias_relu with mixed clamp ---\n");
    float A[4] = {1,0, 0,1};
    float B[4] = {2,3, 4,5};
    float bias[2] = {-10.0f, 0.0f};
    float C[4] = {0};
    bpd_mm_bias_relu_cpu(A, B, bias, C, 2, 2, 2);
    float expected[4] = {0.0f, 3.0f, 0.0f, 5.0f};
    check_array("2x2 mm_bias_relu", C, expected, 4);
}

/* Test 18: Outer product — 1-hot vectors
 * A = [1,0,0,0] (4x1), B = [0,0,1,0] (1x4)
 * C should be a 4x4 matrix with only C[0,2]=1, all others 0
 */
static void test_outer_product(void) {
    printf("\n--- Test 18: outer product 4x1 @ 1x4 ---\n");
    float A[4] = {1,0,0,0};
    float B[4] = {0,0,1,0};
    float C[16] = {0};
    bpd_mm_cpu(A, B, C, 4, 4, 1);
    float expected[16] = {0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    check_array("outer product e0 @ e2^T", C, expected, 16);
}

int main(void) {
    printf("=== BPD CPU Matmul Trivial Correctness Tests ===\n");

    test_1x1();
    test_1x1_neg();
    test_2x2_identity();
    test_2x2_identity_left();
    test_2x2_zero();
    test_2x2_known();
    test_3x3_ones();
    test_4x4_ones();
    test_nonsquare_2x3x2();
    test_dot_product();
    test_powers_of_2();
    test_32x32_ones();
    test_64x64_ones();
    test_512x512_ones();
    test_bias_relu_1x1();
    test_bias_relu_clamp();
    test_bias_relu_2x2();
    test_outer_product();

    printf("\n=== Summary: %d PASSED, %d FAILED ===\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
