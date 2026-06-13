// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* SIMD-aware OpenBLAS Sandybridge SGEMM port.
 *
 * Per Heath's direction 2026-05-20 ~22:50 UTC: "(2) Match it by setting
 * sweepable parameters. Surpass it by sweeping the parameter space to
 * extremes OpenBLAS never found."
 *
 * Today (Day 1 of 3): hand-crafted single-instantiation SIMD kernel matching
 * OpenBLAS's KERNEL16x4_SUB exactly. Goal: prove we can get OpenBLAS-class
 * GFLOPS at 0 ULP. Then generalize the generator in subsequent commits.
 *
 * Algorithm (matches OpenBLAS sgemm_kernel_16x4_sandy.S KERNEL16x4_SUB):
 *
 *   Pack A: row-major (M, K) → blocked (M/16, K, 16) with 16-row contiguous chunks
 *   Pack B: row-major (K, N) → blocked (N/4, K, 4)  with 4-col contiguous chunks
 *   K-block: adaptive_half (rem>=2Q: Q; rem>Q: ceil(rem/2/UM)*UM; else rem) with Q=384
 *
 *   For each 16x4 C tile:
 *     For each K-block:
 *       For k in [k_start..k_end):
 *         ymm0 = packed_A[16*k .. 16*k+7]
 *         ymm1 = packed_A[16*k+8 .. 16*k+15]
 *         For col in 0..3:
 *           ymm_b = broadcast packed_B[4*k + col]
 *           ymm_C_top[col] += ymm0 * ymm_b
 *           ymm_C_bot[col] += ymm1 * ymm_b
 *       Store ymm_C[*] to C[16x4 tile] (read-modify-write with existing C).
 *
 * For (M=N=16, K=4096): one 16x4 tile pass × 4 N-panels per K-block × 11 K-blocks.
 *
 * Compile: gcc -O3 -mavx -shared -fPIC -o build/bpd_mm_simd.so bench/bpd_mm_simd.c
 */
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define Q_BLOCK 384
#define UM 16
#define UN 4

/* Pack A panel: rows [i_start..i_start+UM) × K → packed_A[UM*K] in 16-row major.
 * packed_A[k*UM + r] = A[(i_start + r) * K_orig + k_start + k] for r in 0..UM, k in 0..K_block.
 */
static void pack_a_panel(const float* A_src, int K_orig, int i_start, int k_start, int k_block,
                          float* packed_A) {
    for (int k = 0; k < k_block; ++k) {
        for (int r = 0; r < UM; ++r) {
            packed_A[k * UM + r] = A_src[(i_start + r) * K_orig + (k_start + k)];
        }
    }
}

/* Pack B panel: K × cols [j_start..j_start+UN) → packed_B[UN*K] in 4-col major.
 * packed_B[k*UN + c] = B[(k_start + k) * N_orig + j_start + c].
 */
static void pack_b_panel(const float* B_src, int N_orig, int k_start, int j_start, int k_block,
                          float* packed_B) {
    for (int k = 0; k < k_block; ++k) {
        for (int c = 0; c < UN; ++c) {
            packed_B[k * UN + c] = B_src[(k_start + k) * N_orig + (j_start + c)];
        }
    }
}

/* Inner micro-kernel: 16x4 tile, K-block iterations.
 * Adds the partial to C in place (read-modify-write).
 *
 * Mirrors OpenBLAS KERNEL16x4_SUB + SAVE16x4 exactly.
 */
static void kernel_16x4(const float* packed_A, const float* packed_B,
                         float* C, int N, int k_block) {
    /* 8 ymm registers for C: 2 (top/bot rows) × 4 (cols) = 8.
     * ymm4_top, ymm4_bot for col 0; ymm6_top, ymm6_bot for col 1; etc.
     */
    __m256 c00 = _mm256_setzero_ps();  /* C[0..7][0]   */
    __m256 c01 = _mm256_setzero_ps();  /* C[0..7][1]   */
    __m256 c02 = _mm256_setzero_ps();  /* C[0..7][2]   */
    __m256 c03 = _mm256_setzero_ps();  /* C[0..7][3]   */
    __m256 c10 = _mm256_setzero_ps();  /* C[8..15][0]  */
    __m256 c11 = _mm256_setzero_ps();  /* C[8..15][1]  */
    __m256 c12 = _mm256_setzero_ps();  /* C[8..15][2]  */
    __m256 c13 = _mm256_setzero_ps();  /* C[8..15][3]  */

    for (int k = 0; k < k_block; ++k) {
        __m256 a_top = _mm256_loadu_ps(packed_A + k * UM);       /* A[0..7]  */
        __m256 a_bot = _mm256_loadu_ps(packed_A + k * UM + 8);   /* A[8..15] */

        /* For each of the 4 B columns, broadcast and accumulate.
         * Order matches OpenBLAS: b0, b1 first (cols 0, 1), then b2, b3.
         */
        __m256 b0 = _mm256_broadcast_ss(packed_B + k * UN + 0);
        __m256 b1 = _mm256_broadcast_ss(packed_B + k * UN + 1);
        c00 = _mm256_add_ps(c00, _mm256_mul_ps(a_top, b0));
        c10 = _mm256_add_ps(c10, _mm256_mul_ps(a_bot, b0));
        c01 = _mm256_add_ps(c01, _mm256_mul_ps(a_top, b1));
        c11 = _mm256_add_ps(c11, _mm256_mul_ps(a_bot, b1));

        __m256 b2 = _mm256_broadcast_ss(packed_B + k * UN + 2);
        __m256 b3 = _mm256_broadcast_ss(packed_B + k * UN + 3);
        c02 = _mm256_add_ps(c02, _mm256_mul_ps(a_top, b2));
        c12 = _mm256_add_ps(c12, _mm256_mul_ps(a_bot, b2));
        c03 = _mm256_add_ps(c03, _mm256_mul_ps(a_top, b3));
        c13 = _mm256_add_ps(c13, _mm256_mul_ps(a_bot, b3));
    }

    /* SAVE: add to existing C (read-modify-write) and store.
     * C is row-major (i, j); tile is 16 rows × 4 cols at (0, 0) within this 16x4 sub-tile.
     * Each row stride = N (the full output's N dim).
     */
    /* Row r in 0..7 (top half), col 0..3 */
    float* C_row;
    float buf[8];

    /* For each of the 16 rows, gather the 4 column values from the 8 ymm registers
     * and add to C[row][col_start..col_start+3]. */
    float c00_arr[8], c01_arr[8], c02_arr[8], c03_arr[8];
    float c10_arr[8], c11_arr[8], c12_arr[8], c13_arr[8];
    _mm256_storeu_ps(c00_arr, c00);
    _mm256_storeu_ps(c01_arr, c01);
    _mm256_storeu_ps(c02_arr, c02);
    _mm256_storeu_ps(c03_arr, c03);
    _mm256_storeu_ps(c10_arr, c10);
    _mm256_storeu_ps(c11_arr, c11);
    _mm256_storeu_ps(c12_arr, c12);
    _mm256_storeu_ps(c13_arr, c13);

    for (int r = 0; r < 8; ++r) {
        C[r * N + 0] += c00_arr[r];
        C[r * N + 1] += c01_arr[r];
        C[r * N + 2] += c02_arr[r];
        C[r * N + 3] += c03_arr[r];
    }
    for (int r = 0; r < 8; ++r) {
        C[(r + 8) * N + 0] += c10_arr[r];
        C[(r + 8) * N + 1] += c11_arr[r];
        C[(r + 8) * N + 2] += c12_arr[r];
        C[(r + 8) * N + 3] += c13_arr[r];
    }
    (void)C_row; (void)buf;
}

/* Goto SGEMM main entry: C[M,N] = A[M,K] @ B[K,N].
 * For now: assume M % UM == 0 and N % UN == 0 (M=N=16 satisfies this).
 * Will handle edge cases in a follow-up commit.
 *
 * SIMD-aware: scalar tail loops handle M%UM and N%UN sub-blocks if needed.
 */
void bpd_mm_simd_cpu(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    /* Init C to zero */
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    /* Allocate packed buffers — sized for the maximum K-block we'll see */
    float* packed_A = aligned_alloc(32, UM * Q_BLOCK * sizeof(float));
    float* packed_B = aligned_alloc(32, UN * Q_BLOCK * sizeof(float));

    /* K-block loop (adaptive_half rule matching level3.c) */
    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int min_l;
        if (rem >= 2 * Q_BLOCK) {
            min_l = Q_BLOCK;
        } else if (rem > Q_BLOCK) {
            min_l = ((rem / 2 + UM - 1) / UM) * UM;
        } else {
            min_l = rem;
        }

        /* For each M-tile of UM=16 rows and N-tile of UN=4 cols */
        for (int i_start = 0; i_start + UM <= M; i_start += UM) {
            /* Pack A panel for this M-tile and K-block */
            pack_a_panel(A, K, i_start, ls, min_l, packed_A);

            for (int j_start = 0; j_start + UN <= N; j_start += UN) {
                /* Pack B panel for this N-tile and K-block */
                pack_b_panel(B, N, ls, j_start, min_l, packed_B);

                /* Inner kernel: accumulate 16x4 tile, RMW into C */
                kernel_16x4(packed_A, packed_B, C + i_start * N + j_start, N, min_l);
            }
        }
        ls += min_l;
    }

    free(packed_A);
    free(packed_B);
}
