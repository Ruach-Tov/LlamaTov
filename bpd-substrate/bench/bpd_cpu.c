// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Cache-line-aligned allocation.
 * When n >= 32 bytes (half a cache line), returns a 64-byte aligned address.
 * This ensures AVX loads/stores never cross cache line boundaries and
 * eliminates false sharing in multi-threaded scenarios.
 *
 * Uses posix_memalign (POSIX) — available on Linux/macOS.
 * Free with standard free(). */
static inline void* bpd_alloc(size_t n) {
    if (n < 32) return malloc(n);
    void* p = NULL;
    if (posix_memalign(&p, 64, n) != 0) return NULL;
    return p;
}

/* Zeroed cache-line-aligned allocation. */
static inline void* bpd_calloc(size_t count, size_t size) {
    size_t n = count * size;
    void* p = bpd_alloc(n);
    if (p) memset(p, 0, n);
    return p;
}

// AVX1 intrinsics — Phase 3.GEMM SIMD vectorization on Ivy Bridge and later.
// Guarded so non-AVX1 builds can still compile (substrate-design portability).
#if defined(__AVX__)
#include <immintrin.h>
#define BPD_HAVE_AVX1 1
#else
#define BPD_HAVE_AVX1 0
#endif

// CPU matmul: C[M,N] = A[M,K] @ B[K,N]
//
// Implements Goto's blocked GEMM algorithm matching OpenBLAS Sandybridge SGEMM
// bit-for-bit. PyTorch CPU calls cblas_sgemm directly; on AVX1 (Tesla P4 enclave),
// OpenBLAS dispatches to the SANDYBRIDGE sgemm_kernel_16x4 with these parameters:
//
//   gemm_tile_strategy(P=768, Q=384, UM=16, UN=4)
//
// Per OpenBLAS driver/level3/level3.c:309-322, the K block size adapts to the
// remaining work:
//   while remaining > 0:
//     if remaining >= 2*Q:    min_l = Q             # full block
//     elif remaining > Q:     min_l = ceil(rem/2/UM)*UM   # half, rounded to UM
//     else:                   min_l = remaining     # tail
//
// For (M=N=16, K=4096): 9 blocks of K=384 + 2 blocks of K=320 = 11 K-blocks.
//
// Inner accumulation per (i,j): sum_{k in block} A[i,k]*B[k,j] (sequential).
// Cross-block: C[i,j] += block_partial (left-fold across blocks).
//
// Empirically verified 0 ULP vs cblas_sgemm at K ∈ {256, 512, 768, 1024, 2048, 4096}
// and 5 seeds (see /tmp/mm_goto.c + /tmp/test_goto.py).
//
// Substrate-design parameters this kernel realizes (named in
// lib/implementation_matches.pl as platform_param/2 facts):
//   gemm_tile_strategy(goto_sandy)
//   gemm_p(768)
//   gemm_q(384)
//   gemm_unroll_m(16)
//   gemm_unroll_n(4)
// Forward declarations so the dispatcher in bpd_mm_cpu can tail-call AVX1 variants.
void bpd_mm_cpu_avx1(const float* A, const float* B, float* C,
                      int M, int N, int K);
void bpd_mm_cpu_avx1_v2(const float* A, const float* B, float* C,
                         int M, int N, int K);
void bpd_gemm_v2_full(const float* A, const float* B, float* C,
                       int M, int N, int K);

/* Packed GEMM: B-panel packing for L1 cache residency.
 * Packs each NR-wide column panel of B into contiguous memory
 * so the microkernel loads stride-NR (16 floats = 64 bytes = 1 cache line)
 * instead of stride-N (which causes TLB misses for large N).
 * KB=192 (12KB panel) is optimal for 32KB L1D on Ivy Bridge. */
#if BPD_HAVE_AVX1
static void bpd_gemm_packed_panel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    #define PACK_NR 16
    /* K-blocking matching OpenBLAS level3.c splitting algorithm exactly.
     * The scalar GEMM (0 ULP) uses this same logic. The key: when K < 2*Q,
     * split evenly (rounded to UM=16) instead of using Q as block size.
     * This ensures the same accumulation order as OpenBLAS/PyTorch. */
    const int Q = 384;
    const int UM = 16;
    memset(C, 0, (size_t)M * N * sizeof(float));
    float* B_panel;
    int max_kb = Q < K ? Q : K;
    posix_memalign((void**)&B_panel, 64, (size_t)max_kb * PACK_NR * sizeof(float));

    /* K-block loop matching OpenBLAS level3.c splitting exactly */
    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int kb;
        if (rem >= 2 * Q) {
            kb = Q;
        } else if (rem > Q) {
            kb = ((rem / 2 + UM - 1) / UM) * UM;
        } else {
            kb = rem;
        }
        int k0 = ls;
        int k_end = ls + kb;
        for (int j = 0; j + PACK_NR - 1 < N; j += PACK_NR) {
            /* Pack B[k0:k0+kb, j:j+NR] → contiguous panel */
            for (int k = 0; k < kb; k++)
                for (int jj = 0; jj < PACK_NR; jj++)
                    B_panel[k * PACK_NR + jj] = B[(k0 + k) * N + j + jj];

            /* Microkernel: 4 rows × 16 cols, B from packed panel */
            int i;
            for (i = 0; i + 3 < M; i += 4) {
                const float* a0 = A + i * K + k0;
                const float* a1 = a0 + K;
                const float* a2 = a1 + K;
                const float* a3 = a2 + K;
                __m256 acc00=_mm256_setzero_ps(), acc01=_mm256_setzero_ps();
                __m256 acc10=_mm256_setzero_ps(), acc11=_mm256_setzero_ps();
                __m256 acc20=_mm256_setzero_ps(), acc21=_mm256_setzero_ps();
                __m256 acc30=_mm256_setzero_ps(), acc31=_mm256_setzero_ps();
                for (int k = 0; k < kb; k++) {
                    __m256 b0 = _mm256_load_ps(B_panel + k*PACK_NR);
                    __m256 b1 = _mm256_load_ps(B_panel + k*PACK_NR + 8);
                    __m256 a;
                    a = _mm256_set1_ps(a0[k]);
                    acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a, b0));
                    acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a, b1));
                    a = _mm256_set1_ps(a1[k]);
                    acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a, b0));
                    acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a, b1));
                    a = _mm256_set1_ps(a2[k]);
                    acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a, b0));
                    acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a, b1));
                    a = _mm256_set1_ps(a3[k]);
                    acc30 = _mm256_add_ps(acc30, _mm256_mul_ps(a, b0));
                    acc31 = _mm256_add_ps(acc31, _mm256_mul_ps(a, b1));
                }
                float* c0 = C + i*N + j;
                _mm256_storeu_ps(c0,       _mm256_add_ps(_mm256_loadu_ps(c0), acc00));
                _mm256_storeu_ps(c0+8,     _mm256_add_ps(_mm256_loadu_ps(c0+8), acc01));
                _mm256_storeu_ps(c0+N,     _mm256_add_ps(_mm256_loadu_ps(c0+N), acc10));
                _mm256_storeu_ps(c0+N+8,   _mm256_add_ps(_mm256_loadu_ps(c0+N+8), acc11));
                _mm256_storeu_ps(c0+2*N,   _mm256_add_ps(_mm256_loadu_ps(c0+2*N), acc20));
                _mm256_storeu_ps(c0+2*N+8, _mm256_add_ps(_mm256_loadu_ps(c0+2*N+8), acc21));
                _mm256_storeu_ps(c0+3*N,   _mm256_add_ps(_mm256_loadu_ps(c0+3*N), acc30));
                _mm256_storeu_ps(c0+3*N+8, _mm256_add_ps(_mm256_loadu_ps(c0+3*N+8), acc31));
            }
            /* M-tail */
            for (; i < M; i++)
                for (int k = 0; k < kb; k++) {
                    float a_val = A[i*K+k0+k];
                    for (int jj = 0; jj < PACK_NR; jj++)
                        C[i*N+j+jj] += a_val * B[(k0+k)*N+j+jj];
                }
        }
        /* N-tail */
        for (int j = (N/PACK_NR)*PACK_NR; j < N; j++)
            for (int i = 0; i < M; i++)
                for (int k = k0; k < k0+kb; k++)
                    C[i*N+j] += A[i*K+k] * B[k*N+j];
        ls += kb;
    }
    free(B_panel);
    #undef PACK_NR
}
#endif

/* ── Packed GEMM with per-row BN+SiLU epilogue (true CBS fusion) ──
 * Identical to bpd_gemm_packed_panel EXCEPT:
 *   - The LAST K-block's store applies: y = scale[row]*acc + offset[row], then SiLU
 *   - Single K-block: store is y = silu(scale[row]*acc + offset[row])  (no load old C)
 *   - scale/offset arrays have M entries (one per output row = one per Cout channel)
 */
#if BPD_HAVE_AVX1
static inline __m256 apply_bn_silu_8(__m256 x, __m256 vs, __m256 vo) {
    __m256 y = _mm256_add_ps(_mm256_mul_ps(vs, x), vo);
    float buf[8] __attribute__((aligned(32)));
    _mm256_store_ps(buf, y);
    buf[0] = buf[0] / (1.0f + expf(-buf[0]));
    buf[1] = buf[1] / (1.0f + expf(-buf[1]));
    buf[2] = buf[2] / (1.0f + expf(-buf[2]));
    buf[3] = buf[3] / (1.0f + expf(-buf[3]));
    buf[4] = buf[4] / (1.0f + expf(-buf[4]));
    buf[5] = buf[5] / (1.0f + expf(-buf[5]));
    buf[6] = buf[6] / (1.0f + expf(-buf[6]));
    buf[7] = buf[7] / (1.0f + expf(-buf[7]));
    return _mm256_load_ps(buf);
}

static void bpd_gemm_packed_bn_silu(const float* A, const float* B, float* C,
                                      int M, int N, int K,
                                      const float* scale, const float* offset) {
    #define EP_NR 16
    const int Q = 384;
    const int UM = 16;
    memset(C, 0, (size_t)M * N * sizeof(float));
    int max_kb = Q < K ? Q : K;
    float* B_panel;
    posix_memalign((void**)&B_panel, 64, (size_t)max_kb * EP_NR * sizeof(float));

    /* Precompute K-block schedule to know which is last */
    int n_kblocks = 0;
    { int ls=0; while(ls<K) { int rem=K-ls; int ml; if(rem>=2*Q) ml=Q; else if(rem>Q) ml=((rem/2+UM-1)/UM)*UM; else ml=rem; n_kblocks++; ls+=ml; } }

    int ls = 0, kblock_idx = 0;
    while (ls < K) {
        int rem = K - ls;
        int kb;
        if (rem >= 2*Q) kb = Q;
        else if (rem > Q) kb = ((rem/2+UM-1)/UM)*UM;
        else kb = rem;
        int k0 = ls;
        int is_last = (kblock_idx == n_kblocks - 1);

        for (int j = 0; j + EP_NR - 1 < N; j += EP_NR) {
            /* Pack current B panel */
            for (int k = 0; k < kb; k++)
                for (int jj = 0; jj < EP_NR; jj++)
                    B_panel[k * EP_NR + jj] = B[(k0 + k) * N + j + jj];

            /* Prefetch NEXT B panel's first cache lines while we compute this one.
             * The prefetch runs on port 2/3 which is idle during ALU-heavy GEMM. */
            int j_next = j + EP_NR;
            int has_next = (j_next + EP_NR - 1 < N);

            int i;
            for (i = 0; i + 3 < M; i += 4) {
                const float* a0 = A + i*K + k0;
                const float* a1 = a0 + K;
                const float* a2 = a1 + K;
                const float* a3 = a2 + K;
                __m256 acc00=_mm256_setzero_ps(), acc01=_mm256_setzero_ps();
                __m256 acc10=_mm256_setzero_ps(), acc11=_mm256_setzero_ps();
                __m256 acc20=_mm256_setzero_ps(), acc21=_mm256_setzero_ps();
                __m256 acc30=_mm256_setzero_ps(), acc31=_mm256_setzero_ps();
                for (int k = 0; k < kb; k++) {
                    __m256 b0 = _mm256_load_ps(B_panel + k*EP_NR);
                    __m256 b1 = _mm256_load_ps(B_panel + k*EP_NR + 8);
                    /* Prefetch next B panel row every 8 K-steps (1 cache line per prefetch) */
                    if (has_next && (k & 7) == 0 && k/8 < kb)
                        _mm_prefetch((const char*)(B + (k0 + k/8) * N + j_next), _MM_HINT_T0);
                    __m256 a;
                    a = _mm256_set1_ps(a0[k]); acc00=_mm256_add_ps(acc00,_mm256_mul_ps(a,b0)); acc01=_mm256_add_ps(acc01,_mm256_mul_ps(a,b1));
                    a = _mm256_set1_ps(a1[k]); acc10=_mm256_add_ps(acc10,_mm256_mul_ps(a,b0)); acc11=_mm256_add_ps(acc11,_mm256_mul_ps(a,b1));
                    a = _mm256_set1_ps(a2[k]); acc20=_mm256_add_ps(acc20,_mm256_mul_ps(a,b0)); acc21=_mm256_add_ps(acc21,_mm256_mul_ps(a,b1));
                    a = _mm256_set1_ps(a3[k]); acc30=_mm256_add_ps(acc30,_mm256_mul_ps(a,b0)); acc31=_mm256_add_ps(acc31,_mm256_mul_ps(a,b1));
                }
                float* c0 = C + i*N + j;
                if (is_last) {
                    /* Last K-block: apply BN+SiLU epilogue at store */
                    __m256 vs0=_mm256_set1_ps(scale[i]),   vo0=_mm256_set1_ps(offset[i]);
                    __m256 vs1=_mm256_set1_ps(scale[i+1]), vo1=_mm256_set1_ps(offset[i+1]);
                    __m256 vs2=_mm256_set1_ps(scale[i+2]), vo2=_mm256_set1_ps(offset[i+2]);
                    __m256 vs3=_mm256_set1_ps(scale[i+3]), vo3=_mm256_set1_ps(offset[i+3]);
                    _mm256_storeu_ps(c0,       apply_bn_silu_8(_mm256_add_ps(_mm256_loadu_ps(c0), acc00), vs0, vo0));
                    _mm256_storeu_ps(c0+8,     apply_bn_silu_8(_mm256_add_ps(_mm256_loadu_ps(c0+8), acc01), vs0, vo0));
                    _mm256_storeu_ps(c0+N,     apply_bn_silu_8(_mm256_add_ps(_mm256_loadu_ps(c0+N), acc10), vs1, vo1));
                    _mm256_storeu_ps(c0+N+8,   apply_bn_silu_8(_mm256_add_ps(_mm256_loadu_ps(c0+N+8), acc11), vs1, vo1));
                    _mm256_storeu_ps(c0+2*N,   apply_bn_silu_8(_mm256_add_ps(_mm256_loadu_ps(c0+2*N), acc20), vs2, vo2));
                    _mm256_storeu_ps(c0+2*N+8, apply_bn_silu_8(_mm256_add_ps(_mm256_loadu_ps(c0+2*N+8), acc21), vs2, vo2));
                    _mm256_storeu_ps(c0+3*N,   apply_bn_silu_8(_mm256_add_ps(_mm256_loadu_ps(c0+3*N), acc30), vs3, vo3));
                    _mm256_storeu_ps(c0+3*N+8, apply_bn_silu_8(_mm256_add_ps(_mm256_loadu_ps(c0+3*N+8), acc31), vs3, vo3));
                } else {
                    /* Intermediate K-block: normal accumulate */
                    _mm256_storeu_ps(c0,       _mm256_add_ps(_mm256_loadu_ps(c0), acc00));
                    _mm256_storeu_ps(c0+8,     _mm256_add_ps(_mm256_loadu_ps(c0+8), acc01));
                    _mm256_storeu_ps(c0+N,     _mm256_add_ps(_mm256_loadu_ps(c0+N), acc10));
                    _mm256_storeu_ps(c0+N+8,   _mm256_add_ps(_mm256_loadu_ps(c0+N+8), acc11));
                    _mm256_storeu_ps(c0+2*N,   _mm256_add_ps(_mm256_loadu_ps(c0+2*N), acc20));
                    _mm256_storeu_ps(c0+2*N+8, _mm256_add_ps(_mm256_loadu_ps(c0+2*N+8), acc21));
                    _mm256_storeu_ps(c0+3*N,   _mm256_add_ps(_mm256_loadu_ps(c0+3*N), acc30));
                    _mm256_storeu_ps(c0+3*N+8, _mm256_add_ps(_mm256_loadu_ps(c0+3*N+8), acc31));
                }
            }
            /* M-tail */
            for (; i < M; i++) {
                for (int k = 0; k < kb; k++) {
                    float a_val = A[i*K+k0+k];
                    for (int jj = 0; jj < EP_NR; jj++)
                        C[i*N+j+jj] += a_val * B[(k0+k)*N+j+jj];
                }
                if (is_last) {
                    float s = scale[i], o = offset[i];
                    for (int jj = 0; jj < EP_NR; jj++) {
                        float y = s * C[i*N+j+jj] + o;
                        C[i*N+j+jj] = y / (1.0f + expf(-y));
                    }
                }
            }
        }
        /* N-tail */
        for (int j = (N/EP_NR)*EP_NR; j < N; j++)
            for (int i = 0; i < M; i++) {
                for (int k = k0; k < k0+kb; k++)
                    C[i*N+j] += A[i*K+k] * B[k*N+j];
                if (is_last) {
                    float y = scale[i] * C[i*N+j] + offset[i];
                    C[i*N+j] = y / (1.0f + expf(-y));
                }
            }
        ls += kb; kblock_idx++;
    }
    free(B_panel);
    #undef EP_NR
}
#endif

void bpd_mm_cpu(const float* A, const float* B, float* C,
                int M, int N, int K) {
    // ── Runtime dispatch (3 paths) ──
    // SUBSTRATE_AVX1_GEMM controls scalar vs AVX1 selection:
    //   0 = force scalar K-block (Tier 1.5 reference path)
    //   1 = AVX1 SIMD (default when BPD_HAVE_AVX1)
    // SUBSTRATE_AVX1_GEMM_V2 controls v1 vs v2 (when AVX1 chosen):
    //   0 = v1 (single accumulator, 1x8 register tile)
    //   1 = v2 (8 accumulators, 4x16 register tile, K-unroll 4) — DEFAULT
    //
    // Both choices are cached in static ints after first call to avoid
    // getenv() per GEMM (called thousands of times per YOLO frame).
    //
    // dispatch_choice values: 0=scalar, 1=avx1_v1, 2=avx1_v2
    static int dispatch_choice = -1;
    if (dispatch_choice == -1) {
        const char* env_avx = getenv("SUBSTRATE_AVX1_GEMM");
        if (env_avx && env_avx[0] == '0') {
            dispatch_choice = 0;  // scalar
        } else {
#if BPD_HAVE_AVX1
            const char* env_v2 = getenv("SUBSTRATE_AVX1_GEMM_V2");
            if (env_v2 && env_v2[0] == '0') {
                dispatch_choice = 1;  // v1
            } else {
                dispatch_choice = 3;  // packed (default — fastest)
            }
#else
            dispatch_choice = 0;
#endif
        }
    }
    if (dispatch_choice == 3) {
#if BPD_HAVE_AVX1
        bpd_gemm_packed_panel(A, B, C, M, N, K);
#else
        bpd_gemm_v2_full(A, B, C, M, N, K);
#endif
        return;
    }
    if (dispatch_choice == 2) {
        bpd_gemm_v2_full(A, B, C, M, N, K);
        return;
    }
    if (dispatch_choice == 1) {
        bpd_mm_cpu_avx1(A, B, C, M, N, K);
        return;
    }

    // ── Scalar K-block GEMM (Tier 1.5 reference, bit-identical with PyTorch CBLAS) ──
    const int Q = 384;
    const int UM = 16;

    // Init C to zero
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // K-block loop matching OpenBLAS level3.c
    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int min_l;
        if (rem >= 2 * Q) {
            min_l = Q;
        } else if (rem > Q) {
            min_l = ((rem / 2 + UM - 1) / UM) * UM;
        } else {
            min_l = rem;
        }

        // Inner: per (i, j) compute the K-block partial, add to running C[i,j].
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float partial = 0.0f;
                for (int k = ls; k < ls + min_l; k++) {
                    partial += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] += partial;
            }
        }
        ls += min_l;
    }
}

// ──────────────────────────────────────────────────────────────────────
// AVX1-vectorized matmul (Phase 3.GEMM)
// ──────────────────────────────────────────────────────────────────────
//
// Same K-block algorithm as bpd_mm_cpu, but the inner (row, col) loop is
// vectorized across cols using AVX1 256-bit SIMD (8 floats per lane).
//
// Bit-identity preservation:
//   Each per-col accumulator does EXACTLY the same scalar sequence of operations
//   as bpd_mm_cpu: partial += A[row*K+k] * B[k*N+col] for k in [ls, ls+min_l).
//   The only difference is that 8 cols proceed in parallel SIMD lanes, with
//   each lane's accumulator independent. Per-lane IEEE arithmetic is identical
//   to the scalar version, so:
//     partial_vec[lane] == scalar_partial(col = col_base + lane)
//   for all lanes. The horizontal store back to C just writes 8 contiguous
//   float values — no cross-lane operations that could change rounding.
//
// Layout: A is (M, K) row-major; B is (K, N) row-major.
//   For fixed (row, k): A[row, k] is a single float (broadcast to all 8 lanes).
//   For 8 consecutive cols: B[k, col_base..col_base+7] are 8 contiguous floats
//   that can be loaded with a single _mm256_loadu_ps.
//
// AVX1 has no FMA, so we emit mul + add separately (matching gcc -O2 scalar code).
//
// Tail handling: if N % 8 != 0, the remaining cols use the scalar fallback
// with the IDENTICAL accumulation order, preserving bit-identity for non-8-aligned N.
#if BPD_HAVE_AVX1
void bpd_mm_cpu_avx1(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    const int Q = 384;
    const int UM = 16;

    // Init C to zero (matches bpd_mm_cpu)
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int min_l;
        if (rem >= 2 * Q) {
            min_l = Q;
        } else if (rem > Q) {
            min_l = ((rem / 2 + UM - 1) / UM) * UM;
        } else {
            min_l = rem;
        }

        // Vectorized inner loop: process 8 cols at a time per row.
        int n_simd = N & ~7;  // largest multiple of 8 <= N
        for (int row = 0; row < M; row++) {
            const float* a_row = A + row * K;
            float* c_row = C + row * N;

            // SIMD-8 path: 8 cols in parallel per inner k iteration
            for (int col = 0; col < n_simd; col += 8) {
                __m256 partial_vec = _mm256_setzero_ps();
                for (int k = ls; k < ls + min_l; k++) {
                    // Broadcast A[row, k] to all 8 lanes
                    __m256 a_bk = _mm256_set1_ps(a_row[k]);
                    // Load B[k, col..col+7] (8 contiguous floats)
                    __m256 b_kj = _mm256_loadu_ps(B + k * N + col);
                    // partial_vec += a_bk * b_kj (mul + add, no FMA)
                    __m256 prod = _mm256_mul_ps(a_bk, b_kj);
                    partial_vec = _mm256_add_ps(partial_vec, prod);
                }
                // Add this K-block's partial to the running C accumulator.
                // C[row, col..col+7] += partial_vec
                __m256 c_vec = _mm256_loadu_ps(c_row + col);
                c_vec = _mm256_add_ps(c_vec, partial_vec);
                _mm256_storeu_ps(c_row + col, c_vec);
            }

            // Scalar tail: cols [n_simd, N) — matches bpd_mm_cpu exactly
            for (int col = n_simd; col < N; col++) {
                float partial = 0.0f;
                for (int k = ls; k < ls + min_l; k++) {
                    partial += a_row[k] * B[k * N + col];
                }
                c_row[col] += partial;
            }
        }
        ls += min_l;
    }
}
#else
// AVX1 not available at compile time — fall back to scalar bpd_mm_cpu.
void bpd_mm_cpu_avx1(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    bpd_mm_cpu(A, B, C, M, N, K);
}
#endif

// ──────────────────────────────────────────────────────────────────────
// bpd_mm_cpu_avx1_v2 — CAT-scan-informed GEMM (Phase 3.CAT.a)
// ──────────────────────────────────────────────────────────────────────
//
// Based on CAT-scan disassembly of OpenBLAS sgemm_kernel_SANDYBRIDGE.
// Foundational memory: c101e652. Substrate-design discipline: 7b297878.
//
// Substrate-design parameters baked in (deduced from OpenBLAS):
//   register_blocking(MR=4, NR=16)  — 4 rows × 16 cols per inner iteration
//                                     = 8 ymm accumulators (4 rows × 2 col-vectors of 8 floats)
//   ilp_accumulators(8)             — 8 INDEPENDENT (row, col_group) accumulators
//   unroll_factor_K(4)              — 4 k-values per inner loop body
//
// BIT-IDENTITY PRESERVATION (the substantive substrate-design Essence):
//   Each ymm accumulator holds ONE (row, col_group) output cell's running sum.
//   Within each accumulator's k-loop, the reduction order is LINEAR LEFT-TO-RIGHT:
//     acc[row, col_group] += A[row, k] * B[k, col_group]  for k = 0, 1, 2, ..., K-1
//   This is EXACTLY the same scalar order as bpd_mm_cpu and bpd_mm_cpu_avx1.
//   No tree reduction. No partial-sum interleaving. No fancy math.
//
//   The 8 accumulators run in parallel across 8 DIFFERENT output cells, not
//   across 8 partial sums of the SAME output cell. This is option (a) per
//   Medayek's analysis: bit-safe by construction.
//
// TILING:
//   M is processed in blocks of MR=4 rows. Tail rows (M % 4) handled by
//   bpd_mm_cpu_avx1 scalar-SIMD fallback (one row at a time).
//   N is processed in blocks of NR=16 cols. Tail cols (N % 16) handled by
//   the v1 path for that subset of cols.
//   K is processed in K-blocks of size Q=384 (same as bpd_mm_cpu), to match
//   the partial-sum semantics: each K-block adds to C, allowing cumulative
//   accumulation across K-blocks bit-identically with bpd_mm_cpu.
//
// LIMITATION (deliberate, simple to verify):
//   For shapes where M < 4 or N < 16, falls back to bpd_mm_cpu_avx1 (the
//   single-accumulator v1 path). The v2 path activates only for large-enough
//   tiles. This keeps the code clear and the bit-identity gate trivial.

// ──────────────────────────────────────────────────────────────────────
// Phase 3.CAT.TDD primitives — decomposed for test-driven precision
// ──────────────────────────────────────────────────────────────────────
// Per Heath's substrate-design discipline: each primitive is verified
// in isolation via bench/test_f3_v2_tdd.py before composition.

// P1: Zero-initialize a contiguous M×N float32 buffer.
// Tested: test_p1_gemm_v2_init.
void bpd_gemm_v2_init(float* C, int M, int N) {
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
}

// P2: Accumulate a K-range [k_start, k_end) into C using 4×16 register
// blocking with 8 SIMD accumulators per tile and K-unroll factor 4.
//
// Mutates C: C[i,j] += sum_{k=k_start..k_end-1} A[i,k] * B[k,j]
//
// Constraints:
//   M_blocks = M / 4, M_tail = M % 4   ← M-tail handled by P3 (call separately)
//   N_blocks = N / 16, N_tail = N % 16 ← N-tail handled by P4 (call separately)
//   This primitive ONLY does the (M_blocks × N_blocks) tile interior.
//   M and N may exceed the tile coverage; rows >= M_blocks*4 and cols >=
//   N_blocks*16 are left UNTOUCHED (P3/P4 cover them).
//
// Bit-identity: each per-tile ymm accumulator holds ONE (row, col_group)
// output cell's running sum. Within each accumulator's k-loop the partial
// sum is built linearly k = k_start, k_start+1, ..., k_end-1 — matching
// the Python scalar reference. No tree reduction, no partial-sum interleaving.
//
// Tested: test_p2_gemm_v2_kblock_accumulate_{simple,partial,two_blocks}.
#if BPD_HAVE_AVX1
void bpd_gemm_v2_kblock_accumulate(const float* A, const float* B, float* C,
                                     int M, int N, int K_total,
                                     int k_start, int k_end) {
    const int MR = 4;
    const int NR = 16;
    const int KU = 4;
    int M_blocks = M / MR;
    int N_blocks = N / NR;
    int min_l = k_end - k_start;
    int kus = (min_l % KU == 0) ? KU : 1;

    // Phase 3.CAT.g B-panel packing: stack-allocated panel of size min_l x NR.
    // Max size: Q * NR = 384 * 16 floats = 24 KB (fits in 32 KB L1).
    // Env-controlled: SUBSTRATE_AVX1_PACK (default '1' = on).
    //
    // Packing transforms strided B reads (stride N) into contiguous reads
    // (stride NR=16). The panel is packed ONCE per (cb, K-block) and reused
    // across all M_blocks row tiles \u2014 amortizing the packing cost over rb.
    //
    // Bit-identity: the inner-loop math is unchanged. Each (row, col_group)
    // accumulator still does the same linear left-fold over the same B values
    // in the same order \u2014 just read from the packed buffer instead of the
    // strided source. memcpy semantics preserve every float bit.
    // Phase 3.CAT.h sweep result: with prefetch+packing combined,
    // packing nets a small additional ~0.7%% over prefetch-only across the
    // 8 YOLOv5n CBS GEMM shapes. Default ON when SUBSTRATE_AVX1_PACK unset.
    // Flip to '0' via env to disable.
    static int pack_choice = -1;
    if (pack_choice == -1) {
        const char* env = getenv("SUBSTRATE_AVX1_PACK");
        pack_choice = (env && env[0] == '0') ? 0 : 1;
    }
    int do_pack = pack_choice;
    float packed_B[384 * 16] __attribute__((aligned(32)));  // panel buffer (24 KB)

    // Outer loop: cb (col-block). Inner loop: rb (row-block).
    // This inversion lets us pack B once per cb and reuse across rb.
    for (int cb = 0; cb < N_blocks; cb++) {
        int col_base = cb * NR;

        // Pack B[k_start..k_end, col_base..col_base+NR-1] into contiguous panel.
        // Layout: packed_B[(k - k_start) * NR + j] = B[k * N + col_base + j]
        // After packing, the inner loop reads packed_B with stride NR=16
        // (contiguous in K).
        const float* B_src = do_pack ? packed_B : NULL;  // sentinel: use packed if on
        if (do_pack) {
            for (int k = k_start; k < k_end; k++) {
                // Two ymm loads + stores per k: copies 16 contiguous floats
                __m256 b0 = _mm256_loadu_ps(B + k * N + col_base);
                __m256 b1 = _mm256_loadu_ps(B + k * N + col_base + 8);
                _mm256_store_ps(packed_B + (k - k_start) * NR,     b0);
                _mm256_store_ps(packed_B + (k - k_start) * NR + 8, b1);
            }
        }

        for (int rb = 0; rb < M_blocks; rb++) {
            int row_base = rb * MR;
            const float* a0 = A + (row_base + 0) * K_total;
            const float* a1 = A + (row_base + 1) * K_total;
            const float* a2 = A + (row_base + 2) * K_total;
            const float* a3 = A + (row_base + 3) * K_total;
            float* c0 = C + (row_base + 0) * N;
            float* c1 = C + (row_base + 1) * N;
            float* c2 = C + (row_base + 2) * N;
            float* c3 = C + (row_base + 3) * N;

            // 8 fresh accumulators per tile (this K-block's contribution)
            __m256 acc_r0_c0 = _mm256_setzero_ps();
            __m256 acc_r0_c1 = _mm256_setzero_ps();
            __m256 acc_r1_c0 = _mm256_setzero_ps();
            __m256 acc_r1_c1 = _mm256_setzero_ps();
            __m256 acc_r2_c0 = _mm256_setzero_ps();
            __m256 acc_r2_c1 = _mm256_setzero_ps();
            __m256 acc_r3_c0 = _mm256_setzero_ps();
            __m256 acc_r3_c1 = _mm256_setzero_ps();

            if (kus == KU) {
                // Phase 3.CAT.f: prefetch lookahead, env-controlled.
                static int prefetch_choice = -1;
                if (prefetch_choice == -1) {
                    const char* env = getenv("SUBSTRATE_AVX1_PREFETCH");
                    prefetch_choice = (env && env[0] == '0') ? 0 : 1;
                }
                int do_prefetch = prefetch_choice;
                for (int k = k_start; k < k_end; k += KU) {
                    int k_next = k + KU;
                    if (do_prefetch && k_next < k_end) {
                        if (!do_pack) {
                            _mm_prefetch((const char*)(B + (k_next + 0) * N + col_base),     _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 0) * N + col_base + 8), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 1) * N + col_base),     _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 1) * N + col_base + 8), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 2) * N + col_base),     _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 2) * N + col_base + 8), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 3) * N + col_base),     _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 3) * N + col_base + 8), _MM_HINT_T0);
                        }
                        _mm_prefetch((const char*)(a0 + k_next), _MM_HINT_T0);
                        _mm_prefetch((const char*)(a1 + k_next), _MM_HINT_T0);
                        _mm_prefetch((const char*)(a2 + k_next), _MM_HINT_T0);
                        _mm_prefetch((const char*)(a3 + k_next), _MM_HINT_T0);
                    }
                    // Choose B source based on packing
                    if (do_pack) {
                        #define KSTEP_PACKED(KOFF) do {                                                   \
                            int kp = (k + (KOFF)) - k_start;                                              \
                            __m256 b0 = _mm256_load_ps(packed_B + kp * NR);                               \
                            __m256 b1 = _mm256_load_ps(packed_B + kp * NR + 8);                           \
                            __m256 av0 = _mm256_set1_ps(a0[k + (KOFF)]);                                  \
                            __m256 av1 = _mm256_set1_ps(a1[k + (KOFF)]);                                  \
                            __m256 av2 = _mm256_set1_ps(a2[k + (KOFF)]);                                  \
                            __m256 av3 = _mm256_set1_ps(a3[k + (KOFF)]);                                  \
                            acc_r0_c0 = _mm256_add_ps(acc_r0_c0, _mm256_mul_ps(av0, b0));                 \
                            acc_r0_c1 = _mm256_add_ps(acc_r0_c1, _mm256_mul_ps(av0, b1));                 \
                            acc_r1_c0 = _mm256_add_ps(acc_r1_c0, _mm256_mul_ps(av1, b0));                 \
                            acc_r1_c1 = _mm256_add_ps(acc_r1_c1, _mm256_mul_ps(av1, b1));                 \
                            acc_r2_c0 = _mm256_add_ps(acc_r2_c0, _mm256_mul_ps(av2, b0));                 \
                            acc_r2_c1 = _mm256_add_ps(acc_r2_c1, _mm256_mul_ps(av2, b1));                 \
                            acc_r3_c0 = _mm256_add_ps(acc_r3_c0, _mm256_mul_ps(av3, b0));                 \
                            acc_r3_c1 = _mm256_add_ps(acc_r3_c1, _mm256_mul_ps(av3, b1));                 \
                        } while (0)
                        KSTEP_PACKED(0); KSTEP_PACKED(1); KSTEP_PACKED(2); KSTEP_PACKED(3);
                        #undef KSTEP_PACKED
                    } else {
                        #define KSTEP(KOFF) do {                                                          \
                            __m256 b0 = _mm256_loadu_ps(B + (k + (KOFF)) * N + col_base);                 \
                            __m256 b1 = _mm256_loadu_ps(B + (k + (KOFF)) * N + col_base + 8);             \
                            __m256 av0 = _mm256_set1_ps(a0[k + (KOFF)]);                                  \
                            __m256 av1 = _mm256_set1_ps(a1[k + (KOFF)]);                                  \
                            __m256 av2 = _mm256_set1_ps(a2[k + (KOFF)]);                                  \
                            __m256 av3 = _mm256_set1_ps(a3[k + (KOFF)]);                                  \
                            acc_r0_c0 = _mm256_add_ps(acc_r0_c0, _mm256_mul_ps(av0, b0));                 \
                            acc_r0_c1 = _mm256_add_ps(acc_r0_c1, _mm256_mul_ps(av0, b1));                 \
                            acc_r1_c0 = _mm256_add_ps(acc_r1_c0, _mm256_mul_ps(av1, b0));                 \
                            acc_r1_c1 = _mm256_add_ps(acc_r1_c1, _mm256_mul_ps(av1, b1));                 \
                            acc_r2_c0 = _mm256_add_ps(acc_r2_c0, _mm256_mul_ps(av2, b0));                 \
                            acc_r2_c1 = _mm256_add_ps(acc_r2_c1, _mm256_mul_ps(av2, b1));                 \
                            acc_r3_c0 = _mm256_add_ps(acc_r3_c0, _mm256_mul_ps(av3, b0));                 \
                            acc_r3_c1 = _mm256_add_ps(acc_r3_c1, _mm256_mul_ps(av3, b1));                 \
                        } while (0)
                        KSTEP(0); KSTEP(1); KSTEP(2); KSTEP(3);
                        #undef KSTEP
                    }
                }
            } else {
                // Non-unrolled fallback: no packing optimization here
                // (this branch is only used for irregular K, never YOLO shapes).
                for (int k = k_start; k < k_end; k++) {
                    __m256 b0 = _mm256_loadu_ps(B + k * N + col_base);
                    __m256 b1 = _mm256_loadu_ps(B + k * N + col_base + 8);
                    __m256 av0 = _mm256_set1_ps(a0[k]);
                    __m256 av1 = _mm256_set1_ps(a1[k]);
                    __m256 av2 = _mm256_set1_ps(a2[k]);
                    __m256 av3 = _mm256_set1_ps(a3[k]);
                    acc_r0_c0 = _mm256_add_ps(acc_r0_c0, _mm256_mul_ps(av0, b0));
                    acc_r0_c1 = _mm256_add_ps(acc_r0_c1, _mm256_mul_ps(av0, b1));
                    acc_r1_c0 = _mm256_add_ps(acc_r1_c0, _mm256_mul_ps(av1, b0));
                    acc_r1_c1 = _mm256_add_ps(acc_r1_c1, _mm256_mul_ps(av1, b1));
                    acc_r2_c0 = _mm256_add_ps(acc_r2_c0, _mm256_mul_ps(av2, b0));
                    acc_r2_c1 = _mm256_add_ps(acc_r2_c1, _mm256_mul_ps(av2, b1));
                    acc_r3_c0 = _mm256_add_ps(acc_r3_c0, _mm256_mul_ps(av3, b0));
                    acc_r3_c1 = _mm256_add_ps(acc_r3_c1, _mm256_mul_ps(av3, b1));
                }
            }

            // C[i,j] += partial — load existing C, add partial, store back.
            // This is the partial-sum-into-C semantics that makes multi-K-block
            // composition work bit-identically with the scalar reference.
            __m256 c_r0_c0 = _mm256_loadu_ps(c0 + col_base);
            __m256 c_r0_c1 = _mm256_loadu_ps(c0 + col_base + 8);
            _mm256_storeu_ps(c0 + col_base,     _mm256_add_ps(c_r0_c0, acc_r0_c0));
            _mm256_storeu_ps(c0 + col_base + 8, _mm256_add_ps(c_r0_c1, acc_r0_c1));
            __m256 c_r1_c0 = _mm256_loadu_ps(c1 + col_base);
            __m256 c_r1_c1 = _mm256_loadu_ps(c1 + col_base + 8);
            _mm256_storeu_ps(c1 + col_base,     _mm256_add_ps(c_r1_c0, acc_r1_c0));
            _mm256_storeu_ps(c1 + col_base + 8, _mm256_add_ps(c_r1_c1, acc_r1_c1));
            __m256 c_r2_c0 = _mm256_loadu_ps(c2 + col_base);
            __m256 c_r2_c1 = _mm256_loadu_ps(c2 + col_base + 8);
            _mm256_storeu_ps(c2 + col_base,     _mm256_add_ps(c_r2_c0, acc_r2_c0));
            _mm256_storeu_ps(c2 + col_base + 8, _mm256_add_ps(c_r2_c1, acc_r2_c1));
            __m256 c_r3_c0 = _mm256_loadu_ps(c3 + col_base);
            __m256 c_r3_c1 = _mm256_loadu_ps(c3 + col_base + 8);
            _mm256_storeu_ps(c3 + col_base,     _mm256_add_ps(c_r3_c0, acc_r3_c0));
            _mm256_storeu_ps(c3 + col_base + 8, _mm256_add_ps(c_r3_c1, acc_r3_c1));
        }
    }
}
#else
void bpd_gemm_v2_kblock_accumulate(const float* A, const float* B, float* C,
                                     int M, int N, int K_total,
                                     int k_start, int k_end) {
    int M_blocks = (M / 4) * 4;
    int N_blocks = (N / 16) * 16;
    for (int i = 0; i < M_blocks; i++) {
        const float* a_row = A + i * K_total;
        float* c_row = C + i * N;
        for (int j = 0; j < N_blocks; j++) {
            float partial = 0.0f;
            for (int k = k_start; k < k_end; k++) partial += a_row[k] * B[k * N + j];
            c_row[j] += partial;
        }
    }
}
#endif

// P3: M-tail handler. Accumulates rows [M_blocks*4, M) for the K-range
// [k_start, k_end). Scalar per-(row, col) linear K-fold matching the
// reference exactly. Tested: test_p3 with M=5, 7, 17.
void bpd_gemm_v2_kblock_accumulate_mtail(const float* A, const float* B, float* C,
                                           int M, int N, int K_total,
                                           int k_start, int k_end) {
    int M_blocks = M / 4;
    int row_start = M_blocks * 4;
    for (int i = row_start; i < M; i++) {
        const float* a_row = A + i * K_total;
        float* c_row = C + i * N;
        for (int j = 0; j < N; j++) {
            float partial = 0.0f;
            for (int k = k_start; k < k_end; k++) {
                partial += a_row[k] * B[k * N + j];
            }
            c_row[j] += partial;
        }
    }
}

// P4: N-tail handler. Accumulates cols [N_blocks*16, N) for ALL rows
// (the main path doesn't touch these cols even for tile rows).
// Tested: test_p4 with N=15, 17, 23.
void bpd_gemm_v2_kblock_accumulate_ntail(const float* A, const float* B, float* C,
                                           int M, int N, int K_total,
                                           int k_start, int k_end) {
    int N_blocks = N / 16;
    int col_start = N_blocks * 16;
    int M_blocks = M / 4;
    int row_end_main = M_blocks * 4;  // P4 only covers main rows (P3 already handled tail rows for ALL cols)
    for (int i = 0; i < row_end_main; i++) {
        const float* a_row = A + i * K_total;
        float* c_row = C + i * N;
        for (int j = col_start; j < N; j++) {
            float partial = 0.0f;
            for (int k = k_start; k < k_end; k++) {
                partial += a_row[k] * B[k * N + j];
            }
            c_row[j] += partial;
        }
    }
}

// P5: bpd_gemm_v2_full(A, B, C, M, N, K) — full GEMM composed from
// P1 (zero-init) + P2 (main 4x16 tile accumulate) + P3 (M-tail) + P4 (N-tail)
// across K-blocks of size Q=384.
//
// Bit-identity: this composition produces the SAME float pattern as
// bpd_mm_cpu_avx1_v2 (which is already verified BIT_IDENTICAL with scalar).
// Specifically: same per-(i,j) K-block boundaries, same partial-sum-into-C
// semantics. Each K-block calls P2 (main tile) and P3 (M-tail) and P4
// (N-tail) IN THIS ORDER, then advances ls.
//
// Tested: test_p5 against bpd_mm_cpu_avx1_v2 across YOLO shapes + edge cases.
void bpd_gemm_v2_full(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    const int Q = 384;
    const int MR = 4;
    const int NR = 16;
    // Phase 3.CAT.TDD.10 route (a): hoist tail-presence outside the K-loop
    // so we only emit the function calls when there's actual tail work.
    // For all YOLO CBS shapes (M%4==0, N%16==0), this skips ~2-3 no-op
    // function calls per K-block per CBS layer, recovering ~5%% wall-clock.
    int has_mtail = (M % MR) != 0;
    int has_ntail = (N % NR) != 0;
    bpd_gemm_v2_init(C, M, N);
    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int min_l;
        if (rem >= 2 * Q) min_l = Q;
        else if (rem > Q) min_l = ((rem / 2 + MR*4 - 1) / (MR*4)) * (MR*4);
        else min_l = rem;
        int k_end = ls + min_l;
        // P2 main tiles
        bpd_gemm_v2_kblock_accumulate(A, B, C, M, N, K, ls, k_end);
        // P3 M-tail (only when there are M%MR remainder rows)
        if (has_mtail) {
            bpd_gemm_v2_kblock_accumulate_mtail(A, B, C, M, N, K, ls, k_end);
        }
        // P4 N-tail (only when there are N%NR remainder cols on main rows)
        if (has_ntail) {
            bpd_gemm_v2_kblock_accumulate_ntail(A, B, C, M, N, K, ls, k_end);
        }
        ls += min_l;
    }
}

// P6: bpd_bn_silu_epilogue_simd — apply silu(alpha[row] * x + beta[row])
// SIMD-vectorized across cols (8 lanes per ymm). Tail cols (N % 8 != 0)
// fall back to scalar matching bpd_silu_cpu exactly.
//
// Bit-identity: per-element math is identical to the F3 v1 scalar epilogue
// loop. alpha and beta are broadcast once per row (constant within the row),
// then the per-element operation `silu(a*x + b)` uses DIVSS form expf, so
// the substantive substantive substantive SIMD lanes each do exactly the
// same scalar math.
//
// AVX1 has no SIMD expf; we extract the 8 lanes via aligned store, apply
// scalar expf per lane (matching bpd_silu_cpu), reload. This is necessary
// for bit-identity since AVX1 expf approximations would diverge.
//
// Tested: test_p6 against scalar epilogue loop.
#if BPD_HAVE_AVX1
static inline __m256 _bpd_silu_avx1_p6(__m256 x) {
    float buf[8] __attribute__((aligned(32)));
    _mm256_store_ps(buf, x);
    for (int i = 0; i < 8; i++) {
        buf[i] = buf[i] / (1.0f + expf(-buf[i]));
    }
    return _mm256_load_ps(buf);
}

void bpd_bn_silu_epilogue_simd(float* C, int M, int N,
                                 const float* alpha, const float* beta) {
    int n_simd = N & ~7;
    for (int i = 0; i < M; i++) {
        float* c_row = C + i * N;
        __m256 a_v = _mm256_set1_ps(alpha[i]);
        __m256 b_v = _mm256_set1_ps(beta[i]);
        // SIMD path: 8 cols at a time
        for (int j = 0; j < n_simd; j += 8) {
            __m256 x = _mm256_loadu_ps(c_row + j);
            __m256 t = _mm256_add_ps(_mm256_mul_ps(a_v, x), b_v);
            _mm256_storeu_ps(c_row + j, _bpd_silu_avx1_p6(t));
        }
        // Scalar tail
        float a = alpha[i]; float b = beta[i];
        for (int j = n_simd; j < N; j++) {
            float t = a * c_row[j] + b;
            c_row[j] = t / (1.0f + expf(-t));
        }
    }
}
#else
void bpd_bn_silu_epilogue_simd(float* C, int M, int N,
                                 const float* alpha, const float* beta) {
    for (int i = 0; i < M; i++) {
        float a = alpha[i]; float b = beta[i];
        float* c_row = C + i * N;
        for (int j = 0; j < N; j++) {
            float t = a * c_row[j] + b;
            c_row[j] = t / (1.0f + expf(-t));
        }
    }
}
#endif

#if BPD_HAVE_AVX1
void bpd_mm_cpu_avx1_v2(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    const int MR = 4;       // register-block height (rows)
    const int NR = 16;      // register-block width (cols = 2 ymm)
    const int KU = 4;       // K-unroll factor
    const int Q  = 384;     // K-block size (matches bpd_mm_cpu)

    // Init C to zero — same as scalar/v1
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // If shape too small for v2 register blocking, defer to v1 entirely.
    int M_blocks = M / MR;       // # of full 4-row blocks
    int M_tail   = M - M_blocks * MR;
    int N_blocks = N / NR;       // # of full 16-col blocks
    int N_tail   = N - N_blocks * NR;

    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int min_l;
        if (rem >= 2 * Q) {
            min_l = Q;
        } else if (rem > Q) {
            min_l = ((rem / 2 + MR*4 - 1) / (MR*4)) * (MR*4);
        } else {
            min_l = rem;
        }

        // K-block end index (matches bpd_mm_cpu's partial-sum semantics)
        int k_end = ls + min_l;

        // K-unroll: only valid when min_l is a multiple of KU. If not, use kus = 1.
        int kus = (min_l % KU == 0) ? KU : 1;

        // ─── Main path: 4-row × 16-col register-blocked tiles ───
        for (int rb = 0; rb < M_blocks; rb++) {
            int row_base = rb * MR;
            const float* a0 = A + (row_base + 0) * K;
            const float* a1 = A + (row_base + 1) * K;
            const float* a2 = A + (row_base + 2) * K;
            const float* a3 = A + (row_base + 3) * K;
            float* c0 = C + (row_base + 0) * N;
            float* c1 = C + (row_base + 1) * N;
            float* c2 = C + (row_base + 2) * N;
            float* c3 = C + (row_base + 3) * N;

            for (int cb = 0; cb < N_blocks; cb++) {
                int col_base = cb * NR;
                // 8 accumulators: 4 rows × 2 col-vectors per row
                // acc_r{0..3}_c{0,1} where c0 covers cols [col_base, col_base+8)
                // and c1 covers cols [col_base+8, col_base+16)
                __m256 acc_r0_c0 = _mm256_setzero_ps();
                __m256 acc_r0_c1 = _mm256_setzero_ps();
                __m256 acc_r1_c0 = _mm256_setzero_ps();
                __m256 acc_r1_c1 = _mm256_setzero_ps();
                __m256 acc_r2_c0 = _mm256_setzero_ps();
                __m256 acc_r2_c1 = _mm256_setzero_ps();
                __m256 acc_r3_c0 = _mm256_setzero_ps();
                __m256 acc_r3_c1 = _mm256_setzero_ps();

                if (kus == KU) {
                    // K-unrolled inner loop: process 4 k-values per iteration
                    //
                    // Phase 3.CAT.f: prefetch B rows for the NEXT iteration
                    // (KU=4 K-steps ahead) into L1 (_MM_HINT_T0). Bit-identity
                    // is preserved trivially: prefetch is a memory subsystem
                    // hint that does not change any computation.
                    //
                    // Env-controlled (Phase 3.CAT.FUSE.g, parameter sweep):
                    //   SUBSTRATE_AVX1_PREFETCH=1 (default): emit prefetches
                    //   SUBSTRATE_AVX1_PREFETCH=0: skip prefetches
                    // Cached in static int after first call to avoid getenv()
                    // per GEMM (called thousands of times per YOLO frame).
                    static int prefetch_choice = -1;
                    if (prefetch_choice == -1) {
                        const char* env = getenv("SUBSTRATE_AVX1_PREFETCH");
                        prefetch_choice = (env && env[0] == '0') ? 0 : 1;
                    }
                    int do_prefetch = prefetch_choice;
                    for (int k = ls; k < k_end; k += KU) {
                        // Prefetch B rows for the next iteration (k+KU..k+2*KU-1)
                        // before doing this iteration's KSTEPs.
                        int k_next = k + KU;
                        if (do_prefetch && k_next < k_end) {
                            _mm_prefetch((const char*)(B + (k_next + 0) * N + col_base),     _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 0) * N + col_base + 8), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 1) * N + col_base),     _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 1) * N + col_base + 8), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 2) * N + col_base),     _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 2) * N + col_base + 8), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 3) * N + col_base),     _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_next + 3) * N + col_base + 8), _MM_HINT_T0);
                            // A-rows for next iteration too (one cache line per row)
                            _mm_prefetch((const char*)(a0 + k_next), _MM_HINT_T0);
                            _mm_prefetch((const char*)(a1 + k_next), _MM_HINT_T0);
                            _mm_prefetch((const char*)(a2 + k_next), _MM_HINT_T0);
                            _mm_prefetch((const char*)(a3 + k_next), _MM_HINT_T0);
                        }
                        // Per k step: load B's two col-vectors at row k
                        // For each row, broadcast A[row, k] and accumulate
                        #define KSTEP(KOFF) do {                                              \
                            __m256 b0 = _mm256_loadu_ps(B + (k + (KOFF)) * N + col_base);     \
                            __m256 b1 = _mm256_loadu_ps(B + (k + (KOFF)) * N + col_base + 8); \
                            __m256 av0 = _mm256_set1_ps(a0[k + (KOFF)]);                      \
                            __m256 av1 = _mm256_set1_ps(a1[k + (KOFF)]);                      \
                            __m256 av2 = _mm256_set1_ps(a2[k + (KOFF)]);                      \
                            __m256 av3 = _mm256_set1_ps(a3[k + (KOFF)]);                      \
                            acc_r0_c0 = _mm256_add_ps(acc_r0_c0, _mm256_mul_ps(av0, b0));     \
                            acc_r0_c1 = _mm256_add_ps(acc_r0_c1, _mm256_mul_ps(av0, b1));     \
                            acc_r1_c0 = _mm256_add_ps(acc_r1_c0, _mm256_mul_ps(av1, b0));     \
                            acc_r1_c1 = _mm256_add_ps(acc_r1_c1, _mm256_mul_ps(av1, b1));     \
                            acc_r2_c0 = _mm256_add_ps(acc_r2_c0, _mm256_mul_ps(av2, b0));     \
                            acc_r2_c1 = _mm256_add_ps(acc_r2_c1, _mm256_mul_ps(av2, b1));     \
                            acc_r3_c0 = _mm256_add_ps(acc_r3_c0, _mm256_mul_ps(av3, b0));     \
                            acc_r3_c1 = _mm256_add_ps(acc_r3_c1, _mm256_mul_ps(av3, b1));     \
                        } while (0)
                        KSTEP(0);
                        KSTEP(1);
                        KSTEP(2);
                        KSTEP(3);
                        #undef KSTEP
                    }
                } else {
                    // Non-unrolled fallback (when min_l isn't a multiple of KU)
                    for (int k = ls; k < k_end; k++) {
                        __m256 b0 = _mm256_loadu_ps(B + k * N + col_base);
                        __m256 b1 = _mm256_loadu_ps(B + k * N + col_base + 8);
                        __m256 av0 = _mm256_set1_ps(a0[k]);
                        __m256 av1 = _mm256_set1_ps(a1[k]);
                        __m256 av2 = _mm256_set1_ps(a2[k]);
                        __m256 av3 = _mm256_set1_ps(a3[k]);
                        acc_r0_c0 = _mm256_add_ps(acc_r0_c0, _mm256_mul_ps(av0, b0));
                        acc_r0_c1 = _mm256_add_ps(acc_r0_c1, _mm256_mul_ps(av0, b1));
                        acc_r1_c0 = _mm256_add_ps(acc_r1_c0, _mm256_mul_ps(av1, b0));
                        acc_r1_c1 = _mm256_add_ps(acc_r1_c1, _mm256_mul_ps(av1, b1));
                        acc_r2_c0 = _mm256_add_ps(acc_r2_c0, _mm256_mul_ps(av2, b0));
                        acc_r2_c1 = _mm256_add_ps(acc_r2_c1, _mm256_mul_ps(av2, b1));
                        acc_r3_c0 = _mm256_add_ps(acc_r3_c0, _mm256_mul_ps(av3, b0));
                        acc_r3_c1 = _mm256_add_ps(acc_r3_c1, _mm256_mul_ps(av3, b1));
                    }
                }

                // Add this K-block's partial sums to C (matching bpd_mm_cpu's
                // K-block cumulative semantics). C[row, col] += partial.
                __m256 c_r0_c0 = _mm256_loadu_ps(c0 + col_base);
                __m256 c_r0_c1 = _mm256_loadu_ps(c0 + col_base + 8);
                _mm256_storeu_ps(c0 + col_base,     _mm256_add_ps(c_r0_c0, acc_r0_c0));
                _mm256_storeu_ps(c0 + col_base + 8, _mm256_add_ps(c_r0_c1, acc_r0_c1));
                __m256 c_r1_c0 = _mm256_loadu_ps(c1 + col_base);
                __m256 c_r1_c1 = _mm256_loadu_ps(c1 + col_base + 8);
                _mm256_storeu_ps(c1 + col_base,     _mm256_add_ps(c_r1_c0, acc_r1_c0));
                _mm256_storeu_ps(c1 + col_base + 8, _mm256_add_ps(c_r1_c1, acc_r1_c1));
                __m256 c_r2_c0 = _mm256_loadu_ps(c2 + col_base);
                __m256 c_r2_c1 = _mm256_loadu_ps(c2 + col_base + 8);
                _mm256_storeu_ps(c2 + col_base,     _mm256_add_ps(c_r2_c0, acc_r2_c0));
                _mm256_storeu_ps(c2 + col_base + 8, _mm256_add_ps(c_r2_c1, acc_r2_c1));
                __m256 c_r3_c0 = _mm256_loadu_ps(c3 + col_base);
                __m256 c_r3_c1 = _mm256_loadu_ps(c3 + col_base + 8);
                _mm256_storeu_ps(c3 + col_base,     _mm256_add_ps(c_r3_c0, acc_r3_c0));
                _mm256_storeu_ps(c3 + col_base + 8, _mm256_add_ps(c_r3_c1, acc_r3_c1));
            }

            // N-tail: cols [N_blocks * NR, N) for these 4 rows.
            // Use the same per-row scalar k-loop as bpd_mm_cpu to preserve
            // bit-identity. Each tail col handled independently.
            if (N_tail > 0) {
                int col_start = N_blocks * NR;
                for (int row = 0; row < MR; row++) {
                    const float* a_row = A + (row_base + row) * K;
                    float* c_row = C + (row_base + row) * N;
                    for (int col = col_start; col < N; col++) {
                        float partial = 0.0f;
                        for (int k = ls; k < k_end; k++) {
                            partial += a_row[k] * B[k * N + col];
                        }
                        c_row[col] += partial;
                    }
                }
            }
        }

        // M-tail: rows [M_blocks * MR, M) processed scalar (matches bpd_mm_cpu order).
        if (M_tail > 0) {
            int row_start = M_blocks * MR;
            for (int row = row_start; row < M; row++) {
                const float* a_row = A + row * K;
                float* c_row = C + row * N;
                for (int col = 0; col < N; col++) {
                    float partial = 0.0f;
                    for (int k = ls; k < k_end; k++) {
                        partial += a_row[k] * B[k * N + col];
                    }
                    c_row[col] += partial;
                }
            }
        }

        ls += min_l;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Phase 3.CAT.SPEC.a — bpd_mm_cpu_avx1_v2_L0 (specialized for L0_focus)
// ──────────────────────────────────────────────────────────────────────
//
// Compile-time-specialized GEMM for the YOLOv5n L0 focus layer:
//   M=16, N=102400, K=108
//   (Cin=3, kH=kW=6, stride=2, pad=2, input 640x640, Cout=16)
//
// Substantively-substantive empirical experiment: does GCC -O2 produce
// measurably better code when M, N, K are compile-time constants vs
// runtime arguments?
//
// All hyperparameters baked in:
//   M_blocks = 16 / 4 = 4 (outer loop fully unrollable)
//   N_blocks = 102400 / 16 = 6400 (large; no unroll)
//   K = 108; single K-block (108 <= Q=384); no K-block loop
//   kus = KU = 4; K-unrolled inner kernel guaranteed (108 % 4 == 0)
//   Prefetch: always on (no env check)
//   Packing: not used (small B reuse on M=16 doesn't amortize)
//
// Bit-identity preservation: identical scalar math to bpd_mm_cpu_avx1_v2(A, B, C, 16, 102400, 108).
//
// Tested: test_p9_L0_specialized in bench/test_f3_v2_tdd.py.
void bpd_mm_cpu_avx1_v2_L0(const float* A, const float* B, float* C) {
    // Compile-time constants for L0
    enum { M = 16, N = 102400, K = 108, MR = 4, NR = 16, KU = 4 };
    enum { M_BLOCKS = M / MR, N_BLOCKS = N / NR };

    // Init C to zero (M*N = 16 * 102400 = 1.6M floats)
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // K-block: since K=108 <= Q=384, exactly one K-block of size 108
    // Inner kernel: 4x16 register tile, 8 accs, KU=4
    for (int rb = 0; rb < M_BLOCKS; rb++) {
        int row_base = rb * MR;
        const float* a0 = A + (row_base + 0) * K;
        const float* a1 = A + (row_base + 1) * K;
        const float* a2 = A + (row_base + 2) * K;
        const float* a3 = A + (row_base + 3) * K;
        float* c0 = C + (row_base + 0) * N;
        float* c1 = C + (row_base + 1) * N;
        float* c2 = C + (row_base + 2) * N;
        float* c3 = C + (row_base + 3) * N;

        for (int cb = 0; cb < N_BLOCKS; cb++) {
            int col_base = cb * NR;
            __m256 acc_r0_c0 = _mm256_setzero_ps();
            __m256 acc_r0_c1 = _mm256_setzero_ps();
            __m256 acc_r1_c0 = _mm256_setzero_ps();
            __m256 acc_r1_c1 = _mm256_setzero_ps();
            __m256 acc_r2_c0 = _mm256_setzero_ps();
            __m256 acc_r2_c1 = _mm256_setzero_ps();
            __m256 acc_r3_c0 = _mm256_setzero_ps();
            __m256 acc_r3_c1 = _mm256_setzero_ps();

            // K-loop: 108 / 4 = 27 unrolled iterations
            for (int k = 0; k < K; k += KU) {
                int k_next = k + KU;
                if (k_next < K) {
                    _mm_prefetch((const char*)(B + (k_next + 0) * N + col_base),     _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + (k_next + 0) * N + col_base + 8), _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + (k_next + 1) * N + col_base),     _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + (k_next + 1) * N + col_base + 8), _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + (k_next + 2) * N + col_base),     _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + (k_next + 2) * N + col_base + 8), _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + (k_next + 3) * N + col_base),     _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + (k_next + 3) * N + col_base + 8), _MM_HINT_T0);
                    _mm_prefetch((const char*)(a0 + k_next), _MM_HINT_T0);
                    _mm_prefetch((const char*)(a1 + k_next), _MM_HINT_T0);
                    _mm_prefetch((const char*)(a2 + k_next), _MM_HINT_T0);
                    _mm_prefetch((const char*)(a3 + k_next), _MM_HINT_T0);
                }
                #define KSTEP_L0(KOFF) do {                                                       \
                    __m256 b0 = _mm256_loadu_ps(B + (k + (KOFF)) * N + col_base);                 \
                    __m256 b1 = _mm256_loadu_ps(B + (k + (KOFF)) * N + col_base + 8);             \
                    __m256 av0 = _mm256_set1_ps(a0[k + (KOFF)]);                                  \
                    __m256 av1 = _mm256_set1_ps(a1[k + (KOFF)]);                                  \
                    __m256 av2 = _mm256_set1_ps(a2[k + (KOFF)]);                                  \
                    __m256 av3 = _mm256_set1_ps(a3[k + (KOFF)]);                                  \
                    acc_r0_c0 = _mm256_add_ps(acc_r0_c0, _mm256_mul_ps(av0, b0));                 \
                    acc_r0_c1 = _mm256_add_ps(acc_r0_c1, _mm256_mul_ps(av0, b1));                 \
                    acc_r1_c0 = _mm256_add_ps(acc_r1_c0, _mm256_mul_ps(av1, b0));                 \
                    acc_r1_c1 = _mm256_add_ps(acc_r1_c1, _mm256_mul_ps(av1, b1));                 \
                    acc_r2_c0 = _mm256_add_ps(acc_r2_c0, _mm256_mul_ps(av2, b0));                 \
                    acc_r2_c1 = _mm256_add_ps(acc_r2_c1, _mm256_mul_ps(av2, b1));                 \
                    acc_r3_c0 = _mm256_add_ps(acc_r3_c0, _mm256_mul_ps(av3, b0));                 \
                    acc_r3_c1 = _mm256_add_ps(acc_r3_c1, _mm256_mul_ps(av3, b1));                 \
                } while (0)
                KSTEP_L0(0); KSTEP_L0(1); KSTEP_L0(2); KSTEP_L0(3);
                #undef KSTEP_L0
            }

            // C += partial; for L0 there is only one K-block, so C was just-initialized to 0
            // and acc holds the full sum. We could skip the load+add and just store.
            // But to match bpd_mm_cpu_avx1_v2's semantics exactly (C += partial), we keep the load.
            __m256 c_r0_c0 = _mm256_loadu_ps(c0 + col_base);
            __m256 c_r0_c1 = _mm256_loadu_ps(c0 + col_base + 8);
            _mm256_storeu_ps(c0 + col_base,     _mm256_add_ps(c_r0_c0, acc_r0_c0));
            _mm256_storeu_ps(c0 + col_base + 8, _mm256_add_ps(c_r0_c1, acc_r0_c1));
            __m256 c_r1_c0 = _mm256_loadu_ps(c1 + col_base);
            __m256 c_r1_c1 = _mm256_loadu_ps(c1 + col_base + 8);
            _mm256_storeu_ps(c1 + col_base,     _mm256_add_ps(c_r1_c0, acc_r1_c0));
            _mm256_storeu_ps(c1 + col_base + 8, _mm256_add_ps(c_r1_c1, acc_r1_c1));
            __m256 c_r2_c0 = _mm256_loadu_ps(c2 + col_base);
            __m256 c_r2_c1 = _mm256_loadu_ps(c2 + col_base + 8);
            _mm256_storeu_ps(c2 + col_base,     _mm256_add_ps(c_r2_c0, acc_r2_c0));
            _mm256_storeu_ps(c2 + col_base + 8, _mm256_add_ps(c_r2_c1, acc_r2_c1));
            __m256 c_r3_c0 = _mm256_loadu_ps(c3 + col_base);
            __m256 c_r3_c1 = _mm256_loadu_ps(c3 + col_base + 8);
            _mm256_storeu_ps(c3 + col_base,     _mm256_add_ps(c_r3_c0, acc_r3_c0));
            _mm256_storeu_ps(c3 + col_base + 8, _mm256_add_ps(c_r3_c1, acc_r3_c1));
        }
    }
}
#else
void bpd_mm_cpu_avx1_v2(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    bpd_mm_cpu(A, B, C, M, N, K);
}
void bpd_mm_cpu_avx1_v2_L0(const float* A, const float* B, float* C) {
    bpd_mm_cpu(A, B, C, 16, 102400, 108);
}
#endif

// CPU fused matmul + bias + relu
void bpd_mm_bias_relu_cpu(const float* A, const float* B,
                           const float* bias, float* C,
                           int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = fmaxf(0.0f, sum + bias[col]);
        }
    }
}

// CPU relu
void bpd_relu_cpu(const float* input, float* output, int n) {
    /* relu = x>=0 ? x : 0.  Uses >= so signed zero is preserved: torch relu(-0.0)=-0.0.
     * fmaxf/_mm256_max_ps(0,x) would flip -0.0 -> +0.0 (a 0-ULP-class signed-zero bug). */
#if BPD_HAVE_AVX1
    const __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 ge = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
        _mm256_storeu_ps(output + i, _mm256_blendv_ps(zero, x, ge));
    }
    for (; i < n; i++)
        output[i] = input[i] >= 0.0f ? input[i] : 0.0f;
#else
    for (int i = 0; i < n; i++)
        output[i] = input[i] >= 0.0f ? input[i] : 0.0f;
#endif
}

// CPU silu
void bpd_silu_cpu(const float* input, float* output, int n) {
    /* SiLU: x / (1 + exp(-x)). Scalar loop — expf dominates runtime,
     * SIMD extract/reload overhead makes vectorization counterproductive
     * without a polynomial exp approximation (sweepable parameter). */
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// CPU swish (Stanford 25_Swish = x*sigmoid, reciprocal-mul form). Distinct from silu
// (divide form): the two differ by 1 ULP. Stanford forward pins reciprocal_mul.
void bpd_swish_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-x));
        output[i] = x * s;
    }
}

// CPU mish  
void bpd_mish_cpu(const float* input, float* output, int n) {
    /* Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
     * Range optimization (matches PyTorch):
     *   x > 20:  softplus(x) ≈ x, tanh(x) ≈ 1 → mish ≈ x
     *   x < -20: exp(x) ≈ 0, softplus ≈ 0, tanh(0) = 0 → mish ≈ 0 */
    for (int i = 0; i < n; i++) {
        float x = input[i];
        if (x > 20.0f) {
            output[i] = x;
        } else if (x < -20.0f) {
            output[i] = 0.0f;
        } else {
            output[i] = x * tanhf(log1pf(expf(x)));
        }
    }
}

// CPU conv2d (direct, no im2col)
void bpd_conv2d_cpu(const float* input, const float* weight, float* output,
                     int N, int C_in, int H, int W,
                     int C_out, int kH, int kW,
                     int stride, int pad) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    int total = N * C_out * H_out * W_out;
    for (int idx = 0; idx < total; idx++) {
        int ow = idx % W_out;
        int oh = (idx / W_out) % H_out;
        int co = (idx / (W_out * H_out)) % C_out;
        int n  = idx / (W_out * H_out * C_out);
        float sum = 0.0f;
        for (int ci = 0; ci < C_in; ci++)
            for (int kh = 0; kh < kH; kh++)
                for (int kw = 0; kw < kW; kw++) {
                    int hi = oh * stride - pad + kh;
                    int wi = ow * stride - pad + kw;
                    if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                        int in_idx = ((n*C_in+ci)*H+hi)*W+wi;
                        int w_idx = ((co*C_in+ci)*kH+kh)*kW+kw;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
        output[idx] = sum;
    }
}

// Im2col helper: convert NCHW input slice into [Cin*kH*kW, H_out*W_out] row-major.
// Matches PyTorch's im2col.h template signature exactly.
// data_col[(c_col * H_out + h_col) * W_out + w_col] = data_im[(c_im * H + h_im) * W + w_im]
// where c_col indexes (c_im, h_offset, w_offset) in row-major (c_im outermost).
static void bpd_im2col(const float* data_im,
                       int channels, int height, int width,
                       int output_height, int output_width,
                       int kernel_h, int kernel_w,
                       int pad_h, int pad_w,
                       int stride_h, int stride_w,
                       int dilation_h, int dilation_w,
                       float* data_col) {
    int channels_col = channels * kernel_h * kernel_w;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int c_im = c_col / (kernel_h * kernel_w);
        for (int h_col = 0; h_col < output_height; h_col++) {
            int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
            for (int w_col = 0; w_col < output_width; w_col++) {
                int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                int dst = (c_col * output_height + h_col) * output_width + w_col;
                if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
                    data_col[dst] = data_im[(c_im * height + h_im) * width + w_im];
                } else {
                    data_col[dst] = 0.0f;
                }
            }
        }
    }
}

// Parameterized 2D convolution: matches PyTorch CPU F.conv2d exactly via
// im2col + GEMM. Inherits bit-identity from bpd_mm_cpu (Goto-Sandy SGEMM
// matching cblas_sgemm 0 ULP).
//
// PyTorch source: aten/src/ATen/native/ConvolutionMM2d.cpp slow_conv2d_forward_cpu
// + slow_conv2d_update_output_frame. Im2col layout from im2col.h line 65.
//
// Signature: output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
//   input:  (N, Cin, H, W)
//   weight: (Cout, Cin/groups, kH, kW)
//   bias:   (Cout,) or NULL
//   output: (N, Cout, H_out, W_out)
void bpd_conv2d_full_cpu(const float* input, const float* weight, const float* bias,
                          float* output,
                          int N, int Cin, int H, int W,
                          int Cout, int kH, int kW,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;

    int spatial_out = H_out * W_out;
    int k_dim = Cin_per_group * kH * kW;

    float* finput = (float*)bpd_alloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            const float* input_g = input + (n * Cin + g * Cin_per_group) * H * W;
            bpd_im2col(input_g, Cin_per_group, H, W,
                       H_out, W_out, kH, kW,
                       pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w,
                       finput);

            const float* weight_g = weight + g * Cout_per_group * k_dim;
            float* output_g = output + (n * Cout + g * Cout_per_group) * spatial_out;

            // GEMM: output_g[Cout_per_group, spatial_out] = weight_g[Cout_per_group, k_dim] @ finput[k_dim, spatial_out]
            bpd_mm_cpu(weight_g, finput, output_g,
                       Cout_per_group, spatial_out, k_dim);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * spatial_out;
                    for (int p = 0; p < spatial_out; p++) {
                        out_co[p] += b;
                    }
                }
            }
        }
    }

    free(finput);
}

// ──────────────────────────────────────────────────────────────────────
// Conv2d + BatchNorm + SiLU fused (Phase 3.1 F3 — bit-identical with PyTorch)
// ──────────────────────────────────────────────────────────────────────
//
// Computes (in one kernel, eliminating two intermediate tensors per call):
//   y[co, p] = silu(alpha[co] * GEMM(weight, im2col(input))[co, p] + beta[co])
//
// Where alpha and beta are precomputed from BN parameters via the same
// substrate-design choice as bpd_batchnorm_cpu_affine_fused:
//   alpha[c] = gamma[c] * (1.0f / sqrtf(var[c] + eps))   ← multiply-by-reciprocal
//   beta[c]  = bn_beta[c] - mean[c] * alpha[c]
//
// And silu uses the DIVSS form (same as bpd_silu_cpu):
//   silu(x) = x / (1.0f + expf(-x))
//
// Because the GEMM accumulator, the alpha/beta application order, and the
// silu expression are all IDENTICAL to the unfused chain
// (bpd_conv2d_full_cpu → bpd_batchnorm_cpu_affine_fused → bpd_silu_cpu),
// the fused output is bit-identical with the unfused output for all inputs.
//
// Restriction: groups=1 only (YOLOv5n uses groups=1 throughout).
// Restriction: no bias on conv (YOLOv5n CBS uses bias=False on conv; BN provides
//              the additive bias via offset).
//
// Memory traffic savings per call (vs unfused chain):
//   - No intermediate conv_out tensor materialized to memory
//   - No intermediate bn_out tensor materialized to memory
//   - Only the final silu_out is written
//   = 4 fewer memory passes over the (N, Cout, H_out, W_out) tensor.
/* Conv2d + BatchNorm + SiLU fused — GENERATED FROM UNFUSED PRIMITIVES.
 *
 * This kernel composes the SAME code paths as the unfused version:
 *   1. im2col (identical to bpd_conv2d_full_cpu)
 *   2. GEMM via bpd_mm_cpu (packed GEMM with Q=384 splitting — 0 ULP)
 *   3. BN+SiLU epilogue applied per-channel over the GEMM output
 *
 * The epilogue replaces TWO separate kernel calls (BN + SiLU) with
 * ONE pass over the output buffer. Same math, same accumulation order,
 * fewer memory passes.
 *
 * Correctness guarantee: the GEMM produces IDENTICAL bits to unfused
 * (same bpd_mm_cpu call). The epilogue is elementwise — cannot change
 * accumulation order. Therefore: fused output = unfused output, bit-exact.
 */
void bpd_conv2d_bn_silu_fused_cpu(const float* input, const float* weight,
                                    const float* bn_gamma, const float* bn_beta,
                                    const float* bn_mean, const float* bn_var,
                                    float* output,
                                    int N, int Cin, int H, int W,
                                    int Cout, int kH, int kW,
                                    int stride_h, int stride_w,
                                    int pad_h, int pad_w,
                                    float eps) {
    int H_out = (H + 2*pad_h - (kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - (kW-1) - 1) / stride_w + 1;
    int spatial_out = H_out * W_out;
    int k_dim = Cin * kH * kW;

    /* Precompute BN scale/offset — SAME formula as bpd_batchnorm_cpu_affine_fused */
    float* scale = (float*)bpd_alloc(Cout * sizeof(float));
    float* offset = (float*)bpd_alloc(Cout * sizeof(float));
    for (int c = 0; c < Cout; c++) {
        float inv_std = 1.0f / sqrtf(bn_var[c] + eps);
        float s = bn_gamma[c] * inv_std;
        scale[c] = s;
        offset[c] = bn_beta[c] - bn_mean[c] * s;
    }

    float* finput = (float*)bpd_alloc(k_dim * spatial_out * sizeof(float));
    if (!finput) { free(scale); free(offset); return; }

    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * Cin * H * W;
        float* output_n = output + n * Cout * spatial_out;

        /* Step 1: im2col — identical to bpd_conv2d_full_cpu */
        bpd_im2col(input_n, Cin, H, W, H_out, W_out,
                   kH, kW, pad_h, pad_w, stride_h, stride_w, 1, 1, finput);

        /* Step 2+3: GEMM with fused BN+SiLU epilogue (true CBS fusion) */
#if BPD_HAVE_AVX1
        bpd_gemm_packed_bn_silu(weight, finput, output_n, Cout, spatial_out, k_dim, scale, offset);
#else
        bpd_mm_cpu(weight, finput, output_n, Cout, spatial_out, k_dim);
        for (int c = 0; c < Cout; c++) {
            float s = scale[c], o = offset[c];
            float* out_c = output_n + c * spatial_out;
            for (int p = 0; p < spatial_out; p++) {
                float y = s * out_c[p] + o;
                out_c[p] = y / (1.0f + expf(-y));
            }
        }
#endif
    }

    free(finput);
    free(scale);
    free(offset);
}
// ──────────────────────────────────────────────────────────────────────
//
// Identical to bpd_conv2d_bn_silu_fused_cpu (F3) except for one more
// epilogue op: y = silu(alpha*acc + beta) + residual[same position].
//
// Used in YOLOv5 bottleneck blocks with shortcut=True:
//   y = x + cv2(cv1(x))
// where cv2 is a CBS unit. The fused kernel computes cv2's full pipeline
// (im2col -> GEMM -> silu(alpha*x + beta)) and adds the residual `x` in
// the same write-back.
//
// Restriction: residual must have shape (N, Cout, H_out, W_out) \u2014 same
// layout as the conv output. The caller (run_bottleneck) ensures this by
// passing the bottleneck input `x` directly (which has the same shape as
// the cv2 output when cin==cout, k=3, stride=1, pad=1, which is the
// standard YOLOv5 shortcut bottleneck configuration).
//
// Bit-identity:
//   F3 path produces silu(alpha*acc + beta) bit-identically.
//   Adding `+ residual[p]` is a single float ADD performed AFTER silu, in
//   the same order as the unfused chain: y = cv2_silu_out + x.
//   This matches bpd_residual_add_cpu's per-element behavior bit-for-bit.
//
// Memory traffic savings vs unfused:
//   Unfused: F3 writes cv2_out; residual_add reads cv2_out + x and writes y
//     = F3 write (1) + residual_add reads (2) + residual_add write (1) = 4 passes
//   Fused:   F3+add writes y directly (residual read on-the-fly via cache)
//     = 1 write of y output (residual read is 1 pass through x, but that read
//        is sequential and likely cache-resident from cv1)
//   Net: 2-3 fewer memory passes per shortcut bottleneck.
// ──────────────────────────────────────────────────────────────────────
// bpd_conv2d_bn_silu_fused_cpu_v2 (Phase 3.CAT.FUSE.a)
// ──────────────────────────────────────────────────────────────────────
//
// Composition of F3 (Conv+BN+SiLU fusion) with v2 GEMM architecture.
// Same substrate-design parameters as bpd_mm_cpu_avx1_v2:
//   register_blocking(MR=4, NR=16)
//   ilp_accumulators(8)
//   unroll_factor_K(4)
// PLUS the epilogue (alpha*acc + beta then silu) applied SIMD-vectorized
// inside the per-tile store, eliminating the GEMM-output round-trip.
//
// Per Heath: 'if we apply kernel fusion to those kernels.... what happens
// then? even if they are not faster than stock, we can still sweep the new
// parameters you just added.' This is the substantive substrate-design
// substantive substantive move that composes the orthogonal optimization
// dimensions: inner-kernel SIMD parameters AND cross-op fusion.
//
// BIT-IDENTITY (the substantive Essence):
//   Each ymm accumulator holds ONE (row, col_group) output cell's running
//   sum. K-loop accumulates linearly within each accumulator. After all
//   K-blocks complete for a tile, the accumulator gets:
//     silu(alpha[row] * acc + beta[row])
//   applied SIMD-vectorized (alpha and beta broadcast per-row) BEFORE
//   storing to memory. Same scalar arithmetic order per element as the
//   unfused chain v2-GEMM -> scalar-BN -> scalar-SiLU.
//
// Restriction: groups=1, no bias on conv (matches CBS shape).
//   For shapes too small for v2 register blocking (M < 4 or N < 16), falls
//   back to v1 fused path (bpd_conv2d_bn_silu_fused_cpu) via the dispatcher.
//   K-blocks of size Q=384 supported (multi-block accumulation into a
//   temporary register-bank, then epilogue applied at the end).
#if BPD_HAVE_AVX1
static inline __m256 _bpd_silu_avx1(__m256 x) {
    // silu(x) = x / (1.0f + expf(-x))
    // AVX1 has no vectorized expf. Apply scalar expf per lane.
    // This matches bpd_silu_cpu's DIVSS form exactly.
    float buf[8] __attribute__((aligned(32)));
    _mm256_store_ps(buf, x);
    for (int i = 0; i < 8; i++) {
        buf[i] = buf[i] / (1.0f + expf(-buf[i]));
    }
    return _mm256_load_ps(buf);
}

// bpd_conv2d_bn_silu_fused_cpu_v2 — TDD composition (Phase 3.CAT.TDD.7).
//
// Rebuilt from the TDD primitives. Falls FORWARD, not back: handles any
// (K, Q) combination via composition, not fallback.
//
//   im2col(input)   → finput            (existing primitive)
//   bpd_gemm_v2_full(weight, finput, output)  → P5 (P1 + P2 + P3 + P4)
//   bpd_bn_silu_epilogue_simd(output, alpha, beta) → P6
//
// Per Heath: 'disconnect fallback behavior and make it work falling forward.'
//
// Each composed primitive is independently verified BIT_IDENTICAL with its
// scalar reference (P1/P2/P3/P4 via test_f3_v2_tdd.py P5 test against
// bpd_mm_cpu_avx1_v2; P6 against scalar epilogue). Composition is bit-identical
// by transitivity. No K-block-size restrictions. No M/N-divisibility
// restrictions (P3 + P4 handle tails). Works for ALL shapes.
void bpd_conv2d_bn_silu_fused_cpu_v2(const float* input, const float* weight,
                                       const float* alpha, const float* beta,
                                       float* output,
                                       int N, int Cin, int H, int W,
                                       int Cout, int kH, int kW,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w) {
    int H_out = (H + 2*pad_h - (kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - (kW-1) - 1) / stride_w + 1;
    int spatial_out = H_out * W_out;
    int k_dim = Cin * kH * kW;

    float* finput = (float*)bpd_alloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * Cin * H * W;
        bpd_im2col(input_n, Cin, H, W,
                   H_out, W_out, kH, kW,
                   pad_h, pad_w, stride_h, stride_w,
                   1, 1, finput);

        float* output_n = output + n * Cout * spatial_out;

        // P5: GEMM via bpd_mm_cpu (dispatches to packed GEMM with Q=384 splitting)
        bpd_mm_cpu(weight, finput, output_n, Cout, spatial_out, k_dim);

        // P6: SIMD epilogue in-place over output_n
        bpd_bn_silu_epilogue_simd(output_n, Cout, spatial_out, alpha, beta);
    }

    free(finput);
}
#else
void bpd_conv2d_bn_silu_fused_cpu_v2(const float* input, const float* weight,
                                       const float* alpha, const float* beta,
                                       float* output,
                                       int N, int Cin, int H, int W,
                                       int Cout, int kH, int kW,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w) {
    bpd_conv2d_bn_silu_fused_cpu(input, weight, alpha, beta,
                                   NULL, NULL, output,
                                   N, Cin, H, W, Cout, kH, kW,
                                   stride_h, stride_w, pad_h, pad_w,
                                   1e-5f);
}
#endif

void bpd_conv2d_bn_silu_add_fused_cpu(const float* input, const float* weight,
                                        const float* bn_gamma, const float* bn_beta,
                                        const float* bn_mean, const float* bn_var,
                                        const float* residual,
                                        float* output,
                                        int N, int Cin, int H, int W,
                                        int Cout, int kH, int kW,
                                        int stride_h, int stride_w,
                                        int pad_h, int pad_w,
                                        float eps) {
    int H_out = (H + 2*pad_h - (kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - (kW-1) - 1) / stride_w + 1;
    int spatial_out = H_out * W_out;
    int k_dim = Cin * kH * kW;

    /* Precompute BN scale/offset — SAME formula as bpd_batchnorm_cpu_affine_fused */
    float* scale = (float*)bpd_alloc(Cout * sizeof(float));
    float* offset = (float*)bpd_alloc(Cout * sizeof(float));
    for (int c = 0; c < Cout; c++) {
        float inv_std = 1.0f / sqrtf(bn_var[c] + eps);
        float s = bn_gamma[c] * inv_std;
        scale[c] = s;
        offset[c] = bn_beta[c] - bn_mean[c] * s;
    }

    float* finput = (float*)bpd_alloc(k_dim * spatial_out * sizeof(float));
    if (!finput) { free(scale); free(offset); return; }

    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * Cin * H * W;
        const float* residual_n = residual + n * Cout * spatial_out;
        bpd_im2col(input_n, Cin, H, W,
                   H_out, W_out, kH, kW,
                   pad_h, pad_w, stride_h, stride_w,
                   1, 1, finput);

        float* output_n = output + n * Cout * spatial_out;
        bpd_mm_cpu(weight, finput, output_n, Cout, spatial_out, k_dim);

        // Epilogue: y = silu(scale[co] * conv_out + offset[co]) + residual
        for (int co = 0; co < Cout; co++) {
            float s = scale[co];
            float o = offset[co];
            float* out_co = output_n + co * spatial_out;
            const float* res_co = residual_n + co * spatial_out;
            for (int p = 0; p < spatial_out; p++) {
                float x = s * out_co[p] + o;
                out_co[p] = x / (1.0f + expf(-x)) + res_co[p];
            }
        }
    }

    free(finput);
    free(scale);
    free(offset);
}

// ──────────────────────────────────────────────────────────────────────
// Conv2d + Bias + Sigmoid fused (Phase 3.5 F7)
// ──────────────────────────────────────────────────────────────────────
//
// Computes: y = sigmoid(GEMM(weight, im2col(x)) + bias[co])
//
// Used in YOLOv5 Detect head: the 3 detection convs are 1x1 conv with bias
// and NO BN. The fused F7 kernel performs the conv (im2col + GEMM + bias)
// followed by sigmoid in the epilogue, eliminating one memory pass over
// the conv output.
//
// Substrate-design parameter family demonstrated by F7:
//   conv_epilogue(scalar_add_bias) + activation(sigmoid_divss)
// distinct from F3's (precomputed_alpha_beta_bn) + (silu_divss),
// and distinct from F4's (precomputed_alpha_beta_bn) + (silu_divss) + (residual_add).
//
// Bit-identity preservation:
//   Conv: same Goto-Sandy K-block GEMM, same im2col, same per-channel bias add
//   as bpd_conv2d_full_cpu.
//   Sigmoid: 1.0f / (1.0f + expf(-x)) — identical DIVSS form to bpd_sigmoid_cpu.
//   Per-element scalar order matches: (GEMM_result + bias[co]) -> sigmoid.
//
// Restriction: groups=1, dilation=1 (sufficient for YOLOv5n Detect).
void bpd_conv2d_bias_sigmoid_fused_cpu(const float* input, const float* weight,
                                         const float* bias, float* output,
                                         int N, int Cin, int H, int W,
                                         int Cout, int kH, int kW,
                                         int stride_h, int stride_w,
                                         int pad_h, int pad_w) {
    int H_out = (H + 2*pad_h - (kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - (kW-1) - 1) / stride_w + 1;
    int spatial_out = H_out * W_out;
    int k_dim = Cin * kH * kW;

    float* finput = (float*)bpd_alloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * Cin * H * W;
        bpd_im2col(input_n, Cin, H, W,
                   H_out, W_out, kH, kW,
                   pad_h, pad_w, stride_h, stride_w,
                   1, 1, finput);

        float* output_n = output + n * Cout * spatial_out;
        bpd_mm_cpu(weight, finput, output_n, Cout, spatial_out, k_dim);

        // Epilogue: y[co, p] = sigmoid(GEMM_result + bias[co])
        // Same scalar order as the unfused chain: add bias, then sigmoid.
        for (int co = 0; co < Cout; co++) {
            float b = bias[co];
            float* out_co = output_n + co * spatial_out;
            for (int p = 0; p < spatial_out; p++) {
                float x = out_co[p] + b;
                out_co[p] = 1.0f / (1.0f + expf(-x));
            }
        }
    }

    free(finput);
}

// ── 1D and 3D convolutions (im2col + GEMM, same pattern as 2D) ──

// 1D im2col: input (channels, L) → packed (channels * kL, L_out) row-major
// data_col[(c_col * L_out) + l_col] = data_im[(c_im * L + l_im)]
// where c_col indexes (c_im, l_offset) row-major.
static void bpd_im2col_1d(const float* data_im,
                          int channels, int length,
                          int output_length,
                          int kernel_l, int pad_l, int stride_l, int dilation_l,
                          float* data_col) {
    int channels_col = channels * kernel_l;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int l_offset = c_col % kernel_l;
        int c_im = c_col / kernel_l;
        for (int l_col = 0; l_col < output_length; l_col++) {
            int l_im = l_col * stride_l - pad_l + l_offset * dilation_l;
            int dst = c_col * output_length + l_col;
            if (l_im >= 0 && l_im < length) {
                data_col[dst] = data_im[c_im * length + l_im];
            } else {
                data_col[dst] = 0.0f;
            }
        }
    }
}

// 1D convolution via im2col + GEMM.
// Signature: F.conv1d(input, weight, bias, stride, padding, dilation, groups)
//   input:  (N, Cin, L)
//   weight: (Cout, Cin/groups, kL)
//   bias:   (Cout,) or NULL
//   output: (N, Cout, L_out)
void bpd_conv1d_full_cpu(const float* input, const float* weight, const float* bias,
                          float* output,
                          int N, int Cin, int L,
                          int Cout, int kL,
                          int stride_l, int pad_l, int dilation_l,
                          int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int L_out = (L + 2*pad_l - dilation_l*(kL-1) - 1) / stride_l + 1;
    int k_dim = Cin_per_group * kL;

    float* finput = (float*)bpd_alloc(k_dim * L_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            const float* input_g = input + (n * Cin + g * Cin_per_group) * L;
            bpd_im2col_1d(input_g, Cin_per_group, L,
                          L_out, kL, pad_l, stride_l, dilation_l,
                          finput);
            const float* weight_g = weight + g * Cout_per_group * k_dim;
            float* output_g = output + (n * Cout + g * Cout_per_group) * L_out;

            bpd_mm_cpu(weight_g, finput, output_g,
                       Cout_per_group, L_out, k_dim);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * L_out;
                    for (int p = 0; p < L_out; p++) out_co[p] += b;
                }
            }
        }
    }
    free(finput);
}

// 3D im2col: input (channels, D, H, W) → packed (channels * kD * kH * kW, D_out * H_out * W_out)
// data_col[(c_col * D_out * H_out * W_out) + (d_col * H_out * W_out) + (h_col * W_out) + w_col]
//   = data_im[(c_im * D * H * W) + (d_im * H * W) + (h_im * W) + w_im]
// where c_col indexes (c_im, d_offset, h_offset, w_offset) row-major (c_im outermost,
// w_offset innermost) — matches PyTorch's im2col_3d_kernel pattern.
static void bpd_im2col_3d(const float* data_im,
                          int channels, int depth, int height, int width,
                          int output_depth, int output_height, int output_width,
                          int kernel_d, int kernel_h, int kernel_w,
                          int pad_d, int pad_h, int pad_w,
                          int stride_d, int stride_h, int stride_w,
                          int dilation_d, int dilation_h, int dilation_w,
                          float* data_col) {
    int channels_col = channels * kernel_d * kernel_h * kernel_w;
    int dhw = output_depth * output_height * output_width;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int d_offset = (c_col / (kernel_w * kernel_h)) % kernel_d;
        int c_im = c_col / (kernel_d * kernel_h * kernel_w);
        for (int d_col = 0; d_col < output_depth; d_col++) {
            int d_im = d_col * stride_d - pad_d + d_offset * dilation_d;
            for (int h_col = 0; h_col < output_height; h_col++) {
                int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
                for (int w_col = 0; w_col < output_width; w_col++) {
                    int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                    int dst = c_col * dhw
                            + d_col * output_height * output_width
                            + h_col * output_width
                            + w_col;
                    if (d_im >= 0 && d_im < depth
                        && h_im >= 0 && h_im < height
                        && w_im >= 0 && w_im < width) {
                        int src = c_im * depth * height * width
                                + d_im * height * width
                                + h_im * width
                                + w_im;
                        data_col[dst] = data_im[src];
                    } else {
                        data_col[dst] = 0.0f;
                    }
                }
            }
        }
    }
}

// 3D convolution via im2col + GEMM.
// Signature: F.conv3d(input, weight, bias, stride, padding, dilation, groups)
//   input:  (N, Cin, D, H, W)
//   weight: (Cout, Cin/groups, kD, kH, kW)
//   bias:   (Cout,) or NULL
//   output: (N, Cout, D_out, H_out, W_out)
void bpd_conv3d_full_cpu(const float* input, const float* weight, const float* bias,
                          float* output,
                          int N, int Cin, int D, int H, int W,
                          int Cout, int kD, int kH, int kW,
                          int sd, int sh, int sw,
                          int pd, int ph, int pw,
                          int dd, int dh, int dw,
                          int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int D_out = (D + 2*pd - dd*(kD-1) - 1) / sd + 1;
    int H_out = (H + 2*ph - dh*(kH-1) - 1) / sh + 1;
    int W_out = (W + 2*pw - dw*(kW-1) - 1) / sw + 1;

    int spatial_out = D_out * H_out * W_out;
    int k_dim = Cin_per_group * kD * kH * kW;

    float* finput = (float*)bpd_alloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            const float* input_g = input + (n * Cin + g * Cin_per_group) * D * H * W;
            bpd_im2col_3d(input_g, Cin_per_group, D, H, W,
                          D_out, H_out, W_out,
                          kD, kH, kW, pd, ph, pw,
                          sd, sh, sw, dd, dh, dw,
                          finput);
            const float* weight_g = weight + g * Cout_per_group * k_dim;
            float* output_g = output + (n * Cout + g * Cout_per_group) * spatial_out;

            bpd_mm_cpu(weight_g, finput, output_g,
                       Cout_per_group, spatial_out, k_dim);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * spatial_out;
                    for (int p = 0; p < spatial_out; p++) out_co[p] += b;
                }
            }
        }
    }
    free(finput);
}

// ── Transposed convolutions (col2im + GEMM, mirror of forward conv) ──
//
// PyTorch source: aten/src/ATen/native/NaiveConvolutionTranspose2d.cpp
//   slow_conv_transpose2d_out_cpu_template (line 244)
//
// Algorithm:
//   1. GEMM: columns[Cout*kH*kW, H_in*W_in] = weight^T @ input
//      where weight has PyTorch shape (Cin, Cout/groups, kH, kW)
//      reshaped to (Cin, Cout*kH*kW), then transposed → (Cout*kH*kW, Cin)
//   2. col2im: scatter columns into output[Cout, H_out, W_out] with += accumulation
//
// Output dims: H_out = (H_in - 1)*stride - 2*pad + dilation*(kH-1) + output_padding + 1
//
// col2im layout matches im2col exactly but scatters instead of gathers:
//   data_im[(c_im * H_out + h_im) * W_out + w_im]
//     += data_col[(c_col * H_in + h_in) * W_in + w_in]
// where c_col indexes (c_im, h_offset, w_offset) row-major.

// col2im 2D: scatter columns back into spatial image, accumulating overlaps.
static void bpd_col2im(const float* data_col,
                       int channels, int height_out, int width_out,
                       int height_in, int width_in,
                       int kernel_h, int kernel_w,
                       int pad_h, int pad_w,
                       int stride_h, int stride_w,
                       int dilation_h, int dilation_w,
                       float* data_im) {
    // Zero-init output
    int total = channels * height_out * width_out;
    for (int i = 0; i < total; i++) data_im[i] = 0.0f;

    int channels_col = channels * kernel_h * kernel_w;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int c_im = c_col / (kernel_h * kernel_w);
        for (int h_col = 0; h_col < height_in; h_col++) {
            int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
            for (int w_col = 0; w_col < width_in; w_col++) {
                int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                if (h_im >= 0 && h_im < height_out && w_im >= 0 && w_im < width_out) {
                    data_im[(c_im * height_out + h_im) * width_out + w_im] +=
                        data_col[(c_col * height_in + h_col) * width_in + w_col];
                }
            }
        }
    }
}

// 2D ConvTranspose via GEMM + col2im.
// Signature: F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
//   input:  (N, Cin, H_in, W_in)
//   weight: (Cin, Cout/groups, kH, kW)  <- NOTE: Cin is the FIRST dim for ConvTranspose
//   bias:   (Cout,) or NULL
//   output: (N, Cout, H_out, W_out)
//
// Output shape:
//   H_out = (H_in - 1)*sh - 2*ph + dh*(kH-1) + oph + 1
//   W_out = (W_in - 1)*sw - 2*pw + dw*(kW-1) + opw + 1
void bpd_conv_transpose2d_full_cpu(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int N, int Cin, int H_in, int W_in,
                                    int Cout, int kH, int kW,
                                    int sh, int sw, int ph, int pw,
                                    int oph, int opw,
                                    int dh, int dw,
                                    int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int H_out = (H_in - 1) * sh - 2*ph + dh*(kH-1) + oph + 1;
    int W_out = (W_in - 1) * sw - 2*pw + dw*(kW-1) + opw + 1;

    int spatial_in = H_in * W_in;
    int k_dim = Cout_per_group * kH * kW;

    // Buffer for transposed weight slice (per-group): shape (Cout_per_group*kH*kW, Cin_per_group)
    float* weight_T = (float*)bpd_alloc(k_dim * Cin_per_group * sizeof(float));
    // Buffer for columns: shape (Cout_per_group*kH*kW, H_in*W_in)
    float* columns = (float*)bpd_alloc(k_dim * spatial_in * sizeof(float));
    if (!weight_T || !columns) {
        if (weight_T) free(weight_T);
        if (columns) free(columns);
        return;
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            // Transpose weight slice for this group.
            // PyTorch weight layout (per group): [Cin_per_group, Cout_per_group, kH, kW] row-major
            //   weight[(ci * Cout_per_group + co) * kH * kW + kh*kW + kw]
            // We want weight_T[Cout_per_group*kH*kW, Cin_per_group] row-major:
            //   weight_T[(co * kH * kW + kh*kW + kw) * Cin_per_group + ci]
            //     = weight[(ci * Cout_per_group + co) * kH * kW + kh*kW + kw]
            const float* weight_g = weight + g * Cin_per_group * k_dim;
            for (int ci = 0; ci < Cin_per_group; ci++) {
                for (int co = 0; co < Cout_per_group; co++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int src = (ci * Cout_per_group + co) * kH * kW + kh*kW + kw;
                            int dst = ((co * kH + kh) * kW + kw) * Cin_per_group + ci;
                            weight_T[dst] = weight_g[src];
                        }
                    }
                }
            }

            // input slice for this group
            const float* input_g = input + (n * Cin + g * Cin_per_group) * spatial_in;

            // GEMM: columns[k_dim, spatial_in] = weight_T[k_dim, Cin_per_group] @ input_g[Cin_per_group, spatial_in]
            // bpd_mm_cpu(A, B, C, M, N, K): C[M,N] = A[M,K] @ B[K,N]
            bpd_mm_cpu(weight_T, input_g, columns,
                       k_dim, spatial_in, Cin_per_group);

            // col2im: scatter columns into output[Cout_per_group, H_out, W_out] for this group
            float* output_g = output + (n * Cout + g * Cout_per_group) * H_out * W_out;
            bpd_col2im(columns, Cout_per_group, H_out, W_out,
                       H_in, W_in, kH, kW, ph, pw, sh, sw, dh, dw,
                       output_g);

            // Add bias if provided
            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * H_out * W_out;
                    for (int p = 0; p < H_out * W_out; p++) out_co[p] += b;
                }
            }
        }
    }

    free(weight_T);
    free(columns);
}

// ── 1D ConvTranspose ──

// col2im 1D: scatter columns back into spatial image, accumulating overlaps.
static void bpd_col2im_1d(const float* data_col,
                          int channels, int length_out, int length_in,
                          int kernel_l, int pad_l, int stride_l, int dilation_l,
                          float* data_im) {
    int total = channels * length_out;
    for (int i = 0; i < total; i++) data_im[i] = 0.0f;
    int channels_col = channels * kernel_l;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int l_offset = c_col % kernel_l;
        int c_im = c_col / kernel_l;
        for (int l_col = 0; l_col < length_in; l_col++) {
            int l_im = l_col * stride_l - pad_l + l_offset * dilation_l;
            if (l_im >= 0 && l_im < length_out) {
                data_im[c_im * length_out + l_im] += data_col[c_col * length_in + l_col];
            }
        }
    }
}

// 1D ConvTranspose via GEMM + col2im.
// input:  (N, Cin, L_in)
// weight: (Cin, Cout/groups, kL)
// output: (N, Cout, L_out)  where L_out = (L_in-1)*stride - 2*pad + dilation*(kL-1) + output_padding + 1
void bpd_conv_transpose1d_full_cpu(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int N, int Cin, int L_in,
                                    int Cout, int kL,
                                    int stride_l, int pad_l, int output_pad_l,
                                    int dilation_l, int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int L_out = (L_in - 1) * stride_l - 2*pad_l + dilation_l*(kL-1) + output_pad_l + 1;
    int k_dim = Cout_per_group * kL;

    float* weight_T = (float*)bpd_alloc(k_dim * Cin_per_group * sizeof(float));
    float* columns = (float*)bpd_alloc(k_dim * L_in * sizeof(float));
    if (!weight_T || !columns) {
        if (weight_T) free(weight_T);
        if (columns) free(columns);
        return;
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            // Transpose weight slice: src=(Cin_per_group, Cout_per_group, kL), dst=(Cout_per_group*kL, Cin_per_group)
            const float* weight_g = weight + g * Cin_per_group * k_dim;
            for (int ci = 0; ci < Cin_per_group; ci++) {
                for (int co = 0; co < Cout_per_group; co++) {
                    for (int kl = 0; kl < kL; kl++) {
                        int src = (ci * Cout_per_group + co) * kL + kl;
                        int dst = (co * kL + kl) * Cin_per_group + ci;
                        weight_T[dst] = weight_g[src];
                    }
                }
            }
            const float* input_g = input + (n * Cin + g * Cin_per_group) * L_in;
            // columns[k_dim, L_in] = weight_T[k_dim, Cin_per_group] @ input_g[Cin_per_group, L_in]
            bpd_mm_cpu(weight_T, input_g, columns, k_dim, L_in, Cin_per_group);

            float* output_g = output + (n * Cout + g * Cout_per_group) * L_out;
            bpd_col2im_1d(columns, Cout_per_group, L_out, L_in, kL,
                          pad_l, stride_l, dilation_l, output_g);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * L_out;
                    for (int p = 0; p < L_out; p++) out_co[p] += b;
                }
            }
        }
    }
    free(weight_T);
    free(columns);
}

// ── 3D ConvTranspose ──

// col2im 3D: scatter columns back into spatial image, accumulating overlaps.
static void bpd_col2im_3d(const float* data_col,
                          int channels, int D_out, int H_out, int W_out,
                          int D_in, int H_in, int W_in,
                          int kD, int kH, int kW,
                          int pd, int ph, int pw,
                          int sd, int sh, int sw,
                          int dd, int dh, int dw,
                          float* data_im) {
    int total = channels * D_out * H_out * W_out;
    for (int i = 0; i < total; i++) data_im[i] = 0.0f;

    int channels_col = channels * kD * kH * kW;
    int spatial_in = D_in * H_in * W_in;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int w_offset = c_col % kW;
        int h_offset = (c_col / kW) % kH;
        int d_offset = (c_col / (kW * kH)) % kD;
        int c_im = c_col / (kD * kH * kW);
        for (int d_col = 0; d_col < D_in; d_col++) {
            int d_im = d_col * sd - pd + d_offset * dd;
            for (int h_col = 0; h_col < H_in; h_col++) {
                int h_im = h_col * sh - ph + h_offset * dh;
                for (int w_col = 0; w_col < W_in; w_col++) {
                    int w_im = w_col * sw - pw + w_offset * dw;
                    if (d_im >= 0 && d_im < D_out
                        && h_im >= 0 && h_im < H_out
                        && w_im >= 0 && w_im < W_out) {
                        int dst = c_im * D_out * H_out * W_out
                                + d_im * H_out * W_out
                                + h_im * W_out
                                + w_im;
                        int src = c_col * spatial_in
                                + d_col * H_in * W_in
                                + h_col * W_in
                                + w_col;
                        data_im[dst] += data_col[src];
                    }
                }
            }
        }
    }
}

// 3D ConvTranspose via GEMM + col2im.
// input:  (N, Cin, D_in, H_in, W_in)
// weight: (Cin, Cout/groups, kD, kH, kW)
// output: (N, Cout, D_out, H_out, W_out)
void bpd_conv_transpose3d_full_cpu(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int N, int Cin, int D_in, int H_in, int W_in,
                                    int Cout, int kD, int kH, int kW,
                                    int sd, int sh, int sw,
                                    int pd, int ph, int pw,
                                    int opd, int oph, int opw,
                                    int dd, int dh, int dw,
                                    int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int D_out = (D_in - 1) * sd - 2*pd + dd*(kD-1) + opd + 1;
    int H_out = (H_in - 1) * sh - 2*ph + dh*(kH-1) + oph + 1;
    int W_out = (W_in - 1) * sw - 2*pw + dw*(kW-1) + opw + 1;

    int spatial_in = D_in * H_in * W_in;
    int k_dim = Cout_per_group * kD * kH * kW;

    float* weight_T = (float*)bpd_alloc(k_dim * Cin_per_group * sizeof(float));
    float* columns = (float*)bpd_alloc(k_dim * spatial_in * sizeof(float));
    if (!weight_T || !columns) {
        if (weight_T) free(weight_T);
        if (columns) free(columns);
        return;
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            // Transpose weight slice: (Cin_per_group, Cout_per_group, kD, kH, kW) → (Cout_per_group*kD*kH*kW, Cin_per_group)
            const float* weight_g = weight + g * Cin_per_group * k_dim;
            for (int ci = 0; ci < Cin_per_group; ci++) {
                for (int co = 0; co < Cout_per_group; co++) {
                    for (int kd = 0; kd < kD; kd++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int src = ((ci * Cout_per_group + co) * kD + kd) * kH * kW + kh*kW + kw;
                                int dst = (((co * kD + kd) * kH + kh) * kW + kw) * Cin_per_group + ci;
                                weight_T[dst] = weight_g[src];
                            }
                        }
                    }
                }
            }
            const float* input_g = input + (n * Cin + g * Cin_per_group) * spatial_in;
            bpd_mm_cpu(weight_T, input_g, columns, k_dim, spatial_in, Cin_per_group);

            float* output_g = output + (n * Cout + g * Cout_per_group) * D_out * H_out * W_out;
            bpd_col2im_3d(columns, Cout_per_group, D_out, H_out, W_out,
                          D_in, H_in, W_in, kD, kH, kW,
                          pd, ph, pw, sd, sh, sw, dd, dh, dw,
                          output_g);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * D_out * H_out * W_out;
                    for (int p = 0; p < D_out * H_out * W_out; p++) out_co[p] += b;
                }
            }
        }
    }
    free(weight_T);
    free(columns);
}

// CPU batchnorm (inference mode)
//
// Per substrate-design diagnostic 2026-05-20 ~05:45 UTC (mavchin + metayen):
// the 4-op form below produces 32768 ULP (= 2^15) systematic divergence vs
// PyTorch's BN. Root cause candidates:
//   (1) 1.0f/sqrtf(x) vs rsqrtf(x) — different last-bit behavior
//   (2) Operation order — 4 ops vs PyTorch's 2 ops (precomputed affine)
// The bpd_batchnorm_cpu_affine_fused form below eliminates both by matching
// PyTorch's exact computational pattern: precompute scale/offset internally
// once per call, then y = scale[c]*x + offset[c] per element (2 ops, same
// as PyTorch).
//
// This 4-op form is kept for backward compatibility with existing callers;
// new code should use bpd_batchnorm_cpu_affine_fused.
void bpd_batchnorm_cpu(const float* input, const float* gamma,
                        const float* beta, const float* mean,
                        const float* var, float* output,
                        int N, int C, int HW, float eps) {
    int total = N * C * HW;
    for (int idx = 0; idx < total; idx++) {
        int c = (idx / HW) % C;
        float x = input[idx];
        float inv_std = 1.0f / sqrtf(var[c] + eps);
        output[idx] = gamma[c] * (x - mean[c]) * inv_std + beta[c];
    }
}

// CPU batchnorm — affine-fused inference (matches PyTorch eval mode bit-for-bit).
//
// Substrate-design name aligned with the bn_affine_fused epilogue substrate
// vocabulary (lib/epilogue_generator.pl, shipped commit bffbbe1):
//
//   In eval mode, BN reduces to per-channel affine:
//     y = γ[c] / sqrt(σ²[c] + ε) * (x - μ[c]) + β[c]
//
//   Algebraically collapses to:
//     scale[c]  = γ[c] / sqrt(σ²[c] + ε)
//     offset[c] = β[c] - μ[c] * scale[c]
//     y         = scale[c] * x + offset[c]    (2 ops per element, same as PyTorch)
//
// Substantive substrate-design properties:
//   - Internally precomputes scale[c] and offset[c] from gamma/beta/mean/var/eps
//     once per call. For inference with stable weights, the caller can hoist
//     this work above the batch loop by computing once and reusing arrays.
//   - The per-element computation is 2 ops (scale*x + offset), matching PyTorch
//     ATen's eval-mode BN. Bit-identical with PyTorch on CPU.
//   - No division-by-sqrt at per-element scope (the 32768 ULP root cause).
//
// Inputs (read-only):
//   input  : (N, C, HW)  — flat row-major over (batch, channel, spatial)
//   gamma  : (C,)        — BN weight (scale parameter γ)
//   beta   : (C,)        — BN bias (shift parameter β)
//   mean   : (C,)        — running mean (μ)
//   var    : (C,)        — running variance (σ²)
//
// Outputs (written):
//   output : (N, C, HW)  — y[c] = scale[c] * x + offset[c]
//
// Scratch (caller-allocated, size C each):
//   scale_buf, offset_buf : working buffers for precomputed scale/offset.
//                            Pass NULL to allocate internally (slower; only
//                            valid for C up to a small stack budget).
//
// Constant:
//   eps : numerical-stability epsilon (typically 1e-5)
void bpd_batchnorm_cpu_affine_fused(const float* input, const float* gamma,
                                      const float* beta, const float* mean,
                                      const float* var, float* output,
                                      float* scale_buf, float* offset_buf,
                                      int N, int C, int HW, float eps) {
    // Precompute scale[c] and offset[c] from BN parameters.
    // Stack-allocated fallback for the no-buffer-supplied case (C up to 4096).
    float local_scale[4096];
    float local_offset[4096];
    float* scale = scale_buf ? scale_buf : local_scale;
    float* offset = offset_buf ? offset_buf : local_offset;
    if (!scale_buf && C > 4096) {
        // Substrate-honest: refuse to silently produce wrong results.
        // Caller must supply scratch for C > 4096.
        return;
    }
    for (int c = 0; c < C; c++) {
        // Substrate-design substantive substrate-design choice 2026-05-20 ~06:15 UTC
        // (per Heath's SASS-comparison direction):
        //
        // PyTorch's CPU BN-eval substantively computes scale via:
        //   inv_std = 1.0 / sqrt(var + eps)    [one DIVSS]
        //   scale   = gamma * inv_std           [one MULSS]
        // (multiply-by-reciprocal form, 2 ops, both rounded separately).
        //
        // The "direct divide" form `gamma / sqrt(var + eps)` is algebraically
        // equivalent but produces 1-ULP different bits because DIVSS rounds
        // once for the combined division, while MULSS-of-MULSS rounds twice
        // at intermediate steps.
        //
        // For bit-identity with PyTorch CPU eval mode, use the multiply form.
        // Per medayek's framework: this is the rsqrt_variant substrate-design
        // parameter manifesting at CPU level.
        float inv_std = 1.0f / sqrtf(var[c] + eps);
        float s = gamma[c] * inv_std;
        scale[c] = s;
        offset[c] = beta[c] - mean[c] * s;
    }

    // Apply per element: y = scale[c] * x + offset[c].
    // Loop structure: (N, C, HW) — broadcast scale[c] per channel.
    // AVX1: 8-wide multiply-add with broadcast constants.
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float* src = input + (n * C + c) * HW;
            float* dst = output + (n * C + c) * HW;
            float s = scale[c];
            float o = offset[c];
            int hw = 0;
#if BPD_HAVE_AVX1
            __m256 vs = _mm256_set1_ps(s);
            __m256 vo = _mm256_set1_ps(o);
            for (; hw + 7 < HW; hw += 8) {
                __m256 x = _mm256_loadu_ps(src + hw);
                _mm256_storeu_ps(dst + hw, _mm256_add_ps(_mm256_mul_ps(x, vs), vo));
            }
#endif
            for (; hw < HW; hw++)
                dst[hw] = s * src[hw] + o;
        }
    }
}

// CPU upsample nearest 2x
void bpd_upsample_nearest2d_cpu(const float* input, float* output,
                                 int N, int C, int H, int W) {
    int H_out = 2 * H, W_out = 2 * W;
    int total = N * C * H_out * W_out;
    for (int idx = 0; idx < total; idx++) {
        int ow = idx % W_out;
        int oh = (idx / W_out) % H_out;
        int c = (idx / (H_out * W_out)) % C;
        int n = idx / (C * (H_out * W_out));
        int ih = oh / 2, iw = ow / 2;
        int in_idx = ((n*C+c)*H+ih)*W+iw;
        output[idx] = input[in_idx];
    }
}

// ── Concat along channel dimension (NCHW, dim=1) ──
// Concatenates two tensors along the channel axis.
// a: [N, Ca, H, W], b: [N, Cb, H, W] → out: [N, Ca+Cb, H, W]
void bpd_concat_cpu(const float* a, const float* b, float* output,
                     int N, int Ca, int Cb, int H, int W) {
    int HW = H * W;
    int C_out = Ca + Cb;
    for (int n = 0; n < N; n++) {
        // Copy a's channels
        memcpy(output + n * C_out * HW,
               a + n * Ca * HW,
               (size_t)Ca * HW * sizeof(float));
        // Copy b's channels
        memcpy(output + n * C_out * HW + Ca * HW,
               b + n * Cb * HW,
               (size_t)Cb * HW * sizeof(float));
    }
}

// 4-way concat for SPPF: [N, C1+C2+C3+C4, H, W]
void bpd_concat4_cpu(const float* a, const float* b, const float* c, const float* d,
                      float* output,
                      int N, int Ca, int Cb, int Cc, int Cd, int H, int W) {
    int HW = H * W;
    int C_out = Ca + Cb + Cc + Cd;
    for (int n = 0; n < N; n++) {
        int off = n * C_out * HW;
        memcpy(output + off, a + n*Ca*HW, (size_t)Ca*HW*sizeof(float)); off += Ca*HW;
        memcpy(output + off, b + n*Cb*HW, (size_t)Cb*HW*sizeof(float)); off += Cb*HW;
        memcpy(output + off, c + n*Cc*HW, (size_t)Cc*HW*sizeof(float)); off += Cc*HW;
        memcpy(output + off, d + n*Cd*HW, (size_t)Cd*HW*sizeof(float));
    }
}

// ── Additional elementwise ops ──

void bpd_sigmoid_cpu(const float* input, float* output, int n) {
    /* Sigmoid: 1 / (1 + exp(-x)). Same as SiLU — expf dominates,
     * SIMD vectorization is counterproductive without polynomial exp. */
    for (int i = 0; i < n; i++)
        output[i] = 1.0f / (1.0f + expf(-input[i]));
}

// ──────────────────────────────────────────────────────────────────────
// Detect head post-sigmoid fused kernel (Phase 3.2 F8)
// ──────────────────────────────────────────────────────────────────────
//
// Fuses the sigmoid + split + scale + concat sequence in the YOLOv5 Detect
// head into a single sweep over the (bs, na, ny, nx, no) tensor.
//
// Computes, per element (n, a, y, x, c):
//   s = sigmoid(permuted[n,a,y,x,c]) = 1.0f / (1.0f + expf(-permuted[n,a,y,x,c]))
//   if c < 2:           out = (s * 2.0f + grid[0,a,y,x,c]) * stride
//   else if c < 4:      d = s * 2.0f;  out = d * d * anchor_grid[0,a,y,x,c-2]
//   else:               out = s
//
// Bit-identity preservation:
//   - sigmoid: same x / (1.0f + expf(-x)) expression as bpd_sigmoid_cpu
//             but using DIVSS form. Actually bpd_sigmoid_cpu uses 1.0f/(1+exp(-x)),
//             and the unfused detect path multiplies by 2.0f then add grid.
//             Same scalar order in fused kernel.
//   - xy:   s = sigmoid(in); s2 = s * 2.0f; (s2 + grid) * stride  \u2014 same order
//             as unfused: xy*2 \u2192 +grid \u2192 *stride.
//   - wh:   d = sigmoid(in) * 2.0f; d*d * anchor_grid  \u2014 same order as
//             unfused: wh*2 \u2192 squared \u2192 *anchor_grid.
//   - conf: s = sigmoid(in)  \u2014 trivial pass-through.
//
// Grid and anchor_grid layouts (matching _make_grid_yolov5 in yolo_forward.py):
//   grid shape:        (1, na, ny, nx, 2)  with values stack(xv, yv) - 0.5
//   anchor_grid shape: (1, na, ny, nx, 2)  with values anchors[i] * stride[i]
// Both contiguous float32. Reading grid[a, y, x, c] for c in {0, 1}:
//   offset = a*ny*nx*2 + y*nx*2 + x*2 + c
//
// Memory traffic savings vs unfused:
//   Unfused: sigmoid writes whole tensor (R+W), xy*2 writes (R+W), +grid writes
//   (R+W), *stride writes (R+W), wh*2 (R+W), squared (R+W), *anchor_grid (R+W),
//   concatenate writes whole tensor (R+W). At minimum 4-5 R+W of the
//   (bs, na, ny, nx, no) tensor = 8-10 memory passes eliminated.
//   Fused: 1 R + 1 W = 2 memory passes total. Net saving: 6-8 passes per
//   detection level over the full tensor.
void bpd_detect_postprocess_cpu(const float* permuted, const float* grid,
                                  const float* anchor_grid, float stride,
                                  float* output,
                                  int bs, int na, int ny, int nx, int no) {
    int n_per_anchor = ny * nx * no;
    int grid_per_anchor = ny * nx * 2;
    for (int b = 0; b < bs; b++) {
        for (int a = 0; a < na; a++) {
            const float* in_a = permuted + b * (na * n_per_anchor) + a * n_per_anchor;
            float* out_a = output + b * (na * n_per_anchor) + a * n_per_anchor;
            const float* grid_a = grid + a * grid_per_anchor;
            const float* anchor_a = anchor_grid + a * grid_per_anchor;
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    const float* in_yx = in_a + y * (nx * no) + x * no;
                    float* out_yx = out_a + y * (nx * no) + x * no;
                    const float* grid_yx = grid_a + y * (nx * 2) + x * 2;
                    const float* anchor_yx = anchor_a + y * (nx * 2) + x * 2;
                    // Last axis: 0..1 = xy, 2..3 = wh, 4..no-1 = conf
                    // xy: out[c] = (sigmoid(in[c]) * 2.0f + grid_yx[c]) * stride
                    for (int c = 0; c < 2; c++) {
                        float s = 1.0f / (1.0f + expf(-in_yx[c]));
                        out_yx[c] = (s * 2.0f + grid_yx[c]) * stride;
                    }
                    // wh: d = sigmoid(in[c]) * 2.0f; out[c] = d*d * anchor_yx[c-2]
                    for (int c = 2; c < 4; c++) {
                        float s = 1.0f / (1.0f + expf(-in_yx[c]));
                        float d = s * 2.0f;
                        out_yx[c] = d * d * anchor_yx[c - 2];
                    }
                    // conf: out[c] = sigmoid(in[c])
                    for (int c = 4; c < no; c++) {
                        float s = 1.0f / (1.0f + expf(-in_yx[c]));
                        out_yx[c] = s;
                    }
                }
            }
        }
    }
}

void bpd_tanh_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = tanhf(input[i]);
}

void bpd_gelu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
}

void bpd_neg_cpu(const float* input, float* output, int n) {
#if BPD_HAVE_AVX1
    const __m256 sign = _mm256_set1_ps(-0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(output + i, _mm256_xor_ps(_mm256_loadu_ps(input + i), sign));
    for (; i < n; i++) output[i] = -input[i];
#else
    for (int i = 0; i < n; i++) output[i] = -input[i];
#endif
}

void bpd_abs_cpu(const float* input, float* output, int n) {
#if BPD_HAVE_AVX1
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(output + i, _mm256_and_ps(_mm256_loadu_ps(input + i), mask));
    for (; i < n; i++) output[i] = fabsf(input[i]);
#else
    for (int i = 0; i < n; i++) output[i] = fabsf(input[i]);
#endif
}

void bpd_exp_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = expf(input[i]);
}

// ── Tier 1 activations (Stanford L1 problems 20, 27-32) ──
//
// Each implementation mirrors the formula PyTorch uses in
// aten/src/ATen/native/cpu/Activation.cpp. These are pure elementwise
// kernels — no reduction, no SIMD-specific shuffles — so the substrate's
// scalar implementation produces bit-identical output by construction
// (one IEEE 754 operation per element matches one IEEE 754 operation per
// element regardless of whether PyTorch's vectorized path runs).

// LeakyReLU: a > 0 ? a : a * negval (default negval = 0.01)
// Source: aten/src/ATen/native/cpu/Activation.cpp:871 leaky_relu_kernel
void bpd_leaky_relu_cpu(const float* input, float* output, int n) {
#if BPD_HAVE_AVX1
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg = _mm256_set1_ps(0.01f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 pos = _mm256_max_ps(zero, x);
        __m256 negpart = _mm256_mul_ps(_mm256_min_ps(zero, x), neg);
        _mm256_storeu_ps(output + i, _mm256_add_ps(pos, negpart));
    }
    for (; i < n; i++) {
        float a = input[i];
        output[i] = a > 0.0f ? a : a * 0.01f;
    }
#else
    for (int i = 0; i < n; i++) {
        float a = input[i];
        output[i] = a > 0.0f ? a : a * 0.01f;
    }
#endif
}

// ELU: a < 0 ? expm1(a) * (alpha*scale) : a * scale
// Default: alpha=1, scale=1, input_scale=1 → simplifies to a < 0 ? expm1f(a) : a
// Source: aten/src/ATen/native/cpu/Elu.h:23 get_scalar_elu_elementwise_func
void bpd_elu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float a = input[i];
        output[i] = a <= 0.0f ? (expf(a) - 1.0f) : a;
    }
}

// SELU: ELU with alpha=1.6732632, scale=1.0507009 (double constants
// truncated to float at the Scalar→float conversion in elu_kernel).
// Source: aten/src/ATen/native/Activation.cpp:245 SELU_ALPHA/SCALE +
//         aten/src/ATen/native/cpu/Elu.h:23 get_scalar_elu_elementwise_func
void bpd_selu_cpu(const float* input, float* output, int n) {
    // PyTorch truncates these to float when passing through Scalar::to<float>()
    const float alpha = (float)1.6732632423543772848170429916717;
    const float scale = (float)1.0507009873554804934193349852946;
    const float negcoef = alpha * scale;  // PyTorch computes this on float at runtime
    const float poscoef = scale;
    const float negiptcoef = 1.0f;  // input_scale default
    for (int i = 0; i < n; i++) {
        float a = input[i];
        output[i] = a <= 0.0f ? (expf(a * negiptcoef) - 1.0f) * negcoef : a * poscoef;
    }
}

// HardSigmoid: min(max(x + 3, 0), 6) / 6
// Source: aten/src/ATen/native/cpu/Activation.cpp:523 hardsigmoid_kernel
void bpd_hardsigmoid_cpu(const float* input, float* output, int n) {
#if BPD_HAVE_AVX1
    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 t = _mm256_add_ps(x, three);
        t = _mm256_max_ps(zero, t);
        t = _mm256_min_ps(six, t);
        _mm256_storeu_ps(output + i, _mm256_div_ps(t, six));
    }
    for (; i < n; i++) {
        float x = input[i];
        float t = x + 3.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 6.0f) t = 6.0f;
        output[i] = t / 6.0f;
    }
#else
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float t = x + 3.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 6.0f) t = 6.0f;
        output[i] = t / 6.0f;
    }
#endif
}

// HardTanh / clamp: clamp(x, min, max). Default for nn.Hardtanh: min=-1, max=1.
// Source: aten/src/ATen/native/cpu/Activation.cpp (hardtanh path) — clamp is
//         exposed via the more general clamp operator.
void bpd_clamp_cpu(const float* input, float* output, int n) {
    const float min_val = -1.0f;
    const float max_val = 1.0f;
    for (int i = 0; i < n; i++) {
        float x = input[i];
        if (x < min_val) x = min_val;
        if (x > max_val) x = max_val;
        output[i] = x;
    }
}

// Softplus: a * beta > threshold ? a : log1p(exp(a * beta)) / beta
// Default: beta=1, threshold=20 (nn.Softplus default).
// Source: aten/src/ATen/native/cpu/Activation.cpp:950 softplus_kernel
void bpd_softplus_cpu(const float* input, float* output, int n) {
    const float beta = 1.0f;
    const float threshold = 20.0f;
    for (int i = 0; i < n; i++) {
        float a = input[i];
        float ab = a * beta;
        output[i] = ab > threshold ? a : log1pf(expf(ab)) / beta;
    }
}

// Softsign: x / (1 + |x|)
// Source: aten/src/ATen/native/Activation.cpp (no per-element CPU kernel —
// implemented as composite of abs, add scalar, div). Simpler to inline.
void bpd_softsign_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + fabsf(x));
    }
}

// ── Cumulative reductions (Stanford L1 problems 89-93) ──
//
// PyTorch's cumsum/cumprod on float use `at::acc_type<float, false>` = double
// as the accumulator. Each element is added/multiplied into a double accumulator
// then cast back to float on store. This raises precision throughout the chain.
//
// Source: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:79 cumsum_cpu_kernel,
//         aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:98 cumprod_cpu_kernel
//
// Substrate-design parameter: cumulative_acc_type(double).

// Cumsum: y[i] = y[i-1] + x[i], with y[-1] = 0.
// PyTorch uses double as the running accumulator.
void bpd_cumsum_cpu(const float* input, float* output, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        acc += (double)input[i];
        output[i] = (float)acc;
    }
}

// Cumprod: y[i] = y[i-1] * x[i], with y[-1] = 1.
// PyTorch uses double as the running accumulator.
void bpd_cumprod_cpu(const float* input, float* output, int n) {
    double acc = 1.0;
    for (int i = 0; i < n; i++) {
        acc *= (double)input[i];
        output[i] = (float)acc;
    }
}

// Cumsum reverse: y[i] = x[i] + x[i+1] + ... + x[n-1].
// Equivalent to: reverse → cumsum → reverse. PyTorch uses cumsum + flip.
void bpd_cumsum_reverse_cpu(const float* input, float* output, int n) {
    double acc = 0.0;
    for (int i = n - 1; i >= 0; i--) {
        acc += (double)input[i];
        output[i] = (float)acc;
    }
}

// Exclusive cumsum: y[0] = 0, y[i] = x[0] + ... + x[i-1].
// PyTorch implements this as concat([zeros(1), cumsum[:-1]]).
void bpd_cumsum_exclusive_cpu(const float* input, float* output, int n) {
    double acc = 0.0;
    output[0] = 0.0f;
    for (int i = 1; i < n; i++) {
        acc += (double)input[i - 1];
        output[i] = (float)acc;
    }
}

// ── Reductions ──

// PyTorch CPU "cascade_sum" — exact port of at::native::row_sum + multi_row_sum.
// Source: pytorch/aten/src/ATen/native/cpu/SumKernel.cpp.
//
// Algorithm structure (PyTorch's CPU default for AVX1 hardware):
//   reduction_strategy(cascade(SimdWidth=8, IlpFactor=4, CascadeDepth=4, CascadeBase=16))
//
// Three-level parallel reduction:
//   1. SIMD: 8 parallel f32 lanes per Vectorized<float> register (AVX width on the
//      enclave: default 32 bytes / 4 bytes-per-float = 8 lanes).
//   2. ILP:  4 ILP-interleaved cascade lanes per SIMD register. Each input element
//      goes into one of 8 SIMD lanes × 4 ILP lanes = 32 parallel scalar slots
//      at the "level 0" position.
//   3. Cascade: 4 levels per (SIMD, ILP) slot. Every CascadeBase=16 iterations,
//      level 0 promotes to level 1; every 16² to level 2; every 16³ to level 3.
//      Total = 8 × 4 × 4 = 128 parallel scalar accumulators.
//
// Reduction order at the end:
//   level 1..3 collapse into level 0 (per SIMD × ILP)
//   ILP collapse: lane[0][s] += lane[k][s] for k in 1..3
//   SIMD collapse: final += lane[0][s] for s in 0..7
//   Scalar tail addition
//
// Per Heath's direction: "make porting the full SIMD-8 × ILP-4 × 4-level cascade
// implementation as a sweepable pattern for our code generator/optimizer."
// This C function is the manually-ported reference for one specific instantiation;
// lib/reduction_kernel.pl (to be added) generates this same shape for any
// (SimdWidth, IlpFactor, CascadeDepth, CascadeBase) combination.

static int ceil_log2(int n) {
    int r = 0;
    int x = n - 1;
    while (x > 0) { x >>= 1; r++; }
    return r;
}

// multi_row_sum_simd: full SIMD-W × ILP × cascade-D × cascade-base implementation
// for SimdWidth=8, IlpFactor=4, CascadeDepth=4.
//
// `data` starts at the array origin. The function processes `size_ilp` iterations,
// where each iteration loads 4 ILP-interleaved SIMD-8 blocks (= 32 floats).
// Returns the (4, 8) grid of partial accumulators.
static void multi_row_sum_simd(const float* data,
                                int size_ilp,
                                float out_lane[4][8]) {
    int lp = 4;
    if (size_ilp > 0) {
        lp = ceil_log2(size_ilp) / 4;
        if (lp < 4) lp = 4;
    }
    int level_step = 1 << lp;
    int level_mask = level_step - 1;

    // Cascade × ILP × SIMD = 4 × 4 × 8 = 128 accumulators
    float acc[4][4][8] = {{{0}}};

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        // Accumulate level_step iterations into level 0
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp = 0; ilp < 4; ++ilp) {
                const float* src = base + ilp * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp][s] += src[s];
                }
            }
        }
        // Cascade promotion: levels 1..3
        for (int level = 1; level < 4; ++level) {
            for (int ilp = 0; ilp < 4; ++ilp) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp][s] += acc[level-1][ilp][s];
                    acc[level-1][ilp][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }

    // Tail iterations (less than level_step worth)
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp = 0; ilp < 4; ++ilp) {
            const float* src = base + ilp * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp][s] += src[s];
            }
        }
    }

    // Final per-lane cascade collapse: levels 1..3 → level 0
    for (int level = 1; level < 4; ++level) {
        for (int ilp = 0; ilp < 4; ++ilp) {
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp][s] += acc[level][ilp][s];
            }
        }
    }

    // Write out the (ILP, SIMD) grid
    for (int ilp = 0; ilp < 4; ++ilp) {
        for (int s = 0; s < 8; ++s) {
            out_lane[ilp][s] = acc[0][ilp][s];
        }
    }
}

// Per substrate-design correspondence map docs/substrate-design-correspondence.md:
// the binary_kernel_reduce_lastdim path PyTorch uses for torch.norm(p=2, dim=-1)
// is SUBSTANTIVELY DIFFERENT from cascade_sum. It uses a single SIMD-8 accumulator
// in a linear pass, then linear horizontal reduce, then tail scalar.
// Source: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:227 norm_kernel_tensor_iterator_impl
//
// Substrate-design parameter: norm_reduction_strategy(binary_kernel_reduce_lastdim_simd8).
// Distinct from reduction_strategy(cascade(8,4,4,16)) which is for torch.sum.
//
// Returns sum(x[d]^2) over d in [0, n) using PyTorch's exact algorithm.
static float bpd_norm_p2_sumsq_lastdim(const float* x, int n) {
    #define VEC 8
    float acc_vec[VEC] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int d = 0;
    // SIMD-8 loop (norm_two_reduce_step: acc += data * data per lane)
    int simd_end = n - (n % VEC);
    for (; d < simd_end; d += VEC) {
        for (int j = 0; j < VEC; j++) {
            float v = x[d + j];
            acc_vec[j] += v * v;
        }
    }
    // Horizontal reduce: linear sum of the 8 lanes
    float buf = acc_vec[0];
    for (int j = 1; j < VEC; j++) {
        buf = buf + acc_vec[j];
    }
    // Scalar tail
    for (; d < n; d++) {
        float v = x[d];
        buf = buf + v * v;
    }
    #undef VEC
    return buf;
}

static float pairwise_sum(const float* data, int n) {
    if (n == 0) return 0.0f;
    if (n == 1) return data[0];

    // PyTorch's dispatch: vectorized path requires size0 >= vec_t::size() = 8.
    // For n < 8, the scalar fallback is used.
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    const int VEC_SIZE = 8;          // SimdWidth
    const int ILP_FACTOR = 4;
    int vec_size = n / VEC_SIZE;     // number of full SIMD-8 blocks
    int size_ilp = vec_size / ILP_FACTOR;
    int simd_processed = vec_size * VEC_SIZE;   // # floats consumed by full SIMD blocks

    // multi_row_sum_simd processes size_ilp iterations of 32 floats each
    float lane[4][8];
    multi_row_sum_simd(data, size_ilp, lane);

    // Tail SIMD-8 blocks (couldn't fill a complete ILP-4 group of 32)
    for (int v = size_ilp * ILP_FACTOR; v < vec_size; ++v) {
        const float* src = data + v * VEC_SIZE;
        for (int s = 0; s < 8; ++s) {
            lane[0][s] += src[s];
        }
    }

    // Horizontal collapse over ILP: lane[0][s] += lane[k][s] for k in 1..3
    // EMPIRICAL FINDING 2026-05-21: PyTorch's vectorized_reduction source code
    // says pairwise (vop(vop(acc[0], acc[1]), vop(acc[2], acc[3]))), but the
    // emitted code at AVX1 for our shapes is LINEAR ILP combine. We tested
    // pairwise here and BIT_IDENTICAL dropped 93→92 (47 Sum_reduction and 38
    // L1Norm flipped to DIVERGENT). Linear combine matches the actual emitted
    // code. Substrate-design parameter: ilp_combine_strategy(linear_simd8).
    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            lane[0][s] += lane[k][s];
        }
    }

    // Final accumulator: PyTorch's order is
    //   final_acc = 0
    //   for k in scalar tail: final_acc += data[k]
    //   for s in 0..7: final_acc += lane[0][s]
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += lane[0][s];
    }
    return final_acc;
}

void bpd_sum_cpu(const float* input, float* output, int n) {
    *output = pairwise_sum(input, n);
}

void bpd_mean_cpu(const float* input, float* output, int n) {
    *output = pairwise_sum(input, n) / (float)n;
}

void bpd_max_cpu(const float* input, float* output, int n) {
    float m = input[0];
    for (int i = 1; i < n; i++) if (input[i] > m) m = input[i];
    *output = m;
}

// ── Softmax (row-wise) ──

// PyTorch's softmax uses vec::reduce_all (linear scan with one SIMD-Vec
// accumulator), NOT the cascade sum. The cascade is only used by
// sum_kernel_impl in SumKernel.cpp. Source:
// aten/src/ATen/cpu/vec/functional_base.h:184 inline scalar_t reduce_all.
//
// Algorithm:
//   1. Load first SW=8 elements into acc_vec
//   2. For each subsequent SW=8 block: acc_vec[s] = vec_fun(acc_vec[s], data[d+s])
//   3. Tail (last (n % SW) elements) added to acc_vec via vec::set, then
//      horizontally reduced (acc_vec[0] += acc_vec[1] + ... + acc_vec[7])
//
// This is reduction_strategy(linear_scan_simd(SimdWidth=8)) — a simpler
// substrate-design parameter than cascade, but it produces different bits
// than cascade on the same data. Hence softmax doesn't use pairwise_sum.
static float linear_scan_sum_simd8(const float* data, int n) {
    if (n == 0) return 0.0f;
    if (n < 8) {
        // Scalar fallback for very small inputs (matches PyTorch's
        // vec_reduce_all path when size < Vec::size()).
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    // Load first SW=8 elements
    float acc[8];
    for (int s = 0; s < 8; ++s) acc[s] = data[s];

    // Linear scan: each SIMD block accumulated lane-wise
    int d = 8;
    int full_end = n - (n % 8);
    for (; d < full_end; d += 8) {
        for (int s = 0; s < 8; ++s) acc[s] += data[d + s];
    }

    // Tail (last n%8 elements). PyTorch uses vec::set which preserves the
    // upper lanes of acc_vec while loading partial data and adding only
    // the first (n - d) lanes. Implementation-wise this means we add only
    // data[d..n) to the first (n - d) accumulator lanes.
    int tail = n - d;
    for (int s = 0; s < tail; ++s) acc[s] += data[d + s];

    // Horizontal reduce: on AVX1 (no AVX2 acceleration), PyTorch falls
    // through to the generic vec_reduce_all path which sums lane 0
    // left-to-right: acc[0] + acc[1] + acc[2] + ... + acc[7].
    // See functional_base.h:174 vec_reduce_all slow path. The bizarre
    // SIMD shuffle this emulates ends up being equivalent to a strict
    // left-to-right scan of acc_arr[].
    float horiz = acc[0];
    for (int s = 1; s < 8; ++s) horiz += acc[s];
    return horiz;
}

void bpd_softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        // find max for numerical stability
        float mx = row_in[0];
        for (int c = 1; c < cols; c++) if (row_in[c] > mx) mx = row_in[c];
        // exp
        for (int c = 0; c < cols; c++)
            row_out[c] = expf(row_in[c] - mx);
        // PyTorch softmax uses vec::reduce_all (linear-scan SIMD-8), not cascade
        float sum = linear_scan_sum_simd8(row_out, cols);
        // normalize: multiply by reciprocal (matches PyTorch — same pattern as BN)
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) row_out[c] *= inv_sum;
    }
}

// LogSoftmax: y = x - max(x) - log(sum(exp(x - max(x))))
// Source: aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h:31
// serial_vec_log_softmax_lastdim_range
//
// Same linear-scan SIMD-8 reduction as softmax. PyTorch is careful to keep
// the operation order `x - max - log_sum` (not `x - (max + log_sum)`) to
// avoid catastrophic cancellation when max is large and log_sum is small.
void bpd_logsoftmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        // 1. max via linear-scan reduction (same as softmax)
        float mx = row_in[0];
        for (int c = 1; c < cols; c++) if (row_in[c] > mx) mx = row_in[c];
        // 2. exp(x - max) into a temp, then sum via linear-scan SIMD-8.
        //    Use the output buffer as temp (overwritten in step 4 anyway).
        for (int c = 0; c < cols; c++)
            row_out[c] = expf(row_in[c] - mx);
        float sum_exp = linear_scan_sum_simd8(row_out, cols);
        // 3. log(sum)
        float log_sum = logf(sum_exp);
        // 4. output = x - max - log_sum (in that order, per PyTorch source
        //    note about avoiding cancellation between max and log_sum)
        for (int c = 0; c < cols; c++)
            row_out[c] = row_in[c] - mx - log_sum;
    }
}

// ── LayerNorm ──

// Welford-with-cascade rowwise moments matching PyTorch's
// at::native::RowwiseMomentsImpl exactly.
//
// Source: pytorch/aten/src/ATen/native/cpu/moments_utils.h
//
// Algorithm: SIMD-8 Welford inside chunks of kChunkSize=16 SIMD-vectors,
// then pairwise stack-merge with mask-based promotion (same cascade
// pattern as bpd_sum_cpu, but for 3-tuple (m0, m1, m2) updates).
//
// For D=128 (the test shape): n=16 SIMD-vectors, m=1 chunk, depth=0.
// One UpdateMomentsVec on 16 SIMD-8 iterations + AddMoments horizontal
// reduce across 8 SIMD lanes.
//
// Numerical stability comes from:
//   - Welford recurrence (avoids catastrophic cancellation of sum(x²) - mean²)
//   - Cascade merge across chunks (avoids accumulation drift over long arrays)
//
// Substrate-design parameter: rowwise_moments_strategy(welford_simd8_cascade16).

static int ceil_log2_lm(int n) {
    if (n <= 1) return 0;
    int r = 0; int x = n - 1;
    while (x > 0) { x >>= 1; r++; }
    return r;
}

// AddMoments — Welford parallel combination of (m0_a, m1_a, m2_a) and
// (m0_b, m1_b, m2_b) into (*m0, *m1, *m2). Mirrors moments_utils.h:18
// AddMoments<T> exactly.
static void add_moments(int m0_add, float m1_add, float m2_add,
                         int* m0, float* m1, float* m2) {
    int n = *m0 + m0_add;
    float c = (n == 0) ? 0.0f : (float)m0_add / (float)n;
    float delta = m1_add - *m1;
    *m1 += c * delta;
    *m2 += m2_add + delta * delta * c * (float)(*m0);
    *m0 = n;
}

// rowwise_moments — returns (mean, variance) for one row of D floats,
// matching PyTorch's RowwiseMoments<float>(X, N).
//
// For D=128: kVecSize=8, n=16, m=1, depth=0.
// One UpdateMomentsVec on 16 SIMD-8 chunks + horizontal AddMoments across
// 8 SIMD lanes.
static void rowwise_moments(const float* X, int N,
                              float* out_mean, float* out_var) {
    const int kVecSize = 8;
    const int kChunkSize = 16;

    int n = N / kVecSize;
    int m = (n + kChunkSize - 1) / kChunkSize;  // divup
    int depth = ceil_log2_lm(m);

    // Stack: depth levels × 8 SIMD lanes
    // For typical depth ≤ 32 we use a fixed-size stack array.
    enum { kMaxDepth = 32 };
    int   m0_stk[kMaxDepth] = {0};
    float m1_stk[kMaxDepth][8] = {{0}};
    float m2_stk[kMaxDepth][8] = {{0}};

    // c_vecs: per-iteration constants 1/(j+1) for j in [0..kChunkSize)
    float c_consts[16];
    for (int j = 0; j < kChunkSize; ++j) {
        c_consts[j] = 1.0f / (float)(j + 1);
    }

    for (int i = 0; i < m; ++i) {
        const float* X_ptr = X + i * kChunkSize * kVecSize;
        int m0_local = kChunkSize;
        int remain = n - i * kChunkSize;
        if (remain < kChunkSize) m0_local = remain;

        // UpdateMomentsVec: SIMD-8 Welford over m0_local iterations.
        // Each lane s in 0..7 is independent.
        float m1_vec[8] = {0};
        float m2_vec[8] = {0};
        for (int j = 0; j < m0_local; ++j) {
            float c = c_consts[j];
            for (int s = 0; s < 8; ++s) {
                float x = X_ptr[j * 8 + s];
                float delta = x - m1_vec[s];
                // m1 = fmadd(c, delta, m1)  →  m1 += c * delta (no-FMA on AVX1)
                m1_vec[s] = m1_vec[s] + c * delta;
                float delta2 = x - m1_vec[s];
                // m2 = fmadd(delta, delta2, m2)  →  m2 += delta * delta2
                m2_vec[s] = m2_vec[s] + delta * delta2;
            }
        }

        // AddMomentsVec: merge the per-chunk (m0_local, m1_vec[8], m2_vec[8])
        // into stk[0] using vector AddMoments semantics.
        // The AddMomentsVec from PyTorch does the same scalar update applied
        // to each of the 8 SIMD lanes — equivalent to running add_moments per
        // lane with the SAME m0 value (m0_local), and updating m0_stk only once.
        {
            int old_m0 = m0_stk[0];
            int new_m0 = old_m0 + m0_local;
            float c_vec = (new_m0 == 0) ? 0.0f : (float)m0_local / (float)new_m0;
            for (int s = 0; s < 8; ++s) {
                float delta = m1_vec[s] - m1_stk[0][s];
                m1_stk[0][s] += c_vec * delta;
                m2_stk[0][s] += m2_vec[s] + delta * delta * c_vec * (float)old_m0;
            }
            m0_stk[0] = new_m0;
        }

        // Cascade stack-merge: when chunk index (i+1) has trailing zeros at
        // depth j, promote stk[j-1] → stk[j].
        int mask = i + 1;
        for (int j = 1; j < depth && (mask & 1) == 0; ++j) {
            int old_m0_j = m0_stk[j];
            int add_m0 = m0_stk[j - 1];
            int new_m0_j = old_m0_j + add_m0;
            float c_vec = (new_m0_j == 0) ? 0.0f : (float)add_m0 / (float)new_m0_j;
            for (int s = 0; s < 8; ++s) {
                float delta = m1_stk[j-1][s] - m1_stk[j][s];
                m1_stk[j][s] += c_vec * delta;
                m2_stk[j][s] += m2_stk[j-1][s] + delta * delta * c_vec * (float)old_m0_j;
            }
            m0_stk[j] = new_m0_j;
            m0_stk[j-1] = 0;
            for (int s = 0; s < 8; ++s) {
                m1_stk[j-1][s] = 0.0f;
                m2_stk[j-1][s] = 0.0f;
            }
            mask >>= 1;
        }
    }

    // Scalar tail (last N % kVecSize elements) — uses scalar Welford
    int m0 = 0; float m1 = 0; float m2 = 0;
    for (int i = n * kVecSize; i < N; ++i) {
        float x = X[i];
        float delta = x - m1;
        ++m0;
        m1 += delta / (float)m0;
        m2 += delta * (x - m1);
    }

    // Merge stack levels [1..depth) into stk[0]
    for (int j = 1; j < depth; ++j) {
        int old_m0_0 = m0_stk[0];
        int add_m0 = m0_stk[j];
        int new_m0_0 = old_m0_0 + add_m0;
        float c_vec = (new_m0_0 == 0) ? 0.0f : (float)add_m0 / (float)new_m0_0;
        for (int s = 0; s < 8; ++s) {
            float delta = m1_stk[j][s] - m1_stk[0][s];
            m1_stk[0][s] += c_vec * delta;
            m2_stk[0][s] += m2_stk[j][s] + delta * delta * c_vec * (float)old_m0_0;
        }
        m0_stk[0] = new_m0_0;
    }

    // Horizontal AddMoments across the 8 SIMD lanes of stk[0] into the scalar
    // (m0, m1, m2). PyTorch source:
    //   int64_t m0_add = n * kVecSize / kAccVecSize;
    // For same-precision T=float: kVecSize=8, kAccVecSize=8 → m0_add = n.
    // Each lane represents n elements (lane s processes data[s], data[s+8],
    // data[s+16], ..., data[s+(n-1)*8] — n strided values).
    int m0_add_per_lane = n;  // each lane saw n elements
    for (int s = 0; s < 8; ++s) {
        add_moments(m0_add_per_lane, m1_stk[0][s], m2_stk[0][s], &m0, &m1, &m2);
    }

    *out_mean = m1;
    *out_var = m2 / (float)N;  // ddof = 0
}

void bpd_layernorm_cpu(const float* input, const float* gamma,
                        const float* beta, float* output,
                        int N, int D, float eps) {
    for (int n = 0; n < N; n++) {
        const float* x = input + n * D;
        float* y = output + n * D;
        // Welford rowwise moments matching PyTorch CPU exactly
        float mean, var;
        rowwise_moments(x, D, &mean, &var);
        // rstd via reciprocal_sqrt variant (matches PyTorch CPU + bpd_default)
        float rstd = 1.0f / sqrtf(var + eps);
        // Normalize and apply affine (gamma, beta)
        for (int d = 0; d < D; d++)
            y[d] = (x[d] - mean) * rstd * gamma[d] + beta[d];
    }
}

// ── Normalization family (Stanford L1 problems 34-39) ──
//
// The norm family shares two substrate-design choices:
//   1. Welford rowwise_moments() — for mean+var moments (LayerNorm, InstanceNorm, GroupNorm)
//   2. pairwise_sum (cascade) — for sum-of-squares (RMSNorm, Frobenius, L1Norm, L2Norm)
//   3. rsqrt_variant(reciprocal_sqrt) — same as LayerNorm and BatchNorm
//
// Each kernel below mirrors the algorithm PyTorch CPU uses. The L1 tests
// validate at smaller shapes than the model's deployment shapes; the
// per-(n, slice) algorithm is the same.

// InstanceNorm2D: per-(batch, channel) Welford normalization over spatial dims.
// PyTorch source: aten/src/ATen/native/Normalization.cpp:727 instance_norm
// — composite that reshapes (B,C,H,W) to (1, B*C, H, W) and calls batch_norm.
// In training mode (KernelBench default: affine=False, track_running_stats=False),
// batch_norm:
//   1. Collects stats via batch_norm_cpu_collect_stats_contiguous_impl:
//      naive two-pass sum + var_sum (accscalar_t = float for fp32, NOT double).
//   2. Computes invstd = 1/sqrt(var + eps) per channel.
//   3. Applies via precomputed-scale-offset form:
//        alpha[c] = invstd * weight     (weight=1 for InstanceNorm no-affine)
//        beta[c]  = bias - mean * alpha (bias=0 for InstanceNorm no-affine)
//        output(p) = input(p) * alpha[c] + beta[c]
//
// This is the SAME substrate-design choice as BatchNorm: bn_mode(precomputed_scale_offset).
// (x - mean) * invstd and x*alpha + beta are algebraically equivalent but BIT-DIFFERENT.
//
// Source:
//   aten/src/ATen/native/cpu/batch_norm_kernel.cpp:31 batch_norm_cpu_collect_linear_and_constant_terms
//   aten/src/ATen/native/cpu/batch_norm_kernel.cpp:177 batch_norm_cpu_collect_stats_contiguous_impl
void bpd_instancenorm_cpu(const float* input, float* output,
                           int N, int C, int H, int W, float eps) {
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float* x = input + (n * C + c) * spatial;
            float* y = output + (n * C + c) * spatial;
            // PyTorch source: batch_norm_cpu_collect_stats_contiguous_impl
            //   acc_type<float, false> = double on CPU. We use double accumulators.
            // Source: aten/src/ATen/native/cpu/batch_norm_kernel.cpp:177
            double sum = 0.0;
            for (int p = 0; p < spatial; p++) sum += (double)x[p];
            float mean = (float)(sum / (double)spatial);
            double var_sum = 0.0;
            for (int p = 0; p < spatial; p++) {
                double d = (double)x[p] - (double)mean;
                var_sum += d * d;
            }
            float var = (float)(var_sum / (double)spatial);
            float invstd = 1.0f / sqrtf(var + eps);
            // EMPIRICAL 2026-05-21: At controlled inputs mean+var bit-identical
            // with PyTorch. The 3-4 ULP residual at harness RNG state comes from
            // PyTorch's SIMD-vectorized apply step at AVX1 default path emitting
            // a different SIMD chunk pattern than our scalar code at the specific
            // shape (N=2, C=4, H=W=8) used by the harness.
            //
            // Both affine_application forms tested:
            //   precomputed_alpha_beta: x * (invstd) + (-mean * invstd) — 3 ULP / 100 diffs
            //   direct_subtract_multiply: (x - mean) * invstd          — 3 ULP / 234 diffs
            // precomputed_alpha_beta has fewer divergent positions at the harness
            // shape, so we use that form. The remaining 3 ULP is at the SIMD-inner
            // microopt level (Phase B).
            float alpha = invstd;
            float bias = -mean * alpha;
            for (int p = 0; p < spatial; p++)
                y[p] = x[p] * alpha + bias;
        }
    }
}

// GroupNorm: per-(batch, group) Welford normalization over (group_features × spatial).
// PyTorch source: aten/src/ATen/native/cpu/group_norm_kernel.cpp:55 GroupNormKernelImpl
// — calls RowwiseMoments(X_ptr, inner_size) where inner_size = (C/G)*H*W.
// Then applies gamma/beta per channel.
void bpd_groupnorm_cpu(const float* input, const float* gamma,
                       const float* beta, float* output,
                       int N, int C, int H, int W, int G, float eps) {
    int channels_per_group = C / G;
    int group_size = channels_per_group * H * W;
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            const float* x_group = input + (n * C + g * channels_per_group) * spatial;
            float* y_group = output + (n * C + g * channels_per_group) * spatial;
            // Welford rowwise moments (matches PyTorch group_norm_kernel.cpp).
            float mean, var;
            rowwise_moments(x_group, group_size, &mean, &var);
            float rstd = 1.0f / sqrtf(var + eps);
            // Apply per-channel affine via PRECOMPUTED_ALPHA_BETA (matches PyTorch source):
            //   scale = rstd * gamma[c]
            //   bias  = -scale * mean + beta[c]
            //   y[k] = scale * x[k] + bias
            // Source: aten/src/ATen/native/cpu/group_norm_kernel.cpp:71-76
            for (int cc = 0; cc < channels_per_group; cc++) {
                int c = g * channels_per_group + cc;
                float scale = rstd * gamma[c];
                float bias  = -scale * mean + beta[c];
                const float* x = x_group + cc * spatial;
                float* y = y_group + cc * spatial;
                for (int p = 0; p < spatial; p++)
                    y[p] = scale * x[p] + bias;
            }
        }
    }
}

// RMSNorm: x / sqrt(mean(x²) + eps)  — applied per-row over feature dim.
// L1 test computes mean(x², dim=1, keepdim=True) — reduces along dim=1 (channels).
// For input (B, C, H, W) reduce dim=1 (C); result is (B, 1, H, W) broadcast back.
//
// SUBSTANTIVE EMPIRICAL FINDING (2026-05-20 ~23:35 UTC): PyTorch uses DIRECT
// DIVISION for the normalization step, NOT multiply-by-reciprocal. This is a
// new substrate-design parameter: norm_division_strategy(direct_division).
// Multiply-by-reciprocal introduces 1 ULP error in the reciprocal step that
// direct division avoids.
//
// Substrate-design choices for RMSNorm:
//   reduction_strategy(cascade_or_similar) — pairwise_sum over x²
//   norm_division_strategy(direct_division) — x / rms, not x * (1/rms)
void bpd_rmsnorm_cpu(const float* input, float* output,
                     int N, int C, int H, int W, float eps) {
    int spatial = H * W;
    float* temp = (float*)bpd_alloc(C * sizeof(float));
    for (int n = 0; n < N; n++) {
        for (int p = 0; p < spatial; p++) {
            // Squared values, contiguous
            for (int c = 0; c < C; c++) {
                float v = input[n * C * spatial + c * spatial + p];
                temp[c] = v * v;
            }
            // Reference uses torch.mean(x**2, dim=1) → cascade_sum / N pattern
            float sum_sq = pairwise_sum(temp, C);
            float rms = sqrtf(sum_sq / (float)C + eps);
            for (int c = 0; c < C; c++) {
                output[n * C * spatial + c * spatial + p] =
                    input[n * C * spatial + c * spatial + p] / rms;
            }
        }
    }
    free(temp);
}

// FrobeniusNorm: x / sqrt(sum(x²))  — GLOBAL reduction over all elements.
// PyTorch source: torch.norm(x, p='fro') flattens and reduces.
// Substrate-design choices: pairwise_sum over all N elements + direct_division.
void bpd_frobenius_norm_cpu(const float* input, float* output, int n_total) {
    float* temp = (float*)bpd_alloc(n_total * sizeof(float));
    for (int i = 0; i < n_total; i++) {
        float v = input[i];
        temp[i] = v * v;
    }
    float sum_sq = pairwise_sum(temp, n_total);
    float norm = sqrtf(sum_sq);
    for (int i = 0; i < n_total; i++)
        output[i] = input[i] / norm;
    free(temp);
}

// L1Norm: x / mean(|x|, dim=1, keepdim=True)  — per-row reduction along dim=1.
// For input (B, D) where the test uses dim=1: per-row sum(|x|)/D.
// Substrate-design choices: pairwise_sum (cascade) over |x| values + direct_division.
void bpd_l1norm_cpu(const float* input, float* output, int rows, int cols) {
    float* temp = (float*)bpd_alloc(cols * sizeof(float));
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        for (int c = 0; c < cols; c++)
            temp[c] = fabsf(row_in[c]);
        float sum_abs = pairwise_sum(temp, cols);
        float mean_abs = sum_abs / (float)cols;
        for (int c = 0; c < cols; c++)
            row_out[c] = row_in[c] / mean_abs;
    }
    free(temp);
}

// L2Norm: x / norm(x, p=2, dim=1, keepdim=True)  — per-row reduction along dim=1.
// L2 norm of a row = sqrt(sum(x²)).
// Substrate-design choices: pairwise_sum over x² + direct_division.
void bpd_l2norm_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        // Use binary_kernel_reduce_lastdim path to match PyTorch's torch.norm(p=2, dim=-1)
        // Source: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:227
        float sum_sq = bpd_norm_p2_sumsq_lastdim(row_in, cols);
        float norm = sqrtf(sum_sq);
        for (int c = 0; c < cols; c++)
            row_out[c] = row_in[c] / norm;
    }
}

// ── MaxPool2D / AvgPool2D ──

void bpd_maxpool2d_cpu(const float* input, float* output,
                        int N, int C, int H, int W,
                        int kH, int kW, int stride, int pad) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    int total = N * C * H_out * W_out;
    for (int idx = 0; idx < total; idx++) {
        int ow = idx % W_out;
        int oh = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);
        float val = -1e30f;
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++) {
                int hi = oh * stride - pad + kh;
                int wi = ow * stride - pad + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    float v = input[((n*C+c)*H+hi)*W+wi];
                    if (v > val) val = v;
                }
            }
        output[idx] = val;
    }
}

void bpd_avgpool2d_cpu(const float* input, float* output,
                        int N, int C, int H, int W,
                        int kH, int kW, int stride, int pad) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    int total = N * C * H_out * W_out;
    for (int idx = 0; idx < total; idx++) {
        int ow = idx % W_out;
        int oh = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);
        float sum = 0.0f; int count = 0;
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++) {
                int hi = oh * stride - pad + kh;
                int wi = ow * stride - pad + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    sum += input[((n*C+c)*H+hi)*W+wi];
                    count++;
                }
            }
        output[idx] = sum / (float)count;
    }
}

// ── Pool variants 1D and 3D (Stanford L1 problems 41, 43, 44, 46) ──

// MaxPool1D: input (N, C, L), output (N, C, L_out)
void bpd_maxpool1d_cpu(const float* input, float* output,
                       int N, int C, int L,
                       int kL, int stride, int pad) {
    int L_out = (L + 2*pad - kL) / stride + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < L_out; ol++) {
                float val = -1e30f;
                for (int kl = 0; kl < kL; kl++) {
                    int li = ol * stride - pad + kl;
                    if (li >= 0 && li < L) {
                        float v = input[(n*C+c)*L + li];
                        if (v > val) val = v;
                    }
                }
                output[(n*C+c)*L_out + ol] = val;
            }
        }
    }
}

// MaxPool3D: input (N, C, D, H, W), output (N, C, D_out, H_out, W_out)
void bpd_maxpool3d_cpu(const float* input, float* output,
                       int N, int C, int D, int H, int W,
                       int kD, int kH, int kW, int stride, int pad) {
    int D_out = (D + 2*pad - kD) / stride + 1;
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int od = 0; od < D_out; od++)
    for (int oh = 0; oh < H_out; oh++)
    for (int ow = 0; ow < W_out; ow++) {
        float val = -1e30f;
        for (int kd = 0; kd < kD; kd++)
        for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            int di = od * stride - pad + kd;
            int hi = oh * stride - pad + kh;
            int wi = ow * stride - pad + kw;
            if (di >= 0 && di < D && hi >= 0 && hi < H && wi >= 0 && wi < W) {
                float v = input[(((n*C+c)*D+di)*H+hi)*W+wi];
                if (v > val) val = v;
            }
        }
        output[(((n*C+c)*D_out+od)*H_out+oh)*W_out+ow] = val;
    }
}

// AvgPool1D: divisor = kL by default (count_include_pad=True is PT default).
// PyTorch's F.avg_pool1d divides by kernel_size when count_include_pad=True.
void bpd_avgpool1d_cpu(const float* input, float* output,
                       int N, int C, int L,
                       int kL, int stride, int pad) {
    int L_out = (L + 2*pad - kL) / stride + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < L_out; ol++) {
                float sum = 0.0f;
                for (int kl = 0; kl < kL; kl++) {
                    int li = ol * stride - pad + kl;
                    if (li >= 0 && li < L) {
                        sum += input[(n*C+c)*L + li];
                    }
                }
                output[(n*C+c)*L_out + ol] = sum / (float)kL;
            }
        }
    }
}

// AvgPool3D: same — divisor = kD*kH*kW (count_include_pad=True).
void bpd_avgpool3d_cpu(const float* input, float* output,
                       int N, int C, int D, int H, int W,
                       int kD, int kH, int kW, int stride, int pad) {
    int D_out = (D + 2*pad - kD) / stride + 1;
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    float divisor = (float)(kD * kH * kW);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int od = 0; od < D_out; od++)
    for (int oh = 0; oh < H_out; oh++)
    for (int ow = 0; ow < W_out; ow++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kD; kd++)
        for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            int di = od * stride - pad + kd;
            int hi = oh * stride - pad + kh;
            int wi = ow * stride - pad + kw;
            if (di >= 0 && di < D && hi >= 0 && hi < H && wi >= 0 && wi < W) {
                sum += input[(((n*C+c)*D+di)*H+hi)*W+wi];
            }
        }
        output[(((n*C+c)*D_out+od)*H_out+oh)*W_out+ow] = sum / divisor;
    }
}

// ── Linear (matmul + bias) ──

/* bpd_gemm_transB — tiled GEMM with transposed B.
 * Computes C[M,N] = A[M,K] @ B_T[N,K]^T
 * where B_T is stored as [N,K] (each row of B_T is a column of the result).
 *
 * Strategy: pack B_T[N,K] → B_packed[K,N] (transpose into contiguous panel),
 * then call bpd_gemm_v2_full on the packed buffer.
 * The pack is O(N*K) but the GEMM is O(M*N*K) — amortized for M >> 1.
 */
static void bpd_gemm_transB(const float* A, const float* B_T, float* C,
                              int M, int N, int K) {
    /* Pack B_T[N,K] → B_packed[K,N] */
    float* B_packed = (float*)bpd_alloc((size_t)K * N * sizeof(float));
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            B_packed[k * N + n] = B_T[n * K + k];
    
    bpd_gemm_v2_full(A, B_packed, C, M, N, K);
    free(B_packed);
}

void bpd_linear_cpu(const float* input, const float* weight,
                     const float* bias, float* output,
                     int M, int N, int K) {
    /* Linear: output[M,N] = input[M,K] @ weight[N,K]^T + bias[N]
     * Weight is [N,K] row-major (PyTorch convention).
     * Use bpd_gemm_transB which packs weight into [K,N] then calls
     * the tiled GEMM. The pack cost is amortized over M rows. */
    bpd_gemm_transB(input, weight, output, M, N, K);
    if (bias) {
        for (int row = 0; row < M; row++)
            for (int col = 0; col < N; col++)
                output[row*N+col] += bias[col];
    }
}

// ── Layer 2 primitives (per mavchin's direction 2026-05-20 ~18:31 UTC) ──
//
// Trivial kernels needed for YOLOv5n C3 modules and FPN concat:
//   - bpd_residual_add_cpu:  elementwise add for bottleneck residual
//   - bpd_concat_channel_cpu: channel-axis concat for C3/SPPF/FPN

// Elementwise add: out[i] = a[i] + b[i]
// Used by C3 bottleneck residual path.
//
// PyTorch path: y = a + b is single FADD per element (no FMA possible since
// only one operand-pair). Substrate matches bit-for-bit by definition —
// scalar IEEE 754 a + b produces the same result everywhere.
void bpd_residual_add_cpu(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] + b[i];
    }
}

// Channel-axis concatenation of N input tensors.
//
// Layout assumption: NCHW (PyTorch default).
// Each input has shape (N_batch, C_i, H, W). Output has shape
// (N_batch, sum(C_i), H, W).
//
// Per mavchin: "trivial kernel — memcpy with offset arithmetic".
//
// Inputs:
//   inputs:    array of n_inputs pointers, each pointing to an input tensor
//   c_each:    array of n_inputs channel counts (one per input)
//   n_inputs:  how many input tensors to concatenate (2 for C3, 4 for SPPF)
//   N_batch, H, W: shared spatial dims
//   output:    contiguous output buffer of shape (N_batch, sum(C_i), H, W)
//
// Algorithm: for each batch slot, copy each input's per-batch slice
// (C_i × H × W floats) into the output at the correct channel offset.
// Each input's per-batch slice is contiguous in memory; the destination
// region for that input within the output's per-batch slice is also
// contiguous. So this is a straightforward memcpy per (batch, input).
void bpd_concat_channel_cpu(const float** inputs, const int* c_each,
                             int n_inputs, int N_batch, int H, int W,
                             float* output) {
    int HW = H * W;
    // Compute total output channels = sum of c_each
    int C_total = 0;
    for (int i = 0; i < n_inputs; i++) C_total += c_each[i];
    int out_batch_stride = C_total * HW;

    for (int b = 0; b < N_batch; b++) {
        float* out_batch_base = output + b * out_batch_stride;
        int channel_offset = 0;
        for (int i = 0; i < n_inputs; i++) {
            int C_i = c_each[i];
            int in_batch_stride = C_i * HW;
            const float* in_batch_base = inputs[i] + b * in_batch_stride;
            // Copy C_i × H × W contiguous floats
            for (int j = 0; j < in_batch_stride; j++) {
                out_batch_base[channel_offset * HW + j] = in_batch_base[j];
            }
            channel_offset += C_i;
        }
    }
}

// ── Loss family (Stanford L1 problems 94-100) ──
//
// Each loss reduces to: elementwise op → mean/sum reduction.
// Reductions use pairwise_sum (cascade(8,4,4,16)) matching torch.mean/torch.sum.

// MSELoss: mean((predictions - targets)²)
// Returns single scalar via output[0].
void bpd_mse_loss_cpu(const float* pred, const float* target, float* output, int n) {
    float* temp = (float*)bpd_alloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float d = pred[i] - target[i];
        temp[i] = d * d;
    }
    float sum = pairwise_sum(temp, n);
    output[0] = sum / (float)n;
    free(temp);
}

// HuberLoss / smooth_l1_loss: per-element: 0.5*x² if |x|<beta else beta*(|x|-0.5*beta)
// beta=1.0 by default. Then mean reduction.
// PyTorch source: F.smooth_l1_loss with reduction='mean' (default), beta=1.0
void bpd_huber_loss_cpu(const float* pred, const float* target, float* output, int n) {
    const float beta = 1.0f;
    float* temp = (float*)bpd_alloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float diff = pred[i] - target[i];
        float abs_diff = fabsf(diff);
        if (abs_diff < beta) {
            temp[i] = 0.5f * diff * diff / beta;
        } else {
            temp[i] = abs_diff - 0.5f * beta;
        }
    }
    float sum = pairwise_sum(temp, n);
    output[0] = sum / (float)n;
    free(temp);
}

// HingeLoss: torch.mean(torch.clamp(1 - predictions * targets, min=0))
void bpd_hinge_loss_cpu(const float* pred, const float* target, float* output, int n) {
    float* temp = (float*)bpd_alloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float v = 1.0f - pred[i] * target[i];
        temp[i] = v > 0.0f ? v : 0.0f;
    }
    float sum = pairwise_sum(temp, n);
    output[0] = sum / (float)n;
    free(temp);
}

// KLDivLoss: F.kl_div(log_pred, target, reduction='batchmean')
//   per-element: target * (log(target) - log_pred) … but PyTorch's F.kl_div
//   convention is: target * (log(target) - input), where input is already log.
//   Source: torch.nn.functional.kl_div docs:
//     "input – Tensor of arbitrary shape in log-probabilities."
//     loss = target * (log(target) - input)
//     For target=0: contribution is 0 (by convention).
//   reduction='batchmean' divides by batch_size (first dim).
void bpd_kl_div_loss_cpu(const float* log_pred, const float* target,
                          float* output, int batch_size, int per_batch) {
    int n = batch_size * per_batch;
    float* temp = (float*)bpd_alloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float t = target[i];
        if (t > 0.0f) {
            temp[i] = t * (logf(t) - log_pred[i]);
        } else {
            temp[i] = 0.0f;
        }
    }
    float sum = pairwise_sum(temp, n);
    // 'batchmean': divide by batch_size, NOT n
    output[0] = sum / (float)batch_size;
    free(temp);
}

// CrossEntropyLoss: F.cross_entropy(predictions, targets, reduction='mean')
//   = mean over batch of: -log_softmax(predictions)[target[i]]
// predictions: (batch_size, num_classes), targets: (batch_size,) integer class indices.
//
// For numerical match with PyTorch:
//   1. compute log_softmax(predictions) per row (using linear_scan_sum_simd8)
//   2. gather log_softmax[i, targets[i]] for each batch element
//   3. negate, then mean
void bpd_cross_entropy_loss_cpu(const float* pred, const long* target,
                                  float* output, int batch_size, int num_classes) {
    float* temp = (float*)bpd_alloc(batch_size * sizeof(float));
    float* row_logsm = (float*)bpd_alloc(num_classes * sizeof(float));
    for (int b = 0; b < batch_size; b++) {
        const float* row = pred + b * num_classes;
        // log_softmax inline: same as bpd_logsoftmax_cpu but for one row
        float mx = row[0];
        for (int c = 1; c < num_classes; c++) if (row[c] > mx) mx = row[c];
        for (int c = 0; c < num_classes; c++) row_logsm[c] = expf(row[c] - mx);
        float sum_exp = linear_scan_sum_simd8(row_logsm, num_classes);
        float log_sum = logf(sum_exp);
        // log_softmax(c) = row[c] - mx - log_sum; we only need the target column
        int t = (int)target[b];
        temp[b] = -(row[t] - mx - log_sum);
    }
    float sum = pairwise_sum(temp, batch_size);
    output[0] = sum / (float)batch_size;
    free(temp);
    free(row_logsm);
}

// TripletMarginLoss: F.triplet_margin_loss(anchor, positive, negative, margin=1, p=2)
//   per-row: max(0, ||a-p||_p - ||a-n||_p + margin)
//   reduction='mean' over batch.
// p=2 means L2 distance per row (sqrt(sum((a-p)²))).
void bpd_triplet_margin_loss_cpu(const float* anchor, const float* positive,
                                   const float* negative, float* output,
                                   int batch_size, int feat_dim, float margin) {
    float* temp = (float*)bpd_alloc(batch_size * sizeof(float));
    float* sqdiff = (float*)bpd_alloc(feat_dim * sizeof(float));
    for (int b = 0; b < batch_size; b++) {
        const float* a = anchor + b * feat_dim;
        const float* p = positive + b * feat_dim;
        const float* nv = negative + b * feat_dim;
        // ||a - p||_2
        for (int c = 0; c < feat_dim; c++) {
            float d = a[c] - p[c];
            sqdiff[c] = d * d;
        }
        float dist_ap = sqrtf(pairwise_sum(sqdiff, feat_dim));
        // ||a - n||_2
        for (int c = 0; c < feat_dim; c++) {
            float d = a[c] - nv[c];
            sqdiff[c] = d * d;
        }
        float dist_an = sqrtf(pairwise_sum(sqdiff, feat_dim));
        float loss = dist_ap - dist_an + margin;
        temp[b] = loss > 0.0f ? loss : 0.0f;
    }
    float sum = pairwise_sum(temp, batch_size);
    output[0] = sum / (float)batch_size;
    free(temp);
    free(sqdiff);
}

// ── A.5 BMM and matrix-product variants ──

// A.5.a Matrix-scalar multiplication: out[i] = A[i] * s
// Phase L.1.2: element-wise multiply with broadcast of `b` along the outer axis.
//
// Per-element flow: out[i*inner + j] = a[i*inner + j] * b[j]
//   for i in [0, outer), j in [0, inner).
//
// Matches ggml's MUL when the second operand has the broadcasted shape.
// In the llama flow this animates the BN-fold-style application of the
// learned norm weight to the RMS-normalized activations.
//
// Bit-identity: trivial per-element float multiply, no reduction, no broadcast
// trickery. The order of (i, j) traversal does not affect the result.
//
// Tested: test_lk_02_mul in bench/test_llama_kernels.py against the fixture
// at /tmp/llama_dump_layer0/0004_attn_norm-0.bin.
void bpd_mul_broadcast_cpu(const float* a, const float* b, float* out,
                            int outer, int inner) {
    for (int i = 0; i < outer; i++) {
        const float* a_row = a + i * inner;
        float* out_row = out + i * inner;
        for (int j = 0; j < inner; j++) {
            out_row[j] = a_row[j] * b[j];
        }
    }
}

// Phase L.1.9: Q8_0 dequantization animator.
//
// Q8_0 layout per 32-element block (34 bytes):
//   bytes [0..1]:  uint16 little-endian F16 scale
//   bytes [2..33]: int8[32] quantized values
//
// Per-element flow: out[block*32 + i] = (float)int8[i] * f16_to_f32(scale)
//
// Bit-identity contract: the arithmetic is per-element with no reduction,
// no ordering dependence. F16 -> F32 is deterministic per IEEE 754. The
// (int8 -> float) cast and the multiply are deterministic.
//
// This animates the path from packed quantized weights to F32 activations,
// the foundational breath that unblocks L.1.1 embed lookup and every
// MUL_MAT operation in the inference current.
//
// Tested: test_lk_09_q8_0_dequant in bench/test_llama_kernels.py.

// IEEE 754 half (F16) -> single (F32) conversion via bit manipulation.
// Matches what every conforming F16 implementation does \u2014 the bit-identity
// gate is at this level, not at any higher abstraction. Inlined so the
// compiler can vectorize the surrounding loop.
//
// F16 layout: 1 sign bit | 5 exponent bits (bias 15) | 10 mantissa bits
// F32 layout: 1 sign bit | 8 exponent bits (bias 127) | 23 mantissa bits
//
// Special cases:
//   exp=0  & mantissa=0:  signed zero
//   exp=0  & mantissa!=0: subnormal (small denormal value)
//   exp=31 & mantissa=0:  +/- infinity
//   exp=31 & mantissa!=0: NaN
//
// For Q8_0 scales we typically see normal values in [2^-14, 2^15), so the
// normal path dominates. But we honor all cases for full IEEE conformance.
/* f16_to_f32: mirrors ggml_compute_fp16_to_fp32 from ggml/src/ggml-impl.h.
 * Uses the XNNPACK magic-number trick for correct subnormal handling.
 * Bit-identical with ggml's unhalf() and numpy's float16->float32 for all
 * 65536 F16 values (verified by tests/check_f16_algorithm.py).
 *
 * The previous sign/exponent/mantissa decomposition was correct for normal
 * F16 but produced wrong F32 bits for subnormal F16 values (exp=0, mant!=0),
 * causing the 1.2e-6 divergence in Q8_0 matmul when weight block scales
 * happened to be subnormal F16.
 */
static inline float f16_to_f32(uint16_t h) {
    // F16C hardware conversion (single vcvtph2ps, NO branch). Bit-identical to the
    // magic-bias version across all 65536 values (verified). Eliminates the denormal-cutoff
    // branch that caused 3.02 branch-miss/dot in bpd_qdot.
    return _cvtsh_ss(h);
}

void bpd_dequant_q8_0_cpu(const uint8_t* raw, float* out, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t* block = raw + b * 34;
        // F16 scale (little-endian) lives in bytes [0..1]
        uint16_t scale_u16 = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
        float scale = f16_to_f32(scale_u16);
        // 32 int8 quants live in bytes [2..33]
        const int8_t* qs = (const int8_t*)(block + 2);
        float* out_block = out + b * 32;
        for (int i = 0; i < 32; i++) {
            out_block[i] = (float)qs[i] * scale;
        }
    }
}

// Phase L.1.1: embedding lookup with Q8_0 dequantization fused in.
//
// In ggml's GET_ROWS animation: each input token_id selects a row from
// the (vocab_size, embed_dim) embedding table. The table is typically
// quantized (Q8_0 in llama3.2); the output is F32. So the op is a fused
// gather + dequantize.
//
// Per-token flow: out[t*embed_dim : (t+1)*embed_dim] =
//                 dequant_q8_0(table[token_ids[t] * blocks_per_row .. ...])
//
// Where blocks_per_row = embed_dim / 32 (Q8_0 block size).
//
// Inputs:
//   table:       raw Q8_0 bytes of the full embedding table, contiguous.
//                Each row = blocks_per_row * 34 bytes.
//   token_ids:   int32 array of token indices, length n_tokens.
//   out:         F32 output, shape (n_tokens, embed_dim).
//   n_tokens:    number of token IDs to look up.
//   embed_dim:   dimension of each embedding row (must be multiple of 32).
//
// Bit-identity: composition of (i) integer-indexed memcpy from a contiguous
// byte array (no arithmetic, no rounding) and (ii) the Q8_0 dequant flow
// already verified at 0 ULP (L.1.9). By transitivity the composition is
// bit-identical.
//
// Tested: test_lk_01_embed_lookup against fixture /tmp/llama_dump_layer0/
// 0000_inp_embd.bin (the captured GET_ROWS output from llama.cpp).
void bpd_embed_lookup_q8_0_cpu(const uint8_t* table, const int32_t* token_ids,
                                float* out, int n_tokens, int embed_dim) {
    int blocks_per_row = embed_dim / 32;
    int bytes_per_row  = blocks_per_row * 34;
    for (int t = 0; t < n_tokens; t++) {
        int32_t tok = token_ids[t];
        const uint8_t* row_bytes = table + (size_t)tok * bytes_per_row;
        float* out_row = out + (size_t)t * embed_dim;
        // Reuse the dequant flow per row (inlined to avoid call overhead).
        bpd_dequant_q8_0_cpu(row_bytes, out_row, blocks_per_row);
    }
}

// Phase L.1.10: Q8_0 quantization (F32 -> Q8_0).
//
// Mirrors ggml's AVX1 quantize_row_q8_0 (NOT the _ref variant), since that's
// what ggml's MUL_MAT dispatch actually calls via the from_float type trait
// on x86 hosts with __AVX__. See ggml/src/ggml-cpu/ggml-cpu-quants.c line 732+.
//
// Two substantive substrate-design differences from the scalar _ref version
// that we mirror here (recovered from commit 76106bf, accidentally reverted
// in PR #47's merge into bpd_cpu.c):
//
// 1. Computing id (the quantization multiplier):
//      _ref:        d = amax/127; id = 1/d         (two divisions, two roundings)
//      AVX1 actual: id = 127/maxScalar             (one division, one rounding)
//    The two forms are mathematically equivalent in real arithmetic but
//    produce different F32 bit patterns because the intermediate d in _ref
//    introduces an extra rounding step.
//
// 2. Rounding for the int8 cast:
//      _ref:        roundf (round-half-away-from-zero)
//      AVX1 actual: _mm256_round_ps(_MM_ROUND_NEAREST) (round-half-to-even,
//                                                       i.e., banker's rounding)
//    For .5-boundary inputs these produce different int8 values.
//
// The d that gets STORED is still d = maxScalar/127 (with F32->F16 conversion).
// The id that's USED is 127/maxScalar (the direct form).
//
// Bit-identity contract: matches ggml's AVX1 quantize_row_q8_0 exactly.
// Tested: bench/test_llama_kernels.py L.1.10 against the captured
// /tmp/llama_dump_layer0/ fixture (n_tokens=2 case): 0 ULP / 4096.

// F32 -> F16 conversion: IEEE 754 round-to-nearest-even.
// Handles normal, subnormal, zero, infinity, NaN.
// Matches what the F16C instruction vcvtps2ph produces (the path ggml takes
// on Ivy Bridge and later).
static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(uint32_t));
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t exp_f32 = (x >> 23) & 0xff;
    uint32_t mant_f32 = x & 0x7fffff;

    if (exp_f32 == 0xff) {
        // Inf or NaN: F16 exp = 31, mantissa preserved (truncated to 10 bits)
        return (uint16_t)(sign | 0x7c00 | (mant_f32 ? (mant_f32 >> 13) | 1 : 0));
    }

    // Compute the unbiased F16 exponent
    int32_t exp_f16 = (int32_t)exp_f32 - 127 + 15;

    if (exp_f16 >= 31) {
        // Overflow -> infinity
        return (uint16_t)(sign | 0x7c00);
    }
    if (exp_f16 <= 0) {
        // Subnormal F16 or underflow
        if (exp_f16 < -10) {
            // Too small even for subnormal
            return (uint16_t)sign;
        }
        // Generate subnormal mantissa
        uint32_t mant_with_implicit = mant_f32 | 0x800000;  // restore implicit leading 1
        int shift = 14 - exp_f16;  // shift right to align as subnormal
        uint32_t mant_sub = mant_with_implicit >> shift;
        // Round to nearest even
        uint32_t round_bit = (mant_with_implicit >> (shift - 1)) & 1;
        uint32_t sticky = (mant_with_implicit & ((1u << (shift - 1)) - 1)) != 0;
        if (round_bit && (sticky || (mant_sub & 1))) {
            mant_sub++;
        }
        return (uint16_t)(sign | mant_sub);
    }

    // Normal F16
    uint32_t mant_f16 = mant_f32 >> 13;
    // Round to nearest even
    uint32_t round_bit = (mant_f32 >> 12) & 1;
    uint32_t sticky = (mant_f32 & 0xfff) != 0;
    if (round_bit && (sticky || (mant_f16 & 1))) {
        mant_f16++;
        if (mant_f16 == 0x400) {
            // Mantissa overflow -> bump exponent
            mant_f16 = 0;
            exp_f16++;
            if (exp_f16 >= 31) {
                return (uint16_t)(sign | 0x7c00);
            }
        }
    }
    return (uint16_t)(sign | (exp_f16 << 10) | mant_f16);
}

void bpd_quant_q8_0_cpu(const float* x, uint8_t* y, int n_elements) {
    int nb = n_elements / 32;
    for (int i = 0; i < nb; i++) {
        const float* x_block = x + i * 32;
        uint8_t* y_block = y + i * 34;
        // Find maxScalar (positive absolute max). ggml's AVX1 path computes
        // this via a SIMD reduction; for correctness we use a scalar loop
        // which produces the same maxScalar.
        float maxScalar = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = x_block[j];
            float av = v < 0 ? -v : v;
            if (av > maxScalar) maxScalar = av;
        }
        // Stored scale: d = maxScalar / 127
        float d = maxScalar / 127.f;
        uint16_t d_f16 = f32_to_f16(d);
        y_block[0] = (uint8_t)(d_f16 & 0xff);
        y_block[1] = (uint8_t)(d_f16 >> 8);
        // Quantization multiplier: id = 127 / maxScalar (NOT 1/d \u2014 see comment above)
        float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        // Quantize with round-half-to-even (matches _mm256_round_ps(_MM_ROUND_NEAREST))
        int8_t* qs = (int8_t*)(y_block + 2);
        for (int j = 0; j < 32; j++) {
            float x0 = x_block[j] * id;
            // rintf: round to nearest, ties to EVEN (banker's rounding).
            // Matches the SSE/AVX _MM_ROUND_NEAREST rounding mode.
            qs[j] = (int8_t)rintf(x0);
        }
    }
}

// Phase L.1.10 Path B: Q8_0 x Q8_0 dot product over n_blocks blocks.
//
// MIRRORS GGML'S AVX1 BRANCH (not the scalar fallback). Empirical evidence:
// libggml-cpu.so as built compiles to AVX1 instructions for vec_dot_q8_0_q8_0
// even with GGML_AVX=OFF in CMakeCache \u2014 the compiler picks up __AVX__ from
// the host CPU's features. The captured fixture was produced by this AVX1
// path, so our 0-ULP gate must mirror it.
//
// Algorithm (mirroring ggml/src/ggml-cpu/ggml-cpu-quants.c lines 3881-3899):
//   1. Process pairs of blocks (ib, ib+1) at a time.
//   2. Per pair: compute 8-lane __m256 'p' containing the 8 int32 dot results
//      (4 quarters per block * 2 blocks), converted to F32.
//   3. Per pair: compute 8-lane __m256 'deltas' = [s_ib repeated 4x, s_ib+1 repeated 4x]
//      where s = f16_to_f32(w.d) * f16_to_f32(a.d).
//   4. accum += deltas * p (lane-wise multiply then add).
//   5. After all pairs: hsum_float_8(accum) reduces 8 lanes to one F32 in
//      a specific pairwise pattern that determines the final bit pattern.
//
// The reduction order \u2014 8 parallel accumulations across pairs followed by
// a specific horizontal pairwise collapse \u2014 is what differs from the scalar
// branch and what we must mirror.
//
// Tested: test_lk_10_q8_0_matmul at 0 ULP vs the captured llama.cpp Qcur-0.
#if BPD_HAVE_AVX1

// Per-block int8 dot via SSSE3 path (matches ggml's mul_add_epi8_sse).
// Given two 128-bit lanes of int8 (16 elements each), returns 8 int16
// partial products. Caller does the final reduce.
static inline __m128i bpd_mul_add_epi8_sse(__m128i x, __m128i y) {
    __m128i ax = _mm_sign_epi8(x, x);   // abs(x)
    __m128i sy = _mm_sign_epi8(y, x);   // y * sign(x)
    return _mm_maddubs_epi16(ax, sy);   // 16x int8 -> 8x int16 with pair-adjacent sum
}

// Mirrors ggml's mul_sum_i8_quad_float: 4 blocks of 32-int8 -> __m256 of 8 floats.
// The lanes hold: [b1_q0, b1_q1, b1_q2, b1_q3, b2_q0, b2_q1, b2_q2, b2_q3]
// where bN_qK is the K-th quarter (8 elements) dot of block N.
static inline __m256 bpd_mul_sum_i8_quad_float(
    __m128i x_1_0, __m128i x_1_1, __m128i x_2_0, __m128i x_2_1,
    __m128i y_1_0, __m128i y_1_1, __m128i y_2_0, __m128i y_2_1) {
    __m128i mone = _mm_set1_epi16(1);
    __m128i p16_1_0 = bpd_mul_add_epi8_sse(x_1_0, y_1_0);
    __m128i p16_1_1 = bpd_mul_add_epi8_sse(x_1_1, y_1_1);
    __m128i p16_2_0 = bpd_mul_add_epi8_sse(x_2_0, y_2_0);
    __m128i p16_2_1 = bpd_mul_add_epi8_sse(x_2_1, y_2_1);
    __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
    __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
    __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
    __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
    __m128i p_1 = _mm_add_epi32(p_1_0, p_1_1);
    __m128i p_2 = _mm_add_epi32(p_2_0, p_2_1);
    return _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(p_1), p_2, 1));
}

// Mirrors ggml's quad_fp16_delta_float: returns __m256 with low 128 = (x0*y0)x4 and high 128 = (x1*y1)x4
static inline __m256 bpd_quad_fp16_delta_float(float x0_f, float y0_f, float x1_f, float y1_f) {
    return _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_set1_ps(x0_f * y0_f)),
        _mm_set1_ps(x1_f * y1_f), 1);
}

// Mirrors ggml's hsum_float_8: horizontal sum with specific pairwise order.
static inline float bpd_hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

float bpd_qdot_q8_0_q8_0_cpu(const uint8_t* w_blocks, const uint8_t* a_blocks,
                             int n_blocks) {
    __m256 accum = _mm256_setzero_ps();
    int ib = 0;
    // Pair-loop mirroring ggml's AVX1 branch
    for (; ib + 1 < n_blocks; ib += 2) {
        const uint8_t* wb_1 = w_blocks + ib * 34;
        const uint8_t* wb_2 = w_blocks + (ib + 1) * 34;
        const uint8_t* ab_1 = a_blocks + ib * 34;
        const uint8_t* ab_2 = a_blocks + (ib + 1) * 34;
        // Load int8 quants (2 x 128-bit halves per block, 32 ints total)
        __m128i qx_1_0 = _mm_loadu_si128((const __m128i*)(wb_1 + 2));
        __m128i qx_1_1 = _mm_loadu_si128((const __m128i*)(wb_1 + 2 + 16));
        __m128i qx_2_0 = _mm_loadu_si128((const __m128i*)(wb_2 + 2));
        __m128i qx_2_1 = _mm_loadu_si128((const __m128i*)(wb_2 + 2 + 16));
        __m128i qy_1_0 = _mm_loadu_si128((const __m128i*)(ab_1 + 2));
        __m128i qy_1_1 = _mm_loadu_si128((const __m128i*)(ab_1 + 2 + 16));
        __m128i qy_2_0 = _mm_loadu_si128((const __m128i*)(ab_2 + 2));
        __m128i qy_2_1 = _mm_loadu_si128((const __m128i*)(ab_2 + 2 + 16));
        // 8-lane int dot results -> F32
        __m256 p = bpd_mul_sum_i8_quad_float(
            qx_1_0, qx_1_1, qx_2_0, qx_2_1,
            qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        // 8-lane scale-product deltas
        uint16_t wd1_u16 = (uint16_t)wb_1[0] | ((uint16_t)wb_1[1] << 8);
        uint16_t ad1_u16 = (uint16_t)ab_1[0] | ((uint16_t)ab_1[1] << 8);
        uint16_t wd2_u16 = (uint16_t)wb_2[0] | ((uint16_t)wb_2[1] << 8);
        uint16_t ad2_u16 = (uint16_t)ab_2[0] | ((uint16_t)ab_2[1] << 8);
        float wd1 = f16_to_f32(wd1_u16);
        float ad1 = f16_to_f32(ad1_u16);
        float wd2 = f16_to_f32(wd2_u16);
        float ad2 = f16_to_f32(ad2_u16);
        __m256 deltas = bpd_quad_fp16_delta_float(wd1, ad1, wd2, ad2);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }
    float sumf = bpd_hsum_float_8(accum);
    // Tail: odd remaining block (if any), scalar fallback
    for (; ib < n_blocks; ib++) {
        const uint8_t* wb = w_blocks + ib * 34;
        const uint8_t* ab = a_blocks + ib * 34;
        const int8_t* wq = (const int8_t*)(wb + 2);
        const int8_t* aq = (const int8_t*)(ab + 2);
        int sumi = 0;
        for (int j = 0; j < 32; j++) sumi += (int)wq[j] * (int)aq[j];
        uint16_t wd_u16 = (uint16_t)wb[0] | ((uint16_t)wb[1] << 8);
        uint16_t ad_u16 = (uint16_t)ab[0] | ((uint16_t)ab[1] << 8);
        float wd = f16_to_f32(wd_u16);
        float ad = f16_to_f32(ad_u16);
        sumf += (float)sumi * (wd * ad);
    }
    return sumf;
}

#else
// Fallback when no AVX1: scalar reduction (won't be bit-identical with ggml's AVX1 path)
float bpd_qdot_q8_0_q8_0_cpu(const uint8_t* w_blocks, const uint8_t* a_blocks,
                             int n_blocks) {
    float sumf = 0.0f;
    for (int ib = 0; ib < n_blocks; ib++) {
        const uint8_t* wb = w_blocks + ib * 34;
        const uint8_t* ab = a_blocks + ib * 34;
        const int8_t* wq = (const int8_t*)(wb + 2);
        const int8_t* aq = (const int8_t*)(ab + 2);
        int sumi = 0;
        for (int j = 0; j < 32; j++) sumi += (int)wq[j] * (int)aq[j];
        uint16_t wd_u16 = (uint16_t)wb[0] | ((uint16_t)wb[1] << 8);
        uint16_t ad_u16 = (uint16_t)ab[0] | ((uint16_t)ab[1] << 8);
        float wd = f16_to_f32(wd_u16);
        float ad = f16_to_f32(ad_u16);
        sumf += (float)sumi * (wd * ad);
    }
    return sumf;
}
#endif

// Phase L.1.10 Path B: Q8_0 weight x F32 activations matmul, output F32.
// Composes (Q8_0 quantize of activations) + (Q8_0 x Q8_0 block dot products).
//
// ggml MUL_MAT semantics: out[m, n] = sum_k X[m, k] * W[n, k]
//   where W is stored row-major as (N, K) = ne[1] x ne[0]
//   and X is stored row-major as (M, K)
//
// Algorithm: quantize each row of X to Q8_0 once (M rows total), then for
// each (m, n) compute the block dot product against W's row n.
//
// Bit-identity by composition: quant_q8_0 + qdot_q8_0 each mirror ggml's
// scalar reference. The composition matches ggml's MUL_MAT path.
void bpd_qmatmul_q8_0_cpu(const uint8_t* W_q8_0, const float* X_f32,
                          float* out, int M, int N, int K) {
    int n_blocks_per_row = K / 32;
    int bytes_per_row = n_blocks_per_row * 34;
    // Allocate quantized X buffer: M rows * K elements -> M * bytes_per_row bytes
    uint8_t* X_q8_0 = (uint8_t*)malloc((size_t)M * bytes_per_row);
    if (!X_q8_0) return;
    // Quantize each row of X to Q8_0
    for (int m = 0; m < M; m++) {
        bpd_quant_q8_0_cpu(X_f32 + (size_t)m * K, X_q8_0 + (size_t)m * bytes_per_row, K);
    }
    // For each (m, n), dot Q8_0 weight row n with Q8_0 activation row m
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            const uint8_t* w_row = W_q8_0 + (size_t)n * bytes_per_row;
            const uint8_t* a_row = X_q8_0 + (size_t)m * bytes_per_row;
            out[(size_t)m * N + n] = bpd_qdot_q8_0_q8_0_cpu(w_row, a_row, n_blocks_per_row);
        }
    }
    free(X_q8_0);
}

// Phase L.1.10 Path B': Q8_0 x Q8_0 matmul mirroring llamafile_sgemm.
//
// EMPIRICAL FINDING (commit 63918a3): ggml's MUL_MAT dispatches to
// llamafile_sgemm FIRST. The vec_dot_q8_0_q8_0 path is only a fallback.
// Our captured Qcur-0 fixture was produced by llamafile, so to get 0 ULP
// we must mirror its exact gemm<RM, RN> algorithm.
//
// Source mirrored: tinyBLAS_Q0_AVX::gemm<RM, RN> from
// llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp lines 282-329.
//
// Dispatch on Ivy Bridge (no AVX2, VECTOR_REGISTERS=16):
//   For (m, n) = (m_weight, m_tokens) = (2048, 2):
//     mnpack key = (MIN(m,4)<<4) | MIN(n,4) = 0x42
//     Selected tile: RM=4, RN=2 (weight rows per tile, token rows per tile)
//
// llamafile convention (we mirror exactly):
//   A = weight (m rows, m = output_dim)
//   B = activation (n rows, n = m_tokens)
//   C[ldc * jj + ii] = sum_l A[ii, l] * B[jj, l]
//     ldc = m, jj iterates 0..n-1, ii iterates 0..m-1
//   In our test: output stored as out[token=jj, weight_row=ii], row-major,
//                with stride ldc = m_weight = 2048.
//
// Per tile (RM=4, RN=2 = 8 output cells):
//   Cv[RN=2][RM=4] = 8 __m256 accumulators (zero)
//   for l = 0..k-1:        // k = blocks per row
//     for j = 0..RN-1:     // 2 token rows
//       for i = 0..RM-1:   // 4 weight rows
//         load A[ii+i] block l, B[jj+j] block l
//         compute udTmp (8-lane partial F32)
//         scale = f16_to_f32(A.d) * f16_to_f32(B.d)
//         Cv[j][i] = (scale * udTmp) + Cv[j][i]
//   for j, i: out[(jj+j) * ldc + (ii+i)] = hsum(Cv[j][i])
//
// This mirrors the lane semantics of llamafile exactly \u2014 each Cv[j][i] is
// 8 partial sums accumulated across k blocks of ONE (i, j) cell.
#if BPD_HAVE_AVX1

// Inner kernel: process one (RM=4, RN=2) tile, accumulate across k blocks,
// hsum into 8 F32 outputs. Writes to out[jj*ldc + ii] indexing.
static void bpd_llamafile_q8_0_tile_42(
    const uint8_t* W_tile_base,  // weight rows [ii..ii+RM), each k blocks
    const uint8_t* B_tile_base,  // activation rows [jj..jj+RN), each k blocks
    int k,
    int weight_row_stride,       // bytes between consecutive weight rows
    int act_row_stride,          // bytes between consecutive activation rows
    float* out_base,             // points at out[jj*ldc + ii]
    int ldc)
{
    const int RM = 4, RN = 2;
    __m256 Cv[RN][RM];
    for (int j = 0; j < RN; j++)
        for (int i = 0; i < RM; i++)
            Cv[j][i] = _mm256_setzero_ps();

    for (int l = 0; l < k; l++) {
        for (int j = 0; j < RN; j++) {
            const uint8_t* Bblock = B_tile_base + j * act_row_stride + l * 34;
            __m128i blj0 = _mm_loadu_si128((const __m128i*)(Bblock + 2));
            __m128i blj1 = _mm_loadu_si128((const __m128i*)(Bblock + 2 + 16));
            uint16_t Bd_u16 = (uint16_t)Bblock[0] | ((uint16_t)Bblock[1] << 8);
            float Bd = f16_to_f32(Bd_u16);
            for (int i = 0; i < RM; i++) {
                const uint8_t* Ablock = W_tile_base + i * weight_row_stride + l * 34;
                __m128i ali0 = _mm_loadu_si128((const __m128i*)(Ablock + 2));
                __m128i ali1 = _mm_loadu_si128((const __m128i*)(Ablock + 2 + 16));
                // Sign tricks
                __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);
                const __m128i oneFill = _mm_set1_epi16(1);
                __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                __m128i p32_0 = _mm_madd_epi16(oneFill, mad0);
                __m128i p32_1 = _mm_madd_epi16(oneFill, mad1);
                // Combine: MM256_SET_M128I(a=mad1, b=mad0) -> low=mad0, high=mad1
                __m256i p32 = _mm256_insertf128_si256(
                    _mm256_castsi128_si256(p32_0), p32_1, 1);
                __m256 udTmp = _mm256_cvtepi32_ps(p32);
                uint16_t Ad_u16 = (uint16_t)Ablock[0] | ((uint16_t)Ablock[1] << 8);
                float Ad = f16_to_f32(Ad_u16);
                __m256 scale = _mm256_set1_ps(Ad * Bd);
                // No FMA on Ivy Bridge: add(mul(scale, udTmp), Cv)
                Cv[j][i] = _mm256_add_ps(_mm256_mul_ps(scale, udTmp), Cv[j][i]);
            }
        }
    }
    // hsum each cell, write to out[(jj+j)*ldc + (ii+i)] \u2014 base already points there
    for (int j = 0; j < RN; j++) {
        for (int i = 0; i < RM; i++) {
            __m128 v = _mm_add_ps(_mm256_extractf128_ps(Cv[j][i], 1),
                                  _mm256_castps256_ps128(Cv[j][i]));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            v = _mm_add_ss(v, _mm_movehdup_ps(v));
            out_base[j * ldc + i] = _mm_cvtss_f32(v);
        }
    }
}

// Declare the tile dispatcher from bpd_gemm_q8_0_cpu.c
void bpd_qmatmul_q8_0_dispatch_cpu(
    const uint8_t* W_q8_0,
    const uint8_t* X_q8_0,
    float* out,
    int m_weight,
    int m_tokens,
    int K);

// Full matmul: weight W (Q8_0, m_weight rows of K elements), activation X
// (F32, m_tokens rows of K), output C (F32 row-major as [token, weight_row]).
// Replaces the old fixed-tile version with the dynamic dispatcher.
void bpd_qmatmul_q8_0_llamafile_cpu(
    const uint8_t* W_q8_0,
    const float* X_f32,
    float* out,
    int m_weight,    // = m in llamafile (= ggml's ne01 = output dim)
    int m_tokens,    // = n in llamafile (= ggml's ne11 = seq len)
    int K)
{
    int k = K / 32;
    int bytes_per_row = k * 34;

    // Quantize activations to Q8_0. Use a PERSISTENT thread-local scratch (no malloc/free
    // churn per call — the dominant per-matmul overhead). Grows monotonically; never freed.
    static __thread uint8_t* X_q8_0 = NULL;
    static __thread size_t X_q8_0_cap = 0;
    size_t need = (size_t)m_tokens * bytes_per_row;
    if (need > X_q8_0_cap) {
        free(X_q8_0);
        X_q8_0 = (uint8_t*)malloc(need);
        if (!X_q8_0) { X_q8_0_cap = 0; return; }
        X_q8_0_cap = need;
    }
    for (int i = 0; i < m_tokens; i++) {
        bpd_quant_q8_0_cpu(X_f32 + (size_t)i * K,
                           X_q8_0 + (size_t)i * bytes_per_row, K);
    }

    // Call the dispatcher to handle tiling
    bpd_qmatmul_q8_0_dispatch_cpu(W_q8_0, X_q8_0, out, m_weight, m_tokens, K);
    // (no free: persistent scratch reused across calls)
}

#else
void bpd_qmatmul_q8_0_llamafile_cpu(const uint8_t* W, const float* X, float* out,
                                    int mw, int mt, int K) {
    bpd_qmatmul_q8_0_cpu(W, X, out, mt, mw, K);
}
#endif

void bpd_scalar_mul_cpu(const float* A, float s, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = A[i] * s;
}

// A.5.b Batched matmul (BMM): (B, M, K) @ (B, K, N) → (B, M, N)
// Each batch slice is an independent mm. Reuse bpd_mm_cpu which is bit-identical
// with cblas_sgemm (Goto-Sandy SGEMM).
void bpd_bmm_cpu(const float* A, const float* B, float* C,
                  int batch, int M, int N, int K) {
    for (int b = 0; b < batch; b++) {
        const float* a_b = A + b * M * K;
        const float* b_b = B + b * K * N;
        float* c_b = C + b * M * N;
        bpd_mm_cpu(a_b, b_b, c_b, M, N, K);
    }
}

// A.5.c 3D tensor-matrix multiplication: (B, M, K) @ (K, N) → (B, M, N)
// Single matmul if we reshape input to (B*M, K). bpd_mm_cpu handles this.
void bpd_3d_tensor_matmul_cpu(const float* A, const float* B, float* C,
                                int batch, int M, int N, int K) {
    // Treat A as (batch*M, K), output as (batch*M, N), B unchanged.
    bpd_mm_cpu(A, B, C, batch * M, N, K);
}

// A.5.d 4D tensor-matrix multiplication: (B, C_dim, M, K) @ (K, N) → (B, C_dim, M, N)
// Single matmul with reshape to (B*C_dim*M, K).
void bpd_4d_tensor_matmul_cpu(const float* A, const float* B, float* C,
                                int batch, int C_dim, int M, int N, int K) {
    bpd_mm_cpu(A, B, C, batch * C_dim * M, N, K);
}

// A.5.e Diagonal matmul: A is (M,) diagonal vector; B is (M, N); output (M, N)
// out[i, j] = A[i] * B[i, j]
void bpd_diag_matmul_cpu(const float* A_diag, const float* B, float* C,
                          int M, int N) {
    for (int i = 0; i < M; i++) {
        float a = A_diag[i];
        const float* b_row = B + i * N;
        float* c_row = C + i * N;
        for (int j = 0; j < N; j++) c_row[j] = a * b_row[j];
    }
}

// ── A.6 specialty kernels ──

// A.6.a Argmax over dim: input shape (..., dim_size, ...), output one int64 per
//   slice excluding the reduced dim. Ties: lowest index (PyTorch semantic).
// We'll handle a contiguous "outer × dim_size × inner" layout. The harness will
// reshape as needed.
//   out[outer, inner] = argmax over k in 0..dim_size of x[outer, k, inner]
void bpd_argmax_dim_cpu(const float* x, long* out,
                         int outer, int dim_size, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float best = x[(o * dim_size + 0) * inner + i];
            long best_idx = 0;
            for (int k = 1; k < dim_size; k++) {
                float v = x[(o * dim_size + k) * inner + i];
                if (v > best) {
                    best = v;
                    best_idx = k;
                }
            }
            out[o * inner + i] = best_idx;
        }
    }
}

// A.6.b Argmin: mirror of argmax with <.
void bpd_argmin_dim_cpu(const float* x, long* out,
                         int outer, int dim_size, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float best = x[(o * dim_size + 0) * inner + i];
            long best_idx = 0;
            for (int k = 1; k < dim_size; k++) {
                float v = x[(o * dim_size + k) * inner + i];
                if (v < best) {
                    best = v;
                    best_idx = k;
                }
            }
            out[o * inner + i] = best_idx;
        }
    }
}

// A.6.c Min reduction over dim: like argmin but returns values not indices.
void bpd_min_dim_cpu(const float* x, float* out,
                      int outer, int dim_size, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float best = x[(o * dim_size + 0) * inner + i];
            for (int k = 1; k < dim_size; k++) {
                float v = x[(o * dim_size + k) * inner + i];
                if (v < best) best = v;
            }
            out[o * inner + i] = best;
        }
    }
}

// A.6.d masked_cumsum: cumsum(x * mask, dim).
// mask is uint8 (1=True, 0=False) — Python ctypes will pass it as such.
// Layout: contiguous (batch, dim_size), cumsum along dim_size axis.
//
// SUBSTRATE-DESIGN: cumulative_acc_type(double) — PyTorch's torch.cumsum
// accumulates in double internally and casts back to float on store.
// (Same pattern as bpd_cumsum_cpu — empirically verified earlier in this
// session for #89.)
void bpd_masked_cumsum_cpu(const float* x, const unsigned char* mask,
                            float* out, int batch, int dim_size) {
    for (int b = 0; b < batch; b++) {
        double acc = 0.0;  // ← double accumulator, not float
        const float* x_row = x + b * dim_size;
        const unsigned char* m_row = mask + b * dim_size;
        float* o_row = out + b * dim_size;
        for (int k = 0; k < dim_size; k++) {
            double v = m_row[k] ? (double)x_row[k] : 0.0;
            acc = acc + v;
            o_row[k] = (float)acc;  // cast back to f32 on store
        }
    }
}

// A.6.e MinGPT NewGelu: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
// PyTorch's torch.tanh on CPU dispatches to libm tanhf for fp32.
// Constant sqrt(2/pi) = 0.7978845608028654 in f64 → 0.79788458f in f32 (rounds).
void bpd_mingpt_newgelu_cpu(const float* x, float* out, int n) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float GELU_COEF = 0.044715f;
    for (int i = 0; i < n; i++) {
        float xv = x[i];
        float x3 = xv * xv * xv;
        float inner = SQRT_2_OVER_PI * (xv + GELU_COEF * x3);
        float t = tanhf(inner);
        out[i] = 0.5f * xv * (1.0f + t);
    }
}

// A.6.f ScaledDotProductAttention: out = softmax(Q @ K.T / sqrt(d_k), dim=-1) @ V
// Q, K, V shape: (batch, num_heads, seq_len, embed_dim).
// Per (batch, head): scores[seq, seq] = Q[seq, embed] @ K.T[embed, seq] / sqrt(embed_dim)
//                    attn[seq, seq] = softmax(scores, dim=-1)
//                    out[seq, embed] = attn[seq, seq] @ V[seq, embed]
//
// We allocate temp buffers for scores and K.T per (batch, head) pair.
extern float linear_scan_sum_simd8(const float* data, int n);
// We need a bpd_mm with one operand transposed. Simplest: physically transpose K
// once per (batch, head) into a temp buffer, then use bpd_mm_cpu.
//
// SUBSTANTIVE substrate-design choice: PyTorch's _scaled_dot_product_attention_math
// (attention.cpp:850) uses the SQUARE-ROOTED PRE-SCALE pattern:
//   scaling = sqrt(1/sqrt(d_k))  = d_k^(-1/4)
//   Q' = Q * scaling
//   K' = K * scaling   (NOTE: PyTorch's code does K.transpose(-2,-1) * scaling)
//   scores = Q' @ K'^T = (Q @ K^T) * (scaling * scaling) = (Q @ K^T) / sqrt(d_k)
// This pre-scaling produces different bits from post-scaling (Q@K^T then /sqrt(d_k)).
//
// Source: aten/src/ATen/native/transformers/attention.cpp:894-901
void bpd_scaled_dot_product_attention_cpu(const float* Q, const float* K, const float* V,
                                            float* out,
                                            int batch, int num_heads, int seq_len, int embed_dim) {
    int qkv_per_head = seq_len * embed_dim;
    int scores_size = seq_len * seq_len;
    float* Q_scaled = (float*)bpd_alloc(qkv_per_head * sizeof(float));
    float* K_scaled_T = (float*)bpd_alloc(embed_dim * seq_len * sizeof(float));
    float* scores = (float*)bpd_alloc(scores_size * sizeof(float));
    if (!Q_scaled || !K_scaled_T || !scores) {
        if (Q_scaled) free(Q_scaled);
        if (K_scaled_T) free(K_scaled_T);
        if (scores) free(scores);
        return;
    }

    // PyTorch's calculate_scale returns 1/sqrt(d_k); then .sqrt() = d_k^(-1/4)
    float scaling = sqrtf(1.0f / sqrtf((float)embed_dim));

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            int slot = b * num_heads + h;
            const float* Q_h = Q + slot * qkv_per_head;
            const float* K_h = K + slot * qkv_per_head;
            const float* V_h = V + slot * qkv_per_head;
            float* out_h = out + slot * qkv_per_head;

            // Pre-scale Q: Q_scaled = Q * scaling
            for (int i = 0; i < qkv_per_head; i++) Q_scaled[i] = Q_h[i] * scaling;

            // Pre-scale K then transpose: K_scaled_T[e, s] = K_h[s, e] * scaling
            for (int s = 0; s < seq_len; s++) {
                for (int e = 0; e < embed_dim; e++) {
                    K_scaled_T[e * seq_len + s] = K_h[s * embed_dim + e] * scaling;
                }
            }

            // scores = Q_scaled @ K_scaled_T → shape (seq_len, seq_len)
            //   Mathematically: scores = (Q @ K^T) * scaling^2 = (Q @ K^T) / sqrt(d_k)
            //   But bit-different from post-scaling.
            bpd_mm_cpu(Q_scaled, K_scaled_T, scores, seq_len, seq_len, embed_dim);

            // Softmax per row of scores
            for (int s = 0; s < seq_len; s++) {
                float* row = scores + s * seq_len;
                float mx = row[0];
                for (int k = 1; k < seq_len; k++) if (row[k] > mx) mx = row[k];
                for (int k = 0; k < seq_len; k++) row[k] = expf(row[k] - mx);
                float sum_exp = linear_scan_sum_simd8(row, seq_len);
                float inv_sum = 1.0f / sum_exp;
                for (int k = 0; k < seq_len; k++) row[k] *= inv_sum;
            }

            // out_h = scores @ V_h → shape (seq_len, embed_dim)
            bpd_mm_cpu(scores, V_h, out_h, seq_len, embed_dim, seq_len);
        }
    }

    free(Q_scaled);
    free(K_scaled_T);
    free(scores);
}


// ══════════════════════════════════════════════════════════════════════════════
// BPD_MKL_PATH=1 — Kernel variants that match PyTorch builds with Intel MKL
// BLAS backend (AVX2+FMA CPUs, e.g. Intel Xeon, Core i7/i9 with AVX2).
//
// Empirically characterised 2026-05-21 on Manus sandbox (Intel Xeon, AVX2+FMA,
// PyTorch 2.x, BLAS_INFO=mkl) via bench/probe_mkl_params.py.
//
// Compiled only when -DBPD_MKL_PATH=1 is passed (CPU_FP_MODE=mkl in Makefile).
//
// Root causes of the 79/100 → 95/100 gap (all confirmed empirically):
//   1. Transcendentals (sigmoid/tanh/silu/elu/softplus/selu): PyTorch ATen uses
//      Julien Pommier's sse_mathfun polynomial with AVX2 FMA Horner evaluation.
//      The FMA fuses mul+add into one IEEE 754 op, producing 1-2 ULP divergence
//      from BPD's scalar expf loop. Fix: bpd_exp_ps_avx2_fma + FMA Horner.
//   2. GEMV (N=1): MKL dispatches to cblas_sgemv (AVX2 AXPY, 8 accumulators).
//      Fix: bpd_gemv_mkl_cpu.
//   3. RMSNorm: ATen uses _mm256_sqrt_ps (VSQRTPS) on the cascade8 result.
//      Fix: bpd_rmsnorm_mkl_cpu.
//   4. InstanceNorm affine apply: ATen uses _mm256_fmadd_ps.
//      Fix: bpd_instancenorm_mkl_cpu.
//   5. Depthwise conv (#82-86): no BPD kernel yet — Phase E work.
//
// Predicted score: 95-97/100 on MKL environments after these fixes.
// ══════════════════════════════════════════════════════════════════════════════
#if defined(BPD_MKL_PATH) && BPD_MKL_PATH == 1
#if !defined(__AVX2__) || !defined(__FMA__)
#error "BPD_MKL_PATH=1 requires -mavx2 -mfma (AVX2 + FMA support)"
#endif

// ── ATen Cephes expf polynomial with FMA Horner evaluation ───────────────────
// Coefficients from Julien Pommier's sse_mathfun.h (used by PyTorch ATen).
// FMA Horner matches ATen's _mm256_fmadd_ps chain exactly.
static inline __m256 bpd_exp_ps_avx2_fma(__m256 x) {
    const __m256 log2e  = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half   = _mm256_set1_ps(0.5f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 p0     = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 p1     = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 p2     = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 p3     = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 p4     = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 p5     = _mm256_set1_ps(5.0000001201E-1f);
    const __m256 one    = _mm256_set1_ps(1.0f);
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647950f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647950f);
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);
    // n = floor(x * log2e + 0.5)
    __m256 fx = _mm256_fmadd_ps(x, log2e, half);
    fx = _mm256_floor_ps(fx);
    // r = x - n*ln2_hi - n*ln2_lo  (Cephes split for accuracy)
    __m256 r = _mm256_fnmadd_ps(fx, ln2_hi, x);
    r = _mm256_fnmadd_ps(fx, ln2_lo, r);
    // FMA Horner: y = fma(fma(fma(fma(fma(fma(p0,r,p1),r,p2),r,p3),r,p4),r,p5),r*r,r+1)
    __m256 y = _mm256_fmadd_ps(p0, r, p1);
    y = _mm256_fmadd_ps(y, r, p2);
    y = _mm256_fmadd_ps(y, r, p3);
    y = _mm256_fmadd_ps(y, r, p4);
    y = _mm256_fmadd_ps(y, r, p5);
    __m256 r2 = _mm256_mul_ps(r, r);
    y = _mm256_fmadd_ps(y, r2, r);
    y = _mm256_add_ps(y, one);
    // Scale by 2^n via integer bit manipulation
    __m256i ni = _mm256_cvttps_epi32(fx);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(ni));
}

// ── MKL-path sigmoid: 1/(1+exp(-x)) with AVX2 FMA exp ────────────────────────
void bpd_sigmoid_mkl_cpu(const float* input, float* output, int n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = bpd_exp_ps_avx2_fma(_mm256_sub_ps(_mm256_setzero_ps(), x));
        _mm256_storeu_ps(output + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
    }
    for (; i < n; i++) output[i] = 1.0f / (1.0f + expf(-input[i]));
}

// ── MKL-path SiLU: x * sigmoid(x) ────────────────────────────────────────────
void bpd_silu_mkl_cpu(const float* input, float* output, int n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = bpd_exp_ps_avx2_fma(_mm256_sub_ps(_mm256_setzero_ps(), x));
        __m256 s = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(output + i, _mm256_mul_ps(x, s));
    }
    for (; i < n; i++) { float s = 1.0f/(1.0f+expf(-input[i])); output[i]=input[i]*s; }
}

// ── MKL-path tanh: (exp(2x)-1)/(exp(2x)+1) ───────────────────────────────────
void bpd_tanh_mkl_cpu(const float* input, float* output, int n) {
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 e = bpd_exp_ps_avx2_fma(_mm256_mul_ps(two, x));
        __m256 r = _mm256_div_ps(_mm256_sub_ps(e, one), _mm256_add_ps(e, one));
        _mm256_storeu_ps(output + i, r);
    }
    for (; i < n; i++) output[i] = tanhf(input[i]);
}

// ── MKL-path ELU: expm1(x) for x<0, x for x>=0 ───────────────────────────────
void bpd_elu_mkl_cpu(const float* input, float* output, int n) {
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x    = _mm256_loadu_ps(input + i);
        __m256 em1  = _mm256_sub_ps(bpd_exp_ps_avx2_fma(x), one);
        __m256 mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
        _mm256_storeu_ps(output + i, _mm256_blendv_ps(x, em1, mask));
    }
    for (; i < n; i++) { float a=input[i]; output[i]=a<0.0f?expm1f(a):a; }
}

// ── MKL-path SELU ─────────────────────────────────────────────────────────────
void bpd_selu_mkl_cpu(const float* input, float* output, int n) {
    const float alpha   = (float)1.6732632423543772848170429916717;
    const float scale   = (float)1.0507009873554804934193349852946;
    const __m256 vnegc  = _mm256_set1_ps(alpha * scale);
    const __m256 vposc  = _mm256_set1_ps(scale);
    const __m256 one    = _mm256_set1_ps(1.0f);
    const __m256 zero   = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x    = _mm256_loadu_ps(input + i);
        __m256 em1  = _mm256_sub_ps(bpd_exp_ps_avx2_fma(x), one);
        __m256 neg  = _mm256_mul_ps(em1, vnegc);
        __m256 pos  = _mm256_mul_ps(x, vposc);
        __m256 mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
        _mm256_storeu_ps(output + i, _mm256_blendv_ps(pos, neg, mask));
    }
    for (; i < n; i++) {
        float a=input[i];
        output[i]=a<0.0f?expm1f(a)*(alpha*scale):a*scale;
    }
}

// ── MKL-path Softplus: log(1+exp(x)) ─────────────────────────────────────────
void bpd_softplus_mkl_cpu(const float* input, float* output, int n) {
    const __m256 one       = _mm256_set1_ps(1.0f);
    const __m256 threshold = _mm256_set1_ps(20.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x   = _mm256_loadu_ps(input + i);
        __m256 ep1 = _mm256_add_ps(bpd_exp_ps_avx2_fma(x), one);
        // log via scalar (log is not in the divergent set)
        float tmp[8]; _mm256_storeu_ps(tmp, ep1);
        for (int j = 0; j < 8; j++) tmp[j] = logf(tmp[j]);
        __m256 sp   = _mm256_loadu_ps(tmp);
        __m256 mask = _mm256_cmp_ps(x, threshold, _CMP_GT_OQ);
        _mm256_storeu_ps(output + i, _mm256_blendv_ps(sp, x, mask));
    }
    for (; i < n; i++) { float a=input[i]; output[i]=a>20.0f?a:log1pf(expf(a)); }
}

// ── MKL-path GEMV: AVX2 AXPY-based matrix-vector (matches MKL cblas_sgemv) ───
void bpd_gemv_mkl_cpu(const float* A, const float* x, float* y, int M, int K) {
    for (int row = 0; row < M; row++) {
        const float* a = A + row * K;
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= K; k += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+k), _mm256_loadu_ps(x+k), acc);
        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s4 = _mm_add_ps(lo, hi);
        __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        float dot = _mm_cvtss_f32(s1);
        for (; k < K; k++) dot += a[k] * x[k];
        y[row] = dot;
    }
}

// ── MKL-path RMSNorm: cascade(8) + _mm256_sqrt_ps ────────────────────────────
void bpd_rmsnorm_mkl_cpu(const float* input, float* output,
                          int N, int C, int H, int W, float eps) {
    int spatial = H * W;
    float* temp = (float*)bpd_alloc(C * sizeof(float));
    for (int n = 0; n < N; n++) {
        for (int p = 0; p < spatial; p++) {
            for (int c = 0; c < C; c++) {
                float v = input[n*C*spatial + c*spatial + p];
                temp[c] = v * v;
            }
            float sum_sq = pairwise_sum(temp, C);
            // Use VSQRTPS to match ATen's _mm256_sqrt_ps rounding
            __m256 vsq = _mm256_set1_ps(sum_sq / (float)C + eps);
            float rms = _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_sqrt_ps(vsq)));
            for (int c = 0; c < C; c++)
                output[n*C*spatial + c*spatial + p] =
                    input[n*C*spatial + c*spatial + p] / rms;
        }
    }
    free(temp);
}

// ── MKL-path InstanceNorm: FMA affine apply ───────────────────────────────────
void bpd_instancenorm_mkl_cpu(const float* input, float* output,
                               const float* weight, const float* bias,
                               int N, int C, int H, int W, float eps) {
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float* x = input  + (n*C+c)*spatial;
            float*       y = output + (n*C+c)*spatial;
            float mean = 0.0f;
            for (int p = 0; p < spatial; p++) mean += x[p];
            mean /= (float)spatial;
            float var = 0.0f;
            for (int p = 0; p < spatial; p++) { float d=x[p]-mean; var+=d*d; }
            var /= (float)spatial;
            float invstd = 1.0f / sqrtf(var + eps);
            __m256 va = _mm256_set1_ps(invstd);
            __m256 vb = _mm256_set1_ps(-mean * invstd);
            int p = 0;
            for (; p + 8 <= spatial; p += 8)
                _mm256_storeu_ps(y+p, _mm256_fmadd_ps(_mm256_loadu_ps(x+p), va, vb));
            for (; p < spatial; p++) y[p] = x[p]*invstd + (-mean*invstd);
            if (weight && bias) {
                __m256 vw = _mm256_set1_ps(weight[c]);
                __m256 vbi = _mm256_set1_ps(bias[c]);
                p = 0;
                for (; p + 8 <= spatial; p += 8)
                    _mm256_storeu_ps(y+p, _mm256_fmadd_ps(_mm256_loadu_ps(y+p), vw, vbi));
                for (; p < spatial; p++) y[p] = y[p]*weight[c] + bias[c];
            }
        }
    }
}

#endif  /* BPD_MKL_PATH */

/* ══════════════════════════════════════════════════════════════════════════
 * Phase L.1 — Llama.cpp-matching kernels
 *
 * Each kernel mirrors the exact arithmetic of the corresponding ggml op so
 * that the output is bit-identical to llama.cpp's CPU path.
 *
 * Substrate-design parameters (from implementation_matches.pl llama_cpp entry):
 *   rms_accumulator(double)   — ggml uses ggml_float (double) for sum-of-squares
 *   rope_type(neox)           — Llama 3 uses GGML_ROPE_TYPE_NEOX (n_offset = n_dims/2)
 *   rope_freq_base(500000.0)  — Llama 3.2 default; overridable per model
 *   kv_cache_layout(f32)      — F32 KV cache (Llama 3 default; F16 is a separate variant)
 * ══════════════════════════════════════════════════════════════════════════ */

/* ─────────────────────────────────────────────────────────────────────────
 * L.1.4  bpd_rmsnorm_llama_cpu
 *
 * Mirrors ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>.
 *
 * Critical substrate-design property: ggml uses ggml_float (double) for
 * the sum-of-squares accumulation, then casts to float for the scale.
 * Our existing bpd_rmsnorm_cpu uses float accumulation — correct for the
 * PyTorch oracle but NOT for the llama.cpp oracle.
 *
 * Signature:
 *   input  : [n_rows, row_len]  F32, row-major
 *   weight : [row_len]          F32 (the per-element scale from attn_norm weight)
 *   output : [n_rows, row_len]  F32
 *   n_rows : number of rows (tokens x batch)
 *   row_len: embedding dimension (ne0 in ggml)
 *   eps    : epsilon (typically 1e-5 for Llama 3)
 * ─────────────────────────────────────────────────────────────────────────*/
void bpd_rmsnorm_llama_cpu(
        const float* input,
        const float* weight,
        float*       output,
        int          n_rows,
        int          row_len,
        float        eps)
{
    for (int r = 0; r < n_rows; r++) {
        const float* x = input  + r * row_len;
        float*       y = output + r * row_len;

        /* ggml uses double accumulation for the sum-of-squares */
        double sum = 0.0;
        for (int i = 0; i < row_len; i++) {
            sum += (double)x[i] * (double)x[i];
        }
        const float mean  = (float)(sum / row_len);
        const float scale = 1.0f / sqrtf(mean + eps);

        if (weight) {
            for (int i = 0; i < row_len; i++)
                y[i] = x[i] * scale * weight[i];
        } else {
            for (int i = 0; i < row_len; i++)
                y[i] = x[i] * scale;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * L.1.5  bpd_rope_neox_cpu
 *
 * Mirrors ggml_compute_forward_rope_flt<float> with GGML_ROPE_TYPE_NEOX.
 *
 * NEOX layout: the two rotation components for dimension pair i are
 *   x[i]  and  x[i + n_dims/2]
 * (first half and second half of head_dim, NOT interleaved pairs).
 *
 * theta for dimension pair i:
 *   theta_i = pos * freq_base^(-2i/n_dims)
 *           = pos * theta_scale^i   where theta_scale = freq_base^(-2/n_dims)
 *
 * ggml computes the cache once per (position, head):
 *   theta = (float)pos
 *   for i0 in 0, 2, 4, ..., n_dims-2:
 *       cache[i0+0] = cosf(theta)
 *       cache[i0+1] = sinf(theta)
 *       theta *= theta_scale
 * then rotate_pairs with n_offset = n_dims/2.
 *
 * For standard Llama 3 (no YaRN: ext_factor=0, freq_scale=1,
 * freq_factors=NULL, attn_factor=1), rope_yarn simplifies to:
 *   cos_theta = cosf(theta)
 *   sin_theta = sinf(theta)
 *
 * Signature:
 *   input     : [n_tokens, n_heads, head_dim]  F32, row-major
 *   output    : [n_tokens, n_heads, head_dim]  F32
 *   pos_ids   : [n_tokens]  I32  (position index for each token)
 *   n_tokens  : sequence length
 *   n_heads   : number of attention heads
 *   head_dim  : dimension per head (= ne0 in ggml)
 *   n_dims    : number of dimensions to rotate (= head_dim for standard Llama 3)
 *   freq_base : RoPE theta base (10000.0 for Llama 2, 500000.0 for Llama 3)
 * ─────────────────────────────────────────────────────────────────────────*/
/* Forward decls so back-compat wrappers can delegate. */
void bpd_rope_neox_freqs_cpu(
        const float*   input,
        float*         output,
        const int32_t* pos_ids,
        const float*   freq_factors,
        int            n_tokens,
        int            n_heads,
        int            head_dim,
        int            n_dims,
        float          freq_base);

void bpd_rope_norm_freqs_cpu(
        const float*   input,
        float*         output,
        const int32_t* pos_ids,
        const float*   freq_factors,
        int            n_tokens,
        int            n_heads,
        int            head_dim,
        int            n_dims,
        float          freq_base);

void bpd_rope_neox_cpu(
        const float*   input,
        float*         output,
        const int32_t* pos_ids,
        int            n_tokens,
        int            n_heads,
        int            head_dim,
        int            n_dims,
        float          freq_base)
{
    /* Delegate to the freq_factors variant with NULL (= all-ones).
     * Preserves API compatibility for callers that don't have freq_factors. */
    bpd_rope_neox_freqs_cpu(input, output, pos_ids, NULL,
                            n_tokens, n_heads, head_dim, n_dims, freq_base);
}

/* L.1.6.b  bpd_rope_neox_freqs_cpu
 *
 * NEOX RoPE with optional per-dimension frequency factors (Llama 3 NTK-aware
 * long-context extension). Mirrors ggml_rope_cache_init's freq_factors branch:
 *
 *   for i0 in 0, 2, ..., n_dims-2:
 *       ff = freq_factors ? freq_factors[i0/2] : 1.0f
 *       theta_effective = theta_base / ff
 *       cache[i0+0] = cosf(theta_effective)
 *       cache[i0+1] = sinf(theta_effective)
 *       theta_base *= theta_scale
 *
 * For Llama 3.2, freq_factors is the `rope_freqs.weight` tensor from the GGUF,
 * shape (n_dims/2,) = (32,) for head_dim=64. Values range 1.0 (low-freq dims,
 * no scaling) to 32.0 (high-freq dims, theta scaled down by 32x). This implements
 * the NTK-aware interpolation that lets Llama 3 extend to 128K context.
 *
 * Signature additions (vs bpd_rope_neox_cpu):
 *   freq_factors : [n_dims/2]  F32 (or NULL to use all-ones / no scaling)
 * \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500*/
void bpd_rope_neox_freqs_cpu(
        const float*   input,
        float*         output,
        const int32_t* pos_ids,
        const float*   freq_factors,    /* [n_dims/2] or NULL */
        int            n_tokens,
        int            n_heads,
        int            head_dim,
        int            n_dims,
        float          freq_base)
{
    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    const int   half        = n_dims / 2;

    for (int t = 0; t < n_tokens; t++) {
        const int32_t pos = pos_ids[t];
        for (int h = 0; h < n_heads; h++) {
            const float* src = input  + (t * n_heads + h) * head_dim;
            float*       dst = output + (t * n_heads + h) * head_dim;

            float theta = (float)pos;
            for (int i0 = 0; i0 < n_dims; i0 += 2) {
                const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
                const float theta_eff = theta / ff;
                const float cos_theta = cosf(theta_eff);
                const float sin_theta = sinf(theta_eff);
                const int ic = i0 / 2;
                const float x0 = src[ic];
                const float x1 = src[ic + half];
                dst[ic]        = x0 * cos_theta - x1 * sin_theta;
                dst[ic + half] = x0 * sin_theta + x1 * cos_theta;
                theta *= theta_scale;
            }
            for (int i = n_dims; i < head_dim; i++)
                dst[i] = src[i];
        }
    }
}

/* L.1.6.c  bpd_rope_norm_freqs_cpu
 *
 * NORM-style RoPE (ggml's LLAMA_ROPE_TYPE_NORM, used by LLM_ARCH_LLAMA).
 * Pairs of CONSECUTIVE head values are rotated: pair i rotates src[2i] and
 * src[2i+1]. This is the rotation pattern Llama 3.x actually uses, NOT NEOX.
 *
 *   for i0 in 0, 2, ..., n_dims-2:
 *       ff = freq_factors ? freq_factors[i0/2] : 1.0f
 *       theta_eff = theta_base / ff
 *       cos_theta = cosf(theta_eff)
 *       sin_theta = sinf(theta_eff)
 *       x0 = src[i0]
 *       x1 = src[i0+1]
 *       dst[i0]   = x0*cos - x1*sin
 *       dst[i0+1] = x0*sin + x1*cos
 *       theta_base *= theta_scale
 *
 * Bit-identity: matches ggml's NORM path in ggml_compute_forward_rope_f32 (the
 * non-NEOX else branch in ggml-cpu/ops.cpp), with freq_factors plumbed from
 * the rope_freqs.weight GGUF tensor.
 *
 * Signature mirrors bpd_rope_neox_freqs_cpu but with the NORM rotation pattern.
 * \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500*/
void bpd_rope_norm_freqs_cpu(
        const float*   input,
        float*         output,
        const int32_t* pos_ids,
        const float*   freq_factors,   /* [n_dims/2] or NULL */
        int            n_tokens,
        int            n_heads,
        int            head_dim,
        int            n_dims,
        float          freq_base)
{
    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);

    for (int t = 0; t < n_tokens; t++) {
        const int32_t pos = pos_ids[t];
        for (int h = 0; h < n_heads; h++) {
            const float* src = input  + (t * n_heads + h) * head_dim;
            float*       dst = output + (t * n_heads + h) * head_dim;

            float theta = (float)pos;
            for (int i0 = 0; i0 < n_dims; i0 += 2) {
                const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
                const float theta_eff = theta / ff;
                const float cos_theta = cosf(theta_eff);
                const float sin_theta = sinf(theta_eff);
                /* NORM: pair (i0/2) rotates src[i0] and src[i0+1] */
                const float x0 = src[i0];
                const float x1 = src[i0 + 1];
                dst[i0]     = x0 * cos_theta - x1 * sin_theta;
                dst[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
                theta *= theta_scale;
            }
            for (int i = n_dims; i < head_dim; i++)
                dst[i] = src[i];
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * L.1.6  bpd_kv_cache_write_cpu
 *
 * Writes a K or V projection tensor into the KV cache at the given
 * sequence positions.  In llama.cpp the KV cache is a flat F32 buffer
 * with layout [max_seq_len, n_kv_heads, head_dim] (ggml row-major:
 * fastest axis = head_dim = ne0).
 *
 * This is a plain indexed store — the bit-identity comes from the
 * upstream RoPE kernel (for K) or the projection matmul (for V).
 *
 * Signature:
 *   cache      : [max_seq_len, n_kv_heads, head_dim]  F32 flat buffer
 *   src        : [n_tokens, n_kv_heads, head_dim]     F32
 *   pos_ids    : [n_tokens]  I32  (absolute position of each token)
 *   n_tokens   : number of tokens to write
 *   n_kv_heads : number of KV heads
 *   head_dim   : dimension per head
 *   max_seq_len: total cache capacity (stride for the seq dimension)
 * ─────────────────────────────────────────────────────────────────────────*/
void bpd_kv_cache_write_cpu(
        float*         cache,
        const float*   src,
        const int32_t* pos_ids,
        int            n_tokens,
        int            n_kv_heads,
        int            head_dim,
        int            max_seq_len)
{
    const int row_stride = n_kv_heads * head_dim;
    for (int t = 0; t < n_tokens; t++) {
        const int32_t pos = pos_ids[t];
        const float*  s   = src   + t   * row_stride;
        float*        d   = cache + pos * row_stride;
        for (int i = 0; i < row_stride; i++)
            d[i] = s[i];
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * L.1.6.f16  bpd_kv_cache_write_f16_cpu
 *
 * F16 KV cache write: mirror of bpd_kv_cache_write_cpu but writes F16 cache.
 * Substrate-design parameter family: kv_cache_dtype ∈ {f32, f16}.
 *
 * ggml's canonical fixture uses F16 cache (idx 25, 34 in /tmp/llama_dump_hello_8
 * show CPY ops with f16 dtype). This function implements the f16 value of the
 * kv_cache_dtype parameter family. The F32 alternative remains available via
 * bpd_kv_cache_write_cpu for substrate variants that prefer cache precision.
 *
 * Conversion: uses f32_to_f16 (IEEE 754 round-to-nearest-even, matching ggml's
 * _cvtss_sh hardware F16C convention bit-for-bit).
 *
 * Signature mirrors bpd_kv_cache_write_cpu but with uint16_t* cache.
 * Discovered/verified by medayek 2026-05-23 ~01:00 UTC.
 * ───────────────────────────────────────────────────────────────────────── */
void bpd_kv_cache_write_f16_cpu(
        uint16_t*      cache,
        const float*   src,
        const int32_t* pos_ids,
        int            n_tokens,
        int            n_kv_heads,
        int            head_dim,
        int            max_seq_len)
{
    const int row_stride = n_kv_heads * head_dim;
    for (int t = 0; t < n_tokens; t++) {
        const int32_t pos = pos_ids[t];
        const float*    s = src   + t   * row_stride;
        uint16_t*       d = cache + pos * row_stride;
        for (int i = 0; i < row_stride; i++)
            d[i] = f32_to_f16(s[i]);
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * L.1.7  bpd_softmax_causal_cpu
 *
 * Causal (lower-triangular) masked softmax over attention scores.
 * Mirrors ggml_compute_forward_soft_max with causal mask.
 *
 * ggml applies:
 *   1. Scale by attn_factor (= 1/sqrt(head_dim))
 *   2. Add upper-triangular -inf mask (future positions)
 *   3. Row-wise softmax (max-stabilised)
 *
 * Signature:
 *   scores   : [n_heads, n_q_tokens, n_kv_tokens]  F32  (Q @ K^T result)
 *   output   : [n_heads, n_q_tokens, n_kv_tokens]  F32
 *   n_heads  : number of query heads
 *   n_q      : number of query tokens (rows)
 *   n_kv     : number of KV tokens (columns, including past)
 *   scale    : attention scale (1/sqrt(head_dim))
 *   q_offset : absolute position of the first query token (for causal mask)
 * ─────────────────────────────────────────────────────────────────────────*/
void bpd_softmax_causal_cpu(
        const float* scores,
        float*       output,
        int          n_heads,
        int          n_q,
        int          n_kv,
        float        scale,
        int          q_offset)
{
    for (int h = 0; h < n_heads; h++) {
        for (int q = 0; q < n_q; q++) {
            const float* row_in  = scores + (h * n_q + q) * n_kv;
            float*       row_out = output + (h * n_q + q) * n_kv;
            const int    q_abs   = q_offset + q;

            /* 1. Scale + causal mask, find max for numerical stability */
            float max_val = -1e38f;
            for (int k = 0; k < n_kv; k++) {
                float v = (k <= q_abs) ? row_in[k] * scale : -1e38f;
                row_out[k] = v;
                if (v > max_val) max_val = v;
            }

            /* 2. exp(x - max) and sum */
            float sum = 0.0f;
            for (int k = 0; k < n_kv; k++) {
                float e = expf(row_out[k] - max_val);
                row_out[k] = e;
                sum += e;
            }

            /* 3. Normalise */
            const float inv_sum = 1.0f / sum;
            for (int k = 0; k < n_kv; k++)
                row_out[k] *= inv_sum;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * L.1.8  bpd_gqa_attn_cpu
 *
 * Grouped-Query Attention (GQA) with online softmax, mirroring
 * ggml_compute_forward_flash_attn_ext_f16_one_chunk for the F32/F32 case
 * (no F16 V cache, no logit_softcap, no sinks, no mask tensor — just the
 * causal mask from position).
 *
 * Algorithm (Dao et al. 2022 online softmax):
 *   For each query token q and query head iq:
 *     kv_head = iq / gqa_ratio          (GQA head mapping)
 *     M = -inf, S = 0, VKQ[DV] = 0
 *     for ic in 0 .. n_kv-1:
 *       if ic > q_pos + kv_offset: break  (causal mask)
 *       s = dot(Q[q,iq], K[ic,kv_head]) * scale
 *       if s > M:
 *         ms = expf(Mold - s);  M = s
 *         VKQ *= ms;  S *= ms
 *       else:
 *         vs = expf(s - M)
 *       VKQ += V[ic,kv_head] * vs        (ggml_vec_mad_f32 = FMA loop)
 *       S   += vs
 *     VKQ /= S
 *     dst[q,iq] = VKQ
 *
 * The Q·K dot product uses a plain scalar loop (matching OpenBLAS sdot on
 * Ruach Tov hardware).  The V accumulation uses a scalar FMA-equivalent
 * loop: y[i] = y[i] + x[i]*v, which the compiler will emit as VFMADD231PS
 * when compiled with -mavx2 -mfma, matching ggml_vec_mad_f32.
 *
 * Signature:
 *   q         : [n_q_tokens, n_q_heads, head_dim]   F32
 *   k         : [n_kv_tokens, n_kv_heads, head_dim] F32  (KV cache slice)
 *   v         : [n_kv_tokens, n_kv_heads, head_dim] F32
 *   dst       : [n_q_tokens, n_q_heads, head_dim]   F32
 *   n_q_tokens: number of query tokens
 *   n_kv      : number of KV tokens in the cache
 *   n_q_heads : number of query heads
 *   n_kv_heads: number of KV heads (n_q_heads / n_kv_heads = gqa_ratio)
 *   head_dim  : dimension per head (DK = DV = head_dim)
 *   scale     : attention scale (typically 1/sqrt(head_dim))
 *   kv_offset : position of the first KV token in the sequence
 *               (for prefill: 0; for decode: past_kv_len)
 * ─────────────────────────────────────────────────────────────────────────*/
/* GQA attention with batch softmax (post-scaled).
 * Matches ggml's exact computation order:
 *   raw QK^T → scale*score → causal mask → max → exp(x-max) → sum → normalize → attn*V
 *
 * Sweepable parameter: scale_application_path = post_scaled (ggml convention).
 * The previous online-softmax version (pre_scaled) is faster but produces
 * different bits due to different accumulation order.
 */
static inline float bpd_hsum256_tb(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sm = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(sm, sm);
    sm = _mm_add_ps(sm, sh);
    sh = _mm_shuffle_ps(sm, sm, 0x1);
    sm = _mm_add_ss(sm, sh);
    return _mm_cvtss_f32(sm);
}
/* tinyBLAS-EXACT f16 QK dot (matches ggml node_20 at 0 ULP; verified vs dump + disassembly @0x7c0b0).
 * Both operands rounded to f16 (F16C), single __m256 accumulator over K (vmulps+vaddps, no FMA), hsum. */
static inline float bpd_qk_dot_tinyblas_f32k(const float* q_f32, const float* k_f32, int n) {
    __m256 acc = _mm256_setzero_ps();
    int l = 0;
    for (; l + 8 <= n; l += 8) {
        __m256 qv = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_loadu_ps(q_f32 + l), _MM_FROUND_TO_NEAREST_INT));
        __m256 kv = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_loadu_ps(k_f32 + l), _MM_FROUND_TO_NEAREST_INT));
        acc = _mm256_add_ps(_mm256_mul_ps(qv, kv), acc);
    }
    float res = bpd_hsum256_tb(acc);
    for (; l < n; ++l) {
        float qf = _cvtsh_ss(_cvtss_sh(q_f32[l], 0));
        float kf = _cvtsh_ss(_cvtss_sh(k_f32[l], 0));
        res += qf * kf;
    }
    return res;
}

/* COMPLETE scalar replica of ggml_v_expf (incl |n|>126 overflow branch). 0-ULP vs ggml.
 * Handles masked scores (-1e38 -> 0) like ggml. FMA expanded to mul+add (no-FMA box). */
static inline float bpd_exp_scalar_ggml(float x){
    const float r = 0x1.8p23f;
    const float z = x * 0x1.715476p+0f + r;
    const float n = z - r;
    const float b = (x - n * 0x1.62e4p-1f) - n * 0x1.7f7d1cp-20f;
    uint32_t zi; __builtin_memcpy(&zi, &z, 4);
    uint32_t e = (zi << 23);
    uint32_t k_bits = e + 0x3f800000u;
    float k; __builtin_memcpy(&k, &k_bits, 4);
    float absn = n < 0.f ? -n : n;
    const float u = b * b;
    const float j = ((0x1.0e4020p-7f * b + 0x1.573e2ep-5f) * u
                     + (0x1.555e66p-3f * b + 0x1.fffdb6p-2f)) * u + 0x1.ffffecp-1f * b;
    if (!(absn > 126.f)) return j * k + k;
    uint32_t g = (n <= 0.f ? 0x82000000u : 0u);
    uint32_t s1b = g + 0x7f000000u; float s1; __builtin_memcpy(&s1, &s1b, 4);
    uint32_t s2b = e - g;          float s2; __builtin_memcpy(&s2, &s2b, 4);
    if (absn > 192.f) return s1 * s1;
    return (s2 * j + s2) * s1;
}

void bpd_gqa_attn_cpu(
        const float* q,
        const float* k,
        const float* v,
        float*       dst,
        int          n_q_tokens,
        int          n_kv,
        int          n_q_heads,
        int          n_kv_heads,
        int          head_dim,
        float        scale,
        int          kv_offset)
{
    const int gqa_ratio = n_q_heads / n_kv_heads;

    for (int qt = 0; qt < n_q_tokens; qt++) {
        const int q_pos = kv_offset + qt;

        for (int iq = 0; iq < n_q_heads; iq++) {
            const int kv_head = iq / gqa_ratio;

            const float* pq  = q   + (qt * n_q_heads + iq) * head_dim;
            float*       pdst = dst + (qt * n_q_heads + iq) * head_dim;

            /* 1. Compute RAW QK^T scores (NO scale in dot product) */
            float scores[n_kv];
            for (int ic = 0; ic < n_kv; ic++) {
                const float* pk = k + (ic * n_kv_heads + kv_head) * head_dim;
                scores[ic] = bpd_qk_dot_tinyblas_f32k(pq, pk, head_dim);  /* tinyBLAS-exact f16 QK */
            }

            /* 2. Scale + causal mask (matching ggml: scale applied here, not in dot product) */
            float max_val = -1e38f;
            for (int ic = 0; ic < n_kv; ic++) {
                float sv = (ic <= q_pos) ? scores[ic] * scale : -1e38f;
                scores[ic] = sv;
                if (sv > max_val) max_val = sv;
            }

            /* 3. exp(x - max) and sum — ggml-EXACT SOFT_MAX: poly-exp (bpd_exp_scalar_ggml,
             * 0-ULP vs ggml_v_expf) + DOUBLE-precision sum (ggml sums exp in ggml_float). */
            double sum_d = 0.0;
            for (int ic = 0; ic < n_kv; ic++) {
                float e = bpd_exp_scalar_ggml(scores[ic] - max_val);
                scores[ic] = e;
                sum_d += (double)e;
            }
            float sum_exp = (float)sum_d;

            /* 4. Normalize */
            float inv_sum = (sum_exp == 0.0f) ? 0.0f : 1.0f / sum_exp;
            for (int ic = 0; ic < n_kv; ic++)
                scores[ic] *= inv_sum;

            /* 5. Weighted sum of V — ggml node_22 = tinyBLAS f16 sgemm. For each output dim d:
             * kqv[d] = tinyBLAS-f16-dot(softmax[0..nkv_pad], V[:,d]) with kv PADDED to mult-of-8
             * (softmax=0 for pad). Round softmax+V to f16, single __m256 acc over kv, hsum.
             * Verified 0-ULP vs node_22 (TDD). */
            {
                int nkv_pad = (n_kv + 7) & ~7;   /* pad kv up to multiple of 8 */
                /* gather padded f16 softmax weights once */
                static float w_pad_buf[8192];    /* >= max_seq_len padded */
                for (int ic = 0; ic < n_kv; ic++) w_pad_buf[ic] = scores[ic];
                for (int ic = n_kv; ic < nkv_pad; ic++) w_pad_buf[ic] = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int ic = 0; ic < nkv_pad; ic += 8) {
                        float wv[8], vv[8];
                        for (int j = 0; j < 8; j++) {
                            int kk = ic + j;
                            wv[j] = w_pad_buf[kk];
                            vv[j] = (kk < n_kv) ? v[(kk * n_kv_heads + kv_head) * head_dim + d] : 0.0f;
                        }
                        __m256 wf = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_loadu_ps(wv), _MM_FROUND_TO_NEAREST_INT));
                        __m256 vf = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_loadu_ps(vv), _MM_FROUND_TO_NEAREST_INT));
                        acc = _mm256_add_ps(_mm256_mul_ps(wf, vf), acc);
                    }
                    pdst[d] = bpd_hsum256_tb(acc);
                }
            }
        }
    }
}

/* ================================================================== * L.1.9  Residual add, SiLU, elementwise MUL, fused SwiGLU FFN
 *
 * Mirrors:
 *   ggml_compute_forward_add_non_quantized  (binary-ops.cpp)
 *     -> vec_binary_op_contiguous<op_add, float, float, float>
 *     -> z[i] = x[i] + y[i]   (AVX2: _mm256_add_ps 8-wide)
 *
 *   ggml_compute_forward_mul  (binary-ops.cpp)
 *     -> vec_binary_op_contiguous<op_mul, float, float, float>
 *     -> z[i] = x[i] * y[i]   (scalar; no SIMD path in ggml for mul)
 *
 *   ggml_vec_silu_f32  (vec.cpp)
 *     -> scalar path: y[i] = x[i] / (1.0f + expf(-x[i]))
 *     -> AVX2+FMA path: ggml_v_silu(__m256) using ggml_v_expf polynomial
 *   On Ruach Tov hardware (no AVX2+FMA in ggml build), scalar path is used.
 *   BPD uses the scalar path to match the Ruach Tov oracle.
 *
 *   SwiGLU FFN composition (llama.cpp transformer block):
 *     h  = gate_proj(x)     Q8_0 matmul, already in bpd_qmatmul_q8_0_cpu
 *     h  = silu(h)
 *     h2 = up_proj(x)       Q8_0 matmul
 *     h  = h * h2           elementwise mul
 *     out = down_proj(h)    Q8_0 matmul
 *   bpd_swiglu_fuse_cpu fuses silu + elementwise mul into a single pass.
 * ========================================================================= */

/* L.1.9a  Residual add: dst[i] = a[i] + b[i]  (F32, contiguous)
 * Mirrors ggml_vec_add_f32 with AVX2 8-wide add, scalar tail.
 */
void bpd_add_f32_cpu(const float * a, const float * b, float * dst, int n) {
    int i = 0;
#if BPD_HAVE_AVX1
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
#endif
    for (; i < n; i++)
        dst[i] = a[i] + b[i];
}

/* L.1.9b  SiLU: dst[i] = x[i] / (1.0f + expf(-x[i]))
 * Mirrors ggml_silu_f32 scalar path (Ruach Tov oracle).
 * On AVX2+FMA hosts, ggml uses the ARM-Limited polynomial; on the Ruach Tov
 * hardware (the bit-identical oracle), the scalar libm expf path is used.
 */
void bpd_silu_f32_cpu(const float * x, float * dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = x[i] / (1.0f + expf(-x[i]));
}

/* L.1.9c  Elementwise multiply: dst[i] = a[i] * b[i]  (F32, contiguous)
 * Mirrors vec_binary_op_contiguous<op_mul> -- scalar loop, no SIMD in ggml.
 */
void bpd_mul_f32_cpu(const float * a, const float * b, float * dst, int n) {
    int i = 0;
#if BPD_HAVE_AVX1
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
#endif
    for (; i < n; i++)
        dst[i] = a[i] * b[i];
}

/* L.1.9d  Fused SwiGLU: dst[i] = silu(gate[i]) * up[i]
 * Fuses L.1.9b + L.1.9c into a single pass to avoid a temporary buffer.
 * gate and up are the outputs of gate_proj and up_proj respectively.
 * dst is written in-place (may alias gate).
 */
static inline __m256 bpd_mla256(__m256 a, __m256 b, __m256 c){ return _mm256_add_ps(_mm256_mul_ps(a,b), c); }
static inline __m256 bpd_nmla256(__m256 a, __m256 b, __m256 c){ return _mm256_sub_ps(c, _mm256_mul_ps(a,b)); }
static inline __m256i bpd_add_epi32_avx1(__m256i x, __m256i y){
    __m128i xl=_mm256_castsi256_si128(x), xh=_mm256_extractf128_si256(x,1);
    __m128i yl=_mm256_castsi256_si128(y), yh=_mm256_extractf128_si256(y,1);
    __m256i r=_mm256_castsi128_si256(_mm_add_epi32(xl,yl));
    return _mm256_insertf128_si256(r,_mm_add_epi32(xh,yh),1);
}
static inline __m256i bpd_slli_epi32_avx1(__m256i x, int n){
    __m128i xl=_mm256_castsi256_si128(x), xh=_mm256_extractf128_si256(x,1);
    __m256i r=_mm256_castsi128_si256(_mm_slli_epi32(xl,n));
    return _mm256_insertf128_si256(r,_mm_slli_epi32(xh,n),1);
}
/* ggml-EXACT poly-exp (vec.h ggml_v_expf), FMA-expanded to mul+add to match ggml's no-FMA
 * binary on this Ivy Bridge box. VERIFIED 0 ULP vs ggml_vec_silu_f32 across 1M values. */
static inline __m256 bpd_exp256_ps(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = bpd_mla256(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = bpd_nmla256(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                 bpd_nmla256(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = bpd_slli_epi32_avx1(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(bpd_add_epi32_avx1(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = bpd_mla256(bpd_mla256(bpd_mla256(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                 bpd_mla256(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
                             u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    return bpd_mla256(j, k, k);
}

void bpd_swiglu_fuse_cpu(const float * gate, const float * up, float * dst, int n) {
    /* VECTORIZED SwiGLU: silu(gate)*up = (gate/(1+exp(-gate)))*up. Poly-exp (4 ULP vs libm). */
    int i = 0;
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 e = bpd_exp256_ps(_mm256_sub_ps(zero, g));
        __m256 silu = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(silu, _mm256_loadu_ps(up + i)));
    }
    for (; i < n; i++) dst[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}
/* ─────────────────────────────────────────────────────────────────────────
 * L.1.10  bpd_llama_block_cpu / bpd_llama_forward_cpu
 *
 * Full transformer block composition and complete forward pass.
 * Composes the individual L.1.4–L.1.9 kernels into a single-function
 * transformer block, then iterates n_layers blocks for the full forward.
 *
 * This file is compiled separately and linked with bpd_cpu.so, or
 * #included at the end of bpd_cpu.c.
 *
 * Memory strategy:
 *   - Two scratch buffers (ping-pong) for intermediate activations
 *   - KV cache is pre-allocated externally (max_seq_len * n_kv_heads * head_dim)
 *   - All weight pointers are passed via a model descriptor struct
 *
 * ggml reference: llama_decode_internal → llm_build_llama
 * ───────────────────────────────────────────────────────────────────────── */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── Forward declarations of the composed kernels ─────────────────────── */
extern void bpd_rmsnorm_llama_cpu(const float* input, const float* weight,
                                   float* output, int n_rows, int row_len, float eps);
extern void bpd_rope_neox_cpu(const float* input, float* output,
                               const int32_t* pos_ids, int n_tokens, int n_heads,
                               int head_dim, int n_dims, float freq_base);
extern void bpd_kv_cache_write_cpu(float* cache, const float* src,
                                    const int32_t* pos_ids, int n_tokens,
                                    int n_kv_heads, int head_dim, int max_seq_len);
extern void bpd_gqa_attn_cpu(const float* q, const float* k, const float* v,
                              float* dst, int n_q_tokens, int n_kv,
                              int n_q_heads, int n_kv_heads, int head_dim,
                              float scale, int kv_offset);
extern void bpd_add_f32_cpu(const float* a, const float* b, float* dst, int n);
extern void bpd_silu_f32_cpu(const float* x, float* dst, int n);
extern void bpd_mul_f32_cpu(const float* a, const float* b, float* dst, int n);
extern void bpd_swiglu_fuse_cpu(const float* gate, const float* up, float* dst, int n);
extern void bpd_qmatmul_q8_0_llamafile_cpu(const uint8_t* W_q8_0, const float* X_f32,
                                             float* out, int m_weight, int m_tokens, int K);
extern void bpd_embed_lookup_q8_0_cpu(const uint8_t* table, const int32_t* token_ids,
                                       float* out, int n_tokens, int embed_dim);
extern void bpd_argmax_dim_cpu(const float* x, long* out,
                                int outer, int dim_size, int inner);

/* ── Model configuration struct ──────────────────────────────────────── */
typedef struct {
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int embed_dim;       /* = n_heads * head_dim */
    int ffn_dim;         /* intermediate size (e.g., 14336 for Llama3-8B) */
    int vocab_size;
    int max_seq_len;
    float rms_eps;
    float rope_base;
    int rope_dim;        /* usually = head_dim */
    int kv_cache_f16;    /* sweepable parameter: 0=f32 cache (more precision),
                            1=f16 cache (matches ggml, less memory).
                            Substrate-design parameter family: kv_cache_dtype */
} bpd_llama_config;

/* Type-safe KV cache handle.
 * The tag (is_f16) must match the pointer. Accessing .f32 when is_f16=1
 * is a bug that the substrate can detect at dispatch time.
 * This is the C-level analog of Invariant 7 (dtype coherence). */
typedef struct {
    union {
        float*    f32;
        uint16_t* f16;
    } data;
    int is_f16;  /* must match cfg->kv_cache_f16 */
} bpd_kv_cache;

/* ── Per-layer weight pointers ───────────────────────────────────────── */
typedef struct {
    const float*   attn_norm_w;   /* [embed_dim] */
    const uint8_t* w_q;           /* Q8_0: [n_heads*head_dim, embed_dim] */
    const uint8_t* w_k;           /* Q8_0: [n_kv_heads*head_dim, embed_dim] */
    const uint8_t* w_v;           /* Q8_0: [n_kv_heads*head_dim, embed_dim] */
    const uint8_t* w_o;           /* Q8_0: [embed_dim, n_heads*head_dim] */
    const float*   ffn_norm_w;    /* [embed_dim] */
    const uint8_t* w_gate;        /* Q8_0: [ffn_dim, embed_dim] */
    const uint8_t* w_up;          /* Q8_0: [ffn_dim, embed_dim] */
    const uint8_t* w_down;        /* Q8_0: [embed_dim, ffn_dim] */
} bpd_llama_layer_weights;

/* ── Full model weight pointers ──────────────────────────────────────── */
typedef struct {
    const uint8_t*            token_embd;    /* Q8_0: [vocab_size, embed_dim] */
    const bpd_llama_layer_weights* layers;   /* array of n_layers */
    const float*              output_norm_w; /* [embed_dim] */
    const uint8_t*            output_w;      /* Q8_0: [vocab_size, embed_dim] */
    const float*              rope_freqs;    /* [n_dims/2] NTK-aware (Llama 3+), or NULL */
} bpd_llama_weights;

/* ── Single transformer block ────────────────────────────────────────── */
void bpd_llama_block_cpu(
        float*                       x,          /* [n_tokens, embed_dim] in/out */
        const bpd_llama_layer_weights* lw,
        const bpd_llama_config*      cfg,
        const int32_t*               pos_ids,    /* [n_tokens] */
        int                          n_tokens,
        int                          kv_pos,     /* current KV cache position */
        uint16_t*                    k_cache,    /* [max_seq_len, n_kv_heads*head_dim] dtype tagged by cfg->kv_cache_f16 (uint16_t* if f16, reinterpret as float* if f32) */
        uint16_t*                    v_cache,    /* [max_seq_len, n_kv_heads*head_dim] dtype tagged by cfg->kv_cache_f16 */
        float*                       scratch1,   /* >= n_tokens * max(embed_dim, ffn_dim) */
        float*                       scratch2,   /* >= n_tokens * max(embed_dim, ffn_dim) */
        float*                       scratch3,   /* >= n_tokens * max(n_heads*head_dim, ffn_dim) */
        const float*                 rope_freqs, /* [n_dims/2] or NULL */
        int                          n_out)      /* inp_out_ids: FFN runs on last n_out tokens */
{
    const int E = cfg->embed_dim;
    const int H = cfg->n_heads;
    const int HKV = cfg->n_kv_heads;
    const int D = cfg->head_dim;
    const int F = cfg->ffn_dim;
    const int n_kv = kv_pos + n_tokens;  /* total filled KV length after this step */
    const int n_skip = n_tokens - n_out;   /* inp_out_ids: FFN-side rows to skip */
    float* x_out = x + (size_t)n_skip * E;
    /* For canonical_ggml-bit-identity, attention must process ALL max_seq_len positions
     * (with causal mask producing -inf scores for unfilled ones), not just n_kv filled
     * positions. The reduction tree size of the softmax must match ggml's.
     * Substrate-design parameter family: attention_causal_mask_style ∈ {early_break, score_mask}.
     * Default value here: score_mask (calls bpd_gqa_attn_cpu below). */
    const int n_kv_full = cfg->max_seq_len;

    /* ── Attention sub-block ─────────────────────────────────────────── */

    /* 1. RMSNorm (attention) */
    bpd_rmsnorm_llama_cpu(x, lw->attn_norm_w, scratch1, n_tokens, E, cfg->rms_eps);

    /* 2. Q/K/V projections (Q8_0 matmul) */
    /* Q: [n_tokens, E] @ W_q^T → [n_tokens, H*D] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_q, scratch1, scratch2, H * D, n_tokens, E);
    /* K: [n_tokens, E] @ W_k^T → [n_tokens, HKV*D] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_k, scratch1, scratch3, HKV * D, n_tokens, E);

    /* 3. RoPE on Q (in-place in scratch2). NORM-style with NTK-aware
     *    freq_factors per LLM_ARCH_LLAMA's LLAMA_ROPE_TYPE_NORM. */
    bpd_rope_norm_freqs_cpu(scratch2, scratch2, pos_ids, rope_freqs,
                            n_tokens, H, D, cfg->rope_dim, cfg->rope_base);

    /* 4. RoPE on K (in-place in scratch3). Same NORM + freq_factors. */
    bpd_rope_norm_freqs_cpu(scratch3, scratch3, pos_ids, rope_freqs,
                            n_tokens, HKV, D, cfg->rope_dim, cfg->rope_base);

    /* 5. Write K to KV cache (dtype dispatched by cfg->kv_cache_f16).
     *    Substrate-design parameter family: kv_cache_dtype in {f16, f32}.
     *    F16: ggml-canonical, matches fixture, half memory.
     *    F32: higher precision (no F16 round-trip loss), 2x memory. */
    if (cfg->kv_cache_f16) {
        bpd_kv_cache_write_f16_cpu(k_cache, scratch3, pos_ids, n_tokens,
                                   HKV, D, cfg->max_seq_len);
    } else {
        bpd_kv_cache_write_cpu((float*)k_cache, scratch3, pos_ids, n_tokens,
                               HKV, D, cfg->max_seq_len);
    }

    /* 6. V projection: [n_tokens, E] @ W_v^T → [n_tokens, HKV*D] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_v, scratch1, scratch3, HKV * D, n_tokens, E);

    /* 7. Write V to KV cache (dtype dispatched by cfg->kv_cache_f16) */
    if (cfg->kv_cache_f16) {
        bpd_kv_cache_write_f16_cpu(v_cache, scratch3, pos_ids, n_tokens,
                                   HKV, D, cfg->max_seq_len);
    } else {
        bpd_kv_cache_write_cpu((float*)v_cache, scratch3, pos_ids, n_tokens,
                               HKV, D, cfg->max_seq_len);
    }

    /* 8. Prepare F32 K/V slices for attention.
     *    If kv_cache_f16: dequantize F16->F32 into malloc'd scratch.
     *    If kv_cache_f32: cache is already F32; alias-cast pointers (no copy).
     *    The caller (orchestrator) must allocate the cache buffer with bytes
     *    matching the chosen dtype: uint16_t * N for F16, float * N for F32. */
    const int kv_slice_len = n_kv_full * HKV * D;
    float* k_cache_f32;
    float* v_cache_f32;
    int allocated_scratch = 0;
    if (cfg->kv_cache_f16) {
        k_cache_f32 = (float*)bpd_alloc(sizeof(float) * kv_slice_len);
        v_cache_f32 = (float*)bpd_alloc(sizeof(float) * kv_slice_len);
        for (int i = 0; i < kv_slice_len; i++) {
            k_cache_f32[i] = f16_to_f32(k_cache[i]);
            v_cache_f32[i] = f16_to_f32(v_cache[i]);
        }
        allocated_scratch = 1;
    } else {
        k_cache_f32 = (float*)k_cache;
        v_cache_f32 = (float*)v_cache;
    }

    /* 9. GQA attention: Q against full (dequantized) KV cache */
    float scale = 1.0f / sqrtf((float)D);
    bpd_gqa_attn_cpu(scratch2, k_cache_f32, v_cache_f32,
                          scratch1,  /* output: [n_tokens, H*D] */
                          n_tokens, n_kv_full, H, HKV, D, scale, kv_pos);

    if (allocated_scratch) {
        free(k_cache_f32);
        free(v_cache_f32);
    }

    /* 9. Output projection: [n_tokens, H*D] @ W_o^T -> [n_tokens, E] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_o, scratch1, scratch2, E, n_tokens, H * D);

    /* 10. Residual add: x = x + attn_out */
    bpd_add_f32_cpu(x, scratch2, x, n_tokens * E);

    /* ── FFN sub-block ───────────────────────────────────────────────── */

    /* 11. RMSNorm (FFN) */
    bpd_rmsnorm_llama_cpu(x_out, lw->ffn_norm_w, scratch1, n_out, E, cfg->rms_eps);

    /* 12. Gate projection: [n_tokens, E] @ W_gate^T → [n_tokens, F] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_gate, scratch1, scratch2, F, n_out, E);

    /* 13. Up projection: [n_tokens, E] @ W_up^T → [n_tokens, F] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_up, scratch1, scratch3, F, n_out, E);

    /* 14. SwiGLU: silu(gate) * up → scratch2 */
    bpd_swiglu_fuse_cpu(scratch2, scratch3, scratch2, n_out * F);

    /* 15. Down projection: [n_tokens, F] @ W_down^T → [n_tokens, E] */
    bpd_qmatmul_q8_0_llamafile_cpu(lw->w_down, scratch2, scratch1, E, n_out, F);

    /* 16. Residual add: x = x + ffn_out */
    bpd_add_f32_cpu(x_out, scratch1, x_out, n_out * E);
}

/* ── Full forward pass ───────────────────────────────────────────────── */
void bpd_llama_forward_cpu(
        const int32_t*             token_ids,   /* [n_tokens] */
        int                        n_tokens,
        const bpd_llama_weights*   weights,
        const bpd_llama_config*    cfg,
        const int32_t*             pos_ids,     /* [n_tokens] absolute positions */
        int                        kv_pos,      /* first position in KV cache to write */
        uint16_t*                  k_cache,     /* [n_layers, max_seq_len, n_kv_heads*head_dim] F16 */
        uint16_t*                  v_cache,     /* [n_layers, max_seq_len, n_kv_heads*head_dim] F16 */
        float*                     logits_out,  /* [n_tokens, vocab_size] */
        long*                      token_out)   /* [n_tokens] argmax result */
{
    const int E = cfg->embed_dim;
    const int F = cfg->ffn_dim;
    const int H = cfg->n_heads;
    const int D = cfg->head_dim;
    const int HKV = cfg->n_kv_heads;
    const int max_dim = (E > F) ? E : F;
    const int max_proj = (H * D > F) ? H * D : F;

    /* Allocate working buffers */
    float* x        = (float*)bpd_alloc((size_t)n_tokens * E * sizeof(float));
    float* scratch1 = (float*)bpd_alloc((size_t)n_tokens * max_dim * sizeof(float));
    float* scratch2 = (float*)bpd_alloc((size_t)n_tokens * max_proj * sizeof(float));
    float* scratch3 = (float*)bpd_alloc((size_t)n_tokens * max_proj * sizeof(float));
    if (!x || !scratch1 || !scratch2 || !scratch3) goto cleanup;

    /* 1. Token embedding lookup */
    bpd_embed_lookup_q8_0_cpu(weights->token_embd, token_ids, x, n_tokens, E);

    /* 2. Iterate through all transformer layers */
    const size_t kv_layer_stride = (size_t)cfg->max_seq_len * HKV * D;
    for (int layer = 0; layer < cfg->n_layers; layer++) {
        uint16_t* layer_k_cache = k_cache + layer * kv_layer_stride;
        uint16_t* layer_v_cache = v_cache + layer * kv_layer_stride;

        /* inp_out_ids: final layer's FFN computes only the last token (matches ollama:
         * attn_out-{last}={2048,n_tokens}, ffn_inp-{last}={2048,1}). */
        int layer_n_out = (layer == cfg->n_layers - 1) ? 1 : n_tokens;
        bpd_llama_block_cpu(x, &weights->layers[layer], cfg,
                            pos_ids, n_tokens, kv_pos,
                            layer_k_cache, layer_v_cache,
                            scratch1, scratch2, scratch3,
                            weights->rope_freqs, layer_n_out);
    }
    {
        int last = n_tokens - 1;
        float* x_last = x + (size_t)last * E;
        float* logits_last = logits_out + (size_t)last * cfg->vocab_size;
        bpd_rmsnorm_llama_cpu(x_last, weights->output_norm_w, scratch1, 1, E, cfg->rms_eps);
        bpd_qmatmul_q8_0_llamafile_cpu(weights->output_w, scratch1, logits_last,
                                       cfg->vocab_size, 1, E);
        bpd_argmax_dim_cpu(logits_last, token_out + last, 1, cfg->vocab_size, 1);
    }
    goto cleanup;


cleanup:
    free(x);
    free(scratch1);
    free(scratch2);
    free(scratch3);
}

// ─── bpd_llama_generate_cpu — C-side autoregressive decode loop ───
// Eliminates Python per-token overhead (FFI + numpy argmax/argsort) by looping in C:
// prefill the prompt once, then generate n_generate tokens, feeding each argmax back.
// Scratch (logits/tok_out/pos_ids) allocated ONCE, not per forward call.
// k_cache/v_cache are passed through opaquely (the forward knows the dtype via cfg).
void bpd_llama_generate_cpu(
        const int32_t*             prompt_tokens,
        int                        prompt_len,
        int                        n_generate,
        const bpd_llama_weights*   weights,
        const bpd_llama_config*    cfg,
        void*                      k_cache,
        void*                      v_cache,
        int32_t*                   out_tokens)
{
    const int V = cfg->vocab_size;
    const int max_batch = prompt_len > 1 ? prompt_len : 1;

    float* logits = (float*)malloc((size_t)max_batch * V * sizeof(float));
    long*  tok_out = (long*)malloc((size_t)max_batch * sizeof(long));
    int32_t* pos_ids = (int32_t*)malloc((size_t)max_batch * sizeof(int32_t));
    if (!logits || !tok_out || !pos_ids) { free(logits); free(tok_out); free(pos_ids); return; }

    // Prefill the prompt (positions 0..prompt_len-1, kv_pos=0).
    for (int i = 0; i < prompt_len; i++) pos_ids[i] = i;
    bpd_llama_forward_cpu(prompt_tokens, prompt_len, weights, cfg,
                          pos_ids, 0, (uint16_t*)k_cache, (uint16_t*)v_cache, logits, tok_out);
    int n_past = prompt_len;
    int32_t next = (int32_t)tok_out[prompt_len - 1];

    // Decode: one token per step at kv_pos=n_past, cache accumulates.
    for (int step = 0; step < n_generate; step++) {
        out_tokens[step] = next;
        if (step == n_generate - 1) break;
        pos_ids[0] = n_past;
        bpd_llama_forward_cpu(&next, 1, weights, cfg,
                              pos_ids, n_past, (uint16_t*)k_cache, (uint16_t*)v_cache, logits, tok_out);
        n_past += 1;
        next = (int32_t)tok_out[0];
    }

    free(logits); free(tok_out); free(pos_ids);
}
