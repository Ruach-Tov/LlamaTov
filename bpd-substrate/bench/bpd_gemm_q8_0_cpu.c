// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdint.h>
#include <math.h>
#include <stddef.h>

#if defined(__AVX__)
#include <immintrin.h>
#define BPD_HAVE_AVX1 1
#else
#define BPD_HAVE_AVX1 0
#endif

// Shared f16_to_f32 utility (assumed to be available or defined here)
static inline float bpd_f16_to_f32_local(uint16_t h) {
    // F16C hardware (single vcvtph2ps, branchless). Bit-identical to the bit-assembly
    // version. Eliminates the subnormal branch in the decode tile kernels.
    return _cvtsh_ss(h);
}

#if BPD_HAVE_AVX1

// Generator macro for the 9 tile kernels
// Mirrors llamafile_sgemm tinyBLAS_Q0_AVX::gemm<RM, RN> exactly

/* ggml-EXACT AVX q8_0 dot (ggml_vec_dot_q8_0_q8_0, __AVX__ path): 2 blocks/iter, mul+add, hsum_float_8.
 * ggml uses THIS for M=1 matmuls (tinyBLAS handles M>=2). W and X are q8_0-packed (34B/block). */
static inline __m128i bpd_mul_add_epi8_sse(__m128i x, __m128i y){
    __m128i ax = _mm_sign_epi8(x, x);
    __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}
static inline __m256 bpd_mul_sum_i8_quad(__m128i x10,__m128i x11,__m128i x20,__m128i x21,
                                         __m128i y10,__m128i y11,__m128i y20,__m128i y21){
    const __m128i mone = _mm_set1_epi16(1);
    __m128i p1_0 = _mm_madd_epi16(bpd_mul_add_epi8_sse(x10,y10), mone);
    __m128i p1_1 = _mm_madd_epi16(bpd_mul_add_epi8_sse(x11,y11), mone);
    __m128i p2_0 = _mm_madd_epi16(bpd_mul_add_epi8_sse(x20,y20), mone);
    __m128i p2_1 = _mm_madd_epi16(bpd_mul_add_epi8_sse(x21,y21), mone);
    __m128i p1 = _mm_add_epi32(p1_0, p1_1);
    __m128i p2 = _mm_add_epi32(p2_0, p2_1);
    return _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(p1), p2, 1));
}
static inline float bpd_hsum_float_8(__m256 x){
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
/* W_row, X_row: q8_0-packed rows, K elements = nb=K/32 blocks of 34 bytes each. */
static inline float bpd_vec_dot_q8_0_ggml(int nb, const uint8_t* W_row, const uint8_t* X_row){
    /* param is block count */
    __m256 accum = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2){
        const uint8_t* xb0 = W_row + (size_t)ib*34;       const uint8_t* yb0 = X_row + (size_t)ib*34;
        const uint8_t* xb1 = W_row + (size_t)(ib+1)*34;   const uint8_t* yb1 = X_row + (size_t)(ib+1)*34;
        __m128i qx10 = _mm_loadu_si128((const __m128i*)(xb0+2));
        __m128i qx11 = _mm_loadu_si128((const __m128i*)(xb0+2+16));
        __m128i qx20 = _mm_loadu_si128((const __m128i*)(xb1+2));
        __m128i qx21 = _mm_loadu_si128((const __m128i*)(xb1+2+16));
        __m128i qy10 = _mm_loadu_si128((const __m128i*)(yb0+2));
        __m128i qy11 = _mm_loadu_si128((const __m128i*)(yb0+2+16));
        __m128i qy20 = _mm_loadu_si128((const __m128i*)(yb1+2));
        __m128i qy21 = _mm_loadu_si128((const __m128i*)(yb1+2+16));
        __m256 p = bpd_mul_sum_i8_quad(qx10,qx11,qx20,qx21, qy10,qy11,qy20,qy21);
        uint16_t xd0,yd0,xd1,yd1;
        __builtin_memcpy(&xd0, xb0, 2); __builtin_memcpy(&yd0, yb0, 2);
        __builtin_memcpy(&xd1, xb1, 2); __builtin_memcpy(&yd1, yb1, 2);
        __m256 deltas = _mm256_set_m128(_mm_set1_ps(_cvtsh_ss(xd1)*_cvtsh_ss(yd1)),
                                        _mm_set1_ps(_cvtsh_ss(xd0)*_cvtsh_ss(yd0)));
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }
    return bpd_hsum_float_8(accum);
}

#define DECLARE_Q8_0_TILE_KERNEL(RM, RN) \
void bpd_gemm_q8_0_##RM##_##RN##_cpu( \
    const uint8_t* W_tile_base, \
    const uint8_t* B_tile_base, \
    int k, \
    int weight_row_stride, \
    int act_row_stride, \
    float* out_base, \
    int ldc) \
{ \
    __m256 Cv[RN][RM]; \
    for (int j = 0; j < RN; j++) \
        for (int i = 0; i < RM; i++) \
            Cv[j][i] = _mm256_setzero_ps(); \
\
    for (int l = 0; l < k; l++) { \
        for (int j = 0; j < RN; j++) { \
            const uint8_t* Bblock = B_tile_base + j * act_row_stride + l * 34; \
            __m128i blj0 = _mm_loadu_si128((const __m128i*)(Bblock + 2)); \
            __m128i blj1 = _mm_loadu_si128((const __m128i*)(Bblock + 2 + 16)); \
            uint16_t Bd_u16 = (uint16_t)Bblock[0] | ((uint16_t)Bblock[1] << 8); \
            float Bd = bpd_f16_to_f32_local(Bd_u16); \
            for (int i = 0; i < RM; i++) { \
                const uint8_t* Ablock = W_tile_base + i * weight_row_stride + l * 34; \
                __m128i ali0 = _mm_loadu_si128((const __m128i*)(Ablock + 2)); \
                __m128i ali1 = _mm_loadu_si128((const __m128i*)(Ablock + 2 + 16)); \
                __m128i sepAA0 = _mm_sign_epi8(ali0, ali0); \
                __m128i sepAA1 = _mm_sign_epi8(ali1, ali1); \
                __m128i sepBA0 = _mm_sign_epi8(blj0, ali0); \
                __m128i sepBA1 = _mm_sign_epi8(blj1, ali1); \
                const __m128i oneFill = _mm_set1_epi16(1); \
                __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0); \
                __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1); \
                __m128i p32_0 = _mm_madd_epi16(oneFill, mad0); \
                __m128i p32_1 = _mm_madd_epi16(oneFill, mad1); \
                __m256i p32 = _mm256_insertf128_si256( \
                    _mm256_castsi128_si256(p32_0), p32_1, 1); \
                __m256 udTmp = _mm256_cvtepi32_ps(p32); \
                uint16_t Ad_u16 = (uint16_t)Ablock[0] | ((uint16_t)Ablock[1] << 8); \
                float Ad = bpd_f16_to_f32_local(Ad_u16); \
                __m256 scale = _mm256_set1_ps(Ad * Bd); \
                Cv[j][i] = _mm256_add_ps(_mm256_mul_ps(scale, udTmp), Cv[j][i]); \
            } \
        } \
    } \
    for (int j = 0; j < RN; j++) { \
        for (int i = 0; i < RM; i++) { \
            __m128 v = _mm_add_ps(_mm256_extractf128_ps(Cv[j][i], 1), \
                                  _mm256_castps256_ps128(Cv[j][i])); \
            v = _mm_add_ps(v, _mm_movehl_ps(v, v)); \
            v = _mm_add_ss(v, _mm_movehdup_ps(v)); \
            out_base[j * ldc + i] = _mm_cvtss_f32(v); \
        } \
    } \
}

// Generate all 9 kernels
DECLARE_Q8_0_TILE_KERNEL(1, 1)
DECLARE_Q8_0_TILE_KERNEL(1, 2)
DECLARE_Q8_0_TILE_KERNEL(1, 4)
DECLARE_Q8_0_TILE_KERNEL(2, 1)
DECLARE_Q8_0_TILE_KERNEL(2, 2)
DECLARE_Q8_0_TILE_KERNEL(2, 4)
DECLARE_Q8_0_TILE_KERNEL(4, 1)
DECLARE_Q8_0_TILE_KERNEL(4, 2)
DECLARE_Q8_0_TILE_KERNEL(4, 4)

#else

// Scalar fallback for non-AVX builds
#define DECLARE_Q8_0_TILE_KERNEL_SCALAR(RM, RN) \
void bpd_gemm_q8_0_##RM##_##RN##_cpu( \
    const uint8_t* W_tile_base, \
    const uint8_t* B_tile_base, \
    int k, \
    int weight_row_stride, \
    int act_row_stride, \
    float* out_base, \
    int ldc) \
{ \
    for (int j = 0; j < RN; j++) { \
        for (int i = 0; i < RM; i++) { \
            float sumf = 0.0f; \
            for (int l = 0; l < k; l++) { \
                const uint8_t* Ablock = W_tile_base + i * weight_row_stride + l * 34; \
                const uint8_t* Bblock = B_tile_base + j * act_row_stride + l * 34; \
                const int8_t* wq = (const int8_t*)(Ablock + 2); \
                const int8_t* aq = (const int8_t*)(Bblock + 2); \
                int sumi = 0; \
                for (int q = 0; q < 32; q++) sumi += (int)wq[q] * (int)aq[q]; \
                uint16_t Ad_u16 = (uint16_t)Ablock[0] | ((uint16_t)Ablock[1] << 8); \
                uint16_t Bd_u16 = (uint16_t)Bblock[0] | ((uint16_t)Bblock[1] << 8); \
                float Ad = bpd_f16_to_f32_local(Ad_u16); \
                float Bd = bpd_f16_to_f32_local(Bd_u16); \
                sumf += (float)sumi * (Ad * Bd); \
            } \
            out_base[j * ldc + i] = sumf; \
        } \
    } \
}

DECLARE_Q8_0_TILE_KERNEL_SCALAR(1, 1)
DECLARE_Q8_0_TILE_KERNEL_SCALAR(1, 2)
DECLARE_Q8_0_TILE_KERNEL_SCALAR(1, 4)
DECLARE_Q8_0_TILE_KERNEL_SCALAR(2, 1)
DECLARE_Q8_0_TILE_KERNEL_SCALAR(2, 2)
DECLARE_Q8_0_TILE_KERNEL_SCALAR(2, 4)
DECLARE_Q8_0_TILE_KERNEL_SCALAR(4, 1)
DECLARE_Q8_0_TILE_KERNEL_SCALAR(4, 2)
DECLARE_Q8_0_TILE_KERNEL_SCALAR(4, 4)

#endif


// Declare the 9 tile kernels for the dispatcher
void bpd_gemm_q8_0_1_1_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_1_2_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_1_4_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_2_1_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_2_2_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_2_4_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_4_1_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_4_2_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_4_4_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);

// bpd_qmatmul_q8_0_dispatch_cpu tile dispatcher
// Mirrors mnpack logic from llamafile_sgemm
void bpd_qmatmul_q8_0_dispatch_cpu(
    const uint8_t* W_q8_0,
    const uint8_t* X_q8_0,
    float* out,
    int m_weight,
    int m_tokens,
    int K)
{
    int k = K / 32;
    int bytes_per_row = k * 34;
    int ldc = m_weight;

    for (int jj = 0; jj < m_tokens; ) {
        int n_rem = m_tokens - jj;
        int RN = (n_rem >= 4) ? 4 : (n_rem >= 2) ? 2 : 1;
        
        for (int ii = 0; ii < m_weight; ) {
            int m_rem = m_weight - ii;
            /* BPD access_shape param (matmul_layout.pl): mat-VECTOR (RN==1, decode) has no
             * token reuse to amortize multi-row column-major loads -> use RM=1 row-sequential
             * (stream each contiguous weight row, 7.79 GB/s) not RM=4 (4.02, L1-thrash). */
            int RM = (RN == 1) ? 1 : ((m_rem >= 4) ? 4 : (m_rem >= 2) ? 2 : 1);
            
            const uint8_t* W_tile = W_q8_0 + (size_t)ii * bytes_per_row;
            const uint8_t* X_tile = X_q8_0 + (size_t)jj * bytes_per_row;
            float* out_tile = out + (size_t)jj * ldc + ii;
            
            if (RN == 1) {
                for (int i = 0; i < RM; i++)
                    out_tile[i] = bpd_vec_dot_q8_0_ggml(k, W_tile + (size_t)i * bytes_per_row, X_tile);
            } else if (RM == 4 && RN == 4) {
                bpd_gemm_q8_0_4_4_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            } else if (RM == 4 && RN == 2) {
                bpd_gemm_q8_0_4_2_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            } else if (RM == 4 && RN == 1) {
                bpd_gemm_q8_0_4_1_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            } else if (RM == 2 && RN == 4) {
                bpd_gemm_q8_0_2_4_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            } else if (RM == 2 && RN == 2) {
                bpd_gemm_q8_0_2_2_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            } else if (RM == 2 && RN == 1) {
                bpd_gemm_q8_0_2_1_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            } else if (RM == 1 && RN == 4) {
                bpd_gemm_q8_0_1_4_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            } else if (RM == 1 && RN == 2) {
                bpd_gemm_q8_0_1_2_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            } else {
                bpd_gemm_q8_0_1_1_cpu(W_tile, X_tile, k, bytes_per_row, bytes_per_row, out_tile, ldc);
            }
            
            ii += RM;
        }
        jj += RN;
    }
}
