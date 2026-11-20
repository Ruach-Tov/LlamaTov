#include <stdint.h>
#include <math.h>
#include <stddef.h>

/* Doresh's #11 last-mile fix, second part (Sept 2026): real ggml's scalar
 * Q8_0 dot product (ggml-cpu/arch/x86/quants.c, compiled WITHOUT -mfma
 * for the reference build -- confirmed via its actual compile command,
 * no -m flags beyond baseline) does `sumf += (float)sumi * (wd*xd);` as
 * TWO separately-rounded IEEE-754 operations (multiply, then add). Our
 * own bpd_cpu.so is built WITH -mfma (needed for other kernels), which
 * lets gcc silently CONTRACT this exact expression into a single-
 * rounding vfmadd231ss instruction -- a genuinely different (more
 * precise, but DIFFERENT) floating-point result at the last-bit level.
 * Found via disassembly of the compiled bpd_vec_dot_q8_0_scalar (10
 * vfmadd instructions where 0 were expected) after a real fixture
 * element (row=5,col=1833 of Qcur-0) diverged from ggml's real
 * dlsym'd ggml_vec_dot_q8_0_q8_0 by exactly 2 ULP despite the SOURCE
 * algorithm being bit-identical (verified separately via random-data
 * dlsym comparison).
 *
 * NOTE: `#pragma STDC FP_CONTRACT OFF` was tried FIRST and does NOT
 * reliably suppress contraction under GCC 14 in this file (verified:
 * still produced 1-10 vfmadd instructions depending on exact function
 * structure/inlining, despite being valid, standard-conforming C and
 * correctly scoped at file level) -- this is a known GCC limitation,
 * the pragma is honored inconsistently compared to Clang. The reliable
 * mechanism is the per-function `optimize("fp-contract=off")`
 * attribute below, applied directly to the one function where this
 * matters (bpd_vec_dot_q8_0_scalar) -- verified via disassembly to
 * produce 0 vfmadd instructions regardless of build flags used
 * elsewhere in the file, which is important since the old AVX tile
 * kernels this file also defines (no longer called by the dispatcher,
 * dead code kept for reference) are unaffected either way. */

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


// Declare the 9 AVX tile kernels (kept for reference/future use, but NO
// LONGER CALLED by the dispatcher below -- see Doresh's #11 last-mile
// finding, Sept 2026).
void bpd_gemm_q8_0_1_1_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_1_2_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_1_4_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_2_1_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_2_2_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_2_4_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_4_1_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_4_2_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);
void bpd_gemm_q8_0_4_4_cpu(const uint8_t* W, const uint8_t* B, int k, int ws, int as, float* out, int ldc);

/* Doresh's #11 LAST-MILE FIX (Sept 2026), per Heath's ruling: chase to
 * strict 0-ULP, don't ship with the residual documented-but-unfixed.
 *
 * THE REAL FINDING: disassembled the actual COMPILED llamafile_sgemm in
 * the real reference build's libggml-cpu.so -- it is a bare
 * `xor eax,eax; ret` (unconditional `return false`). Traced why: the
 * GGML_TYPE_Q8_0 case inside llamafile_sgemm (sgemm.cpp) requires
 * __AVX2__||__AVX512F__||__AVX__ (tinyBLAS_Q0_AVX) or
 * __ARM_FEATURE_DOTPROD (tinyBLAS_Q0_ARM) or __MMA__ or PPC -- NONE of
 * these are defined for the reference build (confirmed via CMakeCache:
 * GGML_AVX=OFF, GGML_AVX2=OFF, and the actual compile command has no -m
 * flags beyond baseline x86-64 SSE2). With none of those macros defined,
 * every branch in that switch case is compile-time-dead, and the
 * function's ONLY reachable path is `return false;` -- the compiler
 * folded the WHOLE function to a stub via dead-code elimination.
 *
 * CONSEQUENCE: llamafile_sgemm NEVER runs for ANY matmul in the real
 * reference build, regardless of M/N/K shape. ggml-cpu.c's caller does
 * `if (!llamafile_sgemm(...)) goto UseGgmlGemm1;` -- since it ALWAYS
 * returns false here, EVERY Q8_0 matmul (Q/K/V/O/FFN projections, for
 * ALL M including M>=2) falls through to ggml's generic per-row
 * vec_dot() fallback, calling ggml_vec_dot_q8_0_q8_0 directly for each
 * (row,col) output element. And THAT function's own __AVX2__/__AVX__
 * fast paths are ALSO compile-time-dead for the same reason -- it falls
 * through to the PLAINEST POSSIBLE scalar loop (ggml-cpu/arch/x86/
 * quants.c:1225-1233): for each of the nb=k/32 blocks, a sequential
 * INTEGER accumulation of 32 element-products (`sumi`), then a
 * sequential FLOAT accumulation across blocks
 * (`sumf += sumi*(d_x*d_y)`), in block order 0,1,2,...,nb-1. No SIMD,
 * no tiling, no lane-order to match -- it's simpler than everything
 * previously investigated (the GEMM tile-dispatch table work, and the
 * bpd_vec_dot_q8_0_ggml AVX-intrinsic M==1 special case, were BOTH
 * mirroring compile-time-DEAD code paths in the real reference).
 *
 * THE FIX: replace the tile-based AVX dispatch entirely with ONE plain
 * scalar per-(row,col) dot product matching ggml's real fallback
 * exactly. bpd's OWN existing "scalar fallback" macro
 * (DECLARE_Q8_0_TILE_KERNEL_SCALAR, in the #else branch above, normally
 * only compiled when __AVX__ is undefined) already has EXACTLY the
 * right per-block accumulation order -- it was simply never reachable
 * because bpd_cpu.so is always built with -mavx2 -mfma -mf16c, making
 * __AVX__ defined and routing through the (wrong, for this purpose)
 * AVX-intrinsic path unconditionally. This dispatcher now ALWAYS uses
 * the scalar path, independent of BPD_HAVE_AVX1, since the target we
 * are matching (real ggml on this specific reference build) has no
 * AVX/ARM-dotprod/MMA/PPC compiled in at all. */
__attribute__((optimize("fp-contract=off")))
static float bpd_vec_dot_q8_0_scalar(int nb, const uint8_t* W_row, const uint8_t* X_row) {
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* wblock = W_row + (size_t)ib * 34;
        const uint8_t* xblock = X_row + (size_t)ib * 34;
        const int8_t* wq = (const int8_t*)(wblock + 2);
        const int8_t* xq = (const int8_t*)(xblock + 2);
        int sumi = 0;
        for (int j = 0; j < 32; j++) sumi += (int)wq[j] * (int)xq[j];
        uint16_t wd_u16, xd_u16;
        __builtin_memcpy(&wd_u16, wblock, 2);
        __builtin_memcpy(&xd_u16, xblock, 2);
        float wd = bpd_f16_to_f32_local(wd_u16);
        float xd = bpd_f16_to_f32_local(xd_u16);
        sumf += (float)sumi * (wd * xd);
    }
    return sumf;
}

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

    for (int jj = 0; jj < m_tokens; jj++) {
        const uint8_t* X_row = X_q8_0 + (size_t)jj * bytes_per_row;
        float* out_col = out + (size_t)jj * ldc;
        for (int ii = 0; ii < m_weight; ii++) {
            const uint8_t* W_row = W_q8_0 + (size_t)ii * bytes_per_row;
            out_col[ii] = bpd_vec_dot_q8_0_scalar(k, W_row, X_row);
        }
    }
}
