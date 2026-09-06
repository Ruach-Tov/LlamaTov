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
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float result;
    if (exp == 0) {
        result = (mant == 0) ? 0.0f : ldexpf((float)mant / 1024.0f, -14);
    } else if (exp == 31) {
        result = (mant == 0) ? INFINITY : NAN;
    } else {
        result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    }
    return sign ? -result : result;
}

#if BPD_HAVE_AVX1

// Generator macro for the 9 tile kernels
// Mirrors llamafile_sgemm tinyBLAS_Q0_AVX::gemm<RM, RN> exactly
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
            int RM = (m_rem >= 4) ? 4 : (m_rem >= 2) ? 2 : 1;
            
            const uint8_t* W_tile = W_q8_0 + (size_t)ii * bytes_per_row;
            const uint8_t* X_tile = X_q8_0 + (size_t)jj * bytes_per_row;
            float* out_tile = out + (size_t)jj * ldc + ii;
            
            if (RM == 4 && RN == 4) {
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
