// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <immintrin.h>
#include <stdint.h>
#include <math.h>

static const float C_SQRT2_INV = 0.7071067691f;
static const float C_ERF_P     = 0.3275910914f;
static const float C_ERF_A1    = 0.2548295856f;
static const float C_ERF_A2    = -0.2844967246f;
static const float C_ERF_A3    = 1.4214137793f;
static const float C_ERF_A4    = -1.4531520605f;
static const float C_ERF_A5    = 1.0614054203f;
static const float C_LOG2E     = 1.4426950216f;
static const float C_LN2       = 0.6931471825f;
static const float C_EXP_C1    = 0.9999997020f;
static const float C_EXP_C2    = 0.4999915063f;
static const float C_EXP_C3    = 0.1666765213f;
static const float C_EXP_C4    = 0.0418978222f;
static const float C_EXP_C5    = 0.0082892906f;

static inline __m256 ondnn_exp_avx(__m256 x) {
    __m256 log2e = _mm256_set1_ps(C_LOG2E);
    __m256 ln2   = _mm256_set1_ps(C_LN2);
    __m256 half  = _mm256_set1_ps(0.5f);
    __m256 one   = _mm256_set1_ps(1.0f);
    __m256 clamp_hi = _mm256_set1_ps(88.7228393555f);
    __m256 clamp_lo = _mm256_set1_ps(-87.3365478516f);
    x = _mm256_min_ps(x, clamp_hi);
    x = _mm256_max_ps(x, clamp_lo);
    __m256 x_orig = x;
    __m256 z = _mm256_add_ps(_mm256_mul_ps(x, log2e), half);
    __m256 n = _mm256_round_ps(z, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(x_orig, _mm256_mul_ps(n, ln2));
    __m256 n_adj = _mm256_sub_ps(n, one);
    __m256 p = _mm256_set1_ps(C_EXP_C5);
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(C_EXP_C4));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(C_EXP_C3));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(C_EXP_C2));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(C_EXP_C1));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), one);
    __m256i ni = _mm256_cvtps_epi32(n_adj);
    __m128i ni_lo = _mm256_castsi256_si128(ni);
    __m128i ni_hi = _mm256_extractf128_si256(ni, 1);
    __m128i bias = _mm_set1_epi32(127);
    __m128i elo = _mm_slli_epi32(_mm_add_epi32(ni_lo, bias), 23);
    __m128i ehi = _mm_slli_epi32(_mm_add_epi32(ni_hi, bias), 23);
    __m256 scale = _mm256_castsi256_ps(
        _mm256_insertf128_si256(_mm256_castsi128_si256(elo), ehi, 1));
    return _mm256_mul_ps(_mm256_mul_ps(p, scale), _mm256_set1_ps(2.0f));
}

void bpd_gelu_ondnn_cpu(const float* input, float* output, int n) {
    __m256 sqrt2_inv = _mm256_set1_ps(C_SQRT2_INV);
    __m256 half      = _mm256_set1_ps(0.5f);
    __m256 one       = _mm256_set1_ps(1.0f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 abs_mask  = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 erf_p  = _mm256_set1_ps(C_ERF_P);
    __m256 erf_a1 = _mm256_set1_ps(C_ERF_A1);
    __m256 erf_a2 = _mm256_set1_ps(C_ERF_A2);
    __m256 erf_a3 = _mm256_set1_ps(C_ERF_A3);
    __m256 erf_a4 = _mm256_set1_ps(C_ERF_A4);
    __m256 erf_a5 = _mm256_set1_ps(C_ERF_A5);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 v = _mm256_mul_ps(x, sqrt2_inv);
        __m256 av = _mm256_and_ps(v, abs_mask);
        __m256 v_sign = _mm256_and_ps(v, sign_mask);

        __m256 t = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(erf_p, av)));
        __m256 neg_v2 = _mm256_xor_ps(_mm256_mul_ps(v, v), sign_mask);
        __m256 e = ondnn_exp_avx(neg_v2);

        /* JIT ORDER: (exp * t) first, then Horner4, matches offset 0x186 */
        __m256 et = _mm256_mul_ps(e, t);
        __m256 h = erf_a5;
        h = _mm256_add_ps(_mm256_mul_ps(h, t), erf_a4);
        h = _mm256_add_ps(_mm256_mul_ps(h, t), erf_a3);
        h = _mm256_add_ps(_mm256_mul_ps(h, t), erf_a2);
        h = _mm256_add_ps(_mm256_mul_ps(h, t), erf_a1);

        /* erf_abs = 1 - (exp*t) * h */
        __m256 erf_abs = _mm256_sub_ps(one, _mm256_mul_ps(et, h));
        /* Apply sign: erf = sign(v) * erf_abs */
        __m256 erf_val = _mm256_xor_ps(erf_abs, v_sign);
        /* gelu = 0.5 * x * (1 + erf) */
        __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, erf_val)));

        _mm256_storeu_ps(output + i, result);
    }
    for (; i < n; i++) {
        float x = input[i];
        float v = x * C_SQRT2_INV;
        output[i] = 0.5f * x * (1.0f + erff(v));
    }
}
