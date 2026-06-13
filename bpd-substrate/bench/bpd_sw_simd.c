// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_sw_simd.c — SIMD-striped Smith-Waterman (Farrar 2007)
 *
 * 4 cells simultaneously using SSE 128-bit int32 registers.
 * Query striped: lane L gets positions L*seg_len .. (L+1)*seg_len-1.
 *
 * Key insight (Farrar): process segments sequentially within each column.
 * The diagonal H(i-1,j-1) is the H value saved BEFORE updating in
 * the previous column. The vertical gap F propagates DOWN through
 * segments with a lazy correction loop.
 *
 * Build: gcc -O2 -msse4.1 -shared -fPIC -o build/bpd_sw_simd.so bench/bpd_sw_simd.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <smmintrin.h>  /* SSE4.1 for _mm_max_epi32 */

#define SIMD_LANES 4

static inline int base_idx(char c) {
    switch(c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 0;
    }
}

typedef struct {
    int score;
    int query_end;
    int ref_end;
} sw_simd_result_t;

void bpd_smith_waterman_simd_cpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    sw_simd_result_t* result)
{
    int seg_len = (qlen + SIMD_LANES - 1) / SIMD_LANES;

    /* Build striped query profile */
    __m128i** profile = (__m128i**)malloc(4 * sizeof(__m128i*));
    for (int b = 0; b < 4; b++) {
        profile[b] = (__m128i*)_mm_malloc(seg_len * sizeof(__m128i), 16);
        for (int s = 0; s < seg_len; s++) {
            int scores[SIMD_LANES];
            for (int l = 0; l < SIMD_LANES; l++) {
                int qpos = s + l * seg_len;
                if (qpos < qlen) {
                    scores[l] = (base_idx(query[qpos]) == b) ? match_score : mismatch_score;
                } else {
                    scores[l] = 0;
                }
            }
            profile[b][s] = _mm_set_epi32(scores[3], scores[2], scores[1], scores[0]);
        }
    }

    /* H and E arrays: one vector per segment, representing one column */
    __m128i* vH = (__m128i*)_mm_malloc(seg_len * sizeof(__m128i), 16);
    __m128i* vE = (__m128i*)_mm_malloc(seg_len * sizeof(__m128i), 16);
    /* Store previous column's H for diagonal access */
    __m128i* vHp = (__m128i*)_mm_malloc(seg_len * sizeof(__m128i), 16);

    __m128i v_zero = _mm_setzero_si128();
    __m128i v_gapO = _mm_set1_epi32(gap_open);
    __m128i v_gapE = _mm_set1_epi32(gap_extend);

    for (int s = 0; s < seg_len; s++) {
        vH[s] = v_zero;
        vHp[s] = v_zero;
        vE[s] = v_zero;
    }

    int max_score = 0;
    int max_i = 0, max_j = 0;

    /* Process each reference position (column) */
    for (int j = 0; j < rlen; j++) {
        int rbase = base_idx(ref[j]);
        __m128i* prof = profile[rbase];

        /* The diagonal value for segment 0 comes from the LAST segment
         * of the previous column, shifted right by one lane.
         * This is the H(i-1, j-1) for the first element of each stripe. */
        __m128i v_diag = _mm_slli_si128(vHp[seg_len - 1], 4); /* shift last seg right */

        __m128i v_F = v_zero;

        /* Main loop: process each segment */
        for (int s = 0; s < seg_len; s++) {
            /* Save current H for next column's diagonal */
            __m128i v_H_old = vH[s];

            /* H = diag + score(query[i], ref[j]) */
            __m128i v_H = _mm_add_epi32(v_diag, prof[s]);

            /* E = max(H_prev[i][j-1] - gap_open, E[i][j-1] - gap_extend)
             * H_prev is the H from the PREVIOUS column at same row = vHp[s] */
            __m128i v_E_new = _mm_max_epi32(
                _mm_sub_epi32(vHp[s], v_gapO),
                _mm_sub_epi32(vE[s], v_gapE));
            vE[s] = v_E_new;

            /* H = max(H, E) */
            v_H = _mm_max_epi32(v_H, v_E_new);

            /* F = max(H[i-1][j] - gap_open, F[i-1][j] - gap_extend)
             * Within stripe: F propagates from previous segment */
            v_F = _mm_max_epi32(
                _mm_sub_epi32(v_H, v_gapO),
                _mm_sub_epi32(v_F, v_gapE));

            /* H = max(H, F, 0) */
            v_H = _mm_max_epi32(v_H, v_F);
            v_H = _mm_max_epi32(v_H, v_zero);

            /* Diagonal for next segment = H from previous column at current segment */
            v_diag = vHp[s];

            /* Store H */
            vH[s] = v_H;
        }

        /* Lazy-F correction: F propagates across stripe boundaries */
        for (int iter = 0; iter < SIMD_LANES - 1; iter++) {
            v_F = _mm_slli_si128(v_F, 4);  /* shift left = propagate to next lane */
            v_F = _mm_sub_epi32(v_F, v_gapE);
            int changed = 0;
            for (int s = 0; s < seg_len; s++) {
                __m128i v_H_new = _mm_max_epi32(vH[s], v_F);
                v_H_new = _mm_max_epi32(v_H_new, v_zero);
                __m128i v_cmp = _mm_cmpeq_epi32(v_H_new, vH[s]);
                if (_mm_movemask_epi8(v_cmp) != 0xFFFF) changed = 1;
                vH[s] = v_H_new;
                v_F = _mm_max_epi32(
                    _mm_sub_epi32(v_H_new, v_gapO),
                    _mm_sub_epi32(v_F, v_gapE));
            }
            if (!changed) break;
        }

        /* Track maximum */
        for (int s = 0; s < seg_len; s++) {
            int vals[SIMD_LANES];
            _mm_storeu_si128((__m128i*)vals, vH[s]);
            for (int l = 0; l < SIMD_LANES; l++) {
                if (vals[l] > max_score) {
                    max_score = vals[l];
                    max_i = s + l * seg_len;
                    max_j = j;
                }
            }
        }

        /* Save current H as previous column for next iteration */
        for (int s = 0; s < seg_len; s++) {
            vHp[s] = vH[s];
        }
    }

    result->score = max_score;
    result->query_end = max_i;
    result->ref_end = max_j;

    for (int b = 0; b < 4; b++) _mm_free(profile[b]);
    free(profile);
    _mm_free(vH); _mm_free(vE); _mm_free(vHp);
}

int bpd_sw_simd_score_cpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match, int mismatch,
    int gap_open, int gap_extend)
{
    sw_simd_result_t result;
    bpd_smith_waterman_simd_cpu(query, qlen, ref, rlen,
        match, mismatch, gap_open, gap_extend, &result);
    return result.score;
}
