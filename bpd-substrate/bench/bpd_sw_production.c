// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_smith_waterman_banded.c — Banded Smith-Waterman (CPU + GPU-ready)
 *
 * Restricts the DP computation to a diagonal band of width 2*bandwidth+1
 * around the main diagonal. O(N * bandwidth) instead of O(N * M).
 *
 * For a 10Kbp read against a 10Kbp reference with bandwidth=100:
 *   Full SW: 100M cells
 *   Banded:  2M cells (50x reduction)
 *
 * Build: gcc -O2 -shared -fPIC -o build/bpd_sw_banded.so \
 *          bench/bpd_smith_waterman_banded.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    int score;
    int query_end;
    int ref_end;
    int cigar_len;
    char cigar[4096];
} sw_result_t;

/* Banded Smith-Waterman with affine gap penalties.
 * Only fills cells where |i/qlen - j/rlen| * max(qlen,rlen) <= bandwidth.
 * Equivalently: j_min = max(1, i*rlen/qlen - bandwidth)
 *               j_max = min(rlen, i*rlen/qlen + bandwidth)
 */
void bpd_smith_waterman_banded_cpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    int bandwidth,
    sw_result_t* result)
{
    /* Allocate scoring matrices — only need 2 rows for H, E, F */
    int band_width = 2 * bandwidth + 1;
    int alloc_cols = rlen + 1;

    int* H_prev = (int*)calloc(alloc_cols, sizeof(int));
    int* H_curr = (int*)calloc(alloc_cols, sizeof(int));
    int* E_prev = (int*)calloc(alloc_cols, sizeof(int));
    int* E_curr = (int*)calloc(alloc_cols, sizeof(int));
    int* F_curr = (int*)calloc(alloc_cols, sizeof(int));

    /* For traceback, store direction matrix */
    /* 0=stop, 1=diag, 2=left(E), 3=up(F) */
    char* trace = (char*)calloc((qlen+1) * (rlen+1), sizeof(char));
    #define TRACE(i,j) trace[(i)*(rlen+1)+(j)]

    int max_score = 0, max_i = 0, max_j = 0;

    for (int i = 1; i <= qlen; i++) {
        /* Compute band boundaries for this row */
        int j_center = (int)((long long)i * rlen / qlen);
        int j_min = j_center - bandwidth;
        if (j_min < 1) j_min = 1;
        int j_max = j_center + bandwidth;
        if (j_max > rlen) j_max = rlen;

        /* Clear cells outside band */
        for (int j = 0; j < j_min; j++) {
            H_curr[j] = 0;
            E_curr[j] = 0;
        }

        for (int j = j_min; j <= j_max; j++) {
            int s = (query[i-1] == ref[j-1]) ? match_score : mismatch_score;

            /* E: horizontal gap */
            int e1 = H_curr[j-1] - gap_open;
            int e2 = E_curr[j-1] - gap_extend;
            E_curr[j] = (e1 > e2) ? e1 : e2;

            /* F: vertical gap — need H_prev which is the previous row */
            int f1 = H_prev[j] - gap_open;
            int f2 = F_curr[j] - gap_extend;  /* F_curr[j] from previous row iteration... */
            /* Actually we need F from previous row. Store separately: */
            int f_val = (f1 > f2) ? f1 : f2;

            /* H */
            int h = H_prev[j-1] + s;
            int dir = 1;  /* diagonal */
            if (E_curr[j] > h) { h = E_curr[j]; dir = 2; }
            if (f_val > h) { h = f_val; dir = 3; }
            if (h < 0) { h = 0; dir = 0; }
            H_curr[j] = h;
            F_curr[j] = f_val;  /* store for next row */
            TRACE(i, j) = dir;

            if (h > max_score) {
                max_score = h;
                max_i = i;
                max_j = j;
            }
        }

        /* Clear cells after band */
        for (int j = j_max + 1; j <= rlen; j++) {
            H_curr[j] = 0;
            E_curr[j] = 0;
        }

        /* Swap rows */
        int* tmp = H_prev; H_prev = H_curr; H_curr = tmp;
        /* E doesn't need swapping — we read E_curr[j-1] in the same row */
    }

    result->score = max_score;
    result->query_end = max_i - 1;
    result->ref_end = max_j - 1;

    /* Traceback */
    int ci = max_i, cj = max_j;
    int ops[8192];
    int n_ops = 0;

    while (ci > 0 && cj > 0 && TRACE(ci, cj) != 0 && n_ops < 8190) {
        int dir = TRACE(ci, cj);
        if (dir == 1) { ops[n_ops++] = 0; ci--; cj--; }       /* M */
        else if (dir == 2) { ops[n_ops++] = 2; cj--; }         /* D */
        else if (dir == 3) { ops[n_ops++] = 1; ci--; }         /* I */
        else break;
    }

    /* Reverse and RLE into CIGAR */
    result->cigar[0] = '\0';
    if (n_ops > 0) {
        char* p = result->cigar;
        int k = n_ops - 1;
        while (k >= 0) {
            int op = ops[k];
            int count = 1;
            while (k > 0 && ops[k-1] == op) { count++; k--; }
            char c = (op == 0) ? 'M' : (op == 1) ? 'I' : 'D';
            p += sprintf(p, "%d%c", count, c);
            k--;
        }
        result->cigar_len = p - result->cigar;
    }

    free(H_prev); free(H_curr);
    free(E_prev); free(E_curr); free(F_curr);
    free(trace);
    #undef TRACE
}

/* Score-only interface */
int bpd_sw_banded_score_cpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match, int mismatch,
    int gap_open, int gap_extend,
    int bandwidth)
{
    sw_result_t result;
    bpd_smith_waterman_banded_cpu(query, qlen, ref, rlen,
        match, mismatch, gap_open, gap_extend, bandwidth, &result);
    return result.score;
}

/* ================================================================
 * Seed finder: binary search on suffix array for k-mer matches
 * Returns positions in the reference where the k-mer occurs.
 * ================================================================ */

int bpd_find_seeds(
    const char* ref, int rlen,
    const int* SA,
    const char* kmer, int klen,
    int* positions, int max_positions)
{
    /* Binary search for lower bound */
    int lo = 0, hi = rlen;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int cmp = strncmp(ref + SA[mid], kmer, klen);
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    int start = lo;

    /* Binary search for upper bound */
    hi = rlen;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int cmp = strncmp(ref + SA[mid], kmer, klen);
        if (cmp <= 0) lo = mid + 1;
        else hi = mid;
    }
    int end = lo;

    /* Collect positions */
    int count = 0;
    for (int i = start; i < end && count < max_positions; i++) {
        positions[count++] = SA[i];
    }
    return count;
}

/* ================================================================
 * Seed-and-extend: find seeds, chain nearby seeds, extend with banded SW
 * ================================================================ */

typedef struct {
    int ref_pos;     /* position in reference */
    int query_pos;   /* position in query */
} seed_t;

typedef struct {
    int score;
    int ref_start;
    int ref_end;
    int query_start;
    int query_end;
    char cigar[4096];
} alignment_result_t;

int bpd_seed_and_extend(
    const char* query, int qlen,
    const char* ref, int rlen,
    const int* SA,
    int kmer_size,
    int bandwidth,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    alignment_result_t* result)
{
    /* Step 1: Find all k-mer seeds from the query in the reference */
    int max_seeds = 1000;
    int* seed_positions = (int*)malloc(max_seeds * sizeof(int));
    seed_t* seeds = (seed_t*)malloc(max_seeds * sizeof(seed_t));
    int total_seeds = 0;

    for (int qp = 0; qp <= qlen - kmer_size && total_seeds < max_seeds; qp += kmer_size / 2) {
        int found = bpd_find_seeds(ref, rlen, SA, query + qp, kmer_size,
                                    seed_positions, max_seeds - total_seeds);
        for (int f = 0; f < found && total_seeds < max_seeds; f++) {
            seeds[total_seeds].ref_pos = seed_positions[f];
            seeds[total_seeds].query_pos = qp;
            total_seeds++;
        }
    }

    if (total_seeds == 0) {
        result->score = 0;
        result->cigar[0] = '\0';
        free(seed_positions);
        free(seeds);
        return 0;
    }

    /* Step 2: Find best seed cluster (simple: pick seed with most nearby seeds) */
    int best_seed = 0;
    int best_count = 0;
    for (int i = 0; i < total_seeds; i++) {
        int count = 0;
        int diag_i = seeds[i].ref_pos - seeds[i].query_pos;
        for (int j = 0; j < total_seeds; j++) {
            int diag_j = seeds[j].ref_pos - seeds[j].query_pos;
            if (abs(diag_i - diag_j) <= bandwidth) count++;
        }
        if (count > best_count) {
            best_count = count;
            best_seed = i;
        }
    }

    /* Step 3: Define extension window around best seed */
    int ref_center = seeds[best_seed].ref_pos;
    int ext_start = ref_center - qlen / 2;
    if (ext_start < 0) ext_start = 0;
    int ext_end = ref_center + qlen + qlen / 2;
    if (ext_end > rlen) ext_end = rlen;
    int ext_len = ext_end - ext_start;

    /* Step 4: Banded SW extension */
    sw_result_t sw;
    bpd_smith_waterman_banded_cpu(
        query, qlen,
        ref + ext_start, ext_len,
        match_score, mismatch_score,
        gap_open, gap_extend,
        bandwidth, &sw);

    result->score = sw.score;
    result->ref_start = ext_start;
    result->ref_end = ext_start + sw.ref_end;
    result->query_start = 0;
    result->query_end = sw.query_end;
    strncpy(result->cigar, sw.cigar, 4095);

    free(seed_positions);
    free(seeds);
    return total_seeds;
}

/* ================================================================
 * Quality-aware scoring
 * score = base_score * quality_factor
 * quality_factor = 1 - 10^(-Q/10) for matches
 *                = -(mismatch * quality_factor) for mismatches
 * ================================================================ */

void bpd_smith_waterman_quality_cpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    const uint8_t* quality,  /* Phred quality scores (0-40) */
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    sw_result_t* result)
{
    int* H = (int*)calloc((qlen+1) * (rlen+1), sizeof(int));
    int* E = (int*)calloc((qlen+1) * (rlen+1), sizeof(int));
    int* F = (int*)calloc((qlen+1) * (rlen+1), sizeof(int));
    #define IDX(i,j) ((i)*(rlen+1)+(j))

    int max_score = 0, max_i = 0, max_j = 0;

    for (int i = 1; i <= qlen; i++) {
        /* Quality-adjusted score */
        float q_factor = 1.0f;
        if (quality) {
            int phred = quality[i-1] - 33;  /* ASCII to Phred */
            if (phred < 0) phred = 0;
            if (phred > 40) phred = 40;
            /* Probability of error = 10^(-Q/10) */
            /* For Q=30: p_error = 0.001, q_factor = 0.999 */
            /* For Q=10: p_error = 0.1, q_factor = 0.9 */
            /* For Q=2:  p_error = 0.63, q_factor = 0.37 */
            float p_correct = 1.0f - powf(10.0f, -(float)phred / 10.0f);
            q_factor = p_correct;
        }

        for (int j = 1; j <= rlen; j++) {
            int s;
            if (query[i-1] == ref[j-1]) {
                s = (int)(match_score * q_factor);
            } else {
                s = (int)(mismatch_score * q_factor);
            }

            E[IDX(i,j)] = (H[IDX(i,j-1)] - gap_open > E[IDX(i,j-1)] - gap_extend) ?
                            H[IDX(i,j-1)] - gap_open : E[IDX(i,j-1)] - gap_extend;
            F[IDX(i,j)] = (H[IDX(i-1,j)] - gap_open > F[IDX(i-1,j)] - gap_extend) ?
                            H[IDX(i-1,j)] - gap_open : F[IDX(i-1,j)] - gap_extend;

            int h = H[IDX(i-1,j-1)] + s;
            if (E[IDX(i,j)] > h) h = E[IDX(i,j)];
            if (F[IDX(i,j)] > h) h = F[IDX(i,j)];
            if (h < 0) h = 0;
            H[IDX(i,j)] = h;

            if (h > max_score) { max_score = h; max_i = i; max_j = j; }
        }
    }

    result->score = max_score;
    result->query_end = max_i - 1;
    result->ref_end = max_j - 1;
    result->cigar[0] = '\0';
    result->cigar_len = 0;

    free(H); free(E); free(F);
    #undef IDX
}
