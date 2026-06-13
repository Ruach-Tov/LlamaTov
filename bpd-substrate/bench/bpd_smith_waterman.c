// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_smith_waterman.c — Smith-Waterman local sequence alignment (CPU)
 *
 * GPU kernel generation target for BPD substrate.
 * Reference implementation for bit-identity verification.
 *
 * Algorithm: Smith-Waterman with affine gap penalties
 *   H(i,j) = max(0,
 *                 H(i-1,j-1) + score(q[i], r[j]),
 *                 E(i,j),
 *                 F(i,j))
 *   E(i,j) = max(H(i,j-1) - gap_open, E(i,j-1) - gap_extend)
 *   F(i,j) = max(H(i-1,j) - gap_open, F(i-1,j) - gap_extend)
 *
 * Build: gcc -O2 -shared -fPIC -o build/bpd_sw.so bench/bpd_smith_waterman.c
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* DNA scoring: match=+2, mismatch=-1 */
static int dna_score(char a, char b) {
    return (a == b) ? 2 : -1;
}

/* Encode base to index: A=0, C=1, G=2, T=3 */
static int base_to_idx(char c) {
    switch(c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

/* BLOSUM62-style scoring for protein (simplified) */
static const int BLOSUM62[4][4] = {
    { 2, -1, -1, -1},  /* A vs A,C,G,T */
    {-1,  2, -1, -1},  /* C vs A,C,G,T */
    {-1, -1,  2, -1},  /* G vs A,C,G,T */
    {-1, -1, -1,  2},  /* T vs A,C,G,T */
};

typedef struct {
    int score;           /* best alignment score */
    int query_end;       /* end position in query */
    int ref_end;         /* end position in reference */
    int cigar_len;       /* length of CIGAR string */
    char cigar[4096];    /* CIGAR string (e.g., "8M2I3M1D5M") */
} sw_result_t;

/* Smith-Waterman with affine gap penalties
 * Returns the optimal local alignment score and position.
 */
void bpd_smith_waterman_cpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    sw_result_t* result)
{
    /* Allocate scoring matrices */
    int* H = (int*)calloc((qlen+1) * (rlen+1), sizeof(int));
    int* E = (int*)calloc((qlen+1) * (rlen+1), sizeof(int));
    int* F = (int*)calloc((qlen+1) * (rlen+1), sizeof(int));

    #define IDX(i,j) ((i)*(rlen+1)+(j))

    int max_score = 0, max_i = 0, max_j = 0;

    /* Fill scoring matrix */
    for (int i = 1; i <= qlen; i++) {
        for (int j = 1; j <= rlen; j++) {
            int s = (query[i-1] == ref[j-1]) ? match_score : mismatch_score;

            /* E: best score ending with gap in query (horizontal) */
            int e1 = H[IDX(i, j-1)] - gap_open;
            int e2 = E[IDX(i, j-1)] - gap_extend;
            E[IDX(i,j)] = (e1 > e2) ? e1 : e2;

            /* F: best score ending with gap in reference (vertical) */
            int f1 = H[IDX(i-1, j)] - gap_open;
            int f2 = F[IDX(i-1, j)] - gap_extend;
            F[IDX(i,j)] = (f1 > f2) ? f1 : f2;

            /* H: best score at (i,j) */
            int h = H[IDX(i-1, j-1)] + s;
            if (E[IDX(i,j)] > h) h = E[IDX(i,j)];
            if (F[IDX(i,j)] > h) h = F[IDX(i,j)];
            if (h < 0) h = 0;  /* local alignment: floor at 0 */
            H[IDX(i,j)] = h;

            if (h > max_score) {
                max_score = h;
                max_i = i;
                max_j = j;
            }
        }
    }

    result->score = max_score;
    result->query_end = max_i - 1;
    result->ref_end = max_j - 1;

    /* Traceback to produce CIGAR string */
    int ci = max_i, cj = max_j;
    int cigar_ops[8192]; /* 0=M, 1=I, 2=D */
    int n_ops = 0;

    while (ci > 0 && cj > 0 && H[IDX(ci, cj)] > 0) {
        int s = (query[ci-1] == ref[cj-1]) ? match_score : mismatch_score;

        if (H[IDX(ci, cj)] == H[IDX(ci-1, cj-1)] + s) {
            cigar_ops[n_ops++] = 0; /* M */
            ci--; cj--;
        } else if (H[IDX(ci, cj)] == E[IDX(ci, cj)]) {
            cigar_ops[n_ops++] = 2; /* D (gap in query) */
            cj--;
        } else {
            cigar_ops[n_ops++] = 1; /* I (gap in ref) */
            ci--;
        }
    }

    /* Reverse and run-length encode into CIGAR string */
    result->cigar[0] = '\0';
    if (n_ops > 0) {
        char* p = result->cigar;
        int k = n_ops - 1;
        while (k >= 0) {
            int op = cigar_ops[k];
            int count = 1;
            while (k > 0 && cigar_ops[k-1] == op) { count++; k--; }
            char c = (op == 0) ? 'M' : (op == 1) ? 'I' : 'D';
            p += sprintf(p, "%d%c", count, c);
            k--;
        }
        result->cigar_len = p - result->cigar;
    }

    free(H); free(E); free(F);
    #undef IDX
}

/* Simplified interface for ctypes: score only, no traceback */
int bpd_sw_score_cpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match, int mismatch,
    int gap_open, int gap_extend)
{
    sw_result_t result;
    bpd_smith_waterman_cpu(query, qlen, ref, rlen,
                            match, mismatch, gap_open, gap_extend,
                            &result);
    return result.score;
}
