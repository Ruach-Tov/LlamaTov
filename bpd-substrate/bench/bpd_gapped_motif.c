// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_gapped_motif.c — Gapped motif extension via Smith-Waterman
 *
 * Takes ungapped motif candidates from Gibbs sampling and extends
 * them with affine gaps using our verified Smith-Waterman kernel.
 *
 * The pipeline:
 *   1. Gibbs discovers ungapped motif consensus + positions
 *   2. Extract consensus from PWM (most probable base at each position)
 *   3. For each sequence, SW-align the consensus against a window
 *      around the Gibbs-predicted position
 *   4. The SW score with gaps > ungapped PWM score = evidence of gapped motif
 *   5. Collect CIGAR strings — gaps in the alignment reveal the motif's gap structure
 *
 * Build: gcc -O2 -shared -fPIC -o build/bpd_gapped_motif.so \
 *          bench/bpd_gapped_motif.c bench/bpd_smith_waterman.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Import SW from bpd_smith_waterman.c */
typedef struct {
    int score;
    int query_end;
    int ref_end;
    int cigar_len;
    char cigar[4096];
} sw_result_t;

extern void bpd_smith_waterman_cpu(
    const char* query, int qlen,
    const char* ref, int rlen,
    int match, int mismatch,
    int gap_open, int gap_extend,
    sw_result_t* result);

/* ================================================================
 * Extract consensus motif from PWM (most probable base per position)
 * ================================================================ */
void bpd_pwm_to_consensus(
    const float* pwm,   /* motif_width × 4, log-odds */
    int motif_width,
    char* consensus)     /* output: motif_width + 1 chars */
{
    const char bases[] = "ACGT";
    for (int j = 0; j < motif_width; j++) {
        int best_b = 0;
        float best_score = pwm[j * 4];
        for (int b = 1; b < 4; b++) {
            if (pwm[j * 4 + b] > best_score) {
                best_score = pwm[j * 4 + b];
                best_b = b;
            }
        }
        consensus[j] = bases[best_b];
    }
    consensus[motif_width] = '\0';
}

/* ================================================================
 * Gapped motif result for one sequence
 * ================================================================ */
typedef struct {
    int seq_idx;
    int ungapped_pos;       /* position from Gibbs */
    int ungapped_score;     /* PWM score at that position */
    int gapped_score;       /* SW score with gaps */
    int gapped_query_end;
    int gapped_ref_end;
    char cigar[4096];       /* alignment with gaps */
    int has_gaps;           /* 1 if CIGAR contains I or D */
} gapped_motif_result_t;

/* ================================================================
 * Extend ungapped motif with gaps for all sequences
 * ================================================================ */
int bpd_extend_with_gaps(
    const char** sequences, int n_seqs, const int* seq_lengths,
    const int* ungapped_positions,
    const float* pwm, int motif_width,
    int window_margin,      /* how far around Gibbs position to search */
    int match_score, int mismatch_score,
    int gap_open, int gap_extend,
    gapped_motif_result_t* results)  /* output: n_seqs results */
{
    /* Extract consensus from PWM */
    char* consensus = (char*)malloc(motif_width + 1);
    bpd_pwm_to_consensus(pwm, motif_width, consensus);

    int n_gapped = 0;

    for (int s = 0; s < n_seqs; s++) {
        int pos = ungapped_positions[s];
        int seq_len = seq_lengths[s];

        /* Define search window around Gibbs position */
        int win_start = pos - window_margin;
        if (win_start < 0) win_start = 0;
        int win_end = pos + motif_width + window_margin;
        if (win_end > seq_len) win_end = seq_len;
        int win_len = win_end - win_start;

        /* SW align consensus against window */
        sw_result_t sw;
        bpd_smith_waterman_cpu(
            consensus, motif_width,
            sequences[s] + win_start, win_len,
            match_score, mismatch_score,
            gap_open, gap_extend,
            &sw);

        /* Compute ungapped PWM score for comparison */
        int base_map[256] = {0};
        base_map['A'] = base_map['a'] = 0;
        base_map['C'] = base_map['c'] = 1;
        base_map['G'] = base_map['g'] = 2;
        base_map['T'] = base_map['t'] = 3;

        float pwm_score = 0;
        for (int j = 0; j < motif_width && pos + j < seq_len; j++) {
            int b = base_map[(unsigned char)sequences[s][pos + j]];
            pwm_score += pwm[j * 4 + b];
        }

        /* Check if CIGAR has gaps */
        int has_gaps = 0;
        for (int c = 0; sw.cigar[c]; c++) {
            if (sw.cigar[c] == 'I' || sw.cigar[c] == 'D') {
                has_gaps = 1;
                break;
            }
        }

        results[s].seq_idx = s;
        results[s].ungapped_pos = pos;
        results[s].ungapped_score = (int)(pwm_score * 100);  /* scale for int */
        results[s].gapped_score = sw.score;
        results[s].gapped_query_end = win_start + sw.query_end;
        results[s].gapped_ref_end = win_start + sw.ref_end;
        strncpy(results[s].cigar, sw.cigar, 4095);
        results[s].has_gaps = has_gaps;

        if (has_gaps) n_gapped++;
    }

    free(consensus);
    return n_gapped;
}
