// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_pwm_score.c — Position Weight Matrix scoring (CPU reference)
 *
 * Scores all positions in a sequence against a PWM motif model.
 * This is the inner loop of Gibbs sampling — the computational bottleneck
 * that GPU acceleration actually helps with.
 *
 * A PWM is a W×4 matrix where W = motif width and 4 = {A,C,G,T}.
 * Score at position p = sum of PWM[j][base_at(p+j)] for j=0..W-1.
 *
 * Build: gcc -O2 -shared -fPIC -o build/bpd_pwm.so bench/bpd_pwm_score.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Base encoding: A=0, C=1, G=2, T=3 */
static inline int base_idx(char c) {
    switch(c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 0;
    }
}

/* ================================================================
 * Score all positions in a sequence against a PWM
 * PWM is row-major: pwm[position * 4 + base]
 * scores[i] = sum(pwm[j * 4 + base_at(seq[i+j])]) for j=0..width-1
 * ================================================================ */

void bpd_pwm_score_all(
    const char* seq, int seq_len,
    const float* pwm, int motif_width,
    float* scores)  /* output: seq_len - motif_width + 1 scores */
{
    int n_positions = seq_len - motif_width + 1;
    for (int i = 0; i < n_positions; i++) {
        float score = 0.0f;
        for (int j = 0; j < motif_width; j++) {
            int base = base_idx(seq[i + j]);
            score += pwm[j * 4 + base];
        }
        scores[i] = score;
    }
}

/* ================================================================
 * Build PWM from a set of aligned sequences (count matrix → log-odds)
 * seqs: array of seq_count pointers, each of length motif_width
 * background: {0.25, 0.25, 0.25, 0.25} for uniform
 * pseudocount: typically 0.1 to avoid log(0)
 * ================================================================ */

void bpd_build_pwm(
    const char** seqs, int seq_count, int motif_width,
    const float* background,  /* 4 floats */
    float pseudocount,
    float* pwm)  /* output: motif_width * 4 */
{
    for (int j = 0; j < motif_width; j++) {
        /* Count bases at position j */
        float counts[4] = {0, 0, 0, 0};
        for (int s = 0; s < seq_count; s++) {
            int b = base_idx(seqs[s][j]);
            counts[b] += 1.0f;
        }
        /* Convert to log-odds */
        float total = seq_count + 4 * pseudocount;
        for (int b = 0; b < 4; b++) {
            float freq = (counts[b] + pseudocount) / total;
            pwm[j * 4 + b] = logf(freq / background[b]);
        }
    }
}

/* ================================================================
 * Sample a position proportional to exp(score)
 * Returns index into scores array.
 * Uses the Gumbel-max trick for numerically stable sampling.
 * ================================================================ */

int bpd_sample_position(
    const float* scores, int n_positions,
    uint32_t* rng_state)  /* simple LCG state */
{
    /* Find max for numerical stability */
    float max_score = scores[0];
    for (int i = 1; i < n_positions; i++) {
        if (scores[i] > max_score) max_score = scores[i];
    }

    /* Compute cumulative exp(score - max) */
    float cum_sum = 0.0f;
    float* cum = (float*)malloc(n_positions * sizeof(float));
    for (int i = 0; i < n_positions; i++) {
        cum_sum += expf(scores[i] - max_score);
        cum[i] = cum_sum;
    }

    /* Sample uniform in [0, cum_sum) */
    *rng_state = *rng_state * 1664525u + 1013904223u;  /* LCG */
    float u = ((float)(*rng_state) / 4294967296.0f) * cum_sum;

    /* Binary search */
    int lo = 0, hi = n_positions - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cum[mid] < u) lo = mid + 1;
        else hi = mid;
    }

    free(cum);
    return lo;
}

/* ================================================================
 * One iteration of Gibbs sampling
 * Hold out sequence hold_out_idx, score all positions in it
 * against PWM built from remaining sequences, sample new position.
 * Returns the new position for the held-out sequence.
 * ================================================================ */

int bpd_gibbs_step(
    const char** sequences, int n_seqs, int* seq_lengths,
    int* motif_positions,  /* current motif position in each seq */
    int motif_width,
    int hold_out_idx,
    const float* background,
    float pseudocount,
    uint32_t* rng_state)
{
    /* Build PWM from all sequences except held-out */
    float* pwm = (float*)calloc(motif_width * 4, sizeof(float));
    
    /* Collect aligned motif instances */
    const char** motif_seqs = (const char**)malloc((n_seqs - 1) * sizeof(char*));
    char** motif_copies = (char**)malloc((n_seqs - 1) * sizeof(char*));
    int k = 0;
    for (int s = 0; s < n_seqs; s++) {
        if (s == hold_out_idx) continue;
        motif_copies[k] = (char*)malloc(motif_width + 1);
        memcpy(motif_copies[k], sequences[s] + motif_positions[s], motif_width);
        motif_copies[k][motif_width] = '\0';
        motif_seqs[k] = motif_copies[k];
        k++;
    }
    
    bpd_build_pwm(motif_seqs, n_seqs - 1, motif_width,
                   background, pseudocount, pwm);

    /* Score all positions in held-out sequence */
    int n_pos = seq_lengths[hold_out_idx] - motif_width + 1;
    float* scores = (float*)malloc(n_pos * sizeof(float));
    bpd_pwm_score_all(sequences[hold_out_idx], seq_lengths[hold_out_idx],
                       pwm, motif_width, scores);

    /* Sample new position */
    int new_pos = bpd_sample_position(scores, n_pos, rng_state);

    /* Cleanup */
    for (int i = 0; i < k; i++) free(motif_copies[i]);
    free(motif_copies);
    free(motif_seqs);
    free(pwm);
    free(scores);

    return new_pos;
}
