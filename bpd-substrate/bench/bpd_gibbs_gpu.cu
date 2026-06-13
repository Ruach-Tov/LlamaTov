// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_gibbs_gpu.cu — Collapsed Gibbs sampling with DP-scheduled inner loop
 *
 * KEY INSIGHT: Leave-one-out PWM is a subtraction from a precomputed total.
 * All N held-out sequences can be processed simultaneously on GPU.
 *
 * Instead of N sequential iterations per Gibbs round:
 *   1. Compute total_counts once (parallel over positions)
 *   2. For ALL N sequences simultaneously:
 *      a. counts_without_k = total_counts - counts[k]
 *      b. Convert to log-odds PWM
 *      c. Score all positions in sequence k
 *      d. Sample new position (parallel reduction)
 *
 * Build: nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_61 \
 *          -o build/bpd_gibbs_gpu.so bench/bpd_gibbs_gpu.cu -Wno-deprecated-gpu-targets
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MAX_MOTIF_WIDTH 32
#define NUM_BASES 4

/* ================================================================
 * Kernel 1: Build total count matrix from all current motif positions
 * One thread per (position_in_motif × base) = motif_width × 4
 * ================================================================ */
__global__ void k_build_total_counts(
    const char* __restrict__ sequences,   /* concatenated sequences */
    const int* __restrict__ seq_offsets,   /* start offset of each seq */
    const int* __restrict__ motif_pos,     /* current motif position per seq */
    int n_seqs,
    int motif_width,
    float* __restrict__ total_counts)     /* motif_width × 4 output */
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = idx / NUM_BASES;  /* position in motif */
    int b = idx % NUM_BASES;  /* base index */

    if (j >= motif_width) return;

    float count = 0.0f;
    for (int s = 0; s < n_seqs; s++) {
        int pos = seq_offsets[s] + motif_pos[s] + j;
        char base = sequences[pos];
        int base_val;
        switch(base) {
            case 'A': case 'a': base_val = 0; break;
            case 'C': case 'c': base_val = 1; break;
            case 'G': case 'g': base_val = 2; break;
            case 'T': case 't': base_val = 3; break;
            default: base_val = 0;
        }
        if (base_val == b) count += 1.0f;
    }
    total_counts[j * NUM_BASES + b] = count;
}

/* ================================================================
 * Kernel 2: Score all positions in ALL sequences simultaneously
 * 
 * For each (sequence k, position p):
 *   1. Compute leave-one-out counts: total - contribution of k
 *   2. Convert to log-odds: log((count + pseudo) / (N-1 + 4*pseudo) / bg)
 *   3. Score: sum of log-odds at each motif position
 *
 * Grid: one thread per (sequence, position) pair
 * ================================================================ */
__global__ void k_score_all_leave_one_out(
    const char* __restrict__ sequences,
    const int* __restrict__ seq_offsets,
    const int* __restrict__ seq_lengths,
    const int* __restrict__ motif_pos,
    const float* __restrict__ total_counts,  /* precomputed total */
    int n_seqs,
    int motif_width,
    float pseudocount,
    int max_seq_len,
    float* __restrict__ all_scores)  /* n_seqs × max_positions output */
{
    int seq_idx = blockIdx.y;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (seq_idx >= n_seqs) return;

    int seq_len = seq_lengths[seq_idx];
    int n_positions = seq_len - motif_width + 1;
    if (pos >= n_positions) return;

    int seq_start = seq_offsets[seq_idx];

    /* Compute the motif contribution of THIS sequence (to subtract) */
    /* Then for each position j in the motif, compute leave-one-out log-odds
     * and accumulate the score */
    float score = 0.0f;
    float denom = (n_seqs - 1) + NUM_BASES * pseudocount;

    for (int j = 0; j < motif_width; j++) {
        /* Base at candidate position */
        char cand_base = sequences[seq_start + pos + j];
        int cand_b;
        switch(cand_base) {
            case 'A': case 'a': cand_b = 0; break;
            case 'C': case 'c': cand_b = 1; break;
            case 'G': case 'g': cand_b = 2; break;
            case 'T': case 't': cand_b = 3; break;
            default: cand_b = 0;
        }

        /* Base that sequence k contributes at motif position j */
        char my_base = sequences[seq_start + motif_pos[seq_idx] + j];
        int my_b;
        switch(my_base) {
            case 'A': case 'a': my_b = 0; break;
            case 'C': case 'c': my_b = 1; break;
            case 'G': case 'g': my_b = 2; break;
            case 'T': case 't': my_b = 3; break;
            default: my_b = 0;
        }

        /* Leave-one-out count for the candidate base */
        float count = total_counts[j * NUM_BASES + cand_b];
        if (cand_b == my_b) count -= 1.0f;  /* subtract this seq's contribution */
        
        float freq = (count + pseudocount) / denom;
        score += logf(freq / 0.25f);  /* log-odds vs uniform background */
    }

    all_scores[seq_idx * max_seq_len + pos] = score;
}

/* ================================================================
 * Kernel 3: Find max score per sequence (for stable sampling)
 * Parallel reduction over positions for each sequence.
 * ================================================================ */
__global__ void k_find_max_per_seq(
    const float* __restrict__ all_scores,
    int n_seqs,
    int max_seq_len,
    const int* __restrict__ seq_lengths,
    int motif_width,
    float* __restrict__ max_scores)  /* one per sequence */
{
    int seq_idx = blockIdx.x;
    if (seq_idx >= n_seqs) return;

    int n_pos = seq_lengths[seq_idx] - motif_width + 1;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float my_max = -FLT_MAX;

    /* Grid-stride loop */
    for (int i = tid; i < n_pos; i += blockDim.x) {
        float val = all_scores[seq_idx * max_seq_len + i];
        if (val > my_max) my_max = val;
    }
    sdata[tid] = my_max;
    __syncthreads();

    /* Tree reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) max_scores[seq_idx] = sdata[0];
}

/* ================================================================
 * Kernel 4: Sample positions proportional to exp(score - max)
 * Compute cumulative sum, then binary search with random threshold.
 * One block per sequence.
 * ================================================================ */
__global__ void k_sample_positions(
    const float* __restrict__ all_scores,
    const float* __restrict__ max_scores,
    int n_seqs,
    int max_seq_len,
    const int* __restrict__ seq_lengths,
    int motif_width,
    uint32_t* __restrict__ rng_states,  /* one per sequence */
    int* __restrict__ new_positions)    /* output: one per sequence */
{
    int seq_idx = blockIdx.x;
    if (seq_idx >= n_seqs) return;
    if (threadIdx.x != 0) return;  /* single-threaded per sequence for sampling */

    int n_pos = seq_lengths[seq_idx] - motif_width + 1;
    float max_s = max_scores[seq_idx];

    /* Compute cumulative exp(score - max) */
    float cum_sum = 0.0f;
    for (int i = 0; i < n_pos; i++) {
        cum_sum += expf(all_scores[seq_idx * max_seq_len + i] - max_s);
    }

    /* Sample */
    uint32_t rng = rng_states[seq_idx];
    rng = rng * 1664525u + 1013904223u;
    rng_states[seq_idx] = rng;
    float u = ((float)rng / 4294967296.0f) * cum_sum;

    /* Linear scan to find position */
    float running = 0.0f;
    int chosen = 0;
    for (int i = 0; i < n_pos; i++) {
        running += expf(all_scores[seq_idx * max_seq_len + i] - max_s);
        if (running >= u) { chosen = i; break; }
    }

    new_positions[seq_idx] = chosen;
}

/* ================================================================
 * Host API: Run one full Gibbs round (all N sequences updated)
 * ================================================================ */
extern "C" {

typedef struct {
    int n_seqs;
    int motif_width;
    int max_seq_len;
    float pseudocount;
    int block_size;
} gibbs_config_t;

void bpd_gibbs_round_gpu(
    const char* concat_seqs,    /* all sequences concatenated */
    int total_chars,
    const int* seq_offsets,     /* start of each sequence */
    const int* seq_lengths,     /* length of each sequence */
    int* motif_positions,       /* IN/OUT: current positions */
    uint32_t* rng_states,       /* IN/OUT: per-sequence RNG */
    const gibbs_config_t* config)
{
    int n = config->n_seqs;
    int W = config->motif_width;
    int max_len = config->max_seq_len;
    int bs = config->block_size;
    float pseudo = config->pseudocount;

    /* Device allocations */
    char* d_seqs;
    int *d_offsets, *d_lengths, *d_motif_pos, *d_new_pos;
    float *d_total_counts, *d_all_scores, *d_max_scores;
    uint32_t *d_rng;

    cudaMalloc(&d_seqs, total_chars);
    cudaMalloc(&d_offsets, n * sizeof(int));
    cudaMalloc(&d_lengths, n * sizeof(int));
    cudaMalloc(&d_motif_pos, n * sizeof(int));
    cudaMalloc(&d_new_pos, n * sizeof(int));
    cudaMalloc(&d_total_counts, W * NUM_BASES * sizeof(float));
    cudaMalloc(&d_all_scores, n * max_len * sizeof(float));
    cudaMalloc(&d_max_scores, n * sizeof(float));
    cudaMalloc(&d_rng, n * sizeof(uint32_t));

    cudaMemcpy(d_seqs, concat_seqs, total_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, seq_offsets, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, seq_lengths, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_motif_pos, motif_positions, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rng, rng_states, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Step 1: Build total count matrix */
    int count_threads = W * NUM_BASES;
    k_build_total_counts<<<1, count_threads>>>(
        d_seqs, d_offsets, d_motif_pos, n, W, d_total_counts);

    /* Step 2: Score all positions in all sequences (the big parallel step) */
    int max_positions = max_len - W + 1;
    dim3 grid2((max_positions + bs - 1) / bs, n);
    k_score_all_leave_one_out<<<grid2, bs>>>(
        d_seqs, d_offsets, d_lengths, d_motif_pos, d_total_counts,
        n, W, pseudo, max_len, d_all_scores);

    /* Step 3: Find max score per sequence */
    k_find_max_per_seq<<<n, bs, bs * sizeof(float)>>>(
        d_all_scores, n, max_len, d_lengths, W, d_max_scores);

    /* Step 4: Sample new positions */
    k_sample_positions<<<n, 1>>>(
        d_all_scores, d_max_scores, n, max_len, d_lengths, W,
        d_rng, d_new_pos);

    cudaDeviceSynchronize();

    /* Copy results back */
    cudaMemcpy(motif_positions, d_new_pos, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rng_states, d_rng, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /* Cleanup */
    cudaFree(d_seqs); cudaFree(d_offsets); cudaFree(d_lengths);
    cudaFree(d_motif_pos); cudaFree(d_new_pos);
    cudaFree(d_total_counts); cudaFree(d_all_scores); cudaFree(d_max_scores);
    cudaFree(d_rng);
}

/* Convenience: run multiple Gibbs rounds */
void bpd_gibbs_sampler_gpu(
    const char* concat_seqs, int total_chars,
    const int* seq_offsets, const int* seq_lengths,
    int* motif_positions, uint32_t* rng_states,
    const gibbs_config_t* config,
    int n_iterations)
{
    for (int iter = 0; iter < n_iterations; iter++) {
        bpd_gibbs_round_gpu(concat_seqs, total_chars,
                            seq_offsets, seq_lengths,
                            motif_positions, rng_states, config);
    }
}

} /* extern "C" */
