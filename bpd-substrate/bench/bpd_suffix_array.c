// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_suffix_array.c — Suffix array + LCP array construction (CPU reference)
 *
 * Suffix array: sorted array of all suffix start positions.
 * LCP array: longest common prefix between adjacent suffixes in SA.
 *
 * Together these give the same information as a suffix tree
 * but in a flat, GPU-friendly representation.
 *
 * Algorithm: naive O(n log^2 n) suffix array (good enough for reference).
 * Kasai's O(n) LCP construction from SA and inverse SA.
 *
 * Build: gcc -O2 -shared -fPIC -o build/bpd_sa.so bench/bpd_suffix_array.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ================================================================
 * Suffix Array — naive construction via qsort
 * For reference/verification. GPU version will use radix sort.
 * ================================================================ */

static const char* g_text;
static int g_len;

static int suffix_compare(const void* a, const void* b) {
    int i = *(const int*)a;
    int j = *(const int*)b;
    return strcmp(g_text + i, g_text + j);
}

/* Build suffix array for text[0..n-1].
 * SA must be pre-allocated with n ints.
 * Appends a sentinel '\0' conceptually. */
void bpd_build_suffix_array(const char* text, int n, int* SA) {
    g_text = text;
    g_len = n;
    for (int i = 0; i < n; i++) SA[i] = i;
    qsort(SA, n, sizeof(int), suffix_compare);
}

/* ================================================================
 * LCP Array — Kasai's algorithm
 * Given text and SA, compute LCP[i] = length of longest common
 * prefix between text[SA[i]] and text[SA[i-1]].
 * LCP[0] = 0 by convention.
 * ================================================================ */

void bpd_build_lcp_array(const char* text, int n, const int* SA, int* LCP) {
    /* Build inverse suffix array */
    int* rank = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) rank[SA[i]] = i;

    int k = 0;
    LCP[0] = 0;

    for (int i = 0; i < n; i++) {
        if (rank[i] == 0) { k = 0; continue; }
        int j = SA[rank[i] - 1];
        while (i + k < n && j + k < n && text[i + k] == text[j + k]) k++;
        LCP[rank[i]] = k;
        if (k > 0) k--;
    }

    free(rank);
}

/* ================================================================
 * Motif candidate enumeration from SA + LCP
 * Find all substrings of length >= min_len that occur >= min_count times.
 * These are the candidates for motif discovery.
 * ================================================================ */

typedef struct {
    int sa_start;    /* start index in SA */
    int sa_end;      /* end index in SA (exclusive) */
    int length;      /* length of the repeated substring */
    int count;       /* number of occurrences */
} motif_candidate_t;

/* Find repeated substrings using LCP intervals.
 * Returns the number of candidates found.
 * candidates must be pre-allocated (max_candidates). */
int bpd_find_repeat_candidates(
    const char* text, int n,
    const int* SA, const int* LCP,
    int min_len, int min_count,
    motif_candidate_t* candidates, int max_candidates)
{
    int found = 0;

    /* Simple approach: scan LCP array for runs >= min_len */
    int i = 1;
    while (i < n && found < max_candidates) {
        if (LCP[i] >= min_len) {
            /* Start of a repeat block */
            int start = i - 1;
            int min_lcp = LCP[i];
            while (i < n && LCP[i] >= min_len) {
                if (LCP[i] < min_lcp) min_lcp = LCP[i];
                i++;
            }
            int count = i - start;
            if (count >= min_count) {
                candidates[found].sa_start = start;
                candidates[found].sa_end = i;
                candidates[found].length = min_lcp;
                candidates[found].count = count;
                found++;
            }
        } else {
            i++;
        }
    }
    return found;
}

/* ================================================================
 * Convenience: extract the actual substring for a candidate
 * ================================================================ */

void bpd_get_candidate_string(
    const char* text, const int* SA,
    const motif_candidate_t* cand,
    char* out, int max_out)
{
    int pos = SA[cand->sa_start];
    int len = cand->length;
    if (len >= max_out) len = max_out - 1;
    memcpy(out, text + pos, len);
    out[len] = '\0';
}
