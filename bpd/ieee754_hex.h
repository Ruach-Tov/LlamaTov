// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/*
 * ieee754_hex.h — Canonical IEEE 754 hex float representation.
 *
 * All backends use this format for human-readable float output.
 * Eliminates printf-style representation ambiguity entirely.
 *
 * Format: 8 hex chars for F32 (little-endian byte order as hex).
 * Example: 0.1f → "3dcccccd"
 *
 * Usage:
 *   char hex[9];
 *   f32_to_hex(my_float, hex);
 *   printf("%s\n", hex);
 */
#ifndef IEEE754_HEX_H
#define IEEE754_HEX_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>

static inline void f32_to_hex(float val, char out[9]) {
    uint32_t bits;
    memcpy(&bits, &val, 4);
    snprintf(out, 9, "%08x", bits);
}

static inline float hex_to_f32(const char hex[9]) {
    uint32_t bits;
    sscanf(hex, "%08x", &bits);
    float val;
    memcpy(&val, &bits, 4);
    return val;
}

static inline void f64_to_hex(double val, char out[17]) {
    uint64_t bits;
    memcpy(&bits, &val, 8);
    snprintf(out, 17, "%016lx", (unsigned long)bits);
}

/* Print N floats as canonical hex, with optional decimal annotation */
static inline void dump_f32_hex(const char *name, const float *arr, int n, int max_show) {
    char hex[9];
    printf("  %s: %d elements\n", name, n);
    int show = (max_show > 0 && max_show < n) ? max_show : n;
    for (int i = 0; i < show; i++) {
        f32_to_hex(arr[i], hex);
        printf("    [%4d] %s  (%+.8e)\n", i, hex, arr[i]);
    }
    if (show < n) printf("    ... (%d more)\n", n - show);
}

/* Compare two F32 arrays element-wise, report in canonical hex */
static inline int compare_f32_hex(const char *name, 
                                   const float *gpu, const float *cpu, 
                                   int n, float atol) {
    char g_hex[9], c_hex[9];
    int mismatches = 0;
    float max_diff = 0;
    int worst_idx = 0;
    
    for (int i = 0; i < n; i++) {
        float diff = gpu[i] - cpu[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) { max_diff = diff; worst_idx = i; }
        if (diff > atol) mismatches++;
    }
    
    printf("  %s: %d elements, max_diff=%.2e %s\n", 
           name, n, max_diff, mismatches ? "MISMATCH" : "MATCH");
    
    if (max_diff > 0) {
        /* Show worst 5 */
        printf("    worst at [%d]:\n", worst_idx);
        f32_to_hex(gpu[worst_idx], g_hex);
        f32_to_hex(cpu[worst_idx], c_hex);
        uint32_t g_bits, c_bits;
        memcpy(&g_bits, &gpu[worst_idx], 4);
        memcpy(&c_bits, &cpu[worst_idx], 4);
        printf("      gpu=%s  cpu=%s  xor=%08x\n", g_hex, c_hex, g_bits ^ c_bits);
    } else {
        printf("    BIT-IDENTICAL\n");
    }
    
    return mismatches == 0;
}

#endif /* IEEE754_HEX_H */
