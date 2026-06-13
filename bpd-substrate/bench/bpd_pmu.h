// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_pmu.h — C-callable CPU PMU interface.
 *
 * Counter layout (indices into the out[] array):
 *   [0] cycles
 *   [1] instructions  
 *   [2] cache_misses (hardware)
 *   [3] branch_misses
 *   [4] sse_packed_single
 *   [5] sse_scalar_single
 *   [6] avx_256_packed
 *
 * Usage:
 *   #include "bpd_pmu.h"
 *   bpd_pmu_open();
 *   bpd_pmu_start();
 *   // ... code to profile ...
 *   int64_t c[7]; bpd_pmu_read(c);
 *   printf("cycles=%lld insns=%lld IPC=%.2f\n", c[0], c[1], (double)c[1]/c[0]);
 *   bpd_pmu_stop();
 *   bpd_pmu_close();
 */
#ifndef BPD_PMU_H
#define BPD_PMU_H

#include <stdint.h>

#define BPD_PMU_CYCLES          0
#define BPD_PMU_INSTRUCTIONS    1
#define BPD_PMU_CACHE_MISSES    2
#define BPD_PMU_BRANCH_MISSES   3
#define BPD_PMU_SSE_PACKED      4
#define BPD_PMU_SSE_SCALAR      5
#define BPD_PMU_AVX_256         6
#define BPD_PMU_N_COUNTERS      7

void bpd_pmu_open(void);
void bpd_pmu_start(void);
void bpd_pmu_read(int64_t *out);  /* out must be int64_t[7] */
void bpd_pmu_stop(void);
void bpd_pmu_close(void);

#endif /* BPD_PMU_H */
