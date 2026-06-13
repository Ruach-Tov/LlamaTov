// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_pmu.c — C-callable CPU PMU interface.
 *
 * Generated from the same specification as the Prolog FLI bridge,
 * but projected as plain C functions instead of Prolog predicates.
 *
 * Interface:
 *   bpd_pmu_open()              — open 7 perf counters
 *   bpd_pmu_start()             — reset + enable all counters
 *   bpd_pmu_read(int64_t *out)  — read 7 counters into array
 *   bpd_pmu_stop()              — disable all counters
 *   bpd_pmu_close()             — close file descriptors
 *
 * Counter layout (indices into the out[] array):
 *   [0] cycles
 *   [1] instructions
 *   [2] cache_misses (L1)
 *   [3] branch_misses
 *   [4] sse_packed_single (raw 0x4010)
 *   [5] sse_scalar_single (raw 0x2010)
 *   [6] avx_256_packed (raw 0x0111)
 *
 * Build: gcc -O2 -shared -fPIC -o bpd_pmu.so bpd_pmu.c
 * Usage in C: #include "bpd_pmu.h"
 *             bpd_pmu_open(); bpd_pmu_start();
 *             ... your code ...
 *             int64_t counters[7]; bpd_pmu_read(counters);
 *             bpd_pmu_stop(); bpd_pmu_close();
 */
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

#define BPD_PMU_N_COUNTERS 7

/* Counter definitions: {perf_type, config} */
static const struct { uint32_t type; uint64_t config; } counter_defs[BPD_PMU_N_COUNTERS] = {
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},        /* [0] cycles */
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},      /* [1] instructions */
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},      /* [2] cache_misses */
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},     /* [3] branch_misses */
    {PERF_TYPE_RAW, 0x4010},  /* [4] FP_COMP_OPS_EXE.SSE_FP_PACKED_SINGLE */
    {PERF_TYPE_RAW, 0x2010},  /* [5] FP_COMP_OPS_EXE.SSE_FP_SCALAR_SINGLE */
    {PERF_TYPE_RAW, 0x0111},  /* [6] SIMD_FP_256.PACKED_SINGLE (AVX) */
};

static int fds[BPD_PMU_N_COUNTERS] = {-1,-1,-1,-1,-1,-1,-1};

static long perf_event_open(struct perf_event_attr *attr, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

void bpd_pmu_open(void) {
    struct perf_event_attr pe;
    for (int i = 0; i < BPD_PMU_N_COUNTERS; i++) {
        memset(&pe, 0, sizeof(pe));
        pe.type = counter_defs[i].type;
        pe.size = sizeof(pe);
        pe.config = counter_defs[i].config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        fds[i] = (int)perf_event_open(&pe, 0, -1, -1, 0);
    }
}

void bpd_pmu_start(void) {
    for (int i = 0; i < BPD_PMU_N_COUNTERS; i++) {
        if (fds[i] >= 0) {
            ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
            ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
        }
    }
}

void bpd_pmu_read(int64_t *out) {
    for (int i = 0; i < BPD_PMU_N_COUNTERS; i++) {
        if (fds[i] >= 0) {
            read(fds[i], &out[i], sizeof(int64_t));
        } else {
            out[i] = -1;
        }
    }
}

void bpd_pmu_stop(void) {
    for (int i = 0; i < BPD_PMU_N_COUNTERS; i++) {
        if (fds[i] >= 0) {
            ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
        }
    }
}

void bpd_pmu_close(void) {
    for (int i = 0; i < BPD_PMU_N_COUNTERS; i++) {
        if (fds[i] >= 0) {
            close(fds[i]);
            fds[i] = -1;
        }
    }
}
