// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <time.h>

static long perf_open(int type, int config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.type = type;
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    return syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
}

typedef struct {
    long long cycles, instructions, l1_misses, branch_misses;
    double wall_ms;
} profile_t;

typedef void (*gemm_fn)(const float*, const float*, float*, int, int, int);

profile_t do_profile(gemm_fn fn, const float* A, const float* B, float* C,
                     int M, int N, int K, int reps) {
    profile_t r = {0};
    int fd_cyc = perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    int fd_ins = perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    int fd_l1  = perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    int fd_br  = perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

    if (fd_cyc < 0) { r.cycles = -1; return r; }

    fn(A, B, C, M, N, K); /* warmup */

    ioctl(fd_cyc, PERF_EVENT_IOC_RESET, 0); ioctl(fd_cyc, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_ins, PERF_EVENT_IOC_RESET, 0); ioctl(fd_ins, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_l1,  PERF_EVENT_IOC_RESET, 0); ioctl(fd_l1,  PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_br,  PERF_EVENT_IOC_RESET, 0); ioctl(fd_br,  PERF_EVENT_IOC_ENABLE, 0);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < reps; i++) fn(A, B, C, M, N, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    ioctl(fd_cyc, PERF_EVENT_IOC_DISABLE, 0); read(fd_cyc, &r.cycles, 8);
    ioctl(fd_ins, PERF_EVENT_IOC_DISABLE, 0); read(fd_ins, &r.instructions, 8);
    ioctl(fd_l1,  PERF_EVENT_IOC_DISABLE, 0); read(fd_l1,  &r.l1_misses, 8);
    ioctl(fd_br,  PERF_EVENT_IOC_DISABLE, 0); read(fd_br,  &r.branch_misses, 8);

    close(fd_cyc); close(fd_ins); close(fd_l1); close(fd_br);

    r.wall_ms = ((t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec)) / 1e6 / reps;
    r.cycles /= reps; r.instructions /= reps;
    r.l1_misses /= reps; r.branch_misses /= reps;
    return r;
}

extern void bpd_gemm_v2_full(const float*, const float*, float*, int, int, int);
extern void bpd_mm_cpu_avx1(const float*, const float*, float*, int, int, int);
extern void bpd_mm_cpu_avx1_v2(const float*, const float*, float*, int, int, int);

int main() {
    int M = 1024, N = 1024, K = 1024, REPS = 5;
    float* A = malloc(M*K*sizeof(float));
    float* B = malloc(K*N*sizeof(float));
    float* C = calloc(M*N, sizeof(float));

    srand(42);
    for (int i = 0; i < M*K; i++) A[i] = (float)rand()/RAND_MAX - 0.5f;
    for (int i = 0; i < K*N; i++) B[i] = (float)rand()/RAND_MAX - 0.5f;

    printf("=== CPU Profile: GEMM Tile Variants (1024x1024) ===\n\n");
    printf("%-25s %7s %7s %5s %10s %10s\n",
           "Variant", "ms", "GFLOPS", "IPC", "L1 miss", "Br miss");
    printf("-------------------------------------------------------------------\n");

    struct { const char* name; gemm_fn fn; } v[] = {
        {"avx1 (MR=16,NR=4)", bpd_mm_cpu_avx1},
        {"avx1_v2 (MR=4,NR=16)", bpd_mm_cpu_avx1_v2},
        {"v2_full (K-blocked)", bpd_gemm_v2_full},
    };

    for (int i = 0; i < 3; i++) {
        profile_t p = do_profile(v[i].fn, A, B, C, M, N, K, REPS);
        if (p.cycles < 0) { printf("%-25s  perf failed\n", v[i].name); continue; }
        double gf = 2.0*M*N*K / p.wall_ms / 1e6;
        double ipc = (double)p.instructions / p.cycles;
        printf("%-25s %5.1fms %5.1fGF %4.2f %10lld %10lld\n",
               v[i].name, p.wall_ms, gf, ipc, p.l1_misses, p.branch_misses);
    }

    printf("\nDIAGNOSTIC:\n");
    printf("  IPC < 1.0 = pipeline stalls (register spills, cache misses)\n");
    printf("  High L1 miss = working set exceeds L1 (tile too large)\n");
    printf("  Peak IPC on Ivy Bridge = ~4.0 (2 ALU + 1 load + 1 store)\n");

    free(A); free(B); free(C);
    return 0;
}
