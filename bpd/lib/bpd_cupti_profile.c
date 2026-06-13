// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_cupti_profile.c — CUPTI PC sampling for warp stall analysis
 *
 * Uses the CUPTI Activity API (NOT the Profiler API) for PC sampling.
 * The Activity API works on Pascal (cc 6.1) unlike the Profiler API (cc >= 7.0).
 *
 * PC sampling periodically snapshots which instruction each warp is executing
 * and WHY it is stalled. This gives us per-kernel stall reason distributions.
 *
 * Build: gcc -O2 -shared -fPIC -o build/bpd_cupti_profile.so \
 *          lib/bpd_cupti_profile.c -I$CUPTI_INC -L$CUPTI_LIB -lcupti -lcuda
 *
 * For SWI-Prolog PLF integration: this file provides the C functions
 * that cupti_bridge.c will wrap with PL_register_foreign().
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* Stall reason counters */
typedef struct {
    uint64_t none;
    uint64_t inst_fetch;
    uint64_t exec_dependency;
    uint64_t memory_dependency;
    uint64_t texture;
    uint64_t sync;
    uint64_t constant_memory;
    uint64_t pipe_busy;
    uint64_t memory_throttle;
    uint64_t not_selected;
    uint64_t other;
    uint64_t sleeping;
    uint64_t total_samples;
} stall_counters_t;

/* Global counters — accumulated by the activity callback */
static stall_counters_t g_stalls;
static int g_profiling_enabled = 0;

/* CUPTI error checking */
#define CUPTI_CHECK(call) do { \
    CUptiResult _status = call; \
    if (_status != CUPTI_SUCCESS) { \
        const char* _errstr; \
        cuptiGetResultString(_status, &_errstr); \
        fprintf(stderr, "CUPTI error: %s at %s:%d\n", _errstr, __FILE__, __LINE__); \
        return -1; \
    } \
} while(0)

/* Activity buffer management */
#define BUF_SIZE (32 * 1024)

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *buffer = (uint8_t*)malloc(BUF_SIZE);
    *size = BUF_SIZE;
    *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                                      uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    
    while (1) {
        CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if (status != CUPTI_SUCCESS) break;
        
        if (record->kind == CUPTI_ACTIVITY_KIND_PC_SAMPLING) {
            CUpti_ActivityPCSampling3 *pcRecord = (CUpti_ActivityPCSampling3*)record;
            
            g_stalls.total_samples += pcRecord->samples;
            
            switch (pcRecord->stallReason) {
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE:
                    g_stalls.none += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH:
                    g_stalls.inst_fetch += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY:
                    g_stalls.exec_dependency += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY:
                    g_stalls.memory_dependency += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE:
                    g_stalls.texture += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC:
                    g_stalls.sync += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY:
                    g_stalls.constant_memory += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY:
                    g_stalls.pipe_busy += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE:
                    g_stalls.memory_throttle += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED:
                    g_stalls.not_selected += pcRecord->samples; break;
                case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING:
                    g_stalls.sleeping += pcRecord->samples; break;
                default:
                    g_stalls.other += pcRecord->samples; break;
            }
        }
    }
    
    free(buffer);
}

/* ================================================================
 * Public API
 * ================================================================ */

/* Initialize CUPTI PC sampling */
int bpd_cupti_init(void) {
    memset(&g_stalls, 0, sizeof(g_stalls));
    
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
    
    /* Configure sampling period */
    CUpti_ActivityPCSamplingConfig config;
    config.size = sizeof(config);
    config.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
    /* Need to get the CUcontext */
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    config.samplingPeriod2 = 0;
    
    g_profiling_enabled = 1;
    return 0;
}

/* Flush and collect all PC sampling data */
int bpd_cupti_flush(void) {
    CUPTI_CHECK(cuptiActivityFlushAll(0));
    return 0;
}

/* Get stall counters */
int bpd_cupti_get_stalls(stall_counters_t* out) {
    *out = g_stalls;
    return 0;
}

/* Reset counters for a new profiling run */
void bpd_cupti_reset(void) {
    memset(&g_stalls, 0, sizeof(g_stalls));
}

/* Disable profiling */
int bpd_cupti_shutdown(void) {
    CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
    g_profiling_enabled = 0;
    return 0;
}

/* Print stall report */
void bpd_cupti_print_report(void) {
    uint64_t total = g_stalls.total_samples;
    if (total == 0) {
        printf("No PC sampling data collected.\n");
        return;
    }
    
    printf("=== WARP STALL ANALYSIS (PC Sampling) ===\n");
    printf("Total samples: %lu\n\n", total);
    printf("%-30s %8s %6s\n", "Stall Reason", "Samples", "Pct");
    printf("%-30s %8s %6s\n", "------------------------------", "--------", "------");
    
    #define PRINT_STALL(name, field) \
        if (g_stalls.field > 0) \
            printf("%-30s %8lu %5.1f%%\n", name, g_stalls.field, 100.0 * g_stalls.field / total)
    
    PRINT_STALL("No stall (issuing)", none);
    PRINT_STALL("Instruction fetch", inst_fetch);
    PRINT_STALL("Execution dependency", exec_dependency);
    PRINT_STALL("Memory dependency", memory_dependency);
    PRINT_STALL("Texture", texture);
    PRINT_STALL("Synchronization", sync);
    PRINT_STALL("Constant memory", constant_memory);
    PRINT_STALL("Pipe busy", pipe_busy);
    PRINT_STALL("Memory throttle", memory_throttle);
    PRINT_STALL("Not selected", not_selected);
    PRINT_STALL("Sleeping", sleeping);
    PRINT_STALL("Other", other);
    
    #undef PRINT_STALL
    
    printf("\n=== OPTIMIZATION SUGGESTIONS ===\n");
    float mem_pct = 100.0 * g_stalls.memory_dependency / total;
    float exec_pct = 100.0 * g_stalls.exec_dependency / total;
    float sync_pct = 100.0 * g_stalls.sync / total;
    float throttle_pct = 100.0 * g_stalls.memory_throttle / total;
    
    if (mem_pct > 30)
        printf("  → WARP SHUFFLE: %.1f%% memory dependency stalls — replace global reads with __shfl_up_sync()\n", mem_pct);
    if (mem_pct > 15 && mem_pct <= 30)
        printf("  → SHARED MEMORY: %.1f%% memory dependency — move hot data to shared memory\n", mem_pct);
    if (exec_pct > 20)
        printf("  → ILP: %.1f%% execution dependency — increase instruction-level parallelism\n", exec_pct);
    if (sync_pct > 15)
        printf("  → REDUCE BARRIERS: %.1f%% sync stalls — fewer __syncthreads() calls\n", sync_pct);
    if (throttle_pct > 10)
        printf("  → MEMORY COALESCING: %.1f%% memory throttle — improve access patterns\n", throttle_pct);
    if (mem_pct <= 15 && exec_pct <= 20 && sync_pct <= 15)
        printf("  → KERNEL IS WELL-OPTIMIZED: no dominant stall reason > 15%%\n");
}
