// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* ggml_roofline_profile.c — Comprehensive GPU counter profiling for ggml matmul.
 *
 * Reads exact hardware counters from ALL event domains around ggml's
 * Q8_0 matmul to explain the 42-62% bandwidth utilization gap.
 *
 * Author: mavchin (2026-06-03)
 */

#include <cuda.h>
#include <cupti.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

static const char *events_to_collect[] = {
    /* Domain A: SM activity */
    "active_cycles_pm",
    "active_warps_pm",
    "elapsed_cycles_sm",
    "elapsed_cycles_pm",
    /* Domain B: DRAM */
    "fb_subp0_read_sectors",
    "fb_subp1_read_sectors",
    "fb_subp0_write_sectors",
    "fb_subp1_write_sectors",
    /* Domain C: Load/store instructions by width */
    "gld_inst_8bit",
    "gld_inst_16bit",
    "gld_inst_32bit",
    "gld_inst_64bit",
    "gld_inst_128bit",
    "gst_inst_32bit",
    /* Domain D: Warps + instructions */
    "warps_launched",
    "inst_executed",
    /* Domain E: L2 cache */
    "l2_subp0_read_sector_misses",
    "l2_subp1_read_sector_misses",
    "l2_subp0_read_tex_sector_queries",
    "l2_subp0_read_tex_hit_sectors",
};
#define N_EVENTS (sizeof(events_to_collect) / sizeof(events_to_collect[0]))

static void profile_matmul_counters(int M, int K, int N, const char *name) {
    cuInit(0);
    CUdevice cuDev; cuDeviceGet(&cuDev, 0);
    
    /* ggml setup */
    void *buf = aligned_alloc(64, 16*1024*1024);
    struct ggml_init_params params = {16*1024*1024, buf, true};
    struct ggml_context *ctx = ggml_init(params);
    
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor *c = ggml_mul_mat(ctx, a, b);
    
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
    
    size_t a_sz = ggml_nbytes(a), b_sz = ggml_nbytes(b);
    void *h_a = malloc(a_sz), *h_b = malloc(b_sz);
    memset(h_a, 0x42, a_sz);
    for (int i = 0; i < K*N; i++) ((float*)h_b)[i] = 0.01f;
    ggml_backend_tensor_set(a, h_a, 0, a_sz);
    ggml_backend_tensor_set(b, h_b, 0, b_sz);
    
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    
    /* Warmup */
    ggml_backend_graph_compute(cuda, graph);
    cudaDeviceSynchronize();
    
    /* Get the CUDA context ggml created */
    CUcontext cuCtx;
    cuCtxGetCurrent(&cuCtx);
    
    printf("\n=== %s [%d x %d] x %d (weight=%.1fMB) ===\n", name, M, K, N, a_sz/1e6);
    
    /* Collect each event */
    uint64_t values[N_EVENTS];
    for (int e = 0; e < (int)N_EVENTS; e++) {
        CUpti_EventID evId;
        CUptiResult r = cuptiEventGetIdFromName(cuDev, events_to_collect[e], &evId);
        if (r != CUPTI_SUCCESS) {
            printf("  %-35s (not found: %d)\n", events_to_collect[e], r);
            values[e] = 0;
            continue;
        }
        
        CUpti_EventGroup grp;
        r = cuptiEventGroupCreate(cuCtx, &grp, 0);
        if (r != CUPTI_SUCCESS) { values[e] = 0; continue; }
        
        r = cuptiEventGroupAddEvent(grp, evId);
        if (r != CUPTI_SUCCESS) { cuptiEventGroupDestroy(grp); values[e] = 0; continue; }
        
        uint32_t all = 1;
        cuptiEventGroupSetAttribute(grp, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all);
        
        cuptiEventGroupEnable(grp);
        ggml_backend_graph_compute(cuda, graph);
        cudaDeviceSynchronize();
        
        /* ReadAllEvents (ReadEvent returns DISABLED on Pascal) */
        uint64_t readVals[64]; CUpti_EventID readIds[64];
        size_t valsSz = sizeof(readVals), idsSz = sizeof(readIds);
        size_t numRead = 0;
        r = cuptiEventGroupReadAllEvents(grp, CUPTI_EVENT_READ_FLAG_NONE,
                                          &valsSz, readVals, &idsSz, readIds, &numRead);
        values[e] = (r == CUPTI_SUCCESS && numRead > 0) ? readVals[0] : 0;
        
        cuptiEventGroupDisable(grp);
        cuptiEventGroupDestroy(grp);
    }
    
    /* Print raw counters */
    printf("\n  RAW COUNTERS:\n");
    for (int e = 0; e < (int)N_EVENTS; e++)
        printf("    %-35s %12lu\n", events_to_collect[e], values[e]);
    
    /* Derived analysis */
    uint64_t fb_reads = values[4] + values[5];
    uint64_t fb_writes = values[6] + values[7];
    uint64_t gld_total = values[8]+values[9]+values[10]+values[11]+values[12];
    uint64_t active_cyc = values[0];
    uint64_t elapsed_cyc = values[2];
    uint64_t active_warps = values[1];
    uint64_t warps_launched = (N_EVENTS > 14) ? values[14] : 0;
    uint64_t inst_executed = (N_EVENTS > 15) ? values[15] : 0;
    uint64_t l2_misses = (N_EVENTS > 16) ? values[16] + values[17] : 0;
    uint64_t l2_queries = (N_EVENTS > 18) ? values[18] : 0;
    uint64_t l2_hits = (N_EVENTS > 19) ? values[19] : 0;
    
    printf("\n  DERIVED ANALYSIS:\n");
    printf("    DRAM read sectors:      %lu (%.2f MB)\n", fb_reads, fb_reads*32.0/1e6);
    printf("    DRAM write sectors:     %lu (%.2f MB)\n", fb_writes, fb_writes*32.0/1e6);
    printf("    Weight matrix:          %.2f MB\n", a_sz/1e6);
    printf("    DRAM/weight ratio:      %.2f (1.0=pure stream, <1=L2 absorbs)\n",
           a_sz > 0 ? (fb_reads*32.0) / a_sz : 0);
    
    if (gld_total > 0)
        printf("    Sectors per load:       %.3f\n", (double)fb_reads / gld_total);
    
    printf("    Load width breakdown:   8b=%lu 16b=%lu 32b=%lu 64b=%lu 128b=%lu\n",
           values[8], values[9], values[10], values[11], values[12]);
    
    if (elapsed_cyc > 0) {
        printf("    SM utilization:         %.1f%% (active/elapsed cycles)\n",
               100.0 * active_cyc / elapsed_cyc);
        double avg_warps = (double)active_warps / active_cyc;
        printf("    Avg warps per cycle:    %.1f (max 64 on P4)\n", avg_warps);
        printf("    Occupancy:              %.1f%%\n", 100.0 * avg_warps / 64.0);
    }
    
    if (inst_executed > 0)
        printf("    Instructions executed:  %lu\n", inst_executed);
    
    if (l2_queries > 0)
        printf("    L2 hit rate:            %.1f%%\n", 100.0 * l2_hits / l2_queries);
    if (l2_misses > 0)
        printf("    L2 misses:              %lu (%.2f MB to DRAM)\n", l2_misses, l2_misses*32.0/1e6);
    
    /* Timing for bandwidth */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) ggml_backend_graph_compute(cuda, graph);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    float gbps = ((float)a_sz * 100 / (ms/1000.0f)) / 1e9f;
    printf("    Achieved bandwidth:     %.1f GB/s (%.1f%% of 192 peak)\n", gbps, 100*gbps/192);
    
    /* Cleanup */
    free(h_a); free(h_b);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    ggml_backend_buffer_free(gbuf);
    ggml_backend_free(cuda);
    ggml_free(ctx);
    free(buf);
}

int main() {
    printf("=== COMPREHENSIVE GPU ROOFLINE ANALYSIS: ggml Q8_0 matmul on P4 ===\n");
    profile_matmul_counters(2048, 2048, 1, "Attn proj (decode)");
    profile_matmul_counters(8192, 2048, 1, "FFN gate/up (decode)");
    profile_matmul_counters(2048, 8192, 1, "FFN down (decode)");
    return 0;
}
