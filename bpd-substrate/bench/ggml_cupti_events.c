// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* ggml_cupti_events.c — Collect exact GPU hardware counters around ggml matmul.
 *
 * Uses CUPTI Event API for AGGREGATE TOTALS (not sampling).
 * Callable from Prolog via the FLI (load_foreign_library).
 *
 * Predicates:
 *   gpu_profile_matmul(+M, +K, +N, -Results)
 *     Results = list of counter_name=value pairs
 *
 * Author: mavchin (2026-06-03)
 */

#include <SWI-Prolog.h>
#include <cuda.h>
#include <cupti.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

/* Events to collect */
static const char *event_names[] = {
    "fb_subp0_read_sectors",
    "fb_subp1_read_sectors",
    "fb_subp0_write_sectors",
    "fb_subp1_write_sectors",
    "gld_inst_8bit",
    "gld_inst_16bit",
    "gld_inst_32bit",
    "gld_inst_64bit",
    "gld_inst_128bit",
    "elapsed_cycles_sm",
    "active_warps_pm",
    "active_cycles_pm",
};
#define N_EVENTS (sizeof(event_names) / sizeof(event_names[0]))

static foreign_t pl_gpu_profile_matmul(term_t t_m, term_t t_k, term_t t_n, term_t t_result) {
    int M, K, N;
    PL_get_integer(t_m, &M);
    PL_get_integer(t_k, &K);
    PL_get_integer(t_n, &N);
    
    /* Get the CURRENT CUDA context (created by ggml_backend_cuda_init) */
    CUcontext cuCtx;
    CUdevice cuDev;
    cuInit(0);
    cuDeviceGet(&cuDev, 0);
    
    /* Create ggml context + tensors FIRST (this creates the CUDA context) */
    void *buf = aligned_alloc(64, 16*1024*1024);
    struct ggml_init_params params = {16*1024*1024, buf, true};
    struct ggml_context *ctx = ggml_init(params);
    
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor *c = ggml_mul_mat(ctx, a, b);
    
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
    
    /* NOW get the context that ggml created */
    cuCtxGetCurrent(&cuCtx);
    
    /* Fill */
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
    
    /* Collect events one domain at a time */
    uint64_t values[N_EVENTS];
    memset(values, 0, sizeof(values));
    
    for (int e = 0; e < (int)N_EVENTS; e++) {
        CUpti_EventID eventId;
        CUptiResult cr = cuptiEventGetIdFromName(cuDev, event_names[e], &eventId);
        if (cr != CUPTI_SUCCESS) { values[e] = (uint64_t)-1; continue; }
        
        CUpti_EventGroup group;
        cr = cuptiEventGroupCreate(cuCtx, &group, 0);
        if (cr != CUPTI_SUCCESS) { values[e] = (uint64_t)-2; continue; }
        
        cr = cuptiEventGroupAddEvent(group, eventId);
        if (cr != CUPTI_SUCCESS) {
            cuptiEventGroupDestroy(group);
            values[e] = (uint64_t)-3;
            continue;
        }
        
        /* Profile all domain instances */
        uint32_t all = 1;
        cuptiEventGroupSetAttribute(group, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                    sizeof(all), &all);
        
        cuptiEventGroupEnable(group);
        
        /* Run the kernel */
        ggml_backend_graph_compute(cuda, graph);
        cudaDeviceSynchronize();
        
        /* Read ALL events (ReadEvent returns DISABLED on Pascal, ReadAll works) */
        size_t valsSz = sizeof(uint64_t) * 64;
        uint64_t readVals[64];
        CUpti_EventID readIds[64];
        size_t idsSz = sizeof(readIds);
        size_t numRead = 0;
        
        CUptiResult rr = cuptiEventGroupReadAllEvents(group, CUPTI_EVENT_READ_FLAG_NONE,
                                          &valsSz, readVals, &idsSz, readIds, &numRead);
        if (rr == CUPTI_SUCCESS && numRead > 0) {
            values[e] = readVals[0];
        } else {
            values[e] = 0;
            printf("  WARNING: ReadAll failed for %s: %d, numRead=%zu\n", event_names[e], rr, numRead);
        }
        
        cuptiEventGroupDisable(group);
        cuptiEventGroupDestroy(group);
    }
    
    /* Build result list for Prolog */
    term_t list = PL_new_term_ref();
    term_t head = PL_new_term_ref();
    PL_put_nil(list);
    
    /* Print + build list (reverse order for proper list construction) */
    printf("\n=== GPU Hardware Counters: matmul [%d x %d] x %d ===\n", M, K, N);
    for (int e = N_EVENTS - 1; e >= 0; e--) {
        printf("  %-30s %lu\n", event_names[e], values[e]);
        
        term_t pair = PL_new_term_ref();
        term_t fname = PL_new_term_ref();
        term_t fval = PL_new_term_ref();
        PL_put_atom_chars(fname, event_names[e]);
        PL_put_int64(fval, (int64_t)values[e]);
        
        /* Build name=value */
        PL_cons_functor(pair, PL_new_functor(PL_new_atom("="), 2), fname, fval);
        PL_cons_list(head, pair, list);
        PL_put_term(list, head);
    }
    
    /* Derived metrics */
    uint64_t fb_reads = values[0] + values[1];  /* subp0 + subp1 */
    uint64_t gld_total = values[4]+values[5]+values[6]+values[7]+values[8];
    printf("\n  --- Derived ---\n");
    printf("  fb_read_sectors (total):  %lu\n", fb_reads);
    printf("  gld_instructions (total): %lu\n", gld_total);
    if (gld_total > 0)
        printf("  SECTORS PER LOAD:         %.3f\n", (double)fb_reads / (double)gld_total);
    printf("  fb_read bytes:            %.2f MB\n", fb_reads * 32.0 / 1e6);
    printf("  weight bytes:             %.2f MB\n", a_sz / 1e6);
    
    PL_unify(t_result, list);
    
    /* Cleanup */
    free(h_a); free(h_b);
    ggml_backend_buffer_free(gbuf);
    ggml_backend_free(cuda);
    ggml_free(ctx);
    free(buf);
    
    return TRUE;
}

install_t install(void) {
    PL_register_foreign("gpu_profile_matmul", 4, pl_gpu_profile_matmul, 0);
}
