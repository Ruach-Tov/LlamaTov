// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <cstring>
// trap_diag.cu — STANDING post-kernel fault diagnostic via GPU hardware trap counters.
//
// Heath's "read errors from registers": instead of just "CUDA error" in the log,
// read active_cycles_in_trap + inst_executed_in_trap AFTER a kernel. Nonzero =>
// a warp hit an illegal access (misaligned load, div-by-zero, OOB) — the counter
// localizes WHICH and HOW MANY. Pairs with the e2e profiling capability as the
// error-detection layer (e2e_bench + skyline + THIS).
//
// Usage as a library: trap_diag_begin(); launch_kernel(); trap_diag_end(name);
// Reports: name PASS (trap_cycles=0) or name FAULT (trap_cycles=N, trap_insts=M).
//
// Build: link with -lcupti. Reads via the same cuptiEventGroupReadAllEvents path
// mavchin's bridge uses (cuptiEventGetIdFromName -> Enable -> launch -> ReadAll).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cupti.h>

// the two trap diagnostic events (P4 / GP104, sm_61)
static const char* TRAP_EVENTS[] = { "active_cycles_in_trap", "inst_executed_in_trap" };
#define N_TRAP 2

static CUpti_EventGroup g_trap_group = 0;
static CUcontext        g_ctx = 0;
static CUdevice         g_dev = 0;
static int              g_active = 0;

// Call once at startup (after a CUDA context exists).
int trap_diag_init() {
    cuCtxGetCurrent(&g_ctx);
    if (!g_ctx) { printf("[trap] no CUDA context\n"); return -1; }
    cuCtxGetDevice(&g_dev);
    CUptiResult r = cuptiEventGroupCreate(g_ctx, &g_trap_group, 0);
    if (r != CUPTI_SUCCESS) { printf("[trap] EventGroupCreate failed (%d)\n", r); return -1; }
    int added = 0;
    for (int i = 0; i < N_TRAP; i++) {
        CUpti_EventID id;
        r = cuptiEventGetIdFromName(g_dev, TRAP_EVENTS[i], &id);
        if (r != CUPTI_SUCCESS) { printf("[trap] event %s not found on this GPU\n", TRAP_EVENTS[i]); continue; }
        r = cuptiEventGroupAddEvent(g_trap_group, id);
        if (r == CUPTI_SUCCESS) added++;
    }
    if (!added) { printf("[trap] no trap events available\n"); return -1; }
    return 0;
}

// Begin measuring (enable counters) — call right before the kernel launch.
void trap_diag_begin() {
    if (!g_trap_group) return;
    uint32_t all = 1;
    cuptiEventGroupSetAttribute(g_trap_group,
        CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all);
    cuptiEventGroupEnable(g_trap_group);
    g_active = 1;
}

// End + report — call right after the kernel (and cudaDeviceSynchronize).
// Returns 0 if clean (no trap), 1 if a fault was detected.
int trap_diag_end(const char* kernel_name) {
    if (!g_active) return 0;
    cudaDeviceSynchronize();
    uint64_t vals[N_TRAP] = {0,0};
    size_t vsz = sizeof(vals);
    CUpti_EventID ids[N_TRAP]; size_t isz = sizeof(ids);
    size_t nread = N_TRAP;
    cuptiEventGroupReadAllEvents(g_trap_group, CUPTI_EVENT_READ_FLAG_NONE,
                                 &vsz, vals, &isz, ids, &nread);
    cuptiEventGroupDisable(g_trap_group);
    g_active = 0;
    // map results back by event id->name (order may differ); sum trap cycles
    uint64_t trap_cycles = 0, trap_insts = 0;
    for (size_t k = 0; k < nread; k++) {
        char nm[128]; size_t nl = sizeof(nm);
        cuptiEventGetAttribute(ids[k], CUPTI_EVENT_ATTR_NAME, &nl, nm);
        if (strstr(nm, "active_cycles_in_trap")) trap_cycles = vals[k];
        else if (strstr(nm, "inst_executed_in_trap")) trap_insts = vals[k];
    }
    if (trap_cycles == 0 && trap_insts == 0) {
        printf("[trap] %-28s PASS (no fault: trap_cycles=0)\n", kernel_name);
        return 0;
    }
    printf("[trap] %-28s *** FAULT DETECTED *** trap_cycles=%llu trap_insts=%llu\n",
           kernel_name, (unsigned long long)trap_cycles, (unsigned long long)trap_insts);
    printf("       -> a warp hit an illegal access (misaligned load / div-by-zero / OOB).\n");
    printf("       -> trap_insts localizes how many instructions ran in the fault path.\n");
    return 1;
}

// COMPLETE post-kernel diagnostic: fatal (cudaError) + recoverable (trap counter).
// Call right after the kernel launch + sync. Catches BOTH fault classes.
//   returns 0 = clean, 1 = recoverable trap, 2 = FATAL fault (misaligned/OOB/div0)
int kernel_diag_end(const char* name){
    cudaError_t ke = cudaGetLastError();          // fatal launch/exec error
    cudaError_t se = cudaDeviceSynchronize();
    cudaError_t fatal = (ke != cudaSuccess) ? ke : se;
    if (fatal != cudaSuccess){
        printf("[diag] %-28s *** FATAL FAULT *** %s\n", name, cudaGetErrorString(fatal));
        printf("       -> misaligned load / OOB / div-by-zero. Context aborted; trap counter unreadable.\n");
        printf("       -> localize with: compute-sanitizer (which warp/line), or check ptr alignment.\n");
        return 2;
    }
    // clean fatal-path -> check recoverable traps via the counter
    return trap_diag_end(name) ? 1 : 0;
}
