// kernel_guard.cuh — ROBUST kernel launch + fault-detecting fixtures.
//
// Distills the session's hard-won learning into a reusable capability:
//   - FATAL faults (misaligned load, OOB, div-by-zero) abort the context ->
//     caught by cudaGetLastError/Synchronize return codes (NOT the trap counter).
//   - RECOVERABLE traps -> caught by active_cycles_in_trap (when readable).
//   - SILENT WRONG OUTPUT (the most dangerous: kernel runs clean but wrong numbers,
//     e.g. stale-global blocks_per_row reading wrong layout) -> caught ONLY by a
//     bit-identity correctness check vs a reference. No fault counter sees it.
//
//   + PRE-LAUNCH validation: check pointer alignment for vectorized loads BEFORE the
//     launch, so a misaligned uint4 is rejected with a clear message instead of a
//     fatal context abort. This is the lesson from the AoS +2-offset / SoA bug.
//
// Usage:
//   GUARD_ALIGN(ptr, 16, "weight quants");        // pre-launch: reject misalignment
//   GUARD_LAUNCH(kernel<<<g,b>>>(args), "name");  // launch + fatal/trap check
//   GUARD_VERIFY(out, ref, n, "name");            // correctness: bit-identity gate
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

// ---- 1. PRE-LAUNCH alignment validation (reject the fatal misaligned-load class) ----
// Returns 0 if aligned, 1 if not (caller should abort the launch).
static inline int guard_check_align(const void* ptr, size_t align, const char* what){
    if (((uintptr_t)ptr % align) != 0){
        printf("[guard] PRE-LAUNCH REJECT: %s ptr=%p NOT %zu-byte aligned "
               "(would fault a %zu-bit vectorized load)\n", what, ptr, align, align*8);
        return 1;
    }
    return 0;
}
#define GUARD_ALIGN(ptr, align, what) do{ if(guard_check_align((ptr),(align),(what))) \
    { printf("[guard] launch aborted (alignment).\n"); } }while(0)

// ---- 2. ROBUST LAUNCH: launch + catch FATAL faults (the context-aborting class) ----
// Checks cudaGetLastError (launch config errors) AND cudaDeviceSynchronize (exec faults:
// misaligned/OOB/div0). Returns 0 clean, nonzero = fatal fault (with the error name).
static inline int guard_launch_check(const char* name){
    cudaError_t launch_err = cudaGetLastError();       // bad config / too many resources
    cudaError_t exec_err   = cudaDeviceSynchronize();  // misaligned/OOB/div0 during exec
    cudaError_t err = (launch_err != cudaSuccess) ? launch_err : exec_err;
    if (err != cudaSuccess){
        printf("[guard] %-24s *** FATAL FAULT *** %s\n", name, cudaGetErrorString(err));
        printf("        -> misaligned load / OOB / div-by-zero / bad launch config.\n");
        printf("        -> localize: compute-sanitizer --tool memcheck (exact warp+line).\n");
        return (int)err;
    }
    return 0;
}
#define GUARD_LAUNCH(call, name) do{ (call); guard_launch_check(name); }while(0)

// ---- 3. CORRECTNESS FIXTURE: bit-identity vs reference (catches SILENT WRONG OUTPUT) ----
// The fault counters and error codes CANNOT see a clean-but-wrong kernel. Only this can.
// Compares device output vs a reference buffer, reports ndiff + maxULP. 0 = bit-identical.
static inline int guard_verify_f32(const float* d_out, const float* h_ref, int n, const char* name){
    float* h_out = (float*)malloc((size_t)n*sizeof(float));
    cudaMemcpy(h_out, d_out, (size_t)n*sizeof(float), cudaMemcpyDeviceToHost);
    int ndiff=0, maxulp=0, first=-1;
    for (int i=0;i<n;i++){
        uint32_t a,b; memcpy(&a,&h_out[i],4); memcpy(&b,&h_ref[i],4);
        if (a!=b){ ndiff++; long d=(long)a-(long)b; if(d<0)d=-d; if(d>maxulp)maxulp=(int)d;
                   if(first<0)first=i; }
    }
    free(h_out);
    if (ndiff==0){
        printf("[guard] %-24s CORRECT (bit-identical vs ref, n=%d)\n", name, n);
        return 0;
    }
    printf("[guard] %-24s *** WRONG OUTPUT *** ndiff=%d/%d maxULP=%d first@%d "
           "(NO fault fired — silent corruption, e.g. stale param / wrong layout)\n",
           name, ndiff, n, maxulp, first);
    return 1;
}
#define GUARD_VERIFY(d_out, h_ref, n, name) guard_verify_f32((d_out),(h_ref),(n),(name))

// ---- Combined: the full robust-launch fixture (all three classes) ----
// 1) you GUARD_ALIGN the vectorized-load pointers, 2) GUARD_LAUNCH (fatal),
// 3) GUARD_VERIFY (silent-wrong). A kernel that passes all three is robust AND correct.
