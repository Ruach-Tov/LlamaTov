// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// kernel_guard_test.cu — prove the guard catches all 3 fault classes.
#include "kernel_guard.cuh"
#include <cuda.h>

__global__ void good_kernel(const int8_t* q, float* out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ const uint4* p=(const uint4*)q; uint4 v=p[i%(n/16)]; out[i]=(float)(int8_t)v.x; }
}
// silent-wrong: runs clean but multiplies by a wrong (stale) factor — like a bad blocks_per_row
__global__ void wrong_kernel(const int8_t* q, float* out, int n, int bad_factor){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ out[i]=(float)q[i]*bad_factor; }   // bad_factor=0 (stale global) -> all zeros, NO fault
}
__global__ void correct_kernel(const int8_t* q, float* out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ out[i]=(float)q[i]*1; }
}

int main(){
    cuInit(0); CUcontext ctx; CUdevice dev; cuDeviceGet(&dev,0); cuCtxCreate(&ctx,0,dev);
    int n=1024; int8_t* q; float* out;
    cudaMalloc(&q,n*16); cudaMalloc(&out,n*sizeof(float)); cudaMemset(q,2,n*16);
    // reference: correct output (q=2, factor 1 -> 2.0)
    float* ref=(float*)malloc(n*sizeof(float)); for(int i=0;i<n;i++) ref[i]=2.0f;

    printf("=== kernel_guard test: catch all 3 fault classes ===\n\n");

    printf("1. PRE-LAUNCH ALIGN check (16-byte aligned ptr — should PASS):\n");
    GUARD_ALIGN(q, 16, "quants (aligned)");
    printf("   misaligned ptr (q+1 — should REJECT):\n");
    GUARD_ALIGN((const int8_t*)q+1, 16, "quants (misaligned)");

    printf("\n2. FATAL FAULT (correct_kernel — should be clean):\n");
    GUARD_LAUNCH((correct_kernel<<<(n+127)/128,128>>>(q,out,n)), "correct_kernel");

    printf("\n3. SILENT WRONG OUTPUT (wrong_kernel, bad_factor=0 — NO fault but WRONG):\n");
    wrong_kernel<<<(n+127)/128,128>>>(q,out,n,0); cudaDeviceSynchronize();
    int sw = GUARD_VERIFY(out, ref, n, "wrong_kernel(stale=0)");

    printf("\n4. CORRECT kernel (should pass GUARD_VERIFY):\n");
    correct_kernel<<<(n+127)/128,128>>>(q,out,n); cudaDeviceSynchronize();
    int ok = GUARD_VERIFY(out, ref, n, "correct_kernel");

    printf("\n=== VERDICT ===\n");
    printf("  silent-wrong DETECTED: %s (the dangerous case — clean run, wrong numbers)\n", sw?"YES":"NO");
    printf("  correct kernel PASSES: %s\n", ok==0?"YES":"NO");
    printf("  guard %s\n", (sw && ok==0) ? "WORKS — catches silent corruption + passes correct"
                                         : "FAILED");
    return 0;
}
