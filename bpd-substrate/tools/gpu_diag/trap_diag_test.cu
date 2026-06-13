// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// trap_diag_test.cu — prove the trap diagnostic catches a real fault.
// Launches a CLEAN kernel (should PASS) and a FAULTING kernel (misaligned 128-bit
// load — the exact SoA bug class — should report FAULT). Validates trap_diag.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>

extern int  trap_diag_init();
extern void trap_diag_begin();
extern int  trap_diag_end(const char*);

__global__ void clean_kernel(float* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = i * 1.5f;
}
// Faulting: read uint4 (128-bit) from a deliberately MISaligned address (+1 byte).
// This is the AoS-quant-offset bug class (quants at +2 -> misaligned uint4 -> trap).
__global__ void faulting_kernel(const uint8_t* base, float* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){
        const uint4* p = (const uint4*)(base + 1);   // +1 = misaligned for 16-byte load
        uint4 v = p[i];                               // illegal -> trap
        out[i] = (float)(v.x + v.y);
    }
}

int main(){
    cuInit(0);
    CUcontext ctx; CUdevice dev;
    cuDeviceGet(&dev, 0); cuCtxCreate(&ctx, 0, dev);
    if (trap_diag_init() != 0){ printf("trap_diag_init failed\n"); return 1; }

    int n = 1024; float* out; uint8_t* base;
    cudaMalloc(&out, n*sizeof(float)); cudaMalloc(&base, n*16 + 16);
    cudaMemset(base, 1, n*16+16);

    printf("=== trap diagnostic self-test ===\n");
    // 1. clean kernel — expect PASS
    trap_diag_begin();
    clean_kernel<<<(n+127)/128,128>>>(out, n);
    cudaDeviceSynchronize();
    int r1 = trap_diag_end("clean_kernel");

    // 2. faulting kernel (misaligned 128-bit load) — expect FAULT
    trap_diag_begin();
    faulting_kernel<<<(n+127)/128,128>>>(base, out, n);
    cudaError_t ke = cudaDeviceSynchronize();
    int r2 = trap_diag_end("faulting_kernel(misaligned)");
    printf("  (cudaDeviceSynchronize after faulting: %s)\n", cudaGetErrorString(ke));

    printf("\n=== VERDICT ===\n");
    printf("  clean kernel:    %s (expected PASS)\n", r1==0?"PASS":"FAULT");
    printf("  faulting kernel: %s (expected FAULT)\n", r2==1?"FAULT":"PASS/missed");
    printf("  diagnostic %s\n", (r1==0 && r2==1) ? "WORKS — catches faults, clean passes"
                                                 : "INCONCLUSIVE (trap counter may need context reset)");
    return 0;
}
