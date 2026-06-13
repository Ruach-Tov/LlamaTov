// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* conv_cupti_profile — profile k_conv_implicit's WARP STALL REASONS via CUPTI PC
 * sampling (sm_61 legacy Activity API), to establish a BEFORE/AFTER baseline for
 * software-pipelining. If memory_dependency dominates -> pipelining (hide load
 * latency) is the right fix. Measure, then optimize.
 *
 * Loads a cubin, launches k_conv_implicit at the conv-50 shape under CUPTI.
 * usage: conv_cupti_profile <cubin> [BM BN NTH]
 * Build: nvcc -O3 -arch=sm_61 -o conv_cupti_profile conv_cupti_profile.cu \
 *          <path>/bpd_cupti_profile.c -lcupti -lcuda -I$CUPTI_INC
 * By: Iyun, 2026-06-08
 */
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

extern "C" {
    typedef struct {
        uint64_t inst_fetch, exec_dependency, memory_dependency, texture, sync,
                 constant_memory, pipe_busy, memory_throttle, not_selected, other,
                 none, total_samples;
    } stall_counters_t;
    int  bpd_cupti_init(void);
    int  bpd_cupti_flush(void);
    int  bpd_cupti_get_stalls(stall_counters_t* out);
    void bpd_cupti_reset(void);
    int  bpd_cupti_shutdown(void);
    void bpd_cupti_print_report(void);
}

#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: conv_cupti_profile cubin [BM BN NTH]\n"); return 1; }
    const char* cubin = argv[1];
    long BM = argc>2?atol(argv[2]):128, BN = argc>3?atol(argv[3]):128, NTH = argc>4?atol(argv[4]):512;

    // conv-50 shape: N=32 Cin=64 H=W=56 Cout=128 K=3 -> Ho=Wo=54
    int N=32, Cin=64, H=56, W=56, Cout=128, KH=3, KW=3, Ho=54, Wo=54;
    long ws = (long)Cout*Cin*KH*KW;
    long Nn = (long)N*Ho*Wo, M = Cout;
    size_t bx = (size_t)N*Cin*H*W*4, bw = (size_t)ws*4, bo = (size_t)N*Cout*Ho*Wo*4;

    CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CUcontext ctx; CK(cuCtxCreate(&ctx,0,d));
    CUmodule m; CK(cuModuleLoad(&m,cubin));
    CUfunction fn; CK(cuModuleGetFunction(&fn,m,"k_conv_implicit"));

    // report static shared-mem + register usage of the kernel
    int smem=0, nreg=0, maxthr=0;
    cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);
    cuFuncGetAttribute(&nreg, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
    cuFuncGetAttribute(&maxthr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn);
    printf("=== k_conv_implicit static resources (tile BM=%ld BN=%ld) ===\n", BM, BN);
    printf("  shared_mem/block: %d bytes (%.1f KB)\n", smem, smem/1024.0);
    printf("  registers/thread: %d\n", nreg);
    printf("  max_threads/block: %d   (launch NTH=%ld)\n\n", maxthr, NTH);

    CUdeviceptr X,Wt,O; CK(cuMemAlloc(&X,bx)); CK(cuMemAlloc(&Wt,bw)); CK(cuMemAlloc(&O,bo));
    void* args[] = { &X, &Wt, &O, &N, &Cin, &H, &W, &Cout, &KH, &KW, &Ho, &Wo };
    unsigned gx=(unsigned)((Nn+BN-1)/BN), gy=(unsigned)((M+BM-1)/BM);

    if (bpd_cupti_init() != 0) { printf("CUPTI init failed\n"); return 1; }
    bpd_cupti_reset();
    // launch several times for enough PC samples
    for (int it=0; it<50; it++)
        CK(cuLaunchKernel(fn, gx, gy, 1, (unsigned)NTH, 1, 1, 0, 0, args, 0));
    CK(cuCtxSynchronize());
    bpd_cupti_flush();
    bpd_cupti_print_report();
    bpd_cupti_shutdown();
    return 0;
}
