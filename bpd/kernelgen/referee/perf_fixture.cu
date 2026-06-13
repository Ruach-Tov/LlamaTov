// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* perf_fixture.cu — performance measurement fixture for generated kernels.
 *
 * The perf analog of the differential referee: loads a generated kernel
 * (PTX or cubin) via the driver API, times it with CUDA events (warmup +
 * N iterations, median + min/max), and reports throughput against the P4
 * roofline. Elementwise -> GB/s / %peak-bandwidth; GEMM -> GFLOPS / %peak-fp32.
 *
 * Usage:
 *   perf_fixture elementwise <op> <module.{ptx|cubin}> <N> [iters]
 *   perf_fixture gemm        <op> <module.{ptx|cubin}> <Ndim> [iters]
 *
 * Kernel ABI (matches the 3-param plain-pointer emitter):
 *   elementwise: kernel(const float* src, float* dst, i64 n)
 *   gemm:        kernel(const float* A, const float* B, float* C, i64 n)  [n x n]
 *
 * P4 roofline (Tesla P4, sm_61): PEAK_BW=192.2 GB/s, PEAK_FP32=5.70 TFLOPS.
 *
 * Author: Iyun, 2026-06-07 (perf fixture for the SKB sweep)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cuda.h>

#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s: %s\n",#x,s);exit(1);} }while(0)

static const double PEAK_BW_GBS    = 192.2;   // P4 GDDR5, 256-bit @ 3.003 GHz x2
static const double PEAK_FP32_TFLOP = 5.70;   // 20 SM x 128 core x 2 x 1.113 GHz

static char* slurp(const char* p, size_t* n){
    FILE* f=fopen(p,"rb"); if(!f){printf("open fail %s\n",p);exit(1);}
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    char* b=(char*)malloc(s+1); fread(b,1,s,f); b[s]=0; fclose(f); if(n)*n=s; return b;
}

// median of timing samples (ms)
static double median(std::vector<float>& v){
    std::sort(v.begin(), v.end());
    size_t m=v.size()/2;
    return v.size()%2 ? v[m] : 0.5*(v[m-1]+v[m]);
}

int main(int argc, char** argv){
    if(argc < 5){ printf("usage: perf_fixture <elementwise|gemm> <op> <module> <N> [iters]\n"); return 1; }
    const char* cls=argv[1]; const char* op=argv[2]; const char* mod=argv[3];
    long N=atol(argv[4]); int iters=argc>5?atoi(argv[5]):200;
    int warmup=20;
    bool is_gemm = (strcmp(cls,"gemm")==0);

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev,0));
    CUcontext ctx; CK(cuCtxCreate(&ctx,0,dev));
    size_t msz; char* mbuf=slurp(mod,&msz);
    CUmodule m; CK(cuModuleLoadData(&m,mbuf));
    CUfunction fn; CK(cuModuleGetFunction(&fn,m,op));

    // allocate buffers
    size_t elems = is_gemm ? (size_t)N*N : (size_t)N;
    size_t bytes = elems*sizeof(float);
    CUdeviceptr A,B,C;
    CK(cuMemAlloc(&A,bytes)); CK(cuMemAlloc(&C,bytes));
    if(is_gemm) CK(cuMemAlloc(&B,bytes));
    CK(cuMemsetD32(A, 0x3f000000, elems)); // fill 0.5f
    if(is_gemm) CK(cuMemsetD32(B, 0x3f000000, elems));

    // launch config + args
    int64_t n64 = N;
    void* args_ew[]  = {&A,&C,&n64};
    void* args_gemm[]= {&A,&B,&C,&n64};
    void** args = is_gemm ? args_gemm : args_ew;
    unsigned gx,gy,gz=1, bx,by=1,bz=1;
    if(is_gemm){ bx=16; by=16; gx=(N+15)/16; gy=(N+15)/16; }
    else { bx=256; gx=(unsigned)((N+255)/256); gy=1; }

    CUevent e0,e1; CK(cuEventCreate(&e0,0)); CK(cuEventCreate(&e1,0));

    // warmup
    for(int i=0;i<warmup;i++)
        CK(cuLaunchKernel(fn,gx,gy,gz,bx,by,bz,0,0,args,0));
    CK(cuCtxSynchronize());

    // timed iterations
    std::vector<float> samples;
    for(int i=0;i<iters;i++){
        CK(cuEventRecord(e0,0));
        CK(cuLaunchKernel(fn,gx,gy,gz,bx,by,bz,0,0,args,0));
        CK(cuEventRecord(e1,0));
        CK(cuEventSynchronize(e1));
        float ms; CK(cuEventElapsedTime(&ms,e0,e1));
        samples.push_back(ms);
    }
    double med=median(samples);
    float mn=*std::min_element(samples.begin(),samples.end());
    float mx=*std::max_element(samples.begin(),samples.end());

    printf("=== perf: %s %s  (N=%ld, %d iters, P4) ===\n", cls, op, N, iters);
    printf("  time: median=%.4f ms  min=%.4f  max=%.4f\n", med, mn, mx);
    if(is_gemm){
        double gflops = 2.0*N*N*N / (med*1e-3) / 1e9;
        printf("  GFLOPS: %.1f  (%.1f%% of %.2f TFLOPS peak)\n",
               gflops, gflops/1000.0/PEAK_FP32_TFLOP*100.0, PEAK_FP32_TFLOP);
    } else {
        double gbs = (2.0*bytes) / (med*1e-3) / 1e9; // read N + write N
        printf("  GB/s: %.1f  (%.1f%% of %.1f GB/s peak)\n",
               gbs, gbs/PEAK_BW_GBS*100.0, PEAK_BW_GBS);
    }
    return 0;
}
