// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// perf_kernel — generic timed launcher for our generated cubins on the P4.
// Usage:
//   perf_kernel <cubin> <fname> elemwise <N>
//   perf_kernel <cubin> <fname> reduce   <R> <C>
//   perf_kernel <cubin> <fname> pool1d   <NC> <L> <Lout>
//   perf_kernel <cubin> <fname> pool2d   <NC> <H> <W> <Hout> <Wout>
//   perf_kernel <cubin> <fname> conv2d   <N> <Cin> <H> <W> <Cout> <KH> <KW> <Hout> <Wout> <wsize>
//   perf_kernel <cubin> <fname> gemm     <N>
// Runs 30 timed reps via CUDA events, prints: "median_ms=<m> reps=30".
// Inputs are random device data (perf only — correctness verified elsewhere).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)

static CUdeviceptr dalloc(size_t bytes){ CUdeviceptr p; if(cuMemAlloc(&p,bytes)!=CUDA_SUCCESS) return 0; cuMemsetD8(p,1,bytes); return p; }

int main(int argc,char**argv){
  if(argc<4){printf("usage: perf_kernel cubin fname shape ...\n");return 1;}
  const char* cubin=argv[1]; const char* fname=argv[2]; const char* shape=argv[3];
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CUcontext c; CK(cuCtxCreate(&c,0,d));
  CUmodule m; CK(cuModuleLoad(&m,cubin)); CUfunction fn; CK(cuModuleGetFunction(&fn,m,fname));

  // build args + launch config per shape
  unsigned gx=1,gy=1,bx=1,by=1; std::vector<void*> args; std::vector<long> ints; std::vector<CUdeviceptr> ptrs;
  // reserve so addresses stay stable
  ints.reserve(16); ptrs.reserve(8);
  auto I=[&](long v){ints.push_back(v); return &ints.back();};
  auto P=[&](size_t b){ptrs.push_back(dalloc(b)); return &ptrs.back();};

  if(!strcmp(shape,"elemwise")){
    long N=atol(argv[4]); size_t b=(size_t)N*4;
    args={P(b),P(b),I(N)}; bx=256; gx=(N+255)/256;
  } else if(!strcmp(shape,"reduce")){
    long R=atol(argv[4]),C=atol(argv[5]);
    // TILED reduce: ONE BLOCK PER ROW (256 threads coalesce across columns).
    args={P((size_t)R*C*4),P((size_t)R*4),I(R),I(C)}; bx=256; gx=(unsigned)R;
  } else if(!strcmp(shape,"pool1d")){
    long NC=atol(argv[4]),L=atol(argv[5]),Lo=atol(argv[6]);
    args={P((size_t)NC*L*4),P((size_t)NC*Lo*4),I(NC),I(L),I(Lo)}; bx=128; gx=(NC*Lo+127)/128;
  } else if(!strcmp(shape,"pool2d")){
    long NC=atol(argv[4]),H=atol(argv[5]),W=atol(argv[6]),Ho=atol(argv[7]),Wo=atol(argv[8]);
    // simple thread-per-output pool: total threads.
    args={P((size_t)NC*H*W*4),P((size_t)NC*Ho*Wo*4),I(NC),I(H),I(W),I(Ho),I(Wo)};
    bx=128; gx=(unsigned)(((long)NC*Ho*Wo+127)/128);
  } else if(!strcmp(shape,"pool2d_warp")){
    long NC=atol(argv[4]),H=atol(argv[5]),W=atol(argv[6]),Ho=atol(argv[7]),Wo=atol(argv[8]);
    // TILED pool: ONE WARP per output element -> total*32 lanes (coalesced window reads).
    long lanes=(long)NC*Ho*Wo*32;
    args={P((size_t)NC*H*W*4),P((size_t)NC*Ho*Wo*4),I(NC),I(H),I(W),I(Ho),I(Wo)}; bx=128; gx=(unsigned)((lanes+127)/128);
  } else if(!strcmp(shape,"conv2d")){
    long N=atol(argv[4]),Cin=atol(argv[5]),H=atol(argv[6]),W=atol(argv[7]),Cout=atol(argv[8]),KH=atol(argv[9]),KW=atol(argv[10]),Ho=atol(argv[11]),Wo=atol(argv[12]),ws=atol(argv[13]);
    args={P((size_t)N*Cin*H*W*4),P((size_t)ws*4),P((size_t)N*Cout*Ho*Wo*4),
          I(N),I(Cin),I(H),I(W),I(Cout),I(KH),I(KW),I(Ho),I(Wo)}; bx=128; gx=(N*Cout*Ho*Wo+127)/128;
  } else if(!strcmp(shape,"conv2d_implicit")){
    // implicit-GEMM conv (k_conv_implicit): SAME args as conv2d, but the launch
    // geometry is the GEMM tiling — gx over Nn=N*Ho*Wo (BN=64), gy over M=Cout
    // (BM=64), 256 threads/block. This is the 6.25x-over-naive kernel.
    long N=atol(argv[4]),Cin=atol(argv[5]),H=atol(argv[6]),W=atol(argv[7]),Cout=atol(argv[8]),KH=atol(argv[9]),KW=atol(argv[10]),Ho=atol(argv[11]),Wo=atol(argv[12]),ws=atol(argv[13]);
    // tile geometry the kernel was compiled with (BM,BN,NTH). Default to the
    // autotuned tile BM128 BN128 (NTH=512); override via argv[14..16].
    long tBM = argc>14?atol(argv[14]):128, tBN = argc>15?atol(argv[15]):128;
    long tNTH = argc>16?atol(argv[16]):512;
    long Nn=(long)N*Ho*Wo, M=Cout;
    args={P((size_t)N*Cin*H*W*4),P((size_t)ws*4),P((size_t)N*Cout*Ho*Wo*4),
          I(N),I(Cin),I(H),I(W),I(Cout),I(KH),I(KW),I(Ho),I(Wo)};
    bx=(unsigned)tNTH; gx=(unsigned)((Nn+tBN-1)/tBN); gy=(unsigned)((M+tBM-1)/tBM);
  } else if(!strcmp(shape,"gemm")){
    long N=atol(argv[4]); size_t b=(size_t)N*N*4;
    args={P(b),P(b),P(b),I(N)}; bx=16; by=16; gx=(N+15)/16; gy=(N+15)/16;
  } else { printf("unknown shape %s\n",shape); return 1; }

  // warmup
  for(int i=0;i<5;i++) CK(cuLaunchKernel(fn,gx,gy,1,bx,by,1,0,0,args.data(),0));
  CK(cuCtxSynchronize());
  CUevent e0,e1; cuEventCreate(&e0,0); cuEventCreate(&e1,0);
  std::vector<float> times;
  for(int i=0;i<30;i++){
    cuEventRecord(e0,0);
    CK(cuLaunchKernel(fn,gx,gy,1,bx,by,1,0,0,args.data(),0));
    cuEventRecord(e1,0); cuEventSynchronize(e1);
    float ms=0; cuEventElapsedTime(&ms,e0,e1); times.push_back(ms);
  }
  std::sort(times.begin(),times.end());
  printf("median_ms=%.5f reps=30\n", times[times.size()/2]);
  return 0;
}
