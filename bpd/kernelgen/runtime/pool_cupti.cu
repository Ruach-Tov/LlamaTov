// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* pool_cupti — CUPTI stall profile of k_pool (fused vs non-fused), to find WHY
 * epilogue fusion is neutral. Mavchin's bpd_cupti_profile.c bridge.
 * usage: pool_cupti <cubin>  (maxpool2d K=4 S=1 P=1 shape, matches mp_run) */
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
extern "C" {
  typedef struct { uint64_t inst_fetch, exec_dependency, memory_dependency, texture,
    sync, constant_memory, pipe_busy, memory_throttle, not_selected, other, none, total_samples; } stall_counters_t;
  int bpd_cupti_init(void); int bpd_cupti_flush(void); void bpd_cupti_reset(void);
  int bpd_cupti_shutdown(void); void bpd_cupti_print_report(void);
}
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){
  int NC=256,H=130,W=130,K=4,S=1,P=1,Ho=H+2*P-K+1,Wo=W+2*P-K+1;
  long IN=(long)NC*H*W, OUT=(long)NC*Ho*Wo;
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CUcontext ctx; CK(cuCtxCreate(&ctx,0,d));
  CUmodule m; CK(cuModuleLoad(&m,v[1])); CUfunction fn; CK(cuModuleGetFunction(&fn,m,"k_pool"));
  int smem=0,nreg=0; cuFuncGetAttribute(&smem,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,fn);
  cuFuncGetAttribute(&nreg,CU_FUNC_ATTRIBUTE_NUM_REGS,fn);
  printf("=== k_pool static: regs/thread=%d shared=%dB ===\n", nreg, smem);
  size_t bin=IN*4, bo=OUT*4;
  CUdeviceptr X,O; CK(cuMemAlloc(&X,bin));CK(cuMemAlloc(&O,bo));
  int an[]={NC,H,W,Ho,Wo}; void* a[]={&X,&O,&an[0],&an[1],&an[2],&an[3],&an[4]};
  long lanes=OUT*32; unsigned bx=256, gx=(lanes+bx-1)/bx;
  CK(cuLaunchKernel(fn,gx,1,1,bx,1,1,0,0,a,0)); CK(cuCtxSynchronize());
  if(bpd_cupti_init()!=0){printf("cupti init fail\n");return 1;} bpd_cupti_reset();
  for(int it=0;it<80;it++) CK(cuLaunchKernel(fn,gx,1,1,bx,1,1,0,0,a,0));
  CK(cuCtxSynchronize()); bpd_cupti_flush(); bpd_cupti_print_report(); bpd_cupti_shutdown();
  return 0;
}
