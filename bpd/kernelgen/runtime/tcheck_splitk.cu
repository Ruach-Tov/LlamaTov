// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* tcheck_splitk — harness for the SPLIT-K rect GEMM (two-kernel, workspace).
 * C[MxN]=A[MxK]*B[KxN] via SPLITS K-stripes -> workspace -> deterministic reduce.
 * usage: tcheck_splitk <cubin> <NTH> <BM> <BN> <M> <N> <K> <SPLITS> <mode> [iters]
 *   verify: load A.bin/B.bin -> Csw.bin ; perf: random, time iters -> GFLOPS
 * By: Iyun, 2026-06-08
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)

int main(int c,char**v){
  if(c<10){printf("usage: tcheck_splitk cubin NTH BM BN M N K SPLITS mode [iters]\n");return 1;}
  const char* cub=v[1];
  int NTH=atoi(v[2]),BM=atoi(v[3]),BN=atoi(v[4]),M=atoi(v[5]),N=atoi(v[6]),K=atoi(v[7]),SPLITS=atoi(v[8]);
  const char* mode=v[9]; int iters=c>10?atoi(v[10]):30;
  size_t bA=(size_t)M*K*4, bB=(size_t)K*N*4, bC=(size_t)M*N*4, bW=(size_t)SPLITS*M*N*4;
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CUcontext ctx; CK(cuCtxCreate(&ctx,0,d));
  CUmodule m; CK(cuModuleLoad(&m,cub));
  CUfunction kg,kr; CK(cuModuleGetFunction(&kg,m,"k_gemm_splitk")); CK(cuModuleGetFunction(&kr,m,"k_splitk_reduce"));
  CUdeviceptr A,B,C,W; CK(cuMemAlloc(&A,bA)); CK(cuMemAlloc(&B,bB)); CK(cuMemAlloc(&C,bC)); CK(cuMemAlloc(&W,bW));
  unsigned gx=(N+BN-1)/BN, gy=(M+BM-1)/BM, gz=SPLITS;
  void* ga[]={&A,&B,&W,&M,&N,&K,&SPLITS};
  long MN=(long)M*N; unsigned rgx=(unsigned)((MN+255)/256);
  void* ra[]={&W,&C,&M,&N,&SPLITS};
  if(!strcmp(mode,"verify")){
    float* Ah=(float*)malloc(bA); float* Bh=(float*)malloc(bB); float* Ch=(float*)malloc(bC);
    FILE* fa=fopen("A.bin","rb"); fread(Ah,4,(size_t)M*K,fa); fclose(fa);
    FILE* fb=fopen("B.bin","rb"); fread(Bh,4,(size_t)K*N,fb); fclose(fb);
    CK(cuMemcpyHtoD(A,Ah,bA)); CK(cuMemcpyHtoD(B,Bh,bB));
    CK(cuLaunchKernel(kg,gx,gy,gz,NTH,1,1,0,0,ga,0)); CK(cuCtxSynchronize());
    CK(cuLaunchKernel(kr,rgx,1,1,256,1,1,0,0,ra,0)); CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(Ch,C,bC));
    FILE* fo=fopen("Csw.bin","wb"); fwrite(Ch,4,(size_t)M*N,fo); fclose(fo);
    printf("verify ok: M=%d N=%d K=%d SPLITS=%d grid=(%u,%u,%u)\n",M,N,K,SPLITS,gx,gy,gz);
  } else {
    float* Ah=(float*)malloc(bA); float* Bh=(float*)malloc(bB);
    for(size_t i=0;i<(size_t)M*K;i++) Ah[i]=(float)(i%13)*0.01f;
    for(size_t i=0;i<(size_t)K*N;i++) Bh[i]=(float)(i%7)*0.02f;
    CK(cuMemcpyHtoD(A,Ah,bA)); CK(cuMemcpyHtoD(B,Bh,bB));
    CK(cuLaunchKernel(kg,gx,gy,gz,NTH,1,1,0,0,ga,0));
    CK(cuLaunchKernel(kr,rgx,1,1,256,1,1,0,0,ra,0)); CK(cuCtxSynchronize());
    CUevent t0,t1; CK(cuEventCreate(&t0,0)); CK(cuEventCreate(&t1,0));
    CK(cuEventRecord(t0,0));
    for(int it=0;it<iters;it++){
      CK(cuLaunchKernel(kg,gx,gy,gz,NTH,1,1,0,0,ga,0));
      CK(cuLaunchKernel(kr,rgx,1,1,256,1,1,0,0,ra,0));
    }
    CK(cuEventRecord(t1,0)); CK(cuEventSynchronize(t1));
    float ms=0; CK(cuEventElapsedTime(&ms,t0,t1)); ms/=iters;
    double gf=2.0*M*N*K/(ms*1e-3)/1e9;
    printf("perf: %.4f ms  %.1f GFLOPS  (M=%d N=%d K=%d SPLITS=%d)\n",ms,gf,M,N,K,SPLITS);
  }
  return 0;
}
