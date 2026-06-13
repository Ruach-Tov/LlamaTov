// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){
 const char*cub=v[1]; int NTH=atoi(v[2]),BM=atoi(v[3]),BN=atoi(v[4]),N=atoi(v[5]);
 const char*mode=v[6]; int iters=c>7?atoi(v[7]):30;
 size_t b=(size_t)N*N*4;
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,d));
 CUmodule m;CK(cuModuleLoad(&m,cub));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"k_gemm"));
 CUdeviceptr A,B,C;CK(cuMemAlloc(&A,b));CK(cuMemAlloc(&B,b));CK(cuMemAlloc(&C,b));
 long n=N;void*a[]={&A,&B,&C,&n};
 unsigned gx=(N+BN-1)/BN, gy=(N+BM-1)/BM;
 if(strcmp(mode,"verify")==0){
   float*Ah=(float*)malloc(b),*Bh=(float*)malloc(b),*Ch=(float*)malloc(b);
   FILE*fa=fopen("A.bin","rb");fread(Ah,4,(size_t)N*N,fa);fclose(fa);
   FILE*fb=fopen("B.bin","rb");fread(Bh,4,(size_t)N*N,fb);fclose(fb);
   CK(cuMemcpyHtoD(A,Ah,b));CK(cuMemcpyHtoD(B,Bh,b));
   CK(cuLaunchKernel(fn,gx,gy,1,NTH,1,1,0,0,a,0));CK(cuCtxSynchronize());
   CK(cuMemcpyDtoH(Ch,C,b));FILE*fo=fopen("Csw.bin","wb");fwrite(Ch,4,(size_t)N*N,fo);fclose(fo);
   printf("verify ran NTH=%d grid=%dx%d\n",NTH,gx,gy);
 } else {
   for(int i=0;i<10;i++) CK(cuLaunchKernel(fn,gx,gy,1,NTH,1,1,0,0,a,0));
   CK(cuCtxSynchronize());
   CUevent e0,e1;cuEventCreate(&e0,0);cuEventCreate(&e1,0);
   cuEventRecord(e0,0);
   for(int i=0;i<iters;i++) CK(cuLaunchKernel(fn,gx,gy,1,NTH,1,1,0,0,a,0));
   cuEventRecord(e1,0);cuEventSynchronize(e1);
   float ms;cuEventElapsedTime(&ms,e0,e1);ms/=iters;
   double gf=2.0*N*N*N/(ms*1e-3)/1e9;
   printf("GFLOPS: %.1f (%.1f%% peak)\n",gf,gf/1000.0/5.70*100);
 }
 return 0;}
