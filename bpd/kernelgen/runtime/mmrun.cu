// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){int N=atoi(v[2]);size_t b=(size_t)N*N*4;
 float*Ah=(float*)malloc(b),*Bh=(float*)malloc(b),*Ch=(float*)malloc(b);
 FILE*fa=fopen("A.bin","rb");fread(Ah,4,(size_t)N*N,fa);fclose(fa);FILE*fb=fopen("B.bin","rb");fread(Bh,4,(size_t)N*N,fb);fclose(fb);
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,d));
 CUmodule m;CK(cuModuleLoad(&m,v[1]));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"gemm"));
 CUdeviceptr A,B,C;CK(cuMemAlloc(&A,b));CK(cuMemAlloc(&B,b));CK(cuMemAlloc(&C,b));
 CK(cuMemcpyHtoD(A,Ah,b));CK(cuMemcpyHtoD(B,Bh,b));long n=N;void*a[]={&A,&B,&C,&n};
 unsigned g=(N+15)/16;CK(cuLaunchKernel(fn,g,g,1,16,16,1,0,0,a,0));CK(cuCtxSynchronize());
 CK(cuMemcpyDtoH(Ch,C,b));FILE*fo=fopen("Cmm.bin","wb");fwrite(Ch,4,(size_t)N*N,fo);fclose(fo);printf("ran N=%d\n",N);return 0;}
