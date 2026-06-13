// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// mlirgpu_run <module.ptx|cubin> <op> <N>  : reads in.bin (N f32), runs the
// 3-param MLIR-GPU kernel (src,dst,long n), writes out.bin. cuModuleLoadData
// handles both PTX (JIT) and cubin (already linked).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){
  const char*mod=v[1]; const char*op=v[2]; long N=atol(v[3]);
  float*xh=(float*)malloc(N*4),*ch=(float*)malloc(N*4);
  FILE*fi=fopen("in.bin","rb"); if(!fi){printf("no in.bin\n");return 1;} fread(xh,4,N,fi);fclose(fi);
  CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,d));
  FILE*fp=fopen(mod,"rb");fseek(fp,0,2);long sz=ftell(fp);fseek(fp,0,0);char*b=(char*)malloc(sz+1);fread(b,1,sz,fp);b[sz]=0;fclose(fp);
  CUmodule m;CK(cuModuleLoadData(&m,b));CUfunction fn;CK(cuModuleGetFunction(&fn,m,op));
  CUdeviceptr xd,cd;CK(cuMemAlloc(&xd,N*4));CK(cuMemAlloc(&cd,N*4));CK(cuMemcpyHtoD(xd,xh,N*4));
  int64_t n=N; void*a[]={&xd,&cd,&n};
  CK(cuLaunchKernel(fn,(unsigned)((N+255)/256),1,1,256,1,1,0,0,a,0));CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(ch,cd,N*4));FILE*fo=fopen("out.bin","wb");fwrite(ch,4,N,fo);fclose(fo);
  printf("ran %s N=%ld\n",op,N);return 0;}
