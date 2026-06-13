// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s: %s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){
  const char* op=v[1]; const char* cub=v[2]; const char* out=v[3];
  const int N=1024; float xh[N],ch[N];
  FILE* fi=fopen("/tmp/gpu-work/referee/input.bin","rb"); fread(xh,4,N,fi); fclose(fi);
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CUcontext ctx; CK(cuCtxCreate(&ctx,0,d));
  FILE* fc=fopen(cub,"rb"); fseek(fc,0,SEEK_END); long sz=ftell(fc); fseek(fc,0,SEEK_SET);
  char* buf=(char*)malloc(sz); fread(buf,1,sz,fc); fclose(fc);
  CUmodule m; CK(cuModuleLoadData(&m,buf)); CUfunction fn; CK(cuModuleGetFunction(&fn,m,op));
  CUdeviceptr xd,cd; CK(cuMemAlloc(&xd,N*4)); CK(cuMemAlloc(&cd,N*4)); CK(cuMemcpyHtoD(xd,xh,N*4));
  int64_t z=0,n=N,o=1; void* a[]={&xd,&xd,&z,&n,&o,&cd,&cd,&z,&n,&o};
  CK(cuLaunchKernel(fn,(N+255)/256,1,1,256,1,1,0,0,a,0)); CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(ch,cd,N*4)); FILE* fo=fopen(out,"wb"); fwrite(ch,4,N,fo); fclose(fo);
  printf("MLIR-GPU %s (cubin) on P4: in[4]=%g out[4]=%g\n",op,xh[4],ch[4]); return 0;
}
