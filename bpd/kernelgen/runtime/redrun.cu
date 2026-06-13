// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// redrun <cubin> <R> <C> : reads xr.bin (R*C f32), runs k_reduce, writes outr.bin (R f32)
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){int R=atoi(v[2]),C=atoi(v[3]);
 size_t bin=(size_t)R*C*4, bout=(size_t)R*4;
 float*xh=(float*)malloc(bin),*oh=(float*)malloc(bout);
 FILE*fi=fopen("xr.bin","rb");fread(xh,4,(size_t)R*C,fi);fclose(fi);
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,d));
 CUmodule m;CK(cuModuleLoad(&m,v[1]));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"k_reduce"));
 CUdeviceptr X,O;CK(cuMemAlloc(&X,bin));CK(cuMemAlloc(&O,bout));CK(cuMemcpyHtoD(X,xh,bin));
 void*a[]={&X,&O,&R,&C};unsigned g=(R+127)/128;
 CK(cuLaunchKernel(fn,g,1,1,128,1,1,0,0,a,0));CK(cuCtxSynchronize());
 CK(cuMemcpyDtoH(oh,O,bout));FILE*fo=fopen("outr.bin","wb");fwrite(oh,4,R,fo);fclose(fo);printf("ran R=%d C=%d\n",R,C);return 0;}
