// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// poolrun1d <cubin> NC L Lout  | poolrun2d <cubin> NC H W Hout Wout
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,d));
 CUmodule m;CK(cuModuleLoad(&m,v[1]));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"k_pool"));
 if(c==5){int NC=atoi(v[2]),L=atoi(v[3]),Lo=atoi(v[4]);
   size_t bi=(size_t)NC*L*4,bo=(size_t)NC*Lo*4;float*xh=(float*)malloc(bi),*oh=(float*)malloc(bo);
   FILE*fi=fopen("xp.bin","rb");fread(xh,4,(size_t)NC*L,fi);fclose(fi);
   CUdeviceptr X,O;CK(cuMemAlloc(&X,bi));CK(cuMemAlloc(&O,bo));CK(cuMemcpyHtoD(X,xh,bi));
   void*a[]={&X,&O,&NC,&L,&Lo};unsigned tot=NC*Lo,g=(tot+127)/128;
   CK(cuLaunchKernel(fn,g,1,1,128,1,1,0,0,a,0));CK(cuCtxSynchronize());
   CK(cuMemcpyDtoH(oh,O,bo));FILE*fo=fopen("outp.bin","wb");fwrite(oh,4,(size_t)NC*Lo,fo);fclose(fo);printf("ran1d\n");
 } else {int NC=atoi(v[2]),H=atoi(v[3]),W=atoi(v[4]),Ho=atoi(v[5]),Wo=atoi(v[6]);
   size_t bi=(size_t)NC*H*W*4,bo=(size_t)NC*Ho*Wo*4;float*xh=(float*)malloc(bi),*oh=(float*)malloc(bo);
   FILE*fi=fopen("xp.bin","rb");fread(xh,4,(size_t)NC*H*W,fi);fclose(fi);
   CUdeviceptr X,O;CK(cuMemAlloc(&X,bi));CK(cuMemAlloc(&O,bo));CK(cuMemcpyHtoD(X,xh,bi));
   void*a[]={&X,&O,&NC,&H,&W,&Ho,&Wo};unsigned tot=NC*Ho*Wo,g=(tot+127)/128;
   CK(cuLaunchKernel(fn,g,1,1,128,1,1,0,0,a,0));CK(cuCtxSynchronize());
   CK(cuMemcpyDtoH(oh,O,bo));FILE*fo=fopen("outp.bin","wb");fwrite(oh,4,(size_t)NC*Ho*Wo,fo);fclose(fo);printf("ran2d\n");}
 return 0;}
