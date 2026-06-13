// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* convrun4 <cubin> N Cin H W Cout KH KW Hout Wout wsize [time] — single fused implicit-GEMM conv */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s\n",s);return 1;}}while(0)
int main(int c,char**v){
  int N=atoi(v[2]),Cin=atoi(v[3]),H=atoi(v[4]),W=atoi(v[5]),Cout=atoi(v[6]),KH=atoi(v[7]),KW=atoi(v[8]),Ho=atoi(v[9]),Wo=atoi(v[10]);
  long ws=atol(v[11]); int doTime=(c>12 && !strcmp(v[12],"time"));
  long Nn=(long)N*Ho*Wo, M=Cout;
  size_t bx=(size_t)N*Cin*H*W*4, bw=ws*4, bo=(size_t)N*Cout*Ho*Wo*4;
  float*xh=(float*)malloc(bx),*wh=(float*)malloc(bw),*oh=(float*)malloc(bo);
  FILE*fx=fopen("xc.bin","rb");fread(xh,4,bx/4,fx);fclose(fx);
  FILE*fw=fopen("wc.bin","rb");fread(wh,4,bw/4,fw);fclose(fw);
  CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,d));
  CUmodule m;CK(cuModuleLoad(&m,v[1]));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"k_conv_implicit"));
  CUdeviceptr X,Wt,O;CK(cuMemAlloc(&X,bx));CK(cuMemAlloc(&Wt,bw));CK(cuMemAlloc(&O,bo));
  CK(cuMemcpyHtoD(X,xh,bx));CK(cuMemcpyHtoD(Wt,wh,bw));
  void*a[]={&X,&Wt,&O,&N,&Cin,&H,&W,&Cout,&KH,&KW,&Ho,&Wo};
  unsigned gx=(Nn+63)/64, gy=(M+63)/64;
  CK(cuLaunchKernel(fn,gx,gy,1,256,1,1,0,0,a,0));CK(cuCtxSynchronize());
  if(doTime){CUevent e0,e1;cuEventCreate(&e0,0);cuEventCreate(&e1,0);std::vector<float>t;
    for(int i=0;i<30;i++){cuEventRecord(e0,0);cuLaunchKernel(fn,gx,gy,1,256,1,1,0,0,a,0);cuEventRecord(e1,0);cuEventSynchronize(e1);float ms;cuEventElapsedTime(&ms,e0,e1);t.push_back(ms);}
    std::sort(t.begin(),t.end());printf("median_ms=%.5f\n",t[15]);}
  CK(cuMemcpyDtoH(oh,O,bo));FILE*fo=fopen("outc4.bin","wb");fwrite(oh,4,bo/4,fo);fclose(fo);
  if(!doTime)printf("ran\n");return 0;}
