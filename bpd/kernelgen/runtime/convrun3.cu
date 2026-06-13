// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* convrun3 <cubin> N Cin H W Cout KH KW Hout Wout wsize [time]
 * im2col + gemm_rect + relayout pipeline. xc.bin,wc.bin -> outc3.bin */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){
  int N=atoi(v[2]),Cin=atoi(v[3]),H=atoi(v[4]),W=atoi(v[5]),Cout=atoi(v[6]),KH=atoi(v[7]),KW=atoi(v[8]),Ho=atoi(v[9]),Wo=atoi(v[10]);
  long ws=atol(v[11]); int doTime=(c>12 && !strcmp(v[12],"time"));
  long K=(long)Cin*KH*KW, Nn=(long)N*Ho*Wo, M=Cout;
  size_t bx=(size_t)N*Cin*H*W*4, bw=ws*4, bcol=(size_t)K*Nn*4, bC=(size_t)M*Nn*4, bo=(size_t)N*Cout*Ho*Wo*4;
  float*xh=(float*)malloc(bx),*wh=(float*)malloc(bw),*oh=(float*)malloc(bo);
  FILE*fx=fopen("xc.bin","rb");fread(xh,4,bx/4,fx);fclose(fx);
  FILE*fw=fopen("wc.bin","rb");fread(wh,4,bw/4,fw);fclose(fw);
  CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,d));
  CUmodule m;CK(cuModuleLoad(&m,v[1]));
  CUfunction fi,fg,fr; CK(cuModuleGetFunction(&fi,m,"k_im2col"));CK(cuModuleGetFunction(&fg,m,"k_gemm_rect"));CK(cuModuleGetFunction(&fr,m,"k_relayout"));
  CUdeviceptr X,Wt,Col,Cm,O;
  CK(cuMemAlloc(&X,bx));CK(cuMemAlloc(&Wt,bw));CK(cuMemAlloc(&Col,bcol));CK(cuMemAlloc(&Cm,bC));CK(cuMemAlloc(&O,bo));
  CK(cuMemcpyHtoD(X,xh,bx));CK(cuMemcpyHtoD(Wt,wh,bw));
  int Ni=(int)N;
  auto run=[&](){
    // im2col
    void*ai[]={&X,&Col,&Ni,&Cin,&H,&W,&KH,&KW,&Ho,&Wo};
    CK(cuLaunchKernel(fi,1024,1,1,256,1,1,0,0,ai,0));
    // gemm_rect: grid (ceil(Nn/BN), ceil(M/BM)), block 256
    int Mi=(int)M,Ki=(int)K,Nni=(int)Nn;
    void*ag[]={&Wt,&Col,&Cm,&Mi,&Ki,&Nni};
    unsigned gx=(Nn+63)/64, gy=(M+63)/64;
    CK(cuLaunchKernel(fg,gx,gy,1,256,1,1,0,0,ag,0));
    // relayout
    void*ar[]={&Cm,&O,&Ni,&Cout,&Ho,&Wo};
    CK(cuLaunchKernel(fr,1024,1,1,256,1,1,0,0,ar,0));
  };
  run(); CK(cuCtxSynchronize());
  if(doTime){
    CUevent e0,e1;cuEventCreate(&e0,0);cuEventCreate(&e1,0);std::vector<float> ts;
    for(int i=0;i<30;i++){cuEventRecord(e0,0);run();cuEventRecord(e1,0);cuEventSynchronize(e1);float ms;cuEventElapsedTime(&ms,e0,e1);ts.push_back(ms);}
    std::sort(ts.begin(),ts.end());printf("median_ms=%.5f\n",ts[15]);
  }
  CK(cuMemcpyDtoH(oh,O,bo));FILE*fo=fopen("outc3.bin","wb");fwrite(oh,4,bo/4,fo);fclose(fo);
  if(!doTime)printf("ran\n");
  return 0;
}
