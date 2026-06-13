// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// convrun <cubin> N Cin H W Cout KH KW Hout Wout : reads xc.bin, wc.bin -> outc.bin
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){
 int N=atoi(v[2]),Cin=atoi(v[3]),H=atoi(v[4]),W=atoi(v[5]),Cout=atoi(v[6]),KH=atoi(v[7]),KW=atoi(v[8]),Ho=atoi(v[9]),Wo=atoi(v[10]);
 long cig_times = atol(v[11]); // weight in-channels-per-group * ... we pass weight size directly
 size_t bx=(size_t)N*Cin*H*W*4, bw=cig_times*4, bo=(size_t)N*Cout*Ho*Wo*4;
 float*xh=(float*)malloc(bx),*wh=(float*)malloc(bw),*oh=(float*)malloc(bo);
 FILE*fx=fopen("xc.bin","rb");fread(xh,4,bx/4,fx);fclose(fx);
 FILE*fw=fopen("wc.bin","rb");fread(wh,4,bw/4,fw);fclose(fw);
 CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,d));
 CUmodule m;CK(cuModuleLoad(&m,v[1]));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"k_conv"));
 CUdeviceptr X,Wt,O;CK(cuMemAlloc(&X,bx));CK(cuMemAlloc(&Wt,bw));CK(cuMemAlloc(&O,bo));
 CK(cuMemcpyHtoD(X,xh,bx));CK(cuMemcpyHtoD(Wt,wh,bw));
 void*a[]={&X,&Wt,&O,&N,&Cin,&H,&W,&Cout,&KH,&KW,&Ho,&Wo};unsigned tot=N*Cout*Ho*Wo,g=(tot+127)/128;
 CK(cuLaunchKernel(fn,g,1,1,128,1,1,0,0,a,0));CK(cuCtxSynchronize());
 CK(cuMemcpyDtoH(oh,O,bo));FILE*fo=fopen("outc.bin","wb");fwrite(oh,4,bo/4,fo);fclose(fo);printf("ranconv\n");return 0;}
