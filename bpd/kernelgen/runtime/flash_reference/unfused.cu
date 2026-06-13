// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.

extern "C" __global__ void k_gemm_nt(const float* A,const float* B,float* C,int M,int N,int K){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)M*N)return;
  int i=idx/N,j=idx%N; float a=0.f; for(int k=0;k<K;k++)a+=A[(long)i*K+k]*B[(long)j*K+k]; C[idx]=a; }
extern "C" __global__ void k_gemm_nn(const float* A,const float* B,float* C,int M,int N,int K){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)M*N)return;
  int i=idx/N,j=idx%N; float a=0.f; for(int k=0;k<K;k++)a+=A[(long)i*K+k]*B[(long)k*N+j]; C[idx]=a; }
extern "C" __global__ void k_softmax_scale(const float* in,float* out,int M,int N,float scale){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=M)return; const float* r=in+(long)i*N; float* o=out+(long)i*N;
  float mx=-3.4e38f; for(int j=0;j<N;j++){float v=r[j]*scale;if(v>mx)mx=v;}
  float s=0.f; for(int j=0;j<N;j++){float e=expf(r[j]*scale-mx);o[j]=e;s+=e;} float inv=1.f/s; for(int j=0;j<N;j++)o[j]*=inv; }
