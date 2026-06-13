// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// BPD matmul as shared library for bit_identical verification.
// This is the SAME kernel that our Prolog substrate generates.
#include <cuda_runtime.h>
__global__ void k_mm(const float*A,const float*B,float*C,int M,int N,int K){
    __shared__ float As[64][8],Bs[8][64];
    int ty=threadIdx.y,tx=threadIdx.x,tid=ty*16+tx;
    int row0=blockIdx.y*64+ty*4,col0=blockIdx.x*64+tx*4;
    float c00=0,c01=0,c02=0,c03=0,c10=0,c11=0,c12=0,c13=0;
    float c20=0,c21=0,c22=0,c23=0,c30=0,c31=0,c32=0,c33=0;
    for(int tile=0;tile<(K+7)/8;tile++){int bk=tile*8;
    {int idx=tid;int ar=idx/8,ac=idx%8;As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;idx+=256;ar=idx/8;ac=idx%8;As[ar][ac]=(blockIdx.y*64+ar<M&&bk+ac<K)?A[(blockIdx.y*64+ar)*K+bk+ac]:0;}
    {int idx=tid;int br=idx/64,bc=idx%64;Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;idx+=256;br=idx/64;bc=idx%64;Bs[br][bc]=(bk+br<K&&blockIdx.x*64+bc<N)?B[(bk+br)*N+blockIdx.x*64+bc]:0;}
    __syncthreads();
    for(int i=0;i<8;i++){float a0=As[ty*4][i],a1=As[ty*4+1][i],a2=As[ty*4+2][i],a3=As[ty*4+3][i];float b0=Bs[i][tx*4],b1=Bs[i][tx*4+1],b2=Bs[i][tx*4+2],b3=Bs[i][tx*4+3];c00+=a0*b0;c01+=a0*b1;c02+=a0*b2;c03+=a0*b3;c10+=a1*b0;c11+=a1*b1;c12+=a1*b2;c13+=a1*b3;c20+=a2*b0;c21+=a2*b1;c22+=a2*b2;c23+=a2*b3;c30+=a3*b0;c31+=a3*b1;c32+=a3*b2;c33+=a3*b3;}
    __syncthreads();}
    if(row0+0<M&&col0+0<N)C[(row0+0)*N+col0+0]=c00;if(row0+0<M&&col0+1<N)C[(row0+0)*N+col0+1]=c01;if(row0+0<M&&col0+2<N)C[(row0+0)*N+col0+2]=c02;if(row0+0<M&&col0+3<N)C[(row0+0)*N+col0+3]=c03;
    if(row0+1<M&&col0+0<N)C[(row0+1)*N+col0+0]=c10;if(row0+1<M&&col0+1<N)C[(row0+1)*N+col0+1]=c11;if(row0+1<M&&col0+2<N)C[(row0+1)*N+col0+2]=c12;if(row0+1<M&&col0+3<N)C[(row0+1)*N+col0+3]=c13;
    if(row0+2<M&&col0+0<N)C[(row0+2)*N+col0+0]=c20;if(row0+2<M&&col0+1<N)C[(row0+2)*N+col0+1]=c21;if(row0+2<M&&col0+2<N)C[(row0+2)*N+col0+2]=c22;if(row0+2<M&&col0+3<N)C[(row0+2)*N+col0+3]=c23;
    if(row0+3<M&&col0+0<N)C[(row0+3)*N+col0+0]=c30;if(row0+3<M&&col0+1<N)C[(row0+3)*N+col0+1]=c31;if(row0+3<M&&col0+2<N)C[(row0+3)*N+col0+2]=c32;if(row0+3<M&&col0+3<N)C[(row0+3)*N+col0+3]=c33;
}
extern "C" {
void bpd_sgemm(const float*A,const float*B,float*C,int M,int N,int K){
    dim3 g((N+63)/64,(M+63)/64),b(16,16);
    k_mm<<<g,b>>>(A,B,C,M,N,K);
}
void* gpu_alloc(int n){void*p;cudaMalloc(&p,n);return p;}
void gpu_free(void*p){cudaFree(p);}
void gpu_h2d(void*d,const void*s,int n){cudaMemcpy(d,s,n,cudaMemcpyHostToDevice);}
void gpu_d2h(void*d,const void*s,int n){cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost);}
void gpu_sync(){cudaDeviceSynchronize();}
}
