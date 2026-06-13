// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>

#define Q8P_BLOCK 64

struct q8_1_block { half d; half s; int8_t qs[32]; };

// dp4a with COALESCED layout [nb, N, 64]
__global__ void k_vecmat_dp4a_coal(
    const float *A, const unsigned char *Bq, float *C, int K, int N) {
    int nb = K / 32;
    extern __shared__ q8_1_block sA[];
    for (int kb = threadIdx.x; kb < nb; kb += blockDim.x) {
        float amax = 0;
        for (int i = 0; i < 32; i++) { float v = fabsf(A[kb*32+i]); if(v>amax)amax=v; }
        float d = amax/127.0f, id = d>0 ? 127.0f/amax : 0;
        sA[kb].d = __float2half(d);
        for (int i = 0; i < 32; i++) sA[kb].qs[i] = (int8_t)__float2int_rn(A[kb*32+i]*id);
    }
    __syncthreads();
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    
    float sum = 0;
    for (int kb = 0; kb < nb; kb++) {
        // COALESCED: [kb, col, 64] — adjacent threads read adjacent blocks
        int boff = (kb * N + col) * Q8P_BLOCK;
        float d0 = __half2float(*(const half*)(&Bq[boff]));
        float d1 = __half2float(sA[kb].d);
        const int *v = (const int*)(&Bq[boff + 4]);
        const int *u = (const int*)(sA[kb].qs);
        int s = 0;
        s = __dp4a(v[0],u[0],s); s = __dp4a(v[1],u[1],s);
        s = __dp4a(v[2],u[2],s); s = __dp4a(v[3],u[3],s);
        s = __dp4a(v[4],u[4],s); s = __dp4a(v[5],u[5],s);
        s = __dp4a(v[6],u[6],s); s = __dp4a(v[7],u[7],s);
        sum += d0 * d1 * (float)s;
    }
    C[col] = sum;
}

int main() {
    cublasHandle_t handle; cublasCreate(&handle);
    int shapes[][2] = {{2048,2048},{2048,8192},{8192,2048}};
    const char *names[] = {"proj","FFN up","FFN down"};
    int ITERS = 2000;
    
    printf("=== dp4a COALESCED [nb,N,64] vs cuBLAS ===\n\n");
    
    for (int s=0; s<3; s++) {
        int K=shapes[s][0], N=shapes[s][1], nb=K/32;
        float *hA=(float*)malloc(K*4), *hW=(float*)malloc(K*N*4);
        srand(42);
        for(int i=0;i<K;i++) hA[i]=(float)rand()/RAND_MAX*0.02f-0.01f;
        for(int i=0;i<K*N;i++) hW[i]=(float)rand()/RAND_MAX*0.02f-0.01f;
        
        // Coalesced 64-byte padded [nb, N, 64]
        unsigned char *hQ=(unsigned char*)calloc(nb*N*64,1);
        float *hWd=(float*)calloc(K*N,4);
        for(int col=0;col<N;col++) for(int kb=0;kb<nb;kb++) {
            float *v=hW+col*K+kb*32; float am=0;
            for(int i=0;i<32;i++){float x=fabsf(v[i]);if(x>am)am=x;}
            float d=am>0?am/127.0f:0;
            int off=(kb*N+col)*64;  // COALESCED
            *((__half*)&hQ[off])=__float2half(d);
            for(int i=0;i<32;i++){int8_t q=d>0?(int8_t)roundf(v[i]/d):0;hQ[off+4+i]=(unsigned char)q;hWd[(kb*32+i)*N+col]=d*q;}
        }
        
        float *dA,*dWf,*dC1,*dC2; unsigned char *dWq;
        cudaMalloc(&dA,K*4);cudaMalloc(&dWf,K*N*4);cudaMalloc(&dWq,nb*N*64);
        cudaMalloc(&dC1,N*4);cudaMalloc(&dC2,N*4);
        cudaMemcpy(dA,hA,K*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dWf,hWd,K*N*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dWq,hQ,nb*N*64,cudaMemcpyHostToDevice);
        
        cudaEvent_t t0,t1; cudaEventCreate(&t0);cudaEventCreate(&t1);
        
        for(int i=0;i<20;i++){
            k_vecmat_dp4a_coal<<<(N+255)/256,256,nb*sizeof(q8_1_block)>>>(dA,dWq,dC2,K,N);
            float al=1,be=0;
            cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,1,K,&al,dWf,N,dA,K,&be,dC1,N);
        }
        cudaDeviceSynchronize();
        
        cudaEventRecord(t0);
        for(int i=0;i<ITERS;i++) k_vecmat_dp4a_coal<<<(N+255)/256,256,nb*sizeof(q8_1_block)>>>(dA,dWq,dC2,K,N);
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float dp4a_ms; cudaEventElapsedTime(&dp4a_ms,t0,t1); dp4a_ms/=ITERS;
        
        float al=1,be=0;
        cudaEventRecord(t0);
        for(int i=0;i<ITERS;i++) cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,1,K,&al,dWf,N,dA,K,&be,dC1,N);
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float cublas_ms; cudaEventElapsedTime(&cublas_ms,t0,t1); cublas_ms/=ITERS;
        
        float *hC1=(float*)malloc(N*4),*hC2=(float*)malloc(N*4);
        cudaMemcpy(hC1,dC1,N*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hC2,dC2,N*4,cudaMemcpyDeviceToHost);
        float err=0; for(int i=0;i<N;i++){float d=fabsf(hC1[i]-hC2[i]);if(d>err)err=d;}
        
        printf("[%dx%d] %s:\n", K, N, names[s]);
        printf("  dp4a coalesced: %.3fms  err=%.1e\n", dp4a_ms, err);
        printf("  cuBLAS F32:     %.3fms\n", cublas_ms);
        if (dp4a_ms < cublas_ms) printf("  *** BEATS cuBLAS %.2fx ***\n", cublas_ms/dp4a_ms);
        else printf("  cuBLAS wins %.2fx\n", dp4a_ms/cublas_ms);
        printf("\n");
        
        cudaFree(dA);cudaFree(dWf);cudaFree(dWq);cudaFree(dC1);cudaFree(dC2);
        free(hA);free(hW);free(hQ);free(hWd);free(hC1);free(hC2);
    }
    cublasDestroy(handle);
}
