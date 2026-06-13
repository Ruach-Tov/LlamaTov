// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// BPD upsample_nearest2d — shared library for verification
#include <cuda_runtime.h>

__global__ void k_upsample_nearest2d(const float * __restrict__ input,
                                      float * __restrict__ output,
                                      int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = 2 * H;
    int W_out = 2 * W;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;
    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int c = (idx / (H_out * W_out)) % C;
    int n = idx / (C * (H_out * W_out));
    int ih = oh / 2;
    int iw = ow / 2;
    int in_idx = ((n * C + c) * H + ih) * W + iw;
    output[idx] = input[in_idx];
}

extern "C" {
void bpd_upsample_nearest2d(const float*input, float*output, int N, int C, int H, int W) {
    int total = N * C * (2*H) * (2*W);
    int blocks = (total + 255) / 256;
    k_upsample_nearest2d<<<blocks, 256>>>(input, output, N, C, H, W);
}
void* gpu_alloc(int n){void*p;cudaMalloc(&p,n);return p;}
void gpu_free(void*p){cudaFree(p);}
void gpu_h2d(void*d,const void*s,int n){cudaMemcpy(d,s,n,cudaMemcpyHostToDevice);}
void gpu_d2h(void*d,const void*s,int n){cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost);}
void gpu_sync(){cudaDeviceSynchronize();}
}
