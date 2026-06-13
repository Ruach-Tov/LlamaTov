// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* silu_nvcc.cu — nvcc CUDA-C silu on P4, to confirm cuda-oxide matches the
 * vendor GPU toolchain (both use libdevice __nv_expf). Prints silu(1.0) bits.
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void silu_kernel(const float* x, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        c[i] = v / (1.0f + expf(-v));   // divide form, libdevice expf
    }
}

int main() {
    const int n = 16;
    float xh[16] = {1.0f, -1.0f, 0.0f, 2.0f, -2.0f, 0.5f, -0.5f, 3.0f,
                    -3.0f, 4.0f, -4.0f, 0.1f, -0.1f, 5.0f, -5.0f, 1.5f};
    float *xd, *cd; cudaMalloc(&xd, 64); cudaMalloc(&cd, 64);
    cudaMemcpy(xd, xh, 64, cudaMemcpyHostToDevice);
    silu_kernel<<<1, 16>>>(xd, cd, n);
    cudaDeviceSynchronize();
    float ch[16]; cudaMemcpy(ch, cd, 64, cudaMemcpyDeviceToHost);
    printf("nvcc-GPU silu(1.0) = %.9f  bits=0x%08x\n", ch[0], *(uint32_t*)&ch[0]);
    printf("nvcc-GPU silu(2.0) = %.9f  bits=0x%08x\n", ch[3], *(uint32_t*)&ch[3]);
    return 0;
}
