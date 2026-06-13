// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* torch-cfd stencil kernels — CUDA versions for GPU profiling.
 *
 * Each kernel is one thread per grid point, periodic boundary conditions.
 * Uses reciprocal multiply (BPD_STENCIL_RECIPROCAL=1 equivalent) to match
 * GPU floating-point behavior.
 *
 * Compile: nvcc -shared -Xcompiler -fPIC -o bpd_cfd_gpu.so bpd_cfd_stencils.cu
 *
 * Author: medayek (Collective SME, Verification Methodology)
 * Plan: 9da354ba Phase 2e/2g
 */

extern "C" {

__global__ void bpd_forward_diff_y_gpu(const float *u, float *out,
                                        int ny, int nx, float inv_dy) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny && j < nx) {
        int ip1 = (i + 1) % ny;
        out[i * nx + j] = (u[ip1 * nx + j] - u[i * nx + j]) * inv_dy;
    }
}

__global__ void bpd_forward_diff_x_gpu(const float *u, float *out,
                                        int ny, int nx, float inv_dx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny && j < nx) {
        int jp1 = (j + 1) % nx;
        out[i * nx + j] = (u[i * nx + jp1] - u[i * nx + j]) * inv_dx;
    }
}

__global__ void bpd_backward_diff_y_gpu(const float *u, float *out,
                                         int ny, int nx, float inv_dy) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny && j < nx) {
        int im1 = (i - 1 + ny) % ny;
        out[i * nx + j] = (u[i * nx + j] - u[im1 * nx + j]) * inv_dy;
    }
}

__global__ void bpd_backward_diff_x_gpu(const float *u, float *out,
                                         int ny, int nx, float inv_dx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny && j < nx) {
        int jm1 = (j - 1 + nx) % nx;
        out[i * nx + j] = (u[i * nx + j] - u[i * nx + jm1]) * inv_dx;
    }
}

__global__ void bpd_central_diff_y_gpu(const float *u, float *out,
                                        int ny, int nx, float inv_2dy) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny && j < nx) {
        int ip1 = (i + 1) % ny;
        int im1 = (i - 1 + ny) % ny;
        out[i * nx + j] = (u[ip1 * nx + j] - u[im1 * nx + j]) * inv_2dy;
    }
}

__global__ void bpd_central_diff_x_gpu(const float *u, float *out,
                                        int ny, int nx, float inv_2dx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny && j < nx) {
        int jp1 = (j + 1) % nx;
        int jm1 = (j - 1 + nx) % nx;
        out[i * nx + j] = (u[i * nx + jp1] - u[i * nx + jm1]) * inv_2dx;
    }
}

__global__ void bpd_laplacian_2d_gpu(const float *u, float *out,
                                      int ny, int nx,
                                      float inv_dx2, float inv_dy2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny && j < nx) {
        int ip1 = (i + 1) % ny;
        int im1 = (i - 1 + ny) % ny;
        int jp1 = (j + 1) % nx;
        int jm1 = (j - 1 + nx) % nx;
        out[i * nx + j] =
            (u[im1 * nx + j] - 2.0f * u[i * nx + j] + u[ip1 * nx + j]) * inv_dy2 +
            (u[i * nx + jm1] - 2.0f * u[i * nx + j] + u[i * nx + jp1]) * inv_dx2;
    }
}

__global__ void bpd_divergence_2d_gpu(const float *vx, const float *vy,
                                       float *out, int ny, int nx,
                                       float inv_dx, float inv_dy) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny && j < nx) {
        int im1 = (i - 1 + ny) % ny;
        int jm1 = (j - 1 + nx) % nx;
        float dvx_dx = (vx[i * nx + j] - vx[i * nx + jm1]) * inv_dx;
        float dvy_dy = (vy[i * nx + j] - vy[im1 * nx + j]) * inv_dy;
        out[i * nx + j] = dvx_dx + dvy_dy;
    }
}

} /* extern "C" */
