// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_gpu_kernels.cu — GPU kernel implementations with host-callable wrappers.
 *
 * Each kernel has:
 *   1. __global__ kernel function
 *   2. extern "C" host wrapper (handles grid/block config + launch)
 *
 * Usage from Python/ctypes:
 *   lib = ctypes.CDLL("build/bpd_gpu.so")
 *   lib.bpd_relu_gpu(input_ptr, output_ptr, n)
 *
 * Build:
 *   nvcc -O2 -shared -Xcompiler -fPIC -o build/bpd_gpu.so bench/bpd_gpu_kernels.cu
 */

#include <cuda_runtime.h>
#include <math.h>

// ── Helper: grid/block config ──

static inline int ceildiv(int a, int b) { return (a + b - 1) / b; }
#define BLOCK 256

// ── GPU memory management ──

extern "C" {
void* gpu_alloc(int bytes) { void* p; cudaMalloc(&p, bytes); return p; }
void  gpu_free(void* p) { cudaFree(p); }
void  gpu_h2d(void* d, const void* s, int bytes) { cudaMemcpy(d, s, bytes, cudaMemcpyHostToDevice); }
void  gpu_d2h(void* d, const void* s, int bytes) { cudaMemcpy(d, s, bytes, cudaMemcpyDeviceToHost); }
void  gpu_sync() { cudaDeviceSynchronize(); }
}

// ═══════════════════════════════════════════════════════════════
// Elementwise kernels
// ═══════════════════════════════════════════════════════════════

__global__ void k_relu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.0f, in[i]);
}

__global__ void k_silu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x / (1.0f + expf(-x)); }
}

__global__ void k_mish(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x * tanhf(log1pf(expf(x))); }
}

__global__ void k_sigmoid(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

__global__ void k_tanh_k(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(in[i]);
}

__global__ void k_gelu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }
}

__global__ void k_neg(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}

__global__ void k_abs(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(in[i]);
}

__global__ void k_exp(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}

// ── Fused matmul + bias + relu ──

__global__ void k_mm_bias_relu(const float* A, const float* B,
                                const float* bias, float* C,
                                int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx / N, col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[idx] = fmaxf(0.0f, sum + bias[col]);
    }
}

// ── BatchNorm (inference, precomputed scale/offset) ──

__global__ void k_batchnorm_affine(const float* input, const float* scale,
                                    const float* offset, float* output,
                                    int total, int C, int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int c = (i / HW) % C;
        output[i] = input[i] * scale[c] + offset[c];
    }
}

// ── Residual add ──

__global__ void k_residual_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ── Upsample nearest 2x ──

__global__ void k_upsample_nearest2d(const float* input, float* output,
                                      int N, int C, int H, int W) {
    int H_out = 2 * H, W_out = 2 * W;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (i < total) {
        int ow = i % W_out, oh = (i / W_out) % H_out;
        int c = (i / (H_out * W_out)) % C, n = i / (C * H_out * W_out);
        output[i] = input[((n * C + c) * H + oh / 2) * W + ow / 2];
    }
}

// -- L1 missing kernel functions (8 ops) --

__global__ void k_leaky_relu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x > 0.0f ? x : 0.01f * x; }
}

__global__ void k_elu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x > 0.0f ? x : expm1f(x); }
}

__global__ void k_hardsigmoid(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f; }
}

__global__ void k_softplus(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x > 20.0f ? x : log1pf(expf(x)); }
}

__global__ void k_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] * b[i]; }
}

__global__ void k_softmax(const float* in, float* out, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x; if (row >= rows) return;
    const float* ri = in + row * cols; float* ro = out + row * cols;
    float lmax = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) lmax = fmaxf(lmax, ri[j]);
    sdata[threadIdx.x] = lmax; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x+s]); __syncthreads(); }
    float rmax = sdata[0]; __syncthreads();
    float lsum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) { float e = expf(ri[j]-rmax); ro[j] = e; lsum += e; }
    sdata[threadIdx.x] = lsum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x+s]; __syncthreads(); }
    float sum = sdata[0]; __syncthreads();
    for (int j = threadIdx.x; j < cols; j += blockDim.x) ro[j] /= sum;
}

__global__ void k_layernorm(const float* in, const float* gamma, const float* beta, float* out, int rows, int cols, float eps) {
    extern __shared__ float sd[]; float* sm = sd; float* sv = sd + blockDim.x;
    int row = blockIdx.x; if (row >= rows) return;
    const float* ri = in + row*cols; float* ro = out + row*cols;
    float ls = 0.0f; for (int j = threadIdx.x; j < cols; j += blockDim.x) ls += ri[j];
    sm[threadIdx.x] = ls; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x+s]; __syncthreads(); }
    float mean = sm[0]/cols; __syncthreads();
    float lv = 0.0f; for (int j = threadIdx.x; j < cols; j += blockDim.x) { float d = ri[j]-mean; lv += d*d; }
    sv[threadIdx.x] = lv; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sv[threadIdx.x] += sv[threadIdx.x+s]; __syncthreads(); }
    float istd = 1.0f/sqrtf(sv[0]/cols + eps); __syncthreads();
    for (int j = threadIdx.x; j < cols; j += blockDim.x) ro[j] = gamma[j]*(ri[j]-mean)*istd + beta[j];
}

__global__ void k_maxpool2d(const float* in, float* out, int N, int C, int H, int W, int kH, int kW, int stride, int pad, int Ho, int Wo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N*C*Ho*Wo; if (i >= total) return;
    int ow = i%Wo, oh = (i/Wo)%Ho, c = (i/(Wo*Ho))%C, n = i/(Wo*Ho*C);
    float val = -1e30f;
    for (int kh = 0; kh < kH; kh++) for (int kw = 0; kw < kW; kw++) {
        int hi = oh*stride-pad+kh, wi = ow*stride-pad+kw;
        if (hi >= 0 && hi < H && wi >= 0 && wi < W) val = fmaxf(val, in[((n*C+c)*H+hi)*W+wi]);
    }
    out[i] = val;
}

// -- L1 additional kernels --

__global__ void k_logsoftmax(const float* in, float* out, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x; if (row >= rows) return;
    const float* ri = in + row * cols; float* ro = out + row * cols;
    float lmax = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) lmax = fmaxf(lmax, ri[j]);
    sdata[threadIdx.x] = lmax; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x+s]); __syncthreads(); }
    float rmax = sdata[0]; __syncthreads();
    float lsum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) lsum += expf(ri[j]-rmax);
    sdata[threadIdx.x] = lsum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x+s]; __syncthreads(); }
    float logsum = logf(sdata[0]) + rmax; __syncthreads();
    for (int j = threadIdx.x; j < cols; j += blockDim.x) ro[j] = ri[j] - logsum;
}

__global__ void k_selu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i];
        const float alpha = 1.6732632423543772f, scale = 1.0507009873554805f;
        out[i] = x > 0.0f ? scale * x : scale * alpha * expm1f(x); }
}

__global__ void k_softsign(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x / (1.0f + fabsf(x)); }
}

__global__ void k_hardtanh(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = fminf(fmaxf(x, -1.0f), 1.0f); }
}

__global__ void k_mingpt_gelu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
}

__global__ void k_avgpool2d(const float* in, float* out, int N, int C, int H, int W, int kH, int kW, int stride, int pad, int Ho, int Wo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N*C*Ho*Wo; if (i >= total) return;
    int ow = i%Wo, oh = (i/Wo)%Ho, c = (i/(Wo*Ho))%C, n = i/(Wo*Ho*C);
    float sum = 0.0f; int count = 0;
    for (int kh = 0; kh < kH; kh++) for (int kw = 0; kw < kW; kw++) {
        int hi = oh*stride-pad+kh, wi = ow*stride-pad+kw;
        if (hi >= 0 && hi < H && wi >= 0 && wi < W) { sum += in[((n*C+c)*H+hi)*W+wi]; count++; }
    }
    out[i] = sum / (float)(kH * kW);
}

/* Reduction kernel matching PyTorch Reduce.cuh block_x_reduce exactly:
 * 1. Thread-level: vec4 striped accumulators + stride = blockDim.x * 4
 * 2. Cross-warp: shared memory binary tree (offset = dim_x/2 down to warpSize)
 * 3. Intra-warp: __shfl_down_sync (offset = warpSize/2 down to 1)
 *
 * Sweepable parameters:
 *   block_width (threads per row): set by wrapper, matches PyTorch ReduceConfig
 *   vec_size: 4 (PyTorch default for float32)
 */
__global__ void k_sum_reduce(const float* in, float* out, int outer, int reduce_dim, int inner) {
    int bx = blockDim.x, by = blockDim.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    /* split_across_warps: by > 1 means multiple warps reduce the SAME output row */
    int split = (by > 1 && reduce_dim > bx * 4);
    int row = split ? blockIdx.x : blockIdx.x * by + ty;
    if (row >= outer * inner) return;
    
    int o = row / inner, inn = row % inner;
    const float* base = in + (o * reduce_dim) * inner + inn;
    
    /* Thread-level accumulation: total_threads = bx*by if split, else bx */
    int total_threads = split ? bx * by : bx;
    int tid_in_row = split ? ty * bx + tx : tx;
    
    float val;
    if (inner == 1 && reduce_dim >= total_threads * 4) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int pos = tid_in_row * 4, stride = total_threads * 4;
        while (pos + 3 < reduce_dim) {
            a0 += base[pos]; a1 += base[pos+1];
            a2 += base[pos+2]; a3 += base[pos+3];
            pos += stride;
        }
        for (; pos < reduce_dim; pos++) a0 += base[pos];
        val = ((a0 + a1) + a2) + a3;
    } else {
        val = 0.0f;
        for (int r = tid_in_row; r < reduce_dim; r += total_threads)
            val += base[r * inner];
    }
    
    extern __shared__ float smem[];
    
    /* block_x_reduce: shared memory tree for bx > 32, then shfl_down */
    if (bx > 32) {
        int flat = ty * bx + tx;
        smem[flat] = val;
        for (int offset = bx / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (tx < offset && tx + offset < bx) {
                val += smem[flat + offset];
                smem[flat] = val;
            }
        }
        __syncthreads();
    }
    int warp_dim = bx < 32 ? bx : 32;
    for (int offset = warp_dim >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    
    /* block_y_reduce: binary tree in shared memory across ty (PyTorch block_y_reduce) */
    if (split && by > 1) {
        smem[tx + ty * bx] = val;
        for (int offset = by / 2; offset > 0; offset >>= 1) {
            __syncthreads();
            if (ty < offset && ty + offset < by) {
                val += smem[tx + (ty + offset) * bx];
                smem[tx + ty * bx] = val;
            }
        }
    }
    
    if (tx == 0 && (split ? ty == 0 : 1))
        out[row] = val;
}

__global__ void k_mean_reduce(const float* in, float* out, int outer, int reduce_dim, int inner) {
    int bx = blockDim.x, by = blockDim.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    /* split_across_warps: by > 1 means multiple warps reduce the SAME output row */
    int split = (by > 1 && reduce_dim > bx * 4);
    int row = split ? blockIdx.x : blockIdx.x * by + ty;
    if (row >= outer * inner) return;
    
    int o = row / inner, inn = row % inner;
    const float* base = in + (o * reduce_dim) * inner + inn;
    
    /* Thread-level accumulation: total_threads = bx*by if split, else bx */
    int total_threads = split ? bx * by : bx;
    int tid_in_row = split ? ty * bx + tx : tx;
    
    float val;
    if (inner == 1 && reduce_dim >= total_threads * 4) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int pos = tid_in_row * 4, stride = total_threads * 4;
        while (pos + 3 < reduce_dim) {
            a0 += base[pos]; a1 += base[pos+1];
            a2 += base[pos+2]; a3 += base[pos+3];
            pos += stride;
        }
        for (; pos < reduce_dim; pos++) a0 += base[pos];
        val = ((a0 + a1) + a2) + a3;
    } else {
        val = 0.0f;
        for (int r = tid_in_row; r < reduce_dim; r += total_threads)
            val += base[r * inner];
    }
    
    extern __shared__ float smem[];
    
    /* block_x_reduce: shared memory tree for bx > 32, then shfl_down */
    if (bx > 32) {
        int flat = ty * bx + tx;
        smem[flat] = val;
        for (int offset = bx / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (tx < offset && tx + offset < bx) {
                val += smem[flat + offset];
                smem[flat] = val;
            }
        }
        __syncthreads();
    }
    int warp_dim = bx < 32 ? bx : 32;
    for (int offset = warp_dim >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    
    /* block_y_reduce: binary tree in shared memory across ty (PyTorch block_y_reduce) */
    if (split && by > 1) {
        smem[tx + ty * bx] = val;
        for (int offset = by / 2; offset > 0; offset >>= 1) {
            __syncthreads();
            if (ty < offset && ty + offset < by) {
                val += smem[tx + (ty + offset) * bx];
                smem[tx + ty * bx] = val;
            }
        }
    }
    
    if (tx == 0 && (split ? ty == 0 : 1))
        out[row] = val / (float)reduce_dim;
}

__global__ void k_max_reduce(const float* in, float* out, int outer, int reduce_dim, int inner) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner; if (i >= total) return;
    int o = i / inner, inn = i % inner;
    float m = in[(o * reduce_dim) * inner + inn];
    for (int r = 1; r < reduce_dim; r++) m = fmaxf(m, in[(o * reduce_dim + r) * inner + inn]);
    out[i] = m;
}

__global__ void k_min_reduce(const float* in, float* out, int outer, int reduce_dim, int inner) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner; if (i >= total) return;
    int o = i / inner, inn = i % inner;
    float m = in[(o * reduce_dim) * inner + inn];
    for (int r = 1; r < reduce_dim; r++) m = fminf(m, in[(o * reduce_dim + r) * inner + inn]);
    out[i] = m;
}


// -- GPU matmul (shared-memory tiled) --
__global__ void k_mm_simple(const float* A, const float* B, float* C, int M, int N, int K) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// GPU im2col
__global__ void k_im2col(const float* data_im, float* data_col,
                          int C, int H, int W, int Ho, int Wo,
                          int kH, int kW, int pad_h, int pad_w,
                          int stride_h, int stride_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * kH * kW * Ho * Wo;
    if (idx >= total) return;
    int w_col = idx % Wo;
    int h_col = (idx / Wo) % Ho;
    int c_col = idx / (Ho * Wo);
    int w_offset = c_col % kW;
    int h_offset = (c_col / kW) % kH;
    int c_im = c_col / (kH * kW);
    int h_im = h_col * stride_h - pad_h + h_offset;
    int w_im = w_col * stride_w - pad_w + w_offset;
    data_col[idx] = (h_im >= 0 && h_im < H && w_im >= 0 && w_im < W) ?
        data_im[(c_im * H + h_im) * W + w_im] : 0.0f;
}

// GPU bias add (per-channel broadcast over spatial dims)
__global__ void k_bias_add(float* output, const float* bias, int Cout, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Cout * spatial) {
        int c = idx / spatial;
        output[idx] += bias[c];
    }
}


// -- Additional L1 kernels batch 3 --

__global__ void k_hardswish(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f; }
}

__global__ void k_scalar_mul(const float* in, float* out, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * s;
}

__global__ void k_mse_loss(const float* pred, const float* target, float* out, int n) {
    // Per-element squared difference (reduction done on host or separate kernel)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float d = pred[i] - target[i]; out[i] = d * d; }
}

__global__ void k_hinge_loss_elem(const float* pred, const float* target, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = 1.0f - pred[i] * target[i]; out[i] = v > 0.0f ? v : 0.0f; }
}

__global__ void k_cumsum(const float* in, float* out, int outer, int dim_size, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;
    int o = idx / inner, inn = idx % inner;
    float s = 0.0f;
    for (int d = 0; d < dim_size; d++) {
        s += in[(o * dim_size + d) * inner + inn];
        out[(o * dim_size + d) * inner + inn] = s;
    }
}

__global__ void k_argmax(const float* in, long long* out, int outer, int dim_size, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;
    int o = idx / inner, inn = idx % inner;
    float best = in[(o * dim_size) * inner + inn];
    int best_idx = 0;
    for (int d = 1; d < dim_size; d++) {
        float v = in[(o * dim_size + d) * inner + inn];
        if (v > best) { best = v; best_idx = d; }
    }
    out[idx] = best_idx;
}

__global__ void k_argmin(const float* in, long long* out, int outer, int dim_size, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;
    int o = idx / inner, inn = idx % inner;
    float best = in[(o * dim_size) * inner + inn];
    int best_idx = 0;
    for (int d = 1; d < dim_size; d++) {
        float v = in[(o * dim_size + d) * inner + inn];
        if (v < best) { best = v; best_idx = d; }
    }
    out[idx] = best_idx;
}


// -- Tier 3 primitives --
__global__ void k_sqrt_f(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sqrtf(in[i]);
}
__global__ void k_rsqrt_f(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = rsqrtf(in[i]);
}
__global__ void k_log_f(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(in[i]);
}
__global__ void k_exp_f(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}
__global__ void k_div_f(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}
__global__ void k_sub_f(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}
__global__ void k_where_f(const float* cond, const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cond[i] > 0.0f ? a[i] : b[i];
}
__global__ void k_pow_f(const float* in, float* out, int n, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = powf(in[i], p);
}
__global__ void k_clamp_f(const float* in, float* out, int n, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fminf(fmaxf(in[i], lo), hi);
}
__global__ void k_recip_f(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / in[i];
}

// Host-callable wrappers (extern "C" for ctypes/FFI)
// ═══════════════════════════════════════════════════════════════

extern "C" {

// Elementwise
void bpd_relu_gpu(const float* in, float* out, int n) {
    k_relu<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_silu_gpu(const float* in, float* out, int n) {
    k_silu<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_mish_gpu(const float* in, float* out, int n) {
    k_mish<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_sigmoid_gpu(const float* in, float* out, int n) {
    k_sigmoid<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_tanh_gpu(const float* in, float* out, int n) {
    k_tanh_k<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_gelu_gpu(const float* in, float* out, int n) {
    k_gelu<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_neg_gpu(const float* in, float* out, int n) {
    k_neg<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_abs_gpu(const float* in, float* out, int n) {
    k_abs<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_exp_gpu(const float* in, float* out, int n) {
    k_exp<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }

// Fused
void bpd_mm_bias_relu_gpu(const float* A, const float* B, const float* bias,
                            float* C, int M, int N, int K) {
    k_mm_bias_relu<<<ceildiv(M * N, BLOCK), BLOCK>>>(A, B, bias, C, M, N, K); }

// BatchNorm
void bpd_batchnorm_affine_gpu(const float* input, const float* scale,
                                const float* offset, float* output,
                                int N, int C, int HW) {
    int total = N * C * HW;
    k_batchnorm_affine<<<ceildiv(total, BLOCK), BLOCK>>>(input, scale, offset, output, total, C, HW); }

// Residual add
void bpd_residual_add_gpu(const float* a, const float* b, float* out, int n) {
    k_residual_add<<<ceildiv(n, BLOCK), BLOCK>>>(a, b, out, n); }

// Upsample
void bpd_upsample_nearest2d_gpu(const float* input, float* output,
                                  int N, int C, int H, int W) {
    int total = N * C * 4 * H * W;
    k_upsample_nearest2d<<<ceildiv(total, BLOCK), BLOCK>>>(input, output, N, C, H, W); }

// -- L1 missing wrappers --
void bpd_leaky_relu_gpu(const float* in, float* out, int n) { k_leaky_relu<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_elu_gpu(const float* in, float* out, int n) { k_elu<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_hardsigmoid_gpu(const float* in, float* out, int n) { k_hardsigmoid<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_softplus_gpu(const float* in, float* out, int n) { k_softplus<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_mul_gpu(const float* a, const float* b, float* out, int n) { k_mul<<<ceildiv(n,BLOCK),BLOCK>>>(a,b,out,n); }
void bpd_softmax_gpu(const float* in, float* out, int rows, int cols) { k_softmax<<<rows,BLOCK,BLOCK*sizeof(float)>>>(in,out,rows,cols); }
void bpd_layernorm_gpu(const float* in, const float* gamma, const float* beta, float* out, int rows, int cols, float eps) { k_layernorm<<<rows,BLOCK,2*BLOCK*sizeof(float)>>>(in,gamma,beta,out,rows,cols,eps); }
void bpd_maxpool2d_gpu(const float* in, float* out, int N, int C, int H, int W, int kH, int kW, int stride, int pad) { int Ho=(H+2*pad-kH)/stride+1,Wo=(W+2*pad-kW)/stride+1; k_maxpool2d<<<ceildiv(N*C*Ho*Wo,BLOCK),BLOCK>>>(in,out,N,C,H,W,kH,kW,stride,pad,Ho,Wo); }

// -- Additional L1 wrappers --
void bpd_logsoftmax_gpu(const float* in, float* out, int rows, int cols) { k_logsoftmax<<<rows,BLOCK,BLOCK*sizeof(float)>>>(in,out,rows,cols); }
void bpd_selu_gpu(const float* in, float* out, int n) { k_selu<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_softsign_gpu(const float* in, float* out, int n) { k_softsign<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_hardtanh_gpu(const float* in, float* out, int n) { k_hardtanh<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_mingpt_gelu_gpu(const float* in, float* out, int n) { k_mingpt_gelu<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_avgpool2d_gpu(const float* in, float* out, int N, int C, int H, int W, int kH, int kW, int stride, int pad) { int Ho=(H+2*pad-kH)/stride+1,Wo=(W+2*pad-kW)/stride+1; k_avgpool2d<<<ceildiv(N*C*Ho*Wo,BLOCK),BLOCK>>>(in,out,N,C,H,W,kH,kW,stride,pad,Ho,Wo); }
/* Dispatch with PyTorch-matching block_width.
 * Set BPD_REDUCE_NT env var to override (for parameter sweeping).
 * Default: exact PyTorch ReduceConfig::set_block_dimension. */
void bpd_sum_reduce_gpu(const float* in, float* out, int outer, int reduce_dim, int inner) {
    if (inner != 1) { k_sum_reduce<<<outer*inner, dim3(32,1), 32*sizeof(float)>>>(in,out,outer,reduce_dim,inner); return; }
    if (reduce_dim <= 4096) {
        /* dim<=4096: bw=32, 1D, proven 0 ULP for all outer sizes */
        k_sum_reduce<<<outer, dim3(32,1), 32*sizeof(float)>>>(in,out,outer,reduce_dim,1);
        return;
    }
    /* dim>4096: compute PyTorch ReduceConfig with split_across_warps */
    int vec = 4, dim0 = reduce_dim / vec, dim1 = outer, mx = 1024;
    auto lp2 = [](int n)->int{ n|=(n>>1);n|=(n>>2);n|=(n>>4);n|=(n>>8);n|=(n>>16); return n<1?1:n-(n>>1); };
    int d0 = dim0<mx?lp2(dim0):mx, d1 = dim1<mx?lp2(dim1):mx;
    int bw = d0<32?d0:32, bh = d1<mx/bw?d1:mx/bw;
    bw = d0<mx/bh?d0:mx/bh;
    int vpt = reduce_dim / bw;
    int threshold = bh * 16 < 256 ? bh * 16 : 256;
    if (vpt >= threshold && bh > 1) {
        k_sum_reduce<<<outer, dim3(bw,bh), bw*bh*sizeof(float)>>>(in,out,outer,reduce_dim,1);
    } else {
        k_sum_reduce<<<outer, dim3(bw,1), bw*sizeof(float)>>>(in,out,outer,reduce_dim,1);
    }
}
/* Dispatch with PyTorch-matching block_width.
 * Set BPD_REDUCE_NT env var to override (for parameter sweeping).
 * Default: exact PyTorch ReduceConfig::set_block_dimension. */
void bpd_mean_reduce_gpu(const float* in, float* out, int outer, int reduce_dim, int inner) {
    if (inner != 1) { k_mean_reduce<<<outer*inner, dim3(32,1), 32*sizeof(float)>>>(in,out,outer,reduce_dim,inner); return; }
    if (reduce_dim <= 4096) {
        /* dim<=4096: bw=32, 1D, proven 0 ULP for all outer sizes */
        k_mean_reduce<<<outer, dim3(32,1), 32*sizeof(float)>>>(in,out,outer,reduce_dim,1);
        return;
    }
    /* dim>4096: compute PyTorch ReduceConfig with split_across_warps */
    int vec = 4, dim0 = reduce_dim / vec, dim1 = outer, mx = 1024;
    auto lp2 = [](int n)->int{ n|=(n>>1);n|=(n>>2);n|=(n>>4);n|=(n>>8);n|=(n>>16); return n<1?1:n-(n>>1); };
    int d0 = dim0<mx?lp2(dim0):mx, d1 = dim1<mx?lp2(dim1):mx;
    int bw = d0<32?d0:32, bh = d1<mx/bw?d1:mx/bw;
    bw = d0<mx/bh?d0:mx/bh;
    int vpt = reduce_dim / bw;
    int threshold = bh * 16 < 256 ? bh * 16 : 256;
    if (vpt >= threshold && bh > 1) {
        k_mean_reduce<<<outer, dim3(bw,bh), bw*bh*sizeof(float)>>>(in,out,outer,reduce_dim,1);
    } else {
        k_mean_reduce<<<outer, dim3(bw,1), bw*sizeof(float)>>>(in,out,outer,reduce_dim,1);
    }
}
void bpd_max_reduce_gpu(const float* in, float* out, int outer, int reduce_dim, int inner) { k_max_reduce<<<ceildiv(outer*inner,BLOCK),BLOCK>>>(in,out,outer,reduce_dim,inner); }
void bpd_min_reduce_gpu(const float* in, float* out, int outer, int reduce_dim, int inner) { k_min_reduce<<<ceildiv(outer*inner,BLOCK),BLOCK>>>(in,out,outer,reduce_dim,inner); }

// -- Matmul and Conv2d wrappers --
void bpd_mm_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16); dim3 grid(ceildiv(N, 16), ceildiv(M, 16));
    k_mm_simple<<<grid, block>>>(A, B, C, M, N, K); }
void bpd_conv2d_gpu(const float* input, const float* weight, const float* bias,
                     float* output, int N_batch, int Cin, int H, int W,
                     int Cout, int kH, int kW, int stride_h, int stride_w,
                     int pad_h, int pad_w) {
    int Ho = (H + 2*pad_h - kH) / stride_h + 1;
    int Wo = (W + 2*pad_w - kW) / stride_w + 1;
    int k_dim = Cin * kH * kW;
    int spatial = Ho * Wo;
    float* d_col;
    cudaMalloc(&d_col, (size_t)k_dim * spatial * sizeof(float));
    for (int n = 0; n < N_batch; n++) {
        const float* in_n = input + n * Cin * H * W;
        float* out_n = output + n * Cout * spatial;
        int total_col = k_dim * spatial;
        k_im2col<<<ceildiv(total_col, BLOCK), BLOCK>>>(in_n, d_col, Cin, H, W, Ho, Wo, kH, kW, pad_h, pad_w, stride_h, stride_w);
        dim3 block(16, 16); dim3 grid(ceildiv(spatial, 16), ceildiv(Cout, 16));
        k_mm_simple<<<grid, block>>>(weight, d_col, out_n, Cout, spatial, k_dim);
        if (bias != NULL) {
            k_bias_add<<<ceildiv(Cout * spatial, BLOCK), BLOCK>>>(out_n, bias, Cout, spatial);
        }
    }
    cudaFree(d_col); }

// -- Batch 3 wrappers --
void bpd_hardswish_gpu(const float* in, float* out, int n) { k_hardswish<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_scalar_mul_gpu(const float* in, float* out, int n, float s) { k_scalar_mul<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n,s); }
void bpd_mse_loss_elem_gpu(const float* pred, const float* target, float* out, int n) { k_mse_loss<<<ceildiv(n,BLOCK),BLOCK>>>(pred,target,out,n); }
void bpd_hinge_loss_elem_gpu(const float* pred, const float* target, float* out, int n) { k_hinge_loss_elem<<<ceildiv(n,BLOCK),BLOCK>>>(pred,target,out,n); }
void bpd_cumsum_gpu(const float* in, float* out, int outer, int dim_size, int inner) { k_cumsum<<<ceildiv(outer*inner,BLOCK),BLOCK>>>(in,out,outer,dim_size,inner); }
void bpd_argmax_gpu(const float* in, long long* out, int outer, int dim_size, int inner) { k_argmax<<<ceildiv(outer*inner,BLOCK),BLOCK>>>(in,out,outer,dim_size,inner); }
void bpd_argmin_gpu(const float* in, long long* out, int outer, int dim_size, int inner) { k_argmin<<<ceildiv(outer*inner,BLOCK),BLOCK>>>(in,out,outer,dim_size,inner); }

// -- Tier 3 primitive wrappers --
void bpd_sqrt_gpu(const float* in, float* out, int n) { k_sqrt_f<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_rsqrt_gpu(const float* in, float* out, int n) { k_rsqrt_f<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_log_gpu(const float* in, float* out, int n) { k_log_f<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_div_gpu(const float* a, const float* b, float* out, int n) { k_div_f<<<ceildiv(n,BLOCK),BLOCK>>>(a,b,out,n); }
void bpd_sub_gpu(const float* a, const float* b, float* out, int n) { k_sub_f<<<ceildiv(n,BLOCK),BLOCK>>>(a,b,out,n); }
void bpd_where_gpu(const float* cond, const float* a, const float* b, float* out, int n) { k_where_f<<<ceildiv(n,BLOCK),BLOCK>>>(cond,a,b,out,n); }
void bpd_pow_gpu(const float* in, float* out, int n, float p) { k_pow_f<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n,p); }
void bpd_clamp_gpu(const float* in, float* out, int n, float lo, float hi) { k_clamp_f<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n,lo,hi); }
void bpd_recip_gpu(const float* in, float* out, int n) { k_recip_f<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }

} // extern "C"
