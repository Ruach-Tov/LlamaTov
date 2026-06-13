// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
#include <cuda_runtime.h>
#include <math.h>

#define TILE 32

// ═══════════════════════════════════════════════════════════════
// BPD-Generated Kernel Library for LlamaTov Inference
// Compiled: nvcc -arch=sm_61 (Tesla P4)
// ═══════════════════════════════════════════════════════════════

extern "C" {

// Tiled matmul: C[M,N] = A[M,K] @ B[K,N]
__global__ void k_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        if (row < M && t*TILE+threadIdx.x < K) sA[threadIdx.y][threadIdx.x] = A[row*K+t*TILE+threadIdx.x];
        else sA[threadIdx.y][threadIdx.x] = 0.0f;
        if (t*TILE+threadIdx.y < K && col < N) sB[threadIdx.y][threadIdx.x] = B[(t*TILE+threadIdx.y)*N+col];
        else sB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// Elementwise ops
__global__ void k_relu(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x > 0 ? x : 0; }
}
__global__ void k_silu(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x / (1.0f + expf(-x)); }
}
__global__ void k_gelu(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); }
}
__global__ void k_add(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
__global__ void k_mul(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
__global__ void k_scale(const float *in, float *out, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * s;
}

// RMS Norm: out[i] = (in[i] / rms) * weight[i]  
__global__ void k_rms_norm(const float *in, const float *weight, float *out, int cols, float eps) {
    int row = blockIdx.x;
    // Compute sum of squares for this row
    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = in[row * cols + j];
        sum_sq += v * v;
    }
    // Warp reduction
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum_sq = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __syncthreads();
    float rms = sqrtf(shared[0] / cols + eps);
    // Normalize
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        out[row * cols + j] = (in[row * cols + j] / rms) * weight[j];
    }
}

// ═══════════════════════════════════════════════════════════════
// C API WRAPPERS (keep data on GPU between calls)
// ═══════════════════════════════════════════════════════════════

// GPU memory management
void* gpu_alloc(int bytes) { void *p; cudaMalloc(&p, bytes); return p; }
void gpu_free(void *p) { cudaFree(p); }
void gpu_copy_h2d(void *dst, const void *src, int bytes) { cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); }
void gpu_copy_d2h(void *dst, const void *src, int bytes) { cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); }
void gpu_sync() { cudaDeviceSynchronize(); }

// Op dispatchers (data stays on GPU!)
void gpu_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    k_matmul<<<grid, block>>>(A, B, C, M, N, K);
}

void gpu_add(const float *a, const float *b, float *out, int n) {
    k_add<<<(n+255)/256, 256>>>(a, b, out, n);
}

void gpu_mul(const float *a, const float *b, float *out, int n) {
    k_mul<<<(n+255)/256, 256>>>(a, b, out, n);
}

void gpu_silu(const float *in, float *out, int n) {
    k_silu<<<(n+255)/256, 256>>>(in, out, n);
}

void gpu_gelu(const float *in, float *out, int n) {
    k_gelu<<<(n+255)/256, 256>>>(in, out, n);
}

void gpu_rms_norm(const float *in, const float *weight, float *out, int rows, int cols, float eps) {
    k_rms_norm<<<rows, min(cols, 256)>>>(in, weight, out, cols, eps);
}

void gpu_scale(const float *in, float *out, float s, int n) {
    k_scale<<<(n+255)/256, 256>>>(in, out, s, n);
}

} // extern "C"

// Additional kernels for inference
extern "C" {

// Embedding lookup: out[i,:] = table[indices[i],:]
__global__ void k_embed(const float *table, const int *indices, float *out, int seq_len, int embd) {
    int pos = blockIdx.x;  // which token
    if (pos >= seq_len) return;
    int tok = indices[pos];
    for (int j = threadIdx.x; j < embd; j += blockDim.x) {
        out[pos * embd + j] = table[tok * embd + j];
    }
}

// Device-to-device copy
void gpu_copy_d2d(void *dst, const void *src, int bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}

// Embedding dispatch
void gpu_embed(const float *table, const int *indices, float *out, int seq_len, int embd) {
    k_embed<<<seq_len, min(embd, 256)>>>(table, indices, out, seq_len, embd);
}

// Allocate int array on GPU and copy indices
void* gpu_alloc_int(const int *h_data, int count) {
    int *d;
    cudaMalloc(&d, count * sizeof(int));
    cudaMemcpy(d, h_data, count * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}

// Softmax (row-wise, for attention)
__global__ void k_softmax(float *data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float *r = data + row * cols;
    
    // Find max for numerical stability
    float max_val = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        if (r[j] > max_val) max_val = r[j];
    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    __shared__ float smax;
    if (threadIdx.x == 0) smax = max_val;
    __syncthreads();
    max_val = smax;
    
    // Exp and sum
    float sum = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        r[j] = expf(r[j] - max_val);
        sum += r[j];
    }
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = sum;
    __syncthreads();
    
    // Normalize
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        r[j] /= ssum;
}

void gpu_softmax(float *data, int rows, int cols) {
    k_softmax<<<rows, min(cols, 256)>>>(data, rows, cols);
}

// Causal mask fill (set upper triangle to -inf)
__global__ void k_causal_mask(float *att, int T, int stride) {
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    if (row < T && col < T && col > row) {
        att[row * stride + col] = -1e30f;
    }
}

void gpu_causal_mask(float *att, int n_head, int T) {
    int stride = T;  // stride between columns in attention matrix
    for (int h = 0; h < n_head; h++) {
        dim3 grid(T, (T+255)/256);
        k_causal_mask<<<grid, 256>>>(att + h * T * T, T, stride);
    }
}

} // extern "C"

// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
// Simple correct version: one thread does the reduction, all threads normalize
extern "C" {
__global__ void k_layer_norm(const float *in, const float *weight, const float *bias,
                              float *out, int cols, float eps) {
    int row = blockIdx.x;
    const float *x = in + row * cols;
    float *y = out + row * cols;
    
    __shared__ float s_mean, s_inv_std;
    
    // Thread 0 computes mean and variance (simple, correct)
    if (threadIdx.x == 0) {
        float sum = 0, sum_sq = 0;
        for (int j = 0; j < cols; j++) {
            sum += x[j];
        }
        s_mean = sum / cols;
        for (int j = 0; j < cols; j++) {
            float d = x[j] - s_mean;
            sum_sq += d * d;
        }
        s_inv_std = rsqrtf(sum_sq / cols + eps);
    }
    __syncthreads();
    
    // All threads normalize in parallel
    float mean = s_mean;
    float inv_std = s_inv_std;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        y[j] = (x[j] - mean) * inv_std * weight[j] + bias[j];
    }
}

void gpu_layer_norm(const float *in, const float *weight, const float *bias,
                     float *out, int rows, int cols, float eps) {
    k_layer_norm<<<rows, min(cols, 256)>>>(in, weight, bias, out, cols, eps);
}
} // extern "C"

// RoPE: Rotary Position Embedding
// Applies rotation to pairs of elements: (x0,x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
extern "C" {
__global__ void k_rope(float *q, float *k, int seq_len, int n_head, int head_dim, float theta_base) {
    int pos = blockIdx.x;  // position in sequence
    int h = blockIdx.y;    // head index
    if (pos >= seq_len) return;
    
    int half = head_dim / 2;
    float *q_head = q + (pos * n_head + h) * head_dim;
    float *k_head = k + (pos * n_head + h) * head_dim;
    
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(theta_base, (float)(2 * i) / head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        
        // Q rotation
        float q0 = q_head[i], q1 = q_head[i + half];
        q_head[i] = q0 * cos_a - q1 * sin_a;
        q_head[i + half] = q0 * sin_a + q1 * cos_a;
        
        // K rotation (may have fewer heads for GQA)
    }
}

// Separate version for K with different n_head_kv
__global__ void k_rope_k(float *k, int seq_len, int n_head_kv, int head_dim, float theta_base) {
    int pos = blockIdx.x;
    int h = blockIdx.y;
    if (pos >= seq_len || h >= n_head_kv) return;
    
    int half = head_dim / 2;
    float *k_head = k + (pos * n_head_kv + h) * head_dim;
    
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(theta_base, (float)(2 * i) / head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        
        float k0 = k_head[i], k1 = k_head[i + half];
        k_head[i] = k0 * cos_a - k1 * sin_a;
        k_head[i + half] = k0 * sin_a + k1 * cos_a;
    }
}

void gpu_rope_q(float *q, int seq_len, int n_head, int head_dim, float theta) {
    dim3 grid(seq_len, n_head);
    k_rope<<<grid, min(head_dim/2, 256)>>>(q, NULL, seq_len, n_head, head_dim, theta);
}

void gpu_rope_k(float *k, int seq_len, int n_head_kv, int head_dim, float theta) {
    dim3 grid(seq_len, n_head_kv);
    k_rope_k<<<grid, min(head_dim/2, 256)>>>(k, seq_len, n_head_kv, head_dim, theta);
}

} // extern "C"


// Optimized matmul: TILE_K=128 for higher arithmetic intensity
// 7.1 FLOPs/byte vs 0.47 for naive (15x improvement)
#define TILE_K_OPT 128
#define TILE_MN 32

extern "C" {
__global__ void k_matmul_opt(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE_MN][TILE_K_OPT];
    __shared__ float sB[TILE_K_OPT][TILE_MN];
    int row = blockIdx.y * TILE_MN + threadIdx.y;
    int col = blockIdx.x * TILE_MN + threadIdx.x;
    float sum = 0.0f;
    
    for (int kt = 0; kt < K; kt += TILE_K_OPT) {
        // Cooperative load: each thread loads 4 elements of A and B
        for (int i = 0; i < 4; i++) {
            int kk = i * 32 + threadIdx.x;
            if (row < M && kt + kk < K)
                sA[threadIdx.y][kk] = A[row * K + kt + kk];
            else
                sA[threadIdx.y][kk] = 0.0f;
            
            int kr = i * 32 + threadIdx.y;
            if (kt + kr < K && col < N)
                sB[kr][threadIdx.x] = B[(kt + kr) * N + col];
            else
                sB[kr][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        for (int k = 0; k < TILE_K_OPT && kt + k < K; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

void gpu_matmul_opt(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(TILE_MN, TILE_MN);
    dim3 grid((N+TILE_MN-1)/TILE_MN, (M+TILE_MN-1)/TILE_MN);
    k_matmul_opt<<<grid, block>>>(A, B, C, M, N, K);
}
} // extern "C"


// Specialized vector-matrix multiply for M=1 decode path
// Matches cuBLAS on FFN shapes, within 1.2x on projections
extern "C" {
__global__ void k_vecmat(const float * __restrict__ A,
                          const float * __restrict__ B,
                          float * __restrict__ C,
                          int K, int N) {
    extern __shared__ float sA[];
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        sA[i] = A[i];
    __syncthreads();
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += sA[k] * B[k * N + col];
    C[col] = sum;
}

void gpu_vecmat(const float *A, const float *B, float *C, int K, int N) {
    int smem = K * sizeof(float);
    k_vecmat<<<(N+255)/256, 256, smem>>>(A, B, C, K, N);
}
} // extern "C"
