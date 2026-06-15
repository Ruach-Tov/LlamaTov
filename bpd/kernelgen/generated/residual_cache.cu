// ===================================================================
// residual_cache (KV-Direct) — generated C/CUDA for ggml/llama.cpp
// Recompute K/V from the cached residual instead of caching K/V (13-27x less KV cache).
// SPDX-License-Identifier: LicenseRef-RTAAL-1.1
// ===================================================================

static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}


static __device__ __forceinline__ float block_reduce_sum(float val, float * buf) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lane_id == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();
    val = lane_id < 8 ? buf[lane_id] : 0.0f;
    val = warp_reduce_sum(val);
    return val;
}


__global__ void k_rms_norm(const float * __restrict__ in, const float * __restrict__ weight, float * __restrict__ out, int cols, float eps) {
    int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * x = in + (row * cols);

    // Step 1: strided partial sum (each of 256 threads)
    float tmp = 0.0f;
    for (int col = tid; col < cols; col += 256) {
        float xi = x[col];
        tmp += xi * xi;
    }

    // Step 2: full-block reduction (composed: warp_reduce_sum + cross-warp + warp_reduce_sum)
    __shared__ float s_sum[8];
    tmp = block_reduce_sum(tmp, s_sum);

    // Step 3: normalize
    float scale = rsqrtf((tmp / cols) + eps);
    for (int col = tid; col < cols; col += 256) {
        out[(row * cols) + col] = (x[col] * scale) * weight[col];
    }
}


__global__ void k_vecmat(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ c, int k_dim, int n_dim) {
    extern __shared__ float sA[];
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
        sA[i] = a[i];
    }
    __syncthreads();

    int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (col >= n_dim) {
        return;
    }

    float sum = 0.0;
    for (int k = 0; k < k_dim; k++) {
        sum = sum + (sA[k] * b[(k * n_dim) + col]);
    }

    c[col] = sum;
}


void kv_direct_recompute(const float * __restrict__ residual, const float * __restrict__ attn_norm_w, const float * __restrict__ proj_w, float * __restrict__ kv_out, float * __restrict__ scratch_normed, int embd, int out_dim, float eps) {
    // KV-Direct recompute: normed = rms_norm(residual); kv = proj_w @ normed
    // Step 1: rms_norm(residual) -> scratch_normed  [1 block, 256 threads]
    k_rms_norm<<<1, 256>>>(residual, attn_norm_w, scratch_normed, embd, eps);
    // Step 2: kv = proj_w @ normed  [out_dim cols blocked, k_dim=embd in shared]
    int grid = (out_dim + 255) / 256;
    int shmem = embd * sizeof(float);
    // vecmat with dynamic shared memory: k_vecmat<<<grid, 256, shmem>>>(...)
    k_vecmat<<<grid, 256, shmem>>>(scratch_normed, proj_w, kv_out, embd, out_dim);
}

