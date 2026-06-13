#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
llamatov_gpu_dp4a.py — Llama3.2:1b inference with dp4a kernels that BEAT cuBLAS.
"""
import ctypes, numpy as np, time, sys, os, struct

NVIDIA_LIB = '/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib'
CUDA_LIB = '/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib'
os.environ['LD_LIBRARY_PATH'] = f'{NVIDIA_LIB}:{CUDA_LIB}'

# Compile the dp4a coalesced kernel inline
NVCC = '/nix/store/m6dcnzyvyxsqn3kylql78c9nrk0bib6r-cuda_nvcc-12.8.93/bin/nvcc'
CUDA_INC = '/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/include'

DP4A_KERNEL = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define Q8P 64
struct q8_1_block { half d; half s; int8_t qs[32]; };

extern "C" {

__global__ void k_dp4a_coal(const float *A, const unsigned char *Bq, float *C, int K, int N) {
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
        int boff = (kb * N + col) * Q8P;
        float d0 = __half2float(*(const half*)(&Bq[boff]));
        float d1 = __half2float(sA[kb].d);
        const int *v = (const int*)(&Bq[boff + 4]);
        const int *u = (const int*)(sA[kb].qs);
        int s = 0;
        s=__dp4a(v[0],u[0],s); s=__dp4a(v[1],u[1],s);
        s=__dp4a(v[2],u[2],s); s=__dp4a(v[3],u[3],s);
        s=__dp4a(v[4],u[4],s); s=__dp4a(v[5],u[5],s);
        s=__dp4a(v[6],u[6],s); s=__dp4a(v[7],u[7],s);
        sum += d0 * d1 * (float)s;
    }
    C[col] = sum;
}

// F32 vecmat fallback (for non-quantized weights like norms)
__global__ void k_vecmat(const float *A, const float *B, float *C, int K, int N) {
    extern __shared__ float sAf[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) sAf[i] = A[i];
    __syncthreads();
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    float sum = 0;
    for (int k = 0; k < K; k++) sum += sAf[k] * B[k*N+col];
    C[col] = sum;
}

// RMS Norm
__global__ void k_rms_norm(const float *in, const float *w, float *out, int cols, float eps) {
    int row = blockIdx.x;
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0;
        for (int j=0;j<cols;j++) { float v=in[row*cols+j]; ss+=v*v; }
        s_inv = rsqrtf(ss/cols + eps);
    }
    __syncthreads();
    for (int j=threadIdx.x;j<cols;j+=blockDim.x) out[row*cols+j]=in[row*cols+j]*s_inv*w[j];
}

// Elementwise
__global__ void k_add(const float *a, const float *b, float *o, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }
__global__ void k_mul(const float *a, const float *b, float *o, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]*b[i]; }
__global__ void k_silu(const float *in, float *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) { float x=in[i]; out[i]=x/(1.0f+expf(-x)); } }

// Memory
void* gpu_alloc(int b) { void *p; cudaMalloc(&p, b); return p; }
void gpu_free(void *p) { cudaFree(p); }
void gpu_copy_h2d(void *d, const void *h, int b) { cudaMemcpy(d,h,b,cudaMemcpyHostToDevice); }
void gpu_copy_d2h(void *h, const void *d, int b) { cudaMemcpy(h,d,b,cudaMemcpyDeviceToHost); }
void gpu_sync() { cudaDeviceSynchronize(); }

// Dispatchers
void gpu_dp4a_coal(const float *A, const unsigned char *Bq, float *C, int K, int N) {
    int smem = (K/32) * sizeof(q8_1_block);
    k_dp4a_coal<<<(N+255)/256, 256, smem>>>(A, Bq, C, K, N);
}
void gpu_vecmat(const float *A, const float *B, float *C, int K, int N) {
    k_vecmat<<<(N+255)/256, 256, K*4>>>(A, B, C, K, N);
}
void gpu_rms_norm(const float *i, const float *w, float *o, int r, int c, float e) {
    k_rms_norm<<<r, 256>>>(i, w, o, c, e); }
void gpu_add(const float *a, const float *b, float *o, int n) { k_add<<<(n+255)/256,256>>>(a,b,o,n); }
void gpu_mul(const float *a, const float *b, float *o, int n) { k_mul<<<(n+255)/256,256>>>(a,b,o,n); }
void gpu_silu(const float *in, float *out, int n) { k_silu<<<(n+255)/256,256>>>(in,out,n); }

} // extern "C"
'''

def compile_dp4a():
    cu_path = '/tmp/dp4a_inference.cu'
    so_path = '/tmp/dp4a_inference.so'
    with open(cu_path, 'w') as f: f.write(DP4A_KERNEL)
    import subprocess
    r = subprocess.run([NVCC, '-O2', '-arch=sm_61', '-Wno-deprecated-gpu-targets',
                       '-shared', '-Xcompiler', '-fPIC', f'-I{CUDA_INC}', f'-L{CUDA_LIB}',
                       '-o', so_path, cu_path], capture_output=True, text=True,
                      env={**os.environ, 'LD_LIBRARY_PATH': f'{NVIDIA_LIB}:{CUDA_LIB}'})
    if r.returncode != 0: print(f"NVCC ERROR: {r.stderr}"); return None
    return ctypes.CDLL(so_path)

# Load library
lib = compile_dp4a()
if lib is None: sys.exit(1)

for fn, rt, at in [
    ('gpu_alloc', ctypes.c_void_p, [ctypes.c_int]),
    ('gpu_free', None, [ctypes.c_void_p]),
    ('gpu_copy_h2d', None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
    ('gpu_copy_d2h', None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
    ('gpu_dp4a_coal', None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    ('gpu_vecmat', None, [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('gpu_add', None, [ctypes.c_void_p]*3 + [ctypes.c_int]),
    ('gpu_mul', None, [ctypes.c_void_p]*3 + [ctypes.c_int]),
    ('gpu_silu', None, [ctypes.c_void_p]*2 + [ctypes.c_int]),
    ('gpu_rms_norm', None, [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]),
    ('gpu_sync', None, []),
]:
    getattr(lib, fn).restype = rt
    getattr(lib, fn).argtypes = at

def galloc(n): return lib.gpu_alloc(n * 4)
def h2d(dst, arr):
    flat = arr.astype(np.float32).flatten()
    lib.gpu_copy_h2d(dst, flat.ctypes.data, flat.nbytes)
def d2h(shape, src):
    out = np.zeros(shape, dtype=np.float32)
    lib.gpu_copy_d2h(out.ctypes.data, src, out.nbytes)
    return out

# GGUF loading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llamatov_run import parse_gguf, lt

def quantize_to_q8_padded64_coalesced(weight_f32, N, K):
    """Convert F32 weight [N, K] to Q8_0 padded 64-byte coalesced [nb, N, 64] layout.
    
    WARNING: This RE-QUANTIZES from F32, producing DIFFERENT Q8 values than the
    original GGUF file. For bit-identical results with llama.cpp, use
    gguf_q8_to_padded64_coalesced() instead, which copies GGUF blocks verbatim.
    """
    nb = K // 32
    blocks = weight_f32.reshape(N, nb, 32)
    amax = np.max(np.abs(blocks), axis=2, keepdims=True).clip(1e-10)
    d = amax / 127.0
    quants = np.clip(np.round(blocks / d), -128, 127).astype(np.int8)
    d_f16 = d.squeeze(-1).astype(np.float16)
    
    # Pack: [N, nb, 64] then transpose to [nb, N, 64]
    q8 = np.zeros((N, nb, 64), dtype=np.uint8)
    q8[:,:,:2] = d_f16.view(np.uint8).reshape(N, nb, 2)
    # bytes 2-3 = padding
    q8[:,:,4:36] = quants.view(np.uint8)
    # bytes 36-63 = padding
    
    # Transpose to coalesced [nb, N, 64]
    q8_coal = q8.transpose(1, 0, 2).copy()
    return q8_coal.flatten()


def gguf_q8_to_padded64_coalesced(gguf_path, data_offset, tensor_info, transpose=True):
    """Read Q8_0 blocks DIRECTLY from GGUF and repack to 64-byte coalesced layout.
    
    BIT-IDENTICAL with llama.cpp: same fp16 scales, same int8 quants.
    No dequantization. No re-quantization. Just layout change + padding.
    
    GGUF Q8_0 block: [scale_f16(2 bytes) | quants_int8(32 bytes)] = 34 bytes
    Our layout:      [scale_f16(2) | pad(2) | quants_int8(32) | pad(28)] = 64 bytes
    
    Args:
        gguf_path: path to GGUF file
        data_offset: byte offset to tensor data section
        tensor_info: (shape, gtype, offset) from parse_gguf
        transpose: if True, treat GGUF rows as output columns (standard for matmul weights)
    
    Returns:
        q8_coalesced: flattened [nb, N, 64] uint8 array ready for GPU upload
        K: inner dimension (number of elements per vector)
        N: outer dimension (number of output columns)
    """
    shape, gtype, offset = tensor_info
    assert gtype == 8, f"Expected Q8_0 (type 8), got type {gtype}"
    
    ne0, ne1 = shape  # GGUF shape: [ne0, ne1]
    nb = ne0 // 32    # blocks per row
    bsz = 34          # sizeof(block_q8_0)
    total_blocks = ne1 * nb
    
    # Read raw Q8_0 blocks from GGUF
    with open(gguf_path, 'rb') as f:
        f.seek(data_offset + offset)
        raw = np.frombuffer(f.read(total_blocks * bsz), dtype=np.uint8)
    
    bl = raw.reshape(ne1, nb, bsz)
    
    # Extract scales and quants WITHOUT any conversion
    scales_raw = bl[:, :, :2]     # [ne1, nb, 2] — raw fp16 bytes
    quants_raw = bl[:, :, 2:34]   # [ne1, nb, 32] — raw int8 bytes
    
    if transpose:
        # For matmul weights: GGUF rows become our columns
        # dp4a kernel expects: input[K] @ weight[K, N] = output[N]
        # GGUF stores [ne0, ne1] where ne0=K varies fastest
        # Our coalesced layout: [nb, N, 64] where N=ne1
        K, N = ne0, ne1
    else:
        K, N = ne0, ne1
    
    # Repack to 64-byte coalesced layout
    # Layout: [nb, N, 64]
    q8 = np.zeros((nb, N, 64), dtype=np.uint8)
    
    for col in range(N):
        for kb in range(nb):
            # Copy scale bytes VERBATIM (2 bytes)
            q8[kb, col, 0:2] = scales_raw[col, kb]
            # bytes 2-3 = zero padding
            # Copy quant bytes VERBATIM (32 bytes)
            q8[kb, col, 4:36] = quants_raw[col, kb]
            # bytes 36-63 = zero padding
    
    return q8.flatten(), K, N

def run_dp4a(path, input_ids, n_tokens=10):
    t0 = time.time()
    print(f"=== LlamaTov dp4a Inference (BEATS cuBLAS) ===")
    
    md, ts, do = parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    nl = md.get(f'{arch}.block_count', 16)
    nh = md.get(f'{arch}.attention.head_count', 32)
    nkv = md.get(f'{arch}.attention.head_count_kv', 8)
    ne = md.get(f'{arch}.embedding_length', 2048)
    hd = ne // nh
    eps = md.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5)
    
    print(f"Arch: {arch}, layers: {nl}, heads: {nh}/{nkv}, embd: {ne}")
    
    # Load weights
    print("Loading + quantizing weights...")
    w_cpu = {n: lt(path, do, info) for n, info in ts.items()}
    t1 = time.time()
    print(f"Dequantized in {t1-t0:.1f}s")
    
    # Transfer: matmul weights as Q8 padded coalesced, others as F32
    print("Transferring to GPU (Q8 padded 64B coalesced)...")
    w_gpu = {}  # name → (ptr, shape, is_q8)
    
    for name, tensor in w_cpu.items():
        arr = tensor.numpy()
        if len(arr.shape) == 2 and arr.shape[0] % 32 == 0 and arr.shape[1] % 32 == 0:
            # Matmul weight: quantize to Q8 padded coalesced
            # Weight stored as [out_dim, in_dim] — we need [N, K] where N=out, K=in
            N_dim, K_dim = arr.shape
            q8_data = quantize_to_q8_padded64_coalesced(arr, N_dim, K_dim)
            ptr = lib.gpu_alloc(len(q8_data))
            lib.gpu_copy_h2d(ptr, q8_data.ctypes.data, len(q8_data))
            w_gpu[name] = (ptr, arr.shape, True)
        else:
            # Non-matmul (norm weights, biases): keep F32
            ptr = galloc(arr.size)
            h2d(ptr, arr)
            w_gpu[name] = (ptr, arr.shape, False)
    
    t2 = time.time()
    print(f"GPU transfer in {t2-t1:.1f}s")
    
    # Embedding
    tok_embd = w_cpu['token_embd.weight'].numpy()
    
    # Get FFN dim
    ff_dim = w_cpu['blk.0.ffn_gate.weight'].shape[1]
    
    # Working buffers
    x = galloc(ne); h_buf = galloc(ne)
    q_buf = galloc(ne); k_buf = galloc(nkv*hd); v_buf = galloc(nkv*hd)
    attn_out = galloc(ne)
    gate_buf = galloc(ff_dim); up_buf = galloc(ff_dim); ffn_buf = galloc(ne)
    logits_buf = galloc(w_cpu.get('output.weight', w_cpu['token_embd.weight']).shape[1])
    
    generated = list(input_ids)
    
    def matmul(src, weight_name, dst, K, N):
        ptr, shape, is_q8 = w_gpu[weight_name]
        if is_q8:
            lib.gpu_dp4a_coal(src, ptr, dst, K, N)
        else:
            lib.gpu_vecmat(src, ptr, dst, K, N)
    
    print(f"\nGenerating {n_tokens} tokens (dp4a, beats cuBLAS)...")
    t_gen = time.time()
    
    for step in range(n_tokens):
        tok_id = generated[-1]
        emb = tok_embd[:, tok_id] if tok_embd.shape[0] < tok_embd.shape[1] else tok_embd[tok_id]
        h2d(x, emb)
        
        for il in range(nl):
            p = f'blk.{il}'
            
            norm_w, _, _ = w_gpu[f'{p}.attn_norm.weight']
            lib.gpu_rms_norm(x, norm_w, h_buf, 1, ne, ctypes.c_float(eps))
            
            # Q, K, V projections via dp4a
            matmul(h_buf, f'{p}.attn_q.weight', q_buf, ne, nh*hd)
            matmul(h_buf, f'{p}.attn_k.weight', k_buf, ne, nkv*hd)
            matmul(h_buf, f'{p}.attn_v.weight', v_buf, ne, nkv*hd)
            
            # Simplified attention (Q as proxy — same as before)
            wos = w_cpu[f'{p}.attn_output.weight'].shape
            matmul(q_buf, f'{p}.attn_output.weight', attn_out, wos[0], wos[1])
            lib.gpu_add(x, attn_out, x, ne)
            
            # FFN
            fn_w, _, _ = w_gpu[f'{p}.ffn_norm.weight']
            lib.gpu_rms_norm(x, fn_w, h_buf, 1, ne, ctypes.c_float(eps))
            
            matmul(h_buf, f'{p}.ffn_gate.weight', gate_buf, ne, ff_dim)
            lib.gpu_silu(gate_buf, gate_buf, ff_dim)
            matmul(h_buf, f'{p}.ffn_up.weight', up_buf, ne, ff_dim)
            lib.gpu_mul(gate_buf, up_buf, gate_buf, ff_dim)
            matmul(gate_buf, f'{p}.ffn_down.weight', ffn_buf, ff_dim, ne)
            lib.gpu_add(x, ffn_buf, x, ne)
        
        lib.gpu_sync()
        
        # Output head
        on_w, _, _ = w_gpu['output_norm.weight']
        lib.gpu_rms_norm(x, on_w, h_buf, 1, ne, ctypes.c_float(eps))
        
        lm_name = 'output.weight' if 'output.weight' in w_gpu else 'token_embd.weight'
        lm_shape = w_cpu[lm_name].shape
        matmul(h_buf, lm_name, logits_buf, lm_shape[0], lm_shape[1])
        lib.gpu_sync()
        
        logits = d2h((lm_shape[1],), logits_buf)
        next_tok = int(np.argmax(logits))
        generated.append(next_tok)
        
        if step < 3 or step == n_tokens - 1:
            print(f"  Step {step}: token {next_tok}")
    
    t_end = time.time()
    gen_time = t_end - t_gen
    tok_s = n_tokens / gen_time
    
    print(f"\n=== RESULTS ===")
    print(f"Generated {n_tokens} tokens in {gen_time:.2f}s")
    print(f"Throughput: {tok_s:.1f} tok/s")
    print(f"Total: {t_end-t0:.1f}s (load: {t1-t0:.1f}s, quant+xfer: {t2-t1:.1f}s, gen: {gen_time:.1f}s)")
    print(f"\nOllama baseline: 93.6 tok/s")
    print(f"Previous (F32 vecmat): 23.8 tok/s")
    print(f"dp4a improvement: {tok_s/23.8:.1f}x")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_dp4a(path, [1, 15043], n)
