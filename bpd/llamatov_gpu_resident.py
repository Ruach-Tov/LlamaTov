#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
llamatov_gpu_resident.py — ALL data stays on GPU between ops.
Only CPU↔GPU transfer: embedding in, logits out.
Step 1 toward whole-model kernel fusion.
"""
import ctypes, numpy as np, time, sys, os, torch
import torch.nn.functional as F

NVIDIA_LIB = '/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib'
CUDA_LIB = '/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib'
os.environ['LD_LIBRARY_PATH'] = f'{NVIDIA_LIB}:{CUDA_LIB}'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llamatov_run import parse_gguf, lt, apply_rope
from llamatov_gpu_dp4a import compile_dp4a, quantize_to_q8_padded64_coalesced

def run(path, input_ids, n_tokens=10):
    t0 = time.time()
    print(f"=== LlamaTov GPU-Resident Inference ===")
    
    md, ts, do = parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    nl = md.get(f'{arch}.block_count', 16)
    nh = md.get(f'{arch}.attention.head_count', 32)
    nkv = md.get(f'{arch}.attention.head_count_kv', 8)
    ne = md.get(f'{arch}.embedding_length', 2048)
    hd = ne // nh
    eps = md.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5)
    theta = md.get(f'{arch}.rope.freq_base', 500000.0)
    ff_dim = None
    
    print(f"Arch: {arch}, layers: {nl}, heads: {nh}/{nkv}, embd: {ne}")
    
    # Compile kernels
    lib = compile_dp4a()
    for fn, rt, at in [
        ('gpu_alloc', ctypes.c_void_p, [ctypes.c_int]),
        ('gpu_free', None, [ctypes.c_void_p]),
        ('gpu_copy_h2d', None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
        ('gpu_copy_d2h', None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
        ('gpu_dp4a_coal', None, [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
        ('gpu_rms_norm', None, [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]),
        ('gpu_add', None, [ctypes.c_void_p]*3 + [ctypes.c_int]),
        ('gpu_mul', None, [ctypes.c_void_p]*3 + [ctypes.c_int]),
        ('gpu_silu', None, [ctypes.c_void_p]*2 + [ctypes.c_int]),
        ('gpu_sync', None, []),
    ]:
        getattr(lib, fn).restype = rt; getattr(lib, fn).argtypes = at
    
    # Load weights
    print("Loading weights...")
    w = {n: lt(path, do, info) for n, info in ts.items()}
    t1 = time.time()
    
    # Upload ALL weights to GPU
    print("Uploading weights to GPU...")
    w_gpu = {}  # name → (ptr, K, N) for matmul weights
    w_norm = {}  # name → ptr for norm weights (F32)
    
    for name, tensor in w.items():
        arr = tensor.numpy()
        if len(arr.shape) == 2 and arr.shape[0] % 32 == 0 and arr.shape[1] % 32 == 0:
            K, N = arr.shape
            q8 = quantize_to_q8_padded64_coalesced(arr.T.copy(), N, K)
            ptr = lib.gpu_alloc(len(q8))
            lib.gpu_copy_h2d(ptr, q8.ctypes.data, len(q8))
            w_gpu[name] = (ptr, K, N)
            if ff_dim is None and 'ffn_gate' in name:
                ff_dim = N
        elif len(arr.shape) == 1:
            flat = arr.astype(np.float32)
            ptr = lib.gpu_alloc(flat.nbytes)
            lib.gpu_copy_h2d(ptr, flat.ctypes.data, flat.nbytes)
            w_norm[name] = ptr
    
    # Embedding table on GPU
    tok_embd_np = w['token_embd.weight'].numpy()
    tok_embd_flat = tok_embd_np.astype(np.float32).flatten()
    d_embd_table = lib.gpu_alloc(tok_embd_flat.nbytes)
    lib.gpu_copy_h2d(d_embd_table, tok_embd_flat.ctypes.data, tok_embd_flat.nbytes)
    vocab_size = tok_embd_np.shape[1]
    
    t2 = time.time()
    print(f"Loaded in {t1-t0:.1f}s, uploaded in {t2-t1:.1f}s")
    
    # GPU working buffers — data STAYS on GPU between ops
    d_x = lib.gpu_alloc(ne * 4)
    d_h = lib.gpu_alloc(ne * 4)
    d_q = lib.gpu_alloc(ne * 4)
    d_k = lib.gpu_alloc(nkv * hd * 4)
    d_v = lib.gpu_alloc(nkv * hd * 4)
    d_attn = lib.gpu_alloc(ne * 4)
    d_gate = lib.gpu_alloc(ff_dim * 4)
    d_up = lib.gpu_alloc(ff_dim * 4)
    d_ffn = lib.gpu_alloc(ne * 4)
    d_logits = lib.gpu_alloc(vocab_size * 4)
    
    # Helper: GPU matmul (data already on GPU, stays on GPU)
    def gpu_mm(src_ptr, weight_name, dst_ptr):
        ptr, K, N = w_gpu[weight_name]
        lib.gpu_dp4a_coal(src_ptr, ptr, dst_ptr, K, N)
    
    # Helper: read GPU buffer to CPU numpy
    def gpu_read(ptr, n):
        out = np.zeros(n, dtype=np.float32)
        lib.gpu_copy_d2h(out.ctypes.data, ptr, n * 4)
        return out
    
    # Helper: write CPU numpy to GPU buffer
    def gpu_write(ptr, arr):
        flat = arr.astype(np.float32).flatten()
        lib.gpu_copy_h2d(ptr, flat.ctypes.data, flat.nbytes)
    
    # KV cache (CPU for now — will move to GPU in step 3)
    tokens_list = md.get('tokenizer.ggml.tokens', [])
    MAX_SEQ = 128
    kv_k = torch.zeros(nl, nkv, MAX_SEQ, hd)
    kv_v = torch.zeros(nl, nkv, MAX_SEQ, hd)
    
    generated = list(input_ids)
    pos = 0
    
    print(f"Running ({len(input_ids)} prefill + {n_tokens} decode)...")
    t_gen = time.time()
    
    for step in range(len(input_ids) + n_tokens):
        tok_id = input_ids[step] if step < len(input_ids) else generated[-1]
        
        # Embedding: lookup on CPU, upload once
        emb = tok_embd_np[:, tok_id].astype(np.float32)
        gpu_write(d_x, emb)
        
        for il in range(nl):
            p = f'blk.{il}'
            
            # RMS norm on GPU (data stays on GPU!)
            lib.gpu_rms_norm(d_x, w_norm[f'{p}.attn_norm.weight'], d_h, 1, ne, ctypes.c_float(eps))
            
            # Q/K/V projections on GPU (data stays on GPU!)
            gpu_mm(d_h, f'{p}.attn_q.weight', d_q)
            gpu_mm(d_h, f'{p}.attn_k.weight', d_k)
            gpu_mm(d_h, f'{p}.attn_v.weight', d_v)
            
            # RoPE + Attention: bring Q/K/V to CPU (small tensors)
            lib.gpu_sync()
            q_np = gpu_read(d_q, ne)
            k_np = gpu_read(d_k, nkv * hd)
            v_np = gpu_read(d_v, nkv * hd)
            
            q_t = torch.from_numpy(q_np).unsqueeze(0).unsqueeze(0)
            k_t = torch.from_numpy(k_np).unsqueeze(0).unsqueeze(0)
            v_t = torch.from_numpy(v_np).unsqueeze(0).unsqueeze(0)
            
            q_r, k_r = apply_rope(q_t, k_t, nh, hd, theta, positions=torch.tensor([pos]))
            
            # KV cache
            k_h = k_r.view(1, 1, nkv, hd).transpose(1, 2)
            v_h = v_t.view(1, 1, nkv, hd).transpose(1, 2)
            kv_k[il, :, pos, :] = k_h[0, :, 0, :]
            kv_v[il, :, pos, :] = v_h[0, :, 0, :]
            
            # Attention (CPU)
            q_h = q_r.view(1, 1, nh, hd).transpose(1, 2)
            k_c = kv_k[il, :, :pos+1, :].unsqueeze(0)
            v_c = kv_v[il, :, :pos+1, :].unsqueeze(0)
            rep = nh // nkv
            k_e = k_c.repeat_interleave(rep, dim=1)
            v_e = v_c.repeat_interleave(rep, dim=1)
            scores = (q_h @ k_e.transpose(-2, -1)) / (hd ** 0.5)
            probs = F.softmax(scores, dim=-1)
            attn_out = (probs @ v_e).transpose(1, 2).contiguous().view(1, 1, ne)
            
            # Upload attention result back to GPU
            gpu_write(d_attn, attn_out.numpy().flatten())
            
            # O projection on GPU
            gpu_mm(d_attn, f'{p}.attn_output.weight', d_h)  # reuse d_h as temp
            
            # Residual on GPU
            lib.gpu_add(d_x, d_h, d_x, ne)
            
            # FFN norm on GPU
            lib.gpu_rms_norm(d_x, w_norm[f'{p}.ffn_norm.weight'], d_h, 1, ne, ctypes.c_float(eps))
            
            # FFN on GPU (gate, up, silu, mul, down — all GPU!)
            gpu_mm(d_h, f'{p}.ffn_gate.weight', d_gate)
            lib.gpu_silu(d_gate, d_gate, ff_dim)
            gpu_mm(d_h, f'{p}.ffn_up.weight', d_up)
            lib.gpu_mul(d_gate, d_up, d_gate, ff_dim)
            gpu_mm(d_gate, f'{p}.ffn_down.weight', d_ffn)
            
            # Residual on GPU
            lib.gpu_add(d_x, d_ffn, d_x, ne)
        
        # Output norm + logits on GPU
        lib.gpu_rms_norm(d_x, w_norm['output_norm.weight'], d_h, 1, ne, ctypes.c_float(eps))
        
        lm_name = 'output.weight' if 'output.weight' in w_gpu else 'token_embd.weight'
        if lm_name in w_gpu:
            gpu_mm(d_h, lm_name, d_logits)
        else:
            # Weight-tied: use embedding table matmul
            # Need F32 vecmat for this since embedding might not be Q8
            lib.gpu_sync()
            h_np = gpu_read(d_h, ne)
            logits_np = h_np @ tok_embd_np
            gpu_write(d_logits, logits_np)
        
        lib.gpu_sync()
        logits_np = gpu_read(d_logits, vocab_size)
        next_tok = int(np.argmax(logits_np))
        
        if step == len(input_ids) - 1:
            generated.append(next_tok)
            tok_str = repr(tokens_list[next_tok]) if next_tok < len(tokens_list) else '?'
            print(f"  Prefill → first token: {next_tok} = {tok_str}")
        elif step >= len(input_ids):
            generated.append(next_tok)
            gen_idx = step - len(input_ids)
            if gen_idx < 3 or gen_idx == n_tokens - 1:
                tok_str = repr(tokens_list[next_tok]) if next_tok < len(tokens_list) else '?'
                print(f"  Step {gen_idx}: token {next_tok} = {tok_str}")
    
    t_end = time.time()
    gen_time = t_end - t_gen
    tok_s = n_tokens / gen_time
    
    print(f"\n=== RESULTS ===")
    print(f"Generated: {n_tokens} tokens in {gen_time:.2f}s")
    print(f"Throughput: {tok_s:.1f} tok/s")
    print(f"Previous (round-trip): 12.8 tok/s")
    print(f"Ollama: 89.9 tok/s")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    md, ts, do = parse_gguf(path)
    bos = md.get('tokenizer.ggml.bos_token_id', 1)
    tokens = md.get('tokenizer.ggml.tokens', [])
    hello_id = next((i for i, t in enumerate(tokens) if t == 'Hello'), None)
    if hello_id is None: print("No 'Hello' token"); sys.exit(1)
    print(f"BOS={bos}, Hello={hello_id}")
    run(path, [bos, hello_id], n)
