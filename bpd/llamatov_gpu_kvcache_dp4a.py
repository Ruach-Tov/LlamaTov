#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
llamatov_gpu_kvcache_dp4a.py — CORRECT KV cache + GPU dp4a matmul

Step 1: Replace CPU matmul with GPU dp4a for weight projections
Keep: attention on CPU (cheap), norms on CPU, embedding on CPU
GPU: Q/K/V/O projections + FFN gate/up/down (the expensive matmuls)
"""
import ctypes, numpy as np, time, sys, os, torch
import torch.nn.functional as F

NVIDIA_LIB = '/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib'
CUDA_LIB = '/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib'
os.environ['LD_LIBRARY_PATH'] = f'{NVIDIA_LIB}:{CUDA_LIB}'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llamatov_run import parse_gguf, lt, rms_norm, apply_rope
from llamatov_gpu_dp4a import compile_dp4a, quantize_to_q8_padded64_coalesced, gguf_q8_to_padded64_coalesced

def run(path, input_ids, n_tokens=10):
    t0 = time.time()
    print(f"=== LlamaTov GPU dp4a + KV Cache (CORRECT + FAST) ===")
    
    md, ts, do = parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    nl = md.get(f'{arch}.block_count', 16)
    nh = md.get(f'{arch}.attention.head_count', 32)
    nkv = md.get(f'{arch}.attention.head_count_kv', 8)
    ne = md.get(f'{arch}.embedding_length', 2048)
    hd = ne // nh
    eps = md.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5)
    theta = md.get(f'{arch}.rope.freq_base', 500000.0)
    
    print(f"Arch: {arch}, layers: {nl}, heads: {nh}/{nkv}, embd: {ne}")
    
    # Compile dp4a kernels
    lib = compile_dp4a()
    for fn, rt, at in [
        ('gpu_alloc', ctypes.c_void_p, [ctypes.c_int]),
        ('gpu_copy_h2d', None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
        ('gpu_copy_d2h', None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
        ('gpu_dp4a_coal', None, [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
        ('gpu_rms_norm', None, [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]),
        ('gpu_sync', None, []),
    ]:
        getattr(lib, fn).restype = rt; getattr(lib, fn).argtypes = at
    
    # Load weights
    print("Loading weights...")
    w = {n: lt(path, do, info) for n, info in ts.items()}
    t1 = time.time()
    
    # Upload matmul weights as Q8 padded coalesced to GPU
    # BIT-IDENTICAL: for Q8_0 tensors, copy GGUF blocks verbatim (no re-quantization)
    print("Uploading weights to GPU (Q8 padded, bit-identical)...")
    w_gpu = {}  # name → (gpu_ptr, K, N)
    matmul_names = []
    for il in range(nl):
        p = f'blk.{il}'
        for suffix in ['attn_q.weight', 'attn_k.weight', 'attn_v.weight', 
                        'attn_output.weight', 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight']:
            name = f'{p}.{suffix}'
            if name in ts:
                shape, gtype, offset = ts[name]
                if gtype == 8:  # Q8_0: direct copy from GGUF (bit-identical)
                    q8, K, N = gguf_q8_to_padded64_coalesced(path, do, ts[name])
                    ptr = lib.gpu_alloc(len(q8))
                    lib.gpu_copy_h2d(ptr, q8.ctypes.data, len(q8))
                    w_gpu[name] = (ptr, K, N)
                elif name in w:  # Other quant types: dequant then re-quantize
                    arr = w[name].numpy()
                    K, N = arr.shape
                    q8 = quantize_to_q8_padded64_coalesced(arr.T.copy(), N, K)
                    ptr = lib.gpu_alloc(len(q8))
                    lib.gpu_copy_h2d(ptr, q8.ctypes.data, len(q8))
                    w_gpu[name] = (ptr, K, N)
                matmul_names.append(name)
    
    # Output head
    lm_name = 'output.weight' if 'output.weight' in w else 'token_embd.weight'
    lm = w[lm_name].numpy()
    K_lm, N_lm = lm.shape
    # For output head: CPU F32 matmul (avoids Q8 precision loss on large vocab)
    # Keep ptr_lm as None; logits computed on CPU
    ptr_lm = None
    
    t2 = time.time()
    print(f"Loaded {len(matmul_names)} weights in {t1-t0:.1f}s, uploaded in {t2-t1:.1f}s")
    
    # GPU buffers for matmul I/O
    max_dim = max(ne, max(w[f'blk.0.ffn_gate.weight'].shape[1], 
                          w[f'blk.0.ffn_down.weight'].shape[0]))
    d_in = lib.gpu_alloc(max_dim * 4)
    d_out = lib.gpu_alloc(max_dim * 4)
    d_logits = lib.gpu_alloc(N_lm * 4)
    
    def gpu_matmul(input_vec, weight_name):
        """CPU input → GPU matmul → CPU output"""
        ptr, K, N = w_gpu[weight_name]
        inp = input_vec.numpy().flatten().astype(np.float32)
        lib.gpu_copy_h2d(d_in, inp.ctypes.data, K * 4)
        lib.gpu_dp4a_coal(d_in, ptr, d_out, K, N)
        lib.gpu_sync()
        out = np.zeros(N, dtype=np.float32)
        lib.gpu_copy_d2h(out.ctypes.data, d_out, N * 4)
        return torch.from_numpy(out).unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    
    # KV cache + inference
    tok_embd = w['token_embd.weight']
    tokens_list = md.get('tokenizer.ggml.tokens', [])
    MAX_SEQ = 128
    kv_k = torch.zeros(nl, nkv, MAX_SEQ, hd)
    kv_v = torch.zeros(nl, nkv, MAX_SEQ, hd)
    
    generated = list(input_ids)
    pos = 0
    ff_dim = w['blk.0.ffn_gate.weight'].shape[1]
    
    print(f"Running inference ({len(input_ids)} prefill + {n_tokens} decode)...")
    t_gen = time.time()
    
    for step in range(len(input_ids) + n_tokens):
        if step < len(input_ids):
            tok_id = input_ids[step]
        else:
            tok_id = generated[-1]
        
        x = tok_embd[:, tok_id].unsqueeze(0).unsqueeze(0)
        
        for il in range(nl):
            p = f'blk.{il}'
            h = rms_norm(x, w[f'{p}.attn_norm.weight'], eps)
            
            # GPU dp4a matmul for Q/K/V
            q = gpu_matmul(h, f'{p}.attn_q.weight')
            k = gpu_matmul(h, f'{p}.attn_k.weight')
            v = gpu_matmul(h, f'{p}.attn_v.weight')
            
            # Add biases if present (qwen2, deepseek-r1)
            if f'{p}.attn_q.bias' in w:
                q = q + w[f'{p}.attn_q.bias']
            if f'{p}.attn_k.bias' in w:
                k = k + w[f'{p}.attn_k.bias']
            if f'{p}.attn_v.bias' in w:
                v = v + w[f'{p}.attn_v.bias']
            
            # RoPE (CPU, cheap)
            q_r, k_r = apply_rope(q, k, nh, hd, theta, positions=torch.tensor([pos]))
            
            # KV cache store
            k_h = k_r.view(1, 1, nkv, hd).transpose(1, 2)
            v_h = v.view(1, 1, nkv, hd).transpose(1, 2)
            kv_k[il, :, pos, :] = k_h[0, :, 0, :]
            kv_v[il, :, pos, :] = v_h[0, :, 0, :]
            
            # Attention (CPU, cheap for short sequences)
            q_h = q_r.view(1, 1, nh, hd).transpose(1, 2)
            k_c = kv_k[il, :, :pos+1, :].unsqueeze(0)
            v_c = kv_v[il, :, :pos+1, :].unsqueeze(0)
            rep = nh // nkv
            k_e = k_c.repeat_interleave(rep, dim=1)
            v_e = v_c.repeat_interleave(rep, dim=1)
            scores = (q_h @ k_e.transpose(-2, -1)) / (hd ** 0.5)
            probs = F.softmax(scores, dim=-1)
            attn_out = (probs @ v_e).transpose(1, 2).contiguous().view(1, 1, ne)
            
            # O projection (GPU dp4a)
            y = gpu_matmul(attn_out, f'{p}.attn_output.weight')
            x = x + y
            
            # FFN (GPU dp4a for matmuls, CPU for activation)
            h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], eps)
            gate = gpu_matmul(h2, f'{p}.ffn_gate.weight')
            gate = F.silu(gate)
            up = gpu_matmul(h2, f'{p}.ffn_up.weight')
            ffn = gpu_matmul(gate * up, f'{p}.ffn_down.weight')
            x = x + ffn
        
        pos += 1
        
        # Logits via CPU F32 matmul (bit-identical, avoids Q8 precision loss)
        x_norm = rms_norm(x, w['output_norm.weight'], eps)
        h_np = x_norm.numpy().flatten().astype(np.float32)
        # w[lm_name] from lt() is [K, N] already (lt transposes internally)
        # For vecmat: h[K] @ W[K, N] = logits[N]
        logits_np = h_np @ lm
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
    
    # Compare against Ollama
    print(f"\n=== RESULTS ===")
    print(f"Prefill: {len(input_ids)} tokens")
    print(f"Generated: {n_tokens} tokens in {gen_time:.2f}s")
    print(f"Throughput: {tok_s:.1f} tok/s")
    print(f"Total: {t_end-t0:.1f}s")
    
    gen_toks = generated[len(input_ids):]
    decoded = ''.join(tokens_list[t].replace('Ġ', ' ') if t < len(tokens_list) else '?' for t in gen_toks)
    print(f"Output: {decoded}")
    print(f"\nOllama reference: , I'm looking for a reliable and trustworthy online")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    md, ts, do = parse_gguf(path)
    bos = md.get('tokenizer.ggml.bos_token_id', None)
    eos = md.get('tokenizer.ggml.eos_token_id', None)
    tokens = md.get('tokenizer.ggml.tokens', [])
    hello_id = next((i for i, t in enumerate(tokens) if t == 'Hello'), None)
    if hello_id is None: print("No 'Hello' token"); sys.exit(1)
    
    # Some models use EOS as BOS (e.g., falcon3)
    if bos is None or bos == 0:
        bos = eos if eos else 1
    
    print(f"BOS={bos}, Hello={hello_id}")
    run(path, [bos, hello_id], n)
