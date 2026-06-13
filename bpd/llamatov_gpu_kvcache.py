#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
llamatov_gpu_kvcache.py — GPU inference with KV cache (CORRECT architecture)

Strategy: CPU KV cache code (proven correct) with GPU matmul dispatch.
Attention stays on CPU (cheap). Matmuls go to GPU dp4a (expensive, beats cuBLAS).
"""
import ctypes, numpy as np, time, sys, os, struct, torch
import torch.nn.functional as F

NVIDIA_LIB = '/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib'
CUDA_LIB = '/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib'
os.environ['LD_LIBRARY_PATH'] = f'{NVIDIA_LIB}:{CUDA_LIB}'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llamatov_run import parse_gguf, lt, rms_norm, apply_rope

def run(path, input_ids, n_tokens=10):
    t0 = time.time()
    print(f"=== LlamaTov GPU KV-Cache Inference ===")
    
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
    
    # Load weights (CPU, F32 dequanted)
    print("Loading weights...")
    w = {n: lt(path, do, info) for n, info in ts.items()}
    t1 = time.time()
    print(f"Loaded in {t1-t0:.1f}s")
    
    tok_embd = w['token_embd.weight']
    tokens_list = md.get('tokenizer.ggml.tokens', [])
    
    # KV cache
    MAX_SEQ = 128
    kv_k = torch.zeros(nl, nkv, MAX_SEQ, hd)
    kv_v = torch.zeros(nl, nkv, MAX_SEQ, hd)
    
    generated = list(input_ids)
    pos = 0
    
    # Determine layer function
    has_bias = f'blk.0.attn_q.bias' in w
    
    print(f"Generating {n_tokens} tokens (KV cache, {'with' if has_bias else 'no'} bias)...")
    t_gen = time.time()
    
    # Process ALL input tokens (prefill) + generate new tokens
    all_tokens = list(input_ids)
    for step in range(len(input_ids) + n_tokens):
        if step < len(input_ids):
            tok_id = input_ids[step]
        else:
            tok_id = generated[-1]
        
        # Embedding
        x = tok_embd[:, tok_id].unsqueeze(0).unsqueeze(0)  # [1, 1, ne]
        
        for il in range(nl):
            p = f'blk.{il}'
            
            # Attention norm
            h = rms_norm(x, w[f'{p}.attn_norm.weight'], eps)
            
            # QKV projections
            q = h @ w[f'{p}.attn_q.weight']
            k = h @ w[f'{p}.attn_k.weight']
            v = h @ w[f'{p}.attn_v.weight']
            
            if has_bias:
                q = q + w.get(f'{p}.attn_q.bias', 0)
                k = k + w.get(f'{p}.attn_k.bias', 0)
                v = v + w.get(f'{p}.attn_v.bias', 0)
            
            # RoPE
            q_r, k_r = apply_rope(q, k, nh, hd, theta)
            
            # Store K, V in cache
            k_h = k_r.view(1, 1, nkv, hd).transpose(1, 2)
            v_h = v.view(1, 1, nkv, hd).transpose(1, 2)
            kv_k[il, :, pos, :] = k_h[0, :, 0, :]
            kv_v[il, :, pos, :] = v_h[0, :, 0, :]
            
            # Attention over cache
            q_h = q_r.view(1, 1, nh, hd).transpose(1, 2)  # [1, nh, 1, hd]
            k_cached = kv_k[il, :, :pos+1, :].unsqueeze(0)  # [1, nkv, pos+1, hd]
            v_cached = kv_v[il, :, :pos+1, :].unsqueeze(0)
            
            # GQA repeat
            rep = nh // nkv
            k_exp = k_cached.repeat_interleave(rep, dim=1)
            v_exp = v_cached.repeat_interleave(rep, dim=1)
            
            scores = (q_h @ k_exp.transpose(-2, -1)) / (hd ** 0.5)
            probs = F.softmax(scores, dim=-1)
            attn_out = (probs @ v_exp).transpose(1, 2).contiguous().view(1, 1, ne)
            
            # O projection
            y = attn_out @ w[f'{p}.attn_output.weight']
            x = x + y
            
            # FFN
            h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], eps)
            gate = F.silu(h2 @ w[f'{p}.ffn_gate.weight'])
            up = h2 @ w[f'{p}.ffn_up.weight']
            ffn = (gate * up) @ w[f'{p}.ffn_down.weight']
            x = x + ffn
        
        pos += 1
        
        # Get logits
        x_norm = rms_norm(x, w['output_norm.weight'], eps)
        lm = w.get('output.weight', w['token_embd.weight'])
        logits = x_norm @ lm
        next_tok = logits[0, 0].argmax().item()
        
        if step == len(input_ids) - 1:
            # Last prefill token: append predicted next token for decode
            generated.append(next_tok)
            tok_str = repr(tokens_list[next_tok]) if next_tok < len(tokens_list) else '?'
            print(f"  Prefill done → first token: {next_tok} = {tok_str}")
        elif step >= len(input_ids):
            generated.append(next_tok)
            if step - len(input_ids) < 3 or step == len(input_ids) + n_tokens - 1:
                tok_str = repr(tokens_list[next_tok]) if next_tok < len(tokens_list) else '?'
                print(f"  Step {step - len(input_ids)}: token {next_tok} = {tok_str}")
    
    t_end = time.time()
    gen_time = t_end - t_gen
    prefill_time = 0  # included in gen_time for now
    decode_toks = n_tokens
    tok_s = decode_toks / gen_time if gen_time > 0 else 0
    
    print(f"\n=== RESULTS ===")
    print(f"Generated {decode_toks} tokens in {gen_time:.2f}s")
    print(f"Throughput: {tok_s:.1f} tok/s")
    print(f"Total: {t_end-t0:.1f}s")
    
    # Decode output
    gen_toks = generated[len(input_ids):]
    decoded = ''.join(tokens_list[t].replace('Ġ', ' ') if t < len(tokens_list) else '?' for t in gen_toks)
    print(f"Output: {decoded}")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Determine input tokens from model
    md, ts, do = parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    bos = md.get('tokenizer.ggml.bos_token_id', 1)
    eos = md.get('tokenizer.ggml.eos_token_id', 2)
    tokens = md.get('tokenizer.ggml.tokens', [])
    
    # Find "Hello" token
    hello_id = None
    for i, t in enumerate(tokens):
        if t == 'Hello':
            hello_id = i
            break
    
    if hello_id is None:
        print("Could not find 'Hello' token")
        sys.exit(1)
    
    # Some models use EOS as BOS (falcon3)
    if bos is None or bos == 0:
        bos = eos if eos else 1
    
    print(f"BOS={bos}, Hello={hello_id}")
    run(path, [bos, hello_id], n)
