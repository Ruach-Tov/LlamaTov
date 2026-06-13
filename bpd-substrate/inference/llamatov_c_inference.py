#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
llamatov_c_inference.py — Thin Python wrapper around C inference loop.
Python only loads weights. C does ALL kernel dispatching.
"""
import ctypes, numpy as np, time, sys, os, struct

NVIDIA_LIB = os.environ.get('NVIDIA_LIB', '/usr/lib/x86_64-linux-gnu')
CUDA_LIB = os.environ.get('CUDA_LIB', '/usr/local/cuda/lib64')
os.environ['LD_LIBRARY_PATH'] = f'{NVIDIA_LIB}:{CUDA_LIB}'

lib = ctypes.CDLL("/tmp/llamatov_inference.so")

# C struct mirrors
class LayerWeights(ctypes.Structure):
    _fields_ = [
        ("attn_norm", ctypes.c_void_p),
        ("q", ctypes.c_void_p), ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p), ("o", ctypes.c_void_p),
        ("ffn_norm", ctypes.c_void_p),
        ("gate", ctypes.c_void_p), ("up", ctypes.c_void_p), ("down", ctypes.c_void_p),
        ("q_K", ctypes.c_int), ("q_N", ctypes.c_int),
        ("k_K", ctypes.c_int), ("k_N", ctypes.c_int),
        ("v_K", ctypes.c_int), ("v_N", ctypes.c_int),
        ("o_K", ctypes.c_int), ("o_N", ctypes.c_int),
        ("gate_K", ctypes.c_int), ("gate_N", ctypes.c_int),
        ("up_K", ctypes.c_int), ("up_N", ctypes.c_int),
        ("down_K", ctypes.c_int), ("down_N", ctypes.c_int),
        # Optional biases (qwen2 arch) — NULL if absent
        ("q_bias", ctypes.c_void_p), ("k_bias", ctypes.c_void_p), ("v_bias", ctypes.c_void_p),
    ]

class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("n_layers", ctypes.c_int), ("n_embd", ctypes.c_int),
        ("n_head", ctypes.c_int), ("n_head_kv", ctypes.c_int),
        ("head_dim", ctypes.c_int), ("ff_dim", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("max_seq", ctypes.c_int),
        ("eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
    ]

# KV cache: per-layer K/V GPU buffers. The struct holds two arrays of
# n_layers void* pointers (cast to float* on the C side). Each pointer
# is to a [max_seq * n_kv * head_dim] F32 buffer allocated via gpu_alloc.
class KVCache(ctypes.Structure):
    _fields_ = [
        ("k_cache", ctypes.POINTER(ctypes.c_void_p)),
        ("v_cache", ctypes.POINTER(ctypes.c_void_p)),
    ]

class WorkBuffers(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p), ("h", ctypes.c_void_p),
        ("q", ctypes.c_void_p), ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p), ("attn_out", ctypes.c_void_p),
        ("gate", ctypes.c_void_p), ("up", ctypes.c_void_p), ("ffn", ctypes.c_void_p),
        ("logits", ctypes.c_void_p), ("argmax_result", ctypes.c_void_p),
    ]

# C signature (post-KV-cache integration, commit c93f9c9eb):
#   int generate_token(layers, cfg, buf, kv, position,
#                       embedding, output_norm_w,
#                       lm_head, lm_f32, lm_K, lm_N)
lib.generate_token.restype = ctypes.c_int
lib.generate_token.argtypes = [
    ctypes.POINTER(LayerWeights),     # layers
    ctypes.POINTER(ModelConfig),      # cfg
    ctypes.POINTER(WorkBuffers),      # buf
    ctypes.POINTER(KVCache),          # kv
    ctypes.c_int,                     # position
    ctypes.c_void_p,                  # embedding (GPU ptr to [n_embd] F32)
    ctypes.c_void_p,                  # output_norm_w (F32 norm weight)
    ctypes.c_void_p,                  # lm_head (Q8, NULL for weight-tied)
    ctypes.c_void_p,                  # lm_f32 (F32 fallback for weight-tied)
    ctypes.c_int, ctypes.c_int,       # lm_K, lm_N
]
lib.gpu_alloc.restype = ctypes.c_void_p
lib.gpu_alloc.argtypes = [ctypes.c_int]
lib.gpu_copy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.gpu_copy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.gpu_sync.argtypes = []

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llamatov_run import parse_gguf, lt

def q8_pad64_coalesced(arr):
    """Convert F32 [N,K] to Q8 padded 64B coalesced [nb,N,64]."""
    N, K = arr.shape
    nb = K // 32
    blocks = arr.reshape(N, nb, 32)
    amax = np.max(np.abs(blocks), axis=2, keepdims=True).clip(1e-10)
    d = amax / 127.0
    quants = np.clip(np.round(blocks / d), -128, 127).astype(np.int8)
    d_f16 = d.squeeze(-1).astype(np.float16)
    q8 = np.zeros((N, nb, 64), dtype=np.uint8)
    q8[:,:,:2] = d_f16.view(np.uint8).reshape(N, nb, 2)
    q8[:,:,4:36] = quants.view(np.uint8)
    return q8.transpose(1, 0, 2).copy().flatten()

def upload_f32(arr):
    flat = arr.astype(np.float32).flatten()
    ptr = lib.gpu_alloc(flat.nbytes)
    lib.gpu_copy_h2d(ptr, flat.ctypes.data, flat.nbytes)
    return ptr

def upload_q8(arr):
    q8 = q8_pad64_coalesced(arr)
    ptr = lib.gpu_alloc(len(q8))
    lib.gpu_copy_h2d(ptr, q8.ctypes.data, len(q8))
    return ptr, arr.shape

def run(path, input_ids, n_tokens=20, max_seq=2048):
    t0 = time.time()
    print(f"=== LlamaTov C Inference Loop (KV cache + RoPE + attention on GPU) ===")

    md, ts, do = parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    cfg = ModelConfig()
    cfg.n_layers = md.get(f'{arch}.block_count', 16)
    cfg.n_embd = md.get(f'{arch}.embedding_length', 2048)
    cfg.n_head = md.get(f'{arch}.attention.head_count', 32)
    cfg.n_head_kv = md.get(f'{arch}.attention.head_count_kv', 8)
    cfg.head_dim = cfg.n_embd // cfg.n_head
    cfg.eps = md.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5)
    cfg.vocab_size = md.get(f'{arch}.vocab_size', 128256)
    cfg.max_seq = max_seq
    cfg.rope_theta = md.get(f'{arch}.rope.freq_base', 500000.0)

    print(f"Arch: {arch}, layers: {cfg.n_layers}, heads: {cfg.n_head}/{cfg.n_head_kv}, embd: {cfg.n_embd}")
    print(f"max_seq: {cfg.max_seq}, rope_theta: {cfg.rope_theta}")
    
    # Load weights
    print("Loading weights...")
    w_cpu = {n: lt(path, do, info) for n, info in ts.items()}
    cfg.ff_dim = w_cpu['blk.0.ffn_gate.weight'].shape[1]
    t1 = time.time()
    
    # Upload to GPU
    print("Uploading to GPU (Q8 padded 64B coalesced)...")
    layers = (LayerWeights * cfg.n_layers)()
    for il in range(cfg.n_layers):
        p = f'blk.{il}'
        lw = layers[il]
        lw.attn_norm = upload_f32(w_cpu[f'{p}.attn_norm.weight'].numpy())
        
        for wname, field_ptr, field_K, field_N in [
            ('attn_q.weight', 'q', 'q_K', 'q_N'),
            ('attn_k.weight', 'k', 'k_K', 'k_N'),
            ('attn_v.weight', 'v', 'v_K', 'v_N'),
            ('attn_output.weight', 'o', 'o_K', 'o_N'),
            ('ffn_gate.weight', 'gate', 'gate_K', 'gate_N'),
            ('ffn_up.weight', 'up', 'up_K', 'up_N'),
            ('ffn_down.weight', 'down', 'down_K', 'down_N'),
        ]:
            arr = w_cpu[f'{p}.{wname}'].numpy()
            ptr, shape = upload_q8(arr)
            setattr(lw, field_ptr, ptr)
            setattr(lw, field_K, shape[0])
            setattr(lw, field_N, shape[1])
        
        lw.ffn_norm = upload_f32(w_cpu[f'{p}.ffn_norm.weight'].numpy())

        # Optional QKV biases (qwen2 family). NULL if absent.
        for bias_name, field in [('attn_q.bias', 'q_bias'),
                                  ('attn_k.bias', 'k_bias'),
                                  ('attn_v.bias', 'v_bias')]:
            key = f'{p}.{bias_name}'
            if key in w_cpu:
                setattr(lw, field, upload_f32(w_cpu[key].numpy()))
            else:
                setattr(lw, field, None)

    # Output norm + lm_head
    out_norm = upload_f32(w_cpu['output_norm.weight'].numpy())
    if 'output.weight' in w_cpu:
        # Separate output head
        lm_arr = w_cpu['output.weight'].numpy()
        lm_ptr, lm_shape = upload_q8(lm_arr)
        lm_f32_ptr = None
    else:
        # Weight-tied: reuse token_embd. Upload Q8 for the dp4a path.
        # (The C side has a TODO for proper F32 vecmat — for now we go
        # through Q8 and accept precision loss on large vocabs.)
        lm_arr = w_cpu['token_embd.weight'].numpy()
        lm_ptr, lm_shape = upload_q8(lm_arr)
        lm_f32_ptr = None
    
    # Working buffers
    buf = WorkBuffers()
    ne = cfg.n_embd; ff = cfg.ff_dim; vocab = lm_shape[1]
    buf.x = lib.gpu_alloc(ne*4); buf.h = lib.gpu_alloc(ne*4)
    buf.q = lib.gpu_alloc(ne*4); buf.k = lib.gpu_alloc(cfg.n_head_kv*cfg.head_dim*4)
    buf.v = lib.gpu_alloc(cfg.n_head_kv*cfg.head_dim*4)
    buf.attn_out = lib.gpu_alloc(ne*4)
    buf.gate = lib.gpu_alloc(ff*4); buf.up = lib.gpu_alloc(ff*4); buf.ffn = lib.gpu_alloc(ne*4)
    buf.logits = lib.gpu_alloc(vocab*4)
    buf.argmax_result = lib.gpu_alloc(4)  # single int

    # KV cache: per-layer K/V buffers on GPU.
    # Layout: [max_seq * n_kv * head_dim] floats per layer.
    # The KVCache struct holds two arrays of n_layers void* pointers.
    kv_bytes_per_layer = cfg.max_seq * cfg.n_head_kv * cfg.head_dim * 4
    k_ptrs = (ctypes.c_void_p * cfg.n_layers)()
    v_ptrs = (ctypes.c_void_p * cfg.n_layers)()
    for il in range(cfg.n_layers):
        k_ptrs[il] = lib.gpu_alloc(kv_bytes_per_layer)
        v_ptrs[il] = lib.gpu_alloc(kv_bytes_per_layer)
    kv = KVCache()
    kv.k_cache = k_ptrs
    kv.v_cache = v_ptrs
    kv_total_mb = (kv_bytes_per_layer * 2 * cfg.n_layers) / (1024 * 1024)
    print(f"KV cache allocated: {kv_total_mb:.1f} MB ({cfg.n_layers} layers × max_seq={cfg.max_seq})")

    # Embedding table on CPU. emb_gpu is a scratch buffer for the current token's
    # embedding (one row of token_embd) — refreshed per step via H2D.
    tok_embd = w_cpu['token_embd.weight'].numpy()
    emb_gpu = lib.gpu_alloc(ne * 4)

    t2 = time.time()
    print(f"Loaded in {t1-t0:.1f}s, uploaded in {t2-t1:.1f}s")

    # Sequence position is the absolute count of tokens seen so far.
    # For prefill of an input_ids sequence, run generate_token for each
    # input token (incrementing position each call). The token returned
    # after the last prefill step is the first decoded token.
    position = 0
    generated = list(input_ids)

    print(f"\nPrefilling {len(input_ids)} input tokens then generating {n_tokens} new tokens...")
    t_gen = time.time()

    # Prefill: feed each input token sequentially through the loop so KV
    # cache fills with correct positional K/V at each step.
    for step, tok_id in enumerate(input_ids):
        emb = tok_embd[:, tok_id] if tok_embd.shape[0] < tok_embd.shape[1] else tok_embd[tok_id]
        emb = emb.astype(np.float32)
        lib.gpu_copy_h2d(emb_gpu, emb.ctypes.data, ne * 4)
        token = lib.generate_token(
            layers, ctypes.byref(cfg), ctypes.byref(buf),
            ctypes.byref(kv), position,
            emb_gpu, out_norm,
            lm_ptr, lm_f32_ptr, lm_shape[0], lm_shape[1])
        position += 1
        # The token returned from the last prefill step is the first decoded token.
        if step == len(input_ids) - 1:
            generated.append(token)
            if step < 5:
                print(f"  Prefill step {step}: next token = {token}")

    # Decode: each step uses the last generated token as input.
    for step in range(n_tokens - 1):  # -1 because last prefill already produced first decode
        tok_id = generated[-1]
        emb = tok_embd[:, tok_id] if tok_embd.shape[0] < tok_embd.shape[1] else tok_embd[tok_id]
        emb = emb.astype(np.float32)
        lib.gpu_copy_h2d(emb_gpu, emb.ctypes.data, ne * 4)
        token = lib.generate_token(
            layers, ctypes.byref(cfg), ctypes.byref(buf),
            ctypes.byref(kv), position,
            emb_gpu, out_norm,
            lm_ptr, lm_f32_ptr, lm_shape[0], lm_shape[1])
        position += 1
        generated.append(token)
        if step < 3 or step == n_tokens - 2:
            print(f"  Decode step {step}: token {token}")
    
    lib.gpu_sync()
    t_end = time.time()
    gen_time = t_end - t_gen
    total_steps = len(input_ids) + (n_tokens - 1)  # prefill + decode steps
    tok_s = total_steps / gen_time

    print(f"\n=== RESULTS ===")
    print(f"Generated {n_tokens} new tokens (plus {len(input_ids)} prefill steps)")
    print(f"  in {gen_time:.2f}s total")
    print(f"  total steps: {total_steps}, throughput: {tok_s:.1f} tok/s")
    print(f"")
    print(f"  Previous CPU+KV reference (Python dispatch): 12.8 tok/s")
    print(f"  Previous placeholder-attention C loop:       35.6 tok/s (numerically wrong)")
    print(f"  Projected target with full GPU stack:        30-40 tok/s")
    print(f"  Ollama target:                                93.6 tok/s")
    print(f"")
    print(f"Tokens generated: {generated[len(input_ids):]}")
    print(f"\nTotal: {t_end-t0:.1f}s")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'os.environ.get('GGUF_MODEL_PATH', 'model.gguf')'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run(path, [1, 15043], n)
