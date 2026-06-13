#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
test_ollama_match.py — Compare our C inference loop (with llama-compat RMSNorm)
against Ollama token-by-token.

Uses generate_token() from llamatov_inference.so which has the patched
llama.cpp-compatible RMSNorm reduction order.
"""
import ctypes, numpy as np, struct, os, sys, json, subprocess
sys.path.insert(0, "/tmp")

NVIDIA_LIB = "/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib"
CUDA_LIB = "/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib"
os.environ["LD_LIBRARY_PATH"] = f"{NVIDIA_LIB}:{CUDA_LIB}"

from llamatov_run import parse_gguf, lt
from llamatov_gpu_dp4a import gguf_q8_to_padded64_coalesced

path = "${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
md, ts, data_offset = parse_gguf(path)

# Model config
n_layers = md.get('llama.block_count', 16)
n_embd = md.get('llama.embedding_length', 2048)
n_head = md.get('llama.attention.head_count', 32)
n_head_kv = md.get('llama.attention.head_count_kv', 8)
head_dim = n_embd // n_head
ff_dim = md.get('llama.feed_forward_length', 8192)
vocab_size = md.get('llama.vocab_size', md.get('tokenizer.ggml.tokens', []))
if isinstance(vocab_size, list):
    vocab_size = len(vocab_size)
max_seq = 2048
eps = md.get('llama.attention.layer_norm_rms_epsilon', 1e-5)
rope_theta = md.get('llama.rope.freq_base', 500000.0)

print(f"Model: {n_layers} layers, {n_embd} embd, {n_head} heads, {n_head_kv} kv_heads")
print(f"  ff={ff_dim}, vocab={vocab_size}, eps={eps}, rope_theta={rope_theta}")

# Load library
lib = ctypes.CDLL("/tmp/llamatov_inference.so")
lib.gpu_alloc.restype = ctypes.c_void_p
lib.gpu_alloc.argtypes = [ctypes.c_int]
lib.gpu_copy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.gpu_sync.argtypes = []

# Define structs matching C
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
        ("q_bias", ctypes.c_void_p), ("k_bias", ctypes.c_void_p), ("v_bias", ctypes.c_void_p),
    ]

class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("n_layers", ctypes.c_int), ("n_embd", ctypes.c_int),
        ("n_head", ctypes.c_int), ("n_head_kv", ctypes.c_int),
        ("head_dim", ctypes.c_int), ("ff_dim", ctypes.c_int),
        ("vocab_size", ctypes.c_int), ("max_seq", ctypes.c_int),
        ("eps", ctypes.c_float), ("rope_theta", ctypes.c_float),
    ]

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
        ("logits", ctypes.c_void_p),
        ("argmax_result", ctypes.c_void_p),
    ]

def upload_f32(data):
    """Upload float32 numpy array to GPU, return device pointer."""
    d = lib.gpu_alloc(len(data) * 4)
    lib.gpu_copy_h2d(d, data.ctypes.data, len(data) * 4)
    return d

def upload_bytes(data):
    """Upload raw bytes to GPU."""
    d = lib.gpu_alloc(len(data))
    lib.gpu_copy_h2d(d, data.ctypes.data, len(data))
    return d

print("\nLoading weights to GPU...")

# Load per-layer weights
layers = (LayerWeights * n_layers)()
for i in range(n_layers):
    p = f"blk.{i}"
    
    # Norm weights (F32)
    an = lt(path, data_offset, ts[f"{p}.attn_norm.weight"]).numpy().flatten().astype(np.float32)
    fn = lt(path, data_offset, ts[f"{p}.ffn_norm.weight"]).numpy().flatten().astype(np.float32)
    layers[i].attn_norm = upload_f32(an)
    layers[i].ffn_norm = upload_f32(fn)
    
    # Q8 matmul weights
    for name, attr_k, attr_n, attr_ptr in [
        ("attn_q.weight", "q_K", "q_N", "q"),
        ("attn_k.weight", "k_K", "k_N", "k"),
        ("attn_v.weight", "v_K", "v_N", "v"),
        ("attn_output.weight", "o_K", "o_N", "o"),
        ("ffn_gate.weight", "gate_K", "gate_N", "gate"),
        ("ffn_up.weight", "up_K", "up_N", "up"),
        ("ffn_down.weight", "down_K", "down_N", "down"),
    ]:
        q8, K, N = gguf_q8_to_padded64_coalesced(path, data_offset, ts[f"{p}.{name}"])
        setattr(layers[i], attr_k, K)
        setattr(layers[i], attr_n, N)
        setattr(layers[i], attr_ptr, upload_bytes(q8))
    
    # No biases for llama
    layers[i].q_bias = 0
    layers[i].k_bias = 0
    layers[i].v_bias = 0
    
    if i == 0: print(f"  Layer 0: q=[{layers[0].q_K}×{layers[0].q_N}]")

# Model config
cfg = ModelConfig()
cfg.n_layers = n_layers; cfg.n_embd = n_embd; cfg.n_head = n_head
cfg.n_head_kv = n_head_kv; cfg.head_dim = head_dim; cfg.ff_dim = ff_dim
cfg.vocab_size = vocab_size; cfg.max_seq = max_seq
cfg.eps = eps; cfg.rope_theta = rope_theta

# KV cache
kv = KVCache()
kv_k = (ctypes.c_void_p * n_layers)()
kv_v = (ctypes.c_void_p * n_layers)()
kv_size = max_seq * n_head_kv * head_dim * 4
for i in range(n_layers):
    kv_k[i] = lib.gpu_alloc(kv_size)
    kv_v[i] = lib.gpu_alloc(kv_size)
kv.k_cache = ctypes.cast(kv_k, ctypes.POINTER(ctypes.c_void_p))
kv.v_cache = ctypes.cast(kv_v, ctypes.POINTER(ctypes.c_void_p))

# Work buffers
buf = WorkBuffers()
buf.x = lib.gpu_alloc(n_embd * 4)
buf.h = lib.gpu_alloc(n_embd * 4)
buf.q = lib.gpu_alloc(n_embd * 4)
buf.k = lib.gpu_alloc(n_head_kv * head_dim * 4)
buf.v = lib.gpu_alloc(n_head_kv * head_dim * 4)
buf.attn_out = lib.gpu_alloc(n_embd * 4)
buf.gate = lib.gpu_alloc(ff_dim * 4)
buf.up = lib.gpu_alloc(ff_dim * 4)
buf.ffn = lib.gpu_alloc(ff_dim * 4)
buf.logits = lib.gpu_alloc(vocab_size * 4)
buf.argmax_result = lib.gpu_alloc(4)

# Output norm + lm_head
out_norm = lt(path, data_offset, ts["output_norm.weight"]).numpy().flatten().astype(np.float32)
d_out_norm = upload_f32(out_norm)

# Check if lm_head exists or is weight-tied to token_embd
if "output.weight" in ts:
    lm_q8, lm_K, lm_N = gguf_q8_to_padded64_coalesced(path, data_offset, ts["output.weight"])
    d_lm_head = upload_bytes(lm_q8)
    d_lm_f32 = ctypes.c_void_p(0)
else:
    # Weight-tied: use token_embd.weight as F32
    lm_f32 = lt(path, data_offset, ts["token_embd.weight"]).numpy().astype(np.float32)
    d_lm_f32 = upload_f32(lm_f32.flatten())
    d_lm_head = ctypes.c_void_p(0)
    lm_K = n_embd
    lm_N = vocab_size

# Token embedding table
w_emb = lt(path, data_offset, ts["token_embd.weight"]).numpy().astype(np.float32)  # [n_embd, vocab]

# Setup generate_token
lib.generate_token.restype = ctypes.c_int
lib.generate_token.argtypes = [
    ctypes.POINTER(LayerWeights), ctypes.POINTER(ModelConfig),
    ctypes.POINTER(WorkBuffers), ctypes.POINTER(KVCache),
    ctypes.c_int, ctypes.c_void_p,  # position, embedding
    ctypes.c_void_p,  # output_norm_w
    ctypes.c_void_p,  # lm_head (Q8)
    ctypes.c_void_p,  # lm_f32 (weight-tied)
    ctypes.c_int, ctypes.c_int,  # lm_K, lm_N
]

print("Weights loaded. Running inference...\n")

# BOS token
bos = 128000
hello = 9906
prompt = [bos, hello]

# Tokenizer for decoding
tokens_list = md.get('tokenizer.ggml.tokens', [])

def decode_token(tid):
    if tid < len(tokens_list):
        t = tokens_list[tid]
        if isinstance(t, bytes): t = t.decode('utf-8', errors='replace')
        return t.replace('Ġ', ' ').replace('Ċ', '\n')
    return f"<{tid}>"

our_tokens = []

# Prefill: process BOS and Hello
for pos, tid in enumerate(prompt):
    emb = w_emb[:, tid].copy()
    d_emb = upload_f32(emb)
    token = lib.generate_token(
        layers, ctypes.byref(cfg), ctypes.byref(buf), ctypes.byref(kv),
        pos, d_emb, d_out_norm,
        d_lm_head if d_lm_head else ctypes.c_void_p(0),
        d_lm_f32 if d_lm_f32 else ctypes.c_void_p(0),
        lm_K, lm_N)
    print(f"  Prefill pos={pos} (token {tid}='{decode_token(tid)}') → next={token}='{decode_token(token)}'")
    last_token = token

our_tokens.append(last_token)

# Generate 9 more tokens
for step in range(9):
    pos = len(prompt) + step
    emb = w_emb[:, last_token].copy()
    d_emb = upload_f32(emb)
    token = lib.generate_token(
        layers, ctypes.byref(cfg), ctypes.byref(buf), ctypes.byref(kv),
        pos, d_emb, d_out_norm,
        d_lm_head if d_lm_head else ctypes.c_void_p(0),
        d_lm_f32 if d_lm_f32 else ctypes.c_void_p(0),
        lm_K, lm_N)
    print(f"  Step {step}: token {token}='{decode_token(token)}'")
    our_tokens.append(token)
    last_token = token

our_text = ''.join(decode_token(t) for t in our_tokens)
print(f"\nOur output: {repr(our_text)}")

# Get Ollama output
r = subprocess.run(["curl", "-s", "http://localhost:11434/api/generate", "-d",
    json.dumps({"model":"llama3.2:1b","prompt":"Hello","stream":False,
                "options":{"num_predict":10,"temperature":0},"raw":True})],
    capture_output=True, text=True)
ollama_text = json.loads(r.stdout).get("response","")
print(f"Ollama:     {repr(ollama_text)}")

# Compare
if our_text == ollama_text:
    print(f"\n{'='*60}")
    print("TOKEN-IDENTICAL WITH OLLAMA! 🎉")
    print(f"{'='*60}")
else:
    print(f"\nDIVERGE at some point. Investigating...")
    # Find first divergence
    for i, (a, b) in enumerate(zip(our_text, ollama_text)):
        if a != b:
            print(f"  First difference at char {i}: ours='{a}' ollama='{b}'")
            break
