# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""L.1.10 — Full transformer block composition and forward pass tests.

Tests verify:
  1. Single block self-consistency (run twice with same input → same output)
  2. Single block residual structure (output != input, output is finite)
  3. Full forward pass produces valid logits and argmax
  4. Full forward pass determinism (same input → same output)
  5. Multi-token prefill produces finite outputs for all tokens
  6. KV cache is populated correctly after a forward pass
  7. Incremental decode: single-token decode after prefill uses KV cache

All tests use RANDOM F32 weights (not real model weights) — this tests
the composition wiring, buffer management, and kernel interop. Bit-identical
verification against llama.cpp requires the Ruach Tov fixture.
"""

import ctypes, numpy as np, os, sys, struct

BPD_CPU_SO = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
lib = ctypes.CDLL(BPD_CPU_SO)
print(f"BPD_CPU_SO: {BPD_CPU_SO}")

# ── Type aliases ──────────────────────────────────────────────────────────
c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_int32_p = ctypes.POINTER(ctypes.c_int32)
c_long_p  = ctypes.POINTER(ctypes.c_long)

# ── Config struct ─────────────────────────────────────────────────────────
class BpdLlamaConfig(ctypes.Structure):
    _fields_ = [
        ("n_layers",    ctypes.c_int),
        ("n_heads",     ctypes.c_int),
        ("n_kv_heads",  ctypes.c_int),
        ("head_dim",    ctypes.c_int),
        ("embed_dim",   ctypes.c_int),
        ("ffn_dim",     ctypes.c_int),
        ("vocab_size",  ctypes.c_int),
        ("max_seq_len", ctypes.c_int),
        ("rms_eps",     ctypes.c_float),
        ("rope_base",   ctypes.c_float),
        ("rope_dim",    ctypes.c_int),
    ]

# ── Layer weights struct ──────────────────────────────────────────────────
class BpdLlamaLayerWeights(ctypes.Structure):
    _fields_ = [
        ("attn_norm_w", c_float_p),
        ("w_q",         c_uint8_p),
        ("w_k",         c_uint8_p),
        ("w_v",         c_uint8_p),
        ("w_o",         c_uint8_p),
        ("ffn_norm_w",  c_float_p),
        ("w_gate",      c_uint8_p),
        ("w_up",        c_uint8_p),
        ("w_down",      c_uint8_p),
    ]

# ── Model weights struct ──────────────────────────────────────────────────
class BpdLlamaWeights(ctypes.Structure):
    _fields_ = [
        ("token_embd",    c_uint8_p),
        ("layers",        ctypes.POINTER(BpdLlamaLayerWeights)),
        ("output_norm_w", c_float_p),
        ("output_w",      c_uint8_p),
    ]

# ── Function signatures ───────────────────────────────────────────────────
lib.bpd_llama_block_cpu.restype = None
lib.bpd_llama_block_cpu.argtypes = [
    c_float_p,                                # x
    ctypes.POINTER(BpdLlamaLayerWeights),     # lw
    ctypes.POINTER(BpdLlamaConfig),           # cfg
    c_int32_p,                                # pos_ids
    ctypes.c_int,                             # n_tokens
    ctypes.c_int,                             # kv_pos
    c_float_p,                                # k_cache
    c_float_p,                                # v_cache
    c_float_p,                                # scratch1
    c_float_p,                                # scratch2
    c_float_p,                                # scratch3
]

lib.bpd_llama_forward_cpu.restype = None
lib.bpd_llama_forward_cpu.argtypes = [
    c_int32_p,                                # token_ids
    ctypes.c_int,                             # n_tokens
    ctypes.POINTER(BpdLlamaWeights),          # weights
    ctypes.POINTER(BpdLlamaConfig),           # cfg
    c_int32_p,                                # pos_ids
    ctypes.c_int,                             # kv_pos
    c_float_p,                                # k_cache
    c_float_p,                                # v_cache
    c_float_p,                                # logits_out
    c_long_p,                                 # token_out
]

# ── Helper: create random Q8_0 weight tensor ──────────────────────────────
def make_q8_0_weights(n_rows, n_cols):
    """Create a random Q8_0 weight tensor (n_rows x n_cols).
    Q8_0 format: each block of 32 elements = 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes.
    """
    assert n_cols % 32 == 0
    blocks_per_row = n_cols // 32
    bytes_per_row = blocks_per_row * 34
    total_bytes = n_rows * bytes_per_row
    data = np.zeros(total_bytes, dtype=np.uint8)
    for row in range(n_rows):
        for blk in range(blocks_per_row):
            offset = row * bytes_per_row + blk * 34
            # Random scale as float16 (small values to keep outputs bounded)
            scale = np.float16(np.random.uniform(0.001, 0.01))
            data[offset:offset+2] = np.array([scale], dtype=np.float16).view(np.uint8)
            # Random int8 quants
            data[offset+2:offset+34] = np.random.randint(-127, 128, 32, dtype=np.int8).view(np.uint8)
    return data

# ── Helper: create a tiny model for testing ───────────────────────────────
def make_tiny_model(n_layers=2, n_heads=4, n_kv_heads=2, head_dim=8,
                    ffn_dim=32, vocab_size=64, max_seq_len=32):
    """Create a tiny random model for composition testing."""
    embed_dim = n_heads * head_dim
    np.random.seed(42)

    cfg = BpdLlamaConfig()
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.n_kv_heads = n_kv_heads
    cfg.head_dim = head_dim
    cfg.embed_dim = embed_dim
    cfg.ffn_dim = ffn_dim
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = max_seq_len
    cfg.rms_eps = 1e-5
    cfg.rope_base = 10000.0
    cfg.rope_dim = head_dim

    # Token embedding: [vocab_size, embed_dim] in Q8_0
    token_embd = make_q8_0_weights(vocab_size, embed_dim)

    # Per-layer weights
    layer_weights_arr = (BpdLlamaLayerWeights * n_layers)()
    layer_data = []  # keep references alive

    for i in range(n_layers):
        attn_norm = np.random.uniform(0.5, 1.5, embed_dim).astype(np.float32)
        ffn_norm = np.random.uniform(0.5, 1.5, embed_dim).astype(np.float32)
        w_q = make_q8_0_weights(n_heads * head_dim, embed_dim)
        w_k = make_q8_0_weights(n_kv_heads * head_dim, embed_dim)
        w_v = make_q8_0_weights(n_kv_heads * head_dim, embed_dim)
        w_o = make_q8_0_weights(embed_dim, n_heads * head_dim)
        w_gate = make_q8_0_weights(ffn_dim, embed_dim)
        w_up = make_q8_0_weights(ffn_dim, embed_dim)
        w_down = make_q8_0_weights(embed_dim, ffn_dim)

        layer_weights_arr[i].attn_norm_w = attn_norm.ctypes.data_as(c_float_p)
        layer_weights_arr[i].w_q = w_q.ctypes.data_as(c_uint8_p)
        layer_weights_arr[i].w_k = w_k.ctypes.data_as(c_uint8_p)
        layer_weights_arr[i].w_v = w_v.ctypes.data_as(c_uint8_p)
        layer_weights_arr[i].w_o = w_o.ctypes.data_as(c_uint8_p)
        layer_weights_arr[i].ffn_norm_w = ffn_norm.ctypes.data_as(c_float_p)
        layer_weights_arr[i].w_gate = w_gate.ctypes.data_as(c_uint8_p)
        layer_weights_arr[i].w_up = w_up.ctypes.data_as(c_uint8_p)
        layer_weights_arr[i].w_down = w_down.ctypes.data_as(c_uint8_p)

        layer_data.append((attn_norm, ffn_norm, w_q, w_k, w_v, w_o, w_gate, w_up, w_down))

    # Output norm and output projection
    output_norm = np.random.uniform(0.5, 1.5, embed_dim).astype(np.float32)
    output_w = make_q8_0_weights(vocab_size, embed_dim)

    weights = BpdLlamaWeights()
    weights.token_embd = token_embd.ctypes.data_as(c_uint8_p)
    weights.layers = ctypes.cast(layer_weights_arr, ctypes.POINTER(BpdLlamaLayerWeights))
    weights.output_norm_w = output_norm.ctypes.data_as(c_float_p)
    weights.output_w = output_w.ctypes.data_as(c_uint8_p)

    # Keep all data alive
    model_data = {
        'cfg': cfg, 'weights': weights,
        'token_embd': token_embd, 'output_norm': output_norm, 'output_w': output_w,
        'layer_weights_arr': layer_weights_arr, 'layer_data': layer_data,
    }
    return cfg, weights, model_data

# ── Tests ─────────────────────────────────────────────────────────────────
results = []

def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    print(f"  {status}  {name}: {detail}")

# ── L.1.10a: Single block determinism ────────────────────────────────────
def test_block_determinism():
    cfg, weights, data = make_tiny_model(n_layers=1)
    E = cfg.embed_dim
    n_tokens = 4

    # Create input
    np.random.seed(123)
    x1 = np.random.randn(n_tokens * E).astype(np.float32)
    x2 = x1.copy()
    pos_ids = np.arange(n_tokens, dtype=np.int32)

    # Allocate KV caches and scratch
    kv_size = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k_cache1 = np.zeros(kv_size, dtype=np.float32)
    v_cache1 = np.zeros(kv_size, dtype=np.float32)
    k_cache2 = np.zeros(kv_size, dtype=np.float32)
    v_cache2 = np.zeros(kv_size, dtype=np.float32)

    max_dim = max(E, cfg.ffn_dim)
    max_proj = max(cfg.n_heads * cfg.head_dim, cfg.ffn_dim)
    s1 = np.zeros(n_tokens * max_dim, dtype=np.float32)
    s2 = np.zeros(n_tokens * max_proj, dtype=np.float32)
    s3 = np.zeros(n_tokens * max_proj, dtype=np.float32)

    lw = ctypes.cast(data['layer_weights_arr'], ctypes.POINTER(BpdLlamaLayerWeights))

    # Run block on x1
    lib.bpd_llama_block_cpu(
        x1.ctypes.data_as(c_float_p), lw, ctypes.byref(cfg),
        pos_ids.ctypes.data_as(c_int32_p), n_tokens, 0,
        k_cache1.ctypes.data_as(c_float_p), v_cache1.ctypes.data_as(c_float_p),
        s1.ctypes.data_as(c_float_p), s2.ctypes.data_as(c_float_p),
        s3.ctypes.data_as(c_float_p))

    # Reset scratch and run block on x2
    s1[:] = 0; s2[:] = 0; s3[:] = 0
    lib.bpd_llama_block_cpu(
        x2.ctypes.data_as(c_float_p), lw, ctypes.byref(cfg),
        pos_ids.ctypes.data_as(c_int32_p), n_tokens, 0,
        k_cache2.ctypes.data_as(c_float_p), v_cache2.ctypes.data_as(c_float_p),
        s1.ctypes.data_as(c_float_p), s2.ctypes.data_as(c_float_p),
        s3.ctypes.data_as(c_float_p))

    match = np.array_equal(x1, x2)
    kv_match = np.array_equal(k_cache1, k_cache2) and np.array_equal(v_cache1, v_cache2)
    test("L.1.10a Block determinism", match and kv_match,
         f"output_match={match} kv_match={kv_match}")

# ── L.1.10b: Block produces finite, non-trivial output ───────────────────
def test_block_finite():
    cfg, weights, data = make_tiny_model(n_layers=1)
    E = cfg.embed_dim
    n_tokens = 4

    np.random.seed(456)
    x_orig = np.random.randn(n_tokens * E).astype(np.float32)
    x = x_orig.copy()
    pos_ids = np.arange(n_tokens, dtype=np.int32)

    kv_size = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k_cache = np.zeros(kv_size, dtype=np.float32)
    v_cache = np.zeros(kv_size, dtype=np.float32)

    max_dim = max(E, cfg.ffn_dim)
    max_proj = max(cfg.n_heads * cfg.head_dim, cfg.ffn_dim)
    s1 = np.zeros(n_tokens * max_dim, dtype=np.float32)
    s2 = np.zeros(n_tokens * max_proj, dtype=np.float32)
    s3 = np.zeros(n_tokens * max_proj, dtype=np.float32)

    lw = ctypes.cast(data['layer_weights_arr'], ctypes.POINTER(BpdLlamaLayerWeights))

    lib.bpd_llama_block_cpu(
        x.ctypes.data_as(c_float_p), lw, ctypes.byref(cfg),
        pos_ids.ctypes.data_as(c_int32_p), n_tokens, 0,
        k_cache.ctypes.data_as(c_float_p), v_cache.ctypes.data_as(c_float_p),
        s1.ctypes.data_as(c_float_p), s2.ctypes.data_as(c_float_p),
        s3.ctypes.data_as(c_float_p))

    is_finite = np.all(np.isfinite(x))
    is_different = not np.array_equal(x, x_orig)
    test("L.1.10b Block finite output", is_finite and is_different,
         f"finite={is_finite} different={is_different}")

# ── L.1.10c: Full forward pass produces valid logits ──────────────────────
def test_forward_logits():
    cfg, weights, data = make_tiny_model(n_layers=2)
    n_tokens = 4

    token_ids = np.array([1, 5, 10, 3], dtype=np.int32)
    pos_ids = np.arange(n_tokens, dtype=np.int32)

    kv_layer_size = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k_cache = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)
    v_cache = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)

    logits = np.zeros(n_tokens * cfg.vocab_size, dtype=np.float32)
    tokens_out = np.zeros(n_tokens, dtype=np.int64)

    lib.bpd_llama_forward_cpu(
        token_ids.ctypes.data_as(c_int32_p), n_tokens,
        ctypes.byref(weights), ctypes.byref(cfg),
        pos_ids.ctypes.data_as(c_int32_p), 0,
        k_cache.ctypes.data_as(c_float_p), v_cache.ctypes.data_as(c_float_p),
        logits.ctypes.data_as(c_float_p), tokens_out.ctypes.data_as(c_long_p))

    logits_2d = logits.reshape(n_tokens, cfg.vocab_size)
    is_finite = np.all(np.isfinite(logits_2d))
    has_variance = np.std(logits_2d) > 0
    valid_tokens = all(0 <= t < cfg.vocab_size for t in tokens_out)
    test("L.1.10c Forward logits valid", is_finite and has_variance and valid_tokens,
         f"finite={is_finite} variance={np.std(logits_2d):.4e} tokens={list(tokens_out)}")

# ── L.1.10d: Full forward pass determinism ────────────────────────────────
def test_forward_determinism():
    cfg, weights, data = make_tiny_model(n_layers=2)
    n_tokens = 4

    token_ids = np.array([1, 5, 10, 3], dtype=np.int32)
    pos_ids = np.arange(n_tokens, dtype=np.int32)

    kv_layer_size = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k1 = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)
    v1 = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)
    k2 = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)
    v2 = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)

    logits1 = np.zeros(n_tokens * cfg.vocab_size, dtype=np.float32)
    logits2 = np.zeros(n_tokens * cfg.vocab_size, dtype=np.float32)
    tok1 = np.zeros(n_tokens, dtype=np.int64)
    tok2 = np.zeros(n_tokens, dtype=np.int64)

    lib.bpd_llama_forward_cpu(
        token_ids.ctypes.data_as(c_int32_p), n_tokens,
        ctypes.byref(weights), ctypes.byref(cfg),
        pos_ids.ctypes.data_as(c_int32_p), 0,
        k1.ctypes.data_as(c_float_p), v1.ctypes.data_as(c_float_p),
        logits1.ctypes.data_as(c_float_p), tok1.ctypes.data_as(c_long_p))

    lib.bpd_llama_forward_cpu(
        token_ids.ctypes.data_as(c_int32_p), n_tokens,
        ctypes.byref(weights), ctypes.byref(cfg),
        pos_ids.ctypes.data_as(c_int32_p), 0,
        k2.ctypes.data_as(c_float_p), v2.ctypes.data_as(c_float_p),
        logits2.ctypes.data_as(c_float_p), tok2.ctypes.data_as(c_long_p))

    logits_match = np.array_equal(logits1, logits2)
    tok_match = np.array_equal(tok1, tok2)
    kv_match = np.array_equal(k1, k2) and np.array_equal(v1, v2)
    test("L.1.10d Forward determinism", logits_match and tok_match and kv_match,
         f"logits={logits_match} tokens={tok_match} kv={kv_match}")

# ── L.1.10e: Multi-token prefill ─────────────────────────────────────────
def test_forward_prefill():
    cfg, weights, data = make_tiny_model(n_layers=2, max_seq_len=64)
    n_tokens = 8

    token_ids = np.random.randint(0, cfg.vocab_size, n_tokens, dtype=np.int32)
    pos_ids = np.arange(n_tokens, dtype=np.int32)

    kv_layer_size = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k_cache = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)
    v_cache = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)

    logits = np.zeros(n_tokens * cfg.vocab_size, dtype=np.float32)
    tokens_out = np.zeros(n_tokens, dtype=np.int64)

    lib.bpd_llama_forward_cpu(
        token_ids.ctypes.data_as(c_int32_p), n_tokens,
        ctypes.byref(weights), ctypes.byref(cfg),
        pos_ids.ctypes.data_as(c_int32_p), 0,
        k_cache.ctypes.data_as(c_float_p), v_cache.ctypes.data_as(c_float_p),
        logits.ctypes.data_as(c_float_p), tokens_out.ctypes.data_as(c_long_p))

    logits_2d = logits.reshape(n_tokens, cfg.vocab_size)
    all_finite = np.all(np.isfinite(logits_2d))
    all_valid = all(0 <= t < cfg.vocab_size for t in tokens_out)
    test("L.1.10e Prefill 8 tokens", all_finite and all_valid,
         f"finite={all_finite} valid_tokens={all_valid}")

# ── L.1.10f: KV cache populated ──────────────────────────────────────────
def test_kv_cache_populated():
    cfg, weights, data = make_tiny_model(n_layers=2)
    n_tokens = 4

    token_ids = np.array([1, 5, 10, 3], dtype=np.int32)
    pos_ids = np.arange(n_tokens, dtype=np.int32)

    kv_layer_size = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k_cache = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)
    v_cache = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)

    logits = np.zeros(n_tokens * cfg.vocab_size, dtype=np.float32)
    tokens_out = np.zeros(n_tokens, dtype=np.int64)

    lib.bpd_llama_forward_cpu(
        token_ids.ctypes.data_as(c_int32_p), n_tokens,
        ctypes.byref(weights), ctypes.byref(cfg),
        pos_ids.ctypes.data_as(c_int32_p), 0,
        k_cache.ctypes.data_as(c_float_p), v_cache.ctypes.data_as(c_float_p),
        logits.ctypes.data_as(c_float_p), tokens_out.ctypes.data_as(c_long_p))

    # Check that the first n_tokens positions in layer 0 KV cache are non-zero
    kv_head_dim = cfg.n_kv_heads * cfg.head_dim
    k_layer0 = k_cache[:kv_layer_size].reshape(cfg.max_seq_len, kv_head_dim)
    v_layer0 = v_cache[:kv_layer_size].reshape(cfg.max_seq_len, kv_head_dim)

    k_populated = np.any(k_layer0[:n_tokens] != 0)
    v_populated = np.any(v_layer0[:n_tokens] != 0)
    k_empty_after = np.all(k_layer0[n_tokens:] == 0)
    v_empty_after = np.all(v_layer0[n_tokens:] == 0)

    test("L.1.10f KV cache populated", k_populated and v_populated and k_empty_after and v_empty_after,
         f"k_pop={k_populated} v_pop={v_populated} k_empty_after={k_empty_after} v_empty_after={v_empty_after}")

# ── L.1.10g: Incremental decode after prefill ─────────────────────────────
def test_incremental_decode():
    cfg, weights, data = make_tiny_model(n_layers=2, max_seq_len=64)

    # Prefill with 4 tokens
    prefill_tokens = np.array([1, 5, 10, 3], dtype=np.int32)
    prefill_pos = np.arange(4, dtype=np.int32)

    kv_layer_size = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k_cache = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)
    v_cache = np.zeros(cfg.n_layers * kv_layer_size, dtype=np.float32)

    logits = np.zeros(4 * cfg.vocab_size, dtype=np.float32)
    tokens_out = np.zeros(4, dtype=np.int64)

    lib.bpd_llama_forward_cpu(
        prefill_tokens.ctypes.data_as(c_int32_p), 4,
        ctypes.byref(weights), ctypes.byref(cfg),
        prefill_pos.ctypes.data_as(c_int32_p), 0,
        k_cache.ctypes.data_as(c_float_p), v_cache.ctypes.data_as(c_float_p),
        logits.ctypes.data_as(c_float_p), tokens_out.ctypes.data_as(c_long_p))

    # Now decode one more token at position 4
    next_token = np.array([int(tokens_out[3])], dtype=np.int32)
    next_pos = np.array([4], dtype=np.int32)

    logits_decode = np.zeros(1 * cfg.vocab_size, dtype=np.float32)
    tok_decode = np.zeros(1, dtype=np.int64)

    lib.bpd_llama_forward_cpu(
        next_token.ctypes.data_as(c_int32_p), 1,
        ctypes.byref(weights), ctypes.byref(cfg),
        next_pos.ctypes.data_as(c_int32_p), 4,  # kv_pos = 4 (already have 4 tokens in cache)
        k_cache.ctypes.data_as(c_float_p), v_cache.ctypes.data_as(c_float_p),
        logits_decode.ctypes.data_as(c_float_p), tok_decode.ctypes.data_as(c_long_p))

    decode_finite = np.all(np.isfinite(logits_decode))
    decode_valid = 0 <= tok_decode[0] < cfg.vocab_size
    # KV cache at position 4 should now be populated
    kv_head_dim = cfg.n_kv_heads * cfg.head_dim
    k_layer0 = k_cache[:kv_layer_size].reshape(cfg.max_seq_len, kv_head_dim)
    pos4_populated = np.any(k_layer0[4] != 0)

    test("L.1.10g Incremental decode", decode_finite and decode_valid and pos4_populated,
         f"finite={decode_finite} valid_token={tok_decode[0]} pos4_kv={pos4_populated}")

# ── Run all tests ─────────────────────────────────────────────────────────
test_block_determinism()
test_block_finite()
test_forward_logits()
test_forward_determinism()
test_forward_prefill()
test_kv_cache_populated()
test_incremental_decode()

n_pass = sum(1 for _, p in results if p)
n_fail = sum(1 for _, p in results if not p)
print(f"PASS: {n_pass}  FAIL: {n_fail}  TOTAL: {len(results)}")
sys.exit(0 if n_fail == 0 else 1)
