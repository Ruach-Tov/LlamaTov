# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""LlamaTov inference helpers — quantization dequant + layer operations.

Extends llamatov_loader.py with the operations needed for full transformer
inference. All functions return torch tensors as single Python objects
(not Prolog lists when called via janus_swi).

Per Heath's "maximize Prolog" directive: this is the THIN Python surface
for GPU operations. Prolog orchestrates which functions to call and in
what order. Python does the actual numpy.fromfile + torch ops.

Replaces the embedded builtins:exec helpers in mavchin's llamatov.pl
with a regular Python module file (py_add_lib_dir compatible pattern).
"""

import numpy as np
import torch
import torch.nn.functional as F
import struct


# ────────────────────────────────────────────────────────────────────
# Dequantization (Q2_K, Q3_K, Q6_K)
# ────────────────────────────────────────────────────────────────────

def dequant_q2_k(path, offset, n_elements, shape):
    """Q2_K block dequantization. Each block is 84 bytes for 256 elements."""
    nb = n_elements // 256
    with open(path, 'rb') as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(nb * 84), dtype=np.uint8)
    bl = raw.reshape(nb, 84)
    d = np.frombuffer(bl[:, 80:82].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    dm = np.frombuffer(bl[:, 82:84].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    sc = (bl[:, :16] & 0xF).astype(np.float32) * d[:, None]
    mn = (bl[:, :16] >> 4).astype(np.float32) * dm[:, None]
    qs = bl[:, 16:80]
    q = np.stack([(qs >> (i * 2)) & 3 for i in range(4)], -1).astype(np.float32).reshape(nb, 256)
    si = np.arange(256) // 16
    return torch.from_numpy((sc[:, si] * q - mn[:, si]).reshape(shape).copy())


def dequant_q3_k(path, offset, n_elements, shape):
    """Q3_K block dequantization. Each block is 110 bytes for 256 elements.

    NOTE: this is mavchin's approximate Q3_K; output quality is degraded.
    Real Q3_K has more complex scale extraction.
    """
    nb = n_elements // 256
    with open(path, 'rb') as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(nb * 110), dtype=np.uint8)
    bl = raw.reshape(nb, 110)
    hmask = bl[:, :32]
    qs = bl[:, 32:96]
    sc_raw = bl[:, 96:108]
    d = np.frombuffer(bl[:, 108:110].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    scales = np.zeros((nb, 16), dtype=np.float32)
    for j in range(8):
        scales[:, j] = (sc_raw[:, j] & 0xF).astype(np.float32) - 8
        scales[:, 8 + j] = (sc_raw[:, j] >> 4).astype(np.float32) - 8
    q_lo = np.stack([(qs >> (i * 2)) & 3 for i in range(4)], -1).reshape(nb, 256).astype(np.float32)
    q_hi = np.zeros((nb, 256), dtype=np.float32)
    for j in range(32):
        for k in range(8):
            q_hi[:, j * 8 + k] = ((hmask[:, j] >> k) & 1).astype(np.float32)
    quants = q_lo + q_hi * 4 - 4
    si = np.arange(256) // 16
    return torch.from_numpy((d[:, None] * scales[:, si] * quants).reshape(shape).astype(np.float32).copy())


def dequant_q4_0(path, offset, n_elements, shape):
    """Q4_0 block dequantization. Each block is 18 bytes for 32 elements.

    Block layout (verified against ggml-common.h):
      bytes 0..1   : FP16 scale `d`
      bytes 2..17  : 16 bytes packed 4-bit quants (32 nibbles)
                     Low nibble of byte i is quant 2*i; high nibble is quant 2*i+1.
    Dequantization: value[i] = d * (quant[i] - 8)
    """
    nb = n_elements // 32
    with open(path, 'rb') as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(nb * 18), dtype=np.uint8)
    bl = raw.reshape(nb, 18)
    # Scale: FP16 in first 2 bytes
    d = np.frombuffer(bl[:, :2].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    # Quants: 16 bytes per block, 32 nibbles
    qs = bl[:, 2:18]  # shape (nb, 16)
    # Unpack low and high nibbles
    q_low = (qs & 0x0F).astype(np.float32)   # shape (nb, 16)
    q_high = (qs >> 4).astype(np.float32)    # shape (nb, 16)
    # Interleave: result[0]=low[0], result[1]=high[0], result[2]=low[1], result[3]=high[1], ...
    # Standard ggml layout: quants are laid out as q[0..15]=low_nibbles, q[16..31]=high_nibbles
    # (verified against llama.cpp dequantize_row_q4_0)
    quants = np.concatenate([q_low, q_high], axis=1)  # shape (nb, 32)
    # Apply scale and offset: value = d * (q - 8)
    values = d[:, None] * (quants - 8.0)
    return torch.from_numpy(values.reshape(shape).copy())


def dequant_q4_k(path, offset, n_elements, shape):
    """Q4_K block dequantization. Each block is 144 bytes for 256 elements.

    Block layout (verified against ggml-common.h block_q4_K):
      bytes 0..1     : d   (FP16, super-block scale for scales)
      bytes 2..3     : dmin (FP16, super-block scale for mins)
      bytes 4..15    : 12 bytes packed 6-bit scales and mins (8 sub-blocks × 6 bits × 2)
      bytes 16..143  : 128 bytes packed 4-bit quants (256 nibbles)
    Each super-block has 8 sub-blocks of 32 elements each, with per-sub-block
    scale and min computed from the 6-bit packed values.
    Dequantization: value = d * scale_i * q - dmin * min_i
    """
    nb = n_elements // 256
    with open(path, 'rb') as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(nb * 144), dtype=np.uint8)
    bl = raw.reshape(nb, 144)
    # Super-block scales (FP16 → FP32)
    d = np.frombuffer(bl[:, 0:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    dmin = np.frombuffer(bl[:, 2:4].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    # 12 bytes packed 6-bit scales+mins (8 scales + 8 mins, each 6 bits = 96 bits = 12 bytes)
    scales_bytes = bl[:, 4:16]   # shape (nb, 12)
    # Extract 6-bit values via the standard ggml unpack pattern
    # Reference: get_scale_min_k4 in ggml-common.h
    scales = np.zeros((nb, 8), dtype=np.float32)
    mins = np.zeros((nb, 8), dtype=np.float32)
    for j in range(4):
        # Lower 4 sub-blocks (j=0..3): scale_lo + (scale_hi << 4)
        scales[:, j] = (scales_bytes[:, j] & 0x3F).astype(np.float32)
        mins[:, j]   = (scales_bytes[:, j + 4] & 0x3F).astype(np.float32)
        # Upper 4 sub-blocks (j=4..7)
        scales[:, j + 4] = ((scales_bytes[:, j + 8] & 0x0F) |
                             ((scales_bytes[:, j] >> 6) << 4)).astype(np.float32)
        mins[:, j + 4]   = ((scales_bytes[:, j + 8] >> 4) |
                             ((scales_bytes[:, j + 4] >> 6) << 4)).astype(np.float32)
    # Quants: 128 bytes packed 4-bit (256 nibbles per super-block)
    qs = bl[:, 16:144]  # shape (nb, 128)
    q_low = (qs & 0x0F).astype(np.float32)   # shape (nb, 128)
    q_high = (qs >> 4).astype(np.float32)    # shape (nb, 128)
    # Layout per ggml: sub-blocks 0,2,4,6 use low nibbles; 1,3,5,7 use high nibbles
    # Each sub-block has 32 elements; total = 8 sub-blocks × 32 = 256
    # quants[sb*32 + i] = q_low[sb*16 + i] for even sb, q_high[(sb-1)*16/2*16 + i] for odd sb
    # (verified pattern: first 4 sub-blocks of 32 = low nibbles[0..127]
    #  next 4 sub-blocks of 32 = high nibbles[0..127])
    quants = np.zeros((nb, 256), dtype=np.float32)
    # Sub-blocks 0..3: low nibbles
    for sb in range(4):
        quants[:, sb * 32:(sb + 1) * 32] = q_low[:, sb * 32:(sb + 1) * 32]
    # Sub-blocks 4..7: high nibbles
    for sb in range(4):
        quants[:, (sb + 4) * 32:(sb + 5) * 32] = q_high[:, sb * 32:(sb + 1) * 32]
    # Apply: value = d * scale_sb * q - dmin * min_sb
    sb_idx = np.arange(256) // 32   # shape (256,), values 0..7
    sb_scales_expanded = scales[:, sb_idx]  # shape (nb, 256)
    sb_mins_expanded   = mins[:, sb_idx]    # shape (nb, 256)
    values = (d[:, None] * sb_scales_expanded * quants -
              dmin[:, None] * sb_mins_expanded)
    return torch.from_numpy(values.reshape(shape).astype(np.float32).copy())


def dequant_q6_k(path, offset, n_elements, shape):
    """Q6_K block dequantization. Each block is 210 bytes for 256 elements."""
    nb = n_elements // 256
    with open(path, 'rb') as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(nb * 210), dtype=np.uint8)
    bl = raw.reshape(nb, 210)
    ql = bl[:, :128]
    qh = bl[:, 128:192]
    sc = bl[:, 192:208].view(np.int8).astype(np.float32)
    d = np.frombuffer(bl[:, 208:210].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    q_lo = np.stack([ql & 0xF, ql >> 4], -1).reshape(nb, 256).astype(np.float32)
    q_hi = np.zeros((nb, 256), dtype=np.float32)
    for j in range(64):
        for k in range(4):
            q_hi[:, j * 4 + k] = ((qh[:, j] >> (k * 2)) & 3).astype(np.float32)
    quants = q_lo + q_hi * 16 - 32
    si = np.arange(256) // 16
    return torch.from_numpy((d[:, None] * sc[:, si] * quants).reshape(shape).astype(np.float32).copy())


def dequant_q8_0(path, offset, n_elements, shape):
    """Q8_0 block dequantization. Each block is 34 bytes for 32 elements:
      bytes 0..1  : d (FP16 block scale)
      bytes 2..33 : 32 int8 quants
    Dequant: x[i] = d * qs[i]. (Verified against llama.cpp block_q8_0 in
    ggml-common.h: sizeof == ggml_half + QK8_0. Iyun, 2026-06-08.)
    """
    QK = 32
    BLOCK = 2 + QK            # 34 bytes
    nb = n_elements // QK
    with open(path, 'rb') as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(nb * BLOCK), dtype=np.uint8)
    bl = raw.reshape(nb, BLOCK)
    d = np.frombuffer(bl[:, :2].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    qs = bl[:, 2:].view(np.int8).astype(np.float32)        # (nb, 32)
    values = (d[:, None] * qs).reshape(n_elements)
    return torch.from_numpy(values.reshape(shape).astype(np.float32).copy())


# ────────────────────────────────────────────────────────────────────
# Tensor loading dispatcher (handles all GGUF types)
# ────────────────────────────────────────────────────────────────────

# GGUF type codes (from llama.cpp/gguf.h)
TYPE_F32 = 0
TYPE_F16 = 1
TYPE_Q4_0 = 2
TYPE_Q4_K = 12
TYPE_Q2_K = 10
TYPE_Q3_K = 11
TYPE_Q6_K = 14
TYPE_Q8_0 = 8


def load_tensor_by_type(path, offset, n_elements, shape, type_code):
    """Dispatch tensor loading by GGUF type code.

    Returns torch.Tensor (float32) with the requested shape.
    For quantized types, dequantizes during load.
    For F16, upcasts to F32 (P4 lacks F16 acceleration).
    For unknown types, returns zeros and warns.
    """
    if type_code == TYPE_F32:
        arr = np.fromfile(path, dtype='float32', count=n_elements, offset=offset)
        return torch.from_numpy(arr.copy()).reshape(shape)
    elif type_code == TYPE_F16:
        arr = np.fromfile(path, dtype='float16', count=n_elements, offset=offset)
        return torch.from_numpy(arr.astype('float32').copy()).reshape(shape)
    elif type_code == TYPE_Q4_0:
        return dequant_q4_0(path, offset, n_elements, shape)
    elif type_code == TYPE_Q4_K:
        return dequant_q4_k(path, offset, n_elements, shape)
    elif type_code == TYPE_Q2_K:
        return dequant_q2_k(path, offset, n_elements, shape)
    elif type_code == TYPE_Q3_K:
        return dequant_q3_k(path, offset, n_elements, shape)
    elif type_code == TYPE_Q6_K:
        return dequant_q6_k(path, offset, n_elements, shape)
    elif type_code == TYPE_Q8_0:
        return dequant_q8_0(path, offset, n_elements, shape)
    else:
        print(f"  WARN: unknown type code {type_code}, using zeros")
        return torch.zeros(shape)


# ────────────────────────────────────────────────────────────────────
# Layer-level helpers (called once per layer from Prolog)
# ────────────────────────────────────────────────────────────────────

def layer_norm(x, weight, bias, eps=1e-5):
    """Layer normalization with weight + bias."""
    return F.layer_norm(x, [x.shape[-1]], weight, bias, eps)


def gelu(x):
    """GELU activation."""
    return F.gelu(x)


def silu(x):
    """SiLU activation (for SwiGLU FFN)."""
    return F.silu(x)


def linear(x, weight, bias):
    """Linear: y = x @ weight + bias (GGUF stores [in, out] so no transpose)."""
    if bias is None:
        return x @ weight
    return x @ weight + bias


def attention_qkv_split(qkv, n_heads):
    """Split combined QKV tensor into q, k, v with per-head shape.

    Input: qkv with last dim = 3 * embed_dim
    Output: (q, k, v) each with shape [batch, n_heads, tokens, head_dim]
    """
    B, T, C3 = qkv.shape
    C = C3 // 3
    head_dim = C // n_heads
    q, k, v = qkv.split(C, dim=-1)
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    return q, k, v


def causal_attention(q, k, v):
    """Causal masked attention. Returns the attention output."""
    head_dim = q.shape[-1]
    T = q.shape[-2]
    att = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    att = att.masked_fill(mask, float('-inf'))
    y = F.softmax(att, dim=-1) @ v
    return y


def merge_heads(y, n_heads):
    """Merge attention heads back into a single tensor.

    Input: y with shape [batch, n_heads, tokens, head_dim]
    Output: y with shape [batch, tokens, n_heads * head_dim]
    """
    B, _, T, hd = y.shape
    return y.transpose(1, 2).contiguous().view(B, T, n_heads * hd)


def embed_tokens(token_embd_t, position_embd_t, token_ids):
    """Embed tokens + positions.

    token_embd_t and position_embd_t are .T (transposed) — embeddings stored
    column-wise in GGUF.
    Returns x with shape [1, n_tokens, embed_dim].
    """
    tok = torch.tensor(token_ids, dtype=torch.long)
    pos = torch.arange(len(token_ids), dtype=torch.long)
    x = token_embd_t[tok] + position_embd_t[pos]
    return x.unsqueeze(0)


def transpose(t):
    """Return t.T (for embedding matrices stored column-wise)."""
    return t.T


def argmax_last(logits):
    """Return the argmax of the LAST token's logits as a Python int."""
    return int(logits[0, -1].argmax().item())


def add_tensors(a, b):
    """Element-wise add (for residual connections)."""
    return a + b


def matmul(a, b):
    """Matrix multiplication wrapper (a @ b)."""
    return a @ b


def extract_ollama_eval_stats(response_str):
    """Parse ollama's /api/generate JSON response.

    Returns a 2-tuple (eval_count, eval_duration_ns). Janus represents
    this as `A - B` in Prolog. Used by llamatov_bench.pl.
    """
    import json
    obj = json.loads(response_str)
    return (int(obj.get('eval_count', 0)), int(obj.get('eval_duration', 1)))


# ────────────────────────────────────────────────────────────────────
# Llama-family helpers (RMSNorm, RoPE, SwiGLU, GQA)
# ────────────────────────────────────────────────────────────────────

# ── fact-dispatch switch ─────────────────────────────────────────────────────
# When BPD_FACT_DISPATCH=1, the compute ops delegate to fact_dispatch — kernels
# emitted from Prolog op_expr facts and run on the GPU, instead of torch. The Prolog
# runner (llamatov_llama.pl) is UNCHANGED; flip the flag for a fact-driven forward
# pass (differential vs torch by toggling). (Iyun, 2026-06-08.)
import os as _os
_FACT = _os.environ.get("BPD_FACT_DISPATCH", "0") == "1"
if _FACT:
    import fact_dispatch as _fd


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization (Llama family).

    Per RMSNorm paper: y = x / sqrt(mean(x^2) + eps) * weight
    No bias term (Llama doesn't use biases).
    """
    if _FACT:
        return _fd.rms_norm_fact(x, weight, eps)
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def linear_no_bias(x, weight):
    """Linear without bias: y = x @ weight (Llama-style)."""
    if _FACT:
        return _fd.linear_fact(x, weight)
    return x @ weight


def llama_qkv_split(q_proj, k_proj, v_proj, n_heads, n_heads_kv):
    """Split separate Q, K, V projections into per-head tensors.

    For GQA (Grouped-Query Attention):
      Q has n_heads heads
      K, V have n_heads_kv heads (fewer than Q)
      head_dim is the same for all

    Inputs:
      q_proj: [batch, tokens, n_heads * head_dim]
      k_proj: [batch, tokens, n_heads_kv * head_dim]
      v_proj: [batch, tokens, n_heads_kv * head_dim]

    Outputs (each):
      q: [batch, n_heads, tokens, head_dim]
      k: [batch, n_heads_kv, tokens, head_dim]
      v: [batch, n_heads_kv, tokens, head_dim]
    """
    B, T, _ = q_proj.shape
    head_dim = q_proj.shape[-1] // n_heads
    q = q_proj.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k_proj.view(B, T, n_heads_kv, head_dim).transpose(1, 2)
    v = v_proj.view(B, T, n_heads_kv, head_dim).transpose(1, 2)
    return q, k, v


def precompute_rope_cos_sin(seq_len, head_dim, base=10000.0):
    """Precompute RoPE cos and sin tables for a given sequence length."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary positional embedding to a Q or K tensor.

    Input x: [batch, n_heads, tokens, head_dim]
    cos, sin: [max_seq_len, head_dim/2]   (precomputed for a max length)

    The actual seq_len may be < max_seq_len; we slice cos/sin to match.
    Returns x with RoPE applied — rotates pairs of consecutive dimensions
    by frequencies cos[t, i], sin[t, i].
    """
    head_dim = x.shape[-1]
    seq_len = x.shape[-2]
    half = head_dim // 2
    # Slice cos/sin to actual sequence length
    cos_seq = cos[:seq_len]   # [seq_len, head_dim/2]
    sin_seq = sin[:seq_len]
    # Split last dim into two halves: [..., 0..half-1] and [..., half..head_dim-1]
    x1 = x[..., :half]
    x2 = x[..., half:]
    # Reshape cos/sin to broadcast: [1, 1, seq_len, head_dim/2]
    cos_b = cos_seq.unsqueeze(0).unsqueeze(0)
    sin_b = sin_seq.unsqueeze(0).unsqueeze(0)
    # Rotate: (x1, x2) → (x1*cos - x2*sin, x1*sin + x2*cos)
    rotated_x1 = x1 * cos_b - x2 * sin_b
    rotated_x2 = x1 * sin_b + x2 * cos_b
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


def expand_kv_for_gqa(k, v, n_heads, n_heads_kv):
    """Expand K and V from n_heads_kv to n_heads via repeat_interleave (GQA).

    Each KV head is shared by n_heads / n_heads_kv query heads.
    """
    if n_heads == n_heads_kv:
        return k, v
    repeat_factor = n_heads // n_heads_kv
    k_expanded = k.repeat_interleave(repeat_factor, dim=1)
    v_expanded = v.repeat_interleave(repeat_factor, dim=1)
    return k_expanded, v_expanded


def llama_causal_attention(q, k, v, n_heads, n_heads_kv):
    """Causal masked attention with GQA support.

    Inputs:
      q: [batch, n_heads, tokens, head_dim]
      k, v: [batch, n_heads_kv, tokens, head_dim]

    Output: [batch, n_heads, tokens, head_dim]
    """
    if _FACT:
        return _fd.attention_fact(q, k, v, n_heads, n_heads_kv)
    head_dim = q.shape[-1]
    T = q.shape[-2]
    k_exp, v_exp = expand_kv_for_gqa(k, v, n_heads, n_heads_kv)
    att = (q @ k_exp.transpose(-2, -1)) * (head_dim ** -0.5)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    att = att.masked_fill(mask, float('-inf'))
    y = F.softmax(att, dim=-1) @ v_exp
    return y


def swiglu_ffn(x, gate_w, up_w, down_w):
    """SwiGLU FFN (Llama-style): silu(x @ gate) * (x @ up) → @ down.

    All linears are no-bias.
    """
    if _FACT:
        return _fd.swiglu_fact(x, gate_w, up_w, down_w)
    gate = x @ gate_w
    up = x @ up_w
    inner = F.silu(gate) * up
    return inner @ down_w


def embed_tokens_no_position(token_embd, token_ids):
    """Embed tokens WITHOUT separate position embedding (Llama uses RoPE).

    GGUF stores token_embd.weight as [embed_dim, vocab_size]. To look up
    tokens we index the SECOND dim (vocab axis), then transpose to get
    [n_tokens, embed_dim].

    Returns x with shape [1, n_tokens, embed_dim].
    """
    tok = torch.tensor(token_ids, dtype=torch.long)
    # token_embd is [embed_dim, vocab_size]; transpose then index
    # Equivalent: token_embd[:, tok].T
    x = token_embd.T[tok]  # shape: [n_tokens, embed_dim]
    return x.unsqueeze(0)


# ────────────────────────────────────────────────────────────────────
# Inspection helpers (for Prolog to query tensor state)
# ────────────────────────────────────────────────────────────────────

def tensor_shape(t):
    """Return tensor shape as a tuple."""
    return tuple(t.shape)


def tensor_dim(t, i):
    """Return the i-th dimension as a Python int."""
    return int(t.shape[i])


def tensor_ndim(t):
    """Return number of dimensions."""
    return t.ndim
