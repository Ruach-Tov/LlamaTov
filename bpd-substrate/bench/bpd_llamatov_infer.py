#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import tempfile
"""bpd_llamatov_infer.py — LlamaTov end-to-end inference orchestrator.

Honors medayek's contract from 2026-05-22 05:48 UTC:
  Output JSON file with:
    model, prompt, temperature, n_tokens_generated, tokens (IDs),
    text, per_token_logits_argmax, reference_text, match

Loads a real GGUF model (llama3.2-1b), runs forward through 16 layers
using OUR verified L.1.1-L.1.10 kernels with real Q8_0 weights, produces
output tokens via greedy argmax at temp=0.

Tokenization scope: this Phase L.1 orchestrator takes pre-tokenized input
token IDs (Phase L.2 will implement BPE from GGUF vocab/merges). The
expected reference text + token IDs are obtained by running llama-cli
externally; we compare our output to that reference.

Usage:
  bpd_llamatov_infer.py --gguf <path> --tokens "128000,9906,11,856,836,374" \\
                        --n-generate 8 --out /tmp/llamatov_result.json
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

# ─── ctypes struct definitions (mirror bench/bpd_llama_block.c) ───────────
c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_int32_p = ctypes.POINTER(ctypes.c_int32)
c_long_p = ctypes.POINTER(ctypes.c_long)


class BpdLlamaConfig(ctypes.Structure):
    _fields_ = [
        ("n_layers", ctypes.c_int),
        ("n_heads", ctypes.c_int),
        ("n_kv_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("embed_dim", ctypes.c_int),
        ("ffn_dim", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("max_seq_len", ctypes.c_int),
        ("rms_eps", ctypes.c_float),
        ("rope_base", ctypes.c_float),
        ("rope_dim", ctypes.c_int),
        ("kv_cache_f16", ctypes.c_int),  # sweepable: 0=f32, 1=f16 (matches ggml)
    ]


class BpdLlamaLayerWeights(ctypes.Structure):
    _fields_ = [
        ("attn_norm_w", c_float_p),
        ("w_q", c_uint8_p),
        ("w_k", c_uint8_p),
        ("w_v", c_uint8_p),
        ("w_o", c_uint8_p),
        ("ffn_norm_w", c_float_p),
        ("w_gate", c_uint8_p),
        ("w_up", c_uint8_p),
        ("w_down", c_uint8_p),
    ]


class BpdLlamaWeights(ctypes.Structure):
    _fields_ = [
        ("token_embd", c_uint8_p),
        ("layers", ctypes.POINTER(BpdLlamaLayerWeights)),
        ("output_norm_w", c_float_p),
        ("output_w", c_uint8_p),
        ("rope_freqs", c_float_p),
    ]


# ─── Prolog-driven GGUF tensor offset queries (canonical) ─────────────────
def query_tensor(gguf_path, tensor_name):
    """Query a single tensor's abs_offset+size+type+dims via gguf_query.pl."""
    script = REPO_ROOT / "tests" / "gguf_query.pl"
    result = subprocess.run(
        ["swipl", "-q", "-g", f"consult('{script}'), gguf_query_main", "--",
         str(gguf_path), str(tensor_name)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gguf_query({tensor_name}) failed: {result.stderr}")
    line = result.stdout.strip().splitlines()[-1]
    fields = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            fields[k] = v
    return {
        "name": tensor_name,
        "abs_offset": int(fields["ABS_OFFSET"]),
        "size_bytes": int(fields["SIZE"]),
        "ggml_type": int(fields["TYPE"]),
        "dims": tuple(int(d) for d in fields["DIMS"].split(",")),
    }


# ─── Bulk tensor query: build a Prolog script that queries all tensors at once ─
def query_all_tensors(gguf_path, names):
    """Query offsets for many tensors in ONE swipl invocation (much faster)."""
    # Construct a multi-query Prolog script
    queries = []
    for name in names:
        # member(tensor_info(Name, Dims, Type, Off), Tensors), AbsOff is DS + Off,
        # compute_size(Type, Dims, Size), format('~w|~w|~w|~w|~w~n', [Name, AbsOff, Size, Type, Dims])
        queries.append(f"(member(tensor_info('{name}', D{name.replace('.','_').replace('-','_')}, T{name.replace('.','_').replace('-','_')}, O{name.replace('.','_').replace('-','_')}), Ts), AO{name.replace('.','_').replace('-','_')} is DS + O{name.replace('.','_').replace('-','_')}, compute_size(T{name.replace('.','_').replace('-','_')}, D{name.replace('.','_').replace('-','_')}, S{name.replace('.','_').replace('-','_')}), format('OFFSET|{name}|~w|~w|~w|~w~n', [AO{name.replace('.','_').replace('-','_')}, S{name.replace('.','_').replace('-','_')}, T{name.replace('.','_').replace('-','_')}, D{name.replace('.','_').replace('-','_')}]))")

    script_lines = [
        ":- use_module(library(lists)).",
        ":- use_module('lib/gguf_native_reader').",
        "compute_size(0, Ds, S) :- elem_count(Ds, N), S is N * 4.",
        "compute_size(1, Ds, S) :- elem_count(Ds, N), S is N * 2.",
        "compute_size(8, Ds, S) :- elem_count(Ds, N), B is N // 32, S is B * 34.",
        "compute_size(12, Ds, S) :- elem_count(Ds, N), B is N // 256, S is B * 144.",
        "compute_size(_, Ds, S) :- elem_count(Ds, S).",
        "elem_count([], 1). elem_count([D|R], N) :- elem_count(R, M), N is D * M.",
        "main :-",
        f"    gguf_read('{gguf_path}', _, _, Ts, DS),",
    ]
    for i, name in enumerate(names):
        # Use simple variable naming
        script_lines.append(f"    (member(tensor_info('{name}', D{i}, T{i}, O{i}), Ts) -> ")
        script_lines.append(f"        AO{i} is DS + O{i}, compute_size(T{i}, D{i}, S{i}),")
        script_lines.append(f"        format('OFFSET|{name}|~w|~w|~w|~w~n', [AO{i}, S{i}, T{i}, D{i}])")
        script_lines.append(f"    ; format('MISSING|{name}~n', []))" + ("," if i < len(names)-1 else ","))
    script_lines.append("    true.")
    script_lines.append(":- main.")
    script_lines.append(":- halt.")

    script_src = "\n".join(script_lines)
    script_path = os.path.join(tempfile.gettempdir(), f"_query_all_{os.getuid()}.pl")
    with open(script_path, "w") as f:
        f.write(script_src)

    result = subprocess.run(
        ["swipl", "-q", script_path],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bulk query failed: {result.stderr}")

    tensors = {}
    for line in result.stdout.strip().splitlines():
        if not line.startswith("OFFSET|"):
            continue
        _, name, abs_off, size, gtype, dims_str = line.split("|", 5)
        dims_str = dims_str.strip().lstrip("[").rstrip("]")
        dims = tuple(int(d.strip()) for d in dims_str.split(",")) if dims_str else ()
        tensors[name] = {
            "abs_offset": int(abs_off),
            "size_bytes": int(size),
            "ggml_type": int(gtype),
            "dims": dims,
        }
    return tensors


# ─── GGUF config query via Prolog (single shot) ───────────────────────────
def query_config(gguf_path):
    """Return llama config (n_layers, n_heads, etc.) by reading GGUF metadata."""
    script_src = f"""
:- use_module(library(lists)).
:- use_module('lib/gguf_native_reader').
main :-
    gguf_read('{gguf_path}', _, M, _, _),
    findall(K-V, (member(K-V, M), \\+ K = 'tokenizer.ggml.tokens', \\+ K = 'tokenizer.ggml.merges', \\+ K = 'tokenizer.ggml.token_type'), Filtered),
    forall(member(K-V, Filtered), format('META|~w|~w~n', [K, V])).
:- main.
:- halt.
"""
    script_path = os.path.join(tempfile.gettempdir(), f"_query_config_{os.getuid()}.pl")
    with open(script_path, "w") as f:
        f.write(script_src)

    result = subprocess.run(
        ["swipl", "-q", script_path],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"config query failed: {result.stderr}")

    meta = {}
    for line in result.stdout.splitlines():
        if line.startswith("META|"):
            _, k, v = line.split("|", 2)
            meta[k] = v
    return meta


# ─── Weight tensor loading via mmap ───────────────────────────────────────
class GgufWeightLoader:
    """Loads tensor bytes from a GGUF file via mmap. Holds references so
    numpy arrays stay valid for the lifetime of the loader."""
    def __init__(self, gguf_path):
        self.gguf_path = gguf_path
        # mmap the entire file
        import mmap
        self.fd = open(gguf_path, "rb")
        self.mmap = mmap.mmap(self.fd.fileno(), 0, prot=mmap.PROT_READ)
        self.arrays = []  # keep references so ctypes pointers stay valid

    def load_q8_0(self, info):
        """Load a Q8_0 tensor as raw uint8 bytes."""
        # Read via seek+read instead of np.frombuffer(mmap) to avoid
        # creating memoryview/managedbuffer objects that pin the mmap.
        self.mmap.seek(info["abs_offset"])
        raw = self.mmap.read(info["size_bytes"])
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        self.arrays.append(arr)
        return arr

    def load_f32(self, info):
        """Load an F32 tensor."""
        n_elem = info["size_bytes"] // 4
        self.mmap.seek(info["abs_offset"])
        raw = self.mmap.read(n_elem * 4)
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        self.arrays.append(arr)
        return arr

    def close(self):
        self.arrays.clear()
        self.mmap.close()
        self.fd.close()


# ─── Build the model structs ──────────────────────────────────────────────
def build_model(gguf_path, n_layers=16, kv_cache_f16=1):
    """Load config + all weights, return (cfg_obj, weights_obj, loader)."""
    # 1. Config
    meta = query_config(gguf_path)
    cfg = BpdLlamaConfig()
    cfg.n_layers = int(meta.get('llama.block_count', n_layers))
    cfg.n_heads = int(meta.get('llama.attention.head_count', 32))
    cfg.n_kv_heads = int(meta.get('llama.attention.head_count_kv', 8))
    cfg.embed_dim = int(meta.get('llama.embedding_length', 2048))
    cfg.ffn_dim = int(meta.get('llama.feed_forward_length', 8192))
    cfg.head_dim = cfg.embed_dim // cfg.n_heads
    cfg.vocab_size = 128256  # from tokens array (skipped in our reader, hardcoded)
    cfg.max_seq_len = 128
    cfg.rms_eps = float(meta.get('llama.attention.layer_norm_rms_epsilon', 1e-5))
    cfg.rope_base = float(meta.get('llama.rope.freq_base', 500000.0))
    cfg.rope_dim = int(meta.get('llama.rope.dimension_count', 64))
    cfg.kv_cache_f16 = int(kv_cache_f16)  # sweepable: 1=f16 (ggml), 0=f32 (higher precision)

    print(f"[config] n_layers={cfg.n_layers}, n_heads={cfg.n_heads}, n_kv_heads={cfg.n_kv_heads}")
    print(f"[config] embed_dim={cfg.embed_dim}, ffn_dim={cfg.ffn_dim}, head_dim={cfg.head_dim}")
    print(f"[config] rms_eps={cfg.rms_eps}, rope_base={cfg.rope_base}, rope_dim={cfg.rope_dim}")

    # 2. Build the list of tensor names to query
    tensor_names = [
        "token_embd.weight",
        "output_norm.weight",
        # output.weight may or may not exist (tied to token_embd in some models)
    ]
    # Try output.weight too; if missing we'll use token_embd
    tensor_names.append("output.weight")
    tensor_names.append("rope_freqs.weight")  # llama3 RoPE freq correction (NTK-aware)
    _q8_reg = []  # (lw, field_name, array) for arena packing
    for li in range(cfg.n_layers):
        for suffix in ["attn_norm.weight", "attn_q.weight", "attn_k.weight",
                       "attn_v.weight", "attn_output.weight", "ffn_norm.weight",
                       "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]:
            tensor_names.append(f"blk.{li}.{suffix}")

    print(f"[load] querying {len(tensor_names)} tensor offsets via Prolog...")
    t0 = time.time()
    offsets = query_all_tensors(gguf_path, tensor_names)
    print(f"[load] got {len(offsets)} offsets in {time.time()-t0:.1f}s")

    # 3. Load all weight bytes via mmap
    loader = GgufWeightLoader(gguf_path)
    t0 = time.time()

    # token_embd.weight (Q8_0)
    token_embd_arr = loader.load_q8_0(offsets["token_embd.weight"])
    # output_norm.weight (F32)
    output_norm_arr = loader.load_f32(offsets["output_norm.weight"])
    # output.weight (Q8_0) — fall back to token_embd if missing (tied)
    if "output.weight" in offsets:
        output_w_arr = loader.load_q8_0(offsets["output.weight"])
    else:
        print("[load] output.weight missing; using token_embd.weight (tied)")
        output_w_arr = token_embd_arr

    # Per-layer arrays
    layer_weights_arr = (BpdLlamaLayerWeights * cfg.n_layers)()
    for li in range(cfg.n_layers):
        attn_norm = loader.load_f32(offsets[f"blk.{li}.attn_norm.weight"])
        w_q = loader.load_q8_0(offsets[f"blk.{li}.attn_q.weight"])
        w_k = loader.load_q8_0(offsets[f"blk.{li}.attn_k.weight"])
        w_v = loader.load_q8_0(offsets[f"blk.{li}.attn_v.weight"])
        w_o = loader.load_q8_0(offsets[f"blk.{li}.attn_output.weight"])
        ffn_norm = loader.load_f32(offsets[f"blk.{li}.ffn_norm.weight"])
        w_gate = loader.load_q8_0(offsets[f"blk.{li}.ffn_gate.weight"])
        w_up = loader.load_q8_0(offsets[f"blk.{li}.ffn_up.weight"])
        w_down = loader.load_q8_0(offsets[f"blk.{li}.ffn_down.weight"])

        lw = layer_weights_arr[li]
        lw.attn_norm_w = attn_norm.ctypes.data_as(c_float_p)
        _q8_reg.append((lw, 'w_q', w_q))
        _q8_reg.append((lw, 'w_k', w_k))
        _q8_reg.append((lw, 'w_v', w_v))
        _q8_reg.append((lw, 'w_o', w_o))
        lw.ffn_norm_w = ffn_norm.ctypes.data_as(c_float_p)
        _q8_reg.append((lw, 'w_gate', w_gate))
        _q8_reg.append((lw, 'w_up', w_up))
        _q8_reg.append((lw, 'w_down', w_down))

    print(f"[load] loaded all weights in {time.time()-t0:.1f}s")

    weights = BpdLlamaWeights()
    weights.token_embd = token_embd_arr.ctypes.data_as(c_uint8_p)
    weights.layers = layer_weights_arr
    weights.output_norm_w = output_norm_arr.ctypes.data_as(c_float_p)
    _q8_reg.append((weights, 'output_w', output_w_arr))
    # ─── ARENA PACK (closed-form layout): all q8_0 weights into ONE contiguous,
    # huge-page-aligned buffer, each tensor 64B-aligned. Repoint struct fields to slices.
    import numpy as _np
    _ALIGN = 64
    _HP = 2 * 1024 * 1024  # 2MB huge page
    _off = 0
    _offsets = []
    for (_owner, _field, _arr) in _q8_reg:
        _offsets.append(_off)
        _off = ((_off + _arr.nbytes + _ALIGN - 1) // _ALIGN) * _ALIGN
    _arena_size = ((_off + _HP - 1) // _HP) * _HP
    _arena = _np.zeros(_arena_size, dtype=_np.uint8)
    for (_owner, _field, _arr), _o in zip(_q8_reg, _offsets):
        _bytes = _arr.reshape(-1).view(_np.uint8)
        _arena[_o:_o + _bytes.nbytes] = _bytes
        _ptr = _arena.ctypes.data + _o
        setattr(_owner, _field, ctypes.cast(_ptr, c_uint8_p))
    weights._arena = _arena  # keep alive
    print(f"[arena] packed {len(_q8_reg)} q8_0 tensors into {_arena_size/1048576:.0f}MB contiguous huge-page-aligned buffer")
    # rope_freqs.weight: llama3 NTK-aware frequency correction (theta/ff). ggml applies it;
    # omitting it (NULL) makes high-freq RoPE dims diverge. Load and pass it.
    if "rope_freqs.weight" in offsets:
        rope_freqs_arr = loader.load_f32(offsets["rope_freqs.weight"])
        _keep_rope_freqs = rope_freqs_arr  # keep alive (prevent GC of the backing buffer)
        weights.rope_freqs = rope_freqs_arr.ctypes.data_as(c_float_p)
        print(f"[load] rope_freqs.weight loaded ({len(rope_freqs_arr)} factors): "
              f"{rope_freqs_arr[:4]} ... {rope_freqs_arr[-2:]}")
    else:
        weights.rope_freqs = None
        print("[load] rope_freqs.weight MISSING; RoPE will use ff=1 (may diverge)")

    # Keep references alive
    loader._layer_weights_arr = layer_weights_arr
    return cfg, weights, loader


# ─── Forward pass driver ──────────────────────────────────────────────────
def generate(lib, cfg, weights, prompt_tokens, n_generate, dump_logits_path=None, c_loop=False):
    """Generate n_generate tokens via greedy argmax.

    Buffer sizing (per the structural test):
      k_cache, v_cache: n_layers * max_seq_len * n_kv_heads * head_dim
      logits:           n_tokens * vocab_size  (one logits row PER token position)
      tokens_out:       n_tokens               (one int64 per token position)
    """
    # KV cache: per-layer (max_seq, n_kv_heads, head_dim) dtype dispatched by cfg.kv_cache_f16
    kv_per_layer = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    kv_total = cfg.n_layers * kv_per_layer
    if cfg.kv_cache_f16:
        # F16 cache: uint16_t* buffer of kv_total elements (2 bytes each)
        k_cache = np.zeros(kv_total, dtype=np.uint16)
        v_cache = np.zeros(kv_total, dtype=np.uint16)
        print(f"[config] kv_cache_dtype=f16 (matches ggml, {kv_total * 2} bytes per cache)")
    else:
        # F32 cache: float buffer of kv_total elements (4 bytes each)
        # The C block reinterprets the uint16_t* pointer as float*, so we
        # allocate float32 and view as uint16 for ctypes-passing.
        k_cache_f32 = np.zeros(kv_total, dtype=np.float32)
        v_cache_f32 = np.zeros(kv_total, dtype=np.float32)
        # The underlying buffer is the SAME bytes; just a different view
        k_cache = k_cache_f32.view(dtype=np.uint16)
        v_cache = v_cache_f32.view(dtype=np.uint16)
        print(f"[config] kv_cache_dtype=f32 (higher precision, {kv_total * 4} bytes per cache)")

    generated_tokens = []
    per_token_argmax = []
    all_tokens = list(prompt_tokens)

    if c_loop:
        # ── C-SIDE DECODE LOOP: prefill + generate entirely in C (no Python per-step). ──
        import time as _t
        prompt_arr = np.ascontiguousarray(prompt_tokens, dtype=np.int32)
        out_tokens = np.zeros(n_generate, dtype=np.int32)
        _t0 = _t.time()
        lib.bpd_llama_generate_cpu(
            prompt_arr.ctypes.data_as(c_int32_p),
            len(prompt_tokens),
            n_generate,
            ctypes.byref(weights),
            ctypes.byref(cfg),
            k_cache.ctypes.data_as(ctypes.c_void_p),
            v_cache.ctypes.data_as(ctypes.c_void_p),
            out_tokens.ctypes.data_as(c_int32_p),
        )
        _dt = _t.time() - _t0
        gen = [int(t) for t in out_tokens]
        print(f"[c-loop] generated {n_generate} tokens in {_dt:.2f}s ({_dt/n_generate*1e3:.0f} ms/tok)")
        print(f"[c-loop] tokens: {gen}")
        return gen, gen

    # INCREMENTAL DECODE: prefill the prompt once (kv_pos=0), then feed ONE new token
    # per step at kv_pos=n_past, NEVER resetting the cache -> O(n) not O(n^2).
    # The C forward supports n_tokens>=1 at any kv_pos (n_kv = kv_pos + n_tokens;
    # gqa_attn attends Q over the full accumulated cache).
    n_past = 0
    for step in range(n_generate):
        if step == 0:
            # PREFILL: process the whole prompt in one batch at positions 0..len-1.
            batch = list(prompt_tokens)
            kv_pos = 0
        else:
            # DECODE: feed only the single newly-generated token at position n_past.
            batch = [all_tokens[-1]]
            kv_pos = n_past
        n_in = len(batch)
        if n_past + n_in > cfg.max_seq_len:
            raise RuntimeError(f"sequence length {n_past + n_in} exceeds max_seq_len={cfg.max_seq_len}")
        token_ids = np.ascontiguousarray(batch, dtype=np.int32)
        pos_ids = np.arange(kv_pos, kv_pos + n_in, dtype=np.int32)

        # Logits: n_in * vocab_size F32 (one row per token IN THIS BATCH). Last row = next token.
        logits = np.zeros(n_in * cfg.vocab_size, dtype=np.float32)
        tokens_out = np.zeros(n_in, dtype=np.int64)
        # NOTE: cache is NOT reset — it accumulates across steps.

        t0 = time.time()
        lib.bpd_llama_forward_cpu(
            token_ids.ctypes.data_as(c_int32_p),
            ctypes.c_int(n_in),
            ctypes.byref(weights),
            ctypes.byref(cfg),
            pos_ids.ctypes.data_as(c_int32_p),
            ctypes.c_int(kv_pos),
            k_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)),  # void* — cfg->kv_cache_f16 tags the dtype
            v_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)),
            logits.ctypes.data_as(c_float_p),
            tokens_out.ctypes.data_as(c_long_p),
        )
        dt = time.time() - t0

        # Take the LAST token's logits row for argmax
        last_logits = logits.reshape(n_in, cfg.vocab_size)[-1]
        argmax_logits = int(np.argmax(last_logits))
        next_token = int(tokens_out[-1])

        # Top-k diagnostic
        top_k = 10
        _part = np.argpartition(last_logits, -top_k)[-top_k:]
        topk_idx = _part[np.argsort(last_logits[_part])][::-1]
        topk_vals = last_logits[topk_idx]
        print(f"[gen step {step}] forward {dt:.2f}s (n_in={n_in}) -> token_out[-1]={next_token} (argmax={argmax_logits})")
        print(f"  top-{top_k}: " + ", ".join(f"{int(i)}={float(v):.4f}" for i, v in zip(topk_idx, topk_vals)))

        # Dump logits to .npy if requested
        if dump_logits_path:
            out_path = dump_logits_path.replace(".npy", f"_step{step}.npy")
            np.save(out_path, last_logits)
            print(f"  [dump] saved logits to {out_path}")

        # Use argmax for the next token (temp=0 greedy); tokens_out should equal argmax
        generated_tokens.append(argmax_logits)
        per_token_argmax.append(argmax_logits)
        n_past += n_in
        all_tokens.append(argmax_logits)

    return generated_tokens, per_token_argmax


# ─── Detokenize: use the GGUF's vocab strings (best-effort; full BPE in Phase L.2) ─
def detokenize_simple(token_ids, vocab_path=None):
    """For now: return the token IDs as the 'text' since we don't have the vocab.
    Phase L.2 will implement proper BPE detokenization from GGUF vocab."""
    return f"[token_ids: {','.join(str(t) for t in token_ids)}]"


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, help="Path to the GGUF model file")
    p.add_argument("--so", default="/tmp/bpd_test/build/bpd_cpu.so",
                   help="Path to bpd_cpu.so")
    p.add_argument("--tokens", required=True,
                   help="Comma-separated input token IDs (e.g., '128000,9906,11,856,836,374')")
    p.add_argument("--prompt", default="<unknown>",
                   help="Original prompt text (for reporting only)")
    p.add_argument("--c-loop", action="store_true", help="Use the C-side decode loop (no Python per-step)")
    p.add_argument("--n-generate", type=int, default=8,
                   help="Number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default="/tmp/llamatov_result.json",
                   help="Output JSON path")
    p.add_argument("--reference-tokens", default=None,
                   help="Comma-separated reference token IDs from llama-cli (for match check)")
    p.add_argument("--reference-text", default=None,
                   help="Reference output text from llama-cli")
    p.add_argument("--dump-logits", default=None,
                   help="If set, save full logits vector at each generation step as .npy")
    p.add_argument("--kv-cache-f16", type=int, default=1,
                   help="KV cache dtype: 1=f16 (ggml-canonical, default), 0=f32 (higher precision, 2x memory)")
    args = p.parse_args()

    print(f"[init] loading library: {args.so}")
    lib = ctypes.CDLL(args.so)
    lib.bpd_llama_generate_cpu.restype = None
    lib.bpd_llama_generate_cpu.argtypes = [
        c_int32_p,                              # prompt_tokens
        ctypes.c_int,                           # prompt_len
        ctypes.c_int,                           # n_generate
        ctypes.POINTER(BpdLlamaWeights),        # weights
        ctypes.POINTER(BpdLlamaConfig),         # cfg
        ctypes.c_void_p,                        # k_cache
        ctypes.c_void_p,                        # v_cache
        c_int32_p,                              # out_tokens
    ]
    lib.bpd_llama_forward_cpu.restype = None
    lib.bpd_llama_forward_cpu.argtypes = [
        c_int32_p, ctypes.c_int,
        ctypes.POINTER(BpdLlamaWeights), ctypes.POINTER(BpdLlamaConfig),
        c_int32_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort),  # k_cache, v_cache (uint16_t*) — tagged by cfg.kv_cache_f16
        c_float_p, c_long_p,
    ]

    print(f"[init] loading model: {args.gguf}")
    cfg, weights, loader = build_model(args.gguf, kv_cache_f16=args.kv_cache_f16)

    prompt_tokens = [int(t) for t in args.tokens.split(",")]
    print(f"[input] prompt={args.prompt!r}, tokens={prompt_tokens}")
    print(f"[input] generating {args.n_generate} tokens at temp={args.temperature}")

    t0 = time.time()
    generated, per_step_argmax = generate(lib, cfg, weights, prompt_tokens,
                                          args.n_generate,
                                          dump_logits_path=args.dump_logits,
                                          c_loop=args.c_loop)
    dt = time.time() - t0
    print(f"[done] generated {len(generated)} tokens in {dt:.1f}s")
    print(f"[done] tokens: {generated}")

    # Comparison
    ref_tokens = None
    if args.reference_tokens:
        ref_tokens = [int(t) for t in args.reference_tokens.split(",")]
    match = (generated == ref_tokens) if ref_tokens is not None else None

    result = {
        "model": "llama3.2-1b",
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "temperature": args.temperature,
        "n_tokens_generated": len(generated),
        "tokens": generated,
        "text": detokenize_simple(generated),
        "per_token_logits_argmax": per_step_argmax,
        "reference_tokens": ref_tokens,
        "reference_text": args.reference_text,
        "match": match,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {args.out}")

    if match is False:
        # Find first divergence
        first_div = next((i for i, (a, b) in enumerate(zip(generated, ref_tokens)) if a != b), None)
        print(f"[diverge] FIRST DIVERGENCE at token {first_div}")
        print(f"[diverge]   ours: {generated}")
        print(f"[diverge]   ref:  {ref_tokens}")

    loader.close()
    sys.exit(0 if match is not False else 1)


if __name__ == "__main__":
    main()
