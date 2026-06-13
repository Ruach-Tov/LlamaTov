#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""bpd_layer_bisect.py — bisect end-to-end divergence layer-by-layer.

Approach: drive bpd_llama_block_cpu (single-layer) directly from Python,
capturing the residual stream after each of the 16 layers. Compare each
captured layer-output to the corresponding l_out-N fixture from
llama-eval-callback's trace at /tmp/llama_dump_hello_8.

Output: the first layer (and position within layer) where divergence enters.

The substrate-design discipline applied at the next scale: same empirical
bisection that worked for YOLO Phase 3 (per-kernel 0-ULP gates), now
applied at per-layer granularity.
"""
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

from bpd_llamatov_infer import (
    BpdLlamaConfig, BpdLlamaLayerWeights, BpdLlamaWeights,
    build_model, c_float_p, c_uint8_p, c_int32_p, c_long_p,
)
from llama_fixture_loader import load_manifest, find_op


def ulp_distance(a, b):
    """Per-element ULP distance for F32 arrays."""
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


def compare_tensors(ours, ref, name):
    """Return (max_abs, max_ulp, n_diff, n_total) and print a one-line summary."""
    ours_flat = np.ascontiguousarray(ours, dtype=np.float32).reshape(-1)
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1)
    if ours_flat.shape != ref_flat.shape:
        return None, None, None, None
    max_abs = float(np.abs(ours_flat - ref_flat).max())
    max_ulp, n_diff, n_total = ulp_distance(ours_flat, ref_flat)
    status = "✅" if max_ulp == 0 else ("🟡" if max_abs < 1e-4 else "❌")
    print(f"  {status} {name:30s}: max_abs={max_abs:.6e}, max_ulp={max_ulp}, n_diff={n_diff}/{n_total}")
    return max_abs, max_ulp, n_diff, n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--so", default="/tmp/bpd_test/build/bpd_cpu.so")
    p.add_argument("--tokens", required=True)
    p.add_argument("--fixture-dir", default="/tmp/llama_dump_hello_8")
    args = p.parse_args()

    # Load the C library
    print(f"[init] loading library: {args.so}")
    lib = ctypes.CDLL(args.so)

    # Per-layer block kernel
    lib.bpd_llama_block_cpu.restype = None
    lib.bpd_llama_block_cpu.argtypes = [
        c_float_p,                                # x (in/out)
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

    # Embed-lookup kernel
    lib.bpd_embed_lookup_q8_0_cpu.restype = None
    lib.bpd_embed_lookup_q8_0_cpu.argtypes = [
        c_uint8_p, c_int32_p, c_float_p, ctypes.c_int, ctypes.c_int,
    ]

    # Build the model from real weights
    print(f"[init] loading model: {args.gguf}")
    cfg, weights, loader = build_model(args.gguf)

    # Load the fixture (the captured llama-eval-callback trace)
    print(f"[init] loading fixture: {args.fixture_dir}")
    tensors = load_manifest(args.fixture_dir)
    print(f"[init] {len(tensors)} fixture tensors loaded")

    # Set up the input
    prompt_tokens = [int(t) for t in args.tokens.split(",")]
    n_tokens = len(prompt_tokens)
    print(f"[input] tokens={prompt_tokens}, n={n_tokens}")

    # ─── STAGE 1: embed lookup ────────────────────────────────────────────
    print(f"\n[stage 1] embedding lookup")
    token_ids = np.ascontiguousarray(prompt_tokens, dtype=np.int32)
    x = np.zeros(n_tokens * cfg.embed_dim, dtype=np.float32)
    lib.bpd_embed_lookup_q8_0_cpu(
        weights.token_embd, token_ids.ctypes.data_as(c_int32_p),
        x.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(cfg.embed_dim))

    # Compare to ggml's inp_embd (first occurrence, idx 0)
    ref_inp_embd = find_op(tensors, name_substring="inp_embd", op_desc="GET_ROWS")
    if ref_inp_embd is not None:
        ref = ref_inp_embd.as_numpy()  # shape (n_tokens, embed_dim)
        compare_tensors(x.reshape(n_tokens, cfg.embed_dim), ref, "inp_embd (embed)")

    # ─── STAGE 2: per-layer iteration ─────────────────────────────────────
    pos_ids = np.arange(n_tokens, dtype=np.int32)
    kv_per_layer = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k_cache_all = np.zeros(cfg.n_layers * kv_per_layer, dtype=np.float32)
    v_cache_all = np.zeros(cfg.n_layers * kv_per_layer, dtype=np.float32)
    max_dim = max(cfg.embed_dim, cfg.ffn_dim)
    max_proj = max(cfg.n_heads * cfg.head_dim, cfg.ffn_dim)
    s1 = np.zeros(n_tokens * max_dim, dtype=np.float32)
    s2 = np.zeros(n_tokens * max_proj, dtype=np.float32)
    s3 = np.zeros(n_tokens * max_proj, dtype=np.float32)

    # Loop through each layer, compare residual stream to l_out-N
    first_divergent_layer = None
    for li in range(cfg.n_layers):
        # Reset scratch
        s1[:] = 0; s2[:] = 0; s3[:] = 0
        # Point at layer li's slice of the global KV cache
        k_cache_l = k_cache_all[li * kv_per_layer:(li + 1) * kv_per_layer]
        v_cache_l = v_cache_all[li * kv_per_layer:(li + 1) * kv_per_layer]

        # Take address of the i-th layer's weights struct
        lw_ptr = ctypes.byref(weights.layers[li])

        # Run the block
        lib.bpd_llama_block_cpu(
            x.ctypes.data_as(c_float_p), lw_ptr, ctypes.byref(cfg),
            pos_ids.ctypes.data_as(c_int32_p), ctypes.c_int(n_tokens), ctypes.c_int(0),
            k_cache_l.ctypes.data_as(c_float_p), v_cache_l.ctypes.data_as(c_float_p),
            s1.ctypes.data_as(c_float_p), s2.ctypes.data_as(c_float_p),
            s3.ctypes.data_as(c_float_p))

        # Compare x to l_out-li (the captured layer output)
        ref = find_op(tensors, name_substring=f"l_out-{li}")
        if ref is not None:
            ref_arr = ref.as_numpy()
            res = compare_tensors(x.reshape(n_tokens, cfg.embed_dim), ref_arr, f"layer {li:2d} output")
            if res[1] is not None and res[1] > 0 and first_divergent_layer is None:
                first_divergent_layer = li

    print(f"\n[result] first divergent layer: {first_divergent_layer}")
    loader.close()


if __name__ == "__main__":
    main()
