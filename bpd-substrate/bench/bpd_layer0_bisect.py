#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""bpd_layer0_bisect.py — bisect WITHIN layer 0 to find first divergent op.

Layer 0 captured operations in /tmp/llama_dump_hello_8:
  norm-0          (RMS_NORM of inp_embd)
  attn_norm-0     (MUL: norm-0 * blk.0.attn_norm.weight)
  Qcur-0          (MUL_MAT: weight=attn_q, input=attn_norm-0)
  Kcur-0          (MUL_MAT: weight=attn_k, input=attn_norm-0)
  Vcur-0          (MUL_MAT: weight=attn_v, input=attn_norm-0)
  Qcur-0 (reshaped) -> (rope)
  Kcur-0 (rope, kv_cache write)
  Vcur-0 (kv_cache write)
  Qcur-0 (permuted)
  kq-0            (attention scores)
  kq-0 (soft_max)
  kqv-0           (attention values * scores)
  kqv-0 (merged)
  kqv_out-0       (MUL_MAT: weight=attn_o, input=kqv-0)
  ffn_inp-0       (RESIDUAL_ADD: kqv_out-0 + inp_embd)
  norm-0 (post-attn) - actually labeled differently
  ffn_norm-0      (MUL with ffn_norm weight)
  ffn_gate-0      (MUL_MAT: weight=ffn_gate, input=ffn_norm-0)
  ffn_up-0        (MUL_MAT: weight=ffn_up, input=ffn_norm-0)
  ffn_silu-0      (SiLU)
  ffn_par-0       (MUL: ffn_silu * ffn_up)
  ffn_out-0       (MUL_MAT: weight=ffn_down)
  l_out-0         (RESIDUAL_ADD: ffn_out + ffn_inp)

Strategy: run each sub-kernel from our substrate on the captured inputs,
compare to captured outputs. Find the first sub-kernel that diverges.
"""
import argparse
import ctypes
import sys
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
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


def compare(ours, ref, name):
    ours_flat = np.ascontiguousarray(ours, dtype=np.float32).reshape(-1)
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1)
    if ours_flat.shape != ref_flat.shape:
        print(f"  ⚠️  {name}: shape mismatch ours={ours_flat.shape} ref={ref_flat.shape}")
        return None
    max_abs = float(np.abs(ours_flat - ref_flat).max())
    max_ulp, n_diff, n_total = ulp_distance(ours_flat, ref_flat)
    icon = "✅" if max_ulp == 0 else ("🟡" if max_abs < 1e-4 else "❌")
    print(f"  {icon} {name:35s}: max_abs={max_abs:.6e}, max_ulp={max_ulp}, n_diff={n_diff}/{n_total}")
    return max_ulp, max_abs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--so", default="/tmp/bpd_test/build/bpd_cpu.so")
    p.add_argument("--tokens", required=True)
    p.add_argument("--fixture-dir", default="/tmp/llama_dump_hello_8")
    args = p.parse_args()

    lib = ctypes.CDLL(args.so)
    # Set up needed kernel signatures
    lib.bpd_embed_lookup_q8_0_cpu.restype = None
    lib.bpd_embed_lookup_q8_0_cpu.argtypes = [c_uint8_p, c_int32_p, c_float_p, ctypes.c_int, ctypes.c_int]
    lib.bpd_rmsnorm_llama_cpu.restype = None
    lib.bpd_rmsnorm_llama_cpu.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.bpd_mul_broadcast_cpu.restype = None
    lib.bpd_mul_broadcast_cpu.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int]
    lib.bpd_qmatmul_q8_0_cpu.restype = None
    lib.bpd_qmatmul_q8_0_cpu.argtypes = [c_uint8_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    cfg, weights, loader = build_model(args.gguf)
    tensors = load_manifest(args.fixture_dir)

    prompt_tokens = [int(t) for t in args.tokens.split(",")]
    n_tokens = len(prompt_tokens)

    # STAGE 1: embed lookup
    print("\n[layer 0 bisect — sub-op level]")
    token_ids = np.ascontiguousarray(prompt_tokens, dtype=np.int32)
    inp_embd = np.zeros(n_tokens * cfg.embed_dim, dtype=np.float32)
    lib.bpd_embed_lookup_q8_0_cpu(
        weights.token_embd, token_ids.ctypes.data_as(c_int32_p),
        inp_embd.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(cfg.embed_dim))

    ref_inp = find_op(tensors, name_substring="inp_embd", op_desc="GET_ROWS")
    compare(inp_embd.reshape(n_tokens, cfg.embed_dim), ref_inp.as_numpy(), "inp_embd")

    # STAGE 2: RMSNorm on inp_embd
    norm_out = np.zeros_like(inp_embd)
    # We need layer 0's attn_norm weight; it's in weights.layers[0].attn_norm_w
    # But the rmsnorm uses NO weight inside the formula \u2014 ggml's RMSNorm is
    # the bare RMSNorm; the weight is applied in a separate MUL op.
    # Actually let me check our kernel signature.
    # bpd_rmsnorm_llama_cpu(input, weight, output, n_tokens, embed_dim, eps)
    # ggml's norm-0 is bare (no weight applied yet \u2014 see manifest: "RMS_NORM(inp_embd)")
    # so we pass weight=NULL or ones? Let me look at the test for clarity.

    # Actually: ggml's RMS_NORM is bare. The MUL op multiplies by attn_norm.weight.
    # Our bpd_rmsnorm_llama_cpu signature has a weight arg \u2014 it's optional or always-on?
    # Pass ones to make it a no-op weighting, see if it matches norm-0.
    ones = np.ones(cfg.embed_dim, dtype=np.float32)
    lib.bpd_rmsnorm_llama_cpu(
        inp_embd.ctypes.data_as(c_float_p),
        ones.ctypes.data_as(c_float_p),
        norm_out.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(cfg.embed_dim), ctypes.c_float(cfg.rms_eps))
    ref_norm = find_op(tensors, name_substring="norm-0", op_desc="RMS_NORM")
    compare(norm_out.reshape(n_tokens, cfg.embed_dim), ref_norm.as_numpy(), "norm-0 (bare RMS)")

    # STAGE 3: MUL with attn_norm.weight
    attn_norm_w_arr = np.ctypeslib.as_array(weights.layers[0].attn_norm_w, shape=(cfg.embed_dim,))
    attn_norm_out = np.zeros_like(inp_embd)
    lib.bpd_mul_broadcast_cpu(
        norm_out.ctypes.data_as(c_float_p),
        attn_norm_w_arr.ctypes.data_as(c_float_p),
        attn_norm_out.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(cfg.embed_dim))
    ref_an = find_op(tensors, name_substring="attn_norm-0", op_desc="MUL")
    compare(attn_norm_out.reshape(n_tokens, cfg.embed_dim), ref_an.as_numpy(), "attn_norm-0 (MUL)")

    # STAGE 4: Q projection (matmul attn_norm-0 with attn_q)
    # attn_q.weight shape (2048, 2048) for llama3.2-1b
    embed_dim = cfg.embed_dim
    w_q_arr = np.ctypeslib.as_array(weights.layers[0].w_q,
        shape=(embed_dim * embed_dim // 32 * 34,))
    qcur_out = np.zeros(n_tokens * embed_dim, dtype=np.float32)
    lib.bpd_qmatmul_q8_0_cpu(
        w_q_arr.ctypes.data_as(c_uint8_p),
        attn_norm_out.ctypes.data_as(c_float_p),
        qcur_out.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(embed_dim), ctypes.c_int(embed_dim))
    ref_qcur = find_op(tensors, name_substring="Qcur-0", op_desc="MUL_MAT")
    if ref_qcur:
        compare(qcur_out.reshape(n_tokens, embed_dim), ref_qcur.as_numpy(), "Qcur-0 (MUL_MAT q)")

    loader.close()


if __name__ == "__main__":
    main()
