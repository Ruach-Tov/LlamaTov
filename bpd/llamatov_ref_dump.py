# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""llamatov_ref_dump.py — Reference Intermediate Capture

Instruments the proven CPU+KV attention path (llamatov_gpu_kvcache_dp4a.py
shape) to dump per-layer intermediate tensors at known checkpoints into
numpy files. These known-correct intermediates are the test surface for
verifying the C inference loop kernel-by-kernel.

The reference checkpoints captured per (layer, position):

  pre_attn:    x at the start of the layer (post previous-layer output)
  h_attn:      x after attn_norm   (input to Q/K/V matmuls)
  q_pre_rope:  Q projection before RoPE
  k_pre_rope:  K projection before RoPE
  v:           V projection (no RoPE on V)
  q_post_rope: Q after RoPE
  k_post_rope: K after RoPE
  k_cache:     K cache slice [0..pos+1] for this layer
  v_cache:     V cache slice [0..pos+1] for this layer
  scores:      Q @ K_e^T / sqrt(hd) — attention scores
  probs:       softmax(scores)
  attn_out:    probs @ V_e — attention output
  post_attn:   x after attn + residual
  h_ffn:       x after ffn_norm (input to gate/up matmuls)
  gate:        gate projection
  up:          up projection
  ffn_pre_down: SiLU(gate) * up — input to down matmul
  ffn:         down(ffn_pre_down) — FFN output
  post_ffn:    x after FFN + residual (end of layer)

Plus per-token: logits, next_tok.

USAGE:

  python3 llamatov_ref_dump.py <model.gguf> <prompt> --tokens N --out-dir /tmp/llamatov_ref

OUTPUT LAYOUT:

  /tmp/llamatov_ref/
      meta.json                       # arch, nh, nkv, hd, ne, n_layers, theta, eps
      tok{T}/                         # T = absolute token position (0..prefill+N-1)
          layer{L}/
              {stage}.npy             # numpy array for each stage
          logits.npy
          next_tok.json

VERIFICATION CONTRACT:

For any C inference kernel claiming to implement the same computation:

  given the same x and weights at (layer L, position P), the C kernel's
  output for any stage S must be cosine_sim > 0.9999 with the dumped
  reference, with bounded max_abs_drift (start at 1e-4, relax if the C
  side uses lower precision).

Author: metayen 2026-05-16
Per Heath's "correct first, fast second" mantra. Per mavchin's request
for kernel-level reference data for the C inference loop migration.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Import the same components as llamatov_gpu_kvcache_dp4a.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llamatov_run import parse_gguf, lt, rms_norm, apply_rope


def gpu_matmul_stub(h, weight_name, w):
    """CPU matmul stub — for the reference dump we don't need GPU.

    The reference values are what the model SHOULD compute regardless
    of where the multiplication runs. CPU matmul produces bit-exact
    F32 results which are easier to compare against than dp4a-quantized
    GPU results. The C inference loop's verification will tolerate the
    dp4a quantization drift via cosine_sim threshold.
    """
    weight = w[weight_name]
    # weight stored as F32 [N, K]; matmul h @ weight^T → [..., N]
    return h @ weight.T


def save_tensor(out_dir, stage, tensor):
    """Save a tensor as numpy array. Detach + cast to float32 for portability."""
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    np.save(out_dir / f"{stage}.npy", arr)


def dump_reference(model_path, input_ids, n_tokens, out_root):
    """Run inference and dump every per-layer intermediate to disk.

    This is the reference-capture path. Mirrors llamatov_gpu_kvcache_dp4a.py's
    attention math exactly — same RoPE call signature, same KV cache layout,
    same softmax, same residual structure.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {model_path}...")
    md, w, tokens_list = parse_gguf(model_path)
    arch = md["general.architecture"]
    nh = md.get(f"{arch}.attention.head_count", 32)
    nkv = md.get(f"{arch}.attention.head_count_kv", 8)
    ne = md.get(f"{arch}.embedding_length", 2048)
    nl = md.get(f"{arch}.block_count", 16)
    hd = ne // nh
    eps = md.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)
    theta = md.get(f"{arch}.rope.freq_base", 500000.0)

    # Metadata
    meta = {
        "arch": arch,
        "n_head": int(nh),
        "n_head_kv": int(nkv),
        "head_dim": int(hd),
        "n_embd": int(ne),
        "n_layers": int(nl),
        "rope_theta": float(theta),
        "norm_eps": float(eps),
        "model_path": str(model_path),
        "input_ids": list(map(int, input_ids)),
        "n_decode_tokens": int(n_tokens),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Meta: nh={nh} nkv={nkv} hd={hd} ne={ne} nl={nl} theta={theta} eps={eps}")

    # Embedding table
    tok_embd = w["token_embd.weight"]
    # tok_embd shape: [vocab, n_embd] — index by token id

    # KV cache (host-side, F32, large enough for prefill + decode)
    cache_len = len(input_ids) + n_tokens + 4
    kv_k = torch.zeros((nl, nkv, cache_len, hd), dtype=torch.float32)
    kv_v = torch.zeros((nl, nkv, cache_len, hd), dtype=torch.float32)

    pos = 0
    generated = []

    total_steps = len(input_ids) + n_tokens
    print(f"Dumping reference for {total_steps} steps ({len(input_ids)} prefill + {n_tokens} decode)...")

    for step in range(total_steps):
        if step < len(input_ids):
            tok_id = input_ids[step]
        else:
            tok_id = generated[-1]

        # Per-token output directory
        tok_dir = out_root / f"tok{step:04d}"
        tok_dir.mkdir(exist_ok=True)
        meta_tok = {"step": step, "pos": pos, "tok_id": int(tok_id)}

        # Embedding lookup
        x = tok_embd[tok_id, :].view(1, 1, ne)  # [1, 1, ne]

        save_tensor(tok_dir, "embed", x)

        for il in range(nl):
            layer_dir = tok_dir / f"layer{il:02d}"
            layer_dir.mkdir(exist_ok=True)
            p = f"blk.{il}"

            save_tensor(layer_dir, "pre_attn", x)

            # Attention block
            h = rms_norm(x, w[f"{p}.attn_norm.weight"], eps)
            save_tensor(layer_dir, "h_attn", h)

            q = gpu_matmul_stub(h, f"{p}.attn_q.weight", w)
            k = gpu_matmul_stub(h, f"{p}.attn_k.weight", w)
            v = gpu_matmul_stub(h, f"{p}.attn_v.weight", w)

            # Biases if present (qwen2, deepseek-r1)
            if f"{p}.attn_q.bias" in w:
                q = q + w[f"{p}.attn_q.bias"]
            if f"{p}.attn_k.bias" in w:
                k = k + w[f"{p}.attn_k.bias"]
            if f"{p}.attn_v.bias" in w:
                v = v + w[f"{p}.attn_v.bias"]

            save_tensor(layer_dir, "q_pre_rope", q)
            save_tensor(layer_dir, "k_pre_rope", k)
            save_tensor(layer_dir, "v", v)

            # RoPE — position-aware. Correction 13: positions MUST be
            # the actual decode position. Forgetting this caused the
            # "token 27268 repeating" bug on the CPU path.
            q_r, k_r = apply_rope(q, k, nh, hd, theta, positions=torch.tensor([pos]))

            save_tensor(layer_dir, "q_post_rope", q_r)
            save_tensor(layer_dir, "k_post_rope", k_r)

            # KV cache store
            k_h = k_r.view(1, 1, nkv, hd).transpose(1, 2)
            v_h = v.view(1, 1, nkv, hd).transpose(1, 2)
            kv_k[il, :, pos, :] = k_h[0, :, 0, :]
            kv_v[il, :, pos, :] = v_h[0, :, 0, :]

            # Dump cache slice up to and including this position
            save_tensor(layer_dir, "k_cache", kv_k[il, :, :pos + 1, :])
            save_tensor(layer_dir, "v_cache", kv_v[il, :, :pos + 1, :])

            # Attention
            q_h = q_r.view(1, 1, nh, hd).transpose(1, 2)  # [1, nh, 1, hd]
            k_c = kv_k[il, :, :pos + 1, :].unsqueeze(0)   # [1, nkv, pos+1, hd]
            v_c = kv_v[il, :, :pos + 1, :].unsqueeze(0)   # [1, nkv, pos+1, hd]
            rep = nh // nkv
            k_e = k_c.repeat_interleave(rep, dim=1)        # [1, nh, pos+1, hd]
            v_e = v_c.repeat_interleave(rep, dim=1)
            scores = (q_h @ k_e.transpose(-2, -1)) / (hd ** 0.5)  # [1, nh, 1, pos+1]
            probs = F.softmax(scores, dim=-1)
            attn_out = (probs @ v_e).transpose(1, 2).contiguous().view(1, 1, ne)

            save_tensor(layer_dir, "scores", scores)
            save_tensor(layer_dir, "probs", probs)
            save_tensor(layer_dir, "attn_out", attn_out)

            # O projection + residual
            y = gpu_matmul_stub(attn_out, f"{p}.attn_output.weight", w)
            x = x + y
            save_tensor(layer_dir, "post_attn", x)

            # FFN
            h2 = rms_norm(x, w[f"{p}.ffn_norm.weight"], eps)
            save_tensor(layer_dir, "h_ffn", h2)

            gate = gpu_matmul_stub(h2, f"{p}.ffn_gate.weight", w)
            up = gpu_matmul_stub(h2, f"{p}.ffn_up.weight", w)
            save_tensor(layer_dir, "gate", gate)
            save_tensor(layer_dir, "up", up)

            gate_silu = F.silu(gate)
            ffn_pre_down = gate_silu * up
            save_tensor(layer_dir, "ffn_pre_down", ffn_pre_down)

            ffn = gpu_matmul_stub(ffn_pre_down, f"{p}.ffn_down.weight", w)
            x = x + ffn
            save_tensor(layer_dir, "ffn", ffn)
            save_tensor(layer_dir, "post_ffn", x)

        pos += 1

        # Logits and next token
        x_norm = rms_norm(x, w["output_norm.weight"], eps)
        logits = gpu_matmul_stub(x_norm, "output.weight", w)
        save_tensor(tok_dir, "logits", logits)

        next_tok = int(torch.argmax(logits.flatten()))
        meta_tok["next_tok"] = next_tok
        (tok_dir / "meta.json").write_text(json.dumps(meta_tok, indent=2))

        if step == len(input_ids) - 1:
            generated.append(next_tok)
            print(f"  Step {step} (last prefill) → first token: {next_tok}")
        elif step >= len(input_ids):
            generated.append(next_tok)
            gen_idx = step - len(input_ids)
            print(f"  Step {step} (decode {gen_idx}): token {next_tok}")

    # Top-level generated list
    (out_root / "generated.json").write_text(json.dumps(generated, indent=2))
    print(f"\nDone. {total_steps} step directories written to {out_root}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Reference intermediate capture for C inference loop verification")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--prompt", default="Hello", help="Prompt text (will be tokenized if model has tokenizer)")
    parser.add_argument("--input-ids", nargs="+", type=int, help="Explicit input token IDs (alternative to --prompt)")
    parser.add_argument("--tokens", type=int, default=5, help="Number of decode tokens to generate (default 5 for the Ollama-match test surface)")
    parser.add_argument("--out-dir", default="/tmp/llamatov_ref", help="Output directory for reference dumps")
    args = parser.parse_args()

    if args.input_ids:
        input_ids = args.input_ids
    else:
        # Use a fixed token sequence for reproducibility — same as
        # llamatov_gpu_kvcache_dp4a.py default test surface.
        # "Hello, I'm" tokens for llama3.2 vocab:
        input_ids = [9906, 11, 358, 2846]  # rough; user can override
        print(f"Using default input_ids = {input_ids} (override with --input-ids)")

    dump_reference(args.model, input_ids, args.tokens, args.out_dir)


if __name__ == "__main__":
    main()
