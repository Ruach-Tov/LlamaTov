#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""make_synth_moe_gguf.py — build a tiny synthetic Mixtral-style MoE GGUF (real container, real metadata
keys + tensor names, fake small weights). Validates that gguf_to_graph.py's MoE assumptions
(expert_count / expert_used_count metadata, ffn_gate_inp + ffn_{gate,up,down}_exps tensor names) match
the real llama.cpp GGUF convention — without a 26GB Mixtral download."""
import sys, numpy as np, gguf

ARCH = "llama"           # Mixtral uses arch="llama" with MoE tensors
NL = 2                   # tiny: 2 layers
NE = 8                   # 8 experts
TOPK = 2                 # top-2 routing (Mixtral)
EMBD = 32; FF = 64; NH = 4; NKV = 2; HD = 8; VOCAB = 100

def main(out="/tmp/synth_moe.gguf"):
    w = gguf.GGUFWriter(out, ARCH)
    w.add_block_count(NL)
    w.add_context_length(128)
    w.add_embedding_length(EMBD)
    w.add_feed_forward_length(FF)
    w.add_head_count(NH)
    w.add_head_count_kv(NKV)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    # THE MoE METADATA KEYS the deriver reads:
    w.add_expert_count(NE)              # -> {arch}.expert_count
    w.add_expert_used_count(TOPK)       # -> {arch}.expert_used_count
    w.add_vocab_size(VOCAB)

    def t(name, *shape):
        a = (np.random.randn(*shape) * 0.02).astype(np.float32)
        w.add_tensor(name, a)

    # global tensors
    t("token_embd.weight", VOCAB, EMBD)
    t("output_norm.weight", EMBD)
    t("output.weight", VOCAB, EMBD)
    # per-layer: attention (dense) + MoE FFN (the real stacked-expert tensor names)
    for L in range(NL):
        p = f"blk.{L}."
        t(p+"attn_norm.weight", EMBD)
        t(p+"attn_q.weight", NH*HD, EMBD)
        t(p+"attn_k.weight", NKV*HD, EMBD)
        t(p+"attn_v.weight", NKV*HD, EMBD)
        t(p+"attn_output.weight", EMBD, NH*HD)
        t(p+"ffn_norm.weight", EMBD)
        # THE MoE tensors (real names the deriver expects):
        t(p+"ffn_gate_inp.weight", NE, EMBD)          # the router
        t(p+"ffn_gate_exps.weight", NE, FF, EMBD)     # stacked experts (3D)
        t(p+"ffn_up_exps.weight",   NE, FF, EMBD)
        t(p+"ffn_down_exps.weight", NE, EMBD, FF)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}: arch={ARCH}, {NL} layers, {NE} experts, top-{TOPK}")

if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
