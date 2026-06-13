#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""gguf_to_graph.py — derive a model's compute graph (with dataflow) from a live GGUF, as Prolog facts.

Reads the GGUF tensor list, detects the per-layer architecture GENERICALLY (qkv-bias? fused/split FFN?
GQA?), and emits op(Id, Kind, Inputs, Output) facts that transform_bridge.pl consults. This is the
"load any model -> get its map" step: the structure comes from the actual tensors present, not a
hardcoded assumption. Role inference (in Prolog) never reads the names emitted here.

Usage:  python3 gguf_to_graph.py MODEL.gguf [--layers N] > model_graph.pl
"""
import sys, os, re

def _bpd_root(p=os.path.dirname(os.path.abspath(__file__))):
    while p != "/" and os.path.basename(p) != "bpd":
        p = os.path.dirname(p)
    return p if os.path.basename(p) == "bpd" else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _bpd_root())
sys.path.insert(0, os.path.join(_bpd_root(), "lib"))
import llamatov_run as R


def detect_layer_structure(tensors, layer=0):
    """From the tensors of one layer, return the set of present roles (generic across architectures)."""
    pre = f"blk.{layer}."
    names = {n[len(pre):] for n in tensors if n.startswith(pre)}
    return {
        "qkv_bias":  "attn_q.bias" in names,
        "ffn_gate":  "ffn_gate.weight" in names,        # gated FFN (silu(gate)*up) vs plain MLP
        "attn_norm": "attn_norm.weight" in names,
        "ffn_norm":  "ffn_norm.weight" in names,
        "o_proj":    "attn_output.weight" in names,
    }


def emit_layer(L, struct, out):
    """Emit op/4 facts for one transformer layer. Op ids and activation symbols are namespaced by L.
    Tensor names are the REAL GGUF names — but role inference works on dataflow, not these names."""
    def t(role):  # real GGUF tensor name
        return f"'blk.{L}.{role}'"
    def a(name):  # namespaced activation symbol
        return f"a{L}_{name}"
    P = out.append
    has_bias = struct["qkv_bias"]
    # pre-attn norm
    P(f"op(id({L},attn_norm), ggml_rms_norm, [{a('resid_in')}, {t('attn_norm.weight')}], {a('normed')}).")
    # Q/K/V projections (+ optional bias)
    for proj, sym in (("attn_q","q"), ("attn_k","k"), ("attn_v","v")):
        P(f"op(id({L},{sym}_proj), ggml_mul_mat, [{a('normed')}, {t(proj+'.weight')}], {a(sym+'_raw')}).")
        if has_bias:
            P(f"op(id({L},{sym}_bias), ggml_add, [{a(sym+'_raw')}, {t(proj+'.bias')}], {a(sym+'_b')}).")
    qin = a("q_b") if has_bias else a("q_raw")
    kin = a("k_b") if has_bias else a("k_raw")
    vin = a("v_b") if has_bias else a("v_raw")
    # rope on Q,K (V un-roped)
    P(f"op(id({L},q_rope), ggml_rope, [{qin}], {a('q_rope')}).")
    P(f"op(id({L},k_rope), ggml_rope, [{kin}], {a('k_rope')}).")
    # GQA attention [Q,K,V]
    P(f"op(id({L},attn), flash_attention, [{a('q_rope')}, {a('k_rope')}, {vin}], {a('attn_out')}).")
    # output projection + attn residual
    P(f"op(id({L},o_proj), ggml_mul_mat, [{a('attn_out')}, {t('attn_output.weight')}], {a('o_raw')}).")
    P(f"op(id({L},resid_attn), ggml_add, [{a('resid_in')}, {a('o_raw')}], {a('resid_mid')}).")
    # FFN
    P(f"op(id({L},ffn_norm), ggml_rms_norm, [{a('resid_mid')}, {t('ffn_norm.weight')}], {a('ffn_normed')}).")
    if struct["ffn_gate"]:
        P(f"op(id({L},ffn_gate), ggml_mul_mat, [{a('ffn_normed')}, {t('ffn_gate.weight')}], {a('gate')}).")
        P(f"op(id({L},ffn_up),   ggml_mul_mat, [{a('ffn_normed')}, {t('ffn_up.weight')}], {a('up')}).")
        P(f"op(id({L},ffn_silu), ggml_silu, [{a('gate')}], {a('gact')}).")
        P(f"op(id({L},ffn_gu),   ggml_mul, [{a('gact')}, {a('up')}], {a('gu')}).")
        P(f"op(id({L},ffn_down), ggml_mul_mat, [{a('gu')}, {t('ffn_down.weight')}], {a('down')}).")
    else:  # plain MLP
        P(f"op(id({L},ffn_up),   ggml_mul_mat, [{a('ffn_normed')}, {t('ffn_up.weight')}], {a('up')}).")
        P(f"op(id({L},ffn_silu), ggml_silu, [{a('up')}], {a('gact')}).")
        P(f"op(id({L},ffn_down), ggml_mul_mat, [{a('gact')}, {t('ffn_down.weight')}], {a('down')}).")
    P(f"op(id({L},resid_ffn), ggml_add, [{a('resid_mid')}, {a('down')}], {a('resid_out')}).")


def main():
    if len(sys.argv) < 2:
        print("usage: gguf_to_graph.py MODEL.gguf [--layers N]", file=sys.stderr); sys.exit(1)
    path = sys.argv[1]
    md, ts, _ = R.parse_gguf(path)
    arch = md["general.architecture"]
    nlayers = md.get(f"{arch}.block_count", 0)
    if "--layers" in sys.argv:
        nlayers = int(sys.argv[sys.argv.index("--layers")+1])
    struct = detect_layer_structure(ts, 0)
    out = []
    out.append(f"%% Auto-derived compute graph for arch={arch}, layers={nlayers}.")
    out.append(f"%% structure: {struct}")
    out.append(":- module(model_graph, [op/4]).")
    out.append(":- discontiguous op/4.")
    for L in range(nlayers):
        emit_layer(L, struct, out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
