#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""hf_sublayer_compare.py — HF-side layer-0 sub-op capture + compare vs our captures.
Iyun, 2026-05-29. Closes the sub-op decomposition: first sub-op diverging from 0 ULP =
intrinsic source of the layer-0 divergence. Uses our_sublayer.npz (from sublayer_probe.py or
medayek subop_capture.py). Usage: python3 hf_sublayer_compare.py --gguf PATH [--our NPZ]
"""
import sys, os, argparse
import numpy as np

def ulp_abs(a, b):
    a=np.asarray(a,np.float32).ravel(); b=np.asarray(b,np.float32).ravel()
    n=min(a.size,b.size); a,b=a[:n],b[:n]
    ma=float(np.max(np.abs(a-b))) if n else float("nan")
    ai=a.view(np.int32).astype(np.int64); bi=b.view(np.int32).astype(np.int64)
    ai=np.where(ai<0,np.int64(0x80000000)-ai,ai); bi=np.where(bi<0,np.int64(0x80000000)-bi,bi)
    mu=int(np.max(np.abs(ai-bi))) if n else -1
    return mu, ma, n

def capture_hf(gguf, ids):
    import torch
    from transformers import AutoModelForCausalLM
    md_dir=os.path.dirname(gguf); gf=os.path.basename(gguf)
    print(f"[HF] loading {gf} ...", file=sys.stderr)
    model=AutoModelForCausalLM.from_pretrained(md_dir, gguf_file=gf, torch_dtype=torch.float32, device_map="cpu").eval()
    caps={}; L0=model.model.layers[0]
    def mk(name):
        def h(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[name]=t.detach().float().cpu().numpy()
        return h
    hooks=[]
    hooks.append(L0.input_layernorm.register_forward_hook(mk("attn_norm")))
    hooks.append(L0.self_attn.q_proj.register_forward_hook(mk("q_proj")))
    hooks.append(L0.self_attn.k_proj.register_forward_hook(mk("k_proj")))
    hooks.append(L0.self_attn.v_proj.register_forward_hook(mk("v_proj")))
    hooks.append(L0.self_attn.o_proj.register_forward_hook(mk("o_proj")))
    hooks.append(L0.post_attention_layernorm.register_forward_hook(mk("ffn_norm")))
    if hasattr(L0.mlp, "gate_proj"): hooks.append(L0.mlp.gate_proj.register_forward_hook(mk("ffn_gate")))
    if hasattr(L0.mlp, "up_proj"):   hooks.append(L0.mlp.up_proj.register_forward_hook(mk("ffn_up")))
    if hasattr(L0.mlp, "down_proj"): hooks.append(L0.mlp.down_proj.register_forward_hook(mk("swiglu")))
    hooks.append(L0.register_forward_hook(mk("residual2")))  # full layer output
    import torch as T
    with T.no_grad(): model(T.tensor([ids]))
    for h in hooks: h.remove()
    # squeeze batch
    return {k: (v[0] if v.ndim==3 else v) for k,v in caps.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gguf",required=True)
    ap.add_argument("--our",default=os.path.expanduser("~/tmp/our_sublayer.npz"))
    ap.add_argument("--ids",default="128000,9906")
    a=ap.parse_args(); ids=[int(x) for x in a.ids.split(",")]
    ours=dict(np.load(a.our))
    # map our keys -> hf keys (post_rope_q etc have no clean HF hook; compare the shared ones)
    keymap={"attn_norm":"attn_norm","q_proj":"q_proj","k_proj":"k_proj","v_proj":"v_proj",
            "o_proj":"o_proj","ffn_norm":"ffn_norm","swiglu":"swiglu","residual2":"residual2"}
    hf=capture_hf(a.gguf, ids)
    print(f"[HF] captured: {list(hf.keys())}", file=sys.stderr)
    # ordered by forward-pass position
    order=["attn_norm","q_proj","k_proj","v_proj","o_proj","ffn_norm","swiglu","residual2"]
    print("\n=== SUB-OP DECOMPOSITION (our vs HF, layer 0) ===")
    first_div=None
    for ok in order:
        hk=keymap.get(ok)
        if ok not in ours or hk not in hf: 
            print(f"  {ok:12s}: (no matching capture)"); continue
        mu,ma,n=ulp_abs(ours[ok], hf[hk])
        verdict="0 ULP" if mu==0 else ("small" if mu<64 else "LARGE")
        print(f"  {ok:12s}: max_ULP={mu:>12d}  max_abs={ma:.3e}  -> {verdict}")
        if mu>0 and first_div is None: first_div=ok
    print(f"\nFIRST DIVERGING SUB-OP = {first_div}  (the intrinsic source; everything after propagates)")

if __name__=="__main__":
    main()
