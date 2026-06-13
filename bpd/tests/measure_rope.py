#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""measure_rope.py — measure the rope coordinate vs HF (Iyun, 2026-05-29).
rope = apply_rope(q,k). HF computes post-rope internally; we reconstruct HF's post-rope from HF's
q_proj + HF rotate_half rope. Our post_rope_q is ggml-interleave rope on ggml-layout q.
EXPECTED: diverges — but it is a MATCHED-PAIR CONVENTION difference (ggml-interleave rope on
ggml-layout q/k vs HF-split-half rope on HF-layout q/k), the SAME class as the qkv layout cell.
So rope is arguably ALSO a layout/convention cell, resolvable by the same weave. Measure honestly.
Usage: python3 measure_rope.py --gguf PATH
"""
import sys, os, argparse, numpy as np, torch

def ulp(a,b):
    a=np.asarray(a,np.float32).ravel(); b=np.asarray(b,np.float32).ravel(); n=min(a.size,b.size); a,b=a[:n],b[:n]
    ma=float(np.max(np.abs(a-b))) if n else float("nan")
    ai=a.view(np.int32).astype(np.int64); bi=b.view(np.int32).astype(np.int64)
    ai=np.where(ai<0,np.int64(0x80000000)-ai,ai); bi=np.where(bi<0,np.int64(0x80000000)-bi,bi)
    return (int(np.max(np.abs(ai-bi))) if n else -1), ma

def sorted_check(a,b):
    a=np.sort(np.asarray(a,np.float32).ravel()); b=np.sort(np.asarray(b,np.float32).ravel())
    n=min(a.size,b.size); return float(np.max(np.abs(a[:n]-b[:n])))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gguf",required=True); ap.add_argument("--ids",default="128000,9906")
    a=ap.parse_args(); ids=[int(x) for x in a.ids.split(",")]
    ours=dict(np.load(os.path.expanduser("~/tmp/our_sublayer.npz")))
    our_rope_q=ours["post_rope_q"]
    from transformers import AutoModelForCausalLM
    m=AutoModelForCausalLM.from_pretrained(os.path.dirname(a.gguf), gguf_file=os.path.basename(a.gguf), torch_dtype=torch.float32, device_map="cpu").eval()
    cfg=m.config; nh=cfg.num_attention_heads; hd=cfg.hidden_size//nh
    caps={}
    def mk(n):
        def hk(mod,i,o): caps[n]=(o[0] if isinstance(o,tuple) else o).detach().float()
        return hk
    L0=m.model.layers[0]; h=[L0.self_attn.q_proj.register_forward_hook(mk("q"))]
    with torch.no_grad(): m(torch.tensor([ids]))
    for hk in h: hk.remove()
    # reconstruct HF post-rope: rotate_half on HF q_proj
    q=caps["q"]  # (1,T,nh*hd)
    B,T,_=q.shape; qh=q.view(B,T,nh,hd).transpose(1,2)
    theta=getattr(cfg,"rope_theta",500000.0)
    inv=1.0/(theta**(torch.arange(0,hd,2).float()/hd)); pos=torch.arange(T).float()
    fr=torch.outer(pos,inv); emb=torch.cat([fr,fr],-1); cos=emb.cos(); sin=emb.sin()
    def rot(z): z1,z2=z[...,:hd//2],z[...,hd//2:]; return torch.cat([-z2,z1],-1)
    hf_rope_q=(qh*cos+rot(qh)*sin).transpose(1,2).reshape(B,T,nh*hd)[0].numpy()
    o=our_rope_q
    mu,ma=ulp(o,hf_rope_q); sc=sorted_check(o,hf_rope_q)
    print(f"  rope (post_rope_q) vs HF: max_ULP={mu} max_abs={ma:.3e}")
    print(f"  SORTED max_abs={sc:.3e}  ({'PERMUTATION (layout)' if sc<1e-3 else 'genuine value difference'})")
    print(f"  -> {'matched-pair convention (layout-class, same as qkv)' if sc<1e-3 else 'arithmetic/convention divergence'}")

if __name__=="__main__": main()
