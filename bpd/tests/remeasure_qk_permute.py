#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""remeasure_qk_permute.py — TEST medayek's q/k RoPE permute fix (Iyun, 2026-05-29).
Apply ggml-interleave -> HF-split-half permute to attn_q/attn_k weights, recompute q_proj/k_proj,
compare POSITIONALLY to HF (which had sorted=0 / positional=10.58 before). If permuted q_proj now
matches HF at 0 ULP -> the fix works -> downstream should cascade green. MEASURE, do not assert.
Usage: python3 remeasure_qk_permute.py --gguf PATH
"""
import sys, os, argparse, numpy as np, torch

def ulp_abs(a, b):
    a=np.asarray(a,np.float32).ravel(); b=np.asarray(b,np.float32).ravel()
    n=min(a.size,b.size); a,b=a[:n],b[:n]
    ma=float(np.max(np.abs(a-b))) if n else float("nan")
    ai=a.view(np.int32).astype(np.int64); bi=b.view(np.int32).astype(np.int64)
    ai=np.where(ai<0,np.int64(0x80000000)-ai,ai); bi=np.where(bi<0,np.int64(0x80000000)-bi,bi)
    return (int(np.max(np.abs(ai-bi))) if n else -1), ma

def ggml_to_hf_rope_permute(w, n_head, head_dim):
    # w shape: (n_head*head_dim, in)  ->  permute interleave->split-half per head
    w = w.reshape(n_head, head_dim, -1)
    w_re = w[:, 0::2, :]; w_im = w[:, 1::2, :]
    w_hf = torch.cat([w_re, w_im], dim=1)
    return w_hf.reshape(-1, w.shape[-1])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gguf",required=True); ap.add_argument("--ids",default="128000,9906")
    a=ap.parse_args(); ids=[int(x) for x in a.ids.split(",")]
    sys.path.insert(0,"."); sys.path.insert(0,"..")
    from llamatov_run import parse_gguf, lt, rms_norm
    md,ts,do=parse_gguf(a.gguf); arch=md.get("general.architecture","llama")
    nh=md.get(f"{arch}.attention.head_count",32); nkv=md.get(f"{arch}.attention.head_count_kv",nh)
    ne=md.get(f"{arch}.embedding_length",2048); hd=ne//nh; eps=md.get(f"{arch}.attention.layer_norm_rms_epsilon",1e-5)
    w={n:lt(a.gguf,do,info) for n,info in ts.items()}
    tok=torch.tensor(ids); te=w["token_embd.weight"]
    x=(te.T[tok] if te.shape[0]<te.shape[1] else te[tok]).unsqueeze(0)
    h=rms_norm(x, w["blk.0.attn_norm.weight"], eps)
    wq=w["blk.0.attn_q.weight"]; wk=w["blk.0.attn_k.weight"]
    # our q_proj uses h @ wq. wq is (in, out) in our storage; permute acts on the OUTPUT (per-head) dim.
    # transpose to (out,in), permute rows per head, transpose back.
    def permuted_proj(wmat, heads):
        wt = wmat.T.contiguous()                  # (out, in)
        wp = ggml_to_hf_rope_permute(wt, heads, hd)  # permute output rows per head
        return h @ wp.T                            # (.., out)
    q_unperm = h @ wq; q_perm = permuted_proj(wq, nh)
    k_unperm = h @ wk; k_perm = permuted_proj(wk, nkv)

    # load HF q_proj/k_proj for the positional comparison
    from transformers import AutoModelForCausalLM
    m=AutoModelForCausalLM.from_pretrained(os.path.dirname(a.gguf), gguf_file=os.path.basename(a.gguf),
                                           torch_dtype=torch.float32, device_map="cpu").eval()
    caps={}
    def mk(n):
        def hk(mod,i,o): caps[n]=(o[0] if isinstance(o,tuple) else o).detach().float().numpy()
        return hk
    L0=m.model.layers[0]
    hs=[L0.self_attn.q_proj.register_forward_hook(mk("q")), L0.self_attn.k_proj.register_forward_hook(mk("k"))]
    with torch.no_grad(): m(torch.tensor([ids]))
    for hk in hs: hk.remove()
    hf_q=caps["q"][0]; hf_k=caps["k"][0]
    for label, ours in [("q UNPERMUTED", q_unperm.squeeze(0).detach().numpy()),
                        ("q PERMUTED  ", q_perm.squeeze(0).detach().numpy()),
                        ("k UNPERMUTED", k_unperm.squeeze(0).detach().numpy()),
                        ("k PERMUTED  ", k_perm.squeeze(0).detach().numpy())]:
        ref = hf_q if "q" in label else hf_k
        mu,ma=ulp_abs(ours, ref)
        print(f"  {label} vs HF: max_ULP={mu:>12d} max_abs={ma:.3e}  {'<-- 0 ULP FIX WORKS' if mu==0 else ''}")

if __name__=="__main__": main()
