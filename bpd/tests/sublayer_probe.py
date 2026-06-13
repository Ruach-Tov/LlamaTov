#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""sublayer_probe.py — capture OUR 13 layer-0 sub-op intermediates via REPLAY (runner untouched).
Iyun, 2026-05-29. Pairs with medayek's HF-side hooks: meet at the 13-key comparison; first
sub-op diverging from 0 ULP = intrinsic source of the layer-0 divergence.
Sub-op map (medayek, llamatov_run.py:509-551). Usage: python3 sublayer_probe.py --gguf PATH
"""
import sys, os, argparse
import numpy as np

def our_sublayer(gguf, ids):
    sys.path.insert(0, "."); sys.path.insert(0, "..")
    import torch, torch.nn.functional as F
    from llamatov_run import parse_gguf, lt, rms_norm, apply_rope
    md, ts, do = parse_gguf(gguf); arch = md.get('general.architecture','llama')
    cfg = {'n_head': md.get(f'{arch}.attention.head_count',32),
           'n_head_kv': md.get(f'{arch}.attention.head_count_kv', md.get(f'{arch}.attention.head_count',32)),
           'n_embd': md.get(f'{arch}.embedding_length',2048),
           'rope_theta': md.get(f'{arch}.rope.freq_base',500000.0),
           'norm_eps': md.get(f'{arch}.attention.layer_norm_rms_epsilon',1e-5)}
    w = {n: lt(gguf, do, info) for n, info in ts.items()}
    cap = {}; sq = lambda t: t.squeeze(0).detach().float().cpu().numpy()
    tok = torch.tensor(ids, dtype=torch.long)
    te = w['token_embd.weight']; x = (te.T[tok] if te.shape[0]<te.shape[1] else te[tok]).unsqueeze(0)
    p='blk.0'; nh=cfg['n_head']; nkv=cfg['n_head_kv']; hd=cfg['n_embd']//nh
    # 1. attn_norm
    h = rms_norm(x, w[f'{p}.attn_norm.weight'], cfg['norm_eps']); cap['attn_norm']=sq(h)
    # 2-4. q/k/v proj
    q = h @ w[f'{p}.attn_q.weight']; cap['q_proj']=sq(q)
    k = h @ w[f'{p}.attn_k.weight']; cap['k_proj']=sq(k)
    v = h @ w[f'{p}.attn_v.weight']; cap['v_proj']=sq(v)
    # 5. post_rope
    qr, kr = apply_rope(q, k, nh, hd, cfg['rope_theta']); cap['post_rope_q']=sq(qr); cap['post_rope_k']=sq(kr)
    B,T,_ = qr.shape
    qh = qr.view(B,T,nh,hd).transpose(1,2); kh = kr.view(B,T,nkv,hd).transpose(1,2); vh = v.view(B,T,nkv,hd).transpose(1,2)
    if nkv < nh:
        rep = nh//nkv; kh = kh.repeat_interleave(rep,dim=1); vh = vh.repeat_interleave(rep,dim=1)
    # 6. attn_scores
    att = (qh @ kh.transpose(-2,-1)) * (hd**-0.5); cap['attn_scores']=att.squeeze(0).detach().float().cpu().numpy()
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool), diagonal=1); att = att.masked_fill(mask, float('-inf'))
    # 7. attn_softmax
    sm = F.softmax(att, dim=-1); cap['attn_softmax']=sm.squeeze(0).detach().float().cpu().numpy()
    # 8. attn_v
    y = sm @ vh; cap['attn_v']=y.squeeze(0).detach().float().cpu().numpy()
    y = y.transpose(1,2).contiguous().view(B,T,nh*hd)
    # 9. o_proj
    y = y @ w[f'{p}.attn_output.weight']; cap['o_proj']=sq(y)
    # 10. residual1
    x = x + y; cap['residual1']=sq(x)
    # 11. ffn_norm
    h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], cfg['norm_eps']); cap['ffn_norm']=sq(h2)
    # 12. swiglu
    gate = F.silu(h2 @ w[f'{p}.ffn_gate.weight']); up = h2 @ w[f'{p}.ffn_up.weight']
    ffn = (gate*up) @ w[f'{p}.ffn_down.weight']; cap['swiglu']=sq(ffn)
    # 13. residual2
    x = x + ffn; cap['residual2']=sq(x)
    return cap

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gguf", required=True); ap.add_argument("--ids", default="128000,9906")
    a = ap.parse_args(); ids=[int(x) for x in a.ids.split(",")]
    ours = our_sublayer(a.gguf, ids)
    for k,v in ours.items():
        print(f"  ours[{k}]: shape={tuple(v.shape)} mean={float(np.mean(v)):.4f} std={float(np.std(v)):.4f}", file=sys.stderr)
    out = os.path.expanduser("~/tmp/our_sublayer.npz"); np.savez(out, **ours)
    print(f"saved {len(ours)} sub-op captures -> {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
