#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""measure_residual1.py — measure the residual1 coordinate vs HF (Iyun, 2026-05-29).
residual1 = embedding_input + attn_output (post-attention residual stream). HF exposes it as the
hidden state after the attention block. Clean elementwise add — unambiguous reference.
Expected: RED (propagated) — residual1 adds o_proj, which inherits the q/k layout divergence.
That is honest (red beats tan): residual1 diverges because it ADDS the propagated o_proj, not
because addition is wrong. Usage: python3 measure_residual1.py --gguf PATH
"""
import sys, os, argparse, numpy as np, torch, torch.nn.functional as F

def ulp(a,b):
    a=np.asarray(a,np.float32).ravel(); b=np.asarray(b,np.float32).ravel(); n=min(a.size,b.size); a,b=a[:n],b[:n]
    ma=float(np.max(np.abs(a-b))) if n else float("nan")
    ai=a.view(np.int32).astype(np.int64); bi=b.view(np.int32).astype(np.int64)
    ai=np.where(ai<0,np.int64(0x80000000)-ai,ai); bi=np.where(bi<0,np.int64(0x80000000)-bi,bi)
    return (int(np.max(np.abs(ai-bi))) if n else -1), ma

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gguf",required=True); ap.add_argument("--ids",default="128000,9906")
    a=ap.parse_args(); ids=[int(x) for x in a.ids.split(",")]
    # ours: from the saved sub-op captures
    ours=dict(np.load(os.path.expanduser("~/tmp/our_sublayer.npz")))
    our_r1=ours["residual1"]
    # HF residual1 = hidden state after layer[0]'s self-attention residual add. Capture via the
    # output of self_attn + the input residual. Cleanest: hook the residual stream — HF adds it
    # inside the decoder layer. We capture: input_to_layer + self_attn_output.
    from transformers import AutoModelForCausalLM
    m=AutoModelForCausalLM.from_pretrained(os.path.dirname(a.gguf), gguf_file=os.path.basename(a.gguf), torch_dtype=torch.float32, device_map="cpu").eval()
    caps={}
    def mk(n):
        def hk(mod,i,o):
            caps[n]=(o[0] if isinstance(o,tuple) else o).detach().float().numpy()
            if isinstance(i,tuple) and len(i)>0 and torch.is_tensor(i[0]): caps[n+"_in"]=i[0].detach().float().numpy()
        return hk
    L0=m.model.layers[0]
    # self_attn output + the layer input -> residual1 = layer_input + self_attn_out
    h=[L0.self_attn.register_forward_hook(mk("attn")), L0.register_forward_hook(mk("layerout"))]
    # also need the input to the layer (= embedding for layer 0)
    emb={}
    h.append(m.model.embed_tokens.register_forward_hook(lambda mod,i,o: emb.update(x=o.detach().float().numpy())))
    with torch.no_grad(): m(torch.tensor([ids]))
    for hk in h: hk.remove()
    # HF residual1 = embedding + attn_output
    hf_r1 = (emb["x"][0] + caps["attn"][0]) if "attn" in caps and "x" in emb else None
    if hf_r1 is None:
        print("  residual1: could not assemble HF reference"); return
    mu,ma=ulp(our_r1, hf_r1)
    verdict = "identical(ulp(0))" if mu==0 else ("diverges(ulp(small))" if mu<64 else "diverges(ulp(large))")
    print(f"  residual1 (= embedding + o_proj) vs HF: max_ULP={mu} max_abs={ma:.3e} -> {verdict}")
    print(f"  (expected RED/propagated: residual1 adds o_proj which inherits q/k layout divergence)")

if __name__=="__main__": main()
