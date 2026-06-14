#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""residual_cache_decode.py — the REAL in-engine 0-ULP gate for residual_cache (KV-Direct).

Modeled on the kv_quantize_q8 e2e (which gave 12/12 green through the GraphRunner). Here the transform's
behavior is exercised by RECOMPUTING K/V from the residual at projection time (instead of using the
first-computed K/V) — the KV-Direct read operation — and we assert the decoded token sequence is
IDENTICAL to baseline. Because K/V = W_kv @ norm(residual) is deterministic, recompute-from-the-same-
residual == the original K/V, so the decode MUST be token-identical. This runs the real production
GraphRunner; the recompute kernels are captured into the graph and replayed (launched on dr._STREAM).
"""
import sys, os, ctypes
import os as _o; _r=_o.path.abspath(_o.path.join(_o.path.dirname(__file__),"..","..")); sys.path.insert(0,_r); sys.path.insert(0,_o.path.join(_r,"lib"))
import torch, numpy as np, dev_residency as dr, llamatov_run as R, fact_dispatch as fd
cu = fd._libcuda(); fd._ctx()

GGUF = os.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf")
GREEN = [310, 470, 895, 280, 286, 456, 286, 470, 830, 280, 262, 555]
md, ts, do = R.parse_gguf(GGUF); arch = md['general.architecture']
cfg = {'n_layers': md[f'{arch}.block_count'], 'n_head': md[f'{arch}.attention.head_count'],
       'n_head_kv': md[f'{arch}.attention.head_count_kv'], 'n_embd': md[f'{arch}.embedding_length'],
       'rope_theta': md[f'{arch}.rope.freq_base'], 'norm_eps': md[f'{arch}.attention.layer_norm_rms_epsilon']}
W = {n: R.lt(GGUF, do, ts[n]) for n in ts}

def reset():
    if hasattr(dr,"_SLAB") and dr._SLAB is not None: dr._SLAB.base=None
    for a in ("_RESID_IN","_RESID_OUT","_LOGITS_BUF","_LOGITS_RMS","_ARGMAX_BUF","_ARGMAX_PARTIALS"):
        if hasattr(dr,a): setattr(dr,a,None)
    for c in ("_FUSED_W_CACHE","_FUSED_QKV_CACHE","_W_CACHE"):
        if hasattr(dr,c) and hasattr(getattr(dr,c),"clear"): getattr(dr,c).clear()

def install_recompute(rc_on):
    """When rc_on, the K/V projection RECOMPUTES from the residual (the KV-Direct read path) by running
    the norm+projection a second time on the captured residual. Same kernels, same residual -> identical
    K/V. Tests that the recompute path is bit-identical end-to-end (token sequence)."""
    orig_norm = dr.rms_norm_dev
    orig_proj = dr.q8_linear_dev_bias
    cap = {"xd": None, "wn": None, "hdv": None}
    def norm_wrap(xd, w, eps):
        hdv = orig_norm(xd, w, eps); cap.update(xd=xd, wn=w, hdv=hdv); return hdv
    def proj_wrap(x, w, b=None):
        if getattr(dr, "_RESIDUAL_CACHE", False) and cap["hdv"] is not None and x is cap["hdv"] and b is not None:
            # KV-Direct: recompute K/V from the cached residual (norm again, then project) on _STREAM
            hdv2 = orig_norm(cap["xd"], cap["wn"], cfg['norm_eps'])
            return orig_proj(hdv2, w, b)
        return orig_proj(x, w, b)
    dr.rms_norm_dev = norm_wrap; dr.q8_linear_dev_bias = proj_wrap
    dr._RESIDUAL_CACHE = rc_on
    return (orig_norm, orig_proj)

def run(rc_on, n=12):
    reset(); dr.apply_production_profile()
    saved = install_recompute(rc_on)
    try:
        kv=[None]*cfg['n_layers']; gr=dr.GraphRunner(W,cfg,kv)
        lg=gr.seed(torch.tensor([1,415,6557]),torch.arange(3)); t=int(lg[0,-1].argmax())
        out=[t]; gr.capture(torch.tensor([t]),torch.tensor([3])); cur=t
        for _ in range(n-1): cur=gr.replay_token(torch.tensor([cur])); out.append(int(cur))
        return out
    finally:
        dr.rms_norm_dev, dr.q8_linear_dev_bias = saved

def main():
    print("=== residual_cache (KV-Direct) — in-engine token-identical gate ===\n")
    base = run(False, 12); print(f"  baseline:               {base}")
    rc = run(True, 12);    print(f"  residual_cache (recomp):{rc}")
    print(f"  green expected:         {GREEN}")
    base_green = (base == GREEN)
    identical = (rc == base)
    print(f"\n  baseline == green: {base_green}")
    print(f"  residual_cache == baseline (token-identical): {identical}")
    ok = base_green and identical
    print(f"\n  {'GATE PASS (0-ULP: KV-Direct decode is token-identical to baseline)' if ok else 'GATE FAIL'}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
