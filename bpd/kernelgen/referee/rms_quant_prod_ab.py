# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
import sys, time, statistics
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import numpy as np, torch
import llamatov_run as R, dev_residency as dr, fact_dispatch as fd
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

GGUF=_os2.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf")
md,ts,do=R.parse_gguf(GGUF); arch=md['general.architecture']
cfg={'n_layers':md[f'{arch}.block_count'],'n_head':md[f'{arch}.attention.head_count'],'n_head_kv':md[f'{arch}.attention.head_count_kv'],'n_embd':md[f'{arch}.embedding_length'],'rope_theta':md[f'{arch}.rope.freq_base'],'norm_eps':md[f'{arch}.attention.layer_norm_rms_epsilon']}
w={n:R.lt(GGUF,do,ts[n]) for n in ts}; cu=fd._libcuda()
def reset():
    dr._SLAB.base=None;dr._RESID_IN=None;dr._RESID_OUT=None;dr._LOGITS_BUF=None;dr._LOGITS_RMS=None;dr._ARGMAX_BUF=None;dr._ARGMAX_PARTIALS=None;dr._FUSED_W_CACHE.clear()
def run(rms_fused, ntok, timeit=True):
    reset()
    dr.apply_production_profile()           # the certified ~161 ensemble (graph capture path)
    dr._RMS_QUANT_FUSED = rms_fused
    kv=[None]*cfg['n_layers']; seed=[1,415,6557]; gr=dr.GraphRunner(w,cfg,kv)
    lg=gr.seed(torch.tensor(seed),torch.arange(3)); t=int(lg[0,-1].argmax())
    out=[t]; gr.capture(torch.tensor([t]),torch.tensor([3])); cur=t
    # warmup
    for _ in range(5): cur=gr.replay_token(torch.tensor([cur])); out.append(int(cur))
    if not timeit:
        for _ in range(20): cur=gr.replay_token(torch.tensor([cur])); out.append(int(cur))
        return out
    t0=time.time()
    for _ in range(ntok): cur=gr.replay_token(torch.tensor([cur])); out.append(int(cur))
    return out, ntok/(time.time()-t0)
# correctness first
co=run(False, 0, timeit=False); cn=run(True, 0, timeit=False)
print(f"CORRECTNESS (production graph path): on==off = {co==cn}",flush=True)
print(f"  off[:10]={co[:10]}",flush=True)
print(f"  on [:10]={cn[:10]}",flush=True)
# timing: 6 runs each, 100 replay tokens
N=100
off=[run(False,N)[1] for _ in range(6)]
on =[run(True, N)[1] for _ in range(6)]
mo,mn=statistics.median(off),statistics.median(on)
print(f"\nfused OFF tok/s: {[f'{x:.1f}' for x in off]} median={mo:.2f}",flush=True)
print(f"fused ON  tok/s: {[f'{x:.1f}' for x in on]} median={mn:.2f}",flush=True)
print(f"E2E GAIN (production C/CUDA graph path): {(mn/mo-1)*100:+.2f}%  ({mo:.2f} -> {mn:.2f} tok/s)",flush=True)
