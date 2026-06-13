# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
"""T>1 prefill-internal divergence prober. seed() returns only the LAST prefill row's logits, hiding
internal divergence. This drives the prefill MANUALLY token-by-token (matching forward_pass_resident's
own per-token prefill loop) and captures EACH token's logits sha1, for one config (env toggles).
Driver runs it cold per config -> isolates which toggle makes which prefill-token's logits diverge."""
import sys, os, hashlib
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
def E(k): return os.environ.get(k,"1")!="0"
dr._DEVICE_KV_CACHE=True;dr._DEVICE_ATTN=True;dr._MASKED_ATTN=True;dr._SLAB.enabled=True;dr._GRAPH_PREP=True;dr._KV_MAX_SEQ=256
dr._GEMV_TILED=True;dr._GEMV_TILED_V4=True;dr._RMS_BLOCKROW=True;dr._QUANT_PAR=True;dr._ARGMAX2=True
dr._DEVICE_LOGITS=True;dr._QFUSED=False;dr._ADDRES_FUSED=True;dr._FUSE_QKV=True;dr._FUSE_GATEUP=True
dr._BIAS_FOLD=E("BPD_BIAS_FOLD"); dr._APPEND_KV_FUSED=E("BPD_APPEND_KV_FUSED"); dr._APPEND_INCR_FUSED=E("BPD_APPEND_INCR_FUSED")
# reset module buffers
dr._SLAB.base=None;dr._RESID_IN=None;dr._RESID_OUT=None;dr._LOGITS_BUF=None;dr._LOGITS_RMS=None;dr._ARGMAX_BUF=None;dr._ARGMAX_PARTIALS=None
try: dr._FUSED_W_CACHE.clear()
except: pass
kv=[None]*cfg['n_layers']; seed=[1,415,6557]
# drive prefill token-by-token, capture each token's logits (NOT via seed's last-only return)
sigs=[]
for i,tk in enumerate(seed):
    lg=dr.forward_pass_resident(w,cfg,torch.tensor([tk]),torch.tensor([i]),kv)
    v=lg[0,-1].detach().cpu().numpy().astype(np.float64)
    sigs.append((int(v.argmax()), hashlib.sha1(v.tobytes()).hexdigest()[:12], round(float(v[0]),6)))
tag=f"BIAS={int(dr._BIAS_FOLD)} APPKV={int(dr._APPEND_KV_FUSED)} APPINCR={int(dr._APPEND_INCR_FUSED)}"
print(f"[PREFILL-PERTOKEN {tag}]", flush=True)
for i,(am,sh,l0) in enumerate(sigs):
    print(f"  tok{i}(={seed[i]}): argmax={am} sha1={sh} l0={l0}", flush=True)
