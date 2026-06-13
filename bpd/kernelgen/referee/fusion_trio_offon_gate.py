# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
import sys
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
def run(fold):
    reset()
    dr._DEVICE_KV_CACHE=True;dr._DEVICE_ATTN=True;dr._MASKED_ATTN=True;dr._SLAB.enabled=True;dr._GRAPH_PREP=True;dr._KV_MAX_SEQ=256
    dr._GEMV_TILED=True;dr._GEMV_TILED_V4=True;dr._RMS_BLOCKROW=True;dr._QUANT_PAR=True;dr._ARGMAX2=True
    dr._DEVICE_LOGITS=True;dr._QFUSED=False;dr._ADDRES_FUSED=True;dr._FUSE_QKV=True;dr._FUSE_GATEUP=True
    dr._BIAS_FOLD=fold; dr._APPEND_KV_FUSED=fold; dr._APPEND_INCR_FUSED=fold
    kv=[None]*cfg['n_layers'];seed=[1,415,6557];gr=dr.GraphRunner(w,cfg,kv)
    lg=gr.seed(torch.tensor(seed),torch.arange(3)); t=int(lg[0,-1].argmax())
    out=[t]; gr.capture(torch.tensor([t]),torch.tensor([3])); cur=t
    for _ in range(20): cur=gr.replay_token(torch.tensor([cur])); out.append(int(cur))
    return out
on=run(True); off=run(False)
print(f"DEVICE-LOGITS path: 3-ON ={on}",flush=True)
print(f"DEVICE-LOGITS path: 3-OFF={off}",flush=True)
print(f"TOKEN-EXACT (on==off): {on==off}",flush=True)
