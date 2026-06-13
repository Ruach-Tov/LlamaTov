# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
import sys, numpy as np, torch
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import llamatov_run as R, dev_residency as dr, fact_dispatch as fd
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

md,ts,do=R.parse_gguf(_os2.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf")); arch=md['general.architecture']
cfg={'n_layers':md[f'{arch}.block_count'],'n_head':md[f'{arch}.attention.head_count'],'n_head_kv':md[f'{arch}.attention.head_count_kv'],'n_embd':md[f'{arch}.embedding_length'],'rope_theta':md[f'{arch}.rope.freq_base'],'norm_eps':md[f'{arch}.attention.layer_norm_rms_epsilon']}
w={n:R.lt(_os2.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf"),do,info) for n,info in ts.items()}
def setcfg():
    dr._GEMV_TILED=True; dr._RMS_BLOCKROW=True; dr._QUANT_PAR=True; dr._QFUSED=False; dr._ADDRES_FUSED=False
    dr._DEVICE_KV_CACHE=True;dr._DEVICE_ATTN=True;dr._MASKED_ATTN=True;dr._SLAB.enabled=True;dr._GRAPH_PREP=True;dr._DEVICE_LOGITS=False;dr._KV_MAX_SEQ=256
    dr._SLAB.base=None;dr._RESID_IN=None;dr._RESID_OUT=None;dr._LOGITS_BUF=None;dr._LOGITS_RMS=None;dr._ARGMAX_BUF=None
def l2t(lg): return int(lg[0,-1].argmax())
dr._HOST_OPS=set(); setcfg()
kv=[None]*cfg['n_layers']; g=[1,415,6557]
gr=dr.GraphRunner(w,cfg,kv)
lg=gr.seed(torch.tensor(g),torch.arange(3)); g.append(l2t(lg))
gr.capture(torch.tensor([g[-1]]),torch.tensor([len(g)-1]))
for _ in range(5):
    lg=gr.replay_logits(torch.tensor([g[-1]])); g.append(l2t(lg))
print(f"CLEAN run, D1/D2/D3 active: OK gen={g[4:9]} (assertions no-op when config correct)",flush=True)
dr._HOST_OPS={"rms"}; setcfg()
try:
    gr2=dr.GraphRunner(w,cfg,[None]*cfg['n_layers'])
    gr2.seed(torch.tensor([1,415,6557]),torch.arange(3))
    print("D1 FAILED to fire",flush=True)
except RuntimeError as e:
    print(f"D1 FIRES: {str(e)[:70]}",flush=True)
dr._HOST_OPS=set()
try:
    dr.argmax_dev(0, 0); print("D3 FAILED to fire",flush=True)
except AssertionError as e:
    print(f"D3 FIRES: {str(e)[:55]}",flush=True)
