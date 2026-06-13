# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
"""Cold-subprocess isolation (Bocher's discriminating test): run ONE config in this fresh process
(env-var toggles read at import), emit the step-0 device-logits vector. A driver runs this 4x as
SEPARATE processes (all-ON, BIAS_FOLD=0 alone, APPEND_KV=0 alone, APPEND_INCR=0 alone) and diffs.
This matches Bocher's cold-start subprocess arms — NOT my one-process disc_dl.py (which shared caches
across the flip and could mask a real legacy-path difference)."""
import sys, os
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
# config from ENV (read fresh this process) — exactly Bocher's mechanism
def E(k,d="1"): return os.environ.get(k,d)!="0"
dr._DEVICE_KV_CACHE=True;dr._DEVICE_ATTN=True;dr._MASKED_ATTN=True;dr._SLAB.enabled=True;dr._GRAPH_PREP=True;dr._KV_MAX_SEQ=256
dr._GEMV_TILED=True;dr._GEMV_TILED_V4=True;dr._RMS_BLOCKROW=True;dr._QUANT_PAR=True;dr._ARGMAX2=True
dr._DEVICE_LOGITS=True;dr._QFUSED=False;dr._ADDRES_FUSED=True;dr._FUSE_QKV=True;dr._FUSE_GATEUP=True
dr._BIAS_FOLD=E("BPD_BIAS_FOLD"); dr._APPEND_KV_FUSED=E("BPD_APPEND_KV_FUSED"); dr._APPEND_INCR_FUSED=E("BPD_APPEND_INCR_FUSED")
kv=[None]*cfg['n_layers']; seed=[1,415,6557]
gr=dr.GraphRunner(w,cfg,kv); lg=gr.seed(torch.tensor(seed),torch.arange(3))
v=lg[0,-1].detach().cpu().numpy().astype(np.float64)
tag=f"BIAS={dr._BIAS_FOLD} APPKV={dr._APPEND_KV_FUSED} APPINCR={dr._APPEND_INCR_FUSED}"
# print a compact fingerprint: argmax + top-3 logit values (enough to detect 0.4 drift)
import hashlib; print(f"[{tag}] argmax={int(v.argmax())} sha1={hashlib.sha1(v.tobytes()).hexdigest()[:16]} l0={v[0]:.8f} l310={v[310]:.8f}", flush=True)
