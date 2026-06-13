# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
"""Intermediate-dump prober (Bocher's interaction test): full winning stack, the three fusion
toggles flipped TOGETHER (all-ON vs all-OFF — catches an INTERACTION the each-alone probers miss).
Monkeypatch qkv_fused_dev / q8_linear_dev_bias to dump layer-0 qkv-post-bias per prefill token.
Surfaces the divergent op (not just the collapsed final logits)."""
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
# FULL winning stack (Bocher's env matrix)
dr._DEVICE_KV_CACHE=True;dr._DEVICE_ATTN=True;dr._MASKED_ATTN=True;dr._SLAB.enabled=True;dr._GRAPH_PREP=True;dr._KV_MAX_SEQ=256
dr._GEMV_TILED=True;dr._GEMV_TILED_V4=True;dr._RMS_BLOCKROW=True;dr._QUANT_PAR=True;dr._ARGMAX2=True
dr._DEVICE_LOGITS=True;dr._QFUSED=False;dr._ADDRES_FUSED=True;dr._FUSE_QKV=True;dr._FUSE_GATEUP=True
dr._BIAS_FOLD=E("BPD_BIAS_FOLD"); dr._APPEND_KV_FUSED=E("BPD_APPEND_KV_FUSED"); dr._APPEND_INCR_FUSED=E("BPD_APPEND_INCR_FUSED")

# hook qkv_fused_dev to dump q/k/v of the FIRST call (layer 0) each forward
DUMPS=[]
_orig=dr.qkv_fused_dev
_call=[0]
def _hook(x,wt,p):
    qd,kd,vd=_orig(x,wt,p)
    if p=='blk.0':
        def h(t):
            a=np.empty(t.n,np.float32); cu.cuMemcpyDtoH_v2(a.ctypes.data_as(__import__('ctypes').c_void_p),t.ptr,t.n*4); return a
        try:
            qa,ka,va=h(qd),h(kd),h(vd)
            DUMPS.append((p, hashlib.sha1(qa.tobytes()).hexdigest()[:8], hashlib.sha1(ka.tobytes()).hexdigest()[:8], hashlib.sha1(va.tobytes()).hexdigest()[:8], round(float(qa[0]),6)))
        except Exception as e: DUMPS.append((p,"ERR",str(e)[:30]))
    return qd,kd,vd
dr.qkv_fused_dev=_hook
kv=[None]*cfg['n_layers']; seed=[1,415,6557]
for i,tk in enumerate(seed):
    dr.forward_pass_resident(w,cfg,torch.tensor([tk]),torch.tensor([i]),kv)
tag=f"BIAS={int(dr._BIAS_FOLD)} APPKV={int(dr._APPEND_KV_FUSED)} APPINCR={int(dr._APPEND_INCR_FUSED)}"
print(f"[QKV-L0-DUMP {tag}] (per prefill token, layer-0 qkv-post-bias)",flush=True)
for i,d in enumerate(DUMPS):
    print(f"  call{i}: p={d[0]} q={d[1]} k={d[2]} v={d[3]} q0={d[4] if len(d)>4 else '-'}",flush=True)
