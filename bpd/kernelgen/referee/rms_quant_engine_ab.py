# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
import os, sys, time
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import numpy as np
import torch
import decode_fact as D
import dev_residency as DR
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

GGUF=_os2.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf")
N=int(sys.argv[1]) if len(sys.argv)>1 else 24
cfg, w = D.load_model(GGUF)
def run(fused, ntok):
    DR.apply_production_profile()
    DR._RMS_QUANT_FUSED = fused
    # seed: BOS + first token (mirror the canonical green seed [1,415,6557] start)
    seed_ids=[1,415,6557]
    kv = [None]*cfg["n_layers"]
    toks=[]; 
    # prefill the seed
    pos=0
    for t in seed_ids:
        lg = DR.forward_pass_resident(w, cfg, torch.tensor([t],dtype=torch.long), torch.tensor([pos],dtype=torch.long), kv)
        pos+=1
    nxt=int(np.argmax(lg))
    toks.append(nxt)
    # warmup one
    lg = DR.forward_pass_resident(w, cfg, torch.tensor([nxt],dtype=torch.long), torch.tensor([pos],dtype=torch.long), kv); pos+=1
    nxt=int(np.argmax(lg)); 
    # timed decode
    t0=time.time()
    for _ in range(ntok):
        lg = DR.forward_pass_resident(w, cfg, torch.tensor([nxt],dtype=torch.long), torch.tensor([pos],dtype=torch.long), kv)
        pos+=1; nxt=int(np.argmax(lg)); toks.append(nxt)
    dt=time.time()-t0
    return toks, ntok/dt
print("running fused=OFF...",flush=True)
t_off, tps_off = run(False, N)
print("running fused=ON...",flush=True)
t_on,  tps_on  = run(True, N)
print(f"\nfused OFF: tok/s={tps_off:.1f}  tokens[:8]={t_off[:8]}",flush=True)
print(f"fused ON : tok/s={tps_on:.1f}  tokens[:8]={t_on[:8]}",flush=True)
match = (t_off==t_on)
print(f"\nCORRECTNESS: tokens {'IDENTICAL' if match else 'DIFFER'} ({sum(1 for a,b in zip(t_off,t_on) if a==b)}/{len(t_off)} match)",flush=True)
print(f"E2E GAIN: {(tps_on/tps_off-1)*100:+.2f}%  ({tps_off:.1f} -> {tps_on:.1f} tok/s)",flush=True)
