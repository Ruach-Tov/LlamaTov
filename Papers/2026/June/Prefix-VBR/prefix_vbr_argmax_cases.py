#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
"""prefix_vbr_argmax_cases.py — for each of the 100 prompts, record whether Q8_0 and Prefix-VBR each match
gold's argmax. Surface the disagreement cases: where VBR missed, did Q8_0 also miss (hard prompt) or did
Q8_0 succeed (a real VBR-specific miss)? And vice versa. Heath's question."""
import sys, os
sys.path.insert(0, "/home/iyun/Ruach-Tov/bpd"); sys.path.insert(0, "/home/iyun/Ruach-Tov/bpd/lib")
import torch, numpy as np, llamatov_run as R
from qwen_bpe import QwenBPE
# reuse the exact corpus + prompt builder from the 100-prompt test
import importlib.util
spec = importlib.util.spec_from_file_location("adh100", "/tmp/prefix_vbr_adherence100.py")
adh = importlib.util.module_from_spec(spec); spec.loader.exec_module(adh)

B = os.environ["BLOB"]

def main():
    tok = QwenBPE(B)
    cfg, wg = R.load_model(B)
    keys = [k for k,t in wg.items() if t.dim()==2 and min(t.shape)>=32]
    wq8 = adh.qmodel(wg, adh.q8, keys); wvb = adh.qmodel(wg, adh.vbr, keys)
    prompts = adh.build_prompts(tok)
    print(f"  {len(prompts)} prompts", flush=True)
    cases = {"both_match":0, "both_miss":0, "vbr_miss_q8_ok":[], "q8_miss_vbr_ok":[]}
    for pi, prompt in enumerate(prompts):
        lg,_ = R.generate_logits(B, prompt, preloaded=(cfg,wg));  gt = int(np.argmax(lg))
        l8,_ = R.generate_logits(B, prompt, preloaded=(cfg,wq8)); a8 = int(np.argmax(l8))
        lv,_ = R.generate_logits(B, prompt, preloaded=(cfg,wvb)); av = int(np.argmax(lv))
        m8 = (a8==gt); mv = (av==gt)
        if m8 and mv: cases["both_match"]+=1
        elif not m8 and not mv: cases["both_miss"]+=1
        elif mv and not m8: cases["q8_miss_vbr_ok"].append((pi, tok.decode(prompt), tok.decode([gt]), tok.decode([a8])))
        elif m8 and not mv: cases["vbr_miss_q8_ok"].append((pi, tok.decode(prompt), tok.decode([gt]), tok.decode([av])))
        if pi%20==0: print(f"  ...{pi}", flush=True)
    print(f"\n  both_match={cases['both_match']}  both_miss={cases['both_miss']}", flush=True)
    print(f"  VBR missed but Q8_0 OK: {len(cases['vbr_miss_q8_ok'])}", flush=True)
    for pi,p,g,got in cases["vbr_miss_q8_ok"]:
        print(f"    p{pi}: prompt={p!r}  gold->{g!r}  vbr->{got!r}", flush=True)
    print(f"  Q8_0 missed but VBR OK: {len(cases['q8_miss_vbr_ok'])}", flush=True)
    for pi,p,g,got in cases["q8_miss_vbr_ok"]:
        print(f"    p{pi}: prompt={p!r}  gold->{g!r}  q8->{got!r}", flush=True)

if __name__=="__main__": main()
