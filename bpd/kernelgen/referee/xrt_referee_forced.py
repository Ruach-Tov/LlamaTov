#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""xrt_referee_forced.py — TEACHER-FORCED cross-runtime ULP referee vs llama.cpp/Ollama.

Fixes the free-running flaw: force BOTH engines down the SAME token sequence (Ollama's), so every step
compares the SAME conditioning and divergence CANNOT cascade. For each generated position i, we compute
OUR logits conditioned on (prompt + ollama_tokens[:i]) and compare OUR logprob for ollama_tokens[i] vs
Ollama's logprob for that same token. Apples-to-apples, per-token, no trajectory drift."""
import sys, os, json, urllib.request, math
sys.path.insert(0,"os.environ.get("BPD_ROOT","bpd")"); sys.path.insert(0,"os.environ.get("BPD_ROOT","bpd")/lib"); sys.path.insert(0,"/tmp")
import torch, numpy as np, llamatov_run as R
from qwen_bpe import QwenBPE
URL="http://localhost:11434/api/generate"

def ollama_run(model, prompt, n, topk=20):
    body={"model":model,"prompt":prompt,"stream":False,"raw":True,
          "options":{"num_predict":n,"temperature":0,"top_k":1,"repeat_penalty":1.0,"top_p":1.0,"seed":0},
          "logprobs":True,"top_logprobs":topk}
    req=urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(req,timeout=300).read())
    steps=[]
    for lp in d.get("logprobs",[]):
        chosen=lp.get("token"); clp=lp.get("logprob")
        tl={e["token"]:e["logprob"] for e in lp.get("top_logprobs",[])}
        steps.append((chosen, clp, tl))
    return d.get("response",""), steps

def our_logits_at(blob, full_ids, positions):
    """Forced: for each target position, run a prefill over full_ids[:pos+1] and read logits at last pos.
    Returns {pos: logit_vector}. (N prefills — correct, immune to drift.)"""
    out={}
    for pos in positions:
        ids = full_ids[:pos]      # condition on everything BEFORE the token we're scoring
        gen, _ = R.generate_logits(blob, ids) if hasattr(R,"generate_logits") else (None,None)
        out[pos]=gen
    return out

def logsoftmax(logits, tid):
    m=logits.max(); lse=m+math.log(np.exp(logits-m).sum()); return float(logits[tid]-lse)

def main():
    model=os.environ.get("OLLAMA_MODEL","qwen2.5:0.5b"); blob=os.environ["BLOB"]
    prompt=os.environ.get("PROMPT","Water is made of hydrogen and"); N=int(os.environ.get("N","6"))
    tok=QwenBPE(blob)
    resp, ol_steps = ollama_run(model, prompt, N)
    pids=tok.encode(prompt)
    # build ollama's token ids from its chosen token strings (map via vocab)
    norm=lambda s:s
    ol_ids=[]
    for chosen,_,_ in ol_steps:
        # find the token id whose string == chosen (Ġ-form)
        cid = tok.tok2id.get(chosen)
        if cid is None:
            cid = tok.tok2id.get(chosen.replace(" ","\u0120"))
        ol_ids.append(cid)
    print(f"  model: {model} | Ollama: {resp[:54]!r}")
    print(f"  ollama token ids: {ol_ids}")
    # forced: for each i, condition our engine on prompt + ol_ids[:i], score ol_ids[i]
    print(f"\n  step  ol_tok        ol_logp   our_logp(SAME tok)  |delta|")
    deltas=[]; agree=0
    for i in range(len(ol_ids)):
        if ol_ids[i] is None: continue
        forced = pids + ol_ids[:i]
        lg, _ = R.generate_logits(blob, forced)   # logits at the position predicting token i
        our_lp = logsoftmax(lg, ol_ids[i])
        our_argmax = int(lg.argmax())
        same = (our_argmax == ol_ids[i])
        agree += same
        chosen, ol_lp, _ = ol_steps[i]
        d=abs(our_lp - ol_lp); deltas.append(d)
        print(f"  {i:<4}  {chosen!r:12}  {ol_lp:+7.3f}  {our_lp:+7.3f}            {d:6.3f}  {'agree' if same else 'OUR-ARGMAX-DIFFERS'}")
    print(f"\n  TEACHER-FORCED (same conditioning every step):")
    print(f"  argmax agreement: {agree}/{len([x for x in ol_ids if x is not None])}")
    print(f"  logprob |delta| for the SAME token: mean={np.mean(deltas):.4f} max={np.max(deltas):.4f}")
    print(f"  -> this is the TRUE cross-runtime forward-pass divergence (no trajectory cascade).")

if __name__=="__main__":
    main()
