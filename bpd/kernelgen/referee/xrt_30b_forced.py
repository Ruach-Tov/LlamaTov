#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
"""xrt_30b_forced.py — teacher-forced cross-runtime referee for the 30B (loads model ONCE).
Our engine vs Ollama/llama.cpp, forced down Ollama's token sequence, per-token logprob delta.
For the 30B this loads once and runs N forwards (each ~15min on CPU) — N small."""
import sys, os, json, urllib.request, math, time
sys.path.insert(0,"os.environ.get("BPD_ROOT","bpd")"); sys.path.insert(0,"os.environ.get("BPD_ROOT","bpd")/lib"); sys.path.insert(0,"/tmp")
import numpy as np, llamatov_run as R
from qwen_bpe import QwenBPE
URL="http://localhost:11434/api/generate"
B=os.environ["BLOB"]; MODEL=os.environ.get("OLLAMA_MODEL","qwen3:30b")
PROMPT=os.environ.get("PROMPT","The capital of France is"); N=int(os.environ.get("N","3"))

def ollama(prompt, n, topk=20):
    body={"model":MODEL,"prompt":prompt,"stream":False,"raw":True,
          "options":{"num_predict":n,"temperature":0,"top_k":1,"repeat_penalty":1.0,"top_p":1.0,"seed":0},
          "logprobs":True,"top_logprobs":topk}
    req=urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(req,timeout=600).read())
    steps=[]
    for lp in d.get("logprobs",[]):
        steps.append((lp.get("token"), lp.get("logprob")))
    return d.get("response",""), steps

def logsoftmax(lg, tid):
    m=lg.max(); return float(lg[tid]-(m+math.log(np.exp(lg-m).sum())))

def main():
    tok=QwenBPE(B)
    resp, ol_steps = ollama(PROMPT, N)
    print(f"  Ollama({MODEL}): {resp[:50]!r}", flush=True)
    ol_ids=[]
    for chosen,_ in ol_steps:
        cid=tok.tok2id.get(chosen) or tok.tok2id.get(chosen.replace(" ","\u0120"))
        ol_ids.append(cid)
    print(f"  ollama token ids: {ol_ids}", flush=True)
    pids=tok.encode(PROMPT)
    print(f"  loading 30B once...", flush=True)
    t0=time.time(); cfg,w=R.load_model(B); print(f"  loaded in {time.time()-t0:.0f}s", flush=True)
    deltas=[]; agree=0
    for i in range(len(ol_ids)):
        if ol_ids[i] is None: continue
        forced = pids + ol_ids[:i]
        t1=time.time()
        lg,_ = R.generate_logits(B, forced, preloaded=(cfg,w))
        our_lp = logsoftmax(lg, ol_ids[i]); our_am=int(lg.argmax())
        same=(our_am==ol_ids[i]); agree+=same
        chosen, ol_lp = ol_steps[i]
        d=abs(our_lp-ol_lp); deltas.append(d)
        print(f"  step {i}: ol_tok {chosen!r} ol_logp {ol_lp:+.3f} our_logp {our_lp:+.3f} |delta| {d:.4f} {'agree' if same else 'DIFFERS'}  ({time.time()-t1:.0f}s)", flush=True)
    print(f"\n  TEACHER-FORCED 30B: argmax agreement {agree}/{len([x for x in ol_ids if x is not None])}", flush=True)
    print(f"  logprob |delta| for the SAME token: mean={np.mean(deltas):.4f} max={np.max(deltas):.4f}", flush=True)

if __name__=="__main__": main()
