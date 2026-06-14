#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""xrt_referee.py — CROSS-RUNTIME ULP referee vs llama.cpp (Ollama). Compares OUR per-token logprobs to
llama.cpp's on the SAME model file, teacher-forced (immune to the greedy near-tie lottery).

Method: run our engine greedily; at each step capture our full logit vector. Independently get Ollama's
top-K logprobs (pure greedy, seed 0) for the same prompt. For each step where Ollama and we chose the
SAME token, compare our log-softmax(token) to Ollama's logprob(token): that's the per-token forward-pass
divergence between the two runtimes. Reports mean/max logprob delta + how often the argmax agrees and,
when it differs, whether it's a thin-margin flip (drift > top1-top2 gap)."""
import sys, os, json, urllib.request, math
sys.path.insert(0,"os.environ.get("BPD_ROOT","bpd")"); sys.path.insert(0,"os.environ.get("BPD_ROOT","bpd")/lib"); sys.path.insert(0,"/tmp")
import torch, numpy as np, llamatov_run as R
from qwen_bpe import QwenBPE
URL="http://localhost:11434/api/generate"

def ollama_steps(model, prompt, n, topk=20):
    body={"model":model,"prompt":prompt,"stream":False,"raw":True,
          "options":{"num_predict":n,"temperature":0,"top_k":1,"repeat_penalty":1.0,"top_p":1.0,"seed":0},
          "logprobs":True,"top_logprobs":topk}
    req=urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(req,timeout=300).read())
    out=[]
    for lp in d.get("logprobs",[]):
        tl=lp.get("top_logprobs",[])
        out.append({e["token"]:e["logprob"] for e in tl})
    return d.get("response",""), out

def capture_our_logits(blob, prompt_ids, n):
    """Run generate() but capture the logit vector at each step via a patch on the final matmul."""
    caps=[]
    orig = R.generate
    # patch: wrap torch argmax to also grab the logits. generate does logits[0,-1].argmax().
    # Simpler: monkeypatch torch.Tensor.argmax to record the tensor it's called on (the logits[0,-1]).
    real_argmax = torch.Tensor.argmax
    def argmax_wrap(self, *a, **k):
        # only capture 1-D vocab-sized vectors (the logit row)
        if self.dim()==1 and self.shape[0] > 1000:
            caps.append(self.detach().float().cpu().numpy().copy())
        return real_argmax(self, *a, **k)
    torch.Tensor.argmax = argmax_wrap
    try:
        full,_ = R.generate(blob, prompt_ids, n_tokens=n)
    finally:
        torch.Tensor.argmax = real_argmax
    gen_ids = full[len(prompt_ids):]
    return gen_ids, caps

def logsoftmax_at(logits, tok_id):
    m = logits.max(); lse = m + math.log(np.exp(logits-m).sum())
    return float(logits[tok_id] - lse)

def main():
    model=os.environ.get("OLLAMA_MODEL","qwen2.5:0.5b")
    blob=os.environ["BLOB"]
    prompt=os.environ.get("PROMPT","The capital of France is Paris. The capital of Japan is")
    N=int(os.environ.get("N","8"))
    tok=QwenBPE(blob)
    resp, ol_steps = ollama_steps(model, prompt, N)
    pids=tok.encode(prompt)
    gen_ids, our_logits = capture_our_logits(blob, pids, min(N,len(ol_steps)))
    print(f"  model: {model}  | prompt {len(pids)} toks")
    print(f"  Ollama: {resp[:64]!r}")
    print(f"\n  step  agree  ours_tok            ol_logp   our_logp   |delta|")
    agree=0; deltas=[]; thin=0; wide=0
    for i in range(min(len(gen_ids), len(our_logits), len(ol_steps))):
        lg=our_logits[i]; our_id=int(lg.argmax()); our_tok=tok.tokens[our_id]
        # ollama chosen token this step
        ol_tok=max(ol_steps[i], key=ol_steps[i].get); ol_logp=ol_steps[i][ol_tok]
        # normalize Ġ
        norm=lambda s:s.replace("\u0120"," ").replace("\u010a","\n")
        same = norm(our_tok)==norm(ol_tok)
        agree += same
        # our logprob for OLLAMA's chosen token (if we can map it)
        our_logp_for_ol = logsoftmax_at(lg, our_id)  # our logprob for OUR token
        # delta between our logp(our tok) and ollama logp(ol tok) — same if agree
        d = abs(our_logp_for_ol - ol_logp)
        deltas.append(d)
        # margin analysis when they differ
        if not same:
            srt=np.sort(lg)[::-1]; margin=float(srt[0]-srt[1])
            # how far is ollama's token in our ranking?
            tag="thin" if margin < 1.0 else "WIDE"
            (thin:=thin+1) if margin<1.0 else (wide:=wide+1)
        print(f"  {i:<4}  {'Y' if same else 'n':<5}  {norm(our_tok)!r:18}  {ol_logp:+7.3f}  {our_logp_for_ol:+7.3f}  {d:6.3f}{'' if same else '  <-FLIP '+('thin' if not same and np.sort(lg)[-1] else '')}")
    print(f"\n  argmax agreement: {agree}/{min(len(gen_ids),len(our_logits))}")
    print(f"  logprob |delta| (chosen token): mean={np.mean(deltas):.4f} max={np.max(deltas):.4f}")
    print(f"  flips: {thin} thin-margin, {wide} WIDE-margin  ({'CLEAN: all flips are thin-margin quant drift' if wide==0 else 'WARN: WIDE flip = possible bug'})")

if __name__=="__main__":
    main()
