#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
"""prefix_vbr_logits.py — compare each quantized scheme's FULL LOGIT VECTOR to the unquantized gold.
Finer than token-match: how faithfully does each scheme reproduce the model's actual output distribution?
Metrics vs gold logits [single forward, the prompt's final position]:
  - mean/max |delta logit|       [raw closeness across the whole vocab]
  - KL(softmax_gold || softmax_q) [how different the predicted probability distributions are, in nats]
  - top-1 / top-5 / top-10 agreement [rank preservation]
  - Pearson r between logit vectors [overall shape fidelity]
Heath's question: compare with the logits of the unquantized model."""
import sys, os
sys.path.insert(0, "os.environ.get("BPD_ROOT","bpd")"); sys.path.insert(0, "os.environ.get("BPD_ROOT","bpd")/lib")
import torch, numpy as np, llamatov_run as R

B = os.environ["BLOB"]; BS = 32

def _bs(w):
    flat = w.reshape(-1); nb = flat.shape[0]//BS; body = flat[:nb*BS].reshape(nb,BS)
    s = np.abs(body).max(axis=1,keepdims=True)/127.0; s = np.where(s==0,1e-10,s); return body,s,flat,nb
def q8(w):
    body,s,flat,nb=_bs(w); out=flat.copy(); out[:nb*BS]=(np.round(body/s).clip(-127,127)*s).reshape(-1); return out.reshape(w.shape)
_VBR=[(0,32,7),(32,64,6),(64,96,5),(96,127,4)]
def vbr(w):
    body,s,flat,nb=_bs(w); wn=body/s; sign=np.sign(wn); mag=np.minimum(np.abs(wn),126.9)
    rec=np.zeros_like(mag); filled=np.zeros(mag.shape,bool)
    for lo,hi,bits in _VBR:
        L=2**bits; m=(mag<hi)&(~filled); q=np.clip(np.round((mag-lo)/(hi-lo)*(L-1)),0,L-1)
        rec=np.where(m, lo+q/max(1,L-1)*(hi-lo), rec); filled|=m
    out=flat.copy(); out[:nb*BS]=(rec*sign*s).reshape(-1); return out.reshape(w.shape)
SCHEMES={"q8_0":q8, "prefix_vbr":vbr}

def softmax(x):
    m=x.max(); e=np.exp(x-m); return e/e.sum()
def kl(p,q):  # KL(p||q) in nats
    q=np.clip(q,1e-12,None); return float(np.sum(p*np.log(np.clip(p,1e-12,None)/q)))

def main():
    prompt=[785,6722,315,9625,374]
    cfg,wg=R.load_model(B)
    lg_gold,_=R.generate_logits(B,prompt,preloaded=(cfg,wg))
    lg_gold=lg_gold.astype(np.float64)
    pg=softmax(lg_gold); gtop=np.argsort(lg_gold)[::-1]
    print(f"  gold argmax={int(gtop[0])}  gold top5={gtop[:5].tolist()}", flush=True)
    keys=[k for k,t in wg.items() if t.dim()==2 and min(t.shape)>=32]
    for name,fn in SCHEMES.items():
        wq=dict(wg)
        for k in keys: wq[k]=torch.from_numpy(fn(wg[k].detach().cpu().float().numpy()).astype(np.float32))
        lg,_=R.generate_logits(B,prompt,preloaded=(cfg,wq)); lg=lg.astype(np.float64)
        d=np.abs(lg-lg_gold)
        pq=softmax(lg)
        qtop=np.argsort(lg)[::-1]
        t1=int(qtop[0]==gtop[0])
        t5=len(set(qtop[:5].tolist())&set(gtop[:5].tolist()))
        t10=len(set(qtop[:10].tolist())&set(gtop[:10].tolist()))
        r=float(np.corrcoef(lg,lg_gold)[0,1])
        print(f"  [{name:11s}] mean|dlog|={d.mean():.4f} max|dlog|={d.max():.4f}  "
              f"KL(gold||q)={kl(pg,pq):.3e} nats  top1={t1} top5={t5}/5 top10={t10}/10  pearson_r={r:.6f}", flush=True)

if __name__=="__main__": main()
