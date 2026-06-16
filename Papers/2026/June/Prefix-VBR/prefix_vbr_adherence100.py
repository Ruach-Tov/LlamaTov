#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
"""prefix_vbr_adherence100.py — Prefix-VBR vs Q8_0 adherence to GOLD over 100 DIVERSE prompts
drawn from two maximally-different real registers: RECIPE prose and comp.lang.c (Usenet C-newsgroup) idiom.
Heath's expanded test: 100 prompts for statistical weight. For each prompt, full-vocab logit comparison vs
the unquantized model; aggregate mean deviation / KL, and HOW OFTEN VBR is closer to gold (the robustness
signal). Single forward per prompt (no long generation)."""
import sys, os
sys.path.insert(0, "os.environ.get("BPD_ROOT","bpd")"); sys.path.insert(0, "os.environ.get("BPD_ROOT","bpd")/lib")
import torch, numpy as np, llamatov_run as R
from qwen_bpe import QwenBPE

B = os.environ["BLOB"]; BS = 32

# --- corpus: real RECIPE prose (varied dishes/styles) ---
RECIPES = """In a large pot, bring the chicken stock to a gentle simmer over medium heat.
Add the diced onion, sliced carrots, and celery, and cook until the vegetables soften.
Season the broth with salt, black pepper, and a bay leaf, then let it reduce slowly.
Whisk the eggs with a pinch of sugar until pale, then fold in the sifted flour gradually.
Preheat the oven to 375 degrees and butter a nine inch round cake pan.
Knead the dough on a floured surface for about ten minutes until smooth and elastic.
Let the dough rise in a warm place until it has doubled in size, roughly one hour.
Sear the beef on all sides in a hot skillet before transferring it to the braising pot.
Deglaze the pan with a splash of red wine, scraping up the browned bits from the bottom.
Simmer the tomato sauce uncovered, stirring occasionally, until it thickens and deepens.
Toss the pasta with the sauce and a handful of grated parmesan just before serving.
Marinate the chicken thighs in yogurt, garlic, and spices for at least two hours.
Fold the egg whites into the batter gently so as not to deflate the air you whipped in.
Roast the vegetables on a sheet pan, tossing halfway through, until the edges caramelize.
Reduce the heat to low and let the risotto absorb the stock one ladle at a time.
Garnish the soup with a swirl of cream and a scattering of fresh chopped parsley.
Chill the dough for thirty minutes so the butter firms up before you roll it out.
Bring a large pot of salted water to a rolling boil and add the potatoes whole.
Caramelize the onions slowly over low heat until they turn a deep golden brown.
Whisk together olive oil, lemon juice, and dijon mustard to make a simple vinaigrette."""

# --- corpus: real comp.lang.c (Usenet) idiom ---
CLC = """Strictly speaking, gets() should never be used because it cannot be made safe against buffer overflow.
The behavior is undefined if you dereference a null pointer, and the standard imposes no requirement.
Array names decay to pointers in most contexts, but sizeof is one notable exception to that rule.
You cannot portably assume that a pointer and an integer have the same size on every platform.
The result of i = i++ is undefined because the object is modified twice without an intervening sequence point.
malloc returns void star, and in C you should not cast its return value; doing so can hide a missing header.
A string literal has static storage duration, so attempting to modify it invokes undefined behavior.
The order of evaluation of function arguments is unspecified, so do not rely on side effects between them.
Remember that the null pointer constant is not necessarily represented by all bits zero in memory.
When you increment a pointer, it advances by the size of the type it points to, not by one byte.
The C standard does not guarantee that signed integer overflow wraps around; it is undefined behavior.
Use size_t for object sizes and array indices, since it is the type that sizeof yields.
A function declared with an empty parameter list is not the same as one declared with void in C.
The preprocessor does not understand C syntax, so macros that look like functions can surprise you.
Comparing pointers that do not point into the same array object yields undefined behavior in general.
You should always check the return value of malloc, because it can and does return null on failure.
The difference between two pointers has type ptrdiff_t and is only meaningful within the same array.
A char may be signed or unsigned depending on the implementation, which trips up many beginners.
Volatile tells the compiler that a variable may change in ways it cannot see, so do not optimize it away.
The expression a[i] is defined to be exactly equivalent to star of a plus i, which is why i[a] also works."""

def _bs(w):
    flat=w.reshape(-1); nb=flat.shape[0]//BS; body=flat[:nb*BS].reshape(nb,BS)
    s=np.abs(body).max(axis=1,keepdims=True)/127.0; s=np.where(s==0,1e-10,s); return body,s,flat,nb
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
def softmax(x): m=x.max(); e=np.exp(x-m); return e/e.sum()
def kl(p,q): q=np.clip(q,1e-12,None); return float(np.sum(p*np.log(np.clip(p,1e-12,None)/q)))
def qmodel(wg,fn,keys):
    wq=dict(wg)
    for k in keys: wq[k]=torch.from_numpy(fn(wg[k].detach().cpu().float().numpy()).astype(np.float32))
    return wq

def build_prompts(tok, n=100, plen=8):
    # tokenize each corpus line, take the first plen tokens of each as a prompt; interleave the two registers
    prompts=[]
    for src in (RECIPES, CLC):
        for line in src.strip().split("\n"):
            ids=tok.encode(line.strip())
            if len(ids) >= plen: prompts.append(ids[:plen])
    # we have ~40 lines; to reach 100, also take MID-sentence windows (offset slices) for more diversity
    extra=[]
    for src in (RECIPES, CLC):
        for line in src.strip().split("\n"):
            ids=tok.encode(line.strip())
            if len(ids) >= plen+6: extra.append(ids[4:4+plen])      # offset window
            if len(ids) >= plen+10: extra.append(ids[7:7+plen])     # second offset
    allp = prompts + extra
    return allp[:n]

def main():
    tok=QwenBPE(B)
    cfg,wg=R.load_model(B)
    keys=[k for k,t in wg.items() if t.dim()==2 and min(t.shape)>=32]
    wq8=qmodel(wg,q8,keys); wvb=qmodel(wg,vbr,keys)
    prompts=build_prompts(tok)
    print(f"  {len(prompts)} prompts [recipe + comp.lang.c], {len(keys)} tensors/scheme", flush=True)
    a8m=[]; a8k=[]; avm=[]; avk=[]; t18=t1v=0; vw_m=vw_k=ties=0
    for pi,prompt in enumerate(prompts):
        lg,_=R.generate_logits(B,prompt,preloaded=(cfg,wg)); lg=lg.astype(np.float64); pg=softmax(lg); gt=int(np.argmax(lg))
        l8,_=R.generate_logits(B,prompt,preloaded=(cfg,wq8)); l8=l8.astype(np.float64)
        lv,_=R.generate_logits(B,prompt,preloaded=(cfg,wvb)); lv=lv.astype(np.float64)
        m8=np.abs(l8-lg).mean(); mv=np.abs(lv-lg).mean(); k8=kl(pg,softmax(l8)); kv=kl(pg,softmax(lv))
        a8m.append(m8); a8k.append(k8); avm.append(mv); avk.append(kv)
        t18+=int(np.argmax(l8)==gt); t1v+=int(np.argmax(lv)==gt)
        vw_m+=(mv<m8); vw_k+=(kv<k8); ties+=(mv==m8)
        if pi%10==0: print(f"  ...{pi}/{len(prompts)} (vbr_wins_so_far mean={vw_m})", flush=True)
    n=len(prompts)
    print(f"\n  === AGGREGATE over {n} prompts (recipe + comp.lang.c) ===", flush=True)
    print(f"  q8_0 : mean|dlog|={np.mean(a8m):.4f}  meanKL={np.mean(a8k):.3e}  top1={t18}/{n}", flush=True)
    print(f"  vbr  : mean|dlog|={np.mean(avm):.4f}  meanKL={np.mean(avk):.3e}  top1={t1v}/{n}", flush=True)
    print(f"  VBR closer-to-gold on mean|dlog|: {vw_m}/{n}  | on KL: {vw_k}/{n}  (ties={ties})", flush=True)
    print(f"  ratio: q8 mean / vbr mean = {np.mean(a8m)/np.mean(avm):.2f}x", flush=True)

if __name__=="__main__": main()
