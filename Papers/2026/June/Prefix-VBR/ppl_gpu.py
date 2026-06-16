#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
"""ppl_gpu.py — WikiText-2 perplexity via the GPU engine's single-pass score_seq.
FP16(gold) vs Q8_0 vs Prefix-VBR (round-trip), serialize-on-model. Standard metric: concat wiki.test.raw,
tokenize, slide CTX-token windows, accumulate NLL, PPL=exp(nll/ntok). Env: BLOB, WIKI, CTX=512, MAXTOK."""
import sys, os, math, time
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/home/iyun/Ruach-Tov/bpd"); sys.path.insert(0, "/home/iyun/Ruach-Tov/bpd/lib")
import numpy as np, torch
import llamatov_run as R, llamatov_gpu_qwen3 as G
from qwen_bpe import QwenBPE
import importlib.util
spec = importlib.util.spec_from_file_location("adh", "/tmp/prefix_vbr_adherence100.py")
adh = importlib.util.module_from_spec(spec); spec.loader.exec_module(adh)

B = os.environ["BLOB"]; WIKI = os.environ["WIKI"]
CTX = int(os.environ.get("CTX", "512")); MAXTOK = int(os.environ.get("MAXTOK", "4000"))
DTYPE = torch.float16 if os.environ.get("FP16", "0") == "1" else None

def ppl_over_windows(cfg, w, ids):
    """Slide non-overlapping CTX windows, accumulate NLL via score_seq. (Non-overlapping = standard 'stride=CTX'.)"""
    nll_tot = 0.0; ntok_tot = 0
    n = min(len(ids), MAXTOK)
    for start in range(0, n, CTX):
        window = ids[start:start+CTX]
        if len(window) < 2: break
        nll, ntok = G.score_seq(cfg, w, window)
        nll_tot += nll; ntok_tot += ntok
    return math.exp(nll_tot / max(1, ntok_tot)), ntok_tot

def main():
    tok = QwenBPE(B)
    ids = tok.encode(open(WIKI, encoding="utf-8", errors="ignore").read())
    print(f"  WikiText-2: {len(ids)} tokens (first {MAXTOK}, CTX={CTX}, dtype={'fp16' if DTYPE else 'fp32'})", flush=True)
    cfg, wg = G.load_gpu(B, dtype=DTYPE)
    keys = [k for k, t in wg.items() if t.dim() == 2 and min(t.shape) >= 32]
    # NB: wg is already on GPU; for quant schemes we re-quantize from a CPU copy. Simplest: reload per scheme.
    results = {}
    for name, fn in [("FP16_gold", None), ("Q8_0", adh.q8), ("Prefix_VBR", adh.vbr)]:
        t0 = time.time()
        cfg2, w2 = G.load_gpu(B, quantize_fn=fn, dtype=DTYPE)
        ppl, ntok = ppl_over_windows(cfg2, w2, ids)
        results[name] = ppl
        print(f"  [{name:11s}] PPL={ppl:.4f}  (n={ntok}, {time.time()-t0:.0f}s)", flush=True)
        del w2; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n  === WikiText-2 PERPLEXITY (qwen3:0.6b) ===", flush=True)
    g = results["FP16_gold"]
    for name in ("FP16_gold", "Q8_0", "Prefix_VBR"):
        print(f"  {name:11s}: {results[name]:.4f}   (delta vs FP16: {results[name]-g:+.4f})", flush=True)
    closer = abs(results['Prefix_VBR']-g) < abs(results['Q8_0']-g)
    print(f"  -> Prefix-VBR {'CLOSER to FP16' if closer else 'NOT closer'} than Q8_0", flush=True)

if __name__ == "__main__":
    main()
