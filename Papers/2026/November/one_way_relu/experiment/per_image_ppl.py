#!/usr/bin/env python3
"""per_image_ppl.py — per-image perplexity on the EMNIST test set.

For a checkpoint: softmax NLL of the TRUE class per image -> perplexity
exp(nll). Rank: most-perplexing correct, most-perplexing wrong,
LEAST-perplexing wrong (Heath's conjecture: confidently-wrong errors
are the richest soil for mechanisms).
"""
import json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation
from factorial_emnist import load_emnist, NetE

dev = 'cuda'
BAL = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

def analyze(ckpt, arm):
    net = NetE(make_activation(arm)).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    net.eval()
    xtr, ytr, xte, yte = DATA
    nlls, preds = [], []
    with torch.no_grad():
        for i in range(0, len(xte), 1024):
            logits = net(xte[i:i+1024])
            lp = F.log_softmax(logits, dim=1)
            nlls.append(-lp[torch.arange(len(lp)), yte[i:i+1024]])
            preds.append(logits.argmax(1))
    nll = torch.cat(nlls); pred = torch.cat(preds)
    correct = pred == yte
    ppl = nll.exp()
    out = {'arm': arm, 'ckpt': str(ckpt),
           'acc': correct.float().mean().item(),
           'mean_ppl_correct': ppl[correct].mean().item(),
           'mean_ppl_wrong': ppl[~correct].mean().item()}
    def topk(mask, largest, k=12):
        idx = mask.nonzero().flatten()
        vals = nll[idx]
        order = vals.argsort(descending=largest)[:k]
        sel = idx[order]
        return [{'i': int(i), 'true': BAL[yte[i]], 'pred': BAL[pred[i]],
                 'ppl': float(ppl[i]), 'p_true': float((-nll[i]).exp()),
                 'p_pred': float(F.softmax(net(xte[i:i+1]), 1).max())}
                for i in sel]
    with torch.no_grad():
        out['most_perplexing_correct'] = topk(correct, True)
        out['most_perplexing_wrong'] = topk(~correct, True)
        out['least_perplexing_wrong'] = topk(~correct, False)
    return out

if __name__ == '__main__':
    DATA = load_emnist(dev)
    globals()['DATA'] = DATA
    results = []
    for ckpt, arm in [(sys.argv[1], sys.argv[2])] if len(sys.argv) > 2 else []:
        pass
    import glob
    for arm in ['relu', 'oneway_10']:
        cands = sorted(glob.glob(f'ckpt_e10/{arm}_s200_*.pt'))
        if not cands:
            print(f'no ckpt for {arm}'); continue
        r = analyze(cands[0], arm)
        results.append(r)
        print(f"{arm}: acc {r['acc']:.4f}  ppl(correct) {r['mean_ppl_correct']:.3f}  ppl(wrong) {r['mean_ppl_wrong']:.2f}")
        print('  LEAST-perplexing errors (confidently wrong):')
        for e in r['least_perplexing_wrong'][:6]:
            print(f"    img {e['i']:5d}: true {e['true']} pred {e['pred']}  ppl {e['ppl']:.3f}  p(true) {e['p_true']:.4f}")
        print('  MOST-perplexing errors:')
        for e in r['most_perplexing_wrong'][:3]:
            print(f"    img {e['i']:5d}: true {e['true']} pred {e['pred']}  ppl {e['ppl']:.1f}")
        print('  MOST-perplexing correct:')
        for e in r['most_perplexing_correct'][:3]:
            print(f"    img {e['i']:5d}: true {e['true']} pred {e['pred']}  ppl {e['ppl']:.2f}")
    Path('per_image_ppl.json').write_text(json.dumps(results))
