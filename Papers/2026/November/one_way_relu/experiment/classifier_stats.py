"""Classical classifier-quality stats for one-way vs leaky (Heath, 2026-11-15).

Per arm (epoch-10 checkpoints): confusion matrix, per-class precision/recall/
F1, macro-F1, balanced accuracy, top-5 most-confused class pairs, per-class
F1 DELTA between arms (which letters/digits benefit?), and expected
calibration error (15-bin) since softmax confidence quality is a classical
axis accuracy doesn't see.

Usage: python3 classifier_stats.py --ckpt-dir ckpt_e10 --device cuda
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from factorial_emnist import NetE, load_emnist, N_CLASSES
from activations import make_activation


def eval_model(net, xte, yte, batch=1024):
    probs = []
    with torch.no_grad():
        for i in range(0, len(xte), batch):
            probs.append(torch.softmax(net(xte[i:i+batch]), dim=1))
    P = torch.cat(probs)
    pred = P.argmax(1)
    conf = P.max(1).values
    C = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long)
    for t, p in zip(yte.cpu(), pred.cpu()):
        C[t, p] += 1
    return C, pred, conf


def per_class_f1(C):
    tp = C.diag().float()
    prec = tp / C.sum(0).clamp(min=1).float()
    rec = tp / C.sum(1).clamp(min=1).float()
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    return prec, rec, f1


def ece(conf, correct, bins=15):
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += float(m.float().mean() * (conf[m].mean() - correct[m].float().mean()).abs())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='ckpt_e10')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    _, _, xte, yte = load_emnist(args.device)
    # EMNIST-Balanced class names (mapping file order: 0-9, A-Z, a-z merged set):
    names = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')
    agg = {}
    for ck in sorted(Path(args.ckpt_dir).glob('*.pt')):
        arm = ck.stem.rsplit('_s', 1)[0]
        net = NetE(make_activation(arm)).to(args.device)
        net.load_state_dict(torch.load(ck, map_location=args.device))
        net.eval()
        C, pred, conf = eval_model(net, xte, yte)
        prec, rec, f1 = per_class_f1(C)
        correct = (pred == yte).cpu()
        a = agg.setdefault(arm, {'C': torch.zeros_like(C), 'f1': [], 'ece': [],
                                 'bal': []})
        a['C'] += C
        a['f1'].append(f1)
        a['ece'].append(ece(conf.cpu(), correct))
        a['bal'].append(float(rec.mean()))
    print(f"{'arm':12s} {'macroF1':>8s} {'balAcc':>8s} {'ECE':>7s}")
    for arm, a in agg.items():
        f1m = torch.stack(a['f1']).mean(0)
        print(f"{arm:12s} {float(f1m.mean()):8.4f} "
              f"{sum(a['bal'])/len(a['bal']):8.4f} "
              f"{sum(a['ece'])/len(a['ece']):7.4f}")
    arms = list(agg)
    if len(arms) == 2:
        f1a = torch.stack(agg[arms[0]]['f1']).mean(0)
        f1b = torch.stack(agg[arms[1]]['f1']).mean(0)
        d = f1b - f1a
        order = d.abs().argsort(descending=True)[:8]
        print(f"\nLargest per-class F1 deltas ({arms[1]} - {arms[0]}):")
        for i in order:
            print(f"  class {names[i]:2s}: {float(d[i]):+.4f} "
                  f"({float(f1a[i]):.3f} -> {float(f1b[i]):.3f})")
        # top confusions in the pooled matrices:
        for arm in arms:
            C = agg[arm]['C'].clone()
            C.fill_diagonal_(0)
            flat = C.flatten()
            top = flat.argsort(descending=True)[:5]
            pairs = [(int(t // N_CLASSES), int(t % N_CLASSES), int(flat[t])) for t in top]
            s = ', '.join(f"{names[a]}->{names[b]}:{n}" for a, b, n in pairs)
            print(f"top confusions {arm}: {s}")


if __name__ == '__main__':
    main()
