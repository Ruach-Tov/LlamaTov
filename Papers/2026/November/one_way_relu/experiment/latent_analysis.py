"""Latent-space probes for one-way-leaky checkpoints (night campaign phase 2).

For each saved model: extract f1 (128-d penultimate) features on the EMNIST
test set, then measure:
  1. participation ratio (effective dimensionality of the representation):
     PR = (sum eig)^2 / sum(eig^2) of the feature covariance — how many
     dimensions the network actually USES. Hypothesis: never-dead arms use
     more of the 128.
  2. linear separability: ridge-classifier accuracy on f1 features
     (representation quality independent of the final layer).
  3. class scatter: trace(between-class) / trace(within-class) — Fisher-style
     separability of the latent clusters.
Usage: python3 latent_analysis.py --ckpt-dir ckpt_long --device cuda
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from factorial_emnist import NetE, load_emnist
from activations import make_activation


def f1_features(net, x, batch=1024):
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            net(x[i:i + batch])
            feats.append(net.pre['f1'].clone())
    return torch.cat(feats)


def participation_ratio(F):
    Fc = F - F.mean(0, keepdim=True)
    cov = (Fc.t() @ Fc) / (len(Fc) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def fisher_ratio(F, y, n_classes=47):
    mu = F.mean(0)
    within = 0.0
    between = 0.0
    for c in range(n_classes):
        Fc = F[y == c]
        if len(Fc) < 2:
            continue
        muc = Fc.mean(0)
        within += float(((Fc - muc) ** 2).sum())
        between += len(Fc) * float(((muc - mu) ** 2).sum())
    return between / within


def linear_probe(F, y, Fte, yte):
    # closed-form ridge regression to one-hot targets:
    n, d = F.shape
    Y = torch.zeros(n, 47, device=F.device)
    Y[torch.arange(n), y] = 1.0
    A = F.t() @ F + 1e-2 * torch.eye(d, device=F.device)
    W = torch.linalg.solve(A, F.t() @ Y)
    pred = (Fte @ W).argmax(1)
    return float((pred == yte).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='ckpt_long')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='latent_results.jsonl')
    args = ap.parse_args()
    xtr, ytr, xte, yte = load_emnist(args.device)
    # probe-train on a 20k subsample of train, evaluate on test:
    sub = torch.randperm(len(xtr))[:20000].to(xtr.device)
    results = []
    for ck in sorted(Path(args.ckpt_dir).glob('*.pt')):
        arm = ck.stem.rsplit('_s', 1)[0]
        net = NetE(make_activation(arm)).to(args.device)
        net.load_state_dict(torch.load(ck, map_location=args.device))
        net.eval()
        Ftr = f1_features(net, xtr[sub])
        Fte = f1_features(net, xte)
        r = {'ckpt': ck.name, 'arm': arm,
             'participation_ratio': participation_ratio(Fte),
             'fisher_ratio': fisher_ratio(Fte, yte),
             'linear_probe_acc': linear_probe(Ftr, ytr[sub], Fte, yte)}
        results.append(r)
        print(json.dumps(r), flush=True)
    with open(args.out, 'a') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
