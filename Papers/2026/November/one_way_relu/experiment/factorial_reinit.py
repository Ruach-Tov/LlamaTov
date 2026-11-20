#!/usr/bin/env python3
"""factorial_reinit.py — two initialization/revival experiments (Heath,
2026-11-17), CifarNet testbed.

MODE reinit: when a unit is squelched at an epoch boundary during the
FIRST HALF of training, stochastically re-initialize its incoming
weights (fresh Kaiming draw); second half, no intervention (standard
dynamics). Chart: does re-initialization make units more viable than
natural revival? Tracks per-epoch dead fraction + reinit events +
final viability of reinit'd units.

MODE zerosum: project a fraction of c1 kernels to Sobel-sum-0 at init
(per-kernel mean subtraction => kernel responds to structure, not DC).
Sweep fraction in {0, .25, .5, .75, 1.0} x {relu, oneway_10}.
"""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation
from factorial_cifar import load_cifar, NetC


def dead_mask(net, x):
    with torch.no_grad():
        net(x[:4096])
    return (net.pre['f1'] <= 0).all(dim=0)  # [128] bool


def run_reinit(arm, seed, epochs, device, reinit_half=True, batch=128,
               lr=1e-3):
    torch.manual_seed(seed)
    net = NetC(make_activation(arm)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr)
    xtr, ytr, xte, yte = DATA
    n = len(xtr)
    reinit_log = []   # (epoch, unit) events
    ever_reinit = torch.zeros(128, dtype=torch.bool)
    history = []
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed * 10000 + ep)
        perm = torch.randperm(n, generator=g).to(device)
        net.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
        dm = dead_mask(net, xtr)
        # intervention: first half only
        if reinit_half and ep < epochs // 2 and dm.any():
            dead_idx = dm.nonzero().flatten()
            with torch.no_grad():
                W = net.f1.weight  # [128, in]
                fresh = torch.empty(len(dead_idx), W.shape[1],
                                    device=device)
                nn.init.kaiming_uniform_(fresh, a=5**0.5)
                W[dead_idx] = fresh
                net.f1.bias[dead_idx] = 0.0
                # reset Adam state for these rows:
                st = opt.state.get(net.f1.weight)
                if st and 'exp_avg' in st:
                    st['exp_avg'][dead_idx] = 0
                    st['exp_avg_sq'][dead_idx] = 0
            for u in dead_idx.tolist():
                reinit_log.append((ep, u))
            ever_reinit[dead_idx.cpu()] = True
        net.eval()
        with torch.no_grad():
            correct = sum((net(xte[i:i+2048]).argmax(1) == yte[i:i+2048]).sum().item()
                          for i in range(0, len(xte), 2048))
        history.append({'epoch': ep, 'acc': correct/len(xte),
                        'dead_f1': dm.float().mean().item()})
    final_dead = dead_mask(net, xtr).cpu()
    n_reinit = int(ever_reinit.sum())
    viable_reinit = int((ever_reinit & ~final_dead).sum())
    return {'mode': 'reinit', 'arm': arm, 'seed': seed,
            'reinit_half': reinit_half,
            'final_acc': history[-1]['acc'],
            'reinit_events': len(reinit_log),
            'units_ever_reinit': n_reinit,
            'reinit_viable_at_end': viable_reinit,
            'final_dead_f1': history[-1]['dead_f1'],
            'history': history}


def zerosum_project(net, frac, seed):
    """Project frac of c1 kernels to per-kernel zero sum."""
    g = torch.Generator().manual_seed(seed * 777)
    n_ch = net.c1.weight.shape[0]
    k = int(round(frac * n_ch))
    if k == 0:
        return 0
    chosen = torch.randperm(n_ch, generator=g)[:k]
    with torch.no_grad():
        W = net.c1.weight  # [32, 3, 3, 3]
        for c in chosen.tolist():
            W[c] -= W[c].mean()
    return k


def run_zerosum(arm, seed, frac, epochs, device, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    net = NetC(make_activation(arm)).to(device)
    k = zerosum_project(net, frac, seed)
    opt = torch.optim.Adam(net.parameters(), lr)
    xtr, ytr, xte, yte = DATA
    n = len(xtr)
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed * 10000 + ep)
        perm = torch.randperm(n, generator=g).to(device)
        net.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        correct = sum((net(xte[i:i+2048]).argmax(1) == yte[i:i+2048]).sum().item()
                      for i in range(0, len(xte), 2048))
    dm = dead_mask(net, xtr)
    return {'mode': 'zerosum', 'arm': arm, 'seed': seed, 'frac': frac,
            'kernels_projected': k, 'final_acc': correct/len(xte),
            'final_dead_f1': dm.float().mean().item()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['reinit', 'zerosum'])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results_reinit.jsonl')
    args = ap.parse_args()
    DATA = load_cifar(args.device)
    globals()['DATA'] = DATA
    if args.mode == 'reinit':
        for seed in range(args.seeds):
            for arm, rh in [('relu', False), ('relu', True),
                            ('oneway_10', False)]:
                r = run_reinit(arm, seed, args.epochs, args.device,
                               reinit_half=rh)
                tag = f"{arm}{'+reinit' if rh else ''}"
                print(f"{tag:16s} seed {seed}: acc {r['final_acc']:.4f} "
                      f"dead {r['final_dead_f1']:.3f} "
                      f"reinits {r['reinit_events']} "
                      f"viable {r['reinit_viable_at_end']}/{r['units_ever_reinit']}",
                      flush=True)
                with open(args.out, 'a') as f:
                    f.write(json.dumps(r) + '\n')
    else:
        for seed in range(args.seeds):
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for arm in ['relu', 'oneway_10']:
                    r = run_zerosum(arm, seed, frac, args.epochs,
                                    args.device)
                    print(f"{arm:10s} frac {frac:.2f} seed {seed}: "
                          f"acc {r['final_acc']:.4f} "
                          f"dead {r['final_dead_f1']:.3f}", flush=True)
                    with open(args.out, 'a') as f:
                        f.write(json.dumps(r) + '\n')
