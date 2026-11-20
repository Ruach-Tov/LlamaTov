#!/usr/bin/env python3
"""factorial_vgg.py — VGG-11 (Simonyan & Zissermann config A) on CIFAR-10,
one-way factorial arm with PER-LAYER dead-fraction logging at all 10 ReLU
sites. The depth instrument: how does squelching compound with depth, and
where does one-way's repair land?

Usage: python3 factorial_vgg.py --arms relu leaky_10 oneway_10 \
         --seeds 10 --epochs 10 --device cuda [--optimizer adam|sgd]
"""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation
from factorial_cifar import load_cifar

# VGG-11 conv config (cifar-vgg convention): channels, 'M' = maxpool
CFG_A = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
DROPOUT = 0.0


class VGG11(nn.Module):
    """VGG-11 for 32x32: 8 conv + 2 hidden FC + head = 11 weight layers,
    10 ReLU sites (8 conv + 2 fc). Pre-activations cached per site.
    Optional dropout (Heath's confound hypothesis 2026-11-17: squelching
    may provide dropout-adjacent regularization — so one-way removes both
    the pathology AND an accidental regularizer; explicit dropout should
    restore the benefit if so): p=0.5 on fc sites, standard placement."""
    def __init__(self, act, n_classes=10, dropout=0.0, site_acts=None):
        # site_acts: optional dict site_name -> activation fn, overriding
        # `act` at those sites (per-site convention mixing, Heath
        # 2026-11-17: one-way only at the deepest site(s), etc.)
        super().__init__()
        self.act = act
        self.site_acts = site_acts or {}
        layers = []
        in_c = 3
        self.conv_ids = []
        for v in CFG_A:
            if v == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.Conv2d(in_c, v, 3, padding=1))
                self.conv_ids.append(len(layers) - 1)
                in_c = v
        self.features = nn.ModuleList(layers)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.head = nn.Linear(512, n_classes)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pre = {}  # site_name -> pre-activation (detached)

    def _act_for(self, name):
        return self.site_acts.get(name, self.act)

    def forward(self, x):
        site = 0
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.conv_ids:
                name = f'c{site}'
                self.pre[name] = x.detach()
                x = self._act_for(name)(x)
                site += 1
        x = torch.flatten(x, 1)
        for name, fc in [('f0', self.fc1), ('f1', self.fc2)]:
            x = fc(x)
            self.pre[name] = x.detach()
            x = self._act_for(name)(x)
            x = self.drop(x)
        return self.head(x)


SITES = [f'c{i}' for i in range(8)] + ['f0', 'f1']


def dead_profile(net, x):
    """Per-site fraction of channels/units dead (never active) on a batch."""
    with torch.no_grad():
        net(x[:2048])
    prof = {}
    for s in SITES:
        pre = net.pre[s]
        if pre.dim() == 4:  # conv: channel dead if never positive anywhere
            dead = (pre <= 0).all(dim=(0, 2, 3)).float().mean().item()
        else:
            dead = (pre <= 0).all(dim=0).float().mean().item()
        prof[s] = round(dead, 4)
    return prof


def run(arm, seed, epochs, device, batch=128, lr=1e-3, ckpt_dir='',
        optimizer='adam'):
    torch.manual_seed(seed)
    site_acts = None
    base_arm = arm
    if arm.startswith('mix:'):
        # arm syntax: mix:<sites>  e.g. mix:f0,f1 — one-way at the named
        # sites, strict relu everywhere else.
        sites = arm.split(':', 1)[1].split(',')
        ow = make_activation('oneway_10')
        site_acts = {s: ow for s in sites}
        base_arm = 'relu'
    net = VGG11(make_activation(base_arm), dropout=DROPOUT,
                site_acts=site_acts).to(device)
    if optimizer == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr)
    else:
        opt = torch.optim.SGD(net.parameters(), lr * 10, momentum=0.9)
    xtr, ytr, xte, yte = DATA_CACHE
    n = len(xtr)
    t0 = time.time()
    history = []
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed * 10000 + ep)
        perm = torch.randperm(n, generator=g).to(device)
        net.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, len(xte), 2048):
                correct += (net(xte[i:i+2048]).argmax(1) == yte[i:i+2048]).sum().item()
        acc = correct / len(xte)
        prof = dead_profile(net, xtr)
        history.append({'epoch': ep, 'acc': acc, 'dead': prof})
    if ckpt_dir:
        Path(ckpt_dir).mkdir(exist_ok=True)
        torch.save(net.state_dict(),
                   Path(ckpt_dir) / f'{arm}_s{seed}_e{epochs}.pt')
    return {'arm': arm, 'seed': seed, 'epochs': epochs, 'dataset': 'cifar10',
            'net': 'vgg11', 'optimizer': optimizer,
            'wall_s': time.time() - t0, 'final_acc': history[-1]['acc'],
            'history': history}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    ap.add_argument('--save-ckpt', default='')
    ap.add_argument('--out', default='results_vgg.jsonl')
    ap.add_argument('--dropout', type=float, default=0.0,
                    help='fc dropout p (dropout-confound factorial arm)')
    args = ap.parse_args()
    globals()['DROPOUT'] = args.dropout
    DATA_CACHE = load_cifar(args.device)
    globals()['DATA_CACHE'] = DATA_CACHE
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for arm in args.arms:
            r = run(arm, seed, args.epochs, args.device,
                    ckpt_dir=args.save_ckpt, optimizer=args.optimizer)
            d = r['history'][-1]['dead']
            worst = max(d, key=d.get)
            print(f"{arm:10s} seed {seed}: acc {r['final_acc']:.4f} "
                  f"dead[{worst}]={d[worst]:.3f} "
                  f"mean_dead={sum(d.values())/len(d):.3f} "
                  f"[{r['wall_s']:.0f}s]", flush=True)
            if args.out != '/dev/null':
                with open(args.out, 'a') as f:
                    f.write(json.dumps(r) + '\n')
