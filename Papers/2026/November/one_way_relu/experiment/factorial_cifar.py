#!/usr/bin/env python3
"""factorial_cifar.py — CIFAR-10 arm of the one-way factorial (paired seeds).

Same testbed CLASS as factorial_emnist (conv-conv-pool-fc1-fc2) adapted:
3-channel input, 10 classes, 32x32. Data: HuggingFace cifar10 parquet
(pyarrow), normalized per-channel. Paired seeds: seed k gives identical
init + batch order in every arm.

Usage: python3 factorial_cifar.py --arms relu leaky_10 oneway_10 \
         --seeds 10 --epochs 10 --device cuda [--optimizer adam|sgd] \
         [--save-ckpt DIR] [--out results_cifar.jsonl]
"""
import argparse, io, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation

DATA = Path(__file__).parent / 'data_cifar10'
N_CLASSES = 10


def load_cifar(device):
    import pyarrow.parquet as pq
    from PIL import Image
    xs, ys = {}, {}
    for split, fname in [('train', 'train-00000-of-00001.parquet'),
                         ('test', 'test-00000-of-00001.parquet')]:
        t = pq.read_table(DATA / fname)
        d = t.to_pydict()
        imgs = d['img']
        labels = d['label']
        arr = np.zeros((len(imgs), 32, 32, 3), dtype=np.uint8)
        for i, im in enumerate(imgs):
            arr[i] = np.array(Image.open(io.BytesIO(im['bytes'])))
        x = torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        y = torch.tensor(labels, dtype=torch.long)
        xs[split], ys[split] = x, y
    mean = xs['train'].mean(dim=(0, 2, 3), keepdim=True)
    std = xs['train'].std(dim=(0, 2, 3), keepdim=True)
    for s in xs:
        xs[s] = (xs[s] - mean) / std
    return (xs['train'].to(device), ys['train'].to(device),
            xs['test'].to(device), ys['test'].to(device))


class NetC(nn.Module):
    """CIFAR adaptation of the testbed net: conv3->32, conv32->64, pool,
    fc(14400->128), fc(128->10). Same structural class as NetE."""
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.c1 = nn.Conv2d(3, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.f1 = nn.Linear(64 * 14 * 14, 128)
        self.f2 = nn.Linear(128, N_CLASSES)
        self.pre = {}

    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.c2(x)
        x = F.max_pool2d(self.act(x), 2)
        x = torch.flatten(x, 1)
        pre = self.f1(x)
        self.pre['f1'] = pre.detach()
        x = self.act(pre)
        return self.f2(x)


def dead_stats(net, x):
    with torch.no_grad():
        net(x[:4096])
    pre = net.pre['f1']
    return (pre <= 0).all(dim=0).float().mean().item()


def run(arm, seed, epochs, device, batch=128, lr=1e-3, ckpt_dir='',
        optimizer='adam'):
    torch.manual_seed(seed)
    net = NetC(make_activation(arm)).to(device)
    if optimizer == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr)
    else:
        opt = torch.optim.SGD(net.parameters(), lr * 10, momentum=0.9)
    xtr, ytr, xte, yte = DATA_CACHE
    n = len(xtr)
    history = []
    prev_dead = None
    revivals = 0
    t0 = time.time()
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
            for i in range(0, len(xte), 4096):
                correct += (net(xte[i:i+4096]).argmax(1) == yte[i:i+4096]).sum().item()
        acc = correct / len(xte)
        dead = dead_stats(net, xtr)
        if prev_dead is not None and dead < prev_dead:
            revivals += 1
        prev_dead = dead
        history.append({'epoch': ep, 'acc': acc, 'dead': {'f1': dead}})
    if ckpt_dir:
        Path(ckpt_dir).mkdir(exist_ok=True)
        torch.save(net.state_dict(),
                   Path(ckpt_dir) / f'{arm}_s{seed}_e{epochs}.pt')
    return {'arm': arm, 'seed': seed, 'epochs': epochs, 'dataset': 'cifar10',
            'optimizer': optimizer, 'wall_s': time.time() - t0,
            'revivals_f1': revivals, 'final_acc': history[-1]['acc'],
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
    ap.add_argument('--out', default='results_cifar.jsonl')
    args = ap.parse_args()
    DATA_CACHE = load_cifar(args.device)
    globals()['DATA_CACHE'] = DATA_CACHE
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for arm in args.arms:
            r = run(arm, seed, args.epochs, args.device,
                    ckpt_dir=args.save_ckpt, optimizer=args.optimizer)
            line = (f"{arm:10s} seed {seed}: acc {r['final_acc']:.4f} "
                    f"revivals(f1) {r['revivals_f1']:3d} "
                    f"dead(f1) {r['history'][-1]['dead']['f1']:.3f} "
                    f"[{r['wall_s']:.0f}s]")
            print(line, flush=True)
            if args.out != '/dev/null':
                with open(args.out, 'a') as f:
                    f.write(json.dumps(r) + '\n')
