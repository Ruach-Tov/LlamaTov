#!/usr/bin/env python3
"""necrosis_pilot.py — n=3 epoch-resolution silencing cartography pilot.

VGG-11, Adam, ReLU: after every epoch, record per-unit silencing masks
at all 10 sites + interlayer weight-mass matrices. Output: one JSON per
seed with the full time-resolved spread picture — directed cascade vs
dynamic equilibrium.
"""
import json, time
from pathlib import Path
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation
from factorial_cifar import load_cifar
from factorial_vgg_dropout import VGG11, SITES

dev = 'cuda'


def site_masks(net, x):
    """Per-site boolean dead masks (unit never active on probe batch)."""
    with torch.no_grad():
        net(x[:2048])
    masks = {}
    for s in SITES:
        pre = net.pre[s]
        if pre.dim() == 4:
            masks[s] = (pre <= 0).all(dim=(0, 2, 3)).cpu()
        else:
            masks[s] = (pre <= 0).all(dim=0).cpu()
    return masks


def weight_mass(net):
    """Interlayer |W| mass matrices between adjacent sites (for
    supplier/consumer attribution). conv->conv: [out_ch, in_ch] summed
    over kernel; conv->fc via the flatten channel blocks; fc->fc direct."""
    wm = {}
    convs = [m for m in net.features if isinstance(m, torch.nn.Conv2d)]
    for i in range(1, len(convs)):
        wm[f'c{i-1}->c{i}'] = convs[i].weight.data.abs().sum(dim=(2, 3)).cpu()
    # c7 -> f0 (fc1 sees 512 channels x 1x1 after pools at 32x32 input):
    W = net.fc1.weight.data.abs()  # [512, 512*1*1]
    wm['c7->f0'] = W.reshape(512, 512, -1).sum(dim=2).cpu()
    wm['f0->f1'] = net.fc2.weight.data.abs().cpu()
    return wm


def run(seed, epochs=10, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    net = VGG11(make_activation('relu')).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr)
    xtr, ytr, xte, yte = DATA
    n = len(xtr)
    record = {'seed': seed, 'epochs': []}
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed * 10000 + ep)
        perm = torch.randperm(n, generator=g).to(dev)
        net.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        masks = site_masks(net, xtr)
        with torch.no_grad():
            correct = sum((net(xte[i:i+2048]).argmax(1) == yte[i:i+2048]).sum().item()
                          for i in range(0, len(xte), 2048))
        record['epochs'].append({
            'epoch': ep,
            'acc': correct / len(xte),
            'dead': {s: masks[s].tolist() for s in SITES},
        })
        print(f"seed {seed} ep {ep}: acc {correct/len(xte):.4f} " +
              ' '.join(f"{s}:{masks[s].float().mean()*100:.0f}%" for s in
                       ['c1', 'c7', 'f0']), flush=True)
    # weight mass at end (attribution structure; weights move slowly):
    wm = weight_mass(net)
    record['weight_mass_final'] = {k: v.tolist() for k, v in wm.items()}
    out = Path(f'necrosis_pilot_s{seed}.json')
    out.write_text(json.dumps(record))
    print(f"seed {seed} -> {out} ({out.stat().st_size//1024} KB)", flush=True)


if __name__ == '__main__':
    DATA = load_cifar(dev)
    globals()['DATA'] = DATA
    for seed in range(3):
        run(seed)
    print('PILOT COMPLETE', flush=True)
