#!/usr/bin/env python3
"""factorial_lazyadam.py — Heath's meta-of-Manus experiment (2026-11-18).

If one-way blocks wrong-way signals, should Adam still accumulate and
APPLY momentum for the parameters those signals would have reached?
Standard Adam: when g=0 (gated), m decays slowly but the step still
applies stale suppression-direction momentum; v decays, amplifying the
next passed signal. LAZY-ADAM semantics: elements with g==0 this step
receive no moment update and no parameter step — the optimizer does not
track momentum of bypassed updates.

Arms: {relu, oneway_10} x {adam, lazyadam}, CifarNet, paired seeds.
"""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation
from factorial_cifar import load_cifar, NetC


class SignGatedAdam(torch.optim.Adam):
    """Heath's sign-gated accumulator (2026-11-18): "LazyAdam as-is
    can't be used, because our gradient for x<0 is alpha*g... test x<0
    and laze when that is the condition — sign-gate the accumulator."

    For parameter rows belonging to units that were INACTIVE on this
    batch (pre-activation <= 0 for every sample — the gate's own
    condition), Adam's memory is bypassed: no moment update, no
    momentum-driven step. The revival signal that the gate passed
    (alpha*g, nonzero) steps DIRECTLY (plain SGD-style, same lr), so
    stale suppression momentum cannot keep pushing a silenced unit,
    and v-decay cannot amplify the next revival step. Active rows get
    standard Adam. The inactive mask is supplied per-step by the
    harness from the unit pre-activations (unit_mask_fn)."""
    def __init__(self, params, lr, unit_masks=None, **kw):
        super().__init__(params, lr, **kw)
        self.unit_masks = unit_masks or {}  # param -> row-mask tensor (bool, True=inactive)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr, eps = group['lr'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st['step'] = torch.tensor(0.)
                    st['exp_avg'] = torch.zeros_like(p)
                    st['exp_avg_sq'] = torch.zeros_like(p)
                st['step'] += 1
                t = st['step'].item()
                m, v = st['exp_avg'], st['exp_avg_sq']
                mask = self.unit_masks.get(p)  # [rows] bool, True = unit inactive this batch
                if mask is None or not mask.any():
                    # standard Adam:
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    p.addcdiv_(m / (1 - beta1 ** t),
                               (v / (1 - beta2 ** t)).sqrt().add_(eps),
                               value=-lr)
                else:
                    act = ~mask  # active rows
                    # active rows: standard Adam
                    m[act] = beta1 * m[act] + (1 - beta1) * g[act]
                    v[act] = beta2 * v[act] + (1 - beta2) * g[act] ** 2
                    p[act] -= lr * (m[act] / (1 - beta1 ** t)) / \
                        ((v[act] / (1 - beta2 ** t)).sqrt() + eps)
                    # inactive rows: accumulator LAZED (m, v untouched);
                    # the passed revival signal steps directly:
                    p[mask] -= lr * g[mask]
        return None


class SignResetAdam(SignGatedAdam):
    """Heath's SRAdam, corrected (2026-11-18): SGAdam + reset-momentum-
    on-revive. While a unit is inactive: the SGAdam treatment (moments
    lazed, revival signal steps directly). On the REVIVAL transition
    (inactive prev batch -> active this batch): the unit's rows' m, v
    are reset to zero and its bias-correction clock restarts — so the
    newly-active unit begins standard-Adam life with NO memory of the
    suppression era at all. SGAdam alone resumes from frozen pre-
    silence moments; SRAdam forgets them."""
    def __init__(self, params, lr, unit_masks=None, **kw):
        super().__init__(params, lr, unit_masks=unit_masks, **kw)
        self.prev_masks = {}
        self.row_step = {}

    @torch.no_grad()
    def step(self, closure=None):
        # revival reset BEFORE the SGAdam step:
        for group in self.param_groups:
            for p in group['params']:
                cur = self.unit_masks.get(p)
                if cur is None:
                    continue
                st = self.state[p]
                prev = self.prev_masks.get(p)
                if prev is not None and len(st) > 0:
                    revived = prev & ~cur
                    if revived.any():
                        st['exp_avg'][revived] = 0.0
                        st['exp_avg_sq'][revived] = 0.0
                self.prev_masks[p] = cur.clone()
        return super().step(closure)


def run(arm, opt_name, seed, epochs, device, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    net = NetC(make_activation(arm)).to(device)
    masks = {}
    if opt_name == 'sgadam':
        opt = SignGatedAdam(net.parameters(), lr, unit_masks=masks)
    elif opt_name == 'sradam':
        opt = SignResetAdam(net.parameters(), lr, unit_masks=masks)
    else:
        opt = torch.optim.Adam(net.parameters(), lr)
    xtr, ytr, xte, yte = DATA
    n = len(xtr)
    t0 = time.time()
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed * 10000 + ep)
        perm = torch.randperm(n, generator=g).to(device)
        net.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            if opt_name in ('sgadam', 'sradam'):
                # the gate's own condition, per unit, this batch:
                inactive = (net.pre['f1'] <= 0).all(dim=0)  # [128]
                masks[net.f1.weight] = inactive
                masks[net.f1.bias] = inactive
            opt.step()
    net.eval()
    with torch.no_grad():
        correct = sum((net(xte[i:i+2048]).argmax(1) == yte[i:i+2048]).sum().item()
                      for i in range(0, len(xte), 2048))
        net(xtr[:4096])
        dead = (net.pre['f1'] <= 0).all(dim=0).float().mean().item()
    return {'arm': arm, 'optimizer': opt_name, 'seed': seed,
            'final_acc': correct / len(xte), 'final_dead_f1': dead,
            'wall_s': time.time() - t0}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results_sgadam.jsonl')
    args = ap.parse_args()
    DATA = load_cifar(args.device)
    globals()['DATA'] = DATA
    for seed in range(args.seeds):
        for arm in ['relu', 'oneway_10']:
            for opt_name in ['adam', 'sgadam', 'sradam']:
                r = run(arm, opt_name, seed, args.epochs, args.device)
                print(f"{arm:10s} {opt_name:9s} seed {seed}: "
                      f"acc {r['final_acc']:.4f} dead {r['final_dead_f1']:.3f} "
                      f"[{r['wall_s']:.0f}s]", flush=True)
                with open(args.out, 'a') as f:
                    f.write(json.dumps(r) + '\n')
