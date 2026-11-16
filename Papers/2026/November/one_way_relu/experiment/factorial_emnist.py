"""Five-arm one-way-leaky factorial on EMNIST-Balanced (47 classes, 131k).

Same design as factorial.py (relu doc section 7.4) on a task with real
accuracy headroom: typical small-CNN ceiling ~87-90% (vs MNIST's saturated
99%+), so arm differences can show in ACCURACY, not just dead-unit census.

EMNIST gotcha handled: images are stored transposed relative to MNIST
orientation; we transpose each image on load (harmless for learning either
way, but keeps human-inspectable renders sane and matches published
baselines).
"""
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation, ARMS
from factorial import Net, dead_stats, _read_idx

DATA = Path(__file__).parent / 'data_emnist'
N_CLASSES = 47


def _read_idx_dir(name):
    import gzip, struct
    import numpy as np
    with gzip.open(DATA / (name + '.gz'), 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
    return torch.from_numpy(data.copy())


def load_emnist(device):
    xtr = _read_idx_dir('emnist-balanced-train-images-idx3-ubyte').float()
    ytr = _read_idx_dir('emnist-balanced-train-labels-idx1-ubyte').long()
    xte = _read_idx_dir('emnist-balanced-test-images-idx3-ubyte').float()
    yte = _read_idx_dir('emnist-balanced-test-labels-idx1-ubyte').long()
    # EMNIST orientation fix: transpose H/W
    xtr = xtr.transpose(1, 2).unsqueeze(1) / 255.0
    xte = xte.transpose(1, 2).unsqueeze(1) / 255.0
    return xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)


class NetE(Net):
    def __init__(self, act):
        super().__init__(act)
        self.f2 = nn.Linear(128, N_CLASSES)


def run(arm, seed, epochs, device, batch=128, lr=1e-3, ckpt_dir='', optimizer='adam', amp_log=False):
    torch.manual_seed(seed)
    net = NetE(make_activation(arm)).to(device)
    if optimizer == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr)
    elif optimizer == 'sgd':
        # momentum-SGD, the honest classical baseline; lr tuned 10x Adam's
        # (standard practice for SGD on small CNNs)
        opt = torch.optim.SGD(net.parameters(), lr * 10, momentum=0.9)
    else:
        raise ValueError(optimizer)
    xtr, ytr, xte, yte = load_emnist(device)
    n = len(xtr)
    census_x = xtr[:512]
    history = []
    amp_records = []
    prev_dead = None
    revivals = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for bi, i in enumerate(range(0, n, batch)):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(net(xtr[idx]), ytr[idx])
            loss.backward()
            if amp_log and bi % 200 == 0:
                # Adam-amplification measurement (Heath, 2026-11-16):
                # for the f1 weight matrix, rows whose unit was DEAD this
                # batch receive only revival-channel gradient (alpha-scaled).
                # Record mean |g| on those rows now, then after opt.step()
                # measure the actual |delta w| — the ratio (|dw|/lr)/|g| is
                # Adam's amplification factor A at this alpha. Under SGD,
                # A should be ~1 (modulo momentum); under Adam, ~1/|g|-ish.
                with torch.no_grad():
                    dead_rows = (net.pre['f1'] <= 0).all(dim=0)
                    g = net.f1.weight.grad
                    if dead_rows.any():
                        g_dead = float(g[dead_rows].abs().mean())
                        w_before = net.f1.weight[dead_rows].clone()
                    else:
                        g_dead = 0.0
            opt.step()
            if amp_log and bi % 200 == 0 and g_dead > 0:
                with torch.no_grad():
                    dw = float((net.f1.weight[dead_rows] - w_before).abs().mean())
                    eff_lr = lr * (10 if optimizer == 'sgd' else 1)
                    amp = (dw / eff_lr) / g_dead if g_dead > 0 else 0.0
                    amp_records.append({'epoch': ep, 'batch': bi,
                                        'g_dead': g_dead, 'dw': dw,
                                        'amp': amp})
        with torch.no_grad():
            ds = dead_stats(net, census_x)
            z = net.pre['f1']
            dead_now = (z <= 0).all(dim=0)
            if prev_dead is not None:
                revivals += int((prev_dead & ~dead_now).sum())
            prev_dead = dead_now
            # batched test eval (18.8k test imgs):
            correct = 0
            for j in range(0, len(xte), 1024):
                correct += int((net(xte[j:j+1024]).argmax(1) == yte[j:j+1024]).sum())
            acc = correct / len(xte)
        history.append({'epoch': ep, 'dead': ds, 'acc': acc, 'loss': float(loss)})
    if ckpt_dir:
        Path(ckpt_dir).mkdir(exist_ok=True)
        torch.save(net.state_dict(),
                   Path(ckpt_dir) / f'{arm}_s{seed}_e{epochs}.pt')
    return {'arm': arm, 'seed': seed, 'epochs': epochs, 'dataset': 'emnist-balanced',
            'optimizer': optimizer,
            'wall_s': time.time() - t0, 'revivals_f1': revivals,
            'final_acc': history[-1]['acc'], 'history': history,
            'amp_records': amp_records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='*', default=ARMS)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results_emnist.jsonl')
    ap.add_argument('--save-ckpt', default='', help='dir to save final model state per run')
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    ap.add_argument('--amp-log', action='store_true',
                    help='log Adam-amplification factor on dead-row f1 grads')
    args = ap.parse_args()
    outp = Path(args.out)
    for arm in args.arms:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            r = run(arm, seed, args.epochs, args.device,
                    ckpt_dir=args.save_ckpt, optimizer=args.optimizer,
                    amp_log=args.amp_log)
            with outp.open('a') as f:
                f.write(json.dumps(r) + '\n')
            print(f"{arm:10s} seed {seed}: acc {r['final_acc']:.4f} "
                  f"revivals(f1) {r['revivals_f1']:3d} "
                  f"dead(f1) {r['history'][-1]['dead']['f1']:.3f} "
                  f"[{r['wall_s']:.0f}s]", flush=True)


if __name__ == '__main__':
    main()
