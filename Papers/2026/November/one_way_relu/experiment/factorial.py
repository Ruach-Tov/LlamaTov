"""Five-arm one-way-leaky factorial on MNIST (relu doc section 7.4).

Measures per arm x seed: dead-unit fraction over epochs, revival events,
revive/suppress error-sign ratio at dead units (the measure-not-assume
counter), test accuracy, loss. Device-agnostic: --device cpu|cuda.

Usage:
  python factorial.py --arms relu --seeds 1 --epochs 2      # smoke
  python factorial.py --epochs 10 --seeds 5                 # full (P4: ~30min)
"""
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation, ARMS


class Net(nn.Module):
    """conv32-conv64-fc128-fc10; activation injected per arm."""
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.f1 = nn.Linear(64 * 12 * 12, 128)
        self.f2 = nn.Linear(128, 10)
        # pre-activation stash for dead-unit accounting:
        self.pre = {}

    def forward(self, x):
        z1 = self.c1(x); self.pre['c1'] = z1.detach()
        x = self.act(z1)
        z2 = self.c2(x); self.pre['c2'] = z2.detach()
        x = F.max_pool2d(self.act(z2), 2).flatten(1)
        z3 = self.f1(x); self.pre['f1'] = z3.detach()
        return self.f2(self.act(z3))


DATA_DIR = Path(__file__).parent / 'data'


def _read_idx(name):
    """Parse IDX format directly (no torchvision dependency)."""
    import gzip, struct
    import numpy as np
    with gzip.open(DATA_DIR / (name + '.gz'), 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
    return torch.from_numpy(data.copy())


def load_mnist(device):
    """MNIST from local IDX files (data/ dir, fetched once by setup)."""
    xtr = _read_idx('train-images-idx3-ubyte').float().unsqueeze(1) / 255.0
    ytr = _read_idx('train-labels-idx1-ubyte').long()
    xte = _read_idx('t10k-images-idx3-ubyte').float().unsqueeze(1) / 255.0
    yte = _read_idx('t10k-labels-idx1-ubyte').long()
    return (xtr.to(device), ytr.to(device), xte.to(device), yte.to(device))


def dead_stats(net, xsample):
    """Fraction of units with pre-activation <= 0 for EVERY sample in batch
    (the dying-ReLU census), per layer."""
    net(xsample)
    out = {}
    for k, z in net.pre.items():
        flat = z.transpose(0, 1).reshape(z.shape[1], -1) if z.dim() == 4 else z.t()
        dead = (flat <= 0).all(dim=1)
        out[k] = float(dead.float().mean())
    return out


def run(arm, seed, epochs, device, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    act = make_activation(arm)
    net = Net(act).to(device)
    opt = torch.optim.Adam(net.parameters(), lr)
    xtr, ytr, xte, yte = load_mnist(device)
    n = len(xtr)
    census_x = xtr[:512]
    history = []
    prev_dead = None
    revivals = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            ds = dead_stats(net, census_x)
            # revival accounting on f1 (per-unit dead flags):
            z = net.pre['f1']
            dead_now = (z <= 0).all(dim=0)
            if prev_dead is not None:
                revivals += int((prev_dead & ~dead_now).sum())
            prev_dead = dead_now
            acc = (net(xte).argmax(1) == yte).float().mean().item()
        history.append({'epoch': ep, 'dead': ds, 'acc': acc,
                        'loss': float(loss)})
    return {'arm': arm, 'seed': seed, 'epochs': epochs,
            'wall_s': time.time() - t0, 'revivals_f1': revivals,
            'final_acc': history[-1]['acc'], 'history': history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='*', default=ARMS)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='results.jsonl')
    args = ap.parse_args()
    torch.set_num_threads(6)
    outp = Path(args.out)
    for arm in args.arms:
        for seed in range(args.seeds):
            r = run(arm, seed, args.epochs, args.device)
            with outp.open('a') as f:
                f.write(json.dumps(r) + '\n')
            print(f"{arm:10s} seed {seed}: acc {r['final_acc']:.4f} "
                  f"revivals(f1) {r['revivals_f1']:3d} "
                  f"dead(f1) {r['history'][-1]['dead']['f1']:.3f} "
                  f"[{r['wall_s']:.0f}s]", flush=True)


if __name__ == '__main__':
    main()
