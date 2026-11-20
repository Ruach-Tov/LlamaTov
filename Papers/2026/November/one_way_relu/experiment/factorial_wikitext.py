#!/usr/bin/env python3
"""factorial_wikitext.py — WikiText-2 transformer-LM arm of the one-way
factorial (paired seeds).

2-layer decoder-only transformer: d_model=128, 4 heads, FFN 512 with the
ARM activation (the experiment's site — ReLU in max(0,xW1)W2). Word-level
WT2, tied embeddings, seq_len 128. Metric: validation perplexity.
Squelch metric: fraction of layer-0 FFN channels never active over the
validation set.

Usage: python3 factorial_wikitext.py --arms relu leaky_10 oneway_10 \
         --seeds 5 --epochs 8 --device cuda [--optimizer adam|sgd]
"""
import argparse, json, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from activations import make_activation

DATA = Path(__file__).parent / 'data_wikitext2'
LAYERNORM = True


def load_wikitext(device, seq_len=128):
    import pyarrow.parquet as pq
    def text_of(fname):
        t = pq.read_table(DATA / fname)
        return '\n'.join(x for x in t.to_pydict()['text'] if x.strip())
    train_txt = text_of('train-00000-of-00001.parquet')
    val_txt = text_of('validation-00000-of-00001.parquet')
    # word-level vocab from train:
    words = train_txt.split()
    from collections import Counter
    counts = Counter(words)
    vocab = ['<unk>'] + [w for w, c in counts.most_common() if c >= 3]
    stoi = {w: i for i, w in enumerate(vocab)}
    def encode(txt):
        return torch.tensor([stoi.get(w, 0) for w in txt.split()],
                            dtype=torch.long)
    tr, va = encode(train_txt), encode(val_txt)
    def batchify(d, bsz):
        nb = len(d) // (bsz * seq_len)
        d = d[:nb * bsz * seq_len].reshape(bsz, -1)
        return d
    return (batchify(tr, 32).to(device), batchify(va, 32).to(device),
            len(vocab), seq_len)


class Block(nn.Module):
    def __init__(self, d, heads, ffn, act, layernorm=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        LN = nn.LayerNorm if layernorm else (lambda _: nn.Identity())
        self.ln1 = LN(d)
        self.ln2 = LN(d)
        self.w1 = nn.Linear(d, ffn)
        self.w2 = nn.Linear(ffn, d)
        self.act = act
        self.pre = None

    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=mask, need_weights=False)
        x = x + a
        pre = self.w1(self.ln2(x))
        self.pre = pre.detach()
        x = x + self.w2(self.act(pre))
        return x


class LM(nn.Module):
    def __init__(self, vocab, act, d=128, heads=4, ffn=512, layers=2,
                 seq_len=128, layernorm=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList(
            [Block(d, heads, ffn, act, layernorm) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d) if layernorm else nn.Identity()
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.full((T, T), float('-inf'),
                                     device=idx.device), diagonal=1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln_f(x))


def evaluate(net, va, seq_len):
    net.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, va.shape[1] - 1, seq_len):
            x = va[:, i:i + seq_len]
            y = va[:, i + 1:i + 1 + seq_len]
            if y.shape[1] < x.shape[1]:
                x = x[:, :y.shape[1]]
            logits = net(x)
            tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction='sum').item()
            n += y.numel()
    return math.exp(tot / n)


def dead_ffn(net, va, seq_len):
    """Fraction of layer-0 FFN channels with pre<=0 on ALL val positions."""
    alive = None
    net.eval()
    with torch.no_grad():
        for i in range(0, min(va.shape[1] - 1, 20 * seq_len), seq_len):
            x = va[:, i:i + seq_len]
            net(x)
            pre = net.blocks[0].pre  # [B, T, ffn]
            a = (pre > 0).any(dim=(0, 1))
            alive = a if alive is None else (alive | a)
    return 1.0 - alive.float().mean().item()


def run(arm, seed, epochs, device, optimizer='adam'):
    torch.manual_seed(seed)
    tr, va, vocab, seq_len = DATA_CACHE
    net = LM(vocab, make_activation(arm), seq_len=seq_len,
             layernorm=LAYERNORM).to(device)
    if optimizer == 'adam':
        opt = torch.optim.Adam(net.parameters(), 1e-3)
    else:
        opt = torch.optim.SGD(net.parameters(), 1e-2, momentum=0.9)
    t0 = time.time()
    history = []
    nbatch = (tr.shape[1] - 1) // seq_len
    for ep in range(epochs):
        g = torch.Generator().manual_seed(seed * 10000 + ep)
        order = torch.randperm(nbatch, generator=g)
        net.train()
        for j in order.tolist():
            i = j * seq_len
            x = tr[:, i:i + seq_len]
            y = tr[:, i + 1:i + 1 + seq_len]
            if y.shape[1] < x.shape[1]:
                x = x[:, :y.shape[1]]
            opt.zero_grad()
            logits = net(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        ppl = evaluate(net, va, seq_len)
        dead = dead_ffn(net, va, seq_len)
        history.append({'epoch': ep, 'val_ppl': ppl, 'dead_ffn0': dead})
        if math.isnan(ppl) or math.isinf(ppl) or ppl > 1e6:
            break  # diverged — record and stop this run
    return {'arm': arm, 'seed': seed, 'epochs': epochs,
            'dataset': 'wikitext2', 'optimizer': optimizer,
            'wall_s': time.time() - t0, 'final_ppl': history[-1]['val_ppl'],
            'history': history}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    ap.add_argument('--out', default='results_wikitext.jsonl')
    ap.add_argument('--no-layernorm', action='store_true',
                    help='RENEGADE: replace all LayerNorms with Identity')
    args = ap.parse_args()
    globals()['LAYERNORM'] = not args.no_layernorm
    if args.no_layernorm:
        print('RENEGADE MODE: LayerNorm -> Identity everywhere', flush=True)
    DATA_CACHE = load_wikitext(args.device)
    globals()['DATA_CACHE'] = DATA_CACHE
    print(f'vocab={DATA_CACHE[2]}, train_tokens~{DATA_CACHE[0].numel()}',
          flush=True)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for arm in args.arms:
            r = run(arm, seed, args.epochs, args.device,
                    optimizer=args.optimizer)
            print(f"{arm:10s} seed {seed}: ppl {r['final_ppl']:8.2f} "
                  f"dead(ffn0) {r['history'][-1]['dead_ffn0']:.3f} "
                  f"[{r['wall_s']:.0f}s]", flush=True)
            if args.out != '/dev/null':
                with open(args.out, 'a') as f:
                    f.write(json.dumps(r) + '\n')
