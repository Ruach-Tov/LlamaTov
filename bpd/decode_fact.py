# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
#!/usr/bin/env python3
"""Fact-driven KV-cache decode (T=1 steps) — the architectural lever.
Mirrors llamatov_run.generate but routes rms_norm / linears / attention through
fact_dispatch. Prefill once, then incremental T=1 decode. Measures tok/s.

forward_pass() is the SINGLE shared loop body: decode() injects the fact-driven
int8 ops; decode_referee.py imports the SAME function and injects int8 or fp32.
Reference and subject cannot drift because they are literally one code path
(Iyun's structural point, msg ad93dcf0; refactor by Bocher, 37fc14abd follow-up).
"""
import os, sys, time
_BPD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BPD)
sys.path.insert(0, os.path.join(_BPD, "lib"))
import numpy as np, torch, torch.nn.functional as F
import llamatov_run as R
import fact_dispatch as fd


def q8lin(h, w, b=None):
    """The fact-driven int8 linear: Q8_0 dp4a GEMV emitted from a Prolog fact."""
    y = fd.q8_0_linear_from_fp32(h, w)
    if b is not None and not isinstance(b, int):
        y = y + b
    return y


def fp32lin(h, w, b=None):
    """The fp32 reference linear (torch matmul, no quantization)."""
    y = h @ w
    if b is not None and not isinstance(b, int):
        y = y + b
    return y


def load_model(path):
    """Parse GGUF, build cfg, load tensors. Shared by decode() and the referee."""
    md, ts, do = R.parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    cfg = {'n_layers': md.get(f'{arch}.block_count', 24),
           'n_head': md.get(f'{arch}.attention.head_count', 14),
           'n_head_kv': md.get(f'{arch}.attention.head_count_kv', md.get(f'{arch}.attention.head_count', 14)),
           'n_embd': md.get(f'{arch}.embedding_length', 896),
           'rope_theta': md.get(f'{arch}.rope.freq_base', 1000000.0),
           'norm_eps': md.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6),
           'arch': arch}
    w = {n: R.lt(path, do, info) for n, info in ts.items()}
    return cfg, w


def forward_pass(w, cfg, tok, positions, kv_cache, lin, rms):
    """ONE forward over `tok` (prefill: all ids, T=n; decode: [last], T=1).
    Mutates kv_cache (list[nl] of (k,v), each [1,nkv,T_total,hd]).
    lin(h, w, b=None) and rms(x, weight) are injected — int8 (q8lin +
    fd.rms_norm_fact) for the fact-driven subject, fp32 for the reference.
    Returns logits [1, Tcur, vocab]."""
    nh, nkv = cfg['n_head'], cfg['n_head_kv']
    hd = cfg['n_embd'] // nh
    nl = cfg['n_layers']
    emb = w['token_embd.weight']
    x = (emb.T[tok] if emb.shape[0] < emb.shape[1] else emb[tok]).unsqueeze(0)  # [1,Tcur,E]
    for il in range(nl):
        p = f'blk.{il}'
        h = rms(x, w[f'{p}.attn_norm.weight'])
        q_cur = lin(h, w[f'{p}.attn_q.weight'], w.get(f'{p}.attn_q.bias'))
        k_cur = lin(h, w[f'{p}.attn_k.weight'], w.get(f'{p}.attn_k.bias'))
        v_cur = lin(h, w[f'{p}.attn_v.weight'], w.get(f'{p}.attn_v.bias'))
        q_cur, k_cur = R.apply_rope(q_cur, k_cur, nh, hd, cfg['rope_theta'], positions=positions)
        B, Tc, _ = q_cur.shape
        q = q_cur.view(B, Tc, nh, hd).transpose(1, 2)
        k_new = k_cur.view(B, Tc, nkv, hd).transpose(1, 2)
        v_new = v_cur.view(B, Tc, nkv, hd).transpose(1, 2)
        if kv_cache[il] is not None:
            kc, vc = kv_cache[il]; k = torch.cat([kc, k_new], 2); v = torch.cat([vc, v_new], 2)
        else:
            k, v = k_new, v_new
        kv_cache[il] = (k, v)
        if nkv < nh:
            rep = nh // nkv; k_att = k.repeat_interleave(rep, 1); v_att = v.repeat_interleave(rep, 1)
        else:
            k_att, v_att = k, v
        T_total = k_att.shape[2]
        att = (q @ k_att.transpose(-2, -1)) * (hd ** -0.5)
        if T_total > 1 and Tc > 1:   # prefill causal mask
            m = torch.triu(torch.ones(Tc, T_total, dtype=torch.bool), diagonal=1 + (T_total - Tc))
            att = att.masked_fill(m.unsqueeze(0).unsqueeze(0), float('-inf'))
        y = F.softmax(att, dim=-1) @ v_att
        y = y.transpose(1, 2).contiguous().view(B, Tc, nh * hd)
        y = lin(y, w[f'{p}.attn_output.weight'])
        x = x + y
        h2 = rms(x, w[f'{p}.ffn_norm.weight'])
        gl = lin(h2, w[f'{p}.ffn_gate.weight']); gate = gl / (1.0 + torch.exp(-gl))
        up = lin(h2, w[f'{p}.ffn_up.weight'])
        x = x + lin(gate * up, w[f'{p}.ffn_down.weight'])
    # final norm + logits
    x_last = rms(x[:, -1:, :], w['output_norm.weight'])
    lm = w.get('output.weight', w.get('token_embd.weight'))
    logits = (x_last @ lm.T) if lm.shape[-1] == cfg['n_embd'] else (x_last @ lm)
    return logits


def step_tensors(generated, step):
    """The per-step (tok, positions) pair: prefill at step 0, T=1 decode after."""
    if step == 0:
        return (torch.tensor(generated, dtype=torch.long),
                torch.arange(len(generated)))
    return (torch.tensor([generated[-1]], dtype=torch.long),
            torch.tensor([len(generated) - 1]))


def decode(path, input_ids, n_tokens=10, mode="int8"):
    """The fact-driven KV-cache decode. mode: int8 (default, fact-driven) | fp32."""
    cfg, w = load_model(path)
    nl = cfg['n_layers']
    print(f"Arch {cfg['arch']} layers {nl} heads {cfg['n_head']}/{cfg['n_head_kv']} embd {cfg['n_embd']}", flush=True)
    t0 = time.time()
    print(f"Loaded {len(w)} tensors in {time.time()-t0:.1f}s", flush=True)

    lin = q8lin if mode == "int8" else fp32lin
    rms = (lambda x, wn: fd.rms_norm_fact(x, wn, cfg['norm_eps'])) if mode == "int8" \
          else (lambda x, wn: R.rms_norm(x, wn, cfg['norm_eps']))

    kv_cache = [None] * nl
    generated = list(input_ids)
    step_times = []
    for step in range(n_tokens):
        t_step = time.time()
        tok, positions = step_tensors(generated, step)
        logits = forward_pass(w, cfg, tok, positions, kv_cache, lin, rms)
        nxt = int(logits[0, -1].argmax().item())
        generated.append(nxt)
        dt = time.time() - t_step; step_times.append(dt)
        print(f"  step {step}: token {nxt}  ({dt:.2f}s, {'prefill' if step==0 else 'decode'})", flush=True)
    # tok/s on the decode steps (exclude prefill step 0)
    dec = step_times[1:]
    if dec:
        print(f"DECODE tok/s: {len(dec)/sum(dec):.2f} ({sum(dec)/len(dec)*1000:.0f}ms/token), prefill={step_times[0]:.2f}s", flush=True)
    print(f"GENERATED: {generated[len(input_ids):]}", flush=True)
    return generated


if __name__ == "__main__":
    decode(_os2.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf"), [1, 415, 6557], n_tokens=6)
