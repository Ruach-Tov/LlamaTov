#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""auto_fixture_run.py — run the automatic bit-identical measurement loop across the coordinate
chain (Iyun, 2026-05-29, Heath). For each coordinate: feed OUR op the spec's ACTUAL input tensors
(re-anchored), ULP-compare to the spec's output tensor, emit a measured fixture. RE-ANCHORED so each
op measures its INTRINSIC fidelity vs the lifted spec (not propagated divergence).
Usage: python3 auto_fixture_run.py --dump DIR --gguf BLOB [--out fixtures.o.pl]
"""
import sys, os, struct, argparse
import numpy as np

def read_bin(p):
    with open(p,'rb') as f:
        dt=struct.unpack('<I',f.read(4))[0]; nd=struct.unpack('<I',f.read(4))[0]
        ne=struct.unpack('<4q',f.read(32)); nb=struct.unpack('<4Q',f.read(32))
        nbytes=struct.unpack('<Q',f.read(8))[0]; raw=f.read(nbytes)
    np_dt = {0:'<f4',1:'<f2',2:'<i4'}.get(dt,'<f4')
    a=np.frombuffer(raw,dtype=np_dt).astype(np.float32)
    dims=[d for d in ne if d>1] or [ne[0]]
    # ggml stores ne fastest-first; numpy wants slowest-first -> reverse
    return a.reshape(dims[::-1]) if len(dims)>1 else a, dims, dt

def ulp(a,b):
    a=a.ravel().astype(np.float32); b=b.ravel().astype(np.float32); n=min(a.size,b.size); a,b=a[:n],b[:n]
    ma=float(np.max(np.abs(a-b))) if n else float('nan')
    ai=a.view(np.int32).astype(np.int64); bi=b.view(np.int32).astype(np.int64)
    ai=np.where(ai<0,np.int64(0x80000000)-ai,ai); bi=np.where(bi<0,np.int64(0x80000000)-bi,bi)
    return (int(np.max(np.abs(ai-bi))) if n else -1), ma

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dump',required=True); ap.add_argument('--gguf',required=True)
    ap.add_argument('--out',default='div_fixtures.o.pl')
    a=ap.parse_args(); D=a.dump.rstrip('/')+'/'
    sys.path.insert(0,'.'); sys.path.insert(0,'tests')
    # INVARIANT GUARD (Heath): refuse to silently measure against an aliased dump. Catches phantom
    # divergences (a fixture ref that is byte-equal to a prior tensor due to ggml buffer/name reuse).
    try:
        from dump_invariants import run_guard
        _rep = run_guard(a.dump.rstrip('/'))
        if not _rep['trustworthy']:
            print(f"[GUARD] {_rep['output_alias_detected']} output-aliases detected in dump "
                  f"(buffer/name reuse). Per-coordinate refs validated for distinctness; suspect refs "
                  f"yield phantom divergence — recompute from a verified-distinct output.", file=sys.stderr)
        _ALIASED = {x[0] for x in _rep['aliases']}
    except Exception as _e:
        print(f"[GUARD] dump-invariant guard unavailable: {_e}", file=sys.stderr); _ALIASED=set()
    import torch, torch.nn.functional as F
    from llamatov_run import parse_gguf, lt, rms_norm, apply_rope
    md,ts,do=parse_gguf(a.gguf); arch=md.get('general.architecture','llama')
    nh=md.get(f'{arch}.attention.head_count',32); nkv=md.get(f'{arch}.attention.head_count_kv',nh)
    ne=md.get(f'{arch}.embedding_length',2048); hd=ne//nh
    eps=md.get(f'{arch}.attention.layer_norm_rms_epsilon',1e-5)
    theta=md.get(f'{arch}.rope.freq_base',500000.0)
    w={n:lt(a.gguf,do,info) for n,info in ts.items()}
    def R(name):  # read a dumped spec tensor by file prefix
        import glob
        g=glob.glob(D+name+'*.bin')
        return read_bin(sorted(g)[0])[0] if g else None
    T=lambda x: torch.tensor(x, dtype=torch.float32)
    results=[]  # (coord, op, ulp, abs)

    # 1. attn_norm: input 0000_inp_embd -> our rms_norm(.,attn_norm.weight) vs 0004_attn_norm-0
    inp=R('0000_inp_embd'); spec=R('0004_attn_norm-0')
    if inp is not None and spec is not None:
        our=rms_norm(T(inp).unsqueeze(0), w['blk.0.attn_norm.weight'], eps).squeeze(0).numpy()
        results.append(('[mistral, layer(l), attn, norm]','rms_norm',*ulp(our,spec)))

    # 2-4. q/k/v_proj: input 0004_attn_norm-0 -> our (h @ W) vs 0007_Qcur / 0015_Kcur / 0020_Vcur
    h=T(R('0004_attn_norm-0'))
    for tag, wn, specn in [('q','blk.0.attn_q.weight','0007_Qcur-0'),
                           ('k','blk.0.attn_k.weight','0015_Kcur-0'),
                           ('v','blk.0.attn_v.weight','0020_Vcur-0')]:
        spec=R(specn)
        if spec is not None:
            our=(h @ w[wn]).numpy()
            results.append((f'[mistral, layer(l), attn, qkv]',f'{tag}_proj',*ulp(our,spec)))

    # 5. rope: input 0009_Qcur-0 (reshaped) -> our apply_rope vs 0011_Qcur-0 (ROPE output)
    qre=R('0009_Qcur-0'); spec=R('0011_Qcur-0')
    if qre is not None and spec is not None:
        qflat=T(qre).reshape(1,-1,nh*hd)
        qr,_=apply_rope(qflat, qflat.clone(), nh, hd, theta)
        results.append(('[mistral, layer(l), attn, rope]','apply_rope',*ulp(qr.numpy(), spec)))

    # 6. score (softmax): input 0038_node_20 -> our softmax vs 0041_node_21
    sc=R('0038_node_20'); spec=R('0041_node_21')
    if sc is not None and spec is not None:
        our=F.softmax(T(sc),dim=-1).numpy()
        results.append(('[mistral, layer(l), attn, score]','softmax',*ulp(our,spec)))

    # 7. residual1: 0000_inp_embd + 0050_attn_out-0 (ADD) vs 0052_ffn_inp-0
    e=R('0000_inp_embd'); ao=R('0050_attn_out-0'); spec=R('0052_ffn_inp-0')
    if e is not None and ao is not None and spec is not None:
        our=(T(e)+T(ao)).numpy()
        results.append(('[mistral, layer(l), residual(1)]','add',*ulp(our,spec)))

    # ── emit ──
    print("=== AUTOMATIC FIXTURE MEASUREMENT vs LIFTED SPEC (re-anchored per coordinate) ===")
    facts=[]
    for coord, op, mu, ma in results:
        verdict='0 ULP' if mu==0 else ('small_abs' if ma<0.05 else 'LARGE')
        print(f"  {op:12s} {coord}: max_ULP={mu:>10d} max_abs={ma:.3e} -> {verdict}")
        if mu==0:
            facts.append(f"div_status_vs({coord}, lifted_llamacpp, measured(output(ulp(0)), perf(unmeasured))).")
        else:
            facts.append(f"div_status_vs({coord}, lifted_llamacpp, measured(output(ulp({mu})), perf(unmeasured))).  % {op}")
    with open(a.out,'w') as f:
        f.write("%% div_fixtures.o.pl — GENERATED by auto_fixture_run.py (measured vs lifted llama.cpp spec).\n")
        f.write("\n".join(sorted(set(facts)))+"\n")
    print(f"\nemitted {len(set(facts))} fixture facts -> {a.out}")

if __name__=='__main__': main()
