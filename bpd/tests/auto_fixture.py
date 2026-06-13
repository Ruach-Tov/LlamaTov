#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""auto_fixture.py — AUTOMATIC bit-identical measurement vs the LIFTED SPEC (Iyun, 2026-05-29, Heath).
The missing foundation: systematically measure every coordinate against the actual lifted llama.cpp
computation (llama-eval-callback per-op dump), populate 0-ULP fixtures, emit div_status_vs facts.
This is what fusion is GATED on (cannot auto-fuse until bit-identical fixturing across the chain).

Pipeline:
  1. run llama-eval-callback with LLAMA_DUMP_DIR -> per-op spec tensors + manifest.tsv
  2. parse manifest (idx, name, op, dtype, shape) + load each .bin (the spec reference)
  3. map ggml-op tensors -> Mistral coordinates (RMS_NORM->norm, MUL_MAT Qcur->q_proj, ROPE->rope...)
  4. run OUR op on the same input (from llamatov_run primitives) + ULP-compare to the spec tensor
  5. emit div_status_vs(Coord, lifted_llamacpp, measured(output(ulp(N)))) -> auto-populate fixtures

Usage: python3 auto_fixture.py --dump DIR --gguf BLOB [--out fixtures.o.pl]
"""
import sys, os, struct, argparse
import numpy as np

DTYPE = {0: ('<f4', np.float32), 1: ('<f2', np.float16), 2: ('<i4', np.int32)}

def read_bin(path):
    """Read a dumped tensor: dtype_code, n_dims, ne[4], nb[4], n_bytes, data."""
    with open(path, 'rb') as f:
        dtype_code = struct.unpack('<I', f.read(4))[0]
        n_dims = struct.unpack('<I', f.read(4))[0]
        ne = struct.unpack('<4q', f.read(32))
        nb = struct.unpack('<4Q', f.read(32))
        n_bytes = struct.unpack('<Q', f.read(8))[0]
        raw = f.read(n_bytes)
    np_dt = DTYPE.get(dtype_code, ('<f4', np.float32))[0]
    arr = np.frombuffer(raw, dtype=np_dt)
    # ne is ggml order [ne0,ne1,ne2,ne3] (fastest-first); reshape to the meaningful dims
    dims = [d for d in ne if d > 1] or [ne[0]]
    return arr.astype(np.float32), tuple(dims), dtype_code

def parse_manifest(dump_dir):
    """manifest.tsv: idx \t name \t op \t dtype \t shape. Return list of (idx, name, op, shape)."""
    rows = []
    mpath = os.path.join(dump_dir, 'manifest.tsv')
    with open(mpath) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 5:
                rows.append(dict(idx=parts[0], name=parts[1], op=parts[2], dtype=parts[3], shape=parts[4]))
    return rows

# map a (name, op) from the spec dump -> a Mistral coordinate path (layer 0)
def coord_for(name, op):
    n = name.lower()
    if op == 'RMS_NORM' and 'norm' in n:           return '[mistral, layer(l), attn, norm]'  # first norm = attn_norm
    if op == 'MUL_MAT' and 'qcur' in n:            return '[mistral, layer(l), attn, qkv]'   # q_proj
    if op == 'MUL_MAT' and 'kcur' in n:            return '[mistral, layer(l), attn, qkv]'   # k_proj
    if op == 'MUL_MAT' and 'vcur' in n:            return '[mistral, layer(l), attn, qkv]'   # v_proj
    if op == 'ROPE':                               return '[mistral, layer(l), attn, rope]'
    if op == 'GET_ROWS' and 'embd' in n:           return '[mistral, embed, token]'
    if op == 'SOFT_MAX':                           return '[mistral, layer(l), attn, score]'
    return None

def find_bin(dump_dir, idx, name):
    safe = name
    for c in '/ ()': safe = safe.replace(c, '_')
    p = os.path.join(dump_dir, f'{idx}_{safe}.bin')
    return p if os.path.exists(p) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True); ap.add_argument('--gguf', required=True)
    ap.add_argument('--out', default='div_fixtures.o.pl')
    a = ap.parse_args()
    rows = parse_manifest(a.dump)
    print(f"[manifest] {len(rows)} spec tensors dumped", file=sys.stderr)
    # for THIS first version: report what the spec dump gives us per coordinate (the reference set),
    # so we see the coverage the automatic measurement will populate. (Our-side op-run wires in next.)
    coverage = {}
    for r in rows:
        c = coord_for(r['name'], r['op'])
        if c:
            b = find_bin(a.dump, r['idx'], r['name'])
            if b:
                arr, dims, dt = read_bin(b)
                coverage.setdefault(c, []).append((r['op'], dims, dt, b))
    print(f"\n=== SPEC-REFERENCE COVERAGE (coordinates with a lifted-spec tensor available) ===")
    for c, ts in sorted(coverage.items()):
        ops = ','.join(sorted(set(t[0] for t in ts)))
        print(f"  {c}: {len(ts)} spec tensors [{ops}]")
    print(f"\n{len(coverage)} coordinates have a lifted-spec reference tensor (the automatic-measurement"
          f" substrate). Next: run OUR op on the same input + ULP-compare -> populate measured fixtures.")

if __name__ == '__main__':
    main()
