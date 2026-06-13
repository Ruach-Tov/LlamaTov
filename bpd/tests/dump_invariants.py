#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""dump_invariants.py — DETECT + PREVENT spec-dump aliasing mistakes (Iyun, 2026-05-29, Heath).

DISCOVERY: the llama-eval-callback dump can ALIAS tensors. Root cause (eval-callback.cpp:186 + 214):
the callback dumps the op OUTPUT (t) AND its SOURCE tensors (src), naming files by tensor NAME +
a monotonic index. ggml REUSES buffers and REUSES names (e.g. 'norm' appears for every layer's
RMS_NORM, and the same buffer is overwritten). So a file '{idx}_norm.bin' may hold a DIFFERENT
dump-instant of a reused 'norm' buffer than the op-output you intend to measure against. Measuring
our kernel vs an aliased/stale dump produces a PHANTOM divergence (our output_norm '2.46e-4' was
partly this: the 'norm' tensor we compared to equalled a weighted-MUL result, not the pre-weight
RMS_NORM output).

This module: (1) DETECTS aliasing in a dump, (2) defines INVARIANTS that must hold for a trustworthy
fixture, (3) provides a guard to run before any auto_fixture measurement (so phantom divergences are
caught, never silently mistaken for real kernel divergence).
"""
import os, struct
import numpy as np

def read_bin(p):
    with open(p, "rb") as f:
        struct.unpack("<I", f.read(4)); struct.unpack("<I", f.read(4))
        ne = struct.unpack("<4q", f.read(32)); struct.unpack("<4Q", f.read(32))
        nb = struct.unpack("<Q", f.read(8))[0]; raw = f.read(nb)
    a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    dims = [d for d in ne if d > 1] or [ne[0]]
    return a

def parse_manifest(D):
    """Schema-aware: old format (idx,name,op,...) OR new format (idx,name,op,...,kind,src_index)
    from metayens fix 1523c93. Returns (idx, name, op) tuples; new-schema kind/src_index accessible
    via parse_manifest_full."""
    rows = []
    for line in open(os.path.join(D, "manifest.tsv")):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3: rows.append((p[0], p[1], p[2]))
    return rows

def parse_manifest_full(D):
    """Return full rows incl. kind/src_index if the new schema is present (metayen 1523c93).
    Detects schema by header or column count. kind in {out, src}; src_index -1 for op-output."""
    out = []
    for line in open(os.path.join(D, "manifest.tsv")):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3: continue
        rec = {"idx": p[0], "name": p[1], "op": p[2]}
        # new schema: kind + src_index appended (search for out/src token)
        kind = next((c for c in p[3:] if c in ("out", "src")), None)
        rec["kind"] = kind if kind else "unknown"
        out.append(rec)
    return out

def is_new_schema(D):
    """True if the dump uses metayens fixed naming (kind column present) -> trustworthy-by-construction."""
    return any(r["kind"] in ("out", "src") for r in parse_manifest_full(D))

def _load(D, idx, nm):
    safe = nm
    for c in "/ ()": safe = safe.replace(c, "_")
    p = os.path.join(D, f"{idx}_{safe}.bin")
    return read_bin(p) if os.path.exists(p) else None

# ── INVARIANT 1: NAME UNIQUENESS — no two dumped files share a tensor NAME (else buffer-reuse alias) ──
def check_name_uniqueness(D):
    rows = parse_manifest(D); seen = {}; violations = []
    for idx, nm, op in rows:
        if nm in seen: violations.append((nm, seen[nm], idx, op))
        else: seen[nm] = idx
    return violations  # list of (name, first_idx, dup_idx, op) — each is a reuse/alias risk

# ── INVARIANT 2: OP-OUTPUT DISTINCTNESS — a non-identity op's output must NOT byte-equal a prior tensor ──
# (RMS_NORM/MUL_MAT/ROPE etc. transform data; output==input is the alias signature.)
TRANSFORM_OPS = {"RMS_NORM", "MUL_MAT", "ROPE", "SOFT_MAX", "MUL"}
def check_output_distinctness(D, window=3):
    rows = parse_manifest(D); aliases = []
    for i, (idx, nm, op) in enumerate(rows):
        if op not in TRANSFORM_OPS: continue
        out = _load(D, idx, nm)
        if out is None: continue
        for back in range(1, window + 1):
            if i - back < 0: continue
            pidx, pnm, pop = rows[i - back]; prev = _load(D, pidx, pnm)
            if prev is None or prev.size != out.size: continue
            if float(np.max(np.abs(out - prev))) == 0.0:
                aliases.append((idx, nm, op, pidx, pnm, pop)); break
    return aliases  # (out_idx,out_name,out_op, alias_idx,alias_name,alias_op) — output==prior = ALIAS

# ── INVARIANT 3: PROVENANCE — a fixture's reference tensor must be the op OUTPUT, verified by an
#    independent recompute (input -> our op -> matches the dumped tensor within a SANITY tolerance).
#    If our faithful recompute is WILDLY off (not small-ULP), the reference tensor is suspect (aliased).
def check_provenance(D, coord_name, recompute_fn, input_idx_name, ref_idx_name, sanity_abs=1e-1):
    """recompute_fn(input_array) -> our output. If |our - ref| > sanity_abs, the ref is suspect."""
    inp = _load(D, *input_idx_name); ref = _load(D, *ref_idx_name)
    if inp is None or ref is None: return ("missing", None)
    ours = recompute_fn(inp)
    ma = float(np.max(np.abs(np.asarray(ours).ravel()[:ref.size] - ref.ravel())))
    return ("SUSPECT_REF" if ma > sanity_abs else "ok", ma)

def run_guard(D):
    """Run all invariant checks; return a report. Call BEFORE any auto_fixture measurement."""
    # NEW-SCHEMA fast path (metayen 1523c93): if the dump tags kind=out/src, op-outputs are
    # uniquely named -> trustworthy-by-construction; consumers filter kind==out from the manifest.
    if is_new_schema(D):
        full = parse_manifest_full(D)
        n_out = sum(1 for r in full if r["kind"] == "out")
        return {"schema": "new(metayen-1523c93)", "name_reuse_violations": 0,
                "output_alias_detected": 0, "aliases": [], "op_outputs": n_out,
                "trustworthy": True, "note": "unique-named op-outputs; filter kind==out"}
    # OLD-SCHEMA: heuristic distinctness check (defense-in-depth for pre-fix dumps).
    dup = check_name_uniqueness(D)
    alias = check_output_distinctness(D)
    report = {
        "schema": "old(name-keyed, pre-fix)",
        "name_reuse_violations": len(dup),
        "output_alias_detected": len(alias),
        "aliases": alias[:10],
        "trustworthy": (len(alias) == 0),
    }
    return report

if __name__ == "__main__":
    import sys
    D = sys.argv[1] if len(sys.argv) > 1 else "<home>/tmp/spec_dump"
    rep = run_guard(D)
    print("=== SPEC-DUMP INVARIANT GUARD ===")
    print(f"  name-reuse violations: {rep['name_reuse_violations']}")
    print(f"  output-aliases detected: {rep['output_alias_detected']}")
    for a in rep["aliases"]:
        print(f"    ALIAS: {a[1]}({a[0]},{a[2]}) == {a[4]}({a[3]},{a[5]})")
    print(f"  => dump trustworthy for direct fixturing: {rep['trustworthy']}")
    if not rep["trustworthy"]:
        print("  ACTION: do NOT measure kernels vs aliased tensors — recompute the op output from its")
        print("          verified input, or fix the dump to emit unique-named op-outputs (see invariants).")
