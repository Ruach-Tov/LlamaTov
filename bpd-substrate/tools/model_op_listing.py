#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
model_op_listing.py — render the ENTIRE model's op program from the lifted ggml
graph manifest, in the LCARS/FORTRAN tri-register style.

  - lowercase headers (the hi-tech "we don't shout" frame)
  - ALL-CAPS canonical ggml opcodes (the program mnemonics)
  - Mixed-Case English prose under "what it computes"
  - %04d two-column num (our sequence) + node (ggml id)
  - dtype column, with f16 nodes FLAGGED  (the f16-constriction map)

Usage:
  model_op_listing.py [manifest.tsv]                 # full listing
  model_op_listing.py manifest.tsv --f16-only        # only the f16 ops (the suspects)
  model_op_listing.py manifest.tsv --layer 0         # one layer's ops
  model_op_listing.py manifest.tsv --summary         # op-type + dtype tallies
  model_op_listing.py manifest.tsv --diff soa        # show a transform as a diff-in-context
  model_op_listing.py manifest.tsv --diff soa --context 2 --layer 0

Manifest TSV columns: node_id | tensor_name | op_type | dtype | shape | tap | src
"""
import sys, re, argparse

# ── Mixed-Case prose templates, produced from (op, tensor) — never stored prose ──
def base_tensor(raw):
    return re.sub(r"\s*\((reshaped|transposed|permuted|view|cont|copy of .*)\)", "", raw).strip()

def describe(op, name, dtype):
    bt = base_tensor(name) or name
    if op == "MUL_MAT":   return f"Matmul \u2192 {bt}"
    if op == "MUL":       return f"Elementwise multiply \u2192 {bt}"
    if op == "ADD":       return f"Residual add \u2192 {bt}"
    if op == "RMS_NORM":  return f"RMS-norm {bt}"
    if op == "ROPE":      return f"RoPE on {bt}"
    if op == "SOFT_MAX":  return f"Softmax({bt})"
    if op == "SILU":      return f"SiLU {bt}"
    if op == "RESHAPE":   return f"Reshape {bt}"
    if op == "VIEW":      return f"View into {bt}"
    if op == "PERMUTE":   return f"Permute {bt}"
    if op == "TRANSPOSE": return f"Transpose {bt}"
    if op == "CONT":      return f"Contiguous {bt}"
    if op == "CPY":
        m = re.search(r"copy of (.+)\)", name)
        if dtype == "f16": return f"f32 \u2192 f16 cast into {bt}"
        return f"Copy \u2192 {bt}"
    if op == "GET_ROWS":  return f"Embed lookup \u2192 {bt}"
    if op == "NONE":      return f"Leaf {bt}"
    return f"{op} \u2192 {bt}"

def layer_of(name):
    m = re.search(r"[-_]l?(\d+)\b", name) or re.search(r"-(\d+)$", name)
    return int(m.group(1)) if m else None

# ── Transform specs: which ops a layout transform touches, and how it's marked ──
# Each spec returns (affects(row)->bool, before_line(row), after_line(row), provenance).
# SoA: the Q8_0 weight-consuming MUL_MATs (gate/up/down + attn q/k/v/o projections).
TRANSFORMS = {
    "soa": {
        "title": "AoS \u2192 SoA Q8_0 weight layout",
        "affects": lambda r: r["op"] == "MUL_MAT",   # MUL_MATs read Q8_0 weights
        "before": "weight layout: array_of_structs  [d|qs0..31] x nb  (34B blk, straddles 128B line)",
        "after":  "weight layout: struct_of_arrays   quants[](32B-aligned) + scales[]  (coalesced)",
        "equiv":  "equiv_proof(soa_repack_0ulp, byte_permutation_value_preserving)  [bit-identical]",
        # provenance: measured verdict tagged with the test harness that emitted it.
        # If no harness fact is present, the verdict line is omitted (no hand-authored numbers).
        "verdict": None,   # set via --verdict-facts (mechanically-emitted), else omitted
    },
}

def load_verdict_facts(path):
    """Load mechanically-emitted layout_perf verdicts: node_id -> verdict string.
    Format (TSV): node_id \t test_harness \t cpu_speedup \t gpu_speedup \t max_ulp
    Only facts WITH a test_harness provenance are accepted."""
    out = {}
    if not path: return out
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5: continue
        node, test, cpu, gpu, ulp = p[0], p[1], p[2], p[3], p[4]
        out[node] = f"verdict: {cpu}x cpu / {gpu}x gpu, max_ulp={ulp}  [test({test})]"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?", default="<repo>/n64chk/manifest.tsv")
    ap.add_argument("--f16-only", action="store_true")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--diff", choices=sorted(TRANSFORMS), default=None,
                    help="show a layout transform as a diff, in op-program context")
    ap.add_argument("--context", type=int, default=2,
                    help="context ops shown around each changed op (--diff mode)")
    ap.add_argument("--verdict-facts", default=None,
                    help="TSV of mechanically-emitted verdicts (node\\ttest\\tcpu\\tgpu\\tulp); "
                         "only provenance-tagged facts are shown")
    a = ap.parse_args()

    rows = []
    for line in open(a.manifest):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5: continue
        rows.append(dict(node=p[0], name=p[1], op=p[2], dtype=p[3], shape=p[4]))

    if a.summary:
        from collections import Counter
        print("\n  model op program \u2014 summary")
        print("  " + "\u2501"*50)
        print(f"  total ops: {len(rows)}")
        print("\n  by opcode:")
        for op, c in Counter(r["op"] for r in rows).most_common():
            print(f"    {op:<12} {c:>5}")
        print("\n  by dtype:")
        for dt, c in Counter(r["dtype"] for r in rows).most_common():
            flag = "  \u2190 f16 constriction" if dt == "f16" else ""
            print(f"    {dt:<6} {c:>5}{flag}")
        print(f"\n  f16 ops by opcode (the constriction map):")
        for op, c in Counter(r["op"] for r in rows if r["dtype"]=="f16").most_common():
            print(f"    {op:<12} {c:>5}")
        return

    sel = rows
    if a.f16_only:
        sel = [r for r in rows if r["dtype"] == "f16"]
    if a.layer is not None:
        sel = [r for r in sel if layer_of(r["name"]) == a.layer]

    def row_line(i, r):
        num = f"{i:04d}"; op = r["op"]; dt = r["dtype"]
        dtflag = dt + ("*" if dt == "f16" else " ")
        return (f"{num:<6}{r['node']:<6}{op:<12}{dtflag:<6}{r['shape'][:15]:<16}"
                f"{r['name'][:33]:<34}{describe(op, r['name'], dt)}")

    # ── DIFF MODE: show the transform inline, in op-program context ──
    if a.diff:
        spec = TRANSFORMS[a.diff]
        verdicts = load_verdict_facts(a.verdict_facts)
        affects = spec["affects"]
        changed = [i for i, r in enumerate(sel) if affects(r)]
        changed_set = set(changed)
        # rows to show: each changed op +/- context window
        show = set()
        for i in changed:
            for j in range(max(0, i-a.context), min(len(sel), i+a.context+1)):
                show.add(j)
        print(f"\n  model op program \u2014 diff: {spec['title']}"
              + (f"  \u2014 layer {a.layer}" if a.layer is not None else ""))
        print("  (\u00b1 marks the transformed ops; surrounding context unchanged)")
        hdr = f"    {'num':<6}{'node':<6}{'ggml-op':<12}{'dtype':<6}{'shape':<16}{'tensors':<34}what it computes"
        print(hdr); print("  " + "\u2501" * (len(hdr)-2))
        prev = -1
        for j in sorted(show):
            if prev >= 0 and j != prev + 1:
                print("    \u22ee")                       # elision marker between context blocks
            r = sel[j]; line = row_line(j+1, r)
            if j in changed_set:
                # the changed op: show -old/+new layout, then the op line marked
                print(f"  - {line}")
                print(f"  + {line}")
                print(f"      \u2502 - {spec['before']}")
                print(f"      \u2502 + {spec['after']}")
                print(f"      \u2502   {spec['equiv']}")
                v = verdicts.get(r["node"])
                if v:
                    print(f"      \u2502   {v}")
                else:
                    print(f"      \u2502   (no provenance-tagged verdict fact for node {r['node']}; "
                          f"supply --verdict-facts)")
            else:
                print(f"    {line}")                     # context op, unchanged
            prev = j
        print(f"\n  {len(changed)} ops transformed (of {len(sel)} shown-context / "
              f"{len([r for r in rows if affects(r)])} total affected)")
        return

    title = "model op program"
    if a.f16_only: title += "  \u2014 f16 ops only (constriction map)"
    if a.layer is not None: title += f"  \u2014 layer {a.layer}"
    print(f"\n  {title}")
    hdr = f"  {'num':<6}{'node':<6}{'ggml-op':<12}{'dtype':<6}{'shape':<16}{'tensors':<34}what it computes"
    print(hdr)
    print("  " + "\u2501" * (len(hdr)-2))
    for i, r in enumerate(sel, 1):
        print("  " + row_line(i, r))
    print(f"\n  {len(sel)} ops"
          + ("  (f16 = the precision-constricted nodes; * marks f16)" if not a.f16_only else ""))

if __name__ == "__main__":
    main()
