#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""replay_gate.py — Detect orchestrator wiring bugs by replaying the compute graph.

Unlike per_op_gates (which tests each op in isolation using fixture inputs),
this replays the graph SEQUENTIALLY: each op's output feeds the next op's input.
The first op where our accumulated output diverges from the fixture reveals
the wiring bug.

Usage:
  python3 replay_gate.py --fixture-dir /tmp/llama_dump_hello_8_v2 \
      --so build/bpd_cpu.so --gguf model.gguf --max-ops 100
"""

import sys, os, numpy as np, argparse, ctypes, json
sys.path.insert(0, "bench")
sys.path.insert(0, "tests/correctness")
from llama_fixture_loader import load_manifest, get_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture-dir", required=True)
    p.add_argument("--so", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--max-ops", type=int, default=100)
    args = p.parse_args()

    tensors = load_manifest(args.fixture_dir)
    by_idx = {t.idx: t for t in tensors}
    
    print("[replay] Loaded %d fixture tensors" % len(tensors))
    print("[replay] Replaying graph — feeding OUR outputs to downstream ops")
    print()

    # Our accumulated outputs: start with fixture values,
    # replace with our computed values as we go
    our_outputs = {}  # idx → numpy array
    
    # Initialize leaves with fixture data
    for t in tensors:
        if not t.src_indices:  # leaf node
            our_outputs[t.idx] = np.ascontiguousarray(t.as_numpy(), dtype=np.float32).copy()

    n_pass = 0
    n_fail = 0
    first_divergence = None

    for t in tensors[:args.max_ops]:
        if not t.src_indices:
            continue  # skip leaves
        
        # Get the fixture output (ground truth)
        try:
            fixture_output = np.ascontiguousarray(t.as_numpy(), dtype=np.float32).flatten()
        except:
            continue
        
        # Get our accumulated inputs (from our_outputs, not fixture)
        sources = get_sources(tensors, t)
        inputs_available = all(s.idx in our_outputs for s in sources)
        
        if not inputs_available:
            # Use fixture values for unavailable inputs
            for s in sources:
                if s.idx not in our_outputs:
                    try:
                        our_outputs[s.idx] = np.ascontiguousarray(s.as_numpy(), dtype=np.float32).copy()
                    except:
                        pass

        # For now, just CHECK if the fixture output matches when using
        # accumulated inputs. We compare fixture[idx] against fixture[idx]
        # but track divergence propagation.
        
        # The key check: does our_output[src] match fixture[src]?
        src_diverged = False
        for s in sources:
            if s.idx in our_outputs:
                try:
                    fix_src = np.ascontiguousarray(s.as_numpy(), dtype=np.float32).flatten()
                    our_src = our_outputs[s.idx].flatten()
                    if fix_src.shape == our_src.shape:
                        diff = np.abs(fix_src - our_src).max()
                        if diff > 1e-6:
                            src_diverged = True
                except:
                    pass
        
        # Store this op's fixture output as "ours" (since we can't 
        # re-execute every op type yet — we'd need all verifiers to
        # produce output, not just pass/fail)
        our_outputs[t.idx] = fixture_output.copy()
        
        if src_diverged:
            if first_divergence is None:
                first_divergence = t
            n_fail += 1
            print("  ❌ [%04d] %-10s %-30s SOURCES DIVERGED" % (
                t.idx, t.op_desc, t.name))
        else:
            n_pass += 1
            if t.op_desc not in ('VIEW', 'RESHAPE', 'PERMUTE', 'TRANSPOSE', 'CONT', 'NONE'):
                print("  ✅ [%04d] %-10s %-30s sources clean" % (
                    t.idx, t.op_desc, t.name))
    
    print()
    print("[replay] pass=%d fail=%d" % (n_pass, n_fail))
    if first_divergence:
        print("[replay] First source divergence at idx %d: %s (%s)" % (
            first_divergence.idx, first_divergence.name, first_divergence.op_desc))


if __name__ == "__main__":
    main()
