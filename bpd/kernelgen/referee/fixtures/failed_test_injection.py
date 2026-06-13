#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""FAILED-TEST INJECTION — gate-the-gates probe #1 (Bocher, 2026-06-13).
For each core comparison gate: feed it a KNOWN-BAD artifact and confirm it REDS.
A gate that cannot fail cannot protect. Each probe constructs a deliberate
mismatch (single bit flip, single element perturbation, wrong-shape, NaN) and
asserts the gate returns FAIL — if it returns PASS, the gate is BROKEN (silently
green = the dangerous class)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.abspath(".."))
from fusion_gate import compare_outputs, GateResult

results = []
def probe(name, expect_fail, got):
    ok = (got.passed == (not expect_fail))
    results.append((name, "GATE-OK" if ok else "GATE-BROKEN", f"passed={got.passed} mismatches={got.mismatches} max_ulp={got.max_ulp}"))

# 1. bit_exact gate: identical arrays must PASS (control)
a = np.random.RandomState(1).randn(896).astype(np.float32)
probe("bitexact/identical(control)", expect_fail=False, got=compare_outputs(a, a.copy(), "bit_exact"))

# 2. single LAST-element bit flip must FAIL
b = a.copy(); b_view = b.view(np.uint32); b_view[-1] ^= 1
probe("bitexact/1ulp-last-elem", expect_fail=True, got=compare_outputs(a, b, "bit_exact"))

# 3. single MIDDLE bit flip (high bit of mantissa) must FAIL
c = a.copy(); c_view = c.view(np.uint32); c_view[448] ^= (1 << 22)
probe("bitexact/mantissa-mid", expect_fail=True, got=compare_outputs(a, c, "bit_exact"))

# 4. NaN injection must FAIL (NaN != NaN bitwise? both raw-bit equal NaNs PASS - probe the dangerous direction)
d = a.copy(); d[0] = np.nan
probe("bitexact/one-nan", expect_fail=True, got=compare_outputs(a, d, "bit_exact"))

# 5. BOTH arms same NaN bits: bit-exact says PASS — document the semantics
e = a.copy(); e[0] = np.nan
f = a.copy(); f[0] = np.nan
probe("bitexact/both-nan-same-bits(semantics)", expect_fail=False, got=compare_outputs(e, f, "bit_exact"))

# 6. tolerance gate: drift JUST UNDER must PASS, JUST OVER must FAIL
g = a.copy(); g[10] += 1e-8
probe("tolerance(1e-7)/under", expect_fail=False, got=compare_outputs(a, g, ("tolerance", 1e-7)))
h = a.copy(); h[10] += 1e-6
probe("tolerance(1e-7)/over", expect_fail=True, got=compare_outputs(a, h, ("tolerance", 1e-7)))

# 7. all-zeros vs all-zeros (degenerate control — the stale-cubin shape: kernel no-ops, both zero)
z = np.zeros(896, np.float32)
probe("bitexact/zeros-vs-zeros(degenerate-semantics)", expect_fail=False, got=compare_outputs(z, z.copy(), "bit_exact"))

for name, verdict, detail in results:
    print(f"{verdict:12s} {name:45s} {detail}")
n_broken = sum(1 for _, v, _ in results if v == "GATE-BROKEN")
print(f"\n{len(results)} probes, {n_broken} GATE-BROKEN")
sys.exit(1 if n_broken else 0)
