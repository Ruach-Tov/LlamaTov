#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# FAILED-TEST INJECTION probe #3: long_run_referee's drift/margin/near-flip layer.
# Construct fake A/B step records with a KNOWN drift and a KNOWN near-flip; verify
# the metrics detect both (drift>0 reported, near-flip censused, divergence located).
import numpy as np, json
def steps(seed, n=5, drift_at=None, drift_mag=0.0, fliptok_at=None):
    rng = np.random.RandomState(seed); out = []
    for i in range(n):
        v = rng.randn(64).astype(np.float32)
        v[10] = 5.0; v[20] = 4.99  # engineered thin margin: 0.01
        tok = 10
        if fliptok_at is not None and i == fliptok_at: tok = 20
        if drift_at is not None and i == drift_at: v[10] += drift_mag
        out.append({"step": i, "tok": tok, "runner_up": 20, "margin": float(v[10]-v[20]),
                    "logits_u32": v.view(np.uint32).tolist()})
    return out
A = steps(7); B_drift = steps(7, drift_at=2, drift_mag=0.004)  # drift 0.004 < margin 0.01: tokens same, drift nonzero, near-flip (margin<5x drift? 0.01<0.02 yes)
B_flip = steps(7, fliptok_at=3)
# replicate the referee's comparison math
def analyze(A, B):
    drifts, nf, div = [], [], None
    for sa, sb in zip(A, B):
        if sa["tok"] != sb["tok"]: div = sa["step"]; break
        la = np.array(sa["logits_u32"],np.uint32).view(np.float32); lb = np.array(sb["logits_u32"],np.uint32).view(np.float32)
        d = float(np.abs(la-lb).max()); drifts.append(d)
        if d > 0 and sa["margin"] < 5*d: nf.append(sa["step"])
    return max(drifts) if drifts else 0.0, nf, div
d1, nf1, div1 = analyze(A, steps(7))
print("control: max_drift=%.4g nearflips=%s div=%s -> %s" % (d1, nf1, div1, "GATE-OK" if (d1==0 and not nf1 and div1 is None) else "GATE-BROKEN"))
d2, nf2, div2 = analyze(A, B_drift)
print("injected-drift(0.004@s2, margin 0.01): max_drift=%.4g nearflips=%s -> %s" % (d2, nf2, "GATE-OK" if (abs(d2-0.004)<1e-6 and nf2==[2]) else "GATE-BROKEN"))
d3, nf3, div3 = analyze(A, B_flip)
print("injected-tokenflip@s3: div=%s -> %s" % (div3, "GATE-OK" if div3==3 else "GATE-BROKEN"))
