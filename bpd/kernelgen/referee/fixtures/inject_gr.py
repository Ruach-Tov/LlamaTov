#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import json, sys
import numpy as np
def arm(gen, logits_seed, perturb_step=None):
    steps = []
    rng = np.random.RandomState(logits_seed)
    for i, tok in enumerate(gen[3:]):
        v = rng.randn(64).astype(np.float32)
        if perturb_step is not None and i == perturb_step:
            vv = v.view(np.uint32); vv[7] ^= 1
        steps.append({"step": i, "tok": int(tok), "logits_u32": v.view(np.uint32).tolist(), "dlen": 3 + i})
    return {"gen": gen, "steps": steps,
            "path_attestation": {"device_logits": True, "env_toggles": {}, "folded_logits_executed": True}}
gen = [1, 415, 6557, 310, 470, 895]
eager = arm(gen, 42); graph_same = arm(gen, 42); graph_bitflip = arm(gen, 42, perturb_step=1)
def compare(e, g):
    fails = []
    for se, sg in zip(e["steps"], g["steps"]):
        if se["logits_u32"] != sg["logits_u32"]:
            fails.append(se["step"])
    return fails
f1 = compare(eager, graph_same); f2 = compare(eager, graph_bitflip)
ok1 = "GATE-OK" if not f1 else "GATE-BROKEN"
ok2 = "GATE-OK" if f2 == [1] else "GATE-BROKEN"
print("control(identical arms): fails=%s -> %s" % (f1, ok1))
print("injected(1-bit flip step1): fails=%s -> %s" % (f2, ok2))
graph_wrongpath = arm(gen, 42); graph_wrongpath["path_attestation"]["folded_logits_executed"] = False
att = eager["path_attestation"] != graph_wrongpath["path_attestation"]
print("attestation-mismatch detected: %s -> %s" % (att, "GATE-OK" if att else "GATE-BROKEN"))
