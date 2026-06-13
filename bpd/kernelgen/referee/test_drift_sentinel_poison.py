#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os, sys as _sys
import os as _os2
_REPO = _os2.environ.get("LLAMATOV_ROOT") or _os2.path.abspath(_os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), *[".."]*8))

def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

"""
test_drift_sentinel_poison.py — GATE-THE-GATES for the drift sentinel (defense #3).

Bocher's principle: a gate that can't fail can't protect. The sentinel passes on the healthy shipping
artifact (proven); this proves it FAILS on poisoned inputs — so its green is meaningful.

Three poisons, one per check:
  1. gguf_validate: point it at a crossword-attack GGUF (a known-malformed model) -> must report DRIFT
     (the validator must REJECT it -> the all-pass check fails).
  2. scanner vocabulary: feed it nonsense op names the engine never uses -> must report DRIFT (the
     scanner can't classify them -> below the >=4 threshold).
  (check #3, decode-scan-wired, is a consult-cleanliness check; poisoning it would mean breaking the
   real file, which is out of scope here — its gate-the-gates is the smoke probe's singleton detector.)

Run: python3 test_drift_sentinel_poison.py   (exit 0 = the sentinel correctly REDs on poison)
"""
import sys, os, glob
sys.path.insert(0, _os2.path.join(_REPO, "bpd/kernelgen/referee"))
import drift_sentinel as ds

ROOT = "<repo>"
ATTACKS = glob.glob(f"{ROOT}/bpd/tests/crossword_attacks/*.gguf")

failures = []
print("=== GATE-THE-GATES: drift sentinel must RED on poison ===")

# Poison 1: a malformed model. The healthy model passes; an attack model must be flagged as drift.
if ATTACKS:
    bad = sorted(ATTACKS)[0]
    saved = ds.MODEL
    ds.MODEL = bad
    ok, detail = ds.check_gguf_validate()
    ds.MODEL = saved
    if ok is False:
        print(f"  POISON-1 gguf_validate(attack model) -> correctly REDs ({os.path.basename(bad)}: {detail[:60]})")
    else:
        print(f"  POISON-1 FAIL: attack model {os.path.basename(bad)} was NOT flagged (ok={ok})")
        failures.append("gguf_validate did not reject a crossword attack")
else:
    print("  POISON-1 SKIP: no crossword attack GGUFs found")

# Poison 2: nonsense vocabulary the scanner can't classify.
saved = ds.ENGINE_OPS
ds.ENGINE_OPS = ["frobnicate", "zorptangle", "quuxify", "bibblewop", "grommish", "snarfle"]
ok, detail = ds.check_scanner_vocabulary()
ds.ENGINE_OPS = saved
if ok is False:
    print(f"  POISON-2 scanner(nonsense vocab) -> correctly REDs ({detail[:60]})")
else:
    print(f"  POISON-2 FAIL: nonsense vocabulary was NOT flagged (ok={ok}, {detail})")
    failures.append("scanner classified nonsense ops")

# Sanity: with the REAL inputs restored, the sentinel must still PASS (no false alarm).
ok1, _ = ds.check_gguf_validate()
ok2, _ = ds.check_scanner_vocabulary()
if ok1 and ok2:
    print("  SANITY healthy inputs -> still PASS (the gate greens on good, reds on bad)")
else:
    print(f"  SANITY FAIL: healthy inputs no longer pass (gguf={ok1}, scanner={ok2})")
    failures.append("false alarm on healthy inputs after restore")

print()
if failures:
    print(f"GATE-THE-GATES FAILED: {len(failures)} — the sentinel cannot detect what it claims to:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("GATE-THE-GATES PASS: the drift sentinel REDs on poison and greens on health — it can protect.")
sys.exit(0)
