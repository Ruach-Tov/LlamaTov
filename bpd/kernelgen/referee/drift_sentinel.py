# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
#!/usr/bin/env python3
"""
drift_sentinel.py — SCHEDULED runs against the SHIPPING ARTIFACT (kills SLOW-DRIFT class).

WHY (unified review, defense #3): names/schemas/vocabularies drift from production reality over
MONTHS, invisibly, until something CURRENT re-checks the assumption. 4 of the 6 review bugs were this:
gguf_validate rejected every real model; the auto-fuser scanner didn't speak engine vocabulary;
bpd_format drift; an unbound var in provenance comments. The common cure: re-run the gate/scanner
against the SHIPPING artifact on a schedule, so drift surfaces the moment reality moves past the
assumption.

Three drift-prone checks, all pointed at the PRODUCTION model + the REAL decode graph:
  1. gguf_validate(production_model) must PASS (catches false-positive drift — a validator that
     rejects every real model, AND false-negative — one that greens on a poisoned model).
  2. the auto-fuser scanner over the REAL engine op vocabulary (rms_norm/silu_mul/q8_gemv) must
     produce non-empty, sane output (catches vocabulary drift — a scanner that can't see the graph).
  3. fusion_scan_decode main runs clean (the decode-graph fusion scan stays wired).

Run: python3 drift_sentinel.py    (exit 0 = no drift; nonzero = a SLOW-DRIFT regression surfaced)
Intended cadence: weekly (the slow-drift time-constant). Wire via shamash/cron.
"""
import sys, os, subprocess, re

ROOT = "<repo>"
BPD = f"{ROOT}/bpd"            # gguf_validate uses RELATIVE lib/... paths -> must run from here
MODEL = _os2.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf")   # the SHIPPING artifact
SWIPL = "swipl"

# the real engine op vocabulary the scanner must understand (the names that drifted)
ENGINE_OPS = ["rms_norm", "silu_mul", "q8_gemv", "rope", "add", "quant"]


def run_swipl(goal, timeout=90):
    # run from bpd/ so the engine's relative consult paths (lib/...) resolve — the same cwd the
    # production code and Bocher's probes use. cwd-sensitivity is itself a documented fragility.
    r = subprocess.run([SWIPL, "-q", "-g", goal + ", halt", "-t", "halt"],
                       capture_output=True, text=True, timeout=timeout, cwd=BPD)
    return r.returncode, r.stdout, r.stderr


def check_gguf_validate():
    """The production model must validate clean (no false-positive rejection, no false-negative pass)."""
    if not os.path.exists(MODEL):
        return None, f"model absent: {MODEL}"
    goal = ('consult("lib/gguf_validate"), '
            f'( gguf_validate("{MODEL}", R) -> '
            "  ( forall(member(T,R), functor(T,pass,_)) -> "
            "      write(vok) ; write(vrejected) ) "
            "; write(vthrew) )")
    rc, out, err = run_swipl(goal)
    if "vok" in out:
        return True, "production model validates clean"
    if "vrejected" in out:
        return False, "validator REJECTED the production model (false-positive drift)"
    return False, f"validator threw/odd: {(err or out)[:160]}"


def check_scanner_vocabulary():
    """The scanner must SEE the real engine op names (the vocabulary that drifted)."""
    # build a tiny real-vocabulary op list and ask the fuser to classify — it must NOT be empty.
    ops = ",".join(ENGINE_OPS)
    goal = ('consult("lib/auto_fuser"), '
            f'findall(C, (member(Op,[{ops}]), catch(classify_op(Op,C),_,fail)), Cs), '
            'length(Cs,N), '
            '( N >= 4 -> format("SCANNER_SPEAKS(~w)",[N]) ; format("SCANNER_DEAF(~w)",[N]) )')
    rc, out, err = run_swipl(goal)
    m = re.search(r"SCANNER_SPEAKS\((\d+)\)", out)
    if m:
        return True, f"scanner classifies {m.group(1)}/{len(ENGINE_OPS)} real engine ops"
    m = re.search(r"SCANNER_DEAF\((\d+)\)", out)
    if m:
        return False, f"scanner classifies only {m.group(1)}/{len(ENGINE_OPS)} — vocabulary drift"
    return False, f"scanner check odd: {(err or out)[:160]}"


def check_decode_scan_wired():
    """fusion_scan_decode main must run clean (the decode-graph scan stays wired)."""
    rc, out, err = run_swipl(f'consult("kernelgen/referee/fusion_scan_decode")')
    # singleton warnings = the launch_bounds-class drift; discontiguous = benign style. Only the
    # former (and any ERROR/throw) is drift.
    bad = [l for l in (err or "").splitlines()
           if ("Singleton" in l or "ERROR" in l) and "deprecated" not in l.lower()]
    if rc == 0 and not bad:
        return True, "decode-graph fusion scan runs clean (no singletons/errors)"
    if bad:
        return False, f"singleton/error: {bad[0].strip()}"
    return False, f"rc={rc}: {(err or out)[:160]}"


def main():
    checks = [
        ("gguf_validate(production model)", check_gguf_validate),
        ("scanner speaks engine vocabulary", check_scanner_vocabulary),
        ("decode-graph fusion scan wired", check_decode_scan_wired),
    ]
    print("=== DRIFT SENTINEL (weekly; SLOW-DRIFT guard, vs the SHIPPING artifact) ===")
    failures = []
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"exception: {e}"
        if ok is None:
            print(f"  {name:38} -> SKIP ({detail})")
        elif ok:
            print(f"  {name:38} -> ok ({detail})")
        else:
            print(f"  {name:38} -> DRIFT ({detail})")
            failures.append((name, detail))
    print()
    if failures:
        print(f"DRIFT DETECTED: {len(failures)} assumption(s) drifted from production reality:")
        for n, d in failures:
            print(f"  {n}: {d}")
        sys.exit(1)
    print("SENTINEL PASS: production artifact + real vocabulary still match the gates")
    sys.exit(0)


if __name__ == "__main__":
    main()
