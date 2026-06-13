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
smoke_emitters.py — PER-PUSH SMOKE PROBE for the GEMV emitters (kills FAST-REGRESSION class).

WHY (unified review, defense #2): optimization velocity sheds collateral. The launch_bounds wave
(9fdc171f) put `BlockSz is BM*32` into four serial-mode clauses where BM is unbound -> is/2 threw
mid-emit -> a partial 4-line .cu -> the sweep reference was garbage and every verdict read BAD. The
singleton warning named the exact lines at every consult — the dead canary nobody read. This probe
makes the canary AUDIBLE and the partial-emit DETECTABLE, fast, in the change path.

What it checks for EVERY emit mode:
  1. consult is WARNING-FREE (canary audibility — a singleton/discontiguous warning fails the smoke).
  2. emit produces a COMPLETE kernel (not the partial-.cu shape: must have the __global__ head, a
     balanced closing brace, and the expected kernel name).
  3. (optional, --compile) the kernel nvcc-compiles via the SHARED toolchain module.

Run: python3 smoke_emitters.py [--compile]   (exit 0 = all modes pass; nonzero = a regression)
"""
import sys, os, subprocess, re, tempfile
sys.path.insert(0, _BPD); sys.path.insert(0, _os.path.join(_BPD, "lib"))

EMIT = _os2.path.join(_REPO, "bpd/kernelgen/emitters/q8_0_from_facts")
SWIPL = "swipl"

# (mode-term, expected kernel name) for each GEMV emit mode.
MODES = [
    ("tiled_v4(16,128)",          "k_q8_0_gemv"),
    ("tiled_v4_reghoist(16,128)", "k_q8_0_gemv"),
    ("tiled_v4_addres(16,128)",   "k_q8_0_gemv_addres"),
    ("tiled_v4_silu(16,128)",     "k_q8_0_gemv_silu"),
    ("tiled(16,256,1)",           "k_q8_0_gemv"),
    ("canonical_serial_gemv",     "k_q8_0_gemv"),
    ("dp4a",                      "k_q8_0_gemv"),
]


def emit_mode(mode, out):
    goal = (f'consult("{EMIT}"), q8_0_op_expr(E), '
            f'emit_from_fact(E,[mode({mode})],"{out}"), halt')
    r = subprocess.run([SWIPL, "-q", "-g", goal], capture_output=True, text=True, timeout=60)
    return r.stdout, r.stderr


def check_warning_free(stderr):
    """Canary audibility: any Warning in the consult output is a failure (singleton, discontiguous)."""
    lines = [l for l in stderr.splitlines()
             if "Warning" in l and "deprecated" not in l.lower()]
    return (len(lines) == 0, lines[:4])


def check_complete(out, kname):
    """The partial-.cu detector: a complete kernel has the __global__ head with the right name AND a
    balanced final brace (the launch_bounds bug produced a 4-line truncated file)."""
    if not os.path.exists(out):
        return False, "no output file (emit threw)"
    src = open(out).read()
    if f"void __launch_bounds__" not in src and "__global__" not in src:
        return False, "no __global__ kernel head"
    if kname not in src:
        return False, f"kernel name {kname} absent"
    # balanced braces + a closing brace near the end (truncated emits end mid-statement)
    if src.count("{") != src.count("}"):
        return False, f"unbalanced braces ({src.count('{')} vs {src.count('}')})"
    if len(src.splitlines()) < 10:
        return False, f"suspiciously short ({len(src.splitlines())} lines — partial emit?)"
    return True, f"{len(src.splitlines())} lines, balanced"


def maybe_compile(out):
    try:
        import toolchain as tc
    except Exception as e:
        return None, f"toolchain import: {e}"
    cubin = out.replace(".cu", ".cubin")
    if os.path.exists(cubin):
        os.remove(cubin)
    cmd = tc.nvcc_compile_cmd(out, cubin)
    r = subprocess.run(cmd, capture_output=True, text=True, env=tc.nvcc_env(), timeout=120)
    return os.path.exists(cubin), (r.stderr[:200] if not os.path.exists(cubin) else "ok")


def main():
    do_compile = "--compile" in sys.argv
    tmp = tempfile.mkdtemp(prefix="smoke_emit_")
    failures = []
    print("=== EMITTER SMOKE PROBE (per-push; FAST-REGRESSION guard) ===")
    for mode, kname in MODES:
        out = os.path.join(tmp, f"{mode.split('(')[0]}.cu")
        stdout, stderr = emit_mode(mode, out)
        wf, wlines = check_warning_free(stderr)
        ok, detail = check_complete(out, kname)
        line = f"  {mode:28} -> "
        if not wf:
            failures.append((mode, "WARNING", wlines))
            line += f"FAIL[warnings: {wlines[0] if wlines else '?'}]"
        elif not ok:
            failures.append((mode, "INCOMPLETE", detail))
            line += f"FAIL[{detail}]"
        else:
            line += f"ok ({detail})"
            if do_compile:
                cok, cdet = maybe_compile(out)
                if cok is False:
                    failures.append((mode, "COMPILE", cdet))
                    line += f" COMPILE-FAIL[{cdet}]"
                elif cok:
                    line += " +compiles"
        print(line)
    print()
    if failures:
        print(f"SMOKE FAILED: {len(failures)} mode(s) regressed:")
        for m, kind, det in failures:
            print(f"  {m}: {kind} — {det}")
        sys.exit(1)
    print(f"SMOKE PASS: all {len(MODES)} GEMV emit modes warning-free + complete"
          + (" + compile" if do_compile else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
