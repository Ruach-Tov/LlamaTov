#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""ci_build_test_dashboard.py — Complete CI pipeline for enclave.

Called by webhook_runner.py after git pull.
Builds .so files from source, runs test harness, regenerates dashboard.

Build → Test → Facts → Dashboard. All from tracked source code.
"""
import subprocess
import os
import sys
import time

# ============================================================
# Paths — all relative to repos, no /tmp/ for source or artifacts
# ============================================================

# The Ruach-Tov repo (has test harness, dashboard generator, facts)
RT_REPO = os.path.expanduser("~/Ruach-Tov")

# The bpd-substrate repo (has .ll files, .c files, Makefile)
# www user: ~/repo/bpd-substrate. Others: ~/Ruach-Tov/bpd-substrate
for _try in ["~/repo/bpd-substrate", "~/Ruach-Tov/bpd-substrate", "~/bpd-substrate"]:
    _p = os.path.expanduser(_try)
    if os.path.isdir(_p):
        BPD_REPO = _p
        break
else:
    BPD_REPO = None

# All generated output goes here — NOT /tmp/, NOT in git
GENERATED = "/tmp/bpd-generated"
BUILD_DIR = os.path.join(GENERATED, "build")
OUTPUT_DIR = os.path.join(GENERATED, "output")

# Tools
LLVM_AS = "/nix/store/a8hbpr6c8rhdcvgr19r8gnqifnnfb9q3-llvm-19.1.7/bin/llvm-as"
LLC = "/nix/store/a8hbpr6c8rhdcvgr19r8gnqifnnfb9q3-llvm-19.1.7/bin/llc"
LLVM_DEV = "/nix/store/3b3vq1i94xncrs41aypsayjacgjhg000-llvm-19.1.7-dev"
LLVM_LIB = "/nix/store/yryd32d76cmrdc5npg49j5f1zipbxdfw-llvm-19.1.7-lib"
SWIPL_BASE = "/nix/store/jn4yixfq3qjdl3d4g6hfvl8nnn2pjhc5-swi-prolog-9.2.9/lib/swipl"

# Python with torch
PY = "/nix/store/gc38dj7srpf7c0nlrc0g8x72bcmglmiv-python3-3.12.11-env/bin/python3"
PYPATH = (
    "/nix/store/r3m9fwhp3fmp0zwi32d8a31yi4a1pkqf-python3.12-torch-2.11.0"
    "/lib/python3.12/site-packages:"
    "/nix/store/m8zsv491f72nfm3c41j5sif1c5kgbksj-python3-3.12.11-env"
    "/lib/python3.12/site-packages"
)

def log(msg):
    print(f"[ci] {time.strftime('%H:%M:%S')} {msg}", flush=True)

def run(cmd, cwd=None, env=None, timeout=60):
    """Run a command, return (success, stdout)."""
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            log(f"  FAILED: {' '.join(cmd[:3])}")
            if r.stderr:
                log(f"  stderr: {r.stderr[:200]}")
            return False, r.stdout
        return True, r.stdout
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT: {' '.join(cmd[:3])}")
        return False, ""


def step_pull():
    """Pull both repos."""
    log("=== PULL ===")
    ok1, _ = run(["git", "pull", "--ff-only"], cwd=RT_REPO, timeout=30)
    log(f"  Ruach-Tov: {'ok' if ok1 else 'FAILED'}")
    
    if os.path.isdir(BPD_REPO):
        ok2, _ = run(["git", "pull", "--ff-only"], cwd=BPD_REPO, timeout=30)
        log(f"  bpd-substrate: {'ok' if ok2 else 'FAILED'}")
    else:
        log(f"  bpd-substrate: not found at {BPD_REPO}, skipping")
        ok2 = False
    
    return ok1


def step_build():
    """Build .so files from source."""
    log("=== BUILD ===")
    os.makedirs(BUILD_DIR, exist_ok=True)
    
    results = {}
    
    # 1. Build bpd_cpu.so from bpd_cpu.c (hand-written C reference kernels)
    bpd_cpu_c = os.path.join(BPD_REPO, "bench/bpd_cpu.c")
    bpd_cpu_so = os.path.join(BUILD_DIR, "bpd_cpu.so")
    
    if os.path.exists(bpd_cpu_c):
        ok, _ = run(["gcc", "-O2", "-shared", "-fPIC", "-msse3", "-mno-avx", "-mno-fma",
                      "-o", bpd_cpu_so, bpd_cpu_c, "-lm"])
        log(f"  bpd_cpu.so: {'ok' if ok else 'FAILED'}")
        results['cpu'] = ok
    else:
        log(f"  bpd_cpu.c not found at {bpd_cpu_c}")
        results['cpu'] = False
    
    # 2. Build bpd_llvm.so from Prolog-generated LLVM IR
    # Step 2a: Build the Prolog foreign interface .so
    bpd_llvm_c = os.path.join(BPD_REPO, "lib/bpd_llvm.c")
    bpd_llvm_elem_c = os.path.join(BPD_REPO, "lib/bpd_llvm_elem.c")
    prolog_so = os.path.join(BUILD_DIR, "bpd_llvm.so")
    elem_so = os.path.join(BUILD_DIR, "bpd_llvm_elem.so")
    
    if os.path.exists(bpd_llvm_c):
        ok, _ = run(["gcc", "-shared", "-fPIC", "-O2",
                      f"-I{LLVM_DEV}/include", f"-I{SWIPL_BASE}/include",
                      bpd_llvm_c,
                      f"-L{LLVM_LIB}/lib", "-lLLVM-19",
                      f"-Wl,-rpath,{LLVM_LIB}/lib",
                      "-o", prolog_so])
        log(f"  bpd_llvm_prolog.so: {'ok' if ok else 'FAILED'}")
    
    if os.path.exists(bpd_llvm_elem_c):
        ok, _ = run(["gcc", "-shared", "-fPIC", "-O2",
                      f"-I{LLVM_DEV}/include", f"-I{SWIPL_BASE}/include",
                      bpd_llvm_elem_c,
                      f"-L{LLVM_LIB}/lib", "-lLLVM-19",
                      f"-Wl,-rpath,{LLVM_LIB}/lib",
                      "-o", elem_so])
        log(f"  bpd_llvm_elem_prolog.so: {'ok' if ok else 'FAILED'}")
    
    # Step 2b: Use Prolog to emit LLVM IR, then compile to .so
    emit_ll = os.path.join(BUILD_DIR, "bpd_unary_all.ll")
    emit_bc = os.path.join(BUILD_DIR, "bpd_unary_all.bc")
    emit_o  = os.path.join(BUILD_DIR, "bpd_unary_all.o")
    llvm_so = os.path.join(BUILD_DIR, "bpd_unary_llvm.so")
    
    if os.path.exists(prolog_so) and os.path.exists(elem_so):
        # Generate IR from Prolog
        prolog_cmd = (
            f'use_module(library(shlib)),'
            f'load_foreign_library("{prolog_so[:-3]}"),'
            f'load_foreign_library("{elem_so[:-3]}"),'
            f'llvm_create_module(bpd, Mod),'
            f'llvm_emit_unary(Mod, relu, bpd_relu),'
            f'llvm_emit_unary(Mod, silu, bpd_silu),'
            f'llvm_emit_unary(Mod, sigmoid, bpd_sigmoid),'
            f'llvm_emit_unary(Mod, softplus, bpd_softplus),'
            f'llvm_emit_unary(Mod, leaky_relu, bpd_leaky_relu),'
            f'llvm_emit_unary(Mod, elu, bpd_elu),'
            f'llvm_emit_unary(Mod, softsign, bpd_softsign),'
            f'llvm_dump_ir(Mod),'
            f'llvm_dispose_module(Mod),'
            f'halt'
        )
        ok, ir_text = run(["swipl", "-g", prolog_cmd], timeout=30)
        if ok and ir_text:
            with open(emit_ll, 'w') as f:
                f.write(ir_text)
            log(f"  Prolog emitted {len(ir_text)} bytes of LLVM IR")
            
            # Also append the hand-written .ll files (tanh, sum, scale, etc)
            for ll_name in ['bpd_tanh_fix.ll', 'bpd_sum_sse3.ll', 'bpd_exp_fix.ll',
                           'bpd_scale_cumsum.ll',
                           'bpd_losses.ll', 'bpd_pool.ll', 'bpd_conv.ll']:
                ll_path = os.path.join(BPD_REPO, "lib", ll_name)
                if os.path.exists(ll_path):
                    # These are standalone modules — compile separately
                    bc_path = os.path.join(BUILD_DIR, ll_name.replace('.ll', '.bc'))
                    o_path = os.path.join(BUILD_DIR, ll_name.replace('.ll', '.o'))
                    run([LLVM_AS, ll_path, "-o", bc_path])
                    run([LLC, "-filetype=obj", "-O2", "-mattr=+sse3", 
                         "-o", o_path, bc_path])
            
            # Compile the Prolog-emitted IR
            ok1, _ = run([LLVM_AS, emit_ll, "-o", emit_bc])
            ok2, _ = run([LLC, "-filetype=obj", "-O2", "-mattr=+sse3",
                          "-o", emit_o, emit_bc])
            
            if ok1 and ok2:
                # Link all .o files into one .so
                o_files = [emit_o]
                for ll_name in ['bpd_tanh_fix', 'bpd_sum_sse3', 'bpd_exp_fix',
                               'bpd_scale_cumsum',
                               'bpd_losses', 'bpd_pool', 'bpd_conv']:
                    o_path = os.path.join(BUILD_DIR, ll_name + '.o')
                    if os.path.exists(o_path):
                        o_files.append(o_path)
                
                ok, _ = run(["gcc", "-shared", "-o", llvm_so] + o_files + ["-lm"])
                log(f"  bpd_unary_llvm.so: {'ok' if ok else 'FAILED'} ({len(o_files)} objects)")
                results['llvm'] = ok
            else:
                log("  LLVM compile failed")
                results['llvm'] = False
        else:
            log("  Prolog IR emission failed")
            results['llvm'] = False
    else:
        log("  Prolog .so not available, skipping LLVM emit")
        results['llvm'] = False
    
    # Also build vec_dot from bpd_llvm.c via Prolog
    # (the vec_dot emitter uses the bpd_llvm.so foreign interface)
    dot_ll = os.path.join(BUILD_DIR, "bpd_dot.ll")
    dot_bc = os.path.join(BUILD_DIR, "bpd_dot.bc")
    dot_o  = os.path.join(BUILD_DIR, "bpd_dot.o")
    
    if os.path.exists(prolog_so):
        prolog_cmd = (
            f'use_module(library(shlib)),'
            f'load_foreign_library("{prolog_so[:-3]}"),'
            f'llvm_create_module(bpd, Mod),'
            f'llvm_emit_vec_dot(Mod, 8, 0, bpd_dot),'
            f'llvm_dump_ir(Mod),'
            f'llvm_dispose_module(Mod),'
            f'halt'
        )
        ok, ir_text = run(["swipl", "-g", prolog_cmd], timeout=30)
        if ok and ir_text:
            with open(dot_ll, 'w') as f:
                f.write(ir_text)
            run([LLVM_AS, dot_ll, "-o", dot_bc])
            run([LLC, "-filetype=obj", "-O2", "-mattr=+sse3", "-o", dot_o, dot_bc])
            if os.path.exists(dot_o):
                # Add to the llvm .so
                log("  vec_dot object built")
    
    return results


def step_test():
    """Run the test harness with the built .so files."""
    log("=== TEST ===")
    
    harness = os.path.join(RT_REPO, "bpd/tests/verify_llvm_ops_auto.py")
    if not os.path.exists(harness):
        log(f"  Harness not found: {harness}")
        return False
    
    env = dict(os.environ)
    # Point the harness at our build dir
    env["BPD_LLVM_SO"] = os.path.join(BUILD_DIR, "bpd_unary_llvm.so")
    env["BPD_CPU_SO"] = os.path.join(BUILD_DIR, "bpd_cpu.so")
    env["FACTS_PATH"] = os.path.join(OUTPUT_DIR, "llvm_op_match.o.pl")
    
    ok, output = run([PY, harness], cwd=os.path.join(RT_REPO, "bpd"),
                     env=env, timeout=300)
    log(f"  Harness: {'ok' if ok else 'FAILED'}")
    if output:
        log(f"  Output (last 300): {output[-300:]}")
    return ok


def step_dashboard():
    """Regenerate dashboard SVG from Prolog facts."""
    log("=== DASHBOARD ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    svg_path = os.path.join(OUTPUT_DIR, "llvm_match_status.o.svg")
    bpd_dir = os.path.join(RT_REPO, "bpd")
    
    ok, _ = run(["swipl", "-g", "main, halt", "llvm_match_status.pl", "--", svg_path],
                cwd=bpd_dir, timeout=15)
    log(f"  SVG: {'ok' if ok else 'FAILED'}")
    
    if ok and os.path.exists(svg_path):
        size = os.path.getsize(svg_path)
        log(f"  {svg_path} ({size} bytes)")
    
    return ok


def main():
    log("=" * 60)
    log("CI PIPELINE: build → test → dashboard")
    log("=" * 60)
    
    step_pull()
    results = step_build()
    step_test()
    step_dashboard()
    
    log("=" * 60)
    log("DONE")


if __name__ == "__main__":
    main()
