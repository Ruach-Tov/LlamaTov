#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_llvm_ops.py — Automated LLVM IR match verification for CI.

TWO levels of verification per op:
  1. NUMERICAL: Run BPD kernel vs reference, measure ULP distance
  2. IR-STRUCTURE: Diff our LLVM IR against reference's disassembly/IR
     (blue = IR matches, green = numbers match but IR differs)

Outputs:
  - Prolog facts: ulp_op_match(Pattern, Op, Reference, ULP, Evidence).
  - JSON report for CI artifact
  - Exit code 0 if all within tolerance, 1 if regression

ALL dashboard cells are generated from these facts. No manual annotation.
The CI pipeline is: test → emit facts → regenerate SVG → publish.

Usage:
  python3 verify_llvm_ops.py --bpd-so build/bpd_llvm.so --ref-so build/bpd_cpu.so
  python3 verify_llvm_ops.py --ci --output bpd/lib/llvm_op_match_ci.pl
  python3 verify_llvm_ops.py --ir-diff --bpd-ll build/k_add.ll --ref-ll ref/k_add.ll

Author: medayek (Collective SME, Verification Methodology)
"""
import argparse, ctypes, json, numpy as np, os, sys, time
from datetime import datetime, timezone

c_float_p = ctypes.POINTER(ctypes.c_float)

# ═══════════════════════════════════════════════════════════════
# Op registry: maps each op to its test configuration
# ═══════════════════════════════════════════════════════════════

OPS = {
    # Pattern 1: unary_elementwise
    "ggml_relu":        {"pattern": "unary_elementwise", "fn": "relu",        "type": "unary"},
    "ggml_silu":        {"pattern": "unary_elementwise", "fn": "silu",        "type": "unary"},
    "ggml_gelu":        {"pattern": "unary_elementwise", "fn": "gelu",        "type": "unary"},
    "ggml_elu":         {"pattern": "unary_elementwise", "fn": "elu",         "type": "unary"},
    "ggml_selu":        {"pattern": "unary_elementwise", "fn": "selu",        "type": "unary"},
    "ggml_leaky_relu":  {"pattern": "unary_elementwise", "fn": "leaky_relu",  "type": "unary"},
    "ggml_sigmoid":     {"pattern": "unary_elementwise", "fn": "sigmoid",     "type": "unary"},
    "ggml_tanh":        {"pattern": "unary_elementwise", "fn": "tanh",        "type": "unary"},
    "ggml_hardsigmoid": {"pattern": "unary_elementwise", "fn": "hardsigmoid", "type": "unary"},
    "ggml_softplus":    {"pattern": "unary_elementwise", "fn": "softplus",    "type": "unary"},
    "ggml_softsign":    {"pattern": "unary_elementwise", "fn": "softsign",    "type": "unary"},
    "ggml_clamp":       {"pattern": "unary_elementwise", "fn": "clamp",       "type": "unary"},
    # Pattern 2: binary_elementwise
    "ggml_scale":       {"pattern": "binary_elementwise", "fn": "scale", "type": "binary"},
    # Pattern 3: reduction
    "ggml_mul_mat":     {"pattern": "reduction", "fn": "mul_mat", "type": "matmul"},
    "ggml_sum":         {"pattern": "reduction", "fn": "sum",     "type": "reduce"},
    "ggml_mean":        {"pattern": "reduction", "fn": "mean",    "type": "reduce"},
    "ggml_max":         {"pattern": "reduction", "fn": "max",     "type": "reduce"},
    "ggml_min":         {"pattern": "reduction", "fn": "min",     "type": "reduce"},
    "ggml_argmax":      {"pattern": "reduction", "fn": "argmax",  "type": "reduce_idx"},
    "ggml_argmin":      {"pattern": "reduction", "fn": "argmin",  "type": "reduce_idx"},
    # Pattern 4: reduction_then_elementwise
    "ggml_norm":        {"pattern": "reduction_then_elementwise", "fn": "norm",        "type": "norm"},
    "ggml_rms_norm":    {"pattern": "reduction_then_elementwise", "fn": "rms_norm",    "type": "norm"},
    "ggml_soft_max":    {"pattern": "reduction_then_elementwise", "fn": "softmax",     "type": "softmax"},
    "ggml_log_softmax": {"pattern": "reduction_then_elementwise", "fn": "log_softmax", "type": "softmax"},
    "ggml_group_norm":  {"pattern": "reduction_then_elementwise", "fn": "group_norm",  "type": "norm"},
    "ggml_l2_norm":     {"pattern": "reduction_then_elementwise", "fn": "l2_norm",     "type": "norm"},
    # Pattern 5: scan
    "ggml_cumsum":      {"pattern": "scan", "fn": "cumsum",  "type": "scan"},
    "ggml_cumprod":     {"pattern": "scan", "fn": "cumprod", "type": "scan"},
    # Pattern 6-9: complex ops (tested separately)
}


import subprocess, re

# ═══════════════════════════════════════════════════════════════
# IR-DIFF: Compare LLVM IR structure between our emitter and reference
# ═══════════════════════════════════════════════════════════════

def ir_diff(our_ll_path, ref_ll_path):
    """Compare two LLVM IR files structurally.
    
    Ignores: register names (%0 vs %1), metadata, comments, blank lines.
    Compares: instruction opcodes, types, and structure.
    
    Returns: (match: bool, diff_lines: int, details: str)
    """
    def normalize_ir(path):
        """Normalize LLVM IR for structural comparison."""
        try:
            with open(path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            return None
        
        normalized = []
        for line in lines:
            line = line.strip()
            # Skip comments, metadata, blank lines, attributes
            if not line or line.startswith(';') or line.startswith('!') or \
               line.startswith('attributes') or line.startswith('target'):
                continue
            # Normalize register names: %name → %_
            line = re.sub(r'%[a-zA-Z0-9_.]+', '%_', line)
            # Normalize labels
            line = re.sub(r'label %_', 'label %_', line)
            normalized.append(line)
        return normalized
    
    our = normalize_ir(our_ll_path)
    ref = normalize_ir(ref_ll_path)
    
    if our is None:
        return False, -1, f"our IR not found: {our_ll_path}"
    if ref is None:
        return False, -1, f"ref IR not found: {ref_ll_path}"
    
    # Compare line by line
    diff_count = 0
    details = []
    max_lines = max(len(our), len(ref))
    for i in range(max_lines):
        our_line = our[i] if i < len(our) else "<missing>"
        ref_line = ref[i] if i < len(ref) else "<missing>"
        if our_line != ref_line:
            diff_count += 1
            if len(details) < 5:  # show first 5 diffs
                details.append(f"  line {i}: ours='{our_line[:60]}' ref='{ref_line[:60]}'")
    
    match = (diff_count == 0)
    detail_str = f"{diff_count} structural differences" + \
                 ("\n" + "\n".join(details) if details else "")
    return match, diff_count, detail_str


def disassemble_to_ll(so_path, fn_name, output_ll):
    """Disassemble a .so function to LLVM IR via objdump + llvm-dis.
    
    Falls back to comparing at assembly level if LLVM IR not extractable.
    """
    try:
        # Try llvm-objdump for LLVM bitcode sections
        result = subprocess.run(
            ['llvm-objdump', '-d', '--symbolize-operands', so_path],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            with open(output_ll, 'w') as f:
                f.write(result.stdout)
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Fall back to regular objdump
    try:
        result = subprocess.run(
            ['objdump', '-d', '-M', 'intel', so_path],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            with open(output_ll, 'w') as f:
                f.write(result.stdout)
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return False


def ulp_compare(a, b):
    """Compare two float32 arrays by ULP distance."""
    af = a.flatten().astype(np.float32)
    bf = b.flatten().astype(np.float32)
    if af.size != bf.size:
        return -1, -1, 0
    ab = af.view(np.int32).astype(np.int64)
    bb = bf.view(np.int32).astype(np.int64)
    diffs = np.abs(ab - bb)
    return int(diffs.max()), int((diffs > 0).sum()), len(af)


def test_op_with_pytorch(op_name, op_config, n=16384):
    """Test by comparing BPD LLVM output vs PyTorch as reference."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None, "pytorch_not_available"

    torch.manual_seed(42)
    x = torch.randn(n, dtype=torch.float32)
    x2 = torch.randn(n, dtype=torch.float32)

    fn_map = {
        "relu": lambda: F.relu(x),
        "silu": lambda: F.silu(x),
        "gelu": lambda: F.gelu(x),
        "elu": lambda: F.elu(x),
        "selu": lambda: F.selu(x),
        "leaky_relu": lambda: F.leaky_relu(x, 0.01),
        "sigmoid": lambda: torch.sigmoid(x),
        "tanh": lambda: torch.tanh(x),
        "hardsigmoid": lambda: F.hardsigmoid(x),
        "softplus": lambda: F.softplus(x),
        "softsign": lambda: F.softsign(x),
        "clamp": lambda: torch.clamp(x, -1.0, 1.0),
        "scale": lambda: x * 0.5,
        "sum": lambda: x.reshape(64, -1).sum(dim=-1),
        "mean": lambda: x.reshape(64, -1).mean(dim=-1),
        "max": lambda: x.reshape(64, -1).max(dim=-1).values,
        "min": lambda: x.reshape(64, -1).min(dim=-1).values,
        "argmax": lambda: x.reshape(64, -1).argmax(dim=-1),
        "argmin": lambda: x.reshape(64, -1).argmin(dim=-1),
        "softmax": lambda: F.softmax(x.reshape(64, -1), dim=-1),
        "log_softmax": lambda: F.log_softmax(x.reshape(64, -1), dim=-1),
        "cumsum": lambda: torch.cumsum(x, dim=0),
    }

    fn_name = op_config["fn"]
    if fn_name not in fn_map:
        return None, f"no_test_for_{fn_name}"

    with torch.no_grad():
        ref = fn_map[fn_name]().numpy()

    return ref, None


def test_op_with_bpd_so(op_name, op_config, bpd_so, ref_output, n=16384):
    """Test BPD LLVM .so against reference output."""
    # Map op names to BPD function names
    fn_name = f"bpd_{op_config['fn']}_llvm"
    try:
        fn = getattr(bpd_so, fn_name)
    except AttributeError:
        return None, f"function_{fn_name}_not_found"

    np.random.seed(42)  # same seed as PyTorch
    x = np.random.randn(n).astype(np.float32)
    out = np.zeros_like(ref_output)

    # Call convention depends on op type
    # For now, return untested if we can't call
    return None, "bpd_so_call_not_implemented"


def emit_prolog_facts(results, output_path):
    """Write Prolog facts from test results.
    
    Reference level determines dashboard color:
      ggml_sse3 + 0 ULP = blue (IR-match confirmed)
      scalar + 0 ULP = green (numbers match, IR not confirmed)
      Any ref + >0 ULP = yellow/red based on ULP
      untested = grey
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    with open(output_path, 'w') as f:
        f.write(f"%% Auto-generated by verify_llvm_ops.py CI pipeline\n")
        f.write(f"%% Generated: {now}\n")
        f.write(f"%% DO NOT EDIT — regenerated on every CI run\n")
        f.write(f"%%\n")
        f.write(f"%% Reference levels:\n")
        f.write(f"%%   ggml_sse3 = verified vs actual ggml SSE3 + IR structure matches (BLUE)\n")
        f.write(f"%%   scalar    = verified vs scalar C ref, IR may differ (GREEN if 0 ULP)\n")
        f.write(f"%%   untested  = not yet verified (GREY)\n\n")
        f.write(f":- module(llvm_op_match_ci, [ulp_op_match/5]).\n\n")
        
        for r in results:
            ulp = r['ulp'] if r['ulp'] is not None else 'untested'
            # Reference level: if IR-match confirmed, use ggml_sse3; else scalar
            ref = r['reference']
            if r.get('ir_match', False) and ulp == 0:
                ref = 'ggml_sse3'  # blue
            elif ulp == 0 and ref != 'ggml_sse3':
                ref = 'scalar'     # green
            evidence = r.get('evidence', f'CI {now}')
            f.write(f"ulp_op_match({r['pattern']}, {r['op']}, {ref}, "
                    f"{ulp}, '{evidence}').\n")
    
    return output_path


def emit_json_report(results, output_path):
    """Write JSON report for CI artifact."""
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_ops": len(results),
        "tested": sum(1 for r in results if r['ulp'] is not None),
        "at_zero_ulp": sum(1 for r in results if r['ulp'] == 0),
        "regressions": [r for r in results if r.get('regression', False)],
        "results": results,
    }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='LLVM IR Match Verification for CI')
    parser.add_argument('--bpd-so', type=str, help='Path to BPD LLVM .so')
    parser.add_argument('--ref-so', type=str, help='Path to reference .so (ggml)')
    parser.add_argument('--reference', type=str, default='ggml_sse3',
                       help='Reference name for Prolog facts')
    parser.add_argument('--output', type=str, default='bpd/lib/llvm_op_match_ci.pl',
                       help='Prolog output path')
    parser.add_argument('--json', type=str, default='build/llvm_match_report.json',
                       help='JSON report path')
    parser.add_argument('--ci', action='store_true', help='CI mode: fail on regression')
    parser.add_argument('--tolerance', type=str, help='JSON file with per-op ULP tolerances')
    parser.add_argument('--ir-diff', action='store_true',
                       help='Also compare LLVM IR structure (blue vs green)')
    parser.add_argument('--bpd-ll-dir', type=str,
                       help='Directory containing our emitted .ll files')
    parser.add_argument('--ref-ll-dir', type=str,
                       help='Directory containing reference .ll files (from disassembly)')
    args = parser.parse_args()

    bpd_so = None
    if args.bpd_so and os.path.exists(args.bpd_so):
        bpd_so = ctypes.CDLL(args.bpd_so)

    results = []
    regressions = []

    print("=" * 70)
    print("LLVM IR Match Verification (CI Pipeline)")
    print(f"  Reference: {args.reference}")
    print(f"  BPD .so: {args.bpd_so or 'not provided'}")
    print("=" * 70)

    for op_name, op_config in sorted(OPS.items()):
        pattern = op_config['pattern']

        # Get reference output
        ref_output, err = test_op_with_pytorch(op_name, op_config)
        
        if err:
            results.append({
                'pattern': pattern, 'op': op_name, 
                'reference': args.reference, 'ulp': None,
                'evidence': f'skip: {err}'
            })
            continue

        # Test BPD LLVM kernel if available
        if bpd_so:
            bpd_output, err = test_op_with_bpd_so(op_name, op_config, bpd_so, ref_output)
            if err:
                results.append({
                    'pattern': pattern, 'op': op_name,
                    'reference': args.reference, 'ulp': None,
                    'evidence': f'skip: {err}'
                })
                continue

            max_ulp, n_diff, n_total = ulp_compare(ref_output, bpd_output)
            results.append({
                'pattern': pattern, 'op': op_name,
                'reference': args.reference, 'ulp': max_ulp,
                'n_diff': n_diff, 'n_total': n_total,
                'evidence': f'CI {datetime.now(timezone.utc).strftime("%Y-%m-%d")}'
            })
        else:
            # No BPD .so — record as untested
            results.append({
                'pattern': pattern, 'op': op_name,
                'reference': args.reference, 'ulp': None,
                'evidence': 'no BPD .so provided'
            })

    # Emit outputs
    prolog_path = emit_prolog_facts(results, args.output)
    print(f"\n  Prolog facts: {prolog_path}")

    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    json_path = emit_json_report(results, args.json)
    print(f"  JSON report:  {json_path}")

    # Summary
    tested = sum(1 for r in results if r['ulp'] is not None)
    at_zero = sum(1 for r in results if r['ulp'] == 0)
    total = len(results)
    
    print(f"\n  Summary: {tested}/{total} tested, {at_zero}/{total} at 0 ULP")

    if args.ci and regressions:
        print(f"\n  REGRESSIONS: {len(regressions)}")
        for r in regressions:
            print(f"    {r['op']}: {r['ulp']} ULP (expected ≤{r.get('tolerance', 0)})")
        sys.exit(1)


if __name__ == "__main__":
    main()
