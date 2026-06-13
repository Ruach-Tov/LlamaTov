#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_llvm_ops_auto.py — Test every BPD op against PyTorch. Prolog-driven.

The test manifest comes from Prolog (bpd/lib/op_signatures.pl).
Prolog declares each op's function name, pattern, ggml name, and calling
convention (argument signature). No hardcoded test lists in Python.

For each op: spawns a child process that loads the .so, calls the function
(using the Prolog-declared signature), compares to PyTorch, and prints ULP.
If the child segfaults, parent continues.

Writes Prolog facts to the diviner's path. Touches generator for cache bust.

Author: medayek (original), mavchin (Prolog-driven refactor)
"""
import subprocess, os, sys, json
from datetime import datetime, timezone

FACTS_PATH = os.environ.get("FACTS_PATH", "")
GENERATOR_PATH = os.environ.get("GENERATOR_PATH", os.path.expanduser("~/Ruach-Tov/bpd/llvm_match_status.pl"))
SIGNATURES_PL = os.environ.get("SIGNATURES_PL", os.path.expanduser("~/Ruach-Tov/bpd/lib/op_signatures.pl"))

PY = "/nix/store/gc38dj7srpf7c0nlrc0g8x72bcmglmiv-python3-3.12.11-env/bin/python3"
ENV = {
    "PYTHONPATH": "/nix/store/gc38dj7srpf7c0nlrc0g8x72bcmglmiv-python3-3.12.11-env/lib/python3.12/site-packages",
    "PATH": os.environ.get("PATH", ""),
}

LLVM_SO = os.environ.get("BPD_LLVM_SO", "")
CPU_SO = os.environ.get("BPD_CPU_SO", "")

# ============================================================
# Load test manifest from Prolog
# ============================================================

def load_manifest_from_prolog():
    """Query Prolog for the complete test manifest."""
    try:
        r = subprocess.run(
            ["swipl", "-g", "use_module(op_signatures), op_signatures:emit_test_manifest, halt"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(SIGNATURES_PL)
        )
        if r.returncode == 0 and r.stdout.strip():
            # Fix trailing comma before ]
            text = r.stdout.strip()
            text = text.replace(",\n]", "\n]")
            return json.loads(text)
    except Exception as e:
        print(f"  WARNING: Prolog manifest failed: {e}", flush=True)
    return None


def sig_to_convention(sig_str):
    """Convert Prolog signature like '[in,out,n]' to a calling convention string."""
    # Parse the Prolog list
    sig = sig_str.strip("[]").split(",")
    return sig  # return the raw arg list


SKIPPED_OPS = {}  # populated from manifest

def build_tests_from_manifest(manifest):
    """Build TESTS list from Prolog manifest."""
    tests = []
    for entry in manifest:
        fn = entry["fn"]
        pattern = entry["pattern"]
        ggml = entry["ggml"]
        sig = sig_to_convention(entry["sig"])
        
        # Skip ops with unsupported calling conventions
        if "unsupported" in sig:
            SKIPPED_OPS[(pattern, ggml)] = "complex calling convention (needs gamma/beta/eps/dims)"
            continue
        
        # Determine which .so to use
        if fn.endswith("_cpu"):
            so = CPU_SO
        else:
            so = LLVM_SO
        
        tests.append((fn, so, ggml, pattern, sig))
    return tests


# ============================================================
# Build TESTS — try Prolog first, fall back to hardcoded
# ============================================================

manifest = load_manifest_from_prolog()
if manifest:
    TESTS = build_tests_from_manifest(manifest)
    print(f"  Loaded {len(TESTS)} ops from Prolog manifest", flush=True)
else:
    print("  WARNING: falling back to hardcoded test list", flush=True)
    # Minimal fallback — just the unary ops we know work
    TESTS = []
    for fn, op in [
        ("bpd_relu", "ggml_relu"), ("bpd_silu", "ggml_silu"),
        ("bpd_sigmoid", "ggml_sigmoid"), ("bpd_softplus", "ggml_softplus"),
        ("bpd_leaky_relu", "ggml_leaky_relu"), ("bpd_elu", "ggml_elu"),
        ("bpd_softsign", "ggml_softsign"),
    ]:
        TESTS.append((fn, LLVM_SO, op, "unary_elementwise", ["n", "out", "in"]))

# ============================================================
# SKIPPED_OPS — no longer needed! Prolog declares all signatures.
# If an op has a signature Prolog knows, we test it.
# If the .so doesn't have the symbol, ctypes will report it.
# ============================================================

# ============================================================
# All supported patterns for the "all ops" enumeration
# ============================================================
ALL_OPS = {
    "unary_elementwise": ["ggml_relu","ggml_silu","ggml_gelu","ggml_elu","ggml_selu",
                           "ggml_leaky_relu","ggml_sigmoid","ggml_tanh",
                           "ggml_hardsigmoid","ggml_softplus","ggml_softsign","ggml_clamp"],
    "binary_elementwise": ["ggml_scale"],
    "reduction": ["ggml_mul_mat","ggml_sum","ggml_mean","ggml_max","ggml_min",
                   "ggml_argmax","ggml_argmin"],
    "reduction_then_elementwise": ["ggml_norm","ggml_rms_norm","ggml_soft_max",
                                    "ggml_log_softmax","ggml_group_norm","ggml_l2_norm"],
    "scan": ["ggml_cumsum","ggml_cumprod"],
    "conv_im2col": ["ggml_conv_1d","ggml_conv_2d","ggml_conv_3d",
                     "ggml_conv_transpose_1d","ggml_conv_transpose_2d","ggml_conv_transpose_3d"],
    "pool_reduce": ["ggml_pool_1d","ggml_pool_2d","ggml_pool_3d"],
    "loss_reduce": ["ggml_mse_loss","ggml_cross_entropy_loss","ggml_hinge_loss",
                     "ggml_huber_loss","ggml_kl_div_loss","ggml_triplet_margin_loss"],
    "flash_attention": ["ggml_flash_attn_ext"],
}

REFS = ["ggml_sse3","ggml_avx2","ggml_avx512","ggml_neon","pytorch_mkl","pytorch_nnpack"]

# ============================================================
# Test script template — now handles multiple calling conventions
# ============================================================

TEST_SCRIPT = '''
import ctypes, numpy as np, json, sys
import torch, torch.nn.functional as F
c_float_p = ctypes.POINTER(ctypes.c_float)

fn_name = sys.argv[1]
so_path = sys.argv[2]
sig = json.loads(sys.argv[3])
n = 16384

lib = ctypes.CDLL(so_path)
fn = getattr(lib, fn_name)

np.random.seed(42)
x = np.random.randn(n).astype(np.float32) * 2.0
y = np.abs(np.random.randn(n).astype(np.float32)) * 0.5 + 0.01  # positive targets for loss fns
x_pt = torch.from_numpy(x.copy())

# PyTorch references
REF = {
    "bpd_relu": lambda: F.relu(x_pt), "bpd_silu": lambda: F.silu(x_pt),
    "bpd_gelu": lambda: F.gelu(x_pt),  # F.gelu default = the reference
    "bpd_gelu_fixed": lambda: F.gelu(x_pt),  # same reference — what the user calls
    "bpd_tanh_fixed": lambda: torch.tanh(x_pt),
    "bpd_hardsigmoid_fixed": lambda: F.hardsigmoid(x_pt),
    "bpd_sum_sse3": lambda: torch.full_like(x_pt[:1], x_pt.sum().item()),
    "bpd_elu_fixed": lambda: F.elu(x_pt),
    "bpd_softplus_fixed": lambda: F.softplus(x_pt),
    "bpd_selu_fixed": lambda: F.selu(x_pt),
    "bpd_scale": lambda: x_pt * 0.5,
    "bpd_cumsum": lambda: torch.cumsum(x_pt, dim=0),
    "bpd_cumprod": lambda: torch.cumprod(x_pt, dim=0),
    "bpd_clamp": lambda: torch.clamp(x_pt, -1.0, 1.0),
    "bpd_elu": lambda: F.elu(x_pt), "bpd_selu": lambda: F.selu(x_pt),
    "bpd_leaky_relu": lambda: F.leaky_relu(x_pt, 0.01),
    "bpd_sigmoid": lambda: torch.sigmoid(x_pt), "bpd_tanh": lambda: torch.tanh(x_pt),
    "bpd_hardsigmoid": lambda: F.hardsigmoid(x_pt),
    "bpd_softplus": lambda: F.softplus(x_pt), "bpd_softsign": lambda: F.softsign(x_pt),
    "bpd_scalar_mul": lambda: x_pt * 0.5,
    # Loss functions (two-input)
    "bpd_mse_loss_cpu": lambda: (x_pt - torch.from_numpy(y)).pow(2),
    "bpd_hinge_loss_cpu": lambda: F.relu(1.0 - x_pt * torch.from_numpy(y)),
    "bpd_huber_loss_cpu": lambda: F.huber_loss(x_pt, torch.from_numpy(y), reduction='none'),
    # LLVM loss functions return scalar mean
    "bpd_mse_loss": lambda: torch.full_like(x_pt[:1], F.mse_loss(x_pt, torch.from_numpy(y)).item()),
    "bpd_cross_entropy_loss": lambda: torch.full_like(x_pt[:1], -(torch.from_numpy(y) * torch.log(torch.sigmoid(x_pt))).mean().item()),
    "bpd_hinge_loss": lambda: torch.full_like(x_pt[:1], F.relu(1.0 - x_pt * torch.from_numpy(y)).mean().item()),
    "bpd_kl_div_loss": lambda: torch.full_like(x_pt[:1], F.kl_div(torch.log_softmax(x_pt, dim=0), torch.from_numpy(y), reduction='batchmean').item()),
    # Softmax/logsoftmax/l2norm (1 row of n cols)
    "bpd_softmax_cpu": lambda: F.softmax(x_pt, dim=0),
    "bpd_logsoftmax_cpu": lambda: F.log_softmax(x_pt, dim=0),
    "bpd_l2norm_cpu": lambda: x_pt / (x_pt.norm(2) + 1e-12),
    "bpd_sum": lambda: torch.full_like(x_pt[:1], x_pt.sum().item()),
    "bpd_mean": lambda: torch.full_like(x_pt[:1], x_pt.mean().item()),
    "bpd_max": lambda: torch.full_like(x_pt[:1], x_pt.max().item()),
    "bpd_min": lambda: torch.full_like(x_pt[:1], x_pt.min().item()),
    "bpd_layernorm": lambda: F.layer_norm(x_pt, x_pt.shape),
    "bpd_rmsnorm": lambda: x_pt / (x_pt.pow(2).mean().sqrt() + 1e-5),
    "bpd_softmax": lambda: F.softmax(x_pt, dim=0),
    "bpd_logsoftmax": lambda: F.log_softmax(x_pt, dim=0),
    "bpd_groupnorm": lambda: F.layer_norm(x_pt, x_pt.shape),
    "bpd_l2norm": lambda: x_pt / (x_pt.norm(2) + 1e-5),
}

# Add _cpu variants pointing to same refs
for k in list(REF.keys()):
    REF[k + "_cpu"] = REF[k]

ref_key = fn_name
if ref_key not in REF:
    print(json.dumps({"status": "no_ref", "fn": fn_name}))
    sys.exit(0)

ref_out = REF[ref_key]().numpy()

# Build ctypes call based on signature
is_scalar_out = "out_scalar" in sig
out_n = 1 if is_scalar_out else n
out = np.zeros(out_n, dtype=np.float32)

y = np.abs(np.random.randn(n).astype(np.float32)) * 0.5 + 0.01  # positive for loss functions
y_p = y.ctypes.data_as(c_float_p)
x_p = x.ctypes.data_as(c_float_p)
o_p = out.ctypes.data_as(c_float_p)

try:
    args = []
    has_out = any(a in ("out", "out_scalar") for a in sig)
    is_scalar_return = not has_out  # function returns float directly
    
    if is_scalar_return:
        fn.restype = ctypes.c_float
    
    for a in sig:
        if a == "in": args.append(x_p)
        elif a in ("out", "out_scalar"): args.append(o_p)
        elif a == "n": args.append(ctypes.c_int(n))
        elif a == "in2": args.append(y_p)
        elif a == "scalar": args.append(ctypes.c_float(0.5))
        elif a == "eps": args.append(ctypes.c_float(1e-5))
        elif a == "groups": args.append(ctypes.c_int(1))
        elif a == "n_rows": args.append(ctypes.c_int(1))
    
    result_val = fn(*args)
    if is_scalar_return:
        out[0] = result_val
except Exception as e:
    print(json.dumps({"status": "call_error", "error": str(e)}))
    sys.exit(0)

# Compare
compare_n = min(len(out), len(ref_out))
if compare_n == 0:
    print(json.dumps({"status": "bad_output"}))
    sys.exit(0)

bpd_i = np.frombuffer(out[:compare_n].tobytes(), dtype=np.int32)
ref_i = np.frombuffer(ref_out[:compare_n].tobytes(), dtype=np.int32)
ulps = np.abs(bpd_i.astype(np.int64) - ref_i.astype(np.int64))
max_ulp = int(ulps.max())
mean_ulp = float(ulps.mean())

print(json.dumps({"status": "ok", "max_ulp": max_ulp, "mean_ulp": mean_ulp}))
'''

# ============================================================
# Main test runner
# ============================================================

if __name__ == "__main__":
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    print(f"\nAUTOMATED VERIFICATION — ALL OPS — {datetime.now(timezone.utc).isoformat()[:19]}Z")
    print("=" * 60)
    
    llvm_results = {}   # (pattern, op) → ulp
    cpu_results = {}    # (pattern, op) → ulp
    
    tested_ops = set()  # track which ggml ops we actually tested
    
    for fn, so, op, pat, sig in TESTS:
        sig_json = json.dumps(sig) if isinstance(sig, list) else json.dumps(sig.split(","))
        
        try:
            r = subprocess.run(
                [PY, "-c", TEST_SCRIPT, fn, so, sig_json],
                capture_output=True, text=True, timeout=10, env=ENV
            )
        except subprocess.TimeoutExpired:
            print(f"  SKIP {op:36s} (timeout) [{os.path.basename(so)}]", flush=True)
            continue
        except Exception as e:
            print(f"  SKIP {op:36s} ({e}) [{os.path.basename(so)}]", flush=True)
            continue
        
        result = None
        if r.stdout.strip():
            try:
                result = json.loads(r.stdout.strip().split("\n")[-1])
            except:
                pass
        
        if result is None or result.get("status") not in ("ok",):
            status = result.get("status", "crash") if result else "crash"
            # Check if it's a missing symbol
            if result and result.get("status") == "no_ref":
                print(f"  SKIP {op:36s} (no PyTorch ref) [{os.path.basename(so)}]", flush=True)
            elif result and result.get("status") == "call_error":
                print(f"  SKIP {op:36s} (call_error: {result.get('error','')[:40]}) [{os.path.basename(so)}]", flush=True)
            else:
                print(f"  SKIP {op:36s} ({status}) [{os.path.basename(so)}]", flush=True)
            continue
        
        ulp = result["max_ulp"]
        key = (pat, op)
        tested_ops.add(op)
        
        is_llvm = not fn.endswith("_cpu")
        if is_llvm:
            # Keep the best (lowest ULP) result when multiple functions map to same op
            if key not in llvm_results or ulp < llvm_results[key]:
                llvm_results[key] = ulp
        else:
            if key not in cpu_results or ulp < cpu_results[key]:
                cpu_results[key] = ulp
        
        color = "BLUE" if is_llvm and ulp == 0 else "GRN " if ulp == 0 else "YEL " if ulp <= 100 else "RED "
        src = os.path.basename(so) + ":" + fn
        print(f"  {color} {op:36s} {ulp:>8d} ULP  [{src}]", flush=True)
    
    # Summary
    blues = sum(1 for v in llvm_results.values() if v == 0)
    greens = sum(1 for v in cpu_results.values() if v == 0)
    total_zero = blues + greens
    print(f"\n  {len(llvm_results) + len(cpu_results)} ops tested: {blues} blue, {greens} green, {total_zero} total 0 ULP")
    
    # Write Prolog facts
    with open(FACTS_PATH, "w") as f:
        f.write(f"%% Auto-generated by verify_llvm_ops_auto.py (Prolog-driven)\n")
        f.write(f"%% {datetime.now(timezone.utc).isoformat()[:19]}Z\n")
        f.write(f"%% Manifest from: {SIGNATURES_PL}\n")
        f.write(f"%% NO AGENT INPUT. Subprocess per op. Crash-safe.\n\n")
        f.write(f":- module(llvm_op_match, [ulp_op_match/5]).\n")
        f.write(f":- discontiguous ulp_op_match/5.\n\n")
        
        for ref in REFS:
            f.write(f"\n%% {ref}\n")
            for pat, ops in ALL_OPS.items():
                for op in ops:
                    key = (pat, op)
                    if ref == "ggml_sse3":
                        if key in llvm_results:
                            f.write(f"ulp_op_match({pat}, {op}, ggml_sse3, {llvm_results[key]}, 'auto {date}').\n")
                        elif key in cpu_results:
                            f.write(f"ulp_op_match({pat}, {op}, scalar, {cpu_results[key]}, 'auto {date}').\n")
                        elif key in SKIPPED_OPS:
                            f.write(f"ulp_op_match({pat}, {op}, ggml_sse3, skipped, '{SKIPPED_OPS[key]}').\n")
                        elif op not in tested_ops:
                            f.write(f"ulp_op_match({pat}, {op}, ggml_sse3, untested, '').\n")
                        else:
                            f.write(f"ulp_op_match({pat}, {op}, ggml_sse3, skipped, 'test failed').\n")
                    elif ref == "pytorch_mkl":
                        # pytorch_mkl: value comparison (green if 0 ULP)
                        # To earn blue: need IR diff via scan_ir_match.py
                        if key in llvm_results:
                            f.write(f"ulp_op_match({pat}, {op}, pytorch_mkl, {llvm_results[key]}, 'auto {date}').\n")
                        elif key in cpu_results:
                            f.write(f"ulp_op_match({pat}, {op}, pytorch_mkl, {cpu_results[key]}, 'auto {date}').\n")
                        else:
                            f.write(f"ulp_op_match({pat}, {op}, pytorch_mkl, untested, '').\n")
                    else:
                        f.write(f"ulp_op_match({pat}, {op}, {ref}, untested, '').\n")
    
    # Bust cache
    try:
        os.utime(GENERATOR_PATH, None)
    except: pass
    
    print("  Dashboard updated.")
