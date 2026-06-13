#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
logit_gate.py — two-tier logit-level correctness gate for kernel fusions.
Compares stock vs variant on the 6 prompt strata, at DECODE (not just prefill),
classifying each as GREEN (logits bitwise identical) / YELLOW (tokens match, logits
differ) / RED (tokens differ). This closes the hole that hid the SoA-decode-broken-since-v2
bug: the old gate compared PREFILL logit sums; this compares DECODE-generated tokens + logits.

Usage:
  logit_gate.py --stock /path/stock/llama-cli --variant /path/variant/llama-cli \
                --model GGUF --variant-env GGML_CUDA_Q8_0_SOA=1 [--n-predict 32]

Requires the binaries built with a logit dump: --logit-dump FILE writes, per generated
token, the full fp32 logit vector (the entry point medayek is requesting from mavchin).
Falls back to token-only comparison (RED/GREEN-token) if no logit dump available.
"""
import argparse, subprocess, sys, os, struct, json, tempfile, hashlib

# The 6 prompt strata (medayek's stratified coverage across the activation space)
STRATA = {
    "minimal":      "Hello",
    "code":         "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "multilingual": "\u05e9\u05dc\u05d5\u05dd \u4f60\u597d \u3053\u3093\u306b\u3061\u306f m%s\U0001f600 the capital of France is",
    "long_context": "In the beginning " * 64 + "the most important thing to remember is",
    "repetitive":   "the the the the the the the the the the the the the the the",
    "adversarial":  "qux\x00zZ\u200b 99999 \U0001f9ea\u2603 ;;;; \\x41\\x42 \u05d0\u05d1\u05d2\u05d3 \t\t end",
}

def run(binary, model, prompt, n_predict, env_extra, logit_dump=None, extra_ld=None):
    """Run generation, return (tokens_text, logit_dump_path_or_None)."""
    env = dict(os.environ)
    if env_extra:
        for kv in env_extra:
            k, _, v = kv.partition("=")
            env[k] = v
    _bindir=os.path.dirname(binary)
    _parts=["/run/opengl-driver/lib", _bindir]
    if extra_ld: _parts.append(extra_ld)
    _parts.append(env.get("LD_LIBRARY_PATH",""))
    env["LD_LIBRARY_PATH"]=":".join(p for p in _parts if p)
    dump = logit_dump and tempfile.mktemp(suffix=".logits")
    cmd = [binary, "-m", model, "-ngl", "99", "-n", str(n_predict),
           "-p", prompt, "--temp", "0", "-c", "512", "-no-cnv"]
    if dump:
        cmd += ["--logit-dump", dump]   # the entry point requested from mavchin
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"crashed": True, "reason": "TIMEOUT (hung)", "text": "", "rc": None}, None
    # Detect crash: nonzero/negative exit (signal), core dump, CUDA error, or empty output that should not be empty.
    rc = r.returncode
    stderr = r.stderr or ""
    crashed = False; reason = ""
    if rc is None or rc < 0:                       # killed by signal (e.g. -6 SIGABRT, -11 SIGSEGV)
        crashed = True; reason = "SIGNAL %s" % rc
    elif rc != 0:                                   # nonzero exit
        crashed = True; reason = "EXIT %d" % rc
    elif "dumped core" in stderr or "Segmentation" in stderr:
        crashed = True; reason = "CORE DUMP"
    elif "CUDA error" in stderr or "an illegal memory access" in stderr or "cudaError" in stderr:
        crashed = True; reason = "CUDA ERROR"
    elif "CUDA driver is a stub" in stderr:
        crashed = True; reason = "CPU STUB (LD_LIBRARY_PATH missing driver)"
    elif not (r.stdout or "").strip():              # empty output where tokens were expected = not a valid result
        crashed = True; reason = "EMPTY OUTPUT (no tokens produced)"
    return {"crashed": crashed, "reason": reason, "text": r.stdout or "", "rc": rc,
            "stderr_tail": stderr[-300:]}, (dump if dump and os.path.exists(dump) else None)

def read_logits(path):
    """Read per-token fp32 logit vectors: [n_tokens][vocab] as raw float32. Returns bytes-per-token list."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    # format: int32 n_tokens, int32 vocab, then n_tokens*vocab float32
    if len(data) < 8:
        return None
    n_tok, vocab = struct.unpack("<ii", data[:8])
    body = data[8:]
    stride = vocab * 4
    return [body[i*stride:(i+1)*stride] for i in range(n_tok)]

def classify(stock_run, var_run, stock_logits, var_logits):
    # A CRASH is never a pass and never a fail - it is an ERROR, surfaced distinctly so it
    # can NEVER be mistaken for test results. (Two empty/crashed outputs must NOT compare as identical.)
    if stock_run.get("crashed"):
        return "ERROR", "STOCK run crashed: " + stock_run.get("reason", "?") + " (baseline invalid - cannot compare)"
    if var_run.get("crashed"):
        return "ERROR", "VARIANT run CRASHED: " + var_run.get("reason", "?") + " (NOT a pass - the kernel faulted)"
    stock_txt = stock_run["text"]; var_txt = var_run["text"]
    """Two-tier gate: GREEN (logits bit-identical) / YELLOW (tokens match, logits differ) / RED."""
    # extract just the generated continuation (after the prompt echo) by comparing token text
    tokens_match = (stock_txt == var_txt)
    if not tokens_match:
        return "RED", "tokens differ"
    if stock_logits is None or var_logits is None:
        return "GREEN-token", "tokens identical (no logit dump - logit-identity UNVERIFIED)"
    if len(stock_logits) != len(var_logits):
        return "YELLOW", f"token count differs in logits ({len(stock_logits)} vs {len(var_logits)})"
    for i, (a, b) in enumerate(zip(stock_logits, var_logits)):
        if a != b:
            # find first differing float for diagnostics
            return "YELLOW", f"tokens match but logit vector differs at gen-token {i}"
    return "GREEN", "logit vectors bitwise identical (certified bit-identical)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant-env", action="append", default=[])
    ap.add_argument("--stock-env", action="append", default=[])
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--logit-dump", action="store_true", help="request per-token logit dump from binaries")
    ap.add_argument("--extra-ld", default=None)
    ap.add_argument("--strata", nargs="*", default=list(STRATA.keys()))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = {}
    overall = "GREEN"
    rank = {"GREEN":0, "GREEN-token":1, "YELLOW":2, "RED":3, "FALSE-PASS":4, "ERROR":5}
    for name in args.strata:
        prompt = STRATA[name]
        s_run, s_lg = run(args.stock, args.model, prompt, args.n_predict, args.stock_env, args.logit_dump, args.extra_ld)
        v_run, v_lg = run(args.variant, args.model, prompt, args.n_predict, args.variant_env, args.logit_dump, args.extra_ld)
        sl = read_logits(s_lg); vl = read_logits(v_lg)
        verdict, reason = classify(s_run, v_run, sl, vl)
        results[name] = {"verdict": verdict, "reason": reason}
        if rank.get(verdict, 5) > rank.get(overall, 0):
            overall = verdict
        mark = {"GREEN":"\u2705","GREEN-token":"\U0001f7e2","YELLOW":"\u26a0\ufe0f","RED":"\u274c","FALSE-PASS":"\U0001f6a8","ERROR":"\U0001f4a5"}.get(verdict,"?")
        print(f"  {mark} {name:14s} {verdict:12s} {reason}")
    print(f"\n  OVERALL: {overall}  (gate passes only on GREEN across ALL strata at DECODE)")
    if args.json:
        print(json.dumps({"overall": overall, "strata": results}, indent=2))
    sys.exit(0 if overall == "GREEN" else 1)

if __name__ == "__main__":
    main()
