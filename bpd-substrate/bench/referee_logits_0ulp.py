#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""referee_logits_0ulp.py — 0-ULP LOGIT-LEVEL end-to-end referee for llama vs ggml/ollama.

Compares the actual LOGIT BITS of our forward against ggml's (via llama-eval-callback dump),
catching sub-margin divergence that token-identity cannot see. Gates "bit-identical e2e".

INPUT IS NOW PARAMETRIZED and SINGLE-SOURCED so ggml and ours can't drift:
  --tokens "128000,9906,..."   exact token IDs (fed IDENTICALLY to both ggml and ours)
  --prompt "Hello, my name is" text prompt (tokenized via the eval-callback; the SAME
                                 leaf tokens it produces are then fed to our forward)
Prefers the token-input eval-callback (llama-eval-callback-tok, honors LLAMA_INPUT_TOKENS)
so both sides consume the EXACT same tokens. Falls back to text-prompt mode otherwise.

Usage:
  python3 bench/referee_logits_0ulp.py --so build/bpd_cpu.so --tokens 128000,9906,11
  python3 bench/referee_logits_0ulp.py --so X.so --prompt "The quick brown fox"
  BPD_CPU_SO=... python3 bench/referee_logits_0ulp.py            # env fallback, default prompt
Exit 0 = BIT_IDENTICAL logits, 1 = DIVERGENT.
"""
import os, sys, struct, subprocess, tempfile, glob, argparse
import numpy as np

DEFAULT_GGUF = "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
# token-input build (honors LLAMA_INPUT_TOKENS) preferred; text build is the fallback.
EVAL_CB_TOK  = "<repo>/eval_callback_patched/llama-eval-callback-tok"
EVAL_CB_TEXT = "/tmp/llama_cpp_test/build/bin/llama-eval-callback"
INFER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpd_llamatov_infer.py")

def load_ggml_bin(p):
    raw = open(p, "rb").read()
    nb = struct.unpack("<Q", raw[72:80])[0]
    return np.frombuffer(raw[80:80+nb], dtype=np.float32)

def read_leaf_tokens(dump_dir):
    """Recover the exact tokens ggml actually processed (from the dumped input leaf)."""
    for pat in ("0001_*src1.bin", "*leaf*src1.bin", "*inp_tokens*.bin"):
        g = sorted(glob.glob(os.path.join(dump_dir, pat)))
        if g:
            raw = open(g[0], "rb").read()
            nb = struct.unpack("<Q", raw[72:80])[0]
            toks = np.frombuffer(raw[80:80+nb], dtype=np.int32).tolist()
            if toks:
                return toks
    return None

def dump_ggml_logits(gguf, eval_cb, tokens=None, prompt=None, n_gen=1):
    """Run ggml, dump result_output. If tokens given + tok-build, feed exact tokens.
    Returns (logits, actual_tokens_ggml_used)."""
    d = tempfile.mkdtemp(prefix="ggml_ref_")
    env = dict(os.environ, LLAMA_DUMP_DIR=d, LD_LIBRARY_PATH=os.path.dirname(eval_cb))
    cmd = [eval_cb, "--model", gguf, "-n", str(n_gen), "--ctx-size", "512", "-ngl", "0", "--threads", "1"]
    if tokens is not None and eval_cb == EVAL_CB_TOK:
        env["LLAMA_INPUT_TOKENS"] = ",".join(str(t) for t in tokens)
        cmd += ["--prompt", "x"]   # placeholder; overridden by LLAMA_INPUT_TOKENS
    else:
        cmd += ["--prompt", prompt if prompt is not None else "Hello, my name is"]
    subprocess.run(cmd, capture_output=True, env=env, timeout=120)
    g = glob.glob(f"{d}/*result_output*.bin")
    if not g:
        raise RuntimeError("ggml did not dump result_output (check eval-callback path / args)")
    used = read_leaf_tokens(d)
    return load_ggml_bin(g[0]), used

def dump_our_logits(so, gguf, tokens, n_gen=1):
    out = tempfile.mktemp(suffix=".npy")
    subprocess.run(["python3", INFER, "--so", so, "--gguf", gguf,
                    "--tokens", ",".join(str(t) for t in tokens), "--n-generate", str(n_gen),
                    "--dump-logits", out, "--out", "/tmp/_ref.json"],
                   capture_output=True, env=dict(os.environ, OMP_NUM_THREADS="1"), timeout=120)
    p = out.replace(".npy", "_step0.npy")
    return np.load(p).astype(np.float32).reshape(-1)

def main():
    ap = argparse.ArgumentParser(description="0-ULP logit referee (parametrized input)")
    ap.add_argument("--so", default=os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so"))
    ap.add_argument("--gguf", default=os.environ.get("GGUF", DEFAULT_GGUF))
    ap.add_argument("--tokens", default=os.environ.get("TOKENS"),
                    help="exact token IDs, comma-separated (fed identically to ggml+ours)")
    ap.add_argument("--prompt", default=os.environ.get("PROMPT"),
                    help="text prompt (tokenized by ggml; its leaf tokens drive ours)")
    ap.add_argument("--eval-callback", default=None,
                    help="override eval-callback path (default: token-build if --tokens, else text)")
    ap.add_argument("--n-generate", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    # Resolve input to a single source of truth.
    tokens = None
    if a.tokens:
        tokens = [int(t) for t in a.tokens.split(",") if t.strip() != ""]
    # choose eval-callback: token-build when we have explicit tokens and it exists
    if a.eval_callback:
        eval_cb = a.eval_callback
    elif tokens is not None and os.path.exists(EVAL_CB_TOK):
        eval_cb = EVAL_CB_TOK
    else:
        eval_cb = EVAL_CB_TEXT
    # default input if neither given
    if tokens is None and a.prompt is None:
        a.prompt = "Hello, my name is"

    # Run ggml first; if prompt-mode, recover the EXACT tokens it used so ours matches.
    ggml, ggml_tokens = dump_ggml_logits(a.gguf, eval_cb, tokens=tokens, prompt=a.prompt, n_gen=a.n_generate)
    if tokens is None:
        if ggml_tokens is None:
            raise RuntimeError("prompt-mode: could not recover ggml's leaf tokens to drive ours")
        tokens = ggml_tokens   # SINGLE SOURCE: ours uses exactly what ggml tokenized
    elif ggml_tokens is not None and ggml_tokens[:len(tokens)] != tokens[:len(ggml_tokens)]:
        if not a.quiet:
            print(f"WARNING: ggml processed tokens {ggml_tokens} != requested {tokens} "
                  f"(token-build may not have engaged)")

    ours = dump_our_logits(a.so, a.gguf, tokens, n_gen=a.n_generate)
    n = min(len(ours), len(ggml)); ours, ggml = ours[:n], ggml[:n]
    u = np.abs(ours.view(np.int32).astype(np.int64) - ggml.view(np.int32).astype(np.int64))
    maxulp = int(u.max()); ndiff = int((u != 0).sum())
    maxabs = float(np.abs(ours.astype(np.float64) - ggml).max())
    verdict = "BIT_IDENTICAL" if maxulp == 0 else "DIVERGENT"
    tok_show = tokens[:8] + (["..."] if len(tokens) > 8 else [])
    print(f"LOGIT REFEREE (vs ggml): maxULP={maxulp} ndiff={ndiff}/{n} maxabs={maxabs:.3e} "
          f"argmax ours={int(ours.argmax())} ggml={int(ggml.argmax())}  {verdict}"
          + ("" if a.quiet else f"  [tokens={tok_show}]"))
    return 0 if maxulp == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
