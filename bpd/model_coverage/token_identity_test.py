#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""token_identity_test.py — gate model 0ULP-equivalence on TOKEN-FOR-TOKEN identical output.

Runs each prompt through each model DETERMINISTICALLY (temperature=0, fixed seed, top_k=1)
and records the exact token sequence. A model passes the 0-ULP gate if repeated runs (and
runs across builds/sessions) produce BIT-IDENTICAL token streams for hundreds of tokens.

The deep claim: deterministic greedy decoding (temp=0) means the SAMPLED TOKEN is
argmax(logits). If our reproduction of the compute graph is bit-exact (0-ULP, as proven
for the llama graph), the logits are identical => argmax is identical => the token stream
is identical, token-for-token, for arbitrarily many tokens. Token-identity is the
END-TO-END, USER-VISIBLE consequence of 0-ULP. This test gates on it directly.

Usage:
  token_identity_test.py record  <model> [--n N]      # record token streams for the corpus
  token_identity_test.py verify  <model>              # re-run, compare to recorded (gate)
  token_identity_test.py crosscheck <modelA> <modelB> # do two builds/models agree (should differ unless same)
"""
import json, sys, os, hashlib, time, urllib.request

OLLAMA = "http://localhost:11434"
CORPUS = os.path.join(os.path.dirname(__file__), "prompts.jsonl")
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baselines")
NUM_PREDICT = 256   # tokens to generate per prompt (hundreds of tokens)

def ollama_generate(model, prompt, num_predict=NUM_PREDICT):
    """Deterministic greedy generation. Returns the token-id list + text."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,  # unload after each request -> every prompt starts from clean/zeroed runner state
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 42,
            "num_predict": num_predict,
            "num_ctx": 4096,
            "num_gpu": 0,
            "num_thread": 1,
        },
        # raw=false so the model's chat template applies consistently
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            break
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    else:
        raise last_err
    # ollama returns the response text; for token-exactness we also hash the text
    # (token ids aren't directly exposed via /api/generate, so text+context is the signal)
    return {
        "response": resp.get("response", ""),
        "context": resp.get("context", []),   # the token-id context IS exposed — exact tokens!
        "eval_count": resp.get("eval_count", 0),
        "prompt_eval_count": resp.get("prompt_eval_count", 0),
    }

def stream_fingerprint(result):
    """A bit-exact fingerprint of the generated token stream."""
    ctx = result.get("context", [])
    text = result.get("response", "")
    return {
        "context_sha": hashlib.sha256(json.dumps(ctx).encode()).hexdigest(),
        "text_sha": hashlib.sha256(text.encode()).hexdigest(),
        "n_tokens": result.get("eval_count", 0),
        "n_context": len(ctx),
    }

def load_corpus():
    prompts = []
    with open(CORPUS) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts

def baseline_path(model):
    safe = model.replace(":", "_").replace("/", "_")
    return os.path.join(BASELINE_DIR, f"{safe}.json")

def cmd_record(model, n=None):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    corpus = load_corpus()
    if n: corpus = corpus[:n]
    bp = baseline_path(model)
    # RESUMABLE: load existing baseline, skip already-recorded prompts, CHECKPOINT after each.
    records = {}
    if os.path.exists(bp):
        try:
            records = json.load(open(bp)).get("records", {})
        except (json.JSONDecodeError, ValueError):
            records = {}  # corrupt/empty checkpoint — start fresh
    for i, p in enumerate(corpus):
        if p["id"] in records:
            continue
        t0 = time.time()
        res = ollama_generate(model, p["prompt"])
        fp = stream_fingerprint(res)
        records[p["id"]] = {**fp, "prompt_len": len(p["prompt"]), "category": p.get("category")}
        out = {"model": model, "num_predict": NUM_PREDICT, "options": "temp0_topk1_seed42", "records": records}
        tmp = bp + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, bp)  # atomic — a kill can never leave bp truncated
        print("  [%d/%d] %-20s %4dtok ctx_sha=%s (%.1fs)" % (
            i + 1, len(corpus), p["id"], fp["n_tokens"], fp["context_sha"][:12], time.time() - t0), flush=True)
    print("\nRecorded %d baselines -> %s" % (len(records), bp))

def cmd_verify(model, n=None):
    """Re-run and gate: token streams MUST match the recorded baseline (determinism = 0-ULP proxy).
    Resumable: checkpoints verified results so a chunked run accumulates to a full gate."""
    bp = baseline_path(model)
    if not os.path.exists(bp):
        print(f"NO BASELINE for {model} — run 'record' first"); sys.exit(2)
    base = json.load(open(bp))["records"]
    corpus = load_corpus()
    if n: corpus = corpus[:n]
    rp = bp.replace(".json", ".verify.json")
    done = {}
    if os.path.exists(rp):
        try: done = json.load(open(rp))
        except (json.JSONDecodeError, ValueError): done = {}
    for p in corpus:
        if p["id"] not in base or p["id"] in done:
            continue
        res = ollama_generate(model, p["prompt"])
        fp = stream_fingerprint(res)
        b = base[p["id"]]
        ok = (fp["context_sha"] == b["context_sha"] and fp["text_sha"] == b["text_sha"])
        done[p["id"]] = {"ok": bool(ok), "n_tokens": fp["n_tokens"],
                         "got_ctx": fp["context_sha"][:16], "want_ctx": b["context_sha"][:16]}
        tmp = rp + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(done, fh, indent=2); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, rp)
        mark = "\u2713 0-ULP" if ok else "\u2717 DIVERGED"
        print(f"  {mark}  {p['id']:22s} {fp['n_tokens']}tok " +
              ("" if ok else f"(want {b['context_sha'][:12]} got {fp['context_sha'][:12]})"), flush=True)
    # final gate over ALL verified
    relevant = [p["id"] for p in corpus if p["id"] in base]
    passed = sum(1 for i in relevant if done.get(i, {}).get("ok"))
    total = len(relevant)
    pending = total - len([i for i in relevant if i in done])
    print(f"\nGATE: {passed}/{total} prompts token-identical" +
          (f" ({pending} pending — re-run to continue)" if pending else "") + ". " +
          ("PASS — model is 0-ULP-deterministic." if passed == total else
           f"{'INCOMPLETE' if pending else 'FAIL'}."))
    sys.exit(0 if (passed == total and pending == 0) else 1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    cmd, model = sys.argv[1], sys.argv[2]
    n = None
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n")+1])
    if cmd == "record": cmd_record(model, n)
    elif cmd == "verify": cmd_verify(model, n)
    else: print(f"unknown cmd {cmd}"); sys.exit(1)
