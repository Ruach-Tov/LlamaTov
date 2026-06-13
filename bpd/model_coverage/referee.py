#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""referee.py — the TOKEN-IDENTITY REFEREE: the gate for all Llama improvements.

Heath's mandate: not just per-op 0-ULP — actual MODEL OUTPUT. Our kernels vs Ollama,
same prompt -> same tokens. Every improvement (f32 KV-cache, 0-ULP fusions, perf) is
GATED by this referee. If tokens change, the improvement is REJECTED.

Reliability is paramount. The referee uses the state-isolation diagnosed earlier:
  keep_alive:0 -> ollama unloads the model after each request -> every prompt starts from
  a clean/zeroed runner state -> deterministic, order-independent token streams.

Usage:
  referee.py baseline   [--model M] [--n N]   # record the reference token streams (ground truth)
  referee.py check      [--model M] [--n N]   # candidate must reproduce baseline EXACTLY (the gate)
  referee.py selfcheck  [--model M] [--n N]   # determinism sanity: baseline==baseline (no candidate)

Exit 0 = PASS (all token-identical), 1 = FAIL (some prompt's tokens changed), 2 = setup error.
"""
import json, sys, os, hashlib, time, urllib.request, argparse

OLLAMA = "http://localhost:11434"
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "prompts.jsonl")
BASELINE_DIR = os.path.join(HERE, "baselines")
NUM_PREDICT = 256

def generate(model, prompt, retries=3):
    """Deterministic greedy generation from a CLEAN runner state (keep_alive:0).
    Returns the exact token-id context + reply. Retries on transient runner errors."""
    body = {
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": 0,  # unload after -> next prompt starts from zeroed/clean state
        "options": {
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 42,
            "num_predict": NUM_PREDICT, "num_ctx": 4096,
            "num_gpu": 0,      # CPU = the bit-exact ggml-cpu path
            "num_thread": 1,   # single-threaded = deterministic reductions
        },
    }
    req = urllib.request.Request(f"{OLLAMA}/api/generate",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            return resp.get("context", []), resp.get("response", ""), resp.get("eval_count", 0)
        except Exception as e:
            last = e; time.sleep(3 * (attempt + 1))
    raise last

def fingerprint(ctx, text):
    return {
        "ctx_sha": hashlib.sha256(json.dumps(ctx).encode()).hexdigest(),
        "text_sha": hashlib.sha256(text.encode()).hexdigest(),
    }

def load_corpus(n=None):
    prompts = [json.loads(l) for l in open(CORPUS) if l.strip()]
    return prompts[:n] if n else prompts

def baseline_path(model):
    return os.path.join(BASELINE_DIR, f"{model.replace(':','_').replace('/','_')}.referee.json")

def atomic_save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def cmd_baseline(model, n=None):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    corpus = load_corpus(n)
    bp = baseline_path(model)
    records = {}
    if os.path.exists(bp):
        try: records = json.load(open(bp)).get("records", {})
        except (json.JSONDecodeError, ValueError): records = {}
    for i, p in enumerate(corpus):
        if p["id"] in records:  # resumable
            continue
        t0 = time.time()
        ctx, text, ntok = generate(model, p["prompt"])
        records[p["id"]] = {**fingerprint(ctx, text), "n_tokens": ntok,
                            "category": p.get("category"), "len_class": p.get("len_class")}
        atomic_save(bp, {"model": model, "config": "temp0_topk1_seed42_gpu0_thread1_keepalive0",
                         "num_predict": NUM_PREDICT, "records": records})
        print(f"  [{i+1}/{len(corpus)}] {p['id']:22s} {ntok:4d}tok ctx={records[p['id']]['ctx_sha'][:12]} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nBASELINE recorded: {len(records)} prompts -> {bp}")

def cmd_check(model, n=None, label="CHECK"):
    bp = baseline_path(model)
    if not os.path.exists(bp):
        print(f"NO BASELINE for {model} — run 'baseline' first"); sys.exit(2)
    base = json.load(open(bp))["records"]
    corpus = load_corpus(n)
    rp = bp.replace(".referee.json", f".{label.lower()}.json")
    done = {}
    if os.path.exists(rp):
        try: done = json.load(open(rp))
        except (json.JSONDecodeError, ValueError): done = {}
    for p in corpus:
        if p["id"] not in base or p["id"] in done:
            continue
        ctx, text, ntok = generate(model, p["prompt"])
        fp = fingerprint(ctx, text)
        b = base[p["id"]]
        ok = fp["ctx_sha"] == b["ctx_sha"] and fp["text_sha"] == b["text_sha"]
        done[p["id"]] = {"ok": bool(ok), "n_tokens": ntok}
        atomic_save(rp, done)
        mark = "\u2713" if ok else "\u2717 TOKENS CHANGED"
        print(f"  {mark}  {p['id']:22s} {ntok}tok" + ("" if ok else f"  (want {b['ctx_sha'][:12]} got {fp['ctx_sha'][:12]})"), flush=True)
    relevant = [p["id"] for p in corpus if p["id"] in base]
    passed = sum(1 for i in relevant if done.get(i, {}).get("ok"))
    total = len(relevant)
    pending = total - sum(1 for i in relevant if i in done)
    print(f"\nREFEREE {label}: {passed}/{total} token-identical" +
          (f" ({pending} pending — re-run to continue)" if pending else "") + ". " +
          ("PASS \u2713" if passed == total and pending == 0 else
           ("INCOMPLETE" if pending else f"FAIL \u2717 ({total-passed} rejected)")))
    sys.exit(0 if (passed == total and pending == 0) else 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "check", "selfcheck"])
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "baseline": cmd_baseline(a.model, a.n)
    elif a.cmd == "check": cmd_check(a.model, a.n, "CHECK")
    elif a.cmd == "selfcheck": cmd_check(a.model, a.n, "SELFCHECK")
