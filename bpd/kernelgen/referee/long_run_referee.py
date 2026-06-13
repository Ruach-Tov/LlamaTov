#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""long_run_referee.py — at-scale token-exactness + margin/drift instrument (pillar C core).

WHAT IT MEASURES (the three reports):
  1. TOKEN AGREEMENT: A-config vs B-config generations across VARIED prompts, N tokens each.
  2. MARGIN STATISTICS: per-step top1-top2 margin; min/median; and when dual-config,
     the SAFETY FACTOR = min(margin) / max(drift) — "tokens matched WITH ROOM TO SPARE".
  3. NEAR-FLIP CENSUS: every step where margin < 5x drift, with coordinates
     (prompt, step, top-2 tokens, margin, drift) — forensics pre-collected for any future flip.

MODES:
  single-config: margins only (fragility audit of one path).
  dual-config:   the migration/fusion certifier (e.g. canonical-rms vs torch-order,
                 fused vs unfused, q8-vs-fp32 as the DECLARED quantization class).

DISCIPLINES HONORED (REFERENCE_DISCIPLINE.md):
  (i) path-matching: each arm records a path attestation; cross-arm comparison
      refuses on mismatch UNLESS the mismatch is the DECLARED experiment variable.
  (ii) quant-matching: comparing across quantizations is SOFT by construction here —
      this instrument MEASURES the variance class, it never calls it a bug.

Arms run as subprocesses (cold start, documented seed only — capture-state provenance).

Usage (enclave):
  python3 long_run_referee.py --config-a "" --config-b "BPD_RMS_BLOCKROW=1" \
      --n-tokens 30 --json /tmp/longrun.json
  python3 long_run_referee.py --config-a "BPD_DEVICE_LOGITS=1" --n-tokens 30   # single-config audit

Author: Bocher, 2026-06-12. The at-scale leg of the fusion-series dossier standard,
the rmsnorm-migration flip certification, and the seed of the Ollama clocks+bits report.
"""
import os, sys, json, subprocess, argparse, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_ENV = dict(os.environ, BPD_DEVICE_KV_CACHE="1", BPD_DEVICE_ATTN="1",
                BPD_MASKED_ATTN="1", BPD_SLAB="1", BPD_GRAPH_PREP="1",
                BPD_GRAPH="1")  # arms run the PRODUCTION graphed path

# Varied prompts: different lengths, domains, and token-id regimes.
PROMPTS = [
    [1, 415, 6557],                      # canonical (the gate's house prompt)
    [1, 785, 3974, 13876],               # Iyun's near-tie finder (layer-3 history)
    [1, 9707, 11, 1246, 525, 498],       # conversational opener
    [1, 3838, 374, 279, 7290, 315],      # "what is the meaning of"
    [1, 641, 279, 7167, 11, 1052, 572],  # narrative opener
    [1, 16429, 264, 6018, 19120, 25],    # code-flavored
]

ARM_SRC = r'''
import os, sys, json
import numpy as np
sys.path.insert(0, os.environ["BPD_LIB"]); sys.path.insert(0, os.environ["BPD_DIR"])
import decode_fact as D
import dev_residency as DR

N      = int(os.environ["N_TOKENS"])
PROMPT = json.loads(os.environ["ARM_PROMPT"])

cfg, w = D.load_model(os.environ.get("GGUF", ""))
# PROFILE-PINNED ARMS (case-study #3 closure): both arms run the NAMED production
# ensemble; only declared toggle overrides may differ.
DR.apply_production_profile()
_overrides = json.loads(os.environ.get("ARM_TOGGLE_OVERRIDES", "{}"))
for _k, _v in _overrides.items():
    setattr(DR, _k, _v)
_att = DR.attest_profile()

kv = [None] * cfg['n_layers']
out = {"prompt": PROMPT, "steps": [],
       "profile_attest": {k: str(v) for k, v in sorted(_att.items())},
       "toggle_overrides": _overrides,
       "path_attestation": {
           k: v for k, v in sorted(os.environ.items()) if k.startswith("BPD_")
           and k not in ("BPD_LIB", "BPD_DIR")}}

gen = list(PROMPT)
# PRODUCTION-PATH ARMS: seed the captured graph with the prompt, then replay per
# token — the path that ships. (Bare eager forward_pass_resident asks an eager
# question about a graphed reality: with fusions toggled OFF the eager legacy
# route legitimately leaves device-residency and computes host-fp32 logits —
# an honest engine, a wrong instrument question. Lesson of 2026-06-12.)
gr = DR.GraphRunner(w, cfg, kv)
tok, pos = D.step_tensors(gen, 0)
lg0 = gr.seed(tok, pos)
captured = False
for step in range(N):
    if step == 0:
        lg = lg0
    elif not captured:
        tok, pos = D.step_tensors(gen, step)
        gr.capture(tok, pos)           # capture pass: records the graph, returns None
        captured = True
        lg = gr.replay_logits(gen[-1]) # replay the SAME token for its logits
    else:
        lg = gr.replay_logits(gen[-1])
    v = lg[0, -1].detach().cpu().numpy().astype(np.float32) if hasattr(lg, 'detach') \
        else np.asarray(lg, np.float32).reshape(-1)
    top2 = np.argpartition(v, -2)[-2:]
    top2 = top2[np.argsort(v[top2])][::-1]
    t1, t2 = int(top2[0]), int(top2[1])
    out["steps"].append({"step": step, "tok": t1, "runner_up": t2,
                         "margin": float(v[t1] - v[t2]),
                         "logits_u32": v.view(np.uint32).tolist()})
    gen.append(t1)
out["gen"] = gen[len(PROMPT):]
out["path_attestation"]["folded_logits_executed"] = (
    getattr(DR, "_LOGITS_RMS", None) is not None)
print(json.dumps(out))
'''

def run_arm(extra_env_str, prompt, n_tokens, gguf, overrides=None):
    e = dict(BASE_ENV, N_TOKENS=str(n_tokens), ARM_PROMPT=json.dumps(prompt))
    bpd = os.path.abspath(os.path.join(HERE, "..", ".."))
    e["BPD_LIB"] = os.path.join(bpd, "lib"); e["BPD_DIR"] = bpd
    if gguf: e["GGUF"] = gguf
    if overrides: e["ARM_TOGGLE_OVERRIDES"] = overrides
    for kv_pair in extra_env_str.split():
        if "=" in kv_pair:
            k, v = kv_pair.split("=", 1); e[k] = v
    p = subprocess.run([sys.executable, "-c", ARM_SRC], env=e,
                       capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return {"error": p.stderr[-2000:]}
    return json.loads(p.stdout.strip().splitlines()[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", default="", help="extra env for arm A (e.g. 'BPD_RMS_BLOCKROW=1')")
    ap.add_argument("--config-b", default=None, help="extra env for arm B (omit for single-config audit)")
    ap.add_argument("--overrides-b", default=None,
                    help="JSON dict of DR module attrs for arm B (profile-pinned mode)")
    ap.add_argument("--n-tokens", type=int, default=30)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS), help="how many of the prompt set to run")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    dual = a.config_b is not None or a.overrides_b is not None
    print(f"=== LONG-RUN REFEREE — {'dual' if dual else 'single'}-config, "
          f"{a.prompts} prompts x {a.n_tokens} tokens ===")
    print(f"arm A env: {a.config_a or '(base)'}" +
          (f"\narm B env: {a.config_b or '(base)'}" if dual else ""))

    all_margins, near_flips, agreements, results = [], [], [], []
    for pi, prompt in enumerate(PROMPTS[:a.prompts]):
        A = run_arm(a.config_a, prompt, a.n_tokens, a.gguf)
        if "error" in A:
            print(f"prompt {pi}: ARM-A ERROR (UNTESTED): {A['error'][:300]}"); return 2
        rec = {"prompt_index": pi, "prompt": prompt, "gen_a": A["gen"]}
        margins_a = [s["margin"] for s in A["steps"]]
        all_margins += margins_a
        if dual:
            B = run_arm(a.config_b or "", prompt, a.n_tokens, a.gguf, overrides=a.overrides_b)
            if "error" in B:
                print(f"prompt {pi}: ARM-B ERROR (UNTESTED): {B['error'][:300]}"); return 2
            pa_a = A.get("profile_attest", {}); pa_b = B.get("profile_attest", {})
            base_diff = {k: (pa_a.get(k), pa_b.get(k)) for k in set(pa_a) | set(pa_b)
                         if pa_a.get(k) != pa_b.get(k)
                         and k not in (json.loads(a.overrides_b or "{}").keys())}
            if base_diff:
                print(f"prompt {pi}: ARMS RAN DIFFERENT BASE PROFILES {base_diff} — "
                      f"UNTESTED-AS-INTENDED, refused."); return 2
            fa = A.get("path_attestation", {}).get("folded_logits_executed")
            fb = B.get("path_attestation", {}).get("folded_logits_executed")
            if fa != fb:
                print(f"prompt {pi}: ARMS ON DIFFERENT LOGITS PATHS "
                      f"(A folded={fa}, B folded={fb}) — UNTESTED-AS-INTENDED, "
                      f"comparison refused."); return 2
            rec["gen_b"] = B["gen"]
            agree = A["gen"] == B["gen"]
            agreements.append(agree)
            # per-step drift (max |logit_a - logit_b|) while sequences agree
            import numpy as np
            drifts = []
            for sa, sb in zip(A["steps"], B["steps"]):
                if sa["tok"] != sb["tok"]: break
                la = np.array(sa["logits_u32"], np.uint32).view(np.float32)
                lb = np.array(sb["logits_u32"], np.uint32).view(np.float32)
                d = float(np.abs(la - lb).max())
                drifts.append(d)
                if d > 0 and sa["margin"] < 5 * d:
                    near_flips.append({"prompt": pi, "step": sa["step"],
                                       "top": sa["tok"], "runner_up": sa["runner_up"],
                                       "margin": sa["margin"], "drift": d})
            rec["max_drift"] = max(drifts) if drifts else 0.0
            rec["token_agreement"] = "FULL" if agree else \
                f"DIVERGED at step {next(i for i,(x,y) in enumerate(zip(A['gen'],B['gen'])) if x!=y)}"
            print(f"prompt {pi}: agree={rec['token_agreement']}  max_drift={rec['max_drift']:.3e}  "
                  f"min_margin={min(margins_a):.4f}")
        else:
            print(f"prompt {pi}: gen={A['gen'][:8]}...  min_margin={min(margins_a):.4f}  "
                  f"median={statistics.median(margins_a):.4f}")
        results.append(rec)

    print(f"\n--- MARGIN STATISTICS ({len(all_margins)} decisions) ---")
    print(f"min={min(all_margins):.4f}  median={statistics.median(all_margins):.4f}  "
          f"max={max(all_margins):.4f}")
    summary = {"results": results, "margin_min": min(all_margins),
               "margin_median": statistics.median(all_margins), "near_flips": near_flips}
    if dual:
        max_drift = max((r.get("max_drift", 0.0) for r in results), default=0.0)
        sf = (min(all_margins) / max_drift) if max_drift > 0 else float("inf")
        summary["max_drift"] = max_drift; summary["safety_factor"] = sf
        summary["all_agree"] = all(agreements)
        print(f"max_drift={max_drift:.3e}  SAFETY FACTOR (min_margin/max_drift) = {sf:.1f}x")
        print(f"near-flip census: {len(near_flips)} steps within 5x" +
              (f" — {near_flips}" if near_flips else " — NONE"))
        verdict = "LONG-RUN VERIFIED ✓ — token-exact at scale" if all(agreements) else \
                  "LONG-RUN DIVERGENCE ✗ — see per-prompt reports (classify per REFERENCE_DISCIPLINE)"
        print("\n" + verdict)
    if a.json:
        with open(a.json, "w") as f: json.dump(summary, f, indent=1)
    return 0 if (not dual or summary.get("all_agree")) else 1

if __name__ == "__main__":
    sys.exit(main())
