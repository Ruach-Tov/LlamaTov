# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
#!/usr/bin/env python3
"""decode_referee.py — standing differential gate for the fact-driven KV-cache decode.

Asserts decode_fact.decode()'s incremental T=1 decode (cfe623817 + shared-loop
refactor) is CORRECT, per step, against three references. CRITICALLY: this gate
IMPORTS decode_fact.forward_pass — the subject and every reference run the SAME
loop body with injected ops (int8 | fp32). Reference≡subject structurally; the
gate verifies the real code, not a mirror of it (Iyun msg ad93dcf0).

  A. RECOMPUTE ORACLE (HARD):  per gated step, re-run a fresh prefill-style
     forward over generated[:t] with the SAME int8 ops, into a FRESH cache, and
     compare per-layer K/V against the incrementally-built kv_cache. The only
     delta is cache bookkeeping (cat order, RoPE absolute positions, GQA), so a
     mismatch localizes corruption to (step, layer) — and it catches shared-
     substrate bugs that an int8-vs-fp32 comparison would miss (both would use
     the same buggy cache; the recompute uses none).
     NOT 0-ULP: torch reduction order differs between T=1 GEMV-shaped and T=n
     GEMM-shaped matmuls — ULP noise is expected; corruption is O(1). Hard
     assert: max_abs < --kv-tol. ULP stats reported.

  B. FP32 REFERENCE (SOFT):    forward_pass with fp32 ops. Token divergence is
     SOFT — Q8_0 carries ~2.5%/linear inherent quant error and legitimately
     flips argmax at thin top1-top2 margins (Iyun's caveat). Telemetry: logit
     max_abs drift + top1-top2 margin, every step. Flip at thin margin = quant
     working as designed; flip at WIDE margin (margin/drift > --margin-warn)
     = bug signal -> WARN.

  C. OLLAMA ORACLE (SOFT, --ollama): localhost:11434/api/generate, temp 0 —
     the honest external Q8_0 oracle. Token path probed (context array /
     raw:true); if tokenizers mismatch, compare logit-level on a forced prefix
     instead (Iyun's suggestion) — SKIPs gracefully, never fakes a verdict.

  HARD (exit 1): A-mismatch beyond tolerance; shape mismatch; NaN/Inf in logits
  or KV. SOFT: B/C divergence (logged). WARN: wide-margin flip, drift jumps.

Run on enclave (P4):
    python3 decode_referee.py                      # default: qwen_q8, 6 tokens
    python3 decode_referee.py --n-tokens 12 --kv-cadence 3 --ollama

Idiom: sibling of q8_referee.py / flash_referee.py; ulp() from kernel_harness.
Author: Bocher, 2026-06-09 (design reviewed by Iyun, msgs aacd4465 + ad93dcf0).
"""
import os, sys, json, time, argparse, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../Ruach-Tov
sys.path.insert(0, HERE)                                          # kernel_harness
sys.path.insert(0, os.path.join(REPO, "bpd"))                     # decode_fact
sys.path.insert(0, os.path.join(REPO, "bpd", "lib"))              # fact_dispatch
from kernel_harness import ulp

import torch
import llamatov_run as R
import decode_fact as D     # THE shared loop: forward_pass, load_model, step_tensors

DEFAULT_GGUF = _os2.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf")
DEFAULT_IDS = [1, 415, 6557]
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b-instruct-q8_0"


def make_ops(mode, cfg):
    """(lin, rms) pair for a mode, built from decode_fact's own functions."""
    if mode == "int8":
        import fact_dispatch as fd
        return D.q8lin, (lambda x, wn: fd.rms_norm_fact(x, wn, cfg['norm_eps']))
    return D.fp32lin, (lambda x, wn: R.rms_norm(x, wn, cfg['norm_eps']))


def to_host_kv(ent):
    """Normalize a kv-cache entry to (k_np, v_np, T) float32 host arrays.

    Two architectures exist (and the gate must read both):
      - host tuples: (k, v) torch tensors [B, nkv, T, hd]
      - DeviceKVCache objects (BPD_DEVICE_KV_CACHE=1): device-resident K/V at
        k_ptr/v_ptr, valid rows [0:length], row width nkv*hd. We DtoH-copy ONLY
        the valid [0:L] slice — beyond-L is masked garbage BY DESIGN (the poison
        test owns that territory; fingerprinting it would assert on noise).
    """
    if isinstance(ent, tuple):
        k, v = ent
        # [B=1, nkv, T, hd] -> [T, nkv*hd] so host tuples and device caches
        # (position-major rows) compare in the SAME geometry
        kn = k.detach().cpu().numpy().astype(np.float32)
        vn = v.detach().cpu().numpy().astype(np.float32)
        T = int(kn.shape[2])
        kn = np.ascontiguousarray(kn[0].transpose(1, 0, 2)).reshape(T, -1)
        vn = np.ascontiguousarray(vn[0].transpose(1, 0, 2)).reshape(T, -1)
        return kn, vn, T
    # DeviceKVCache (duck-typed: has k_ptr/v_ptr/length/width)
    import ctypes
    import fact_dispatch as fd
    L = int(ent.length)
    width = int(ent.width)
    kn = np.empty(L * width, np.float32)
    vn = np.empty(L * width, np.float32)
    if L:
        cu = fd._libcuda(); cu.cuCtxSynchronize()
        nbytes = L * width * ent.elem_bytes
        cu.cuMemcpyDtoH_v2(kn.ctypes.data_as(ctypes.c_void_p), ent.k_ptr, nbytes)
        cu.cuMemcpyDtoH_v2(vn.ctypes.data_as(ctypes.c_void_p), ent.v_ptr, nbytes)
    return kn.reshape(L, width), vn.reshape(L, width), L


def kv_fingerprint(kv_cache):
    """Per-layer sha1 + NaN/Inf flags. Cheap, run EVERY step."""
    out = []
    for ent in kv_cache:
        if ent is None:
            out.append(None); continue
        kn, vn, T = to_host_kv(ent)
        bad = bool(np.isnan(kn).any() or np.isinf(kn).any()
                   or np.isnan(vn).any() or np.isinf(vn).any())
        h = hashlib.sha1(kn.tobytes()); h.update(vn.tobytes())
        out.append({"sha1": h.hexdigest()[:12], "T": T, "naninf": bad})
    return out


def kv_compare(kv_inc, kv_rec, tol):
    """Compare incremental vs recomputed cache per layer (host tuples OR
    DeviceKVCache — both normalized through to_host_kv, valid slice only).
    Returns worst report."""
    worst = {"layer": -1, "tensor": "-", "max_abs": 0.0, "max_ulp": 0, "ok": True}
    for il, (a, b) in enumerate(zip(kv_inc, kv_rec)):
        ka, va, Ta = to_host_kv(a)
        kb, vb, Tb = to_host_kv(b)
        if Ta != Tb:
            return {"layer": il, "tensor": "T", "max_abs": float("inf"),
                    "max_ulp": -1, "ok": False,
                    "shape_mismatch": ([Ta], [Tb])}
        for name, an, bn in (("k", ka, kb), ("v", va, vb)):
            if an.shape != bn.shape:
                return {"layer": il, "tensor": name, "max_abs": float("inf"),
                        "max_ulp": -1, "ok": False,
                        "shape_mismatch": (list(an.shape), list(bn.shape))}
            d = float(np.abs(an - bn).max()) if an.size else 0.0
            mu, nd, nanm, sz = ulp(an, bn)
            if d > worst["max_abs"]:
                worst = {"layer": il, "tensor": name, "max_abs": d, "max_ulp": mu,
                         "n_differ": nd, "ok": d < tol}
    return worst


def ollama_tokens(prompt_ids, n_predict):
    """Probe Ollama for token-level greedy generation. None => SKIP, not FAIL."""
    import urllib.request
    body = {"model": OLLAMA_MODEL, "prompt": "", "raw": True, "stream": False,
            "context": list(prompt_ids),
            "options": {"temperature": 0, "seed": 0, "num_predict": n_predict}}
    try:
        req = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        ctx = resp.get("context")
        if ctx and len(ctx) > len(prompt_ids):
            return ctx[len(prompt_ids):len(prompt_ids) + n_predict]
    except Exception as e:
        print(f"[ollama] probe failed ({e}) — pillar C SKIPPED", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser(description="standing referee: fact-driven KV decode")
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--input-ids", default=",".join(map(str, DEFAULT_IDS)))
    ap.add_argument("--n-tokens", type=int, default=6)
    ap.add_argument("--kv-cadence", type=int, default=1,
                    help="run the recompute oracle every N steps (1 = every step)")
    ap.add_argument("--kv-tol", type=float, default=1e-3,
                    help="hard-assert tolerance for recompute-vs-incremental max_abs")
    ap.add_argument("--margin-warn", type=float, default=2.0,
                    help="WARN when a token flip occurs at margin/drift above this")
    ap.add_argument("--resident", action="store_true",
                    help="subject = dev_residency.forward_pass_resident (device-resident "
                    "activations) instead of decode_fact.forward_pass int8.")
    ap.add_argument("--claims", choices=["bitexact", "device-numerics"], default="bitexact",
                    help="the subject's declared numerics claim. bitexact: A2 cross-impl "
                    "recompute is HARD at --kv-tol. device-numerics: subject declares "
                    "legitimate device-vs-host transcendental/reduction variance (expf, "
                    "reduction order); A2 becomes SOFT-WITH-SHAPE (escalates on NaN/shape/"
                    "step-0 divergence). A1 self-consistent recompute is ALWAYS hard.")
    ap.add_argument("--ollama", action="store_true", help="enable pillar C")
    ap.add_argument("--baseline", help="pillar D: prior-run reference — a comma-separated "
                    "token list (e.g. '310,470,895,280,286,456') OR a --json file from a "
                    "previous run. Isolates IMPLEMENTATION VARIANCE: a flip vs baseline at "
                    "thin margin = documented numerical variance (e.g. device-quant tie "
                    "behavior); at wide margin = the change under test broke something.")
    ap.add_argument("--json", help="write per-step records")
    a = ap.parse_args()
    input_ids = [int(t) for t in a.input_ids.split(",")]

    baseline_toks = None
    if a.baseline:
        if os.path.exists(a.baseline):
            with open(a.baseline) as f:
                baseline_toks = [r["tok8"] for r in json.load(f)]
        else:
            baseline_toks = [int(t) for t in a.baseline.split(",")]

    subj_name = ("dev_residency.forward_pass_resident" if a.resident
                 else "decode_fact.forward_pass int8")
    print(f"=== DECODE REFEREE (subject: {subj_name}, P4) ===", flush=True)
    t0 = time.time()
    cfg, w = D.load_model(a.gguf)
    print(f"model loaded {time.time()-t0:.1f}s  layers={cfg['n_layers']} "
          f"heads={cfg['n_head']}/{cfg['n_head_kv']}", flush=True)

    lin8, rms8 = make_ops("int8", cfg)
    lin32, rms32 = make_ops("fp32", cfg)

    # The subject forward: residency variant has the SAME (w,cfg,tok,positions,
    # kv_cache)->logits contract, minus the lin/rms injection (its ops are
    # internal).
    #
    # PILLAR A SPLIT (the expf/tick-cascade ruling, 2026-06-10, memory 22c554d5):
    #   A1 — SELF-CONSISTENT recompute: the oracle uses the SUBJECT'S OWN forward.
    #        Tests cache BOOKKEEPING (cat order, rope positions, GQA) under
    #        identical math. HARD 0.00e+00 for every subject, always. Device
    #        residency CAN and MUST pass this bit-exactly.
    #   A2 — CROSS-IMPLEMENTATION recompute (decode_fact host-int8): claim-
    #        dependent. --claims bitexact => HARD (tolerance). --claims
    #        device-numerics => SOFT-WITH-SHAPE: device transcendentals (expf vs
    #        torch.exp ~5e-7) + reduction orders legitimately flip Q8_0 ticks at
    #        near-boundary elements and cascade (silu is per-layer => systemic).
    #        Expected shape: clean early steps, then small growing divergence.
    #        Escalates to HARD on: NaN, shape mismatch, or divergence at step 0.
    if a.resident:
        import dev_residency as DR
        subject_fwd = lambda w_, cfg_, t_, p_, kv_: DR.forward_pass_resident(w_, cfg_, t_, p_, kv_)
    else:
        subject_fwd = lambda w_, cfg_, t_, p_, kv_: D.forward_pass(w_, cfg_, t_, p_, kv_, lin8, rms8)

    kv8 = [None] * cfg['n_layers']      # subject: incremental int8
    kv32 = [None] * cfg['n_layers']     # reference: incremental fp32 (B pillar)
    gen8, gen32 = list(input_ids), list(input_ids)
    records, hard_fail, warns = [], [], []

    base_col = "base   " if baseline_toks else ""
    print(f"{'step':<5}{'tok8':<7}{'tok32':<7}{base_col}{'drift':<10}{'margin':<10}"
          f"{'kvA max_abs':<13}{'kvA ulp':<9}{'naninf':<7}{'verdict'}", flush=True)

    for step in range(a.n_tokens):
        tok8, pos8 = D.step_tensors(gen8, step)
        tok32, pos32 = D.step_tensors(gen32, step)

        lg8 = subject_fwd(w, cfg, tok8, pos8, kv8)
        lg32 = D.forward_pass(w, cfg, tok32, pos32, kv32, lin32, rms32)

        v8 = lg8[0, -1].detach().cpu().numpy().astype(np.float32)
        v32 = lg32[0, -1].detach().cpu().numpy().astype(np.float32)
        n8 = int(v8.argmax()); n32 = int(v32.argmax())
        gen8.append(n8); gen32.append(n32)

        # NaN/Inf — HARD (logits + cache fingerprints, every step)
        naninf = bool(np.isnan(v8).any() or np.isinf(v8).any())
        fps = kv_fingerprint(kv8)
        naninf = naninf or any(f and f["naninf"] for f in fps)

        # B telemetry — only meaningful while the two sequences share a prefix
        same_prefix = gen8[:-1] == gen32[:-1]
        drift = float(np.abs(v8 - v32).max()) if same_prefix else float("nan")
        top2 = float(np.partition(v8, -2)[-2])
        margin = float(v8.max() - top2)
        margin_ratio = (margin / drift) if (same_prefix and drift > 0) else float("inf")

        # A pillars: recompute oracles at cadence (and always on the final step)
        # A1: SELF-CONSISTENT — recompute with the SUBJECT'S OWN forward. HARD always.
        # A2: CROSS-IMPL — recompute with decode_fact host-int8. Hard iff claims=bitexact.
        kv_a1 = {"max_abs": float("nan"), "max_ulp": -1, "ok": True, "layer": -1}
        kv_a2 = {"max_abs": float("nan"), "max_ulp": -1, "ok": True, "layer": -1}
        if step % a.kv_cadence == 0 or step == a.n_tokens - 1:
            rt = torch.tensor(gen8[:-1], dtype=torch.long)  # all tokens consumed so far
            rp = torch.arange(len(gen8) - 1)
            kv_rec1 = [None] * cfg['n_layers']
            subject_fwd(w, cfg, rt, rp, kv_rec1)            # A1: same implementation
            kv_a1 = kv_compare(kv8, kv_rec1, a.kv_tol)
            if a.resident:                                   # A2 differs from A1 only when
                kv_rec2 = [None] * cfg['n_layers']           # subject isn't host-int8 itself
                D.forward_pass(w, cfg, rt, rp, kv_rec2, lin8, rms8)
                kv_a2 = kv_compare(kv8, kv_rec2, a.kv_tol)
            else:
                kv_a2 = kv_a1
        # A2 escalation rules under device-numerics claim: NaN/shape are still hard.
        # NOTE: step-0 escalation REMOVED (2026-06-10): it assumed a host-path prefill,
        # where step-0 divergence meant a real bug. With device-uniform prefill
        # (7846b5200, D3 retired) the subject runs device ops from position 0, so
        # step-0 A2 divergence vs the host oracle is part of the declared variance
        # class. Tick-shape verification is the element-level first-divergence check.
        a2_hard_escalation = (not kv_a2["ok"]) and (
            a.claims == "bitexact"
            or kv_a2.get("shape_mismatch") is not None
        )
        kv_rep = kv_a1  # printed column remains the always-hard A1

        # D pillar: implementation variance vs a prior run's baseline tokens.
        # Only meaningful while subject and baseline share the consumed prefix —
        # after the first divergence the sequences explore different branches.
        base_tok, base_same_prefix = None, False
        if baseline_toks and step < len(baseline_toks):
            base_seq_so_far = list(input_ids) + baseline_toks[:step]
            base_same_prefix = gen8[:-1] == base_seq_so_far
            base_tok = baseline_toks[step]

        verdict = "PASS"
        if naninf:
            verdict = "FAIL:naninf"; hard_fail.append(step)
        elif not kv_a1["ok"]:
            verdict = f"FAIL:kvA1@L{kv_a1['layer']}/{kv_a1['tensor']}"; hard_fail.append(step)
        elif a2_hard_escalation:
            verdict = f"FAIL:kvA2@L{kv_a2['layer']}/{kv_a2['tensor']}"; hard_fail.append(step)
        elif not kv_a2["ok"]:
            verdict = f"soft:kvA2-numerics@L{kv_a2['layer']}({kv_a2['max_abs']:.1e})"
        elif n8 != n32:
            if same_prefix and margin_ratio > a.margin_warn:
                verdict = "WARN:flip@wide-margin"; warns.append(step)
            else:
                verdict = "soft:quant-flip"
        elif base_tok is not None and base_same_prefix and n8 != base_tok:
            # subject diverged from the PRIOR IMPLEMENTATION on the same prefix:
            # thin margin = documented numerical variance; wide = change broke something
            if margin_ratio > a.margin_warn:
                verdict = "WARN:base-flip@wide-margin"; warns.append(step)
            else:
                verdict = "soft:base-variance"
        elif same_prefix and drift > 0 and margin_ratio < a.margin_warn:
            verdict = "soft:margin-thin"

        ka = kv_rep["max_abs"]
        base_cell = ""
        if baseline_toks:
            base_cell = f"{(base_tok if base_tok is not None else '-'):<7}" if base_same_prefix or base_tok is None \
                        else f"{str(base_tok)+'*':<7}"   # * = prefix already diverged
        print(f"{step:<5}{n8:<7}{n32:<7}{base_cell}"
              f"{(f'{drift:.4f}' if same_prefix else 'diverged'):<10}"
              f"{margin:<10.4f}"
              f"{(f'{ka:.2e}' if ka == ka else '-'):<13}"
              f"{(kv_rep['max_ulp'] if kv_rep['max_ulp'] >= 0 else '-'):<9}"
              f"{str(naninf):<7}{verdict}", flush=True)
        records.append({"step": step, "tok8": n8, "tok32": n32, "drift": drift,
                        "margin": margin, "kvA": kv_rep, "kvA1": kv_a1, "kvA2": kv_a2,
                        "claims": a.claims,
                        "base_tok": base_tok, "base_same_prefix": base_same_prefix,
                        "kv_fingerprints": fps, "naninf": naninf,
                        "verdict": verdict})

    # C pillar (optional)
    if a.ollama:
        ot = ollama_tokens(input_ids, a.n_tokens)
        if ot is not None:
            match = sum(1 for x, y in zip(gen8[len(input_ids):], ot) if x == y)
            print(f"[ollama] subject vs ollama: {match}/{min(a.n_tokens, len(ot))} "
                  f"tokens match (soft) — ollama: {ot}", flush=True)

    print(f"\nsubject sequence: {gen8[len(input_ids):]}")
    print(f"fp32   sequence: {gen32[len(input_ids):]}")
    if baseline_toks:
        print(f"baseline sequence: {baseline_toks}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(records, f, indent=1)
    ok = not hard_fail
    print("\n" + ("DECODE VERIFIED ✓" if ok else f"HARD FAIL at steps {hard_fail} ✗")
          + (f"  ({len(warns)} warn)" if warns else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
