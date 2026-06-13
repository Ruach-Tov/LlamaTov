#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""gr_referee.py — GRAPH-REPLAY EQUIVALENCE gate (GR): graphed ≡ eager, bit-exact, HARD.

THE RULING (Bocher, accepted by Iyun): CUDA-graph capture is pure launch-overhead
removal of IDENTICAL kernels with IDENTICAL fixed pointers. There is NO legitimate
numerics variance between eager and replayed execution. ANY GR nonzero = capture bug
(stale pointer / missed dependency / host-baked value), NEVER declarable-soft.

ASSERTIONS per replay step i (token i+1 of decode):
  GR  (HARD): graphed logits == eager logits, BIT-EXACT (uint32 view equality)
  TOK (HARD): graphed argmax == eager argmax
  V2  (HARD): device *len_ptr == L0 + i  (counter contract v2 — device value via DtoH)
  V3  (HARD): the just-appended cache row (slot L0+i-1, layer-by-layer) matches the
              eager arm's same-step row bit-exact — catches no-advance (one slot
              rewritten N times: the a1bf30a8f bug) AND wrong-stride advance.
First divergence localized to (replay, layer, slot, max_abs).

ARM ISOLATION: module-global slab/resid/cache state is shared in-process, so the two
arms run in SUBPROCESSES, each from a cold start running ONLY the documented seed —
which institutionalizes the cold-start test (capture-state provenance: a graph must
capture from a KNOWN, documented state; delta 4f6d553e).

Run on enclave:  python3 gr_referee.py [--n-replays 5] [--json out.json]
Author: Bocher, 2026-06-10. Calibration target: a1bf30a8f (known-RED append-offset).
"""
import os, sys, json, subprocess, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, BPD_DEVICE_KV_CACHE="1", BPD_DEVICE_ATTN="1",
           BPD_MASKED_ATTN="1", BPD_SLAB="1", BPD_GRAPH_PREP="1")
IDS = [1, 415, 6557]

ARM_SRC = r'''
import os, sys, json, ctypes
import numpy as np
sys.path.insert(0, os.environ["BPD_LIB"]); sys.path.insert(0, os.environ["BPD_DIR"])
import decode_fact as D
import dev_residency as DR
import fact_dispatch as fd
import torch

MODE = os.environ["ARM_MODE"]            # "eager" | "graph"
N    = int(os.environ["N_REPLAYS"])
IDS  = json.loads(os.environ["ARM_IDS"])

cfg, w = D.load_model(os.environ.get("GGUF", ""))
nl = cfg['n_layers']
kv = [None] * nl
out = {"mode": MODE, "steps": []}
# ARMS-RAN-SAME-PATH ATTESTATION (execution attestation leg (b): the oracle ran
# the DECLARED PATH with the DECLARED QUANTIZATION). Record this arm's path
# coordinates; the verdict layer asserts both arms match. Born from the layer-4
# lesson: GR once compared host-fp32 (eager) vs folded-q8 (graphed) without
# declaring it — never again. See REFERENCE_DISCIPLINE.md.
out["path_attestation"] = {
    "device_logits": os.environ.get("BPD_DEVICE_LOGITS", "") == "1",
    "env_toggles": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith("BPD_") and k not in ("BPD_LIB", "BPD_DIR",
                    "BPD_GRAPH")},  # BPD_GRAPH differs BY DESIGN between arms
}

def cache_state():
    st = []
    for ent in kv:
        if ent is None or isinstance(ent, tuple):
            st.append(None); continue
        # DEVICE length is the truth under replay; host mirror is stale BY DESIGN
        # (no host calls in a replayed graph). Read device len, slice rows by it.
        dl = int(ent.length)
        if getattr(ent, "len_ptr", None):
            cu = fd._libcuda(); cu.cuCtxSynchronize()
            v_ = ctypes.c_int(-1); cu.cuMemcpyDtoH_v2(ctypes.byref(v_), ent.len_ptr, 4)
            dl = v_.value
        host_mirror = int(ent.length)
        saved = ent.length
        try:
            ent.length = dl                       # k_slice_host slices [0:length]
            k = ent.k_slice_host(); v = ent.v_slice_host()
        finally:
            ent.length = saved                    # restore mirror untouched
        st.append({"L": dl, "host_mirror": host_mirror,
                   "k_last": k[-1].reshape(-1).view(np.uint32).tolist() if len(k) else [],
                   "v_last": v[-1].reshape(-1).view(np.uint32).tolist() if len(v) else []})
    return st

def dev_len():
    for ent in kv:
        if ent is not None and not isinstance(ent, tuple) and getattr(ent, "len_ptr", None):
            cu = fd._libcuda(); cu.cuCtxSynchronize()
            v = ctypes.c_int(-1); cu.cuMemcpyDtoH_v2(ctypes.byref(v), ent.len_ptr, 4)
            return v.value
    return -1

gen = list(IDS)
if MODE == "eager":
    for step in range(N + 2):           # prefill(+1 decode parity with graph arm) + N replays
        tok, pos = D.step_tensors(gen, step)
        lg = DR.forward_pass_resident(w, cfg, tok, pos, kv)
        v = lg[0, -1].detach().cpu().numpy().astype(np.float32) if hasattr(lg, 'detach') else np.asarray(lg, np.float32).reshape(-1)
        nxt = int(v.argmax()); gen.append(nxt)
        out["steps"].append({"step": step, "tok": nxt,
                             "logits_u32": v.view(np.uint32).tolist(),
                             "dev_len": dev_len(), "cache": cache_state()})
else:
    gr = DR.GraphRunner(w, cfg, kv)
    # seed = documented precondition establishment + prefill of the prompt.
    # seed() takes ONE (tok,pos) forward; feed the whole prompt as the seed forward.
    tok, pos = D.step_tensors(gen, 0)
    lg = gr.seed(tok, pos)
    v = np.asarray(lg, np.float32).reshape(-1) if not hasattr(lg, 'detach') else lg[0, -1].detach().cpu().numpy().astype(np.float32)
    nxt = int(v.argmax()); gen.append(nxt)
    out["steps"].append({"step": 0, "tok": nxt, "logits_u32": v.view(np.uint32).tolist(),
                         "dev_len": dev_len(), "cache": cache_state()})
    # capture on the FIRST decode token (runs it once, eagerly-recorded)
    tok, pos = D.step_tensors(gen, 1)
    gr.capture(tok, pos)
    out["L_capture"] = max((int(e.length) for e in kv
                            if e is not None and not isinstance(e, tuple)), default=-1)
    # the capture run itself executed step 1's kernels while recording? NO — under
    # BeginCapture kernels are RECORDED not executed. First replay executes step 1.
    for i in range(N + 1):
        t = gen[-1]
        lg = gr.replay_logits(t)
        v = lg[0, -1].detach().cpu().numpy().astype(np.float32) if hasattr(lg, 'detach') else np.asarray(lg, np.float32).reshape(-1)
        nxt = int(v.argmax()); gen.append(nxt)
        out["steps"].append({"step": 1 + i, "tok": nxt,
                             "logits_u32": v.view(np.uint32).tolist(),
                             "dev_len": dev_len(), "cache": cache_state()})
out["gen"] = gen
# routing tell: _LOGITS_RMS written iff the folded device-logits path executed
out["path_attestation"]["folded_logits_executed"] = (
    getattr(DR, "_LOGITS_RMS", None) is not None)
print(json.dumps(out))
'''

def run_arm(mode, n, gguf, env):
    e = dict(env, ARM_MODE=mode, N_REPLAYS=str(n), ARM_IDS=json.dumps(IDS),
             BPD_LIB=os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", "bpd", "lib"),
             BPD_DIR=os.path.join(os.path.dirname(os.path.dirname(HERE)), "..", "bpd"))
    # normalize: referee lives in bpd/kernelgen/referee → bpd is two up
    bpd = os.path.abspath(os.path.join(HERE, "..", ".."))
    e["BPD_LIB"] = os.path.join(bpd, "lib"); e["BPD_DIR"] = bpd
    if gguf: e["GGUF"] = gguf
    if mode == "graph": e["BPD_GRAPH"] = "1"
    p = subprocess.run([sys.executable, "-c", ARM_SRC], env=e,
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return {"error": p.stderr[-2000:], "mode": mode}
    return json.loads(p.stdout.strip().splitlines()[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-replays", type=int, default=4)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    print("=== GR REFEREE (graphed ≡ eager, bit-exact, HARD) ===", flush=True)
    eager = run_arm("eager", a.n_replays, a.gguf, ENV)
    if "error" in eager:
        print("EAGER ARM ERROR (UNTESTED):\n" + eager["error"]); return 2
    graph = run_arm("graph", a.n_replays, a.gguf, ENV)
    if "error" in graph:
        print("GRAPH ARM ERROR (UNTESTED):\n" + graph["error"]); return 2

    # ARMS-RAN-SAME-PATH ASSERTION (hard): the two arms must have run the same
    # declared logits path and the same BPD env (minus BPD_GRAPH, which differs
    # by design). If not, any comparison is UNTESTED-AS-INTENDED, not a verdict.
    pa_e = eager.get("path_attestation", {}); pa_g = graph.get("path_attestation", {})
    if pa_e != pa_g:
        diffs = {k: (pa_e.get(k), pa_g.get(k))
                 for k in set(pa_e) | set(pa_g) if pa_e.get(k) != pa_g.get(k)}
        print("ARMS RAN DIFFERENT PATHS (UNTESTED-AS-INTENDED):")
        for k, (ve, vg) in diffs.items():
            print(f"  {k}: eager={ve!r} graph={vg!r}")
        print("\nGR comparison refused — fix the path mismatch first.")
        return 2
    print(f"path attestation: both arms {pa_e}")
    print(f"eager gen: {eager['gen']}")
    print(f"graph gen: {graph['gen']}")
    print(f"{'replay':<8}{'tokE':<8}{'tokG':<8}{'GR(logits)':<14}{'dlenE':<7}{'dlenG':<7}{'V3(cache)':<22}{'verdict'}", flush=True)
    hard, records = [], []
    n = min(len(eager["steps"]), len(graph["steps"]))
    for i in range(n):
        se, sg = eager["steps"][i], graph["steps"][i]
        le = np.array(se["logits_u32"], np.uint32); lg = np.array(sg["logits_u32"], np.uint32)
        gr_ok = np.array_equal(le, lg)
        gr_txt = "bit-exact" if gr_ok else f"{(le!=lg).sum()}/{le.size} differ"
        tok_ok = se["tok"] == sg["tok"]
        len_ok = se["dev_len"] == sg["dev_len"]
        # V3: just-appended row per layer (device-length-sliced both arms)
        v3_ok, v3_txt = True, "rows match"
        for il, (ce, cg) in enumerate(zip(se["cache"], sg["cache"])):
            if ce is None or cg is None: continue
            if ce["L"] != cg["L"]:
                v3_ok, v3_txt = False, f"L{il}: devlen {ce['L']}vs{cg['L']}"; break
            if ce["k_last"] != cg["k_last"] or ce["v_last"] != cg["v_last"]:
                v3_ok, v3_txt = False, f"L{il} slot{ce['L']-1} row differs"; break
        # INVERTED-STALENESS (graph purity): under replay the HOST mirror MUST be
        # frozen at L_capture — if it advanced, a host call leaked into the replay
        # path. Staleness is the PROOF of zero-host-involvement.
        if v3_ok and i >= 1 and "L_capture" in graph:
            Lcap = graph["L_capture"]
            gm = max((c["host_mirror"] for c in sg["cache"] if c), default=-1)
            if Lcap >= 0 and gm != Lcap:
                v3_ok, v3_txt = False, f"HOST LEAK: mirror {gm}!=Lcap {Lcap}"
        ok = gr_ok and tok_ok and len_ok and v3_ok
        verdict = "PASS" if ok else "FAIL:" + ("GR" if not gr_ok else "tok" if not tok_ok else "len" if not len_ok else "V3")
        if not ok: hard.append(i)
        print(f"{i:<8}{se['tok']:<8}{sg['tok']:<8}{gr_txt:<14}{se['dev_len']:<7}{sg['dev_len']:<7}{v3_txt:<22}{verdict}", flush=True)
        records.append({"replay": i, "tokE": se["tok"], "tokG": sg["tok"], "gr_ok": bool(gr_ok),
                        "len_e": se["dev_len"], "len_g": sg["dev_len"], "v3": v3_txt, "verdict": verdict})
    if a.json:
        with open(a.json, "w") as f: json.dump(records, f, indent=1)
    print("\n" + ("GR VERIFIED ✓ — graph is bit-equivalent to eager" if not hard
                  else f"GR FAIL at replays {hard} ✗ — capture bug (never soft)"))
    return 0 if not hard else 1

if __name__ == "__main__":
    sys.exit(main())
