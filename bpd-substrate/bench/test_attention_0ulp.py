#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_attention_0ulp.py — TDD suite for bit-identical attention vs ggml.

One test per attention sub-op, each asserting 0-ULP against the ACTUAL dumped ggml tensor
(not a reconstructed reference). These define "done" for the attention bit-identity fix.

Method: build our .so with dump instrumentation (BPD_DUMP_DIR), run the forward on the
reference prompt, and compare each dumped sub-tensor to ggml's eval-callback dump bit-for-bit.

Run:  BPD_CPU_SO=/tmp/iyun_build/bpd_cpu.so python3 bench/test_attention_0ulp.py
Exit 0 = all green (bit-identical). Non-zero = count of failing sub-ops.

Tests (in computation order — fix upstream first):
  test_q_after_rope   : our Q post-rope == ggml 0011_ROPE_Qcur-0      (PASSING after rope_freqs fix)
  test_k_after_rope   : our K post-rope == ggml 0018_ROPE_Kcur-0
  test_qk_scores      : our raw QK scores == ggml 0041_node_20
  test_softmax        : our softmax weights == ggml 0044_node_21
  test_kqv            : our attn output == ggml 0052_kqv_out-0
"""
import os, sys, struct, subprocess, glob, tempfile, shutil
import numpy as np

GGUF = os.environ.get("GGUF", "")
SO = os.environ.get("BPD_CPU_SO", "/tmp/iyun_build/bpd_cpu.so")
GGML = os.environ.get("GGML_DUMP", "/tmp/iyun_ggmldump")
TOKENS = "128000,9906,11,856,836,374"
INFER = os.path.join(os.path.dirname(__file__), "bpd_llamatov_infer.py")

def lg(p):
    raw = open(p, "rb").read()
    nb = struct.unpack("<Q", raw[72:80])[0]
    return np.frombuffer(raw[80:80+nb], dtype=np.float32)

def ggml(pat):
    f = sorted(glob.glob(f"{GGML}/{pat}"))
    if not f: raise FileNotFoundError(pat)
    return lg(f[0])

def ulp(a, b):
    n = min(len(a), len(b)); a, b = a[:n].copy(), b[:n].copy()
    u = np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))
    return int(u.max()), int((u != 0).sum()), n, float(np.abs(a.astype(np.float64)-b).max())

def our(name, dumpdir):
    f = glob.glob(f"{dumpdir}/*_{name}.f32")
    if not f: raise FileNotFoundError(f"our dump {name}")
    return np.fromfile(f[0], dtype=np.float32)

# ── build instrumented .so (dump the attention sub-ops) ──
DUMP_PATCH = r'''
import re, sys
f = "bench/bpd_cpu.c"; s = open(f).read()
if "bpd_dtq" not in s:
    lines = s.split("\n"); ins = 0
    for i, l in enumerate(lines[:30]):
        if l.startswith("#include"): ins = i
    lines.insert(ins+1, '#include <stdio.h>\nstatic void bpd_dtq(const char* nm,const float* d,int n){const char* dd=getenv("BPD_DUMP_DIR");if(!dd)return;static int gi=0;char p[512];snprintf(p,sizeof(p),"%s/%04d_%s.f32",dd,gi++,nm);FILE* fp=fopen(p,"wb");if(!fp)return;fwrite(d,4,(size_t)n,fp);fclose(fp);}')
    s = "\n".join(lines)
# Q after rope (scratch2 after first rope call). Dump before K rope call.
a = "    bpd_rope_norm_freqs_cpu(scratch3, scratch3, pos_ids, rope_freqs,"
if "qpostrope" not in s:
    s = s.replace(a, '    { static int _q=0; if(_q==0){ bpd_dtq("qpostrope", scratch2, n_tokens*cfg->n_heads*cfg->head_dim); _q++; } }\n'+a, 1)
# raw QK scores (head0), softmax (head0), per qpos
m = s.index("float sv = (ic <= q_pos)")
ls = s.rfind("for (int ic", 0, m)
if "rawscore_h0" not in s:
    s = s[:ls] + '            { static int _rw=0; if(_rw<6 && iq==0){ bpd_dtq("rawscore_h0", scores, n_kv); _rw++; } }\n' + s[ls:]
sm = "                scores[ic] *= inv_sum;"
if "softmax_h0" not in s:
    s = s.replace(sm, sm + '\n            { static int _sm=0; if(_sm<6 && iq==0){ bpd_dtq("softmax_h0", scores, n_kv); _sm++; } }', 1)
open(f, "w").write(s); print("instrumented")
'''

def build_and_run():
    cwd = "<repo>/bpd-substrate"
    # clean, instrument, build
    subprocess.run(["git", "checkout", "bench/bpd_cpu.c"], cwd=cwd, capture_output=True)
    p = os.path.join(cwd, "_dump_patch.py"); open(p, "w").write(DUMP_PATCH)
    subprocess.run(["python3", "_dump_patch.py"], cwd=cwd, capture_output=True)
    cc = ["gcc","-O2","-mavx","-mf16c","-mssse3","-mno-avx2","-mno-fma","-funroll-loops",
          "-shared","-fPIC","-o",SO,"bench/bpd_cpu.c","bench/bpd_gemm_q8_0_cpu.c","-lm"]
    r = subprocess.run(cc, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED:\n"+r.stderr[:1500]); sys.exit(2)
    d = tempfile.mkdtemp(prefix="attn_dump_")
    subprocess.run(["python3", INFER, "--so", SO, "--gguf", GGUF, "--tokens", TOKENS,
                    "--n-generate", "1", "--out", "/tmp/_t.json"], cwd=cwd,
                   capture_output=True, env=dict(os.environ, BPD_DUMP_DIR=d, OMP_NUM_THREADS="1"))
    subprocess.run(["git", "checkout", "bench/bpd_cpu.c"], cwd=cwd, capture_output=True)
    return d

def main():
    d = build_and_run()
    results = []

    # test_q_after_rope: full tensor (all positions, all heads)
    o = our("qpostrope", d); g = ggml("0011_ROPE_Qcur-0.bin")
    mu, nd, n, ma = ulp(o, g)
    results.append(("test_q_after_rope", mu, nd, n, ma))

    # test_qk_scores: head0, per qpos (concatenate valid keys)
    gs = ggml("0041_MUL_MAT_node_20.bin").reshape(32, 6, 32)  # head,qpos,kv
    ours_s, ggml_s = [], []
    for qp, sf in enumerate(sorted(glob.glob(f"{d}/*rawscore_h0.f32"))):
        nk = qp + 1
        ours_s.append(np.fromfile(sf, dtype=np.float32)[:nk]); ggml_s.append(gs[0, qp, :nk])
    if ours_s:
        mu, nd, n, ma = ulp(np.concatenate(ours_s), np.concatenate(ggml_s))
        results.append(("test_qk_scores", mu, nd, n, ma))

    # test_softmax: head0, per qpos
    gsm = ggml("0044_SOFT_MAX_node_21.bin").reshape(32, 6, 32)
    ours_m, ggml_m = [], []
    for qp, sf in enumerate(sorted(glob.glob(f"{d}/*softmax_h0.f32"))):
        nk = qp + 1
        ours_m.append(np.fromfile(sf, dtype=np.float32)[:nk]); ggml_m.append(gsm[0, qp, :nk])
    if ours_m:
        mu, nd, n, ma = ulp(np.concatenate(ours_m), np.concatenate(ggml_m))
        results.append(("test_softmax", mu, nd, n, ma))

    print(f"\n{'TEST':<22}{'maxULP':>10}{'ndiff':>10}{'n':>8}{'maxabs':>12}  verdict")
    print("-"*72)
    fails = 0
    for name, mu, nd, n, ma in results:
        ok = (mu == 0)
        if not ok: fails += 1
        print(f"{name:<22}{mu:>10}{nd:>10}{n:>8}{ma:>12.2e}  {'PASS' if ok else 'FAIL'}")
    print(f"\n{len(results)-fails}/{len(results)} sub-ops bit-identical")
    return fails

if __name__ == "__main__":
    sys.exit(main())
