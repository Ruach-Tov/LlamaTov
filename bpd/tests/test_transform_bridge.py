# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_transform_bridge.py — the role-based model-transformation bridge on the live qwen2 graph.

Derives the compute graph from the actual GGUF (gguf_to_graph.py), then runs transform_bridge.pl's
role inference + model_transform(qwen, turboquant) and asserts the role counts + transform precision.
Requires SWI-Prolog + the production GGUF ($LLAMATOV_MODEL or the default).
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BPD = os.path.dirname(HERE)
LIB = os.path.join(BPD, "lib")
MODEL = os.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf")


def test_role_bridge_on_live_qwen2(tmp_path=None):
    if not os.path.exists(MODEL):
        import pytest; pytest.skip(f"model not present: {MODEL}")
    # 1. derive the graph from the live GGUF
    graph_pl = "/tmp/_tb_qwen_graph.pl"
    env = dict(os.environ, PYTHONPATH=f"{BPD}:{LIB}")
    with open(graph_pl, "w") as f:
        r = subprocess.run([sys.executable, os.path.join(LIB, "gguf_to_graph.py"), MODEL],
                           stdout=f, stderr=subprocess.PIPE, env=env, timeout=120)
    assert r.returncode == 0, f"graph derivation failed: {r.stderr.decode()[:300]}"
    nops = sum(1 for l in open(graph_pl) if l.startswith("op("))
    assert nops == 456, f"expected 456 ops for qwen2-24L, got {nops}"

    # 2. run the bridge roles via a Prolog goal
    goal = f'''
      use_module('{LIB}/transform_bridge'),
      use_module('{graph_pl}'),
      findall(op(I,K,In,O), model_graph:op(I,K,In,O), G),
      meta_attach_points(G, kv_projection, KV), length(KV, NKV),
      meta_attach_points(G, q_projection, Q), length(Q, NQ),
      meta_attach_points(G, skip_connection, SK), length(SK, NSK),
      meta_attach_points(G, ffn_projection, FFN), length(FFN, NF),
      model_transform(G, turboquant, _, applied(_,_,provenance(P))),
      findall(T, member(tensor_encoding(T,turboquant), P), Enc), length(Enc, NE),
      format("RESULT ~w ~w ~w ~w ~w~n", [NKV, NQ, NSK, NF, NE]), halt
    '''
    r = subprocess.run(["swipl", "-q", "-g", goal, "-t", "halt"],
                       capture_output=True, text=True, timeout=120)
    out = r.stdout
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"no result: {out}\n{r.stderr[:300]}"
    nkv, nq, nsk, nf, ne = map(int, line[0].split()[1:6])
    assert nkv == 48, f"kv_projection: {nkv} (expected 48)"
    assert nq == 24, f"q_projection: {nq} (expected 24)"
    assert nsk == 48, f"skip_connection: {nsk} (expected 48 true residuals)"
    assert nf == 96, f"ffn_projection: {nf} (expected 96)"
    assert ne == 48, f"turboquant encoded: {ne} (expected 48 K/V outputs)"


if __name__ == "__main__":
    test_role_bridge_on_live_qwen2()
    print("PASS: role-based transform bridge correct on the live qwen2 graph")
