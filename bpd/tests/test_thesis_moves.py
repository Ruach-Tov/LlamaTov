#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_thesis_moves.py — guards the 5 thesis-fidelity moves (fixes infra-debt #3).

These modules (flash schedule, layer-from-graph, AST epilogue fusion, schedule
vocabulary, norm/softmax from facts) were built + manually verified but UNGUARDED —
a refactor could silently break them. Local, <2s, swipl-only (no GPU). Each test
asserts the Prolog predicate produces the expected structure / generates valid code.
Author: Iyun, 2026-06-08
"""
import os, subprocess, shutil
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EMIT = os.path.join(REPO, "bpd", "kernelgen", "emitters")
SCHED = os.path.join(REPO, "bpd", "kernelgen", "schedule")
FACTS = os.path.join(REPO, "bpd", "lib", "robust_op_match.pl")
HAVE_SWIPL = shutil.which("swipl") is not None
skip_no_swipl = pytest.mark.skipif(not HAVE_SWIPL, reason="swipl required")

def _swipl(goal, extra_loads=None):
    loads = "".join(f"use_module('{m}'), " for m in (extra_loads or []))
    g = f"use_module('{FACTS}',[op_expr/2]), {loads} ({goal} -> write(ok) ; write(no)), halt"
    r = subprocess.run(["swipl", "-q", "-g", g, "-t", "halt"],
                       capture_output=True, text=True, timeout=30)
    return ("ok" in r.stdout), r.stdout, r.stderr


# ── MOVE 1: flash_attn_schedule ──────────────────────────────────────────────
@skip_no_swipl
def test_flash_schedule_exists():
    ok, out, err = _swipl("flash_attn_schedule(tuned_d128, schedule(flash, _))",
                          [f"{EMIT}/flash_attention.pl"])
    assert ok, f"flash_attn_schedule(tuned_d128) missing: {out} {err}"

@skip_no_swipl
def test_flash_schedule_lowers_vectorized(tmp_path):
    out_cu = str(tmp_path / "f.cu")
    ok, _, err = _swipl(
        f'recognize_attention([bpd_matmul,bpd_softmax,bpd_matmul], Spec), '
        f'flash_attn_schedule(tuned_d128, Sch), '
        f'emit_flash_schedule(Spec, Sch, 128, "{out_cu}")',
        [f"{EMIT}/flash_attention.pl"])
    assert ok and os.path.exists(out_cu), f"emit_flash_schedule failed: {err}"
    body = open(out_cu).read()
    # the vectorized (float4) winning structure
    assert "float4" in body and "__shfl_down_sync" in body, "not the warp+vectorized kernel"
    assert "// LAUNCH:" in body, "no launch contract"


# ── MOVE 2: transformer layer from the recognized graph ──────────────────────
@skip_no_swipl
def test_layer_plan_recognizes_attention():
    # the 17-op layer should yield a plan containing a flash_attn step
    ops = ("[op(rms_norm,a,1),op(matmul,b,2),op(matmul,c,3),op(matmul,d,4),"
           "op(matmul,e,5),op(scale,f,6),op(softmax,g,7),op(matmul,h,8),"
           "op(matmul,i,9),op(add,j,10),op(rms_norm,k,11),op(matmul,l,12),"
           "op(matmul,m,13),op(silu,n,14),op(mul,o,15),op(matmul,p,16),op(add,q,17)]")
    ok, out, err = _swipl(
        f"layer_plan({ops}, Plan), memberchk(flash_attn(_), Plan)",
        [f"{EMIT}/transformer_layer.pl"])
    assert ok, f"layer_plan did not recognize the attention diamond: {out} {err}"

@skip_no_swipl
def test_layer_plan_fuses_swiglu():
    # matmul -> [silu, mul] should appear as a fused_chain epilogue
    ops = "[op(matmul,a,1),op(silu,b,2),op(mul,c,3)]"
    ok, out, err = _swipl(
        f"layer_plan({ops}, Plan), memberchk(fused_chain(op(matmul,_,_), Tail), Plan), "
        f"Tail = [op(silu,_,_),op(mul,_,_)]",
        [f"{EMIT}/transformer_layer.pl"])
    assert ok, f"swiglu not fused into an epilogue chain: {out} {err}"


# ── MOVE 3: epilogue fusion at the AST (backend-neutral) ─────────────────────
@skip_no_swipl
def test_epilogue_lowers_cuda_and_mlir():
    # relu->scale must lower to BOTH cuda and mlir from one fold
    ok, out, err = _swipl(
        "epilogue_backends([bpd_relu,bpd_scaling], [cuda(C), mlir(Stmts, _)]), "
        "atom_length(C, LC), LC > 0, Stmts = [_|_]",
        [f"{EMIT}/epilogue_fusion.pl"])
    assert ok, f"epilogue did not lower to both backends: {out} {err}"

@skip_no_swipl
def test_epilogue_identity_is_var():
    ok, _, err = _swipl("epilogue_cuda([], \"v\")", [f"{EMIT}/epilogue_fusion.pl"])
    assert ok, f"empty epilogue should be identity 'v': {err}"


# ── MOVE 4: schedule vocabulary complete ─────────────────────────────────────
@skip_no_swipl
@pytest.mark.parametrize("sched", [
    "tiled_elementwise(float4,grid_stride)",
    "tiled_pool(8,8)",
    "tiled_conv(128,128,32,8,4)",
    "tiled_flash(16,32,4)",
])
def test_schedule_vocabulary(sched):
    ok, out, err = _swipl(
        f"tile_schedule(_, {sched}, schedule(_, _, Prims)), Prims = [_|_]",
        [f"{SCHED}/schedule_ir.pl"])
    assert ok, f"schedule {sched} not well-formed: {out} {err}"


# ── MOVE 5: norm/softmax from facts ──────────────────────────────────────────
@skip_no_swipl
def test_rmsnorm_emits_from_fact(tmp_path):
    out_cu = str(tmp_path / "rms.cu")
    ok, _, err = _swipl(
        f'op_expr(bpd_rmsnorm, R), emit_from_fact(R, [], "{out_cu}")',
        [f"{EMIT}/norm_softmax_from_facts.pl"])
    assert ok and os.path.exists(out_cu), f"rmsnorm not emitted from fact: {err}"
    body = open(out_cu).read()
    assert "k_rmsnorm" in body and "rsqrtf" in body and "ss +=" in body

@skip_no_swipl
def test_softmax_emits_from_fact(tmp_path):
    out_cu = str(tmp_path / "sm.cu")
    ok, _, err = _swipl(
        f'op_expr(bpd_softmax, Sm), emit_from_fact(Sm, [], "{out_cu}")',
        [f"{EMIT}/norm_softmax_from_facts.pl"])
    assert ok and os.path.exists(out_cu), f"softmax not emitted from fact: {err}"
    body = open(out_cu).read()
    assert "k_softmax" in body and "expf" in body


# ── Q8_0 quantized dot (llama path): the fact lowers to scalar + dp4a backends ──
@skip_no_swipl
def test_q8_0_fact_exists():
    ok, out, err = _swipl("q8_0_op_expr(q8_0_dot(block(32), scale(fp16), quant(int8)))",
                          [f"{EMIT}/q8_0_from_facts.pl"])
    assert ok, f"q8_0_op_expr missing or wrong shape: {out} {err}"

@skip_no_swipl
def test_q8_0_emits_scalar(tmp_path):
    out = str(tmp_path / "q8s.cu")
    ok, _, err = _swipl(f'q8_0_op_expr(E), emit_from_fact(E, [mode(scalar)], "{out}")',
                        [f"{EMIT}/q8_0_from_facts.pl"])
    assert ok and os.path.exists(out), f"q8_0 scalar not emitted: {err}"
    body = open(out).read()
    assert "k_q8_0_gemv" in body and "isum" in body and "__half2float" in body

@skip_no_swipl
def test_q8_0_emits_dp4a(tmp_path):
    out = str(tmp_path / "q8d.cu")
    ok, _, err = _swipl(f'q8_0_op_expr(E), emit_from_fact(E, [mode(dp4a)], "{out}")',
                        [f"{EMIT}/q8_0_from_facts.pl"])
    assert ok and os.path.exists(out), f"q8_0 dp4a not emitted: {err}"
    body = open(out).read()
    assert "__dp4a" in body, "dp4a variant must use the __dp4a intrinsic"
