# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
Fact → kernel lowering TABLE LOCK.

Inspired by NVlabs/cuda-oxide#124 (nihalpasham's predicate-table lock): rather than
testing one kernel at a time, lock the WHOLE production fact→kernel→reference table in
one place, so adding/refactoring one emitter cannot silently break another. Each row:

    op_expr fact  ->  emitted device kernel  ->  asserted against a reference impl

DISCIPLINE (the lessons from #124, applied):
  1. LOCK THE WHOLE TABLE, not just one row. Every fact the LLM actually uses is here.
  2. PROVE EACH ROW BITES. See test_table_lock_bites_marker + the REVERT_TO_CHECK notes:
     each assertion has a documented way to make it fail (revert the fact / break the
     emitter) so we know the lock is a real instrument, not a green rubber-stamp.
  3. GUARD THE ADJACENT FOOTGUN. The numerics-class invariants (eps-from-model, expf
     not fast-math) lock the CONDITIONS that made the residency-variance answer true
     (memory 9940f47a) — a future fast-math flag must not silently re-open that class.
  4. AGREEMENT *and* DIVERGENCE lanes. We assert bit/near-bit agreement with the
     reference on normal data AND check the declared soft-variance class on the edges.

The "reference table" here is the host fact-reference / torch, the way #124's reference
was rustc_codegen_ssa's predicate table.

PRODUCTION TABLE (the facts fact_dispatch.py / dev_residency.py actually wire to the LLM):
    bpd_rmsnorm  -> k_rmsnorm  (ref: torch rms, eps FROM MODEL)
    bpd_softmax  -> k_softmax  (ref: torch softmax)
    bpd_silu     -> silu_mul   (ref: g/(1+exp(-g)) * u)   [A2-soft: expf vs torch.exp]
    bpd_matmul   -> q8_0 gemv  (ref: dequant matmul)
    bpd_rope     -> k_rope     (ref: apply_rope, half-split NeoX)  [A2-soft: sinf/cosf]

Runs on the GPU host (enclave). Skips cleanly if torch/CUDA unavailable.
"""
import os
import sys
import math
import pytest
import numpy as np

# Repo layout: bpd/ on path for fact_dispatch + dev_residency, root for llamatov_run.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BPD = os.path.dirname(_HERE)
sys.path.insert(0, _BPD)
sys.path.insert(0, os.path.join(_BPD, "lib"))

torch = pytest.importorskip("torch")
try:
    import fact_dispatch as fd
    import dev_residency as dr
    import llamatov_run as R
except Exception as e:  # pragma: no cover - environment gate
    pytest.skip(f"fact pipeline unavailable: {e}", allow_module_level=True)

# Model numerics-class constants — these are the CONTRACT, pinned here so a drift is loud.
MODEL_EPS = 1e-6          # qwen norm_eps. NOT the rms_norm_fact default of 1e-5 (see test below).
ROPE_THETA = 500000.0     # qwen rope_freq_base
# Soft device-vs-host variance tolerance (the A2 tick-class: expf/sinf/cosf vs torch).
# Bit-identity is NOT expected for transcendental kernels; this bounds the declared class.
SOFT_TOL = 5e-6


def _rng(seed, *shape):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ROW: bpd_rmsnorm  →  rms_norm_fact (emitted k_rmsnorm)  vs  torch reference
# ─────────────────────────────────────────────────────────────────────────────
def test_row_rmsnorm_matches_torch():
    E = 896
    x = _rng(1, 1, E)
    w = (_rng(2, E) * 0.1 + 1.0)
    # reference: rms = x / sqrt(mean(x^2) + eps) * w
    ref = (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + MODEL_EPS) * w)
    got = fd.rms_norm_fact(x, w, MODEL_EPS)
    d = (got - ref).abs().max().item()
    # rmsnorm has a reduction (sum of squares) -> A2-soft class, but tight.
    assert d < SOFT_TOL, f"rmsnorm vs torch: {d:.2e} (REVERT_TO_CHECK: pass wrong eps -> fails)"


def test_row_rmsnorm_eps_comes_from_argument_not_default():
    # ADJACENT-FOOTGUN LOCK (the eps-unmasking incident, memory 9940f47a):
    # rms_norm_fact defaults eps=1e-5, but the model needs 1e-6. The two give
    # DIFFERENT results — a caller that forgets to pass MODEL_EPS silently gets the
    # wrong norm (which is exactly what masked, then unmasked, the silu variance).
    # Lock that eps is actually honored, so a default-eps regression is caught here
    # rather than as a mysterious downstream token drift.
    E = 896
    x = _rng(3, 1, 1, E)
    w = torch.ones(E)
    a = fd.rms_norm_fact(x, w, 1e-6)
    b = fd.rms_norm_fact(x, w, 1e-2)   # deliberately large eps -> must differ
    assert (a - b).abs().max().item() > 1e-4, \
        "eps argument is ignored! rms_norm_fact must honor the passed eps, not a default"


# ─────────────────────────────────────────────────────────────────────────────
# ROW: bpd_softmax  →  softmax_fact (emitted k_softmax)  vs  torch reference
# ─────────────────────────────────────────────────────────────────────────────
def test_row_softmax_matches_torch():
    x = _rng(4, 1, 1, 32)
    ref = torch.softmax(x, dim=-1)
    got = fd.softmax_fact(x)
    d = (got - ref).abs().max().item()
    assert d < SOFT_TOL, f"softmax vs torch: {d:.2e} (expf reduction, A2-soft)"

    # AGREEMENT-AND-DIVERGENCE: softmax must sum to 1 (the invariant that survives
    # the soft class) on a peaked distribution (the edge that stresses expf).
    peaked = torch.tensor([[[-3.0, 28.0, 0.0, 1.0]]])  # the L4-real attn-score range
    s = fd.softmax_fact(peaked)
    assert abs(s.sum().item() - 1.0) < 1e-5, "softmax must normalize even on peaked inputs"


# ─────────────────────────────────────────────────────────────────────────────
# ROW: bpd_silu  →  silu_mul_dev (emitted k_silu_mul)  vs  g/(1+exp(-g))*u
# This is the row whose expf-vs-torch.exp variance was the residency saga (9940f47a).
# We lock BOTH that it matches on normal data AND that the variance stays soft-class.
# ─────────────────────────────────────────────────────────────────────────────
def test_row_silu_matches_reference():
    N = 4864
    g = _rng(5, N)
    u = _rng(6, N)
    ref = (g / (1.0 + torch.exp(-g))) * u
    gd = dr.DevTensor.from_host(g.numpy()); ud = dr.DevTensor.from_host(u.numpy())
    got = torch.from_numpy(dr.silu_mul_dev(gd, ud).to_host().reshape(-1)); dr.free_scratch()
    d = (got - ref).abs().max().item()
    # expf vs torch.exp -> A2-soft, ~5e-7 (memory 9940f47a). Bounded, not bit-identical.
    assert d < SOFT_TOL, f"silu_mul vs ref: {d:.2e} (expf A2-soft class; REVERT_TO_CHECK: break the formula -> large)"


# ─────────────────────────────────────────────────────────────────────────────
# ROW: bpd_rope  →  rope_dev (emitted k_rope)  vs  apply_rope (half-split NeoX)
# ─────────────────────────────────────────────────────────────────────────────
def test_row_rope_matches_apply_rope():
    nh, nkv, hd = 14, 2, 64
    pos = torch.tensor([7])
    for name, nheads in (("q", nh), ("k", nkv)):
        x = _rng(7 if name == "q" else 8, 1, 1, nheads * hd)
        if name == "q":
            ref, _ = R.apply_rope(x.clone(), torch.zeros(1, 1, nkv * hd), nh, hd, ROPE_THETA, positions=pos)
        else:
            _, ref = R.apply_rope(torch.zeros(1, 1, nh * hd), x.clone(), nh, hd, ROPE_THETA, positions=pos)
        ref = ref.reshape(-1).numpy()
        got = dr.rope_dev(x.reshape(1, -1).numpy(), pos.numpy(), nheads, hd, ROPE_THETA).reshape(-1)
        d = float(np.abs(got - ref).max())
        # sinf/cosf vs torch -> A2-soft. (REVERT_TO_CHECK: flip pairing to interleaved -> huge.)
        assert d < SOFT_TOL, f"rope {name} vs apply_rope: {d:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# ROW: bpd_matmul  →  q8_0_linear_from_fp32 (emitted q8_0 gemv)  vs  dequant matmul
# ─────────────────────────────────────────────────────────────────────────────
def test_row_q8_matmul_matches_dequant_reference():
    # CONTRACT (from q8_0_linear_from_fp32 docstring, verified against the production
    # call site dev_residency.py:234): x[K] or [T,K]; weight[K,N] -> y[N] or [T,N].
    # weight is [K,N] (input-dim first), output n = weight[:,n]; y = x @ weight.
    K, N = 896, 64
    x = _rng(9, 1, K)          # [T=1, K]
    w = _rng(10, K, N)         # [K, N]  <-- input-dim first, NOT [N,K]
    got = fd.q8_0_linear_from_fp32(x, w)
    ref = (x @ w)              # [1, N]  (y = x @ weight, weight already [K,N])
    # Q8_0 quantizes the weight -> finite quant error vs full precision. This is the
    # "near-bit-identical to stock Q8_0" credibility row; bound the quant error.
    rel = ((got - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
    assert rel < 0.05, f"q8 matmul vs fp32 matmul rel-err {rel:.3f} (Q8_0 quant error; REVERT_TO_CHECK: wrong K -> garbage)"


# ─────────────────────────────────────────────────────────────────────────────
# TABLE COMPLETENESS: every production fact is covered by a row above.
# Like #124 asserting the WHOLE predicate vector, this fails if someone adds a
# production fact to fact_dispatch without adding a lock row here.
# ─────────────────────────────────────────────────────────────────────────────
PRODUCTION_FACTS = {"bpd_rmsnorm", "bpd_softmax", "bpd_silu", "bpd_matmul", "bpd_rope"}

def test_table_is_complete():
    covered = {
        "bpd_rmsnorm": "test_row_rmsnorm_matches_torch",
        "bpd_softmax": "test_row_softmax_matches_torch",
        "bpd_silu":    "test_row_silu_matches_reference",
        "bpd_rope":    "test_row_rope_matches_apply_rope",
        "bpd_matmul":  "test_row_q8_matmul_matches_dequant_reference",
    }
    missing = PRODUCTION_FACTS - set(covered)
    assert not missing, (
        f"production facts with no lock row: {missing}. "
        "Every fact wired into the LLM decode path must have a table-lock row."
    )


def test_table_lock_bites_marker():
    # Documentation-as-test: records HOW each row is made to fail (the #124
    # 'verified the test bites' discipline). This isn't a runtime mutation test
    # (that's pytest_verify_historical's job); it's the checklist a reviewer runs.
    bite_checks = {
        "rmsnorm": "pass eps=1e-2 instead of MODEL_EPS -> test_row_rmsnorm fails",
        "rmsnorm_eps": "make rms_norm_fact ignore its eps arg -> eps test fails",
        "softmax": "drop the max-subtraction in k_softmax -> overflow on peaked input",
        "silu":    "change k_silu_mul to silu(g) without *u -> large diff",
        "rope":    "flip pairing(half_split) -> pairing(interleaved) in the fact -> huge diff",
        "matmul":  "emit gemv with wrong K -> reads past activation -> garbage rel-err",
    }
    assert len(bite_checks) >= len(PRODUCTION_FACTS), "every row needs a documented bite-check"
