# Milestone: 0-ULP Generated Inference Engine Surpasses Ollama on a Tesla P4

**Date:** 2026-06-12
**Branch:** `iyun/gpu-cuda-oxide`
**Model:** Qwen2.5-0.5B-Instruct, Q8_0 (8-bit) quantization (`qwen2.5:0.5b-instruct-q8_0`)
**Hardware:** NVIDIA Tesla P4 (Pascal, sm_61, ~20 SMs, GDDR5)
**Author:** Iyun (Ruach Tov collective), with Heath Hunnicutt

---

## Headline Result

A transformer inference engine **generated from declarative Prolog facts** (not hand-written CUDA)
runs Qwen2.5-0.5B Q8_0 at **~156.6 tok/s** on a Tesla P4, versus **~144.5 tok/s** for Ollama
(llama.cpp/ggml) on the *same hardware and model* — **~108% of Ollama** — while producing
**token-identical, bit-identical (0-ULP) output**.

| Engine | tok/s (warm, 100-token, P4, Q8_0) | Correctness |
|---|---|---|
| **Ours (generated, BPD)** | **~156.6** | 0-ULP, token-exact to reference |
| Ollama (llama.cpp/ggml) | ~144.5 | reference |

The claim is one of **structure, not degree**: not "comparable quality within epsilon," but
*faster AND bit-identical — run `diff` yourself*. There is no tolerance for a skeptic to pull on.

> Measurement discipline note: both numbers are measured warm, `num_predict=100`, same P4, same
> model. The fully unassailable comparison is an **interleaved A/B harness** (same loop, back-to-back,
> one JSON), which is being built (the "Ollama arm"). Until that lands, this is "our careful
> measurement vs Ollama's careful measurement under matched conditions" — honest, and pending the
> interleaved instrument for the bulletproof form.

---

## The Arc (3 days)

`9.26 → 15.20 → 21.89 → 115 → 120 → 138 → 143.41 → 144.44 → ~156.6 tok/s`

A ~17-fold improvement, **every rung certified** — each optimization is bit-exact-by-construction,
0-ULP-against-a-canonical-reference, or matched-quantization-proven. The engine never traded
correctness for speed.

---

## What Was Implemented (this milestone push)

All optimizations are **generated from facts**, never hand-copied into kernels. The architecture
separates two kinds of fact:
- **Correctness contract** (`reduction_order`) — what gets summed, in what order. The 0-ULP law.
- **Performance contracts** (`weight_access`, `epilogue_compute`) — how bytes are fetched, what
  compute is folded into a store. Orthogonal to correctness; each backend lowers them in its own
  idiom (CUDA / Rust-oxide / Torch-reference), so an optimization's *intent* translates to **all**
  generated backends.

### 1. 128-bit vector weight loads (`mode(tiled_v4)`) — commit `10557759`
CUPTI-from-Prolog stall counters *prescribed* the lever: the tiled GEMV's texture stall (22%) was
narrow 32-bit weight loads. Switched to 128-bit vector loads (CUDA `int4` type = 4×int32 = 16 bytes
— **a load width, NOT 4-bit quantization; the model is Q8_0/8-bit throughout**). Texture stall
collapsed 22% → 2.7%, ~1.45× on the GEMV, BIT-EXACT. → 138 tok/s.

### 2. Canonical-order GEMV (0-ULP) — commit `3acd315c`
Declared `reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma), tree(shfl_down,5))` as a
fact and rendered a serial reference from it; pair-gate = 0 ULP. The residual epsilon was
**FMA contraction** (the per-block multiply-add fuses to one rounding) — making it explicit in the
declared order is the thesis at its sharpest. The decode path became **0-ULP top to bottom**
(rmsnorm ∞-safety, quant bit-exact, GEMV canonical, argmax bit-exact).

### 3. Residual-add epilogue (`tiled_v4_addres`) — commit `20863d39`
Fold `Y[row] = gemv + residual[row]` into the GEMV store. Element-local, bit-exact, eliminates 720
`k_add` launches/decode-window. +5.9% (135.35 → 143.41 tok/s).

### 4. QKV + gate/up fusion — commit `6716ac42` (the big occupancy win)
q/k/v read the same input (attn-norm output) with different weights; same for gate/up. Concatenate
the weights along the output dim → ONE GEMV → slice into device views. The win is **occupancy**:
k/v projections were 8-block launches (starved on a ~20-SM GPU); fused, q+k+v = 1152 rows = 72
blocks. Activation quantized once, shared by all rows. BIT-EXACT (concatenation only adds rows;
each row's reduction is unchanged). +7.3% e2e, token-exact. This was the decisive push past Ollama.

**Provenance:** this came from chasing the attn `constant_memory` stall (20.7%), which
register-hoisting could *not* fix (the SASS showed Pascal's XMAD wants the constant-bank operand).
The stall was a **proxy for small-shape occupancy starvation** — Heath's instinct that "the
statistic may be hiding other inefficiencies." QKV fusion was the real fix.

### 5. Backend-neutral performance axes — commits `207f5005`, `af42bd0b`
- `weight_access` [lane_strided, warp_contiguous] — a coalescing knob. **Corrected finding:**
  lane_strided is *already* near-optimal (1.2 sectors/load ≈ ideal for 16-byte loads); the GEMV runs
  at **86% of the P4's *achievable* 141.5 GB/s** (not 63% of the theoretical 192). The GEMV is at
  the DRAM wall — the same wall ggml/Ollama hit. (Lesson: measure achievable, not theoretical.)
- `epilogue_compute` [none, add_resid, bias, silu, bias_silu] — the "hide compute under memory"
  axis. A memory-bound GEMV leaves the SMs idle during DRAM stalls, so element-local compute folded
  into the store runs in that shadow. Measured: SiLU (a transcendental) folds into the gate GEMV at
  **+0.2%** — the `exp` is hidden. Validated, 0-ULP-gated (`gate_tiled_v4_silu.py`, max_ulp=0),
  available as infrastructure (not wired in the current config — it conflicts with gate/up fusion).

---

## What We Measured (and the negative results that mattered)

The methodology was a closed loop: **measure (CUPTI stall counters) → prescribe → generate (emitter
mode) → verify (bit-exact gate + re-measure)**. Three optimizations were *vetoed by measurement*
before shipping — each a valuable negative result:

- **quant→GEMV fusion (qfused):** bit-exact but **0.55× slower** — re-quantizes the activation
  N/BM times. *Validity ≠ profitability.* (commit `36c8832f`)
- **register-hoisting** the constant_memory stall: SASS showed the compiler already hoists; Pascal
  XMAD prefers the constant-bank operand. *The stall label's prescription was futile; the SASS
  adjudicated.*
- **coalesced-load rearrangement (`warp_contiguous`):** the sector arithmetic + an achievable-
  bandwidth measurement showed the GEMV is already at the wall. *Measure achievable, not theoretical.*

Bandwidth facts (measured on the P4):
- Achievable DRAM bandwidth (stream copy): **141.5 GB/s** (= 74% of theoretical 192, normal GDDR5)
- Vocab GEMV: **121 GB/s = 86% of achievable** — at the physical wall
- sectors_per_load (tiled v4): **1.2** (near-ideal for 16-byte loads)

Post-fusion profile (CUPTI trace): `k_q8_0_gemv` 52.7%, `k_q8_0_gemv_addres` 19.5%,
`k_attn_decode_masked` 7.8% (the new #3 bottleneck, surfaced by removing projection-launch noise),
`k_rmsnorm` 5.8%, `k_rope` 4.5%, `k_quant_q8` 4.5%. GEMV launches/token: ~169 → ~97.

---

## The Frontier Beyond (mapped, per the publish discipline)

- **The GEMV is at the DRAM wall** — bandwidth is not a further lever (matching Ollama's floor).
- **Real remaining headroom is in non-GEMV compute** — chiefly `k_attn_decode_masked` (7.8%),
  which is compute-bound, not bandwidth-bound.
- **`epilogue_compute` compute-hiding** has more reach (e.g. bias-into-QKV) where it doesn't
  conflict with an existing fusion.
- Byte-reduction (e.g. Q4) is off the table — it would break the 0-ULP correctness contract.

---

## Why This Matters

The contribution is not "we wrote a fast kernel." It is a **system that generates, measures,
prescribes, and certifies its own kernels** — a closed loop where CUPTI stall counters (read from
Prolog) prescribe optimizations, an emitter renders them from declarative facts, and a bit-exact
gate certifies them — and the result *beats the incumbent, bit-for-bit, on a ~$90 commodity GPU*.

The optimizations live as **backend-neutral facts**, so the same intent lowers to CUDA, Rust, and
a Torch reference. Correctness (`reduction_order`) and performance (`weight_access`,
`epilogue_compute`) are separate, orthogonal contracts — which is *why* a performance change can be
proven 0-ULP by construction across every backend.


---

## Post-Milestone Addendum (same day): ~157.4 tok/s via small-piece tidying + a hardening fix

After the milestone, we kept measuring *past* the point we stood on (the publish discipline:
map the frontier beyond the result). Two things came of it: a clean further ~+2.2%, and a
structural safety fix that closed a dangerous failure mode.

### The frontier, mapped by rigorous veto (5 levers ruled out, not guessed)
The GEMV families are **72-74% of decode and at the DRAM wall** — proven by *ablation*, not
assumed. Five candidate levers were each built and **measurement-vetoed**:
1. quant→GEMV fusion (qfused): 0.55× (re-quantizes the activation N/BM times).
2. register-hoisting the constant_memory stall: the compiler already hoists (SASS-confirmed).
3. coalesced-load rearrangement (warp_contiguous): GEMV already at 86% of *achievable* 141.5 GB/s.
4. split-K attention at short L: loses (combine overhead) — banked for long-context (wins L>~100).
5. vocab-sync removal: 0.48× — the staging *amortizes* the activation reads 16-fold; the sync
   stall is the **price of a net-positive trade**, not reclaimable.

This yielded a doctrine refinement: **stall-reason ≠ stall-opportunity.** The counter prescribes
*where* the time goes; only the ablation adjudicates whether it is *reclaimable*.

### The small-piece tidying (+2.2%, every fusion bit-exact AND token-exact)
The "down in the noise" bookkeeping/bias kernels (~1% each) compounded when folded — *more* than
their profile share, because eliminating a launch per layer (24×/token) cuts graph-replay overhead
disproportionately:

| Fusion | commit | win | what it folded |
|---|---|---|---|
| bias-into-GEMV | `650f902d` | +0.86% | the q/k/v bias `k_add` → the GEMV store (via the addres kernel, bias as "residual") |
| K/V append fusion | `e775e0ba` | +0.78% | two twin `k_append_at_len` launches → one `k_append_kv` |
| append + incr fusion | `d2fb0979` | +0.94% | `k_incr_len` (`*len += 1`) → folded into `k_append_kv_incr` |

Result: **~154 → ~157.4 tok/s** (~109% of Ollama). The profile is now the cleanest of the session:
**10 kernels (was 12), −16% launches** (20,970 → 17,514). The bookkeeping kernels (`k_add`,
`k_incr_len`, one append) are gone. The small pieces are genuinely exhausted — every remaining
kernel is at the hardware floor (GEMVs), banked for another regime (attn split-K), or irreducible
compute (rmsnorm, rope, quant).

### A trap caught, then disarmed structurally (`db19f10b`)
The append-fusion's *first* measurement looked like **+2.2 tok/s** — but `_build_inline` had cached
a **stale cubin** (built before the new kernel existed), so the "fused" kernel was a **no-op that
silently corrupted the KV cache** — "faster" only because it skipped its writes. A **false speedup
that was actually a correctness break** (the inverse of every honest-red bug — a *dishonest green*).
The bit-exact gate caught it (`token-exact: False`, written bytes `[0,0,0]`). This is precisely why
**gate-before-claiming** is load-bearing — without it, a correctness break ships as a "win."

The fix (structural, not a workaround): `_build_inline` now **content-addresses** the cubin cache
(key = name + SHA1(source)). A source change yields a different cubin path, so a stale cubin can
**never** silently no-op a new kernel again. The trap is permanently disarmed for all inline kernels.

### Certification
GR full-stack VERIFIED 6/6 bit-exact at both the milestone pin (`d82d421c`) and the post-tidy HEAD
(`d2fb0979`) — the three fusions are graph-equivalent to their eager selves. The HARD-class
full-vs-full long-run (all fusion toggles ON vs OFF, same quantization both arms, SF=∞ expected)
runs via the per-fusion env toggles (`BPD_BIAS_FOLD`, `BPD_APPEND_KV_FUSED`, `BPD_APPEND_INCR_FUSED`).

### What this chapter shows
The milestone was not a ceiling but a *floor we then mapped*. The dominant cost (GEMV) is at the
hardware's physical limit — proven by ablation. The remaining tok/s came from tidying the smallest
pieces (bit-exact, compounding), and the process *hardened the engine* (the content-addressed cache)
and *sharpened the doctrine* (stall-reason ≠ stall-opportunity; validity ≠ profitability ≠ priority).
Every claim gated; every negative result recorded; the frontier honestly mapped.


---

## Official Certification (2026-06-12, evening): the small-pieces +2.2% is HARD-class certified

The three small-piece fusions (bias-into-GEMV, K/V append, append+incr) are **officially certified
on the production config** via the complete artifact chain:

1. **Kernel XOR=0 pair-gates** (per-fusion, bit-exact in isolation)
2. **GR full-stack green** (`d2fb0979`, `db19f10b` — 6/6 bit-exact, graph≡eager)
3. **Profile-pinned HARD-class long-run, SAFETY FACTOR = ∞** (`/tmp/longrun_official.json` at
   `c9eda457`): production profile both arms ± the three fusion toggles, 6 prompts × 30 tokens =
   180 decisions. **Full agreement every prompt; max_drift = 0.000e+00 across all 180; near-flip
   census NONE.** The thinnest decision in the run (margin 0.0027, a new record thin-tie, beating
   the sentinel's 0.0038) sat over ZERO drift.
4. **Composition gate** (`5c5228a3`) refusing the only divergent world (QFUSED × fusion-toggle).

### The named production profile + the config-profile gate
`apply_production_profile()` / `attest_profile()` (`74b144b8`) make the certified ~157.4 tok/s
ensemble a **single named source of truth** — self-attesting, function-call-set (no env ambiguity).
`assert_certified_composition()` (`5c5228a3`) refuses uncertified compositions loudly at `seed()`.

### Five comparison coordinates, enforced structurally
The hunt that produced this cert revealed that an oracle's identity has **five coordinates**, each
now enforced: **path** (attested), **quant** (profile-matched), **shape** (production
seed/capture/replay, T=1 decode AND T=3 prefill), **ensemble** (profile-pinned), and **mechanism**
(function-call, self-attesting — env-vs-module-attr was the last invisible coordinate).

### Three dishonest-silent-paths, structurally killed
The hunt surfaced and eliminated three "silent lie" failure shapes — each given a *structural
impossibility proof*, not just a fix:
1. **stale-cubin dishonest-green** (a no-op kernel as a false speedup) → content-addressed cubin
   cache (`db19f10b`)
2. **crash-into-host-fp32 dishonest-fallback** (a false oracle) → loud `_GRAPH_PREP` precondition
   assertion (`4bd44ce4`)
3. **bias-toggle silent-kernel-switch dishonest-composition** → config-profile refusal (`5c5228a3`)

The principle: *honest reds are cheap; silent lies are the expensive class. An engine must never
silently switch its correctness contract — the silence is the gun.*

**Zero production bugs throughout.** The certified config (QFUSED=False, fusions-ON, v4) was correct
the entire time; the entire hunt was about making the *comparison* as honest as the engine already
was — and ended with the lessons compiled into *structure*: the next engineer (or the next session
after a context reset) cannot make these three mistakes even by trying.


---

## Chain-Audit Fusions (2026-06-12, Heath driving): ~157.8 → ~161.3 tok/s (~111.6%)

After the SF=∞ certification, Heath directed a compute-chain audit: *"look over the chain for fusion
that eliminates launches; consider whether whole CUDA Contexts could be merged, or any malloc/memcpy
that can be moved outside the inference."*

### Audit findings (the structural items were already optimal)
- **CUDA contexts:** a single context (`fd._ctx()`, cached). Nothing to merge.
- **malloc:** the `_SlabArena` allocates the device arena ONCE and bump-allocates — the per-layer
  `cuMemAlloc`/`cuMemFree` (the latter *synchronizes*) were already hoisted out.
- **HtoD:** weights are device-cached (`_dev_const`/`_dev_weight`); captured-graph replay has zero
  host involvement (even `len_ptr` is device-resident). No per-token re-upload.

### Three launch/round-trip-elimination fusions (all bit-exact, token-exact)
| Fusion | commit | win | what it eliminated |
|---|---|---|---|
| residual-carry | `b31b002f` | +0.63% | the per-layer `_memcpy_dtod` — the down-proj writes its (gemv+resid) result DIRECTLY into the fixed `_resid_carry` buffer (the next layer's input) via a new `out=` param. 24 graph nodes/token gone. |
| rope-QK | `ff651499` | +1.13% | one of the two rope launches — q and k are CONTIGUOUS slices of the fused-QKV output, roped identically, so one launch over (nh+nkv)=16 heads ropes both (guarded by an explicit contiguity check). |
| silu-into-quant | `220baed0` | +0.87% | `k_silu_mul` + the `gu` global round-trip — a new `k_silu_mul_quant` computes silu(g)*u and quantizes in one kernel, feeding the silu'd values straight into the warp-amax+quantize. |

**Combined: ~157.8 → ~161.3 tok/s, +2.6%, all bit-exact and token-exact.**
Now **~161.3 tok/s = ~111.6% of Ollama (144.5)** — the `110pct` tag is comfortably exceeded.

### Notes
- The silu-into-quant fold is the **gate/up-fusion-compatible** cousin of the earlier-vetoed
  `tiled_v4_silu` epilogue: folding silu into the QUANT side (not the GEMV side) sidesteps the
  conflict with gate/up fusion that vetoed the epilogue version. A lever blocked from one direction
  opened from another.
- Each fusion is a named toggle in `PRODUCTION_PROFILE` (`_RESID_CARRY_FUSED`, `_ROPE_QK_FUSED`,
  `_SILU_QUANT_FUSED`, all default ON) — so the config-profile gate covers them, and the referee
  re-certifies the new profile.
- The post-audit per-layer chain: `rms_norm → qkv_gemv(+bias) → rope_qk(fused) →
  append_kv_incr(fused) → attn_decode → o_gemv(+resid) → rms_norm → gateup_gemv →
  [silu+quant fused into down] → down_gemv(+resid→carry)`. Tight — no obvious remaining
  launch-elimination fusion (o-proj has no elementwise to fold; rms_norm/attn_decode are their own
  necessary passes; attn_decode is the banked split-K territory).
