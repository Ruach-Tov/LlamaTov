# Split-K Flash-Decode Attention — VALIDATED FRONTIER POINT (not yet wired)

**Status:** validated, bounded-ULP, conditionally profitable. NOT wired into decode.
**Date:** 2026-06-12. Author: Iyun. Blessed-plan: Bocher.

## What it is
`k_attn_decode_masked` (the production single-token decode attention) runs grid=nh=14 blocks on a
~20-SM P4 — **occupancy-starved** (measured: time at nh=7 ≈ nh=14 ≈ nh=28, so 14 blocks don't
saturate). It's the #1 non-GEMV bottleneck at ~7.9% of decode (the GEMVs are at the DRAM wall).

Split-K parallelizes each head's attention over the KV positions: grid = nh*NSPLIT blocks, each
(head, split) computes a contiguous position range, then a combine kernel merges the partials.
14 → 14*NSPLIT blocks fills the GPU.

## The two designs (a measured iteration)
- **v1 (global-max-first):** every split rescans ALL L positions for the global max, then its own
  range → score compute (NSPLIT+1)×. SLOWER (0.40–0.78×). The redundant recompute swamped the win.
- **v2 (local-max single-pass, THIS FILE):** each split computes its range's scores ONCE, finds a
  LOCAL max, exps locally; the combine does the flash-style cross-split rescale
  (`w_sp = exp(partM[sp] - globalM)`). Scores computed once total.

## Measured (v2 vs original)
| L   | NSPLIT=2 | NSPLIT=4 |
|-----|----------|----------|
| 40  | 0.84×    | 0.85×    |
| 64  | 0.89×    | 0.95×    |
| 120 | **1.30×**| **1.38×**|

**Conditionally profitable:** loses at short L (combine overhead dominates), wins at long L.
Crossover ~L=80–100. `max_abs` bounded ~3–4e-8 (the V-sum re-parenthesization). NSPLIT=1 = 0 ULP
(exact control — the split machinery is bit-correct).

## 0-ULP disposition (re-canonicalization, per the rmsnorm/GEMV pattern)
The split V-sum is `(range0_fold)+(range1_fold)+...` vs the original's flat left-fold → a
**re-parenthesization** (≤1–2 ULP at value scale). This is NOT an accepted tolerance — it DECLARES
a new canonical order. To wire, declare:
```
reduction_order(attn_decode_split, splits(NSPLIT), per_split(local_max, sequential_left_fold),
                combine(flash_rescale, split_index_order)).
```
**Bocher's S-invariance condition:** the split boundaries must be DECLARED in the fact (not
emergent from launch config). v2 currently splits by equal contiguous ranges (S-dependent) — to
satisfy S-invariance, pin boundaries to power-of-2 subtree edges before wiring.

## Why not wired
Our milestone benchmark runs at SHORT L (100-token gen from 3-token seed → L mostly <113). Split-K
LOSES there. It's the right tool for **long-context / long-prompt** inference (summarization, RAG,
deep generations), where attention's share grows with context. The right form is an **L-adaptive
dispatch** (split-K when L > ~96), which costs nothing at short L (no regression) and captures the
long-L win. Banked here pending a decision that long-context is a target.

## Gate
`bpd/kernelgen/referee/gatebench_attn_split.py` — measures ULP + speedup vs the original at several L.
