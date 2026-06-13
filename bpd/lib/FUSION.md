# The fusion framework — validity + profitability

Fusion combines adjacent ops to eliminate intermediate-tensor traffic. The
framework has **two dimensions**, both required:

1. **Validity** — *can* we fuse? (region-compatible, non-escaping, class-compatible)
2. **Profitability** — *should* we, given the shape? (bytes saved vs recompute cost)

> **Validity ≠ profitability.** A fusion can be perfectly correct yet *slower*.
> The im2col-into-GEMM conv fusion was the worked example: bit-correct but
> measured 22.8ms fused vs 15.8ms 2-stage on the P4 — recompute exceeded the
> memory saved. A generic optimizer must *decline* such fusions.

## The modules

| Module | Role |
|--------|------|
| `fusion_rules.pl` | `fusion_rule(Name, Pre, Post, EquivClass)` — CHiLL-style rules |
| `symbolic_fusion.pl` | **validity**: `fusion_valid/2`, `region_matches/2`, `no_escape/2`, `op_class_compatible/2` |
| `op_classify.pl` | **general op classifier**: `classify_expr/2` derives an op's class (elementwise/reduction/spatial/normalization/matmul/layout) from its op_expr AST — backend-independent |
| `fusion_analyzer.pl` | `find_fusible_chains/2`, `can_fuse/3` — operates on the general bpd ops (via `op_classify`) *and* ggml ops (backward-compat table) |
| `fusion_cost.pl` | **profitability**: `fusion_profitable/3` |
| `iterative_fusion.pl` | `fixpoint_fuse/4` — applies rules to fixpoint, **gated by `fusion_profitable`** |
| `apply_fusion.pl` | the rewrite |

## The cost model (`fusion_cost.pl`)

`fusion_profitable(Rule, Bindings, Verdict)` → `always | profitable(M) | unprofitable(D)`.

- **Epilogue / elementwise-chain / layout** rules → `always` (recompute factor 1,
  trivial per-element work — no gate needed).
- **Generator-prologue** rules (im2col / gather / broadcast-expand) → cost-gated:

  ```
  profitable  iff  recompute_factor × gen_regen_bytes  <  bytes_saved (= 8/elem)
  ```

  - `recompute_factor(gemm_b_input(M,BM))` = `ceil(M/BM)` — a generated GEMM
    B-tile is recomputed once per output M-block.
  - `gen_regen_bytes` (calibrated on the P4): **im2col = 9** (> the 8 it saves:
    index ALU + non-coalesced gather), **broadcast/copy = 2** (cache-friendly
    re-read), **elementwise = 2**.

  → im2col is unprofitable at any factor ≥ 1 (matches measurement). broadcast is
  profitable up to factor 3.

## The gate (`iterative_fusion.pl`)

`fixpoint_iterate` filters enumerated (valid) fusions through
`fusion_is_profitable` *before* applying:

```prolog
enumerate_with_facts(Rules, Facts, AllFusions),
include(fusion_is_profitable(Facts), AllFusions, Fusions),   % the gate
... apply first profitable fusion ...
```

`fusion_is_profitable/2` dispatch (clause order matters — generator_prologue with
cut first, so the always-rule probe never queries the cost model with unbound
bindings):
- `generator_prologue` → consult `fusion_cost` with bindings from the facts.
- `epilogue / elementwise_chain / layout` → `always`, pass through.

## Tests

- `tests/test_symbolic_fusion.pl` — validity (5)
- `tests/test_apply_fusion_multi_rule.pl` — multi-rule application (8)
- `tests/test_fusion_profitability_gate.pl` — the cost gate (7): im2col declined,
  elementwise/epilogue/layout allowed, broadcast factor-1 allowed / factor-8 declined
- `tests/test_kernelbench_l2.pl` — fusion coverage on the 100 L2 problems, with
  train/test rule generalization

## The lesson baked in

Fusion that *doesn't* pan out still teaches the system something durable. The
im2col negative result produced the cost model, which now prevents the whole
*class* of unprofitable generator-prologue fusions going forward.
