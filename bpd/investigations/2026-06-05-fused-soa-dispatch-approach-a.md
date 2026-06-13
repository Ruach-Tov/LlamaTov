# Fused SoA Dispatch — Approach A (Chained Two-Launch)
## 2026-06-05 evening, mavhir at Heath's direction

## Heath's directive

> "Start working on the fused SoA dispatch (wiring fusion_local.gate → our
> FUSE_SILU kernel path) and focus first on bit-identical correctness
> between our SoA compute graph and stock ggml."

## What was built

Tonight's substrate at `<external-build-tree>/ggml-src/src/ggml-cuda/`:

- **`soa_dispatch_fused.inc`** (new) — chained two-launch fused dispatch:
  - Allocate scratch buffer via `ggml_cuda_pool_alloc<float>(ctx.pool(), nrows)`
  - Launch 1: `gemv_soa<128, FUSE_SILU=1>` on gate weight (SoA) → scratch
    (writes `silu(gate_dot)` per row)
  - Launch 2: `gemv_soa<128, FUSE_SILU=0>` on up weight (SoA), passing
    `up_result=scratch` and (if present) `residual=x_bias_col` → dst
    (computes `up_dot * silu(gate_dot) + x_bias`)
- **`soa_dispatch_residual.inc`** (new) — handles the x_bias-only-no-gate case.
- **`mmvq.cu` gate logic** — lifted the fusion-args exclusion; now enters
  SoA path when shadow registered AND (no fusion, OR SWIGLU fusion handleable:
  gate shadow exists + no gate_bias). Unsupported fusion shapes fall through
  to stock (preserves correctness fix from `dae6d508e`).
- **`soa_shadow.cuh`** — pre-existing infrastructure, no changes. The
  `soa_preallocate_shadows` already registers every Q8_0 weight with
  `ne[1] <= 16384` from the graph, including gate weights — so `soa_shadow_lookup(fusion_local.gate)` returns a valid entry.

## Empirical results

### Correctness

End-to-end on Llama-3.2-1B Q8_0, Tesla P4, prompt "Hello":

- STOCK: "Hello! How can I assist you today?"
- SoA + Approach A fused dispatch: "Hello! How can I assist you today?" ✓

Tokens match stock exactly.

### Bit-identity (Heath's first-priority gate)

Element-wise comparison of `dst` for the fused FFN-up dispatch, captured via
`FUSED_STOCK_*` / `FUSED_SOA_*` probes in mmvq.cu (since stripped):

| Phase | Result |
|-------|--------|
| call=1 warmup blk.0-2 | 1800-2000/8192 bit-exact, max_abs ~1e-7 (per-matmul ~1-2 ULP) |
| call=1 warmup blk.3-15 | Cascade — max_abs grows 5e-4 to 1.3e-1 across layers |
| **call=2 REAL prefill blk.15** | **2333/8192 bit-exact, max_abs 7.15e-7 (~1-2 ULP)** |
| call=3 decode-1 blk.15 | Cascade, max_abs 5.9e-2 |

**Bit-identity NOT achieved.** Approach A inherits gemv_soa's underlying
~1-2 ULP drift vs stock's `vec_dot_q_cuda` (the kernel-level drift Iyun
measured tonight via her fixed harness vs CPU truth).

The chained-launch design itself is mathematically clean — the gate→up
multiplication transits global memory losslessly, and multiply is commutative
at IEEE 754. The divergence source is the per-matmul drift of gemv_soa vs
stock, NOT the chaining strategy.

### Performance

Tesla P4, `llama-bench`, Llama-3.2-1B Q8_0, 5 repetitions:

| Configuration | pp512 (prefill) | tg128 (decode) | Δ decode vs stock |
|---|---|---|---|
| **Stock** | 2998.38 ± 22 t/s | **90.92** t/s | baseline |
| **SoA bare** (fused→stock fallback = tonight's correctness fix only) | 2991.49 ± 19 t/s | **88.30** t/s | **−2.9%** |
| **SoA + Approach A fused** | 2931.63 ± 14 t/s | **84.19** t/s | **−7.4%** |

Consistent across run lengths (tg128/tg256/tg512 deltas identical within
noise).

**Observation**: hand-written SoA loses on Pascal/P4 at decode in BOTH
modes. The non-fused-only mode loses 2.9%; the fused-dispatch mode adds
another 4.5% loss (two kernel launches + pool alloc + global-memory
scratch round-trip vs stock's single-kernel parallel-accumulator design).

## Major empirical finding (resolves mavchin's open mechanism question)

The fused dispatch fires at **EVERY** `blk.N.ffn_up.weight` (all 16 layers
of Llama-3.2-1B), not just blk.15. Verified empirically via per-call
FUSED_STOCK / FUSED_SOA probe captures during both stock and SoA runs.
`fusion_local.gate` is non-nullptr at every FFN ffn_up dispatch.

This resolves the open mechanism question flagged in the
`bpd/patches/soa-fusion-gate-fix-2026-06-05.patch.md` patch note (commit
`9a552e829`): mavchin's earlier "all-nil at mmvq" measurement sampled
the wrong path or hit a graph-capture-vs-eager state difference. Tonight's
correctness fix `dae6d508e` was doing real work at 16 layers, not 1.

This ALSO corrects my own minute-30-of-the-box framing ("blk.15-only
fusion path is the bug surface") which was empirically wrong: the
fusion fires uniformly across the model, and the SoA path was silently
dropping it at every dispatch, not just blk.15. The "blk.15-only"
appearance came from the file-existence-guarded probes capturing only
the first dispatch per tensor name — those happened to hit blk.15 first
in some interleaving.

## What the empirical results mean for the larger trajectory

The hand-written gemv_soa has TWO failures vs stock:

1. **~1-2 ULP drift per matmul** vs stock's `vec_dot_q_cuda` (correctness
   gap — small enough that tokens stay correct, but not bit-identical
   by construction)
2. **−2.9% to −7.4% throughput** vs stock at decode (performance gap —
   the SoA coalescing premise is not winning on Pascal/P4)

Both are losses. Neither is what the SoA work was supposed to deliver.

The substantive case-for-the-re-vector (declarative pipeline) is now
strengthened by tonight's empirical evidence at TWO axes:

- **Bit-identity**: `swiglu_fused_emitter.pl` (commit `aee5312d4`,
  2026-06-03) already produces SwiGLU fusion bit-identically from facts
  (transitive 0 ULP, fused==unfused==ggml). Extending the same
  declarative substrate with a SoA-weight-load fact would give Q8_0
  SoA fused dispatch **0-ULP by construction** — eliminating the
  per-matmul drift Approach A inherits.
- **Performance**: the declarative substrate could choose AoS-vs-SoA
  per-platform per-shape based on measured perf, rather than committing
  to one hand-implementation that loses on Pascal. The lowering layer
  could emit either kernel-variant from the same facts.

The week of hand-fighting + tonight's perf measurement together provide
the empirical evidence for the declarative thesis: hand-construction
delivers both correctness AND performance fragility; declarative
lowering delivers both by construction.

## Open question (held for mavchin + Iyun)

The mechanism question raised in patch note `9a552e829` is **resolved
by tonight's measurement**: the correctness fix WAS doing real work
because fusion_local.gate IS non-null at every FFN ffn_up dispatch.
mavchin's prior "all-nil at mmvq" sampling missed the firing path.

This closes the open thread cleanly. The fix from `dae6d508e` stands
on the empirical fact: fusion fires uniformly across the model; the
SoA path silently dropped it at every dispatch; the one-line gate
restored correctness at every dispatch.

## State of the substrate (for handoff)

Files in play at `<external-build-tree>/ggml-src/src/ggml-cuda/`:

- `mmvq.cu` — modified to add fused-dispatch handling at lines ~1213-1255.
  Original fusion-gate fix preserved as fallback path.
- `soa_dispatch_block.inc` — unchanged (bare-matmul SoA path).
- `soa_dispatch_fused.inc` — new, contains Approach A's chained two-launch.
- `soa_dispatch_residual.inc` — new, handles x_bias-only case.
- Build at `<external-build-tree>/libggml-cuda.so.0.13.1`.

The build is held untouched per Iyun's guidance — mavchin+Iyun's
mechanism question (now resolved) was on this exact build, and the
artifact is substrate-of-record.

## Decision points (awaiting Heath)

1. Commit Approach A to bpd-substrate or canonical bpd/ tree, or keep
   as enclave-only artifact-of-investigation?
2. Strip Approach A from enclave (revert to just the correctness fix
   from `dae6d508e`)?
3. Begin re-vector to the declarative path (extend `prolog_to_llvm.pl`'s
   `emit_q8_0_dot` with the SoA-weight-load fact + matmul chain)?

The empirical evidence supports option 3 as the substantive next move:
both correctness and performance gaps of hand-written gemv_soa point
at declarative lowering as the right substrate. But this is Heath's
strategic call.
