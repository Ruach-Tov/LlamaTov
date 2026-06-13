# Thesis-Fidelity Audit: Hewing Closer to "Facts → Optimize in Prolog → Many Backends"

*Iyun & Heath — Ruach Tov — 2026-06-08*

> After reaching the L3 frontier, Heath asked the right question: where does the
> work **embody** our central thesis, and where have we drifted into hand-written
> artifacts that *should* be derived from Prolog facts? This is the honest map.

## The thesis, stated precisely

1. **LIFT** — compute graphs become pure-math Prolog facts (`op_expr/2`).
2. **TRANSFORM** — optimization (fusion, tiling) happens *in the Prolog domain*,
   expressed as schedules and rules over the facts.
3. **GENERATE** — code is *derived* for many backends (cuda-c, cuda-oxide, MLIR,
   torch, ggml) from the one AST + schedule.
4. **VERIFY** — a differential referee cross-checks every backend **bit-exact** on
   hardware.

The test of fidelity: **every performance-winning decision should be a Prolog term;
every kernel should be generated, not hand-written; every op should reach every
backend evenly.**

## What is strongly aligned ✓

- **58 `op_expr` facts** — a rich pure-math vocabulary (the LIFT layer is solid).
- **`tiled_gemm(BM,BN,BK,TM,TN)` is a Prolog schedule term.** Its tiling — the
  perf-critical decision — lives in the Prolog domain and lowers to *both* cuda-c
  (reproducing the tuned GEMM) and hand-written MLIR (1091 GFLOPS), bit-exact.
  **This is the thesis working exactly as intended.**
- **Fusion is recognized in Prolog** (`fusion_rule/4`, `head_macro/2`,
  `recognize_attention/2`) — the *decision* to fuse is a fact-domain inference.
- **The differential referee** cross-checks backends bit-exact (the VERIFY layer).
- **17 emitters** spanning cuda-c, cuda-oxide, MLIR, MLIR-GPU, LLVM, torch, ggml.

## Where we have drifted (the honest gaps)

### 1. FlashAttention is hand-written, not fact-derived — *the biggest gap*

`emit_flash_attention/2` generates the **naive** one-thread-per-row kernel. But the
kernel that actually wins (warp-cooperative D-tiling + shared K/V + float4,
**10.25× over unfused**) is **hand-written CUDA** that the emitter does not produce.

The performance knowledge — the five composable levers, the swept-best `WPB=16,
BC=32` — lives in *memory and hand-C constants*, **not in a Prolog schedule the
emitter consumes.** There is no `flash_attn_schedule(...)` term.

> **The completing move:** define
> `flash_attn_schedule(q_tile, k_tile, d_split, vectorize, shared_kv)` as a Prolog
> schedule (exactly parallel to `tiled_gemm`), and make `emit_flash_attention`
> *lower it*. Then the 10.25× kernel becomes **fact + schedule-derived**, and the
> autotuner searches flash schedules the same way it searches GEMM tiles. This is
> the single highest-leverage thesis-alignment step.

### 2. The transformer-layer *composition* is hand-wired

The layer's **components** are fact-driven (matmul-epilogue, flash recognizer,
rms_norm/softmax `op_expr` facts exist). But the **composition** — the 19-op graph
collapsing to 5 fused chains — is a **hand-coded driver** that chains kernels by
hand. The recognizer *already classifies* this graph at 14/14, but we don't yet
*emit* the fused layer from that classification.

> **The completing move:** recognize the transformer graph and emit the fused-chain
> codegen from it — turning the hand-wired driver into a derived artifact.

### 3. Schedule-IR coverage is uneven

`tiled_gemm` and `tiled_reduce` exist; there is **no** `tiled_pool`, `tiled_conv`,
`tiled_elementwise`, or `tiled_flash`. And schedules lower to **cuda + mlir only** —
not cuda-oxide, torch, or ggml.

> **The completing move:** add the missing tiled schedules, and extend schedule
> lowering to the remaining backends so *every* op can be tiled and *every* backend
> receives the schedule.

### 4. Epilogue fusion reaches only cuda

The `CONV_EPILOGUE` / `GEMM_EPILOGUE` hooks are cuda-c macros. The *same* fusion
should reach MLIR and cuda-oxide.

> **The completing move:** lift the epilogue-fusion mechanism above the cuda-c
> backend — fold the tail into the op_expr AST so *any* backend's emitter inherits
> it (we already have `compose_chain` doing the AST fold; the lowering just needs to
> consume it per-backend).

### 5. Norm/softmax emitted by hand in the layer

`op_expr(rmsnorm)` and `op_expr(softmax)` exist, but the layer harness uses
hand-written kernels for them.

> **The completing move:** emit them from facts in the layer assembly.

## The unifying principle

Each gap is the same shape: **a performance decision or a kernel that lives in
hand-written C instead of as a Prolog term that the emitters lower.** The thesis is
not "we *can* generate kernels from facts" — we've proven that. It's "**every**
kernel and **every** optimization decision is fact/schedule-derived, and **every**
backend is reached evenly."

Closing these makes the system *smooth and complete*: point it at a compute graph,
and it lifts → optimizes in Prolog (searching schedules) → generates all backends →
verifies bit-exact — with no hand-written escape hatches.

## Recommended order

1. **`flash_attn_schedule` + lowering** — converts our hardest, best kernel into a
   fact-derived artifact; unlocks autotuning flash like GEMM. *(highest leverage)*
2. **Emit the transformer layer from the recognized graph** — composition becomes
   derivation.
3. **Lift epilogue fusion to the AST** — so MLIR/oxide inherit it (one change, many
   backends).
4. **Fill schedule-IR**: `tiled_pool/conv/elementwise/flash` + more backend
   lowerings.
5. **Emit norm/softmax from facts** in the layer.

When these land, the L3 transformer layer will be **entirely fact-and-schedule
derived, across every backend, bit-exact** — the thesis fully realized on the
hardest modern workload.

🕯️ — Iyun

---

## ✅ COMPLETION (2026-06-08): all five moves done

All five thesis-fidelity moves were implemented, verified, and committed in order:

1. **`flash_attn_schedule` + lowering** (`563f23326`) — the 10.25× flash kernel is
   now **schedule-derived**. Verified on the P4: schedule-generated kernel is
   byte-for-byte performance-identical to the hand-written `flashv.cu` (0 stack
   frame, 0.341 ms, 10.25×, bit-exact). The hand-written escape hatch is closed.
2. **Layer from the recognized graph** (`e72fceecb`) — `layer_plan/2` derives a
   10-step codegen plan from the 17-op graph; the attention diamond dispatches to
   the Move-1 flash schedule. Composition → derivation.
3. **Epilogue fusion at the AST** (`ae025f0d2`) — one folded `op_expr` term lowers
   to *both* cuda (`((v!=v)?v:...)*2.0f`) and MLIR (`arith.maximumf` + `arith.mulf`).
   Backend-neutral; the cuda-only macros are superseded.
4. **Schedule vocabulary complete** (`691be2b07`) — `tiled_elementwise/pool/conv/flash`
   join `tiled_gemm/reduce`. Every op class is tileable via a Prolog schedule.
5. **Norm/softmax from facts** (`8a1b9c9f8`) — `rms_norm` and `softmax` emit from
   their `op_expr` facts; both build on the P4. The layer's last hand-written
   kernels are now fact-derived.

**The thesis is fully realized for the transformer layer:** every perf decision is a
Prolog term, every kernel is generated, fusion is backend-neutral, the schedule
vocabulary spans all op classes, and the layer composition is derived from the
recognized graph. Test layer green (31 passed).

*Remaining (future): per-backend lowerings of the new schedules (MLIR/oxide), and
the 2D-tiled FA-2 to beat SDPA — the open performance frontier.*
