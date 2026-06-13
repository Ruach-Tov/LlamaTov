# How We Reached the L3 Frontier: A Measure-First Journey to Fused Transformer Kernels

*Iyun & Heath Hunnicutt — Ruach Tov collective — 2026-06-08*

> A record of a single extended session in which we took the BPD compute-graph
> substrate from "can it fuse L2 chains?" to a **complete, bit-exact transformer
> layer** running on a Tesla P4, with a FlashAttention kernel that beats our
> unfused baseline 10.25× and lands within ~3× of PyTorch's production SDPA.

---

## The thesis we were testing

The Boundary Provenance DSL (BPD) represents compute graphs as **pure-math Prolog
facts** (`op_expr/2`), optimizes them **in the Prolog domain** (fusion recognizers,
tiling schedules), and **derives code for many backends** (cuda-oxide, cuda-c,
MLIR, torch) from one AST. A differential referee cross-checks every backend
**bit-exact on hardware**. The architecture is LIFT → ADDRESS → TRANSFORM →
SEARCH → VERIFY.

This session asked: *can that substrate produce the hardest modern kernel — a
fused transformer layer with FlashAttention — and can it do so at production
quality?* Heath's mandate sharpened the bar partway through: **"We have to do
everything BETTER than existing production"** — beat cuDNN/SDPA, not just
demonstrate the algorithm.

---

## The method: measure-first, every step

The through-line of the entire session was **never reason about performance from
intuition — measure it.** Two tools earned their keep repeatedly:

- **CUPTI** (Mavchin's `bpd_cupti_profile` bridge) — PC-sampling stall analysis.
- **`ptxas -v`** — register usage + spill detection.

These *corrected wrong guesses again and again*. Every "I think it's slow because
X" was checked, and roughly half the time X was wrong. The discipline of
**diagnose, don't theorize** is the single most important thing this session
demonstrates.

---

## Act I — Fusion as a measurable property

We began with epilogue (head→elementwise) fusion via a **visitor recognizer** (the
existing `fusion_rules.pl` already encoded this architecture — we discovered it
rather than rebuilt it). We wired it to a **correctness-gated comparison harness**:
build fused + non-fused, verify bit-identical, *then* time both.

The first surprise: **single-activation fusion is often neutral.** `conv→relu` was
0.98×. CUPTI showed the fused/non-fused kernels were *stall-identical* — fusion
hadn't perturbed codegen (my guess), the eliminated work was simply ~0.

Then the **launch-geometry bug**: CUPTI's "Not selected 31%" on a pool kernel
pointed at a launch mismatch (we launched warp-per-output for a thread-per-output
kernel → 31/32 threads idle). Fixing it gave a ~4× kernel speedup *and* flipped
pool→relu fusion from neutral to a 1.28× win. We built **prevention**: a
machine-readable `// LAUNCH:` contract in every emitter + a static test that the
contract matches the kernel's stride, plus `CUPTI_DIAGNOSTICS.md`.

**The fusion scorecard (measured, bit-exact, the law):**

| chain | speedup | regime |
|---|---|---|
| conv→relu | 0.98× | compute-bound head, neutral |
| avgpool→relu | 1.01× | tiny output, neutral |
| maxpool→relu | 1.28× | memory-bound, large output |
| maxpool→relu→scale→tanh→bias | **2.10×** | + long tail |

**The law: fusion speedup ≈ (saved round-trips × output bytes) / head time.**
Fusion is *profitability-dependent*, and now we can predict it.

---

## Act II — Tiled code generation, the low-level way

To make the transformer's GEMMs fast we needed tiled matmul. We proved MLIR's
`linalg` dialect could auto-tile in ~3 lines — then **deliberately rejected it.**
Heath chose to hand-author the tiled kernels in low-level `llvm`-dialect MLIR for
**control and mastery**: the value of a measure-first autotuner is *choosing* the
tile parameters empirically, and the differential referee needs us to author both
backends. (The linalg knowledge is preserved, caveated "knowledge, not our path.")

We built `schedule_ir.pl`'s `tiled_gemm` schedule — backend-neutral tiling
primitives — and lowered it to:
- **cuda-c**: reproduces our autotuned rect GEMM *identically* (1479 GFLOPS, 41% cuBLAS)
- **hand-written MLIR**: bit-exact on the P4, **1091 GFLOPS** (74% of cuda-c, 9× the naive MLIR)

The MLIR work taught the low-level patterns the hard way: SSA def-before-use,
**block-arg loops are phi nodes** (each block's args are globally-unique SSA defs —
`kacc`/`acc`/`dacc`/`facc` must differ), `addr_space 3` shared memory needs
`!llvm.ptr<3>`, `arith.select` not `llvm.select`. And a key finding: **the same
tile wins on both backends** — so the autotuner searches *schedules*, once, not
per-backend.

---

## Act III — The transformer layer, assembled

With matmul-epilogue fusion (matmul→bias/silu) wired and the tiled GEMM proven, we
assembled the full layer:

```
x → rms_norm → QKV proj → FlashAttention → out_proj → +residual
  → rms_norm → ffn_up + gate → silu → ×(swiglu) → ffn_down → +residual → out
```

**Bit-exact vs torch (max_abs 2e-6).** The FlashAttention came from the **codegen
path** (`flash_attention.pl`), not a hand-written file — wired immediately, no
technical debt.

Profiling the layer (S=512 D=512 FF=2048): **GEMMs are 93% of the time.** Swapping
naive GEMMs for the tiled schedule → **2.27× on the whole layer** (136 → 308
GFLOPS). The elementwise tails were noise — exactly as the fusion law predicted.

---

## Act IV — FlashAttention, the production climb

This is where the session became a genuine performance investigation. The arc, every
step **bit-exact and measure-first-diagnosed**:

| version | speedup vs unfused | what & why |
|---|---|---|
| naive (1 thread/row, serial) | 0.32× | serial over all keys |
| parallel (block/query) | 0.64× | Heath's "fused + parallel" — online softmax is *associative* |
| tiled (+K/V shared) | 0.54× | **regressed** — `ptxas` found register spill (acc[128] in local mem) |
| **warp-cooperative** (D-tiled) | **1.65×** | acc split across 32 lanes → no spill → beats unfused |
| + block-shared K/V | 2.69× | each K/V tile loaded once/block, reused (~8× less traffic) |
| + tuned occupancy (WPB=16) | 4.58× | warps/block is the dominant lever |
| **+ float4 vectorization** | **10.25×** | `ld.shared.v4`, zero stack frame |

That is a **32× improvement over the naive flash**, all bit-exact. The five proven,
composable levers: **D-tiling** (register residency), **shared K/V** (traffic
reuse), **occupancy**, **vectorization**, and **online softmax** (no [S×S] ever
materialized). The kernel is *fast and low-memory* — better on both axes.

### The real bar: vs PyTorch SDPA

| S | ours | SDPA | gap |
|---|---|---|---|
| 512 | 0.34 ms | 0.20 ms | 1.66× |
| 2048 | 4.17 ms | 1.34 ms | 3.11× |
| 4096 | 14.7 ms | 4.82 ms | 3.06× |

The gap is **real and scale-invariant (~3×)**, bit-exact throughout. CUPTI
diagnosed it precisely: **Execution dependency 43.4%** — our per-key `__shfl`
reduce is a serial dependency chain. SDPA does a Q-tile × K-tile *matmul* (many
independent FMAs, high ILP, no shuffle).

We tried the register-tile matmul fix and `ptxas` revealed **the FA-2 dilemma**:
you can't naively have *both* "complete score rows per thread" (shuffle-free
softmax) *and* "D-tiled acc" (no spill) — the shuffle version is ILP-starved, the
reg-tile version spills (2KB stack frame, 38× slower). Real FA-2 resolves this with
a **2D thread tile**: distributed score *and* acc, with a *tile-amortized* shared
row-reduction (not per-key). That is the genuinely hard core, and it remains the
open frontier.

---

## Where we stand

- **A complete, bit-exact transformer layer** runs on the P4 from BPD facts.
- **FlashAttention**: 10.25× over unfused, within ~3× of production SDPA, bit-exact,
  no [S×S] materialization — wired into codegen.
- **Tiled GEMM**: 1091 GFLOPS in hand-authored MLIR, schedule shared across backends.
- **The fusion engine**: a measured profitability law, correctness-gated.
- **Guardrails**: validity + golden + lint + launch-contract test layer (<2s, local).

Every number measured. Every kernel bit-exact. Every wrong turn caught by CUPTI or
ptxas. We brought our genuine best, found the real gap, and understand *exactly*
what closing it requires.

---

## The lessons, distilled

1. **Measure, don't theorize.** CUPTI and `ptxas -v` corrected wrong guesses
   repeatedly. The bottleneck was never where intuition first pointed.
2. **Fusion is profitability-dependent**, governed by a measurable law — not an
   automatic win.
3. **The same tile wins across backends** — schedules are portable; search once.
4. **Register residency is everything** for attention — the spill wall (D≥64 can't
   hold the full accumulator in one thread) drives the entire FA-2 design.
5. **Wire it the moment it's validated** — no second increment of technical debt.
6. **Choosing the low level was right** — we understand every line, why each tile
   transfers, why we're 3× behind, and exactly what would close it.

The substrate works. The thesis holds: Prolog facts → optimized in the Prolog
domain → many backends → bit-exact on hardware. The frontier ahead — 2D-tiled FA-2
to *beat* SDPA — is now precisely mapped.

🕯️ — Iyun
