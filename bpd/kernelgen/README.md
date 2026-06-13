# bpd/kernelgen — multi-backend kernel generator

Generate GPU/CPU compute kernels from **declarative Prolog facts** across
multiple backends. The facts are the single source of truth; each backend is a
projection of the same fact, and the differential referee cross-checks them on
hardware so disagreements auto-localize bugs.

**Single source of truth:** `bpd/lib/robust_op_match.pl`
(`robust_op_match(Pattern, Op, Reference, Tier, Evidence)` — the formulation +
pinned FP coordinates: nan_propagation, accumulation_order, fma, etc.)

## Backends (6 backends, one IR)

| # | Backend | Emitter | Reaches GPU (P4, sm_61) | Referee-verified |
|---|---------|---------|----|----|
| 1 | Rust / cuda-oxide | `emitters/oxide_from_facts.pl` | ✅ | ✅ |
| 2 | C++ / CUDA (nvcc) | `emitters/cuda_c_from_facts.pl` | ✅ | ✅ |
| 3 | MLIR (CPU) | `emitters/mlir_from_facts.pl` | — (CPU) | ✅ |
| 3g| MLIR → NVVM → PTX (GPU) | `emitters/mlir_gpu_from_facts.pl` | ✅ | ✅ |
| 4 | LLVM IR (direct) | `emitters/llvm_from_facts.pl` | ✅ (NVPTX) | ✅ |
| 5 | PyTorch | `emitters/torch_from_facts.pl` | (CPU oracle) | ✅ |
| 6 | **ggml** | `emitters/ggml_from_facts.pl` | (graph-builder C) | — |

GEMM/matmul (reduction): `emitters/gemm_from_facts.pl` (+ tiled/autotuned
`gemm_tiled_from_space.pl`).

## The op_expr IR — the general representation

All backends derive from **op_expr**, the neutral AST (~58 ops, 6 fundamental
shapes: elementwise, axis-reduce, normalization, pool, conv, matmul). This is the
heart of the system — see **`emitters/EXPR_IR.md`**. Each backend is a
`lower_<backend>/2` projection; ggml (backend #6) is a coarse-grained
graph-builder so it maps at the op level (one node per op).

## Multi-op chains (L2)

KernelBench L2 problems are chains of L1 primitives. The L2 pipeline lifts a
chain, composes its fusable tail into one kernel, gates fusion on profitability,
and verifies against torch — see **`L2_PIPELINE.md`**. The chain-lifter is the
single source of truth for both the op_expr facts *and* the ggml problem encoding.

## Fusion (validity + profitability)

The fusion framework has two dimensions: *validity* (can we?) and *profitability*
(should we, given the shape?). A fusion can be correct yet slower — the cost model
(`bpd/lib/fusion_cost.pl`) decides, and the apply gate enforces it. See
**`../lib/FUSION.md`**.

## Generator parameters (the bit-identity contract is a *choice*)

- `emitters/nan_mode.pl` — **propagate** (IEEE, matches torch, NaN guard) vs
  **fast** (assume-no-NaN, faster). Honors the fact's `nan_propagation(ieee)`.
- `emitters/fma_mode.pl` — **strict** (mul+add, 2 roundings, matches torch-CPU)
  vs **contract** (fused fma, 1 rounding, matches nvcc -O3 / cuBLAS-style).
  Honors the fact's pinned `fma(strict|contract)`.

## The differential referee

`referee/multibackend_referee.py` — generates each op on every backend, runs on
the P4 over a fixed input, cross-checks every pair (+ vs torch-CPU oracle) with
ULP, separating real ULP from NaN-semantic (`+Nnan`) and signed-zero (`+Nsz`)
mismatches. Reused ULP machinery from `bpd/bench/stanford_referee.py`.
`referee/{relu,silu}_nvcc.cu` — nvcc same-device reference kernels.

## MLIR-GPU runtime

`runtime/` — the MLIR→GPU launch path:
- `mlir_gpu_pipeline.sh` — lower a gpu-dialect `.mlir` to PTX/cubin
- `cubin_launch.cu` — load a cubin + launch on the P4 (driver API)
- `mlir_gpu_launch.cu` / `mlir_gpu_launch2.cu` — PTX launch / libdevice-linked launch
- `mlir_harness.c` — CPU harness for MLIR-CPU kernels

MLIR-GPU pipeline (per op):
```
mlir-opt --convert-scf-to-cf
         --nvvm-attach-target="chip=sm_61 O=3 [l=libdevice.10.bc]"
         --convert-gpu-to-nvvm --reconcile-unrealized-casts
         --gpu-module-to-binary="format={isa|bin}"
```
Transcendentals (math.tanh/erf/exp → __nv_*) require `l=libdevice.10.bc` +
`format=bin` (links libdevice at LLVM level → ELF cubin). Elementwise (relu)
uses `format=isa` → PTX.

## cuda-oxide build

`build/` — reproducible build of the clean-room Rust→PTX compiler:
- `build-env.sh` — nix toolchain (nightly-2026-04-03, llvm-21 llc, CUDA 12.8)
- `ruachtov-c8b3103-compile-fixes.patch` — fixes atop `bpd/patches/ruachtov-sm61-cmath.patch`
- `cuda-oxide-build-README.md` — full reproduction recipe

The cuda-oxide checkout itself lives on the enclave at
`<data>/cuda-oxide` (clone of NVlabs/cuda-oxide@c8b3103 + patches);
its origin is NVlabs so its git history has no cloud backup — these build files
are the reproducible recipe.

## Key findings (recorded in ruach-memory)

- **Transcendental cross-device 1-ULP**: GPU libdevice (__nv_expf/tanhf/erff) ≠
  CPU torch (SLEEF) by ~1 ULP. Right GPU reference = nvcc-on-same-GPU.
- **Matmul = FMA, not reduction order**: naive GEMM is bit-identical across
  cuda-oxide / nvcc / torch *iff* accumulation order matches AND fma=strict.
- **PyTorch 2.7 cannot run on the P4** (Pascal dropped) — validates the whole
  clean-room thesis; nvcc-on-P4 is the perf reference.
- **Found + fixed a novel upstream cuda-oxide bug**: float `!=` lowered to
  `fcmp one` (ordered) instead of `une` → `x != x` folded false for NaN.
  Reported: NVlabs/cuda-oxide issue #123, PR #124.
- **conv2d im2col+GEMM = 6.25×** (2.2% → 13.6% of FP peak); the direct
  im2col-into-GEMM *fusion* was a measured **negative result** (recompute >
  materialize on the P4) — which produced the fusion cost model.
- **L1: 90/100 recognized, 83 verified** via per-problem param extraction.
- **L2: 100/100 chains lift + resolve**; pipeline validated end-to-end (problem
  12 fused rel=0.0). The lifter is the single source for op_expr + ggml encoding.

## Subsystem docs

- **`emitters/EXPR_IR.md`** — the op_expr IR + the 6 backends.
- **`L2_PIPELINE.md`** — multi-op chains: lift → compose → fuse → verify.
- **`../lib/FUSION.md`** — the fusion framework (validity + cost model + gate).

— Iyun, 2026-06-07 / -08 (op_expr IR, ggml backend #6, fusion cost model, L2 pipeline)
