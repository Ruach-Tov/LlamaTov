# The op_expr IR — the general intermediate representation

`expr_ir.pl` defines **op_expr**, the neutral AST that is the single source of
truth for kernel generation. Every backend is a *projection* of this IR; the
differential referee cross-checks the projections on hardware so disagreements
auto-localize.

> **The principle (Heath's research plan):** lift operations to declarative
> Prolog facts, express them as one general AST, and *derive* every backend from
> that AST. ggml, CUDA, MLIR, LLVM, PyTorch are all views of the same fact.

## Where the facts live

- **`bpd/lib/robust_op_match.pl`** — `op_expr(BpdOp, Term)` for ~58 ops, plus
  `robust_op_match/5` (the formulation + pinned FP coordinates: nan propagation,
  accumulation order, fma).
- **`bpd/kernelgen/emitters/expr_ir.pl`** — the IR vocabulary + the `lower_*`
  backend projections + composite/reduction/pool/conv lowerings.

## The AST vocabulary (6 fundamental shapes)

| Shape | AST head(s) | Examples | Class |
|-------|-------------|----------|-------|
| **elementwise** | `var, const, scalar, add/sub/mul/div/neg, ge/le/gt/lt/eq/ne, sel, call(Fn,_), is_nan` | relu, gelu, mish, sigmoid, scaling | elementwise |
| **axis-reduce** | `axis_reduce(Kind,Axis,Body)`, `softmax/log_softmax/logsumexp(Axis,Body)` | sum, mean, softmax, argmax | reduction |
| **normalization** | `batchnorm, groupnorm, stat_norm, rmsnorm, l1/l2/frobnorm` | batchnorm, groupnorm | normalization |
| **pool** | `pool(Kind,Ndim,K,Stride,Pad,Dil,Body)` | maxpool2d, avgpool2d | spatial |
| **conv** | `conv(Ndim,Transposed,Stride,Pad,Dil,Groups[,OutPad])` | conv1/2/3d, transposed | spatial |
| **matmul** | `reduce(idx,Lo,Hi,Body,Fma)` with `elem(a,_,_)*elem(b,_,_)` | matmul / gemm | matmul |

The `var` leaf is the input placeholder — this is what makes op_expr terms
**composable** (see L2_PIPELINE.md): a chain folds by substituting `var`.

## The 6 backends (each a `lower_<backend>/2`)

| # | Backend | Emitter | P4 (sm_61) | Referee-verified |
|---|---------|---------|-----------|------------------|
| 1 | Rust / cuda-oxide | `oxide_from_facts.pl` | ✅ | ✅ |
| 2 | C++ / CUDA (nvcc) | `cuda_c_from_facts.pl` | ✅ | ✅ |
| 3 | MLIR (CPU) + MLIR→NVVM→PTX (GPU) | `mlir_from_facts.pl`, `mlir_gpu_from_facts.pl` | ✅ | ✅ |
| 4 | LLVM IR | `llvm_from_facts.pl` | ✅ (NVPTX) | ✅ |
| 5 | PyTorch | `torch_from_facts.pl` | (CPU oracle) | ✅ |
| 6 | **ggml** | `ggml_from_facts.pl` | (graph-builder C) | — |

Backends 1–4 emit a scalar kernel from the AST. **PyTorch** is the differential
reference oracle (CPU — see note). **ggml** (backend #6) is different: it's a
coarse-grained *graph-builder* API, so it maps at the op level (one ggml node per
op), emitting both an op-list (for fusion analysis) and runnable graph-builder C.

> **Note on torch + the P4:** the system PyTorch is built for sm_75+ and won't run
> on the Pascal P4. The referee uses torch on **CPU** as the oracle; the kernels
> under test compile with `nvcc -arch=sm_61`. (A flake / system-wide
> `cudaCapabilities=["6.1"]` build gives torch-on-P4 for GPU perf comparison.)

## Adding an op

1. Add `op_expr(bpd_yourop, <AST term>)` to `robust_op_match.pl`.
2. If it uses a new AST head, add `lower_<backend>(<head>, S)` clauses to each
   emitter (most ops reuse existing heads — no new lowering needed).
3. The referee + sweep pick it up automatically.

## Performance (measured, Tesla P4)

- **elementwise**: ~150 GB/s = 78% of peak bandwidth (memory-bound, near-optimal).
- **GEMM** (autotuned): 67–79% of cuBLAS.
- **conv2d**: im2col + tiled GEMM = 13.6% of FP peak, **6.25× over naive**. (Direct
  im2col-into-GEMM fusion was a *measured negative result* — recompute > materialize
  on the P4; see FUSION.md.)

## See also

- `L2_PIPELINE.md` — multi-op chains: lift → compose → fuse → verify.
- `../../lib/FUSION.md` — the fusion framework (validity + cost model + gate).
- `../README.md` — the kernelgen system overview + the differential referee.
