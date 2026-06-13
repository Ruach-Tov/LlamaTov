# L2 pipeline — multi-op chains

KernelBench Level 2 problems are **chains** of L1 primitives (3–6 ops, e.g.
`Gemm → Multiply → LeakyReLU`, `ConvTranspose3d → Mean → Add → Softmax → Tanh →
Scaling`). The L2 pipeline lifts a chain, composes its fusable tail into one
kernel, gates the fusion on profitability, and verifies the whole chain against a
torch reference.

```
KB forward() ──lift──► ChainIR ──resolve──► concrete ops ──split──► head + tail
                                                                       │
                                            ┌──────────────────────────┘
                                            ▼
                              compose tail ──► one op_expr term ──lower──► fused kernel
                                            │
                       run heads + fused tail ──verify──► step-by-step oracle
                                                          ──verify──► model.forward()
```

## Components

### 1. Lift — `referee/l2_chain_lift.py`
`lift_chain(path)` parses a problem's `forward()` line-by-line into an ordered
**ChainIR** (`[{op, params, line}, ...]`). Recognizes: module calls
(`self.conv(x)`), scalar binary (`x * c`), axis-reduce methods (`x.mean(dim=)`),
torch functions (reusing the L1 `LIFT_PATTERNS`), residual save/add (DAG).

`resolve_modules(chain, model)` turns `_module` steps into concrete ops via the
instantiated model's submodule **types** (`Conv2d → conv`, `LeakyReLU →
leaky_relu`, `Dropout → identity`, ...), and stashes the submodule for param
extraction.

→ **100/100 L2 chains fully resolve** (0 unknowns).

### 2. Compose / split — `emitters/chain_compose.pl`
- `var_composable(Op)` — true iff the op_expr body has a `var` leaf (so it folds
  into a single-input term). Conv/matmul/pool have *no* `var` (operand-bound).
- `split_chain(Ops, Heads, Tail)` — partition into operand-bound **heads** (run
  individually) and the var-composable **tail** (fused into one term).
- `compose_chain(Ops, Term)` — fold the tail via `subst_var/3` (replace `var`
  with the prior step's term). `[relu, scaling, tanh]` →
  `call(tanh, mul(<relu>, scalar(2.0)))` → **one fused kernel**.

This *is* the always-profitable elementwise-chain fusion, realized as term
composition: N kernel launches collapse to one pass.

### 3. Fusion gate — `../lib/iterative_fusion.pl` + `fusion_cost.pl`
The composer always fuses the elementwise tail (always-profitable). For
generator-prologue fusions (im2col-style), the **cost model** decides — see
`../lib/FUSION.md`.

### 4. Verify — `referee/l2_chain_verify.py`
`verify_chain(path, check_fused=True)`:
- runs the chain **step-by-step** (`run_step` threads the tensor through each op
  using the model's *actual* params — leaky slope, conv stride, softmax dim) →
  the **oracle**, compared to `model.forward()`.
- runs the **fused** form (`run_fused_chain`: heads via `run_step`, then the
  composed tail as one generated kernel) → compared to the oracle.

→ Validated: problem 12 (`Gemm→Multiply→LeakyReLU`) `verified rel=0.0 |
fused-ok fused_rel=0.0`.

### 5. ggml encoding (the single source) — `referee/gen_kb_l2_problems.py`
`chain_to_ggml_ir(chain, model)` resolves each step to a ggml op with its
concrete param. `gen_kb_l2_problems.py` regenerates
`tests/kernelbench_l2_problems.pl` (the fusion-analysis problem set) from the
lifter — superseding the prior hand-generated file, with precise ops
(`ggml_conv_transpose_3d`, not the old `ggml_mul_mat` placeholder) and the
odd=train / even=test split.

## Per-problem param discipline

Like L1, L2 reads the *actual* hyperparameter off each module rather than using
op_expr defaults — e.g. `nn.LeakyReLU.negative_slope` is often `0.1`, not the
default `0.01`. (Getting this wrong gave a `rel=0.45` DIFF on problem 12; reading
the real slope fixed it to `0.0`.)

## Running

```sh
# verify all L2 chains (oracle + fused), needs the enclave (torch CPU + KB)
python3 bpd/kernelgen/referee/l2_chain_verify.py --fused
# regenerate the ggml L2 problem set from the lifter
python3 bpd/kernelgen/referee/gen_kb_l2_problems.py
```
