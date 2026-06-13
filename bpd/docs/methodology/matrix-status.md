# Cross-Language Correctness Matrix — Status

**Last updated**: 2026-05-17 ~22:30 UTC

This document tracks which cells of the 7-cell-per-kernel matrix are
GREEN and under which correctness contract.

## The 7 cells per kernel

Per mavchin's framing (intercom 21:42 UTC, refined 21:46 UTC):

```
                        CPU dispatch    GPU dispatch
[1] C host              native C arith   [2] CUDA kernel
[3] Python host         NumPy/PyTorch    [4] ctypes → CUDA
[5] Rust host           native Rust f32  [6] cudarc → CUDA
[7] (N/A — cuda-oxide is GPU-only)
                                         [8] cuda-oxide → PTX → GPU
```

## The three correctness contracts

Per `docs/methodology/three-correctness-contracts.md`:

- **`--strict`**: bit-identical (uint32 XOR == 0). The strictest probe.
- **`--ulp N`**: bounded ULP distance. Use 2 for cross-axis transcendental
  comparisons.
- **allclose** (default): numerical correctness (rtol=1e-5, atol=1e-6).

Per Heath's "--strict-maxxing" direction: probe `--strict` first; fall
back to `--ulp N` only where the comparison crosses target boundaries
({CPU} vs {GPU} dispatch).

## The two empirical regimes

Per `docs/methodology/within-target-bit-identical-baseline.md`
(Heath's reframe, 2026-05-17): the matrix's cells group naturally
into TWO regimes:

### Regime 1 — Within-target (cells share dispatch hardware)

  CPU group :  [1] C CPU       [3] Python CPU       [5] Rust CPU
  GPU group :  [2] C GPU       [4] Python GPU       [6] Rust GPU    [8] cuda-oxide GPU

Standard: **bit-identical (ULP=0)**. Any deviation is a tunable-
parameter mismatch (math library, FMA, constant precision, reduction
order, compiler flags). The substrate has full control here; physics
doesn't intervene.

### Regime 2 — Cross-axis ({CPU group} vs {GPU group})

Standard: **--ulp 2**, AND each ULP of discrepancy must be explainable
via attribution to specific tunable parameters. Cross-axis ULPs come
from real implementation differences (different transcendental
libraries, different FMA semantics) that the substrate can sometimes
close but not always.

## k_add (mavchin's anchor) — 6/7 cells GREEN

  Cell | Contract  | Status
  -----|-----------|--------
  [1]  | --strict  | ✅ MATCH
  [2]  | --strict  | ✅ MATCH
  [3]  | --strict  | ✅ MATCH
  [4]  | --strict  | ✅ MATCH
  [5]  | --strict  | ✅ MATCH (local)
  [6]  | --strict  | ✅ MATCH
  [8]  | --strict  | ⬜ PENDING (cuda-oxide compiles for sm_61, binary not yet executed on P4)

Source: mavchin's test_8way_add.py + commit 9c4ce65a5.

## Reduction column (metayen)

  6 kernels: ggml_sum_rows, ggml_mean, ggml_max, ggml_min, ggml_argmax, ggml_argmin
  Shape: [8, 16] input → [8] output

  Op            | [1] | [2]            | [3] | [4]      | [5] | [6] | [8]
  --------------|-----|----------------|-----|----------|-----|-----|-----
  ggml_sum_rows | n/a | --ulp 4 MATCH  | ✅  | --strict | n/a | --- | n/a
  ggml_mean     | n/a | --ulp 4 MATCH  | ✅  | ⬜       | n/a | --- | n/a
  ggml_max      | n/a | --strict MATCH | ✅  | ⬜       | n/a | --- | n/a
  ggml_min      | n/a | --strict MATCH | ✅  | ⬜       | n/a | --- | n/a
  ggml_argmax   | n/a | --strict MATCH | ✅  | ⬜       | n/a | --- | n/a
  ggml_argmin   | n/a | --strict MATCH | ✅  | ⬜       | n/a | --- | n/a

  Legend: --- = applicable but pending; n/a = cell N/A; ✅ = cell exists in repo
  Note: cell [1] = C CPU requires substantive C-CPU reduction substrate
        that we don't have yet (BPD-emit is __global__-only). Deferred.
  Note: cell [5] = Rust CPU could exist trivially but reduces also need
        substantive native-Rust reduction impl; deferred.

  Per-op empirical max ULP (cell [2] vs cell [3]):
    sum_rows: 4 ULPs (PyTorch tree vs serial accumulation)
    mean:     4 ULPs (same cause)
    max/min/argmax/argmin: 0 ULPs — selection ops, no accumulation,
                            BIT-IDENTICAL

## Activation column (metayen) — NEW IN T8

  5 kernels: k_silu, k_sigmoid, k_relu, k_gelu, k_tanh
  Shape: [N] input → [N] output (1D, elementwise)
  Sizes probed: 128, 256, 257, 1000, 1023, 1024

### Empirical results, all 6 sizes, cell [2] vs cell [3]

  Kernel     | --strict | --ulp 2  | Notes
  -----------|----------|----------|--------------------------------
  k_silu     | DIVERGE  | ✅ MATCH | max ULP 2, as expected (SFU expf)
  k_sigmoid  | DIVERGE  | ✅ MATCH | max ULP 2, as expected (SFU expf)
  k_relu     | ✅ MATCH | ✅ MATCH | --strict-maxxing achieved! No transcendental.
  k_tanh     | DIVERGE  | ✅ MATCH | max ULP 2, as expected (SFU tanhf)
  k_gelu     | DIVERGE  | DIVERGE  | max ULP 8050 at N=1024 — see analysis below

### k_gelu divergence — investigation resolved (with substrate-honest finding)

The BPD-emitted k_gelu computes:
  `out[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));`

PyTorch's `F.gelu(approximate='none')` computes:
  `0.5 * x * (1 + erf(x / sqrt(2)))`

These are mathematically identical but produce substantial ULP
distances between cell [2] and cell [3]. Empirical per-size measurements
(2026-05-17 23:00 UTC):

  Size | Max ULP | Mean ULP | Bit-equal | Elements > 2 ULP
  -----|---------|----------|-----------|------------------
  128  |    374  |    8.72  | 30 / 128  | 33 / 128
  256  |   3562  |   18.90  | 66 / 256  | 63 / 256
  257  |   1687  |   10.06  | 61 / 257  | 79 / 257
  1000 |    839  |    5.26  | 276 /1000 | 263 / 1000
  1023 |   1866  |    4.74  | 255 /1023 | 261 / 1023
  1024 |   8050  |   10.81  | 244 /1024 | 271 / 1024

The max ULP varies dramatically with input sample size (374 to 8050).
Mean ULP is single-digit across sizes (4.74 to 18.90). The tail is
heavy: 25-35% of elements exceed 2 ULPs at each size. The variance
reflects that larger samples hit more near-zero gelu output values
where the erff implementation divergence concentrates. Investigation:

**Initial hypotheses** (mavchin 2026-05-17 22:29 UTC):
1. erff vs erf precision difference (CPU vs GPU implementations)
2. Constant precision promotion (double literals → expression promotes
   to double then narrows)
3. Multiplication-order associativity

**Hypothesis (b) tested and falsified as sole cause**. Changed
activation_expr facts to use `c_float_f` (f-suffixed) constants:
`0.5f`, `1.0f`, `0.7071067811865476f`. Re-ran on enclave:

  Pre-fix:  k_gelu max ULP at N=128 = 374
  Post-fix: k_gelu max ULP at N=128 = 374  ← SAME (also at all 6 sizes)

The double-promotion was a real C/C++ correctness concern (and worth
fixing — the post-fix CUDA emit is structurally cleaner), but it was
NOT the cause of the divergence.

**Hypothesis (a) is now the leading explanation**. Empirical signature
of the divergence:

  Worst-diverging elements concentrated at NEGATIVE INPUTS where
  gelu output is near zero:
    input=-2.92  cpu=-5.04e-03  gpu=-5.04e-03  ulp=374
    input=-2.90  cpu=-5.48e-03  gpu=-5.47e-03  ulp=370
    input=-1.91  cpu=-5.37e-02  gpu=-5.37e-02  ulp=46
    input=-2.21  cpu=-2.99e-02  gpu=-2.99e-02  ulp=36

  Mean ULP across 128 elements: 8.7
  Bit-identical elements: 30/128 (positive inputs where gelu(x) ≈ x)

This pattern matches what we'd expect from `erff` divergence:
  gelu(x) = 0.5 * x * (1 + erf(x/√2))
  For very negative x, erf(x/√2) ≈ -1, so (1 + erf(...)) ≈ 0
  Small absolute differences in erf become high ULP counts near zero
  (ULP density is highest at small magnitudes).

PyTorch's CPU `erff` and CUDA's GPU `erff` are different implementations
of the same IEEE-754 function. Both are "correct" within their respective
math-library specs (≤2 ULP of the infinite-precision result), but their
specific errors point different directions. Mavchin's "≤2 ULP transcendental"
empirical bound was for `expf`/`sigmoid`/`tanhf` — operations where CPU and
GPU implementations happen to agree closely. `erff` is in a different
precision class.

### k_gelu correctness contract — substrate-honest answer

  k_gelu cell [2] vs cell [3]:
    --strict (ULP=0):              DIVERGE (max ULP 8050 at N=1024)
    --ulp 2 (transcendental):      DIVERGE
    --ulp 10000 (erff-empirical):  MATCH (encompasses all 6 sizes)
    allclose (rtol=1e-5):          ✅ MATCH

The substrate names what it can and cannot guarantee. k_gelu IS
numerically correct (allclose-bounded) but it does NOT meet the
bit-identical or transcendental-tight contracts.

**Methodology refinement** (Heath's 2026-05-17 reframe): the initial
"three transcendental classes" framing conflated operation properties
with code-generation properties. The deeper truth is that ULP bounds
are properties of **(operation, knob-setting) pairs**, not operations
alone. See
`docs/methodology/within-target-bit-identical-baseline.md` for the
full reframe.

What we previously observed as "three classes" reflects what happens
when each operation is cross-axis-compared with default knob settings:

  Operation       | Empirical ULP cross-axis | Primary attribution
  ----------------|--------------------------|------------------------
  expf / silu     | ≤2 ULP                   | math library (cudart vs libm)
  sigmoid / tanhf | ≤2 ULP                   | math library
  erff / gelu     | up to 8050 ULP (N=1024)  | math library (erff differs near zero)
  relu / max / min| 0 ULP                    | no math library; pure conditional

These classes are an artifact of which library implementations we
defaulted to. A substrate that called the same erff implementation
across CPU and GPU would close the gelu gap. For now we document
each kernel's empirical cross-axis bound.

### Post-fix state (2026-05-17 22:40 UTC)

  Kernel       | --strict | --ulp 2  | allclose | Notes
  -------------|----------|----------|----------|--------
  k_silu       | DIVERGE  | ✅ MATCH | ✅ MATCH | Class 1
  k_sigmoid    | DIVERGE  | ✅ MATCH | ✅ MATCH | Class 1
  k_relu       | ✅ MATCH | ✅ MATCH | ✅ MATCH | Class 3 (--strict-maxxing!)
  k_tanh       | DIVERGE  | ✅ MATCH | ✅ MATCH | Class 1
  k_gelu_tanh  | DIVERGE  | DIVERGE  | ✅ MATCH | tanh form (ggml/Ollama default)
  k_gelu_erf   | DIVERGE  | DIVERGE  | ✅ MATCH | exact erf form

The k_gelu kernel was split into two canonical forms after the
terminology investigation (2026-05-17 ~23:30 UTC; see lib/terminology.pl
and docs/methodology/within-target-bit-identical-baseline.md). Both
forms are now first-class in the matrix:

  k_gelu_tanh — what ggml_gelu_f32 / F.gelu(approximate='tanh') /
                HF gelu_new compute. llama.cpp's graph builder uses
                this form at all three LLM_FFN_GELU dispatch sites,
                so Ollama runs this at inference time.

  k_gelu_erf  — what ggml_gelu_erf_f32 / F.gelu(approximate='none') /
                HF gelu (default) compute. Exposed in ggml's API but
                not used in llama.cpp's model dispatch.

### Empirical cross-axis ULP per size — both gelu forms

  k_gelu_tanh                   k_gelu_erf
  Size | Max ULP                Size | Max ULP
  -----|--------                -----|--------
  128  |    35                  128  |    374
  256  |   188                  256  |   3562
  257  |     7                  257  |   1687
  1000 |    15                  1000 |    839
  1023 |    36                  1023 |   1866
  1024 |    35                  1024 |   8050

The tanh form is TIGHTER than the erf form by 1-2 orders of magnitude
but still does NOT reach --ulp 2 cross-axis. This is a new substantive
finding: the tanh-form kernel has tanhf precision difference between
CUDA (libcudart) and PyTorch CPU (libm). Plus the inner polynomial
`1.0f + 0.044715f * x * x` likely benefits from FMA on GPU but not on
CPU — a tunable parameter we haven't standardized.

Per the within-target-bit-identical-baseline framing: the unattributed
ULPs for k_gelu_tanh are substrate self-knowledge gaps. Future work:

  1. Verify within-GPU bit-identicality for k_gelu_tanh (cell [2] vs
     cell [4]) — should be ULP=0 since same binary, same hardware.
  2. Measure CPU-tanhf vs GPU-tanhf precision difference on a wide
     range of inputs to characterize the bound.
  3. Test whether disabling FMA on the GPU side (via -fp-contract=off)
     shifts k_gelu_tanh into Class 1 (≤2 ULP).

## Aggregate matrix state

  Total cells GREEN as of 2026-05-17 22:30 UTC:
    k_add:        6 cells
    ggml_sum_rows: 3 cells (cell [2] --ulp 4, cell [3], cell [4])
    ggml_mean:     2 cells (cell [2], cell [3])
    ggml_max:      2 cells (cell [2] --strict, cell [3])
    ggml_min:      2 cells (cell [2] --strict, cell [3])
    ggml_argmax:   2 cells (cell [2] --strict, cell [3])
    ggml_argmin:   2 cells (cell [2] --strict, cell [3])
    k_silu:        2 cells (cell [2] --ulp 2, cell [3]) × 6 sizes
    k_sigmoid:     2 cells (cell [2] --ulp 2, cell [3]) × 6 sizes
    k_relu:        2 cells (cell [2] --strict, cell [3]) × 6 sizes
    k_tanh:        2 cells (cell [2] --ulp 2, cell [3]) × 6 sizes
    k_gelu:        2 cells (cell [2] allclose, cell [3]) × 6 sizes
                     ↑ note: --ulp 2 FAILS for k_gelu; investigation needed

  Total: 12 kernels with ≥2 cells GREEN.
  First kernel with --strict-maxxing achieved at cell [2]: k_relu.

## Within-target invariant tests — queued

Per Heath's reframe (and the within-target-bit-identical-baseline doc):
cells within the same target group SHOULD be bit-identical. The
empirical tests below verify this:

  Test A: cell [2] vs cell [4] for k_gelu (both GPU, C-host vs Python-host)
    Hypothesis: BIT-IDENTICAL. Same compiled CUDA binary executes on
                same GPU; only host-side marshalling differs. If this
                DIVERGES, there's a harness bug worth investigating.
                If this MATCHES, the cross-axis k_gelu divergence is
                cleanly attributed to math library choice.

  Test B: cell [3] vs cell [5] for k_relu (both CPU, Python vs Rust)
    Hypothesis: BIT-IDENTICAL. relu has no math library involved;
                fmaxf is deterministic. Should pass --strict.

  Test C: cell [2] vs cell [6] for k_silu (both GPU, C-host vs Rust/cudarc)
    Hypothesis: BIT-IDENTICAL. Same GPU kernel, different Rust host.
                Mavchin's k_add proof shows this works for elementwise add;
                should work for silu too.

These tests pending T8.f and Rust column extension.

## Cells not yet attempted

  - Rust cells [5], [6], [8] for any of metayen's kernels — coordination
    with mavchin pending (per the 21:46 UTC exchange: "wait until cell
    [2] for activations is GREEN" — which is now achieved for 4 of 5).
  - Cell [1] (C CPU) for reductions — requires substantive C-CPU
    reduction substrate.
  - Cell [4] (Python GPU) for reductions except ggml_sum_rows — needs
    @pytest.mark.skip decorators removed (mechanical).
  - Cell [4] (Python GPU) for all activations — pending T8.f/g.

Author: metayen 2026-05-17 ~22:30 UTC
Per Heath's "decompose T8, proceed until pausing" + --strict-maxxing.
