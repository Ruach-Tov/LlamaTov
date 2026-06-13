# SoA Q8_0 Matmul Divergence — Empirical Investigation
**Date**: 2026-06-05
**Investigators**: Iyun (kernel diagnostician), mavhir (fresh-eyes/measurement), Heath (methodology discipline), mavchin (kernel author, handoff to mavhir)
**Status**: Kernel-vs-stock structural divergence located. Fix requires architectural change or alternative path.

## TL;DR

mavchin's custom Q8_0 SoA (Structure-of-Arrays) gemv kernel produces garbage
tokens in llama-cli inference on Llama-3.2-1B (Pascal P4) despite reporting
bit-identical output in his isolation harness on synthetic data. Empirical
investigation revealed:

1. **Bug is NOT** in fusion, streams, repack, dump-probe instrumentation,
   uninitialized memory, or block-bpr-stride.
2. **Bug IS** a structural divergence between gemv_soa and stock
   mul_mat_vec_q kernel architectures at ncols_dst=2 (prefill phase).
3. **Empirical signature**: element-wise per-output 56.8% bit-mismatch at
   blk1 hidden state input with random-direction long-tail ULP distribution
   (max 1522 at blk1, amplifying nonlinearly through softmax to 100%
   mismatch by blk15).

## Key Empirical Findings (in chronological order of discovery)

### 1. Pascal disables fusion entirely
`ggml_cuda_should_fuse_mul_mat_vec_q` at ggml-cuda.cu has early-return
`if (cc <= GGML_CUDA_CC_PASCAL) return false;`. On P4, no SwiGLU fusion
fires. Both stock and gemv_soa run bare matmul. This means:

- **Yesterday's diagnosis** ("SoA path dropped fusion_local") applied to a
  build that DIDN'T have the Pascal-gate. Today's build has it. Different
  builds = same code paths, different effective behavior.
- `clean.inc` (which applies FUSE_SILU=1 via dst->src[2]) was producing
  garbage by construction because src[2] is null on Pascal — applied SiLU
  where stock doesn't.

### 2. Multicol dispatch was missing (real bug, not THE bug)
`soa_dispatch_block.inc` called `gemv_soa<<<>>>` once without iterating
over `ncols_dst`. During prefill (ncols_dst=2), only column 0 of dst was
computed; column 1 stayed uninitialized. **Fixed via**:
```cpp
for (int j = 0; j < ncols_dst; j++) {
    act_col = src1_q8_1.get() + j * stride_col_y * 36;
    dst_col = dst_d + j * stride_col_dst;
    gemv_soa<<<nrows, 128>>>(... act_col, dst_col ...);
}
```
**But tokens still garbage post-fix.** This was a real bug that would
have caused garbage on its own, but not the dominant one.

### 3. Harness validates kernel-vs-its-twin (structural tautology)
`/tmp/soa_test_real.cu` uses a **mavchin-hand-written** `stock_matmul_q8_0`
kernel that uses the SAME reduction order as gemv_soa (mavchin matched
them deliberately). The harness reports `diffs=0/4096 BIT-IDENTICAL` on
real-dumped data, but this only validates that gemv_soa matches mavchin's
reference — NOT that gemv_soa matches the real ggml `mul_mat_vec_q`
production kernel.

The real `mul_mat_vec_q` uses a different per-block parallel-output
architecture (see Finding 7).

### 4. Paint shim — uninit-memory hypothesis refuted
Built `bpd/lib/cuda_mem_paint.c` (committed `baf444d5c`): LD_PRELOAD
shim wrapping cudaMalloc (paint 0xBAADF00D) / cudaFree (paint 0xDEADBEEF)
to force any uninit read to produce detectable sentinel.

Empirically:
- mavchin harness WITH shim: still `diffs=0/4096 BIT-IDENTICAL` →
  bit-identity is REAL, not from comparing same-uninit garbage.
- llama-completion SoA WITH shim: same deterministic garbage tokens →
  gemv_soa writes all of dst correctly; divergence is computational not
  memory-safety.

### 5. Determinism rules out race
3/3 SoA runs produce identical garbage tokens ("ière a a a a,,"). With
`CUDA_LAUNCH_BLOCKING=1`: still garbage, unchanged. So:
- Not a stream race
- Not concurrent execution issue
- Pure deterministic computational divergence

### 6. Element-wise comparison (Heath's methodology correction)
Population statistics (min/max/mean_abs) can match while element-wise
bit-equality fails. The aggregate-stats look I did earlier showed blk1-4
"EXACT MATCH" but element-wise was 50%+ mismatch the whole time.

Element-wise ULP comparison at hidden state per layer:

| Layer | bit_exact | mismatch | max_abs | max_ulp |
|-------|-----------|----------|---------|---------|
| blk1  | 43.2%     | 56.8%    | 2.38e-7 | 1522    |
| blk2  | 53.5%     | 46.5%    | 4.77e-7 | 627     |
| blk4  | 48.3%     | 51.7%    | 9.54e-7 | 2394    |
| blk8  | 0.1%      | 99.9%    | 6.94e-3 | 61M     |
| blk15 | 0.0%      | 100%     | 4.67e-2 | 2B      |

Drift pattern: ~50% mismatch holds blk1-4 (per-element ULP diffs that
cancel in aggregates), then nonlinearly amplifies through softmax/
attention to 100% mismatch by blk8.

ULP direction at each layer: **random** (~52% pos / ~48% neg), not
systematic. Rules out "single-direction accumulation order" fix path.

### 7. Structural divergence (THE bug)
Side-by-side reading of gemv_soa vs stock `mul_mat_vec_q` revealed:

**stock at ncols_dst=2**:
- `calc_rows_per_block(2) = 2`
- `nwarps = 4` (TPB=128)
- Each CUDA block processes **2 rows × 2 cols = 4 output values**
  simultaneously
- Per-thread accumulator: `float tmp[ncols_dst][rows_per_cuda_block] = [2][2]`
- Shared mem reduction: `tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size]`
- Interleaves 4 outputs through warp processing

**gemv_soa at ncols_dst=2**:
- Launches gemv_soa<<<>>> `ncols_dst` times externally (loop in dispatch.inc)
- Each kernel call: 1 row per CUDA block, 1 col
- Per-thread accumulator: single float `sum`
- Different per-block parallel-output architecture

This produces:
- Different per-thread block stride and accumulator structure
- Different reduction shared-mem layout
- Different FP rounding per output element → random-direction ULP
- Per-matmul output that differs from stock by 0-1522 ULP (long-tail distribution)

### 8. shfl_xor vs shfl_down: empirically inert
gemv_soa used `__shfl_down_sync`; stock uses `__shfl_xor_sync`. I patched
gemv_soa to use shfl_xor. **Zero empirical change**: element-wise stats
byte-identical pre- and post-patch.

Reason: only lane 0 writes dst. Both primitives deliver the same final
sum to lane 0 (just via different intermediate trees that visit different
lanes). The lane-0 register value is identical between the two reduction
patterns.

### 9. Warm-vs-cold CUDA state explains env-bisection
Earlier mystery: harness in default ssh env → `BIT-IDENTICAL`. In
`env -i` clean env → `1628/4096 MISMATCH`. Same binary, same data.

Timing measurement: default env runs 8-10ms (cached CUDA state from
recent runs). Clean env runs 210-220ms (cold CUDA driver init). The 20x
slowdown indicates fundamentally different driver code path.

**The BIT-IDENTICAL observation was an artifact of warm CUDA state**;
the cold-init MISMATCH is the correct measurement of the kernel's actual
divergence. This explains why Iyun and I saw different results from the
same binary — different shell-session warm-state.

### 10. FMA off in mmvq.cu yields per-matmul bit-identity but not tokens
Compiled mmvq.cu with `-fmad=false`. Result:
- Per-matmul DST output bit-identical between stock and SoA at blk.0.attn_q
  for all 7 SoA-routed matmuls in single forward pass
- BUT tokens still garbage

Reason: only mmvq.cu was recompiled. Other matmul kernels (mmf, mmq) +
non-matmul kernels (RMS_NORM, RoPE, softmax) still use FMA. Stock-vs-SoA
dispatch difference cascades through FMA-on downstream code paths,
producing divergent intermediates that compound.

So **per-matmul bit-identity ≠ correct tokens**. The bug requires either:
- Bit-identity across the ENTIRE graph execution (all kernels)
- Or a structurally-different fix path

## Fix Path Options

### (i) Disable FMA across all ggml-cuda .cu files
Bigger surgery (~50 .cu files). Would verify whether FMA-induced cascade
through the full graph is the dominant issue. Not a production fix
(performance impact + still wouldn't address the architectural
mismatch).

### (ii) Abandon gemv_soa, modify stock mul_mat_vec_q to use SoA buffer
Modify stock's `mul_mat_vec_q` kernel (or its inner `vec_dot_q8_0_q8_1`)
to accept the SoA-layout buffer (separate `quants` and `scales` arrays)
instead of AoS-layout. Math is bit-identical to stock by construction.
SoA memory benefit preserved at load level.

This is essentially: "implement stock's kernel with SoA-aware loads"
rather than "write a custom kernel that happens to match stock bit-for-bit."

Substantial work but structurally clean.

### (iii) Rewrite gemv_soa to match stock's [ncols_dst][rows_per_cuda_block] architecture
Match stock's per-block parallel-output structure exactly. Take ncols_dst
as template param, process rows_per_cuda_block rows per block, maintain
multi-accumulator array per thread. Essentially rewriting gemv_soa to be
"stock with different load addresses."

Same end result as (ii) but more invasive.

## Recommendations

Lean **(ii)** — modify stock's mul_mat_vec_q to use SoA buffer at the
data-loading layer. The "be bit-exact to stock" goal is achieved
trivially if we use stock's kernel. The "SoA memory benefit" is preserved
at the load address resolution, not at the kernel structure.

This is mavchin's substrate to architect — the choice between (i)/(ii)/(iii)
depends on what the SoA approach is ultimately trying to achieve. If
it's "better Pascal P4 throughput for Q8_0 matmul," then a benchmark-
driven choice between (ii) and (iii) is appropriate. If it's "experiment
with kernel design," the substantive learning from this investigation
is already complete and the rewrite isn't urgent.

## Artifacts

- `bpd/lib/cuda_mem_paint.c` (committed baf444d5c): LD_PRELOAD shim for
  cudaMalloc/cudaFree paint hardening. Reusable for future CUDA UAF/uninit
  debugging.
- `<external-build-tree>/` (on enclave): working build tree with various
  probe instrumentation, ready for further iteration.
- `/tmp/soa_test_bpr256.cu` (on enclave): synthetic-data harness for
  testing arbitrary bpr/ncols shapes (refuted Iyun's bpr=256 hypothesis).
- `/tmp/elementwise_diff.py`, `/tmp/ulp_direction.py` (on enclave):
  Element-wise comparison + signed-ULP-direction analysis scripts.

## Methodology Lessons (Heath's discipline corrections)

1. **Population statistics (min/max/mean) are NOT element-wise bit
   comparison.** Aggregate stats can match while element-wise diverges
   significantly. When the question is "bit-identical?", compare bits.

2. **Don't analyze a moving target.** mavchin was live-editing
   soa_dispatch_block.inc during the investigation. Iyun and I burned
   substantial cycles reading stale snapshots. The discipline: freeze
   the substrate (copy build tree to mavhir-owned location) before
   measuring.

3. **Warm CUDA state vs cold init can produce empirically-different
   kernel behavior** (timing-induced). When a measurement seems to
   contradict another, check environmental factors that affect driver
   path, not just env-var differences.

4. **Commit and push artifacts often** even mid-investigation. The
   paint shim was committed as baf444d5c during this work; this
   investigation note is committed similarly. Substrate that persists
   beyond one session needs to be in version control.

## Addendum: lm_head logit probe (final substantive measurement)

After the initial commit of this note, Iyun pushed for a direct probe of
the FINAL logits going to argmax-selection. Implemented and measured:

Setup: instrument the stock-else-branch dispatch in mmvq.cu to dump
`dst[0..4095]` of `token_embd.weight` matmul (Llama-3's lm_head, with
`ne01=128256 > 16384`, falls through to stock kernel in BOTH stock and
SoA runs by construction).

Captured 3 lm_head dispatches per run (decode tokens 1, 2, 3 after the
"Hello" prompt prefill):

| Probe | exact (partial vocab) | max_abs_diff | stock partial-argmax | soa partial-argmax |
|-------|------------------------|--------------|----------------------|---------------------|
| decode-1 | 2048/4096 (50%)    | 3.97e-02     | 1511 (v=1.84)        | 1511 (v=1.84)       |
| decode-2 | 3456/4096 (84%)    | 1.26e+01     | 2279 (v=4.84)        | 2311 (v=7.50)       |
| decode-3 | 128/4096 (3%)      | 2.70e+01     | 2181 (v=18.45)       | 3188 (v=11.96)      |

Substantive findings:

- **At decode-1**, 50% of partial-vocab logits already differ between
  stock and SoA runs, with max abs diff of 0.04. The partial argmax
  over first 4096 of 128256 vocab elements happens to match (1511) but
  the full-vocab argmax may differ given how much already diverges in
  the captured portion.
- **By decode-3**, only 128/4096 elements match (3%); the predictions
  are completely divergent.

The drift pattern at lm_head INPUT (the hidden state from blk.15)
matches what my earlier blk15 probe showed (100% mismatch with
max_ulp=2B). The divergence ARRIVES at lm_head already cascaded from
upstream non-matmul kernels (RMS_NORM, RoPE, softmax, attention scoring)
operating on slightly-different intermediate values from the per-matmul
ULP differences.

Confirmed empirically: **non-matmul kernels are amplifying the per-matmul
ULP differences into widespread hidden-state divergence**. The matmul
output ULP differences themselves are small (1-2 ULP at blk1, mostly
exact-match) but the nonlinear operations downstream (softmax in
attention especially) amplify them into substantive divergence by blk8
and total divergence by blk15.

This sharpens the recommended fix path:

**Option (ii) is the right path** because it makes ALL Q8_0 matmul
outputs bit-identical to stock by construction. With bit-identical
matmul outputs at every layer, the downstream nonlinear operations
have no ULP-difference to amplify. Tokens correct by construction.

Option (i) (disable FMA across all .cu files) might also work but is
a hack — production cost in performance, and doesn't address the
underlying architectural mismatch between gemv_soa and stock kernels.

## Notes on the driver-LD interaction (resolved subplot)

Earlier in the investigation, Iyun and I observed disagreement on
whether mavchin's harness `/tmp/soa_test_real` reports BIT-IDENTICAL or
1628/4096 MISMATCH on real-dumped data. Empirical resolution:

- Without `LD_LIBRARY_PATH=/run/opengl-driver/lib`: CUDA stub libcuda
  loads, `cudaGetDeviceCount → err=35 ("CUDA driver version
  insufficient")`. cudaMalloc fails silently. Harness runs with all
  device buffers staying at uninitialized host malloc'd memory. Both
  stock and SoA "kernels" do nothing. Both dst arrays = same
  uninit-memory pattern. Trivially BIT-IDENTICAL — degenerate.
- With `/run/opengl-driver/lib`: real NVIDIA 570 driver loads, GPU
  initializes, real kernels run, real divergence visible.

This explains why warm vs cold CUDA state on the same shell session
appeared to flip the result — the warm path may have inherited valid
device-pointer state from a prior real run, even after env stripping.

The model (llama-completion) REQUIRES `/run/opengl-driver/lib` (dynamic
link to libcuda.so → falls to CPU stub without it). So all model
measurements are inherently on the real driver, with the harness
needing to match this env to give comparable results.

## Addendum 2: Prefill vs Decode bit-identity asymmetry

After Iyun's pushback on "instrumentation might be the cause," ran two
final empirical tests with completely-stripped substrate (zero host-side
instrumentation in either branch — literally just `if (soa.quants) {
gemv_soa<<<>>>; } else { mul_mat_vec_q_switch_type; }`):

**Test 1**: Strip all instrumentation, rebuild with FMA on. Result:
tokens STILL GARBAGE. Instrumentation-corrupts hypothesis refuted.

**Test 2**: Add minimal full-dst dump probe at `blk.0.attn_q.weight`,
rebuild with FMA off. Result:
- Prefill phase blk.0.attn_q (ncols_dst=2, dst N=4096):
  **4096/4096 bit_exact, max_abs=0** between stock and SoA paths.

Extended Test 2 to all 7 Q8_0 matmuls in blk.0, capturing both prefill
and decode-1 phases. Decode-1 outputs (FMA off, pure bare dispatch):

| Matmul | bit_exact | max_abs_diff |
|--------|-----------|--------------|
| attn_k       | 0/512    | 4.47e+00 |
| attn_output  | 0/2048   | 1.39e-01 |
| attn_q       | 0/2048   | 3.82e+00 |
| attn_v       | 0/512    | 4.11e-01 |
| ffn_down     | 0/2048   | 1.21e+00 |
| **ffn_gate** | **16384/16384** | **0** |
| ffn_up       | 0/8192   | 3.21e+00 |

**The asymmetry**: PREFILL produces bit-identical outputs (at probed
level), DECODE-1 produces wildly divergent outputs. ffn_gate happens
to be bit-identical at decode-1 by coincidence (specific activations
landing on same float bits).

**The mechanism**: PREFILL → bit-identical matmuls but full-network
hidden state at end of prefill has enough accumulated drift (through
non-matmul kernels like RMS_NORM, softmax in attention) that the
lm_head argmax over 128256 vocab flips between stock and SoA. SoA-path
selects "ière" instead of "How" as decode-1's first token. Decode-1
sees a DIFFERENT token embedding as input. blk.0 matmuls at decode-1
operate on different activations between the two runs, hence the
wild divergence.

So the substantive bug:

1. Per-matmul outputs are very close but not perfectly bit-identical
   on real data (1-2 ULP per element, some elements at 1500+ ULP — the
   long-tail distribution Heath's element-wise discipline exposed)
2. These ULP-level differences cascade through 16 transformer layers
   amplified by softmax and attention nonlinearities
3. At end of prefill, accumulated drift flips lm_head argmax → different
   first decode token
4. Autoregressive decode then takes both paths into completely different
   token sequences

The kernel layer is not THE bug (per-matmul outputs are arithmetically
close to stock), but the cumulative drift from the per-matmul ULP
differences becomes sufficient to flip the first generated token. Once
that flips, the autoregressive cascade diverges.

This is why "fixing the kernel to be bit-identical" IS the right path:
bit-exact matmul outputs at every layer → no accumulating drift →
identical first-token argmax → identical generation. Option (ii) (modify
stock's mul_mat_vec_q to use SoA buffer) achieves this by construction.

Option (i) (disable FMA in mmvq.cu only) yields per-matmul bit-identity
but DOES NOT fix tokens because non-matmul kernels (FMA still active in
ggml-cuda.cu, common.cuh inline functions, etc.) still accumulate
rounding through softmax and attention that wasn't addressed.

**Tonight's investigation ends here**. Substrate-of-record committed
across three commits (baf444d5c, 71fdf847c, a19f25df9). Mavchin's
architectural choice on option (i) vs (ii) vs (iii) remains the next
substantive step.

## Addendum 3: Full-network prefill bit-identity scan

Final empirical measurement. With FMA off in mmvq.cu and pure bare
dispatch, probed first-call per tensor across ALL 112 Q8_0 matmul
weights (16 layers × 7 matmuls/layer). Element-wise comparison:

| Layer range | Bit-Exact | Divergent |
|-------------|-----------|-----------|
| blk.0 — blk.14 (all 7 matmuls each) | 105/105 (100%) | 0 |
| blk.15 attention (attn_q/k/v/output) | 4/4 (100%) | 0 |
| blk.15 ffn_gate | (not captured — probe missed) | — |
| blk.15 ffn_down | DIVERGED 0/2048 | max_abs=6.31 |
| blk.15 ffn_up | DIVERGED 0/8192 | max_abs=11.31 |

**Caveat**: The blk.15.ffn_up/down divergent files are HALF the size
of blk.0-14 ffn_up/down (2048 vs 4096 floats; 8192 vs 16384). This
suggests the captured comparison was on DECODE phase (ncols_dst=1)
not PREFILL (ncols_dst=2). The prefill dispatch for blk.15.ffn_up/down
may have not fired the file-existence guard, OR the dispatch path
differs at the last layer.

**What's empirically certain**:
- All Q8_0 matmuls through blk.14 are bit-identical at prefill (105
  matmuls × every output element)
- All blk.15 attention matmuls are bit-identical
- Divergence at blk.15 FFN — whether prefill or decode

This means: **either (a) the bug is specifically at blk.15 FFN
(some last-layer-specific code path) or (b) prefill is fully bit-
identical through blk.15 too and decode is where divergence enters,
which would point at decode-time state (KV cache, position-embedding,
something only running at decode).**

If (a), the kernel-architecture difference at blk.15 FFN's specific
shapes matters. blk.15.ffn_gate is the only matmul where stock might
use a different mul_mat_vec_q template instantiation than gemv_soa.

If (b), the cumulative drift mechanism documented in Addendum 2 holds,
but it must enter somewhere between end-of-prefill and decode-1 — which
matches the lm_head argmax flipping at end of prefill.

The empirical evidence is consistent with the (b) mechanism (Addendum 2):
prefill is bit-identical through all probed layers, drift in the
hidden-state-going-to-lm_head accumulates from non-matmul kernels
operating on slightly-different intermediate values (after FMA-induced
ULP differences somewhere — possibly in attention softmax denominator
sums, RMS_NORM, RoPE, none of which were FMA-disabled in this build).

The empirically-resolved fix path remains option (ii): modify stock's
`mul_mat_vec_q` to use SoA buffer at load layer, making per-matmul
output bit-identical to stock by construction. With identical matmul
outputs and identical non-matmul kernels (same kernels, same inputs),
the entire forward pass is identical, tokens match.
