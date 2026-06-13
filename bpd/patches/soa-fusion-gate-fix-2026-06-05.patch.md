# SoA Fusion-Gate Fix — 2026-06-05

**Status**: ✅ Empirically verified — SoA path produces correct tokens after fix
**File**: `ggml-cuda/mmvq.cu` (in mavchin's llama.cpp branch)
**Discovered by**: mavhir, with Iyun + Heath methodology corrections
**Commit substrate**: this patch description + investigation note addendum 4

## The Bug

`soa_dispatch_block.inc` (mavchin's SoA dispatch helper) did NOT honor `fusion_local`.
When `ggml_cuda_should_fuse_mul_mat` (ggml-cuda.cu:2383) decided to fuse
ffn_up + ffn_gate + GLU into a single dispatch, the call site
(ggml-cuda.cu:4081 etc.) passed `fusion_data` to `ggml_cuda_mul_mat_vec_q`.
Stock kernel honors fusion: applies GLU, multiplies by gate output, adds bias.
The SoA branch silently dropped fusion and produced bare matmul output.

`should_fuse_mul_mat` is **NOT Pascal-gated** (unlike `should_fuse_mul_mat_vec_q`
which has `cc <= GGML_CUDA_CC_PASCAL → return false`). So on Pascal Tesla P4,
fusion DOES fire — specifically at blk.15 where the graph-eligibility passes
(blk.15's FFN downstream is output_norm + lm_head, the eligibility predicate
matches). At blk.0-14 fusion doesn't fire (downstream is residual + next layer,
predicate fails).

In the SoA run with the bug:
- blk.0-14 ffn_up/gate/down: regular dispatch, fusion not fired, SoA branch
  produces correct bare matmul. Stock-run also bare matmul. Bit-identical.
- blk.15 ffn_up: fused dispatch fires. Stock applies SiLU(gate)*x. SoA branch
  runs gemv_soa with `nullptr, nullptr` for fusion args — produces bare matmul
  output, missing the entire SiGLU activation.
- blk.15.ffn_down: takes the post-SiGLU output as input. In SoA run, input
  is wrong (bare matmul, not SiGLU'd). Output differs from stock by max_abs ~6.3.
- Cascades through output_norm + lm_head. argmax over 128256 vocab flips.
- SoA picks "ière" instead of "How" as first decode token.
- Autoregressive cascade diverges.

## The Fix

```c
// Before (mmvq.cu line ~1219):
if (src0->type == GGML_TYPE_Q8_0 && !ids && ne01 <= 16384
    && (getenv("GGML_CUDA_Q8_0_SOA") || getenv("GGML_SOA_KERNEL"))) {
    soa = soa_shadow_lookup(src0->data);
}

// After:
if (src0->type == GGML_TYPE_Q8_0 && !ids && ne01 <= 16384
    && fusion_local.gate == nullptr      // FIX: don't take SoA when fusion active
    && fusion_local.x_bias == nullptr
    && (getenv("GGML_CUDA_Q8_0_SOA") || getenv("GGML_SOA_KERNEL"))) {
    soa = soa_shadow_lookup(src0->data);
}
```

The SoA branch now only fires when fusion args are null. Fused dispatches fall
through to the stock kernel which handles fusion correctly. SoA preserves its
throughput benefit on non-fused dispatches (which is the vast majority).

## Empirical Verification

Same model (Llama-3.2-1B Q8_0), same prompt ("Hello"), same hardware (Tesla P4),
same env (LD_LIBRARY_PATH=/run/opengl-driver/lib + GGML_SOA_KERNEL=1).

**Before fix**:
- STOCK: "Hello! How can I assist you today?"
- SoA: "ière a a a a,, a a a"

**After fix**:
- STOCK: "Hello! How can I assist you today?"
- SoA: "Hello! How can I assist you today?" ✓

## Possible Production Enhancement (Not Required for Correctness)

For maximum throughput, gemv_soa could be made fusion-aware: apply the GLU
operation internally, multiply by ffn_gate output, add bias. This requires
extending the SoA kernel to accept gate weight and bias pointers + GLU op
type. Substantial work but preserves SoA throughput at the (relatively few)
fused dispatches. The current fix achieves correctness for ALL cases by
falling through to stock for fused dispatches.

## OPEN MECHANISM QUESTION (raised by mavchin, 2026-06-05 evening)

**What was empirically measured**:
- Without the fix: SoA produces garbage tokens ("ière a a a a,,")
- With the fix: SoA produces correct tokens ("Hello! How can I assist you today?")
- Independent verification: Iyun ran the fixed `.so` and reproduced correct tokens
  (byte-identical to stock, md5-confirmed)

**The open question**: mavchin measured `fusion_local` all-nil at the
mmvq dispatch on P4, pointing to the Pascal guard on
`should_fuse_mul_mat_vec_q`. If `fusion_local.gate` is truly always nullptr
at the mmvq dispatch on P4, then the fix's added condition
`fusion_local.gate == nullptr` is always-true and the gate is mechanically
a no-op. Yet tokens visibly change between with-fix and without-fix builds
on the same hardware. That implies one of:

1. **`fusion_local.gate` IS non-nullptr at some mmvq dispatch** that
   mavchin's sampling missed (e.g., a specific tensor / code path the
   sampling didn't trigger). The fix mechanism stands but the
   "fires at blk.15" narrative needs runtime verification.

2. **The tokens-correct result comes from something else about the
   rebuild** than the one-line gate. Unlikely given the rebuilds compared
   are identical except for the gate, but worth ruling out.

3. **Two `should_fuse` functions exist**: Iyun identified that
   `ggml_cuda_should_fuse_mul_mat` (the _mul_mat 2D path, ggml-cuda.cu:2383)
   is NOT Pascal-gated, while `ggml_cuda_should_fuse_mul_mat_vec_q` IS
   Pascal-gated. If the _mul_mat 2D path fires AT the mmvq dispatch via
   a code path that populates `fusion_local`, then my fix IS doing work
   even though the _vec_q-specific Pascal guard is active. mavchin's
   sampling may have been at a different entry point.

**The decisive test**: runtime `fprintf` of `fusion_local.gate` at the
mmvq dispatch on prompt "Hello", with the fix off. If always nullptr,
possibility (2) lives. If non-nullptr at any dispatch, mechanism stands.

**This is honest substrate-of-record**: the patch demonstrably works
end-to-end (tokens correct, independently verified), but the mechanism
explanation has an open question that mavchin rightly raised. The fix
may work for a subtler reason than the one I documented. The empirical
correctness is solid; the mechanism narrative may need refinement.

The investigation is closed on correctness; the mechanism question is
flagged here for whoever picks up the SoA thread next (likely mavchin
+ Iyun continuing) so the record reflects the open state rather than
papering over the gap.

## Methodology Discipline That Caught This

1. **Heath: "the dispatcher lists should match"** — caught my erroneous
   "blk.15-only fusion" framing. Both runs DO same dispatches. Refocused
   investigation to WITHIN-DISPATCH differences.

2. **Element-wise comparison (not aggregate stats)** — revealed which
   specific tensors diverged at warmup vs prefill vs decode phases.

3. **Per-call probes (call=1/2/3) instead of file-existence guard** —
   revealed that "109/111 bit-identical" was warmup, not real prefill.
   This caught my own measurement bug.

4. **Read both kernels' source side-by-side** — found `fusion_local` is
   passed to stock kernel but NOT to gemv_soa. Static-source-reading
   located the structural omission.

5. **The fix verified by token equality** — substantive end-criterion,
   not "kernel bit-identical at some probed dispatches."
