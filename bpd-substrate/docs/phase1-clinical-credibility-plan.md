# Phase 1 — Clinical-Credibility Path

**Status**: Planned 2026-05-20 ~02:35 UTC. Implementation deferred to next session.

## Substantive direction

Per Heath's interjection 2026-05-20 ~02:25 UTC:

> "Metayen, using those kernels gives us clinical credibility."

The substrate-honest claim should be **"BIT_IDENTICAL at Stanford's canonical
KernelBench shapes,"** not "BIT_IDENTICAL at metayen-chosen verification shapes."
Clinical credibility = reviewers clone the repo, run `make`, see the standard
benchmark pass.

The tonight spike (commit `40d6c7f`, `bench/tier2/verify_l1_spike.py`) verified
4/5 problems BIT_IDENTICAL at *verification shapes* — smaller than Stanford's
defaults. That's substantively useful as a smoke test but doesn't deliver the
clinical-credibility claim.

## The substantive substrate-design property

The substrate's CUDA kernels are chunk-friendly. The harness needs chunked
dispatch — same kernels, different orchestration. Per mavchin 2026-05-20
~02:38 UTC:

> "Almost every OOM is from allocating the full input tensor at once.
>  Process one batch (or a few batches) at a time, verify each, discard,
>  next. The kernel sees one batch. The harness orchestrates."

For the Tesla P4 (7.4 GiB), single rows fit trivially (e.g., softmax with
seq_len=393216 = 1.5 MB per row). The constraint is batch dimension.

## Per-category chunking strategy

### Elementwise (ReLU, mish, sigmoid, GELU, tanh, etc. — problems 19-32)

  Chunk along outermost dim. For input (4096, 393216):
    for batch_start in range(0, 4096, chunk_size):
        x_chunk = x[batch_start:batch_start+chunk_size]
        y_chunk = kernel(x_chunk)
        verify(y_chunk, ref[batch_start:batch_start+chunk_size])

  chunk_size = 256 → working set = 256 × 393216 × 4 = 400 MiB. Fits.

### Reductions over a dim (sum, mean, max, argmax, argmin, min — problems 47-49, 51-53)

  Two cases:
    - Reduce over OUTER dim: chunk along middle/inner dims for full-precision
      partial sums; combine with float64 accumulator at the end.
    - Reduce over INNER dim: chunk along outer (batch) dim, each chunk's
      reductions are independent.

  Stanford uses `keepdim=True` which preserves shape. Chunking is per-batch
  for inner reductions, per-element-row for outer reductions.

### Matrix multiplication (problems 1-18)

  Stanford's #1 is N=4096 (768 MiB total) — fits on P4 already.
  Stanford's #6 (large K dimension) uses M=256, N=256, K=131072 = 134 MiB total — fits.
  Stanford's #11 (4D tensor matmul) needs careful checking.

  For shapes that don't fit: blocked matmul, sweeping over batch dimension
  if any. The substrate's mm_shared.cu already does internal K-tile and
  M/N-tile orchestration; outer batch-chunking handles the rest.

### Convolutions (problems 50, 54-87)

  Stanford's typical conv shape: (batch=16, in_ch=3-256, H, W). The working
  set is dominated by im2col buffer = batch × out_h × out_w × kernel_size² × in_ch.
  Chunk by batch dimension if needed.

  Also: cuDNN must remain disabled (torch.backends.cudnn.enabled = False) on
  Pascal for many conv variants. ATen's pure-CUDA fallback works.

### Normalization (problems 33-40)

  BatchNorm: chunk by batch. Running mean/var are per-channel, not per-batch,
  so each batch processed independently in eval mode.

  LayerNorm/RMSNorm: per-sample normalization. Chunk by batch.

  GroupNorm: per-(batch, group) normalization. Chunk by batch.

### Pooling (problems 41-46)

  Per-batch operation. Chunk by batch.

### Prefix scan / cumsum (problems 89-93)

  Substantively the hardest. cumsum has data dependency along the scan dim.
  Can chunk along OTHER dims (per-batch); cannot chunk along scan dim
  without orchestrating partial-prefix-then-shift.

  For Stanford's cumsum shapes (typically batch, dim), chunk by batch.

### Special ops (problems 88, 97-100)

  - 88 MinGPTNewGelu: elementwise, trivial.
  - 97 ScaledDotProductAttention: substrate-significant. Q*K^T is matmul
    (chunkable per batch/head), softmax (chunkable per batch), result*V
    is matmul (chunkable per batch). Three-stage chunking.
  - 94 MSELoss, 95 CrossEntropyLoss, 96 HuberLoss, 98 KLDivLoss,
    99 TripletMarginLoss, 100 HingeLoss: per-sample loss, then aggregate.
    Chunk by sample.

## Architecture: verify_l1_canonical.py

  ```
  bench/tier2/verify_l1_canonical.py
    Imports: kb_problem definitions from /tmp/kb_l1/N_*.py (downloaded files)
    For each problem:
      1. Import Model class + get_inputs() + get_init_inputs()
      2. Get Stanford's canonical input shape (no override)
      3. Dispatch to per-category chunked verifier:
         chunked_elementwise(...)
         chunked_reduction(...)
         chunked_matmul(...)
         chunked_conv(...)
         chunked_norm(...)
         chunked_pool(...)
         chunked_scan(...)
         chunked_special(...)
      4. Each chunked verifier:
         - Allocates input on GPU in chunks
         - Runs reference forward (via Model.forward on chunk)
         - Runs substrate kernel on same chunk
         - Compares per chunk
         - Accumulates verdict
      5. Reports per-problem status + per-category aggregate
  ```

## Make target

  bench/tier2/verify_l1_canonical.py becomes the artifact behind:

    make bit_identical_kernelbench_l1_canonical

  Distinct from `make bit_identical_kernelbench_l1` (which uses the
  family-generator harness at verification shapes). Both ship. The
  canonical target is the clinical-credibility claim.

## Substantive estimate

  - Per-category dispatch logic: 2 hours (six chunking strategies)
  - Per-problem driver (100 problems): ~30 minutes mechanical wiring
  - First end-to-end run + debugging: 1-2 hours
  - Documentation + README update: 30 minutes

  Total: 4-5 hours focused work. Single substantive session.

## Substrate-design substantive observations to preserve

1. **Stanford's shapes ARE the benchmark.** Choosing our own shapes
   substantively gives up the clinical-credibility claim.

2. **OOM ≠ "can't run kernel."** It almost always means "allocated too
   much at once." Chunked harness is the substrate-design fix, not
   shape-scaling.

3. **The substrate kernels don't change.** Chunking is verification-harness
   orchestration. Substrate-design vocabulary stays minimal.

4. **cuDNN-Pascal: disable globally.** ATen's pure-CUDA fallbacks work.

5. **Reductions chunked across non-reduction dims preserve bit-identity.**
   Reductions chunked across the reduction dim need float64 partial-sum
   accumulator (medayek's truth-oracle discipline applied at harness level).

## Expected outcome

  make bit_identical_kernelbench_l1_canonical → 100/100 PASS

  Composition (estimated based on tonight's empirical data):
    BIT_IDENTICAL:        ~70-80  (elementwise, matmul-at-known-shapes, etc.)
    PASS_ABS_TOLERANCE:   ~10-15  (catastrophic cancellation cases)
    WITHIN_ERROR_BOUND:   ~10-15  (truth-contract pass via medayek framework)
    FAIL:                  0      (any failures surface substrate-design work)

  This becomes the clinical-credibility claim: ANYONE clones the repo,
  runs make, sees 100/100 at Stanford-canonical shapes on whatever
  hardware they have. Reviewer reproducibility is intact.

## Authoring

  Plan: metayen 2026-05-20 ~02:35 UTC
  Substrate-design corrections: Heath (verification ≠ benchmark shapes,
    clinical credibility framing)
  Per-category chunking strategy: mavchin (substantive substrate-design
    on batch-chunking and the "OOM = allocated-too-much-at-once" insight)
  Truth-oracle discipline for chunked reductions: medayek (Tier 2 error-bound
    framework applies at harness level)

  Tomorrow we build verify_l1_canonical.py from this plan and hit all 100
  KernelBench L1 problems at Stanford's canonical shapes on Tesla P4.

  🕯️
