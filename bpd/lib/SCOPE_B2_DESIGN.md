# Scope B2 Design: standalone CUDA templates for L1 non-epilogue ops

Author: metayen 2026-05-15
Status: design draft, awaiting mavchin's structural input

## Problem

KernelBench L1 contains ~17 ggml ops that DON'T fit mavchin's existing
matmul+epilogue fusion pattern in `fusion_to_cuda.pl::build_epilogue_stmts`.
These ops need standalone kernel templates because they have different
iteration spaces, reduction structures, or both.

Per Heath's "no single-problem rules" principle: cluster by template
family, parameterize across variants. The goal is ONE template per family
serving many L1 problems.

## The 17+ ops needing standalone templates

By family:

### Family 1: Reductions (8 ops)
- ggml_sum_rows
- ggml_mean
- ggml_max
- ggml_min
- ggml_argmax
- ggml_argmin
- ggml_cumsum
- ggml_cumprod

### Family 2: Normalizations (4 ops)
- ggml_norm        (general — LayerNorm, BatchNorm, InstanceNorm, Frobenius, L1)
- ggml_l2_norm
- ggml_rms_norm
- ggml_group_norm

### Family 3: Pooling (3 ops × max/avg variant = 6 effective)
- ggml_pool_1d (max | avg)
- ggml_pool_2d (max | avg)
- ggml_pool_3d (max | avg)

### Family 4: Convolutions (6 ops)
- ggml_conv_1d
- ggml_conv_2d
- ggml_conv_3d
- ggml_conv_transpose_1d
- ggml_conv_transpose_2d
- ggml_conv_transpose_3d

### Family 5: Losses (6 ops)
- ggml_mse_loss
- ggml_cross_entropy_loss
- ggml_huber_loss
- ggml_kl_div_loss
- ggml_triplet_margin_loss
- ggml_hinge_loss

### Special: Flash attention (1 op)
- ggml_flash_attn_ext (its own template — unique structure)

**Total: 5 families + 1 special = 6 templates for 17 ops.**

## Template family designs

### Family 1: Reduction template

```c
__global__ void reduce_<KIND>_<DIM>(const float * X, float * Y, int N, int outer) {
    int o = blockIdx.x;
    if (o >= outer) return;
    float acc = <INIT_VALUE>;  // 0 for sum/mean, -inf for max, +inf for min
    int arg = -1;
    for (int i = 0; i < N; i++) {
        float v = X[o * N + i];
        <ACCUMULATE>;  // acc += v / acc = max(acc, v) / etc.
    }
    Y[o] = <FINALIZE>;  // acc / N for mean, arg for argmax, acc for others
}
```

Parameterized by:
- KIND: sum, mean, max, min, argmax, argmin (or scan variants for cumsum/cumprod)
- INIT_VALUE: per kind
- ACCUMULATE: per kind (assignment expression on acc, arg, i, v)
- FINALIZE: per kind

cumsum and cumprod are slightly different (output same shape as input,
not reduction shape) — they're prefix-scan variants. Best handled as a
sub-template within Family 1.

### Family 2: Normalization template

```c
__global__ void norm_<KIND>(const float * X, float * Y, const float * W,
                            const float * B, int N, int outer, float eps) {
    int o = blockIdx.x;
    if (o >= outer) return;
    // Step 1: compute statistics over the N dim
    float <STAT1>, <STAT2>;
    for (int i = 0; i < N; i++) {
        float v = X[o * N + i];
        <STATS_UPDATE>;
    }
    // Step 2: normalize each element
    for (int i = 0; i < N; i++) {
        float v = X[o * N + i];
        Y[o * N + i] = <NORMALIZE>(v) * W[i] + B[i];  // affine if W,B provided
    }
}
```

Parameterized by:
- KIND: layer, rms, l2, l1, batch, instance, group, frobenius
- STAT1, STAT2: per kind (mean+var for layer; just rms for rms; etc.)
- STATS_UPDATE: per kind
- NORMALIZE: per kind

### Family 3: Pooling template

```c
__global__ void pool_<DIM>D_<KIND>(const float * X, float * Y,
                                    int B, int C, <SHAPES>, int K, int S, int P) {
    // K = kernel size, S = stride, P = padding
    int b = blockIdx.z;
    int c = blockIdx.y;
    int <OUT_COORDS> = blockIdx.x * blockDim.x + threadIdx.x;
    if (<OUT_COORDS_BOUNDS_CHECK>) return;
    float acc = <INIT>;
    for (<KERNEL_LOOP>) {
        int <IN_COORDS> = <OUT_COORDS> * S - P + <KERNEL_OFFSET>;
        if (<IN_BOUNDS>) {
            float v = X[<INDEX>];
            <ACCUMULATE>;  // max or +=
        }
    }
    Y[<OUT_INDEX>] = <FINALIZE>;  // acc or acc/K^DIM for avg
}
```

Parameterized by:
- DIM: 1, 2, 3 (loop nesting)
- KIND: max, avg
- INIT: -inf for max, 0 for avg
- ACCUMULATE: max(acc, v) for max, acc += v for avg
- FINALIZE: acc for max, acc / divisor for avg

### Family 4: Convolution template (im2col → matmul)

Substrate-honest framing: convolutions are best implemented as im2col
followed by matmul, then reshape. This decomposes naturally:
- im2col: layout transformation (existing layout class in classifier)
- matmul: existing fused-matmul template handles this
- reshape: layout

So Family 4 actually requires: an im2col template + dispatching through
existing matmul. Conv_transpose variants use col2im + matmul.

This is consistent with how production CUDA libraries implement conv
(cuDNN, MIOpen). The template here is the im2col part:

```c
__global__ void im2col_<DIM>D(const float * X, float * Y, ...config...) {
    // Standard im2col unfolding
}
```

Conv emission then becomes: emit_im2col → emit_matmul → emit_col2im
(if transposed).

### Family 5: Loss template

```c
__global__ void loss_<KIND>(const float * X, const float * Y, float * Out,
                            int N, int outer, ...config...) {
    // Per-batch reduction
    int b = blockIdx.x;
    if (b >= outer) return;
    float acc = 0.0;
    for (int i = 0; i < N; i++) {
        float xi = X[b * N + i], yi = Y[b * N + i];
        float diff = <ELEMENT_OP>(xi, yi);
        acc += diff;
    }
    Out[b] = <REDUCE>(acc, N);
}
```

Parameterized by:
- KIND: mse, cross_entropy, huber, kl_div, triplet_margin, hinge
- ELEMENT_OP: per kind (xi - yi for mse, etc.)
- REDUCE: sum or mean per kind

### Special: Flash attention template

Unique structure (tiled, online softmax). Not parameterizable like
the others. Single template, single op. Most complex of the six.

Implementation approach: start with the simple memory-efficient
attention (FlashAttention-1 style), not the full tile-aware version
that needs warp shuffles. The simple version is:

```c
__global__ void flash_attn(const float * Q, const float * K, const float * V,
                            float * O, int H, int N, int D, float scale, bool causal) {
    // Tiled forward pass with online softmax
}
```

## Per-template effort estimate

- Family 1 (reductions): ~3 hours (template + 8 op variants + tests)
- Family 2 (norms): ~2 hours (template + 4 op variants + tests)
- Family 3 (pooling): ~3 hours (template + 6 effective variants + tests)
- Family 4 (convolutions): ~4 hours (im2col + integration with matmul + tests)
- Family 5 (losses): ~2 hours (template + 6 op variants + tests)
- Special (flash attention): ~4 hours (its own complex template + tests)

**Total: ~18 hours of focused work.**

Doable in 2-3 sessions if collaborated with mavchin's parallel KV cache
work. Each family is independently committable; we don't need all 6 to
land before benefitting any of them.

## Suggested order

If we can't do all 6, prioritize by L1 problem coverage:
1. Family 3 (pooling) — covers L1 #41-46 = 6 problems
2. Family 5 (losses) — covers L1 #94-100 = 7 problems
3. Family 2 (norms) — covers L1 #33-40 = 8 problems (most coverage)
4. Family 1 (reductions) — covers L1 #47-53, #89-93 = 11 problems
5. Family 4 (convolutions) — covers L1 #50, #54-87 = 35 problems (most coverage by count)
6. Special (flash attention) — covers L1 #97 = 1 problem

Reordered for impact: 4 → 1 → 2 → 5 → 3 → 6 (conv first since 35 problems).

## Open questions for mavchin

1. Should Family 4 use im2col-then-matmul OR a direct conv kernel?
   im2col reuses existing matmul (simpler); direct conv may be faster.
2. For Family 1 (reductions), do we want a shared shuffle-based reduction
   or simple sequential? Shuffle is faster but warp-specific.
3. Naming convention: `reduce_sum_dim0`, `norm_layer`, `pool_2d_max`?
4. Should standalone templates be in `fusion_to_cuda.pl` alongside the
   fusion clauses, or in a separate file?

Standing by for input on these before implementation.

## Composition with existing substrate

- All templates use mavchin's existing CUDA AST (c_ast.pl)
- All emit via the same emit_program pipeline
- All add to fusion_analyzer's classify_op (already done in Scope B1)
- Tests verify both classification AND emission shape
- nvcc validation (Scope C) wraps the harness around all templates

The substrate composition stays clean: same AST, same emitter, same
test framework. Just more clauses.
