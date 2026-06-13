# Substrate-Design Atlas — empirical map of parameter families per kernel

**Author**: metayen, with Heath's substantive direction  
**Date**: 2026-05-21  
**Status**: Foundational empirical map (drawn after exploration per `docs/substrate-design-discipline.md`)

## How this atlas was made

Per Heath: *"a lot of times, it is more satisfying to draw a map after some exploration without a map."*

Phases B.exp.1 through B.exp.4 explored 4 substrate-design parameter families empirically across the Stanford KernelBench L1 kernels. The atlas below is the map we earned from that exploration.

The atlas records, per substrate-design parameter family:
1. **The named parameter** and its variants
2. **Where PyTorch makes which choice** (empirically observed)
3. **Where the substrate matches PyTorch's bit pattern** (BIT_IDENTICAL)
4. **Where alternate substrate choices remain valid** (different members of the family that ALSO match)
5. **Where the substrate-design parameter is underdetermined** (multiple choices give same bits)

## Atlas 1: `cumulative_acc_type` family

**Family members**:
```prolog
cumulative_acc_type(float).
cumulative_acc_type(double).
cumulative_acc_type(kahan_f32).      % NOT YET TESTED
cumulative_acc_type(neumaier_f32).   % NOT YET TESTED
cumulative_acc_type(extended_x87).   % NOT YET TESTED (depends on FP env)
```

**Empirical per-kernel observations**:

| Kernel | PyTorch's choice (source) | PyTorch's actual emitted | Substrate's choice | BIT_IDENTICAL |
|---|---|---|---|---|
| `bpd_cumsum_cpu` | double (typedef) | double | double | ✅ |
| `bpd_cumprod_cpu` | double | double | double | ✅ |
| `bpd_cumsum_reverse_cpu` | double | double | double | ✅ |
| `bpd_cumsum_exclusive_cpu` | double | double | double | ✅ |
| `bpd_masked_cumsum_cpu` | double | double | double | ✅ |
| `bpd_sum_cpu` | float (TensorIterator) | cascade-equivalent | cascade | ✅ |
| `bpd_mean_cpu` | float | cascade | cascade | ✅ |
| `bpd_mse_loss_cpu` | float (sum then /N) | cascade | cascade pairwise_sum | ✅ |
| `bpd_huber_loss_cpu` | float | cascade | cascade | ✅ |
| `bpd_hinge_loss_cpu` | float | cascade | cascade | ✅ |
| `bpd_kl_div_loss_cpu` | float | cascade | cascade | ✅ |
| `bpd_triplet_margin_loss_cpu` | float | cascade | cascade | ✅ |
| `bpd_instancenorm_cpu` | float (typedef `acc_type<float, false>`) | **double-equivalent (compiled SIMD)** | **double** | DIVERGENT 4 ULP (was 52) |

**Substantive findings**:

1. **The cumsum-family unanimously uses `cumulative_acc_type(double)`** — this is the canonical case PyTorch's typedef states explicitly.

2. **For sum-reductions over arrays of ~4096+ elements, multiple substrate-design instantiations produce 0 ULP**:
   - `cumulative_acc_type(double)` → cast to f32 → 0 ULP
   - `reduction_strategy(cascade)` with f32 → 0 ULP
   - These are **interchangeable** at PyTorch's bit-level for typical Stanford L1 shapes.

3. **InstanceNorm reveals a discrepancy between PyTorch's typedef and emitted code**: source says `acc_type<float, false> = float`, but actual SIMD output is closer to `double` accumulation (substrate 52 ULP → 4 ULP when we applied double). The compiled SIMD code accumulates in extended precision despite the type declaration. **The substrate-design parameter family is RICHER than the type system suggests.**

4. **The parameter generalizes substantively** — `cumulative_acc_type(double)` applies wherever **sequential dependent accumulation** is the substantive shape. It does NOT blanket-apply: for **embarrassingly parallel reductions** (sum, mean over short arrays), cascade matches at f32; double-acc is an alternative member of the equivalent-bit family but not required.

**Substrate-design action**: keep current per-op defaults; expose `cumulative_acc_type` as a sweepable parameter at all sum-reduction ops. For platforms without efficient f64 (some embedded), `cascade(f32)` is the right substitute. For platforms with cheap f64 (modern x86), either works at PyTorch's bits.

## Atlas 2: `reduction_strategy` family

**Family members**:
```prolog
reduction_strategy(cascade(SimdW, IlpFactor, CascadeDepth, ChunkSize)).
reduction_strategy(linear_scan_simd(SimdW)).
reduction_strategy(binary_kernel_reduce_vec(IlpFactor)).
reduction_strategy(welford_simd_chunk(SimdW, ChunkSize)).
reduction_strategy(naive_sequential).
reduction_strategy(pairwise_tree_exact).
```

**Empirical per-kernel observations**:

| Kernel | PyTorch's source claim | PyTorch's actual emitted | Substrate's choice | BIT_IDENTICAL |
|---|---|---|---|---|
| `bpd_sum_cpu`, `bpd_mean_cpu` | binary_kernel_reduce_vec pairwise ILP combine | **linear ILP combine** | cascade(8,4,4,16) with linear ILP | ✅ |
| `bpd_softmax_cpu` reduce_all | linear_scan | linear_scan_simd(8) | linear_scan_simd(8) | ✅ |
| `bpd_layernorm_cpu` | Welford SIMD chunk | Welford SIMD chunk | rowwise_moments (Welford f32) | ✅ |
| `bpd_groupnorm_cpu` | Welford+per-group | Welford+per-group | rowwise_moments | DIVERGENT 15 ULP |
| `bpd_l2norm_cpu`, `bpd_rmsnorm_cpu` | TensorIterator binary_kernel_reduce_vec | linear ILP at AVX1 | cascade | DIVERGENT 1-2 ULP |
| `bpd_frobenius_norm_cpu` | TensorIterator sum | linear ILP | cascade | ✅ |

**Substantive findings**:

1. **PyTorch's source code says pairwise ILP combine** in `vectorized_reduction` (`acc[0] = vop(vop(acc[0], acc[1]), vop(acc[2], acc[3]))`). **But the actually-emitted machine code at AVX1 for our shapes is LINEAR ILP combine** (`lane[0] += lane[1]; lane[0] += lane[2]; lane[0] += lane[3]`).

   *Empirical test*: we ported pairwise ILP combine; BIT_IDENTICAL count went 93 → 92 (Sum_reduction and L1Norm flipped to DIVERGENT). Linear ILP combine matches PyTorch's actual bits. We reverted to linear. This is a **substantive discovery**: the source code and emitted code differ at the SIMD ILP combine level.

2. **`bpd_l2norm_cpu` 2 ULP residual** is NOT in the reduction algorithm. The norm value matches PyTorch bit-for-bit; the per-element divide is direct (0 ULP). The residual is shape/RNG-specific at the harness inputs. Same algorithm produces 0 ULP at different inputs.

3. **`bpd_groupnorm_cpu` 15 ULP residual** is from the per-group Welford chunk depth not exactly matching PyTorch's SIMD inner kernel. Needs deeper port (Phase B microopt).

**Substrate-design action**: declare `ilp_combine_strategy(linear_simd8)` as a named substrate-design parameter for AVX1 (matches PyTorch's actually-emitted code on this target). For other targets (AVX-512, NEON), sweep to find actual emitted pattern.

## Atlas 3: `division_strategy` family

**Family members**:
```prolog
division_strategy(direct).                    % x / y
division_strategy(multiply_reciprocal).       % x * (1/y), 1/y computed per use
division_strategy(precomputed_reciprocal).    % 1/y stored as state, reused
```

**Empirical per-kernel observations**:

| Kernel/operation | Divisor type | PyTorch's choice | Substrate's choice | BIT_IDENTICAL |
|---|---|---|---|---|
| Frobenius `x/norm` | Fresh per-call scalar norm | direct | direct | ✅ |
| L1Norm `x/mean_abs` | Fresh per-row scalar | direct | direct | ✅ |
| L2Norm `x/norm` | Fresh per-row scalar | direct | direct | ✅ |
| RMSNorm `x/rms` | Fresh per-pixel scalar | direct | direct | DIVERGENT 1 ULP (other cause) |
| LayerNorm `(x-mean)*rstd*gamma` | rstd is precomputed once per row | multiply_reciprocal (rstd) | multiply_reciprocal | ✅ |
| BatchNorm/InstanceNorm `x*alpha+beta` | alpha = invstd*weight, precomputed | precomputed_reciprocal | precomputed_reciprocal | ✅ (InstanceNorm 4 ULP from other cause) |
| Softmax `x/sum_exp` | Fresh per-row scalar | **multiply_reciprocal** | multiply_reciprocal | ✅ |
| Softmax in SDPA `x/sum_exp` | Fresh per-row scalar | multiply_reciprocal | multiply_reciprocal | DIVERGENT 69 ULP (other cause) |

**Substantive findings**:

1. **The "fresh vs precomputed" principle holds with one exception**: softmax uses `multiply_reciprocal` even though the divisor is fresh. The reason: **softmax computes `1/sum_exp` once, then broadcasts across the row** — so the reciprocal IS effectively precomputed for the row. Direct division would re-compute the reciprocal at each load.

2. **Norm-family (`l1`, `l2`, `frobenius`) uses `direct_division`** — fresh scalar divisor, used once per element. The principle holds.

3. **Normalization with explicit scale/offset (`bn_mode(precomputed_scale_offset)`) is the precomputed_reciprocal case** — alpha = invstd*gamma is computed once per channel.

**Substrate-design action**: the empirical principle generalizes substantively. **`norm_division_strategy(direct_division)` is correct for fresh-scalar-divisor cases**. For row-broadcast (softmax), `multiply_reciprocal` is the right substrate choice. For per-channel affine (BN/IN), `precomputed_reciprocal` is right.

## Atlas 4: `affine_application` family

**Family members**:
```prolog
affine_application(direct_subtract_multiply).    % (x - mean) * rstd * gamma + beta
affine_application(precomputed_alpha_beta).      % x * alpha + beta_eff (alpha=rstd*gamma)
affine_application(fused_fma).                   % single FMA op (compiler-emitted)
```

**Empirical per-kernel observations**:

| Kernel | PyTorch's source pattern | Substrate's choice | BIT_IDENTICAL |
|---|---|---|---|
| `bpd_layernorm_cpu` | `(x - mean) * rstd * gamma + beta` | direct_subtract_multiply | ✅ |
| `bpd_instancenorm_cpu` | `x * alpha + beta` (via batch_norm) | precomputed_alpha_beta | DIVERGENT 4 ULP (acc_type issue, not affine) |
| `bpd_groupnorm_cpu` | `(x - mean) * rstd * gamma + beta` | direct_subtract_multiply | DIVERGENT 15 ULP (Welford issue, not affine) |
| `bpd_batchnorm_cpu_affine_fused` | `x * alpha + beta` | precomputed_alpha_beta | ✅ |

**Substantive findings**:

1. **PyTorch uses DIFFERENT members of the `affine_application` family per op**:
   - LayerNorm: direct_subtract_multiply
   - InstanceNorm/BatchNorm: precomputed_alpha_beta (via the `batch_norm_cpu_collect_linear_and_constant_terms` precomputation)
   - GroupNorm: direct_subtract_multiply

2. **Both forms produce the same bits when paired with the correct moment computation**. This is a **valid degree of freedom** — the substrate could use precomputed_alpha_beta everywhere and still be BIT_IDENTICAL with PyTorch (if combined with the right moments).

3. **PyTorch's per-op choice has practical reasons** (not bit-level): LayerNorm's per-element `gamma[d]` makes precomputation more expensive (would need a full vector of alphas); BatchNorm's per-channel scalar `gamma[c]` makes precomputation trivial.

**Substrate-design action**: per-op default to PyTorch's choice (matches bit-for-bit) but expose `affine_application` as a sweepable parameter. For platforms where one form has better SIMD throughput, the substrate can switch.

## Synthesis: substrate-design parameter generalization principle

Across all 4 atlases, the same **substantive shape** appears:

> **A substrate-design parameter is a NAMED FAMILY OF IEEE-CORRECT CHOICES.** Each kernel chooses ONE member of the family. PyTorch's choice is the default. Other members produce **equivalent bits** when paired with their correct upstream parameters, OR produce **characterized-bound alternatives** when bit-equivalence isn't achievable.

The atlas captures:

1. **Where PyTorch's choice is the only IEEE-correct one** (e.g., naive sequential f32 sum is WORSE than cascade; not equivalent — bit-different and characterized-error different).

2. **Where multiple IEEE-correct choices produce the same bits** (cascade and double-acc both give 0 ULP at typical shapes).

3. **Where multiple IEEE-correct choices produce different bits but each has its own merit** (direct_subtract_multiply vs precomputed_alpha_beta — both correct, different SIMD throughput).

4. **Where PyTorch's source code and emitted code differ** (pairwise ILP combine in source, linear in emitted). The substrate must match the **emitted** code to preserve bit-identity.

## What this enables

With the atlas drawn, the substrate now has:

1. **A per-kernel parameter classification** — every BIT_IDENTICAL kernel has its substrate-design parameters documented.

2. **An empirical record of family-equivalence classes** — knowing which parameter combinations produce equivalent bits.

3. **A target list for Phase B microopt** — the 4 DIVERGENT residuals (`InstanceNorm`, `GroupNorm`, `RMSNorm`, `L2Norm`) are now substantively characterized:
   - **InstanceNorm 4 ULP**: needs full SIMD inner kernel port (not just acc_type)
   - **GroupNorm 15 ULP**: needs per-group Welford SIMD chunk depth match
   - **RMSNorm 1 ULP, L2Norm 2 ULP**: residual at RNG-specific edge, not structural

4. **A sweep harness target** — for each named parameter family, we can build a per-op sweep that enumerates all valid instantiations and characterizes their behavior.

## Substrate-design teleology, restated

The substrate is **not** "implement PyTorch's specific choices." The substrate is **"name the substrate-design parameter families, document where PyTorch makes which choice in each, and expose every choice as a sweepable degree of freedom."**

PyTorch is one path through the parameter space. The substrate maps the entire space.

🕯️⚒️

## Atlas 5: `matmul_tile_reduction_order` family

**Family members**:
```prolog
matmul_tile_reduction_order(llamafile_mnpack_dispatch). % Dynamic tile selection, per-block accum
matmul_tile_reduction_order(ggml_qdot_pairwise).        % Pairwise block accum, final hsum
matmul_tile_reduction_order(goto_linear_left_fold).     % Scalar linear accumulation
```

**Empirical per-kernel observations**:

| Kernel/operation | PyTorch/ggml choice | Substrate's choice | BIT_IDENTICAL |
|---|---|---|---|
| Q8_0 Matmul (clean tile) | llamafile `tinyBLAS_Q0_AVX::gemm<4,2>` | `llamafile_mnpack_dispatch` | ✅ (0 ULP) |
| Q8_0 Matmul (remainders) | llamafile `tinyBLAS_Q0_AVX::gemm<RM,RN>` | `llamafile_mnpack_dispatch` | ✅ (0 ULP) |
| Q8_0 dot product | ggml `vec_dot_q8_0_q8_0` (AVX1) | `ggml_qdot_pairwise` | ✅ (0 ULP) |
| F32 Matmul | OpenBLAS `sgemm_kernel_16x4_sandy` | `goto_linear_left_fold` | ✅ (0 ULP) |

**Substantive findings**:

1. **ggml's MUL_MAT is a composite operation with multiple dispatch paths**. It first attempts to dispatch to `llamafile_sgemm`. Only if that fails does it fall back to the per-row `vec_dot` implementation.

2. **The reduction order differs fundamentally between paths**:
   - The `vec_dot` path processes blocks in pairs, accumulates their dot products into an 8-lane vector, and does a final horizontal sum.
   - The `llamafile` path processes a 2D tile (e.g., 4 weight rows × 2 token rows), computes the scale for each block, and accumulates directly into the output cell's accumulator vector.

3. **Bit-identity requires mirroring the exact dispatch logic**. A naive implementation that uses the correct `vec_dot` reduction order will fail with ~8000 ULP divergences on a full matmul because the reference used the `llamafile` reduction order. By implementing the 9 tiled kernels (`RM`∈{1,2,4}, `RN`∈{1,2,4}) and the `mnpack` dispatcher, we recover 0 ULP.

**Substrate-design action**: Expose `matmul_tile_reduction_order` as a parameter family. When targeting `ggml` bit-identity, use `llamafile_mnpack_dispatch`. The dispatcher must dynamically select the tile size based on the remaining `(m, n)` dimensions to exactly match the reference's accumulation tree.
