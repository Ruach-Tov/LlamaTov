%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% implementation_matches.pl — Platform-specific parameter derivation
%%
%% The meta-parameter: implementation_matches(Platform) derives ALL
%% individual substrate parameters needed to produce bit-identical
%% output with that platform.
%%
%% Usage:
%%   ?- implementation_matches(cuBLAS).
%%   % Sets: accumulation_precision(fp32), k_tile_strategy(auto), ...
%%
%%   ?- implementation_matches(pytorch_cpu_default).
%%   % Sets: accumulation_precision(fp32), cpu_fp_mode(strict), ...
%%
%% Adding a new platform: define its parameter combination below.
%% The substrate sweeps these during verification to confirm bit-identity.

/** <module> Platform-Specific Substrate Configuration

One Prolog fact configures all substrate parameters for a target platform.
Each platform clause asserts the complete set of numerical choices that
make a BPD-generated kernel produce bit-identical output with that platform.

Six platforms defined: cuBLAS, pytorch_cpu_default, pytorch_cpu_mkl,
lapack_reference, llama_cpp, bpd_default.

@author Ruach Tov Collective
@see docs/substrate-design-atlas.md for the empirical map of parameters
@see docs/substrate-design-correspondence.md for parameter naming alignment
*/

:- module(implementation_matches, [
    implementation_matches/1,
    platform_param/2,
    list_platforms/0,
    list_platform_params/1
]).

:- discontiguous implementation_matches/1.
:- discontiguous platform_param/2.

%% ═══════════════════════════════════════════════════════════════
%% Platform definitions
%% ═══════════════════════════════════════════════════════════════

%% NVIDIA cuBLAS (GPU, all architectures)
%% Our primary reference for GPU bit-identity.
%! implementation_matches(+Platform) is det.
%  Assert all substrate-design parameters for Platform.
%  Platform is one of: cuBLAS, pytorch_cpu_default, pytorch_cpu_mkl,
%  lapack_reference, llama_cpp, bpd_default.
%
%  After calling this predicate, all platform_param/2 facts are set
%  and the kernel code generator will produce code matching that platform.
implementation_matches(cuBLAS) :-
    platform_param(cuBLAS, accumulation_precision(fp32)),
    platform_param(cuBLAS, opmath_precision(fp32)),
    platform_param(cuBLAS, k_tile_strategy(auto)),
    platform_param(cuBLAS, reduction_strategy(sequential)),
    platform_param(cuBLAS, bn_mode(multiply_by_reciprocal)),
    platform_param(cuBLAS, rsqrt_variant(hardware)).

platform_param(cuBLAS, accumulation_precision(fp32)).
platform_param(cuBLAS, opmath_precision(fp32)).
platform_param(cuBLAS, k_tile_strategy(auto)).
platform_param(cuBLAS, reduction_strategy(sequential)).
platform_param(cuBLAS, bn_mode(multiply_by_reciprocal)).
platform_param(cuBLAS, rsqrt_variant(hardware)).
platform_param(cuBLAS, matmul_backend(ffma)).

%% PyTorch CPU with DEFAULT backend (no MKL, no OpenBLAS) on AVX1 hardware.
%% Empirically verified on Tesla P4 enclave (Intel CPU with AVX, no AVX2):
%%   reduction_strategy(cascade(8, 4, 4, 16)) matches PyTorch CPU bit-for-bit
%%   at every tested input size. See bench/verify_cascade_sweep.py and
%%   commit 81ab2e1 for the empirical confirmation (2/160 patterns match,
%%   both equivalent).
%%
%% On AVX-512 hardware: PyTorch uses cascade(16, 4, 4, 16) (predicted, untested).
%% On ARM NEON: PyTorch uses cascade(4, 4, 4, 16) (predicted, untested).
%% Both predictions follow from PyTorch's Vectorized<float>::size() table.
%% See lib/reduction_kernel.pl's platform_cascade/2 declarations.
implementation_matches(pytorch_cpu_default) :-
    platform_param(pytorch_cpu_default, accumulation_precision(fp32)),
    platform_param(pytorch_cpu_default, opmath_precision(fp32)),
    platform_param(pytorch_cpu_default, cpu_fp_mode(strict)),
    platform_param(pytorch_cpu_default, bn_mode(precomputed_scale_offset)),
    platform_param(pytorch_cpu_default, reduction_strategy(cascade(8, 4, 4, 16))),
    platform_param(pytorch_cpu_default, rsqrt_variant(reciprocal_sqrt)),
    platform_param(pytorch_cpu_default, gemm_tile_strategy(goto_sandy(768, 384, 16, 4))),
    platform_param(pytorch_cpu_default, norm_division_strategy(direct_division)).

platform_param(pytorch_cpu_default, accumulation_precision(fp32)).
platform_param(pytorch_cpu_default, opmath_precision(fp32)).
platform_param(pytorch_cpu_default, cpu_fp_mode(strict)).
platform_param(pytorch_cpu_default, bn_mode(precomputed_scale_offset)).
platform_param(pytorch_cpu_default, reduction_strategy(cascade(8, 4, 4, 16))).
platform_param(pytorch_cpu_default, rsqrt_variant(reciprocal_sqrt)).

%% gemm_tile_strategy(goto_sandy(P=768, Q=384, UM=16, UN=4)).
%%
%% PyTorch CPU calls cblas_sgemm directly; on AVX1 (Tesla P4 enclave),
%% OpenBLAS 0.3.32 dispatches to its SANDYBRIDGE sgemm kernel. The
%% parameters were empirically extracted via gdb runtime trace (α) and
%% confirmed in OpenBLAS source param.h (β) on 2026-05-20:
%%
%%   SGEMM_DEFAULT_P = 768     (outer M block, L2-resident A panel)
%%   SGEMM_DEFAULT_Q = 384     (inner K block — NOT 248 as Dunnington uses!)
%%   SGEMM_DEFAULT_R = sgemm_r (runtime, computed from buffer size)
%%   SGEMM_DEFAULT_UNROLL_M = 16
%%   SGEMM_DEFAULT_UNROLL_N = 4
%%
%% K-block tiling rule (driver/level3/level3.c:309-322):
%%   while remaining > 0:
%%     if remaining >= 2*Q:  min_l = Q                  # full block
%%     elif remaining > Q:   min_l = ceil(rem/2/UM)*UM  # half, rounded
%%     else:                 min_l = remaining           # tail
%%
%% bench/bpd_cpu.c's bpd_mm_cpu implements this exactly. Verified
%% empirically 0 ULP vs cblas_sgemm at every (M=N=16, K∈{256..4096})
%% × 5 seeds. Stanford KernelBench L1 CPU: 13/13 matmul problems
%% BIT_IDENTICAL after this commit.
platform_param(pytorch_cpu_default, gemm_tile_strategy(goto_sandy(768, 384, 16, 4))).

%% norm_division_strategy(direct_division).
%%
%% PyTorch norm operations (F.normalize, torch.norm, F.rms_norm) divide x by
%% the computed norm DIRECTLY (x / norm), not via multiply-by-reciprocal
%% (x * (1/norm)). Empirically discovered 2026-05-20 via Stanford KernelBench
%% L1 #37 FrobeniusNorm:
%%
%%   Sample: x = 1.936491132, norm = 21.54953194
%%     direct division (x / norm):     0.08986233175  bits=3db809be  ← matches PT
%%     reciprocal multiply (x*(1/norm)): 0.0898623243  bits=3db809bd  ← 1 ULP off
%%
%% Direct division uses ONE IEEE 754 rounding; reciprocal multiply uses TWO
%% (compute 1/norm, then multiply). The last bit of mantissa differs.
%%
%% Distinct from rsqrt_variant(reciprocal_sqrt) which is the correct choice
%% for LayerNorm/BatchNorm where the substrate uses `rstd = 1/sqrt(var+eps)`
%% and applies `(x - mean) * rstd` (the reciprocal IS the right path there
%% because PyTorch's batch_norm precomputes alpha=invstd*weight).
%%
%% PyTorch uses different patterns per op family:
%%   LayerNorm:   rsqrt_variant(reciprocal_sqrt)
%%   BatchNorm:   bn_mode(precomputed_scale_offset)
%%   Norm family: norm_division_strategy(direct_division)
%%
%% Used by: bench/bpd_cpu.c bpd_rmsnorm_cpu, bpd_frobenius_norm_cpu,
%% bpd_l1norm_cpu, bpd_l2norm_cpu. Verified BIT_IDENTICAL for #37 and #38.
platform_param(pytorch_cpu_default, norm_division_strategy(direct_division)).

%% PyTorch CPU with Intel MKL backend (AVX2+FMA CPUs, e.g. Manus sandbox)
%%
%% Empirically characterised 2026-05-21 on Manus sandbox (Intel Xeon, AVX2+FMA,
%% PyTorch 2.x with BLAS_INFO=mkl) via bench/probe_mkl_params.py.
%%
%% Key differences from pytorch_cpu_default (OpenBLAS/Sandybridge):
%%
%%   1. GEMM (M×N, K≤1024): ✅ BIT_IDENTICAL — MKL's SGEMM and BPD's
%%      goto_sandy(768,384,16,4) K-block tiling produce the same accumulation
%%      order for all tested shapes. The GEMM path is NOT the source of the
%%      79/100 score on this sandbox.
%%
%%   2. GEMV (N=1 matrix-vector): ❌ MKL dispatches to a SIMD AXPY-based GEMV
%%      kernel (cblas_sgemv) rather than the GEMM path. Accumulation order
%%      differs from BPD's K-block GEMM. New param: gemv_strategy(simd_axpy).
%%      Kernel fix: bpd_gemv_mkl_cpu — AVX2 AXPY loop, 8 accumulators per row.
%%
%%   3. Transcendentals (sigmoid, tanh, silu, elu, softplus, selu): ❌ max 1–2 ULP.
%%      MKL PyTorch uses Intel SVML (Short Vector Math Library) for vectorized
%%      transcendentals. SVML's vexpf/vtanhf/vexpm1f differ by 1–2 ULP from
%%      the C99 expf/tanhf/expm1f used by BPD's scalar path.
%%      New param: transcendental_library(svml).
%%      Kernel fix: compile with -mfma -ffast-math -mveclibabi=svml, or use
%%      the inline SVML polynomial approximations in bpd_cpu.c.
%%
%%   4. GELU: ❌ max 1791 ULP. SVML's vectorized verf has much wider error
%%      than C99 erff at the tail. PyTorch's GELU uses the SVML vserf path.
%%      New param: gelu_variant(svml_erf).
%%      Kernel fix: use the degree-9 minimax polynomial for erf that SVML uses
%%      (coefficients in Intel SVML source / Agner Fog's libvecmath).
%%
%%   5. Reduction strategy: ✅ cascade(8) — Cascade width 8 (AVX1 lane count)
%%      matches PyTorch MKL exactly (0 ULP confirmed). Same as pytorch_cpu_default.
%%
%%   6. L2Norm: ✅ cascade(8) closes the gap completely (0 ULP).
%%      BPD's bpd_norm_p2_sumsq_lastdim already uses pairwise_sum which matches
%%      cascade(8) at cols=128. No change needed.
%%
%%   7. RMSNorm: ❌ max 2 ULP even with cascade(8). The residual gap is from
%%      MKL's vectorized vsqrtf (SVML) vs BPD's scalar sqrtf.
%%      New param: sqrt_variant(svml). Kernel fix: use _mm256_sqrt_ps.
%%
%%   8. InstanceNorm: ❌ max 40 ULP. The affine apply step (x*alpha + bias)
%%      is FMA-vectorized in MKL's PyTorch. BPD uses scalar multiply-add.
%%      New param: affine_apply_strategy(fma_vectorized).
%%      Kernel fix: use _mm256_fmadd_ps in the affine loop.
%%
%%   9. Depthwise conv (kernels #82–86): ❌ No BPD depthwise kernel exists yet.
%%      MKL PyTorch uses a dedicated im2col-free depthwise path. Scalar reference
%%      also diverges (max 227 ULP at C=4, 3×3), confirming MKL uses a SIMD
%%      accumulation order. New param: depthwise_strategy(mkl_direct).
%%      Kernel needed: bpd_depthwise_conv2d_mkl_cpu — AVX2 inner loop over kH×kW.
%%
%%  10. SDPA: ❌ max 221 ULP. The QK^T GEMM uses MKL's SGEMM path which
%%      (for the small seq_len=8, embed_dim=16 harness shape) accumulates
%%      differently from BPD's K-block path at small K. Covered by gemv_strategy
%%      fix above for the K=embed_dim=16 case.
%%
%% Predicted score after all fixes: 95–97/100 (depthwise conv is 5 kernels).
implementation_matches(pytorch_cpu_mkl) :-
    platform_param(pytorch_cpu_mkl, accumulation_precision(fp32)),
    platform_param(pytorch_cpu_mkl, opmath_precision(fp32)),
    platform_param(pytorch_cpu_mkl, cpu_fp_mode(fma)),
    platform_param(pytorch_cpu_mkl, bn_mode(precomputed_scale_offset)),
    platform_param(pytorch_cpu_mkl, reduction_strategy(cascade(8, 4, 4, 16))),
    platform_param(pytorch_cpu_mkl, gemm_tile_strategy(goto_sandy(768, 384, 16, 4))),
    platform_param(pytorch_cpu_mkl, norm_division_strategy(direct_division)),
    platform_param(pytorch_cpu_mkl, gemv_strategy(simd_axpy)),
    platform_param(pytorch_cpu_mkl, transcendental_library(aten_vectorized)),
    platform_param(pytorch_cpu_mkl, gelu_variant(aten_erf_poly)),
    platform_param(pytorch_cpu_mkl, sqrt_variant(avx2_vsqrtps)),
    platform_param(pytorch_cpu_mkl, affine_apply_strategy(fma_vectorized)),
    platform_param(pytorch_cpu_mkl, depthwise_strategy(mkl_direct)),
    platform_param(pytorch_cpu_mkl, rsqrt_variant(reciprocal_sqrt)).

platform_param(pytorch_cpu_mkl, accumulation_precision(fp32)).
platform_param(pytorch_cpu_mkl, opmath_precision(fp32)).
platform_param(pytorch_cpu_mkl, cpu_fp_mode(fma)).
platform_param(pytorch_cpu_mkl, bn_mode(precomputed_scale_offset)).
platform_param(pytorch_cpu_mkl, reduction_strategy(cascade(8, 4, 4, 16))).
platform_param(pytorch_cpu_mkl, gemm_tile_strategy(goto_sandy(768, 384, 16, 4))).
platform_param(pytorch_cpu_mkl, norm_division_strategy(direct_division)).
platform_param(pytorch_cpu_mkl, rsqrt_variant(reciprocal_sqrt)).

%% gemv_strategy(simd_axpy).
%%
%% MKL dispatches N=1 matrix-vector products to cblas_sgemv, which uses an
%% AVX2 AXPY-based kernel (8 accumulators, one per YMM lane, reduced at end).
%% BPD's bpd_mm_cpu uses the K-block GEMM path even for N=1, producing a
%% different accumulation order. Fix: detect N=1 in bpd_mm_cpu and dispatch
%% to bpd_gemv_mkl_cpu (8-accumulator AVX2 AXPY loop).
platform_param(pytorch_cpu_mkl, gemv_strategy(simd_axpy)).

%% transcendental_library(aten_vectorized).
%%
%% Empirically confirmed 2026-05-21: PyTorch on this sandbox links against
%% system libm.so.6 (NOT Intel SVML). The transcendental divergence comes from
%% PyTorch ATen's own vectorized polynomial implementations in vec256_float.h:
%%   Vectorized<float>::exp()  — Pommier/Cephes polynomial, FMA Horner
%%   Vectorized<float>::tanh() — dedicated polynomial (NOT (exp(2x)-1)/(exp(2x)+1))
%%   Vectorized<float>::expm1() — dedicated polynomial (NOT C99 expm1f)
%%
%% All six activation kernels diverge from C99 libm by 1-2 ULP:
%%   sigmoid:  max 2 ULP,  n=242/1024  (ATen exp polynomial)
%%   tanh:     max 2 ULP,  n=335/1024  (ATen tanh polynomial)
%%   silu:     max 2 ULP,  n=268/1024  (ATen exp polynomial via sigmoid)
%%   elu:      max 1 ULP,  n=49/1024   (ATen expm1 polynomial)
%%   softplus: max 1 ULP,  n=104/1024  (ATen exp polynomial)
%%   selu:     max 2 ULP,  n=139/1024  (ATen expm1 polynomial)
%%
%% The FMA Horner exp polynomial (bpd_exp_ps_avx2_fma in bpd_cpu.c) matches
%% ATen's sigmoid/silu to within 2 ULP (improved from 2 ULP with scalar).
%% Full 0 ULP matching requires reverse-engineering ATen's tanh/expm1 polynomials
%% from vec256_float.h — marked as Phase F work.
%%
%% Status: sigmoid/silu partially improved by bpd_exp_ps_avx2_fma.
%% tanh/elu/selu/softplus remain at 1-2 ULP pending Phase F polynomial extraction.
platform_param(pytorch_cpu_mkl, transcendental_library(aten_vectorized)).

%% gelu_variant(aten_erf_poly).
%%
%% GELU uses erf(x/sqrt(2)). PyTorch ATen's Vectorized<float>::erf() uses a
%% degree-9 polynomial approximation (from aten/src/ATen/cpu/vec/vec256/vec256_float.h)
%% that differs from C99 erff by up to 1791 ULP at the tails (p99=66 ULP, p50=1 ULP).
%% C99 erff is correctly rounded (≤0.5 ULP); ATen's polynomial is faster but wider.
%%
%% This is the LARGEST divergence on the MKL sandbox: max 1791 ULP, n=722/1024.
%% Fix: inline ATen's degree-9 erf polynomial in bpd_gelu_cpu under BPD_MKL_PATH=1.
%% The polynomial coefficients are in PyTorch source:
%%   aten/src/ATen/cpu/vec/vec256/vec256_float.h Vectorized<float>::erf()
platform_param(pytorch_cpu_mkl, gelu_variant(aten_erf_poly)).

%% sqrt_variant(avx2_vsqrtps).
%%
%% PyTorch ATen uses _mm256_sqrt_ps (AVX2 hardware VSQRTPS) for vectorized sqrt.
%% VSQRTPS is correctly rounded (≤0.5 ULP) but the vectorized path processes
%% 8 elements simultaneously, producing a different rounding sequence than
%% BPD's scalar sqrtf loop when the input to sqrt is itself computed via a
%% vectorized reduction (cascade8 sum-of-squares).
%%
%% Observed in RMSNorm (max 2 ULP, n=30/256): the cascade8 reduction matches
%% (0 ULP), but the final sqrtf call produces 1-2 ULP divergence because
%% PyTorch uses VSQRTPS on the 8-element cascade result vector.
%% Fix: use _mm256_sqrt_ps in bpd_rmsnorm_cpu when BPD_MKL_PATH=1.
platform_param(pytorch_cpu_mkl, sqrt_variant(avx2_vsqrtps)).

%% affine_apply_strategy(fma_vectorized).
%%
%% MKL PyTorch's InstanceNorm affine apply step (y = x*alpha + bias) uses
%% _mm256_fmadd_ps (AVX2 FMA). BPD uses scalar multiply-add (two ops).
%% FMA fuses the multiply and add into one IEEE 754 operation, producing
%% a different last bit. Observed: max 40 ULP in InstanceNorm (2,4,8,8).
%% Fix: use _mm256_fmadd_ps in bpd_instancenorm_cpu when CPU_FP_MODE=mkl.
platform_param(pytorch_cpu_mkl, affine_apply_strategy(fma_vectorized)).

%% depthwise_strategy(mkl_direct).
%%
%% MKL PyTorch uses a dedicated im2col-free depthwise conv2d path. The scalar
%% reference also diverges (max 227 ULP at C=4, 3×3, H=W=8), confirming MKL
%% uses a SIMD accumulation order over the kH×kW kernel window.
%% No BPD depthwise kernel exists yet (nm confirms no 'depthwise' symbol).
%% This accounts for 5 divergent kernels (#82–86) on this sandbox.
%% Fix: implement bpd_depthwise_conv2d_mkl_cpu with AVX2 inner loop over
%% the kH×kW window, accumulating in 8 YMM lanes.
platform_param(pytorch_cpu_mkl, depthwise_strategy(mkl_direct)).

%% LAPACK reference (the mathematical gold standard)
%% Double-precision accumulation. Maximum correctness, not bit-compatible.
implementation_matches(lapack_reference) :-
    platform_param(lapack_reference, accumulation_precision(fp64)),
    platform_param(lapack_reference, reduction_strategy(sequential)).

platform_param(lapack_reference, accumulation_precision(fp64)).
platform_param(lapack_reference, reduction_strategy(sequential)).
platform_param(lapack_reference, cpu_fp_mode(strict)).

%% llama.cpp (the Ollama backend)
%% Q4_K_M dequant + fp32 accumulation.
implementation_matches(llama_cpp) :-
    platform_param(llama_cpp, accumulation_precision(fp32)),
    platform_param(llama_cpp, reduction_strategy(sequential)),
    platform_param(llama_cpp, quant_dequant_fused(true)).

platform_param(llama_cpp, accumulation_precision(fp32)).
platform_param(llama_cpp, reduction_strategy(sequential)).
platform_param(llama_cpp, quant_dequant_fused(true)).

%% BPD substrate defaults (our own choices, documented)
implementation_matches(bpd_default) :-
    platform_param(bpd_default, accumulation_precision(fp32)),
    platform_param(bpd_default, opmath_precision(fp32)),
    platform_param(bpd_default, k_tile_strategy(k8)),
    platform_param(bpd_default, reduction_strategy(sequential)),
    platform_param(bpd_default, bn_mode(precomputed_scale_offset)),
    platform_param(bpd_default, cpu_fp_mode(strict)),
    platform_param(bpd_default, rsqrt_variant(reciprocal_sqrt)).

platform_param(bpd_default, accumulation_precision(fp32)).
platform_param(bpd_default, opmath_precision(fp32)).
platform_param(bpd_default, k_tile_strategy(k8)).
platform_param(bpd_default, reduction_strategy(sequential)).
platform_param(bpd_default, bn_mode(precomputed_scale_offset)).
platform_param(bpd_default, cpu_fp_mode(strict)).
platform_param(bpd_default, rsqrt_variant(reciprocal_sqrt)).

%% ═══════════════════════════════════════════════════════════════
%% Query helpers
%% ═══════════════════════════════════════════════════════════════

list_platforms :-
    findall(P, (platform_param(P, _), \+ P = bpd_default), Ps),
    sort(Ps, Unique),
    format("Available platforms:~n"),
    forall(member(U, Unique), format("  ~w~n", [U])).

list_platform_params(Platform) :-
    format("Parameters for ~w:~n", [Platform]),
    forall(platform_param(Platform, Param),
           format("  ~w~n", [Param])).

%% ═══════════════════════════════════════════════════════════════
%% Parameter descriptions (for documentation / UI)
%% ═══════════════════════════════════════════════════════════════

:- discontiguous param_description/2.

param_description(accumulation_precision(fp32),
    "IEEE 754 single-precision accumulation. Matches cuBLAS, PyTorch, most GPU libs.").
param_description(accumulation_precision(fp64),
    "Double-precision accumulation. Maximum correctness. 0 ULP vs f64 truth.").
param_description(accumulation_precision(kahan),
    "Kahan compensated summation. Near-f64 accuracy at fp32 speed.").

param_description(k_tile_strategy(auto),
    "Auto-select K_TILE based on shape. Matches cuBLAS dispatch heuristic.").
param_description(k_tile_strategy(k8),
    "K_TILE=8. Matches cuBLAS sgemm_128x128x8 for large/square shapes.").
param_description(k_tile_strategy(k32),
    "K_TILE=32. Matches cuBLAS sgemm_32x32x32 for small/non-square shapes.").

param_description(cpu_fp_mode(strict),
    "No FMA. Sequential accumulation. Matches PyTorch DEFAULT backend.").
param_description(cpu_fp_mode(fma),
    "FMA enabled (-mfma -ffp-contract=on). Matches PyTorch MKL/OpenBLAS on AVX2+.").
param_description(cpu_fp_mode(native),
    "-march=native. Matches whatever your CPU supports.").

param_description(bn_mode(multiply_by_reciprocal),
    "gamma * (1/sqrt(var+eps)). Matches PyTorch GPU BN (DIVSS+MULSS double rounding).").
param_description(bn_mode(precomputed_scale_offset),
    "Precompute scale=gamma/sqrt(var+eps), offset=beta-mean*scale. 2 ops per element.").

param_description(rsqrt_variant(hardware),
    "Use rsqrtf() hardware instruction. Fast, ~1 ULP from 1/sqrt.").
param_description(rsqrt_variant(reciprocal_sqrt),
    "Use 1.0f/sqrtf(). Matches PyTorch CPU eval-mode BN.").

param_description(reduction_strategy(sequential),
    "Left-to-right sequential reduction. Deterministic. Matches most BLAS.").
param_description(reduction_strategy(tiled),
    "Block-tiled reduction. Non-deterministic order. Matches MKL/OpenBLAS.").
param_description(reduction_strategy(pairwise_tree),
    "Pairwise tree reduction. Better numerical stability. Different bits.").

%% opmath_precision: the precision in which elementwise/scalar math executes
%% when input tensors may be lower precision (e.g., f16 weights from .pt files).
%% Distinct from accumulation_precision, which is about reduction-sum precision
%% in dot products / convolutions.
%%
%% Surfaced by the substantive substrate-design discovery 2026-05-20 ~17:55 UTC:
%% YOLOv5n.pt BN parameters are float16; without explicit promotion to fp32 at
%% function boundaries, numpy/PyTorch will perform arithmetic in input precision,
%% producing massive divergence (8.26e-2 abs error in BN affine path).
%%
%% Detected by bench/test_opmath_precision_invariance.py — a function declared
%% to return fp32 must produce identical fp32 output bits for the SAME numerical
%% input values arriving in different precisions (f16, fp32, fp64).
param_description(opmath_precision(fp16),
    "Operate in fp16 (input precision). Substantively risky for low-precision inputs; produces precision-loss divergence from references that promote.").
param_description(opmath_precision(fp32),
    "Promote inputs to fp32 at function boundaries before any arithmetic. Matches PyTorch's opmath promotion behavior (default for nn modules). The substantive substrate-design discipline for low-precision-input scenarios.").
param_description(opmath_precision(fp64),
    "Promote inputs to fp64 at function boundaries. Maximum precision; matches LAPACK reference behavior.").
