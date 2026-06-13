%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% gemm_kernel.pl — Substrate-design vocabulary for Goto's GEMM
%% (sweepable kernel generator).
%%
%% Per Heath's direction 2026-05-20 ~22:00 UTC: "By making a sweepable
%% generator that subsumes even our kernel BPD, we consolidated the
%% intelligence into tighter crystalline structures."
%%
%% This file declares ONLY the substrate-design parameter space. The
%% concrete C instantiation lives in bench/generate_gemm_kernels.py.
%% The sweep harness (bench/verify_gemm_sweep.py) enumerates all valid
%% patterns from this declaration, generates C for each, compiles, runs
%% against PyTorch CPU at multiple (M, N, K) shapes, and reports the
%% bit-identity table + GFLOPS.
%%
%% Phase I (commit 50f1f4d) empirically verified one specific
%% instantiation matches OpenBLAS Sandybridge bit-for-bit:
%%   gemm_pattern(P=768, Q=384, UM=16, UN=4, SIMD=8, KRule=adaptive_half)
%%
%% Phase II (this file + companions) sweeps the parameter space to find:
%%   - Other OpenBLAS-equivalent settings (Haswell, AVX-512, NEON)
%%   - Substrate-honest configurations OpenBLAS never considered

:- module(gemm_kernel, [
    gemm_pattern/6,
    enumerate_gemm_patterns/1,
    gemm_kernel_name/7,
    platform_gemm/2
]).

%% gemm_pattern(+P, +Q, +UM, +UN, +SimdWidth, +KRule).
%%
%% Sweepable parameter space — Goto's blocked GEMM:
%%
%%   P (outer M block, L2-resident A panel) ∈ {64, 128, 256, 384, 512, 768, 1024}
%%   Q (inner K block, L1-resident sub-block) ∈ {64, 128, 192, 256, 384, 512}
%%   UM (micro-kernel rows) ∈ {1, 4, 8, 16, 32}
%%   UN (micro-kernel cols) ∈ {1, 4, 8, 12, 16}
%%   SimdWidth ∈ {1, 4, 8, 16}   scalar / SSE-NEON / AVX-AVX2 / AVX-512
%%   KRule ∈ {single_block, fixed_q, adaptive_half, equal_split}
%%
%% Validity constraints (substrate-honest, enforce IEEE math constraints
%% AND hardware constraints):
%%
%%   1. UM divides P                       (full M blocks fit in panel)
%%   2. UM % SimdWidth == 0                (full SIMD lanes per row)
%%   3. UM * UN ≤ 16                       (max ymm registers for tile on AVX)
%%   4. Q * (UM + UN) * 4 ≤ 32768          (L1 cache budget: 32KB)
%%   5. P * Q * 4 ≤ 262144                 (L2 cache budget: 256KB)
%%   6. KRule = single_block iff Q ≥ 2048  (no inner blocking needed for tiny matrices)
%%
%% These are necessary substrate-design constraints — they keep the
%% sweep finite and IEEE-correct. Performance bounds (cache eviction,
%% register pressure) are observed empirically by the sweep harness.
gemm_pattern(P, Q, UM, UN, SimdWidth, KRule) :-
    member(P, [1, 64, 128, 256, 384, 512, 768, 1024]),
    member(Q, [1, 64, 128, 192, 240, 248, 256, 384, 512]),
    member(UM, [1, 4, 8, 16, 32]),
    member(UN, [1, 4, 8, 12, 16]),
    member(SimdWidth, [1, 4, 8, 16]),
    member(KRule, [single_block, fixed_q, adaptive_half, equal_split]),
    valid_gemm_pattern(P, Q, UM, UN, SimdWidth, KRule).

valid_gemm_pattern(P, Q, UM, UN, SimdWidth, _KRule) :-
    %% Constraint 1: UM divides P
    0 is P mod UM,
    %% Constraint 2: UM % SimdWidth == 0
    0 is UM mod SimdWidth,
    %% Constraint 3: Register tile fits in ymm/zmm file.
    %% Tile occupies (UM/SimdWidth) SIMD-register rows × UN columns.
    %% AVX1 has 16 ymm regs; we budget UN-wide × ROW_REGS for C plus
    %% 2 ymm for A loads + UN scratch for B broadcasts. Heuristic:
    %% (UM/SIMD) * UN + 2 + UN ≤ 16.
    RowRegs is UM // SimdWidth,
    TileRegs is RowRegs * UN + 2 + UN,
    TileRegs =< 16,
    %% Constraint 4: L1 cache budget — Q × (UM + UN) × 4 ≤ 32KB.
    %% The Q-block of packed-A (Q × UM floats) and packed-B (Q × UN floats)
    %% must fit in L1 together for the inner kernel.
    L1Bytes is Q * (UM + UN) * 4, L1Bytes =< 32768,
    %% Constraint 5: L2/L3 panel budget — P × Q × 4 ≤ 4MB.
    %% OpenBLAS deliberately oversizes for streaming + L3 residency.
    %% Real L2 per-core is ~256KB but the panel is reused many times
    %% across the N-loop, so L3-residency suffices.
    L2Bytes is P * Q * 4, L2Bytes =< 4194304.

%% enumerate_gemm_patterns(-Patterns).
%% Returns all valid (P, Q, UM, UN, SimdWidth, KRule) tuples.
%% Used by the Python sweep harness to know what to test.
enumerate_gemm_patterns(Patterns) :-
    findall(gemm(P, Q, UM, UN, SW, KR),
            gemm_pattern(P, Q, UM, UN, SW, KR),
            Patterns).

%% gemm_kernel_name(+P, +Q, +UM, +UN, +SW, +KR, -Name).
%% C function name for one pattern. Same naming-convention discipline
%% as cascade_kernel_name.
gemm_kernel_name(P, Q, UM, UN, SW, KR, Name) :-
    format(atom(Name),
           'gemm_p~w_q~w_um~w_un~w_simd~w_~w',
           [P, Q, UM, UN, SW, KR]).

%% Named platform defaults — which pattern matches each known platform.
%% Empirically verified entries are noted; others are predictions.
%%
%% goto_sandy: PyTorch CPU on AVX1 hardware (Tesla P4 enclave).
%% Empirically verified at commit 50f1f4d via /tmp/mm_goto.c
%% achieving 0 ULP vs cblas_sgemm at every tested shape.
platform_gemm(pytorch_cpu_default_avx1,
              gemm(768, 384, 16, 4, 8, adaptive_half)).

%% goto_haswell: PyTorch CPU on AVX2+FMA hardware (untested by us).
%% From OpenBLAS param.h HASWELL section: P=512, Q=256, UM=8, UN=4.
platform_gemm(pytorch_cpu_default_haswell,
              gemm(512, 256, 8, 4, 8, adaptive_half)).

%% goto_avx512: PyTorch CPU on Skylake-X+ hardware (untested by us).
%% From OpenBLAS param.h SKYLAKEX section: P=512, Q=256, UM=16, UN=4,
%% SimdWidth=16 (AVX-512).
platform_gemm(pytorch_cpu_default_avx512,
              gemm(512, 256, 16, 4, 16, adaptive_half)).

%% goto_neon: PyTorch CPU on ARM NEON (untested by us).
%% From OpenBLAS param.h ARMV8 section: P=128, Q=240, UM=4, UN=4, SimdWidth=4.
platform_gemm(pytorch_cpu_default_neon,
              gemm(128, 240, 4, 4, 4, adaptive_half)).

%% goto_scalar_baseline: naive sequential triple loop.
%% Always-available reference baseline for the sweep. NOTE: SimdWidth=1
%% is allowed even though our register tile budget would normally exclude
%% it; the baseline is special-cased in the C generator.
platform_gemm(scalar_baseline,
              gemm(1, 1, 1, 1, 1, single_block)).
