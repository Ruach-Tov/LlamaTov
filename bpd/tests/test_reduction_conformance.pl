%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_reduction_conformance.pl — enforce that each f32-reduction KERNEL emits the
%% accumulation/reduction axis-value PINNED by its declared reference.
%%
%% Root cause of the attention V-sum regression (2026-06-01): the spec ALREADY had
%%   param_axis(gemm, reduction_grouping, [sequential, tiled, multi_acc(4), ...])
%%   reference_pins(ggml_b5311, reduction_grouping, tiled)
%% but NOTHING verified that our actual kernels EMIT their reference's pinned value.
%% bpd_gqa_attn_cpu emits 'sequential' (scalar a+b+c+d, proven by asm_facts) while its
%% reference ggml emits 'tiled' (8-wide SIMD tree-reduce, ggml_vec_dot_f32). The token
%% referee was blind (sub-argmax-margin), so it shipped. This test makes that a FAIL.
%%
%% kernel_emits_axis/3   — what OUR kernel actually does (ground truth: asm/code-verified).
%% kernel_targets/2      — which reference each kernel must be bit-identical to.
%% empirical_check/2     — the verify harness that proves it at 0-ULP.

:- module(test_reduction_conformance, []).

:- use_module('../lib/ir_param_axes').

%% ── which reference each f32-reduction kernel must match (DECLARED, per Heath) ──
kernel_targets(bpd_gqa_attn_cpu,   ggml_b5311).   %% llama path -> ggml
kernel_targets(bpd_mm_cpu,         torch_matmul_small). %% KernelBench f32 matmul -> torch
kernel_targets(bpd_qmatmul_q8_0,   ggml_b5311).   %% q8_0 decode -> ggml

%% ── what our kernel ACTUALLY emits for the reduction_grouping axis ──
%% (ground truth from asm_facts / code inspection — UPDATE when a kernel changes)
kernel_emits_axis(bpd_gqa_attn_cpu,   reduction_grouping, tiled).      %% FIXED: QK+V-sum tinyBLAS f16 (single __m256 acc + hsum), 0-ULP vs ggml node_20/node_22
kernel_emits_axis(bpd_mm_cpu,         reduction_grouping, sequential). %% matches torch_matmul_small @256-2048
kernel_emits_axis(bpd_qmatmul_q8_0,   reduction_grouping, tiled).      %% q8_0: integer accum, associative

%% ── the empirical 0-ULP harness that PROVES conformance (or fails) ──
empirical_check(bpd_gqa_attn_cpu, 'bench/verify_attention.py').
empirical_check(bpd_mm_cpu,       'bench/verify_mm_avx1.py').

%% ── the conformance check: kernel must emit what its reference pins ──
conforms(Kernel) :-
    kernel_targets(Kernel, Ref),
    kernel_emits_axis(Kernel, reduction_grouping, Emitted),
    ( reference_pins(Ref, reduction_grouping, Pinned)
    -> ( Emitted == Pinned
       -> true
       ;  ( empirical_check(Kernel, H) -> true ; H = '(no empirical harness)' ),
          format("  ~w: emits ~w but reference ~w pins ~w  [verify: ~w]~n",
                 [Kernel, Emitted, Ref, Pinned, H]),
          fail )
    ;  format("  ~w: reference ~w has no reduction_grouping pin~n", [Kernel, Ref]) ).

run :-
    format("~n=== reduction-order conformance ===~n"),
    findall(K, kernel_targets(K, _), Kernels),
    findall(K, (member(K, Kernels), \+ conforms(K)), Failed),
    ( Failed == []
    -> format("ALL KERNELS CONFORM (reduction order matches reference)~n"), halt(0)
    ;  length(Failed, N),
       format("~n~w KERNEL(S) NON-CONFORMANT: ~w~n", [N, Failed]),
       format("Fix: match the reference's reduction order, verify with the listed harness at 0-ULP.~n"),
       halt(1) ).

:- initialization(run).
