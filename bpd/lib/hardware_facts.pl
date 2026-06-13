%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% hardware_facts.pl — Hardware-specific facts for warp optimization.
%%
%% Per Heath's direction: "GPU warp optimization requires facts about
%% the hardware, and, when done correctly, is a huge speed-up that an
%% optimizer can produce."
%%
%% Each hardware target gets a tuple of facts describing its memory
%% hierarchy, parallelism granularity, and performance ratios. The
%% warp optimizer consults these facts when selecting tile sizes,
%% loop transformations, and memory access patterns.
%%
%% This module provides the FACTS. Per-target deterministic
%% compilation: the optimizer picks the optimal layout for the
%% specific hardware, not for a statistical mode.

%% Naming convention (aligned with mavchin's fusion analyzer):
%% - `hw_` prefix on all hardware-fact predicates avoids potential
%%   collisions with similarly-named predicates in other substrates
%%   (per Heath's thesaurus principle).
%% - `cache_line_size/2` is the one exception; predates the convention
%%   and is too widely referenced to rename without coordination.

:- module(hardware_facts, [
    hardware_target/1,
    hw_warp_size/2,
    hw_threads_per_block_max/2,
    hw_blocks_per_sm_max/2,
    hw_shared_memory_per_sm/2,
    hw_shared_memory_per_block_max/2,
    hw_shared_memory_per_block_optin/2,
    hw_registers_per_sm/2,
    hw_registers_per_thread_max/2,
    hw_l1_cache_size/2,
    hw_l2_cache_size/2,
    cache_line_size/2,
    hw_memory_bus_width/2,
    hw_memory_bandwidth_gb_s/2,
    hw_fp32_flops_tflops/2,
    hw_fp16_flops_tflops/2,
    hw_sm_count/2,
    hw_compute_capability/2,
    hw_max_threads_per_sm/2,
    hw_max_warps_per_sm/2,
    hw_clock_mhz/2,
    hw_supports_tensor_cores/2,
    hw_tensor_core_shapes/3
]).


%% clauses grouped by family, not contiguous — declared for warning-free consult.
:- discontiguous cache_line_size/2, hardware_target/1, hw_blocks_per_sm_max/2, hw_clock_mhz/2, hw_compute_capability/2, hw_fp16_flops_tflops/2, hw_fp32_flops_tflops/2, hw_l1_cache_size/2, hw_l2_cache_size/2, hw_max_threads_per_sm/2, hw_max_warps_per_sm/2, hw_memory_bandwidth_gb_s/2, hw_memory_bus_width/2, hw_registers_per_sm/2, hw_registers_per_thread_max/2, hw_shared_memory_per_block_max/2, hw_shared_memory_per_block_optin/2, hw_shared_memory_per_sm/2, hw_sm_count/2, hw_supports_tensor_cores/2, hw_threads_per_block_max/2, hw_warp_size/2.
%% ────────────────────────────────────────────────────────────────────
%% sm_61: NVIDIA Tesla P4 (Pascal architecture, GP104)
%% ────────────────────────────────────────────────────────────────────
%% Heath's actual target hardware. Specs from NVIDIA P4 datasheet.

hardware_target(sm_61).
hw_compute_capability(sm_61, '6.1').

hw_warp_size(sm_61, 32).
hw_threads_per_block_max(sm_61, 1024).
hw_blocks_per_sm_max(sm_61, 32).
hw_max_threads_per_sm(sm_61, 2048).        % max concurrent threads per SM
hw_max_warps_per_sm(sm_61, 64).            % 2048 / 32
hw_sm_count(sm_61, 20).

hw_shared_memory_per_sm(sm_61, 98304).            % 96 KB per SM
hw_shared_memory_per_block_max(sm_61, 49152).     % 48 KB per block (default)
hw_shared_memory_per_block_optin(sm_61, 49152).   % 48 KB (no opt-in on Pascal)
hw_supports_tensor_cores(sm_61, false).            % Pascal has no tensor cores
hw_registers_per_sm(sm_61, 65536).                % 64 KB worth
hw_registers_per_thread_max(sm_61, 255).

hw_l1_cache_size(sm_61, 24576).        % 24 KB (shared with shared mem on Pascal)
hw_l2_cache_size(sm_61, 2097152).      % 2 MB
cache_line_size(sm_61, 128).           % 128 bytes (no hw_ prefix; predates convention)

hw_memory_bus_width(sm_61, 256).       % bits
hw_memory_bandwidth_gb_s(sm_61, 192).  % GB/s peak
hw_clock_mhz(sm_61, 1531).             % max boost clock

hw_fp32_flops_tflops(sm_61, 5.5).      % ~5.5 TFLOPs FP32
hw_fp16_flops_tflops(sm_61, 5.5).      % P4 has no fast FP16 (Pascal w/o Tensor Cores)

%% ────────────────────────────────────────────────────────────────────
%% sm_89: NVIDIA RTX 4090 (Ada Lovelace, AD102)
%% ────────────────────────────────────────────────────────────────────
%% Modern reference target for comparison.

hardware_target(sm_89).
hw_compute_capability(sm_89, '8.9').

hw_warp_size(sm_89, 32).
hw_threads_per_block_max(sm_89, 1024).
hw_blocks_per_sm_max(sm_89, 24).
hw_max_threads_per_sm(sm_89, 1536).
hw_max_warps_per_sm(sm_89, 48).
hw_sm_count(sm_89, 128).

hw_shared_memory_per_sm(sm_89, 102400).            % 100 KB per SM
hw_shared_memory_per_block_max(sm_89, 49152).     % 48 KB per block (default)
hw_shared_memory_per_block_optin(sm_89, 100352).  % 98 KB per block (opt-in)
hw_supports_tensor_cores(sm_89, true).             % Ada Lovelace 4th-gen TC
hw_registers_per_sm(sm_89, 65536).
hw_registers_per_thread_max(sm_89, 255).

hw_l1_cache_size(sm_89, 131072).       % 128 KB L1 (configurable with shared)
hw_l2_cache_size(sm_89, 75497472).     % 72 MB
cache_line_size(sm_89, 128).

hw_memory_bus_width(sm_89, 384).
hw_memory_bandwidth_gb_s(sm_89, 1008).
hw_clock_mhz(sm_89, 2520).

hw_fp32_flops_tflops(sm_89, 82.6).
hw_fp16_flops_tflops(sm_89, 165.2).    % Tensor cores enable 2x FP16 throughput

%% ────────────────────────────────────────────────────────────────────
%% sm_80: NVIDIA A100 (Ampere, GA100)
%% ────────────────────────────────────────────────────────────────────
%% Data-center reference target.

hardware_target(sm_80).
hw_compute_capability(sm_80, '8.0').

hw_warp_size(sm_80, 32).
hw_threads_per_block_max(sm_80, 1024).
hw_blocks_per_sm_max(sm_80, 32).
hw_max_threads_per_sm(sm_80, 2048).
hw_max_warps_per_sm(sm_80, 64).
hw_sm_count(sm_80, 108).

hw_shared_memory_per_sm(sm_80, 167936).             % 164 KB per SM
hw_shared_memory_per_block_max(sm_80, 49152).       % 48 KB per block (default)
hw_shared_memory_per_block_optin(sm_80, 166912).    % 163 KB per block (opt-in)
hw_supports_tensor_cores(sm_80, true).               % Ampere 3rd-gen TC
hw_registers_per_sm(sm_80, 65536).
hw_registers_per_thread_max(sm_80, 255).

hw_l1_cache_size(sm_80, 196608).       % 192 KB (configurable)
hw_l2_cache_size(sm_80, 41943040).     % 40 MB
cache_line_size(sm_80, 128).

hw_memory_bus_width(sm_80, 5120).      % 5120 bit HBM2e
hw_memory_bandwidth_gb_s(sm_80, 1555).
hw_clock_mhz(sm_80, 1410).

hw_fp32_flops_tflops(sm_80, 19.5).
hw_fp16_flops_tflops(sm_80, 312.0).    % Tensor cores

%% ────────────────────────────────────────────────────────────────────
%% Notes on extensibility
%% ────────────────────────────────────────────────────────────────────
%%
%% Adding a new hardware target requires:
%%   1. New hardware_target/1 fact
%%   2. Per-axis facts for each of the predicates above
%%   3. Compute capability declaration
%%
%% The optimizer consults these facts via standard Prolog queries.
%% Each fact is a single source of truth — no duplication.
%%
%% For non-NVIDIA hardware (AMD, Intel, Apple Silicon, FPGAs):
%% the same predicate names apply with different units where needed.
%% E.g., AMD's "wave_size" maps to warp_size/2 (typically 64).

%% ────────────────────────────────────────────────────────────────────
%% Tensor core shapes (NVIDIA WMMA)
%% ────────────────────────────────────────────────────────────────────
%%
%% hw_tensor_core_shapes(+Hardware, +DataType, -Shapes)
%%   For hardware with tensor cores, list the supported tile shapes.
%%   Each shape is mma_shape(M, N, K) — the matmul tile a single
%%   tensor core instruction operates on.
%%
%% Pascal sm_61 has NO tensor cores (the predicate fails for it).
%% Ampere sm_80 supports more types than Ada sm_89; both share the
%% canonical FP16/BF16 shapes.

hw_tensor_core_shapes(sm_80, f16, [mma_shape(16, 16, 16),
                                    mma_shape(32, 8, 16),
                                    mma_shape(8, 32, 16)]).
hw_tensor_core_shapes(sm_80, bf16, [mma_shape(16, 16, 16),
                                     mma_shape(32, 8, 16),
                                     mma_shape(8, 32, 16)]).
hw_tensor_core_shapes(sm_80, tf32, [mma_shape(16, 16, 8)]).
hw_tensor_core_shapes(sm_80, int8, [mma_shape(16, 16, 32)]).

hw_tensor_core_shapes(sm_89, f16, [mma_shape(16, 16, 16),
                                    mma_shape(32, 8, 16),
                                    mma_shape(8, 32, 16)]).
hw_tensor_core_shapes(sm_89, bf16, [mma_shape(16, 16, 16),
                                     mma_shape(32, 8, 16),
                                     mma_shape(8, 32, 16)]).
hw_tensor_core_shapes(sm_89, tf32, [mma_shape(16, 16, 8)]).
hw_tensor_core_shapes(sm_89, int8, [mma_shape(16, 16, 32)]).
hw_tensor_core_shapes(sm_89, fp8, [mma_shape(16, 16, 32)]).   % new in Ada

%% Implication for warp optimizer: when targeting tensor cores, tile
%% dimensions should be MULTIPLES of one of the supported mma_shapes.
%% E.g., for FP16 on sm_80, tile(M, N, K) should have M, N multiples of
%% 16 and K a multiple of 16. The smallest tile leveraging TC is
%% 16x16x16 = 256 multiply-adds per instruction = 4096 FLOPs.
