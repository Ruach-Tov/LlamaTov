%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% generate_cfd_kernels.pl — CFD CUDA kernel generation.
%%
%% Subtask C6 of the CFD beachhead. Assembles all four CFD kernel emits
%% (k_compute_flux, k_update_conservative, k_compute_primitives,
%% k_cfl_condition) into a single .cu output for compilation and
%% verification via medayek's test_cfd_substrate.py harness.
%%
%% PER 2026-05-18 ~20:00 UTC METHODOLOGY:
%%
%% ZERO write/1. Every line from AST terms through emit_program/2.
%% Same generation discipline as generate_llama_kernels.pl.
%%
%% CROSS-DOMAIN HELPER REUSE:
%% k_cfl_condition calls block_reduce_max — a __device__ helper that
%% lives in kernel_templates_llama.pl (commit dc0b8be32). This program
%% file pulls block_reduce_max_helper AND its dependency warp_reduce_max_helper
%% into the CFD output alongside the CFD-specific kernels. Cross-domain
%% dependency at the program-assembly level, not the substrate-emit level.
%%
%% This is the substrate-honest expression of the fact that block reduction
%% is genuinely cross-domain — softmax, rmsnorm (ML), and CFL condition (CFD)
%% all need the same warp_reduce + cross-warp pattern. A future
%% kernel_templates_common.pl would naturally collect such primitives.

:- use_module('lib/c_ast').
:- use_module('lib/kernel_templates_cfd').
:- use_module('lib/kernel_templates_llama').


%% ═════════════════════════════════════════════════════════════════════════════
%% GPU MEMORY MANAGEMENT (shared with generate_llama_kernels.pl)
%% ═════════════════════════════════════════════════════════════════════════════
%%
%% The CFD output exposes the same gpu_alloc / gpu_free / gpu_h2d / gpu_d2h /
%% gpu_sync interface that the ML output exposes. Medayek's test harness
%% allocates GPU buffers, copies host arrays via gpu_h2d, calls gpu_compute_flux,
%% reads back via gpu_d2h, and compares to the Python Roe reference.

mem_alloc(F) :-
    F = c_func(c_type(ptr(c_type(void))), gpu_alloc,
        [param(c_type(int), bytes)],
        [c_decl(c_type(ptr(c_type(void))), p),
         c_expr_stmt(c_call(cudaMalloc, [c_unop('&', c_var(p)), c_var(bytes)])),
         c_return(c_var(p))]).

mem_free(F) :-
    F = c_func(c_type(void), gpu_free,
        [param(c_type(ptr(c_type(void))), p)],
        [c_expr_stmt(c_call(cudaFree, [c_var(p)]))]).

mem_h2d(F) :-
    F = c_func(c_type(void), gpu_h2d,
        [param(c_type(ptr(c_type(void))), dst),
         param(c_type(const_ptr(c_type(void))), src),
         param(c_type(int), bytes)],
        [c_expr_stmt(c_call(cudaMemcpy,
            [c_var(dst), c_var(src), c_var(bytes),
             c_var(cudaMemcpyHostToDevice)]))]).

mem_d2h(F) :-
    F = c_func(c_type(void), gpu_d2h,
        [param(c_type(ptr(c_type(void))), dst),
         param(c_type(const_ptr(c_type(void))), src),
         param(c_type(int), bytes)],
        [c_expr_stmt(c_call(cudaMemcpy,
            [c_var(dst), c_var(src), c_var(bytes),
             c_var(cudaMemcpyDeviceToHost)]))]).

mem_sync(F) :-
    F = c_func(c_type(void), gpu_sync,
        [],
        [c_expr_stmt(c_call(cudaDeviceSynchronize, []))]).


%% ═════════════════════════════════════════════════════════════════════════════
%% PROGRAM ASSEMBLY
%% ═════════════════════════════════════════════════════════════════════════════

generate_all(Code) :-
    %% Cross-domain helpers from kernel_templates_llama.pl:
    %% k_cfl_condition (C5) calls block_reduce_max which in turn calls
    %% warp_reduce_max. Both __device__ helpers must precede the kernel
    %% that uses them.
    warp_reduce_max_helper(WRMH),
    block_reduce_max_helper(BRMH),

    %% CFD kernel emits from kernel_templates_cfd.pl:
    cfd_flux_kernel(k_compute_flux, FluxK),
    cfd_flux_wrapper(k_compute_flux, FluxW),
    cfd_update_conservative_kernel(k_update_conservative, UpdateK),
    cfd_update_conservative_wrapper(k_update_conservative, UpdateW),
    cfd_compute_primitives_kernel(k_compute_primitives, PrimK),
    cfd_compute_primitives_wrapper(k_compute_primitives, PrimW),
    cfd_cfl_condition_kernel(k_cfl_condition, CflK),
    cfd_cfl_condition_wrapper(k_cfl_condition, CflW),

    %% Memory management functions
    mem_alloc(MA), mem_free(MF), mem_h2d(MH), mem_d2h(MD), mem_sync(MS),

    Program = [
        c_include_sys('cuda_runtime.h'),
        c_include_sys('math.h'),
        c_blank,
        c_comment('BPD-generated CUDA kernels for CFD (Sod shock tube)'),
        c_comment('100% AST-generated via Prolog c_ast DCG emitter'),
        c_comment('ZERO write/1 strings. Every line from AST terms.'),
        c_comment('Per the physics-for-physics correctness framing:'),
        c_comment('  reference is analytical Riemann solution (Sod), not another implementation.'),
        c_comment('Roe-vs-Roe bit-identical is the substrate emit verification.'),
        c_blank,
        c_comment('=== __device__ helpers (cross-domain reuse from kernel_templates_llama) ==='),
        WRMH, c_blank,
        BRMH, c_blank,
        c_comment('=== CFD kernels (4 total, all algorithmically aligned with mavchin sod_gpu_kernels.cu) ==='),
        c_blank,
        c_comment('k_compute_flux: 1D stencil with transmissive BC, Roe + Harten entropy fix'),
        FluxK, c_blank,
        c_comment('k_update_conservative: elementwise U -= dt_dx * (F[i+1] - F[i])'),
        UpdateK, c_blank,
        c_comment('k_compute_primitives: cons (rho, rho*u, E) -> prim (rho, u, p)'),
        PrimK, c_blank,
        c_comment('k_cfl_condition: max wavespeed reduction (uses block_reduce_max)'),
        CflK, c_blank,
        c_comment('=== C API wrappers + GPU memory management ==='),
        c_extern_c_open,
        c_blank,
        FluxW, c_blank,
        UpdateW, c_blank,
        PrimW, c_blank,
        CflW, c_blank,
        MA, c_blank, MF, c_blank, MH, c_blank, MD, c_blank, MS,
        c_blank,
        c_extern_c_close
    ],
    emit_program(Program, Code).

test :-
    generate_all(Code),
    write(Code).

:- initialization((test -> halt(0) ; (write('GENERATION FAILED'), nl, halt(1)))).
