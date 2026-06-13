%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% generate_blas_kernels.pl — BLAS CUDA kernel generation.
%%
%% First instance of NVIDIA Tech Level subsumption per Heath's
%% 2026-05-18 ~16:30 UTC framing. Generates a complete .cu file
%% from kernel_templates_blas.pl facts, ready for nvcc compilation
%% and bit-identical verification against cuBLAS sgemv on Tesla P4.
%%
%% PER HEATH'S DIRECTIVE (2026-05-18 ~22:27 UTC):
%%
%%   "Never call into the other code, always learn it so deeply that
%%    we subsume it. The substrate generates the kernel. The kernel
%%    matches cuBLAS. That's the claim."
%%
%% The substrate IS the source of truth. Hand-writing CUDA would be
%% substrate-bypass. Every byte of the .cu output below is derived
%% from c_ast terms via emit_program/2.
%%
%% USAGE (single-shot generation):
%%
%%   swipl bpd/generate_blas_kernels.pl > bpd/build/blas_kernels.cu
%%
%% The initialization/main goal at the bottom emits the .cu source
%% to stdout, then halts. Pipe to a file or capture for compilation.
%%
%% NVCC COMPILATION:
%%
%%   nvcc -arch=sm_61 -O2 -shared -Xcompiler -fPIC \
%%        -o bpd/build/blas_kernels.so bpd/build/blas_kernels.cu
%%
%% PYTHON CTYPES LOAD (for harness):
%%
%%   lib = ctypes.CDLL('./bpd/build/blas_kernels.so')
%%   lib.gpu_alloc.restype = ctypes.c_void_p
%%   lib.gpu_alloc.argtypes = [ctypes.c_int]
%%   lib.gpu_sgemv_cublas_match.argtypes = [
%%       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
%%       ctypes.c_int, ctypes.c_int
%%   ]
%%   ...
%%
%% MATCHES the same conventions as generate_cfd_kernels.pl (commit
%% 12224c4ba) and generate_llama_kernels.pl. Shared GPU memory
%% management interface (gpu_alloc / gpu_free / gpu_h2d / gpu_d2h /
%% gpu_sync) means any harness that already knows how to call CFD or
%% ML kernels can call BLAS kernels with no infrastructure change.

:- use_module('lib/c_ast').
:- use_module('lib/kernel_templates_blas').


%% ═════════════════════════════════════════════════════════════════════════════
%% GPU MEMORY MANAGEMENT (shared with generate_cfd_kernels.pl, _llama_)
%% ═════════════════════════════════════════════════════════════════════════════

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
    %% BLAS kernel emits from kernel_templates_blas.pl:
    sgemv_kernel_substrate_native(k_sgemv_substrate_native, NativeK),
    sgemv_wrapper_substrate_native(k_sgemv_substrate_native, NativeW),
    sgemv_kernel_cublas_match(k_sgemv_cublas_match, CublasK),
    sgemv_wrapper_cublas_match(k_sgemv_cublas_match, CublasW),

    %% Memory management functions
    mem_alloc(MA), mem_free(MF), mem_h2d(MH), mem_d2h(MD), mem_sync(MS),

    Program = [
        c_include_sys('cuda_runtime.h'),
        c_include_sys('math.h'),
        c_blank,
        c_comment('BPD-generated BLAS kernels for NVIDIA Tech Level subsumption.'),
        c_comment('100% AST-generated via Prolog c_ast DCG emitter.'),
        c_comment('ZERO write/1 strings. Every byte derived from c_ast terms.'),
        c_comment(''),
        c_comment('Per Heath 2026-05-18: "Never call into the other code, always learn'),
        c_comment('it so deeply that we subsume it. The substrate generates the kernel.'),
        c_comment('The kernel matches cuBLAS. That''s the claim."'),
        c_comment(''),
        c_comment('Two configurations:'),
        c_comment('  - substrate_native: warp-shuffle, 32 t/row, simple stride'),
        c_comment('  - cublas_match: 32 t/row x 4 rows/block (128 threads/block),'),
        c_comment('    shared-mem x preload, stride-32 accumulation, warp-shuffle'),
        c_comment('    reduction. Same per-thread arithmetic as substrate-native;'),
        c_comment('    structural changes (shared-mem x, multi-row blocks) are'),
        c_comment('    tick-determining per mavchin 2026-05-18 ~22:36 UTC sweep.'),
        c_comment('    The 12 ULP residual gap is compilation-level, not algorithm-level.'),
        c_blank,
        c_comment('=== __global__ kernels ==='),
        c_blank,
        c_comment('Substrate-native: the warp-shuffle baseline (12 ULP gap vs cuBLAS)'),
        NativeK, c_blank,
        c_comment('cuBLAS-match: bit-identical with cuBLAS sgemv on sm_61 (claim)'),
        CublasK, c_blank,
        c_comment('=== C API wrappers + GPU memory management ==='),
        c_raw('extern "C" {'),
        c_blank,
        NativeW, c_blank,
        CublasW, c_blank,
        MA, c_blank, MF, c_blank, MH, c_blank, MD, c_blank, MS,
        c_blank,
        c_raw('} // extern "C"')
    ],
    emit_program(Program, Code).

test :-
    generate_all(Code),
    write(Code).

:- initialization((test -> halt(0) ; (write('GENERATION FAILED'), nl, halt(1)))).
