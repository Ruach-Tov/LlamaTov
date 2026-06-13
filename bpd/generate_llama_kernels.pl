%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% generate_llama_kernels.pl — Thin orchestrator for llama-family CUDA generation.
%%
%% ZERO write/1. Every line from AST terms through emit_program/2.
%% The computation AND infrastructure are both AST-generated.
%%
%% Post-refactor 2026-05-17: kernel templates + activation_expr facts
%% live in lib/kernel_templates_llama.pl. This script now:
%%   - Uses that module
%%   - Defines memory-management functions (gpu_alloc, gpu_free, etc.)
%%   - Assembles the full Program list
%%   - Fires generate_all via :- initialization
%%
%% The split makes the kernel knowledge reusable from other callers
%% (e.g., kernel_emit_bridge.py via consult-as-module) without
%% triggering generation. Per mavchin's authorization (intercom 10:31 UTC)
%% with Heath's confirmation.

:- use_module('lib/c_ast').
:- use_module('lib/kernel_templates_llama').
%% kernel_templates_blas needed for elem_kernel/2 (k_scale), gpu_scale_wrapper/1,
%% gpu_copy_d2d_wrapper/1. Exclude kernel_available_fixes/2 and fix_description/2
%% — those name-clash with kernel_templates_llama's exports of the same predicate
%% names. The clash is harmless (both export the same fix-flag metadata vocabulary)
%% but Prolog requires explicit disambiguation.
:- use_module('lib/kernel_templates_blas', except([kernel_available_fixes/2, fix_description/2])).


%% ═══════════════════════════════════════════════════════════════════
%% GPU MEMORY MANAGEMENT
%% ═══════════════════════════════════════════════════════════════════

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
    F = c_func(c_type(void), gpu_copy_h2d,
        [param(c_type(ptr(c_type(void))), d),
         param(c_type(const_ptr(c_type(void))), h),
         param(c_type(int), b)],
        [c_expr_stmt(c_call(cudaMemcpy, [c_var(d), c_var(h), c_var(b), c_var(cudaMemcpyHostToDevice)]))]).

mem_d2h(F) :-
    F = c_func(c_type(void), gpu_copy_d2h,
        [param(c_type(ptr(c_type(void))), h),
         param(c_type(const_ptr(c_type(void))), d),
         param(c_type(int), b)],
        [c_expr_stmt(c_call(cudaMemcpy, [c_var(h), c_var(d), c_var(b), c_var(cudaMemcpyDeviceToHost)]))]).

mem_sync(F) :-
    F = c_func(c_type(void), gpu_sync, [],
        [c_expr_stmt(c_call(cudaDeviceSynchronize, []))]).


%% ═══════════════════════════════════════════════════════════════════
%% FULL PROGRAM ASSEMBLY
%% ═══════════════════════════════════════════════════════════════════

generate_all(Code) :-
    vecmat_kernel(VK), vecmat_wrapper(VW),
    %% warp_reduce_sum_helper must precede rms_norm_kernel and softmax_kernel;
    %% both use the warp_reduce_sum + cross-warp __shared__ pattern from
    %% llama.cpp. __device__ functions must be declared before their first
    %% use. Per mavchin's 2026-05-18 diagnosis: these reduction patterns
    %% are necessary for bit-identical output with Ollama.
    warp_reduce_sum_helper(WRSH),
    %% warp_reduce_max_helper used by softmax_kernel. Sibling of
    %% warp_reduce_sum (same warp-shuffle reduction, fmaxf instead of +).
    warp_reduce_max_helper(WRMH),
    %% block_reduce_sum/max helpers — full-block cross-warp reduction.
    %% Per Heath's 2026-05-18 ~07:35 UTC: the substrate-honest form of
    %% the block reduction is a callable function (matching llama.cpp's
    %% block_reduce<Op, block_size>(val, shared) template in
    %% ggml-cuda/common.cuh:594), not statement-splicing into each
    %% kernel. Internally these helpers compute warp_id/lane_id as
    %% named const locals — stable names so a future optimizer can
    %% find them by AST pattern-match.
    %% Must precede rms_norm_kernel and softmax_kernel which call them.
    block_reduce_sum_helper(BRSH),
    block_reduce_max_helper(BRMH),
    rms_norm_kernel(RK), rms_norm_wrapper(RW),
    %% softmax_kernel emits the numerically-stable row-wise softmax matching
    %% llama.cpp's soft_max_f32 algorithmic core (no mask/sinks/ALiBi —
    %% sufficient for llama3.2:1b decode-time attention). Per mavchin's
    %% 2026-05-18 ~05:31 UTC: softmax is the second-biggest divergence
    %% source after RMSNorm for the bit-identical Ollama target.
    %% Pass an explicit empty Fixes list to make the bug-for-bug compatibility
    %% position visible at the call site. Per 2026-05-18 ~08:00 UTC substrate-
    %% honesty design: the default substrate-emitted kernel is byte-identical
    %% with the subsumed software (llama.cpp soft_max_f32) INCLUDING any known
    %% defects. Named fixes (kernel_available_fixes/2) opt-in specific repairs.
    %% Switch to softmax_kernel([fix_softmax_phase_inter_race], SoftmaxK) if
    %% the theoretical race-safety is desired in production output.
    softmax_kernel([], SoftmaxK), softmax_wrapper(SoftmaxW),
    binary_elem_kernel(k_add, +, AddK), binary_wrapper(k_add, gpu_add, AddW),
    binary_elem_kernel(k_mul, *, MulK), binary_wrapper(k_mul, gpu_mul, MulW),
    %% silu now uses declarative unary_activation_kernel to match llama.cpp's
    %% exact "1.0f" literal (the hand-written silu_kernel emitted "1.0" which
    %% is a double promoted, potentially ULP-divergent). Per mavchin's
    %% 2026-05-18 diagnosis of CPU-vs-GPU silu divergence in the bit-identical
    %% pursuit. Hand-written silu_kernel/silu_wrapper retained in the substrate
    %% for historical reference but no longer used in the kernel program.
    unary_activation_kernel(k_silu, SiluK), unary_activation_wrapper(k_silu, gpu_silu, SiluW),
    %% Generate ALL activation kernels from BPD facts
    unary_activation_kernel(k_sigmoid, SigmoidK), unary_activation_wrapper(k_sigmoid, gpu_sigmoid, SigmoidW),
    unary_activation_kernel(k_relu, ReluK), unary_activation_wrapper(k_relu, gpu_relu, ReluW),
    %% Production gpu_gelu uses the tanh form — what Ollama / llama.cpp's
    %% graph builder actually runs (see lib/terminology.pl:
    %% terminology_change(ollama_runtime:k_gelu, world:k_gelu_tanh)).
    %% Per the 2026-05-17 investigation: the in-tree k_gelu was the
    %% erf form, which doesn't match production. Switching the
    %% production wrapper to k_gelu_tanh aligns LlamaTov's inference
    %% with Ollama's per-token output at this op.
    unary_activation_kernel(k_gelu_tanh, GeluK), unary_activation_wrapper(k_gelu_tanh, gpu_gelu, GeluW),
    unary_activation_kernel(k_tanh, TanhK), unary_activation_wrapper(k_tanh, gpu_tanh, TanhW),
    %% Per medayek's driver-symbol parity 2026-05-19 ~07:00 UTC:
    %% wire the additional forward-pass kernels into the Program for
    %% file-level swap with bpd/llamatov_kernels.cu. Each variable
    %% binds via the existing kernel_templates_llama/blas predicates.
    embed_kernel(EmbedK), embed_wrapper(EmbedW),
    causal_mask_kernel(CMaskK), causal_mask_wrapper(CMaskW),
    matmul_kernel(MatmulK), matmul_wrapper(MatmulW),
    matmul_opt_wrapper(MatmulOptW),
    rope_kernel(RopeKern), rope_wrapper(RopeQW),
    rope_k_wrapper(RopeKW),
    layer_norm_kernel(LayerNormK), layer_norm_wrapper(LayerNormW),
    %% k_scale is an elem_op fact; emit via elem_kernel/2 factory.
    elem_kernel(k_scale, ScaleK), gpu_scale_wrapper(ScaleW),
    %% gpu_copy_d2d is wrapper-only (no kernel); wraps cudaMemcpy.
    gpu_copy_d2d_wrapper(CopyD2DW),
    mem_alloc(MA), mem_free(MF), mem_h2d(MH), mem_d2h(MD), mem_sync(MS),

    Program = [
        c_include_sys('cuda_runtime.h'),
        c_include_sys('math.h'),
        c_blank,
        c_comment('BPD-generated CUDA kernels for LlamaTov inference'),
        c_comment('100% AST-generated via Prolog c_ast DCG emitter'),
        c_comment('ZERO write/1 strings. Every line from AST terms.'),
        c_comment('Activations generated from BPD activation_expr/3 facts.'),
        c_blank,
        %% Kernels (no extern "C" needed for __global__)
        VK, c_blank,
        %% __device__ helpers must precede kernels that use them.
        %% Order: warp_reduce_sum (used by rms_norm and softmax) then
        %% warp_reduce_max (used by softmax only).
        WRSH, c_blank, WRMH, c_blank,
        %% block_reduce_sum/max are __device__ helpers that compose warp_reduce
        %% with cross-warp shared mem. rms_norm and softmax call these instead
        %% of inlining the reduction pattern.
        BRSH, c_blank, BRMH, c_blank,
        RK, c_blank,
        %% softmax_kernel calls block_reduce_max + block_reduce_sum
        SoftmaxK, c_blank,
        AddK, c_blank, MulK, c_blank,
        SiluK, c_blank, SigmoidK, c_blank, ReluK, c_blank, GeluK, c_blank, TanhK,
        c_blank,
        %% Forward-pass kernels for full llama-class inference.
        %% Order: embed (first in fwd pass), causal_mask, matmul,
        %% rope, layer_norm, scale. None of these have __device__
        %% dependencies; placement is for readability.
        EmbedK, c_blank,
        CMaskK, c_blank,
        MatmulK, c_blank,
        RopeKern, c_blank,
        LayerNormK, c_blank,
        ScaleK, c_blank,
        c_comment('C API wrappers + GPU memory management'),
        c_raw('extern "C" {'),
        c_blank,
        VW, c_blank, RW, c_blank, SoftmaxW, c_blank,
        AddW, c_blank, MulW, c_blank,
        SiluW, c_blank, SigmoidW, c_blank, ReluW, c_blank, GeluW, c_blank, TanhW,
        c_blank,
        %% Forward-pass wrappers (matching driver-symbol parity per
        %% medayek's swap discipline 2026-05-19 ~07:00 UTC).
        EmbedW, c_blank,
        CMaskW, c_blank,
        MatmulW, c_blank,
        MatmulOptW, c_blank,
        RopeQW, c_blank,
        RopeKW, c_blank,
        LayerNormW, c_blank,
        ScaleW, c_blank,
        CopyD2DW, c_blank,
        MA, c_blank, MF, c_blank, MH, c_blank, MD, c_blank, MS,
        c_blank,
        c_raw('} // extern "C"')
    ],
    emit_program(Program, Code).

test :-
    generate_all(Code),
    write(Code).

:- initialization((test -> halt(0) ; (write('GENERATION FAILED'), nl, halt(1)))).
