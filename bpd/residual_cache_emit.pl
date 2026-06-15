%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
%% residual_cache_emit.pl — emit the C/CUDA for residual_cache (KV-Direct) recompute, composed from
%% the existing verified llama kernels (rms_norm + vecmat). Targets the same __global__ CUDA syntax our
%% runner uses and that drops into ggml/llama.cpp's custom-op world (useful to hermes-final on Ollama).
%%
%% KV-Direct read operation: instead of reading cached K/V, recompute from the cached per-token RESIDUAL:
%%   normed = rms_norm(residual, attn_norm_weight)
%%   K      = W_k @ normed         (and V = W_v @ normed)
%% The residual is deterministic of the token, so the recompute is exact (full precision) / bounded
%% (quantized). This emits the recompute as a host-callable composition of the two verified kernels.

:- use_module('lib/c_ast').
:- use_module('lib/kernel_templates_llama').
:- initialization(main).

%% residual_cache_recompute_wrapper(-Wrapper): a host C function that, given the cached residual + the
%% attn_norm weight + the projection weight, launches rms_norm then vecmat to recompute K (or V).
%% This is the KV-Direct read: the orchestration that the existing kernels plug into.
residual_cache_recompute_wrapper(Wrapper) :-
    Wrapper = c_func([], c_type(void), kv_direct_recompute,
        [param(c_type(const_restrict_ptr(c_type(float))), residual),      %% cached per-token residual [embd]
         param(c_type(const_restrict_ptr(c_type(float))), attn_norm_w),   %% rms_norm weight [embd]
         param(c_type(const_restrict_ptr(c_type(float))), proj_w),        %% W_k or W_v [embd x out]
         param(c_type(restrict_ptr(c_type(float))), kv_out),              %% recomputed K or V [out]
         param(c_type(restrict_ptr(c_type(float))), scratch_normed),      %% scratch [embd]
         param(c_type(int), embd),
         param(c_type(int), out_dim),
         param(c_type(float), eps)],
        [c_comment('KV-Direct recompute: normed = rms_norm(residual); kv = proj_w @ normed'),
         c_comment('Step 1: rms_norm(residual) -> scratch_normed  [1 block, 256 threads]'),
         c_cuda_launch(k_rms_norm, c_int(1), c_int(256),
            [c_var(residual), c_var(attn_norm_w), c_var(scratch_normed), c_var(embd), c_var(eps)]),
         c_comment('Step 2: kv = proj_w @ normed  [out_dim cols blocked, k_dim=embd in shared]'),
         c_decl_init(c_type(int), grid, c_binop(/, c_binop(+, c_var(out_dim), c_int(255)), c_int(256))),
         c_decl_init(c_type(int), shmem, c_binop(*, c_var(embd), c_call(sizeof, [c_var(float)]))),
         c_comment('vecmat with dynamic shared memory: k_vecmat<<<grid, 256, shmem>>>(...)'),
         c_cuda_launch(k_vecmat, c_var(grid), c_int(256), c_var(shmem),
            [c_var(scratch_normed), c_var(proj_w), c_var(kv_out), c_var(embd), c_var(out_dim)])]).

main :-
    %% emit the component kernels (already verified) + the recompute orchestration wrapper.
    %% dependency helpers (k_rms_norm calls block_reduce_sum -> warp_reduce_sum) — emit first.
    warp_reduce_sum_helper(H1), emit_c(H1, SH1),
    block_reduce_sum_helper(H2), emit_c(H2, SH2),
    rms_norm_kernel(K1), emit_c(K1, S1),
    vecmat_kernel(K2), emit_c(K2, S2),
    residual_cache_recompute_wrapper(W), emit_c(W, S3),
    Header = '// ===================================================================\n// residual_cache (KV-Direct) — generated C/CUDA for ggml/llama.cpp\n// Recompute K/V from the cached residual instead of caching K/V (13-27x less KV cache).\n// SPDX-License-Identifier: LicenseRef-RTAAL-1.1\n// ===================================================================\n\n',
    atomic_list_concat([Header, SH1, '\n\n', SH2, '\n\n', S1, '\n\n', S2, '\n\n', S3, '\n'], Out),
    ( getenv('RC_OUT', File) -> true ; File = '/tmp/residual_cache.cu' ),
    open(File, write, St), write(St, Out), close(St),
    write(Out), nl,
    halt.
