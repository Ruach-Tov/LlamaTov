%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% kernel_templates_llama.pl — BPD kernel knowledge for llama-family inference.
%%
%% Authoritative source of kernel templates and activation facts that
%% generate_llama_kernels.pl (the script) assembles into the full
%% llamatov_inference.cu output.
%%
%% Pattern matches lib/kernel_templates.pl: module providing per-family
%% generators; root-level script calls them to produce concrete output.
%%
%% Per mavchin's authorization (intercom 10:31 UTC, with Heath's vote of
%% confirmation): extracted from generate_llama_kernels.pl. The script
%% retains :- initialization, generate_all/1, and the Program list
%% assembly. The module owns the knowledge: kernel templates +
%% activation_expr/3 facts.
%%
%% Substrate-honest motivation: generate_llama_kernels.pl had a
%% :- initialization((test -> halt(0) ; ...)) directive that fired
%% generate_all whenever the file was consulted. This prevented
%% callers (like kernel_emit_bridge.py) from reaching individual
%% kernel templates via consult-and-call. Refactoring to a module
%% means: the script is the only entry point that fires generation;
%% the module is reusable substrate.
%%
%% AUTHORITATIVE BPD KNOWLEDGE captured here:
%%
%%   Activation expressions (5 facts):
%%     activation_expr(k_silu, X, Expr)     — x / (1 + exp(-x))
%%     activation_expr(k_sigmoid, X, Expr)  — 1 / (1 + exp(-x))
%%     activation_expr(k_relu, X, Expr)     — fmaxf(0, x)
%%     activation_expr(k_gelu, X, Expr)     — 0.5*x*(1 + erf(x*0.707))
%%     activation_expr(k_tanh, X, Expr)     — tanhf(x)
%%
%%   Kernel templates:
%%     vecmat_kernel/1            — M=1 decode path matmul
%%     vecmat_wrapper/1
%%     rms_norm_kernel/1
%%     rms_norm_wrapper/1
%%     binary_elem_kernel/3       — parameterized by op (k_add: +, k_mul: *)
%%     binary_wrapper/3
%%     unary_activation_kernel/2  — generic; reads activation_expr/3 fact
%%     unary_activation_wrapper/3
%%     silu_kernel/1              — original hand-written form (kept for
%%     silu_wrapper/1               historical bit-equivalence with the
%%                                  pre-BPD output)
%%
%% Author: metayen 2026-05-17
%% Per mavchin's authorization. Refactor of generate_llama_kernels.pl
%% to enable cross-language matrix harness access to activation facts.

:- module(kernel_templates_llama, [
    %% Kernel template predicates
    vecmat_kernel/1,
    vecmat_wrapper/1,
    rms_norm_kernel/1,
    rms_norm_wrapper/1,
    warp_reduce_sum_helper/1,    % __device__ helper, sum reduction within a warp
    warp_reduce_max_helper/1,    % __device__ helper, max reduction within a warp
    block_reduce_sum_helper/1,   % __device__ helper, full-block sum reduction (calls warp_reduce_sum)
    block_reduce_max_helper/1,   % __device__ helper, full-block max reduction (calls warp_reduce_max)
    softmax_kernel/1,            % bug-for-bug compatible softmax (Fixes=[])
    softmax_kernel/2,            % +Fixes list-of-named-fixes, -Kernel AST
    softmax_wrapper/1,           % gpu_softmax C API wrapper
    %% Fix-flag metadata (per 2026-05-18 ~08:00 UTC substrate-honesty design):
    %% the substrate exposes named, individually-toggleable defect-repairs
    %% as first-class harness-discoverable predicates. The default emit is
    %% bug-for-bug compatible with the subsumed software; fixes are
    %% authored acts of substrate authority that disable specific defects.
    kernel_available_fixes/2,    % +KernelPred, -FixList for harness discovery
    fix_description/2,           % +FixAtom, -Description for harness reports
    binary_elem_kernel/3,
    binary_wrapper/3,
    unary_activation_kernel/2,
    unary_activation_wrapper/3,
    silu_kernel/1,
    silu_wrapper/1,
    %% RoPE — Rotary Position Embedding (half-split form)
    rope_kernel/1,
    rope_wrapper/1,
    rope_k_wrapper/1,
    %% Token embedding lookup — gather from table
    embed_kernel/1,
    embed_wrapper/1,
    %% Causal mask — set upper triangle of attention matrix to -inf
    causal_mask_kernel/1,
    causal_mask_wrapper/1,
    %% General matmul (M×K @ K×N → M×N) with tiled shared memory
    matmul_kernel/1,
    matmul_kernel/2,
    matmul_wrapper/1,
    matmul_wrapper/2,
    matmul_opt_wrapper/1,
    matmul_opt_wrapper/2,
    %% LayerNorm — mean+variance normalization with bias (BERT-class)
    layer_norm_kernel/1,
    layer_norm_wrapper/1,
    %% BPD activation facts (exported so callers can introspect)
    activation_expr/3
]).

:- use_module(c_ast).


%% ═══════════════════════════════════════════════════════════════════
%% VECMAT KERNEL (M=1 decode path, matches cuBLAS on FFN shapes)
%% ═══════════════════════════════════════════════════════════════════

vecmat_kernel(Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_vecmat,
        [param(c_type(const_restrict_ptr(c_type(float))), a),
         param(c_type(const_restrict_ptr(c_type(float))), b),
         param(c_type(restrict_ptr(c_type(float))), c),
         param(c_type(int), k_dim),
         param(c_type(int), n_dim)],
        [%% Shared memory for input vector
         c_extern_shared(c_type(float), sA),
         %% Cooperative load of A into shared memory
         c_for(c_decl_init(c_type(int), i, c_member(c_var(threadIdx), x)),
               c_binop(<, c_var(i), c_var(k_dim)),
               c_compound_step(c_var(i), '+=', c_member(c_var(blockDim), x)),
               [c_assign(c_index(c_var(sA), c_var(i)),
                          c_index(c_var(a), c_var(i)))]),
         c_syncthreads,
         c_blank,
         %% Thread computes one output column
         c_decl_init(c_type(int), col,
             c_binop(+, c_binop(*, c_member(c_var(blockIdx), x),
                                    c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_if(c_binop(>=, c_var(col), c_var(n_dim)), [c_return_void]),
         c_blank,
         %% Dot product
         c_decl_init(c_type(float), sum, c_float(0.0)),
         c_for(c_decl_init(c_type(int), k, c_int(0)),
               c_binop(<, c_var(k), c_var(k_dim)),
               c_postfix(++, c_var(k)),
               [c_assign(c_var(sum),
                   c_binop(+, c_var(sum),
                       c_binop(*,
                           c_index(c_var(sA), c_var(k)),
                           c_index(c_var(b),
                               c_binop(+, c_binop(*, c_var(k), c_var(n_dim)),
                                           c_var(col))))))]),
         c_blank,
         c_assign(c_index(c_var(c), c_var(col)), c_var(sum))
        ]).

vecmat_wrapper(W) :-
    W = c_func(c_type(void), gpu_vecmat,
        [param(c_type(const_restrict_ptr(c_type(float))), a),
         param(c_type(const_restrict_ptr(c_type(float))), b),
         param(c_type(restrict_ptr(c_type(float))), c),
         param(c_type(int), k),
         param(c_type(int), n)],
        [c_cuda_launch(k_vecmat,
            c_binop(/, c_paren(c_binop(+, c_var(n), c_int(255))), c_int(256)),
            c_int(256),
            c_binop(*, c_var(k), c_sizeof(c_type(float))),
            [c_var(a), c_var(b), c_var(c), c_var(k), c_var(n)])]).


%% ═══════════════════════════════════════════════════════════════════
%% RMS NORM KERNEL
%% ═══════════════════════════════════════════════════════════════════

%% warp_reduce_sum_helper(-Helper)
%%
%% Emits a __device__ __forceinline__ helper function that performs
%% a warp-level sum reduction via __shfl_xor_sync. Matches the pattern
%% in llama.cpp's ggml-cuda/common.cuh exactly.
%%
%% NOTE: A prior version of this helper used c_raw with text "Verbatim
%% is correct and honest for CUDA hardware-level intrinsics." That
%% framing was substrate-rationalization. The substrate CAN comprehend
%% __device__ qualifiers (via c_func qualifier list), __shfl_xor_sync
%% (via c_call with any function name including double-underscores),
%% __shared__ declarations (via c_shared_decl), and #pragma directives
%% (via c_pragma). The c_raw escape hatch was unnecessary debt that
%% the substrate has now paid down by expressing the helper structurally.
%%
%% This helper must be emitted ONCE per kernel file, ABOVE any kernel
%% that uses it (rms_norm in particular). generate_llama_kernels.pl
%% places it before rms_norm_kernel's output.
%% Per 2026-05-18 ~06:50 UTC substrate-honesty repair: replaced c_raw text
%% with structural c_ast. The substrate now comprehends every element of
%% this helper — qualifiers, pragma, for-loop, compound assignment, the
%% __shfl_xor_sync intrinsic (via c_call), and the return. Byte-identical
%% output to the previous c_raw form (verified empirically).
%%
%% Note c_compound_assign('+=', ...) for "x +=" rather than
%% c_assign(c_var(x), c_binop('+', ...)) which would emit "x = x +".
%% Both have identical C semantics; the compound form matches llama.cpp's
%% source text byte-for-byte.
warp_reduce_sum_helper(Helper) :-
    Helper = c_func(['static', '__device__', '__forceinline__'],
        c_type(float), warp_reduce_sum,
        [param(c_type(float), x)],
        [c_pragma('unroll'),
         c_for(c_decl_init(c_type(int), offset, c_int(16)),
               c_binop('>', c_var(offset), c_int(0)),
               c_compound_step(c_var(offset), '>>=', c_int(1)),
               [c_compound_assign('+=', c_var(x),
                   c_call('__shfl_xor_sync',
                       [c_hex(0xffffffff), c_var(x), c_var(offset), c_int(32)]))]),
         c_return(c_var(x))]).


%% warp_reduce_max_helper(-Helper)
%%
%% Sibling of warp_reduce_sum_helper. Emits a __device__ __forceinline__
%% helper that performs a warp-level max reduction via __shfl_xor_sync.
%% Matches the pattern in llama.cpp's ggml-cuda/common.cuh (line 481).
%%
%% Used by softmax_kernel and any future kernel that needs warp-level max.
%% Like warp_reduce_sum_helper, this was originally shipped as c_raw with
%% the "verbatim is correct and honest" framing — that was rationalization.
%% The substrate has the primitives to express this structurally.
%%
%% Must be emitted ONCE per kernel file, ABOVE any kernel that uses it.
%% Per 2026-05-18 ~06:50 UTC substrate-honesty repair: replaced c_raw text
%% with structural c_ast. Same comprehension argument as warp_reduce_sum_helper.
%% The fmaxf version uses c_assign(c_var(x), c_call(fmaxf, ...)) since there is
%% no compound assignment for max (no x maxeq y operator in C).
warp_reduce_max_helper(Helper) :-
    Helper = c_func(['static', '__device__', '__forceinline__'],
        c_type(float), warp_reduce_max,
        [param(c_type(float), x)],
        [c_pragma('unroll'),
         c_for(c_decl_init(c_type(int), offset, c_int(16)),
               c_binop('>', c_var(offset), c_int(0)),
               c_compound_step(c_var(offset), '>>=', c_int(1)),
               [c_assign(c_var(x),
                   c_call(fmaxf, [c_var(x),
                       c_call('__shfl_xor_sync',
                           [c_hex(0xffffffff), c_var(x), c_var(offset), c_int(32)])]))]),
         c_return(c_var(x))]).


%% block_reduce_sum_helper(-Helper)
%%
%% Full-block sum reduction. Hardcoded for block_size=256 (8 warps).
%% Per llama.cpp's block_reduce template (ggml-cuda/common.cuh:594),
%% specialized to (Op=SUM, block_size=256, T=float).
%%
%% Signature: static __device__ __forceinline__ float block_reduce_sum(float val, float *buf);
%%   - val: per-thread input value
%%   - buf: caller's shared buffer (caller declares __shared__ float buf[8];)
%%   - returns: the sum reduced across all 256 threads in the block
%%
%% Implementation (structural c_ast, with warp_id/lane_id as named const
%% locals — stable names so a future optimizer / warp scheduler can find
%% them by AST pattern-match):
%%   1. const int warp_id = threadIdx.x / 32;
%%   2. const int lane_id = threadIdx.x % 32;
%%   3. val = warp_reduce_sum(val);    -- per-warp reduction
%%   4. if (lane_id == 0) buf[warp_id] = val;
%%   5. __syncthreads();
%%   6. val = (lane_id < 8) ? buf[lane_id] : 0.0f;  -- gather across warps
%%   7. val = warp_reduce_sum(val);    -- final reduction (first warp only effective)
%%   8. return val;
%%
%% Per Heath's 2026-05-18 ~07:35 UTC framing: when the substrate-honest
%% form of an emit unit is a callable function (matching llama.cpp's
%% structural choice), emit it as a __device__ function rather than
%% splicing inline statements. The optimizer can later walk the
%% c_func body and reason about warp_id/lane_id locals via AST
%% pattern-match.
%%
%% Depends on warp_reduce_sum (emitted by warp_reduce_sum_helper/1).
block_reduce_sum_helper(Helper) :-
    Helper = c_func(['static', '__device__', '__forceinline__'],
        c_type(float), block_reduce_sum,
        [param(c_type(float), val),
         param(c_type(ptr(c_type(float))), buf)],
        [c_decl_init(c_type(const(c_type(int))), warp_id,
             c_binop('/', c_member(c_var(threadIdx), x), c_int(32))),
         c_decl_init(c_type(const(c_type(int))), lane_id,
             c_binop('%', c_member(c_var(threadIdx), x), c_int(32))),
         c_assign(c_var(val), c_call(warp_reduce_sum, [c_var(val)])),
         c_if(c_binop('==', c_var(lane_id), c_int(0)),
              [c_assign(c_index(c_var(buf), c_var(warp_id)), c_var(val))]),
         c_syncthreads,
         c_assign(c_var(val),
             c_ternary(c_binop('<', c_var(lane_id), c_int(8)),
                 c_index(c_var(buf), c_var(lane_id)),
                 c_float_f(0.0))),
         c_assign(c_var(val), c_call(warp_reduce_sum, [c_var(val)])),
         c_return(c_var(val))]).


%% block_reduce_max_helper(-Helper)
%%
%% Sibling of block_reduce_sum_helper. Full-block max reduction.
%% Init value is -INFINITY (the identity element for max).
%% All other structure identical to block_reduce_sum.
%%
%% Signature: static __device__ __forceinline__ float block_reduce_max(float val, float *buf);
%%
%% Depends on warp_reduce_max (emitted by warp_reduce_max_helper/1).
block_reduce_max_helper(Helper) :-
    Helper = c_func(['static', '__device__', '__forceinline__'],
        c_type(float), block_reduce_max,
        [param(c_type(float), val),
         param(c_type(ptr(c_type(float))), buf)],
        [c_decl_init(c_type(const(c_type(int))), warp_id,
             c_binop('/', c_member(c_var(threadIdx), x), c_int(32))),
         c_decl_init(c_type(const(c_type(int))), lane_id,
             c_binop('%', c_member(c_var(threadIdx), x), c_int(32))),
         c_assign(c_var(val), c_call(warp_reduce_max, [c_var(val)])),
         c_if(c_binop('==', c_var(lane_id), c_int(0)),
              [c_assign(c_index(c_var(buf), c_var(warp_id)), c_var(val))]),
         c_syncthreads,
         c_assign(c_var(val),
             c_ternary(c_binop('<', c_var(lane_id), c_int(8)),
                 c_index(c_var(buf), c_var(lane_id)),
                 c_var('-INFINITY'))),
         c_assign(c_var(val), c_call(warp_reduce_max, [c_var(val)])),
         c_return(c_var(val))]).


%% softmax_kernel(-Kernel)
%%
%% Emits a numerically-stable row-wise softmax kernel matching llama.cpp's
%% soft_max_f32 algorithmic core for the simple decode-time case (no
%% mask, no sinks, no ALiBi slope, no template specialization).
%%
%% The kernel computes:
%%   for each row r:
%%     max_val = max(x[r, :])
%%     for j: y[r, j] = exp(x[r, j] - max_val)
%%     sum = sum(y[r, :])
%%     for j: y[r, j] = y[r, j] / sum
%%
%% IMPLEMENTATION: 256 threads per block. Three-phase block reduction:
%%   Phase 1: strided max + warp_reduce_max + cross-warp + warp_reduce_max
%%   Phase 2: strided exp(x - max), sum + same cross-warp pattern
%%   Phase 3: strided normalize
%%
%% This matches llama.cpp's block_reduce<block_reduce_method::MAX> and
%% block_reduce<SUM> templates for block_size=256, T=float, with mask=NULL
%% and sinks=NULL. The reduction order is identical (warp_reduce_max/sum
%% via __shfl_xor_sync, cross-warp via __shared__).
%%
%% Per mavchin's 2026-05-18 ~05:31 UTC guidance: softmax is the second-
%% biggest divergence source after RMSNorm for the bit-identical Ollama
%% target. CPU PyTorch softmax differs from GPU softmax in reduction
%% order; this kernel produces the same accumulation tree as llama.cpp.
%%
%% Depends on warp_reduce_max (emitted by warp_reduce_max_helper/1) AND
%% warp_reduce_sum (emitted by warp_reduce_sum_helper/1). Both helpers
%% MUST appear above this kernel in the file.
%% Per 2026-05-18 ~07:20 UTC subtask 3: replaced c_raw body with structural
%% c_ast. Same pattern as rms_norm (commit 864d10058).
%%
%% Per 2026-05-18 ~08:00 UTC subtask 4B.g: introduces the fix-flag pattern.
%% The arity-1 form softmax_kernel/1 is preserved as a thin wrapper that
%% calls softmax_kernel/2 with an empty fix list — the bug-for-bug-compatible
%% default that exactly emulates llama.cpp's soft_max_f32 source structure.
%% The arity-2 form accepts a list of named fixes that disable specific
%% known defects in the subsumed software.
%%
%% Available fixes (per kernel_available_fixes/2):
%%   fix_softmax_phase_inter_race
%%     Adds a __syncthreads() between Phase 1 and Phase 2 of the block
%%     reduction. Defends against the theoretical race where slow warps
%%     in Phase 1's gather read could overlap with fast warps' Phase 2
%%     writes to the shared buffer. llama.cpp does not include this sync;
%%     analysis suggests the strided exp+sum loop acts as a de facto
%%     barrier in practice. Apply when bit-identical-with-llama.cpp is
%%     not required and theoretical race-safety is preferred.
%%
%% See bpd/docs/methodology/subsumed-software-fix-catalog.md for full
%% diagnosis and rationale of each fix.

softmax_kernel(Kernel) :-
    %% Default: bug-for-bug compatible with llama.cpp soft_max_f32.
    %% Empty fix list means "exactly emulate the subsumed software."
    softmax_kernel([], Kernel).

softmax_kernel(Fixes, Kernel) :-
    %% Build phase 2 prelude: conditional pre-syncthreads based on Fixes.
    ( member(fix_softmax_phase_inter_race, Fixes)
    -> Phase2Prelude = [
           c_comment('fix_softmax_phase_inter_race: barrier before Phase 2 reuses buf_iw'),
           c_syncthreads
       ]
    ;  Phase2Prelude = []
    ),
    %% Assemble the full kernel body. The fixed-position parts are stable;
    %% only Phase2Prelude varies by fix-list.
    KernelHead = [%% int row = blockIdx.x;
        c_decl_init(c_type(int), row, c_member(c_var(blockIdx), x)),
        %% const int tid = threadIdx.x;
        c_decl_init(c_type(const(c_type(int))), tid, c_member(c_var(threadIdx), x)),
        %% const float *x = in + row * ncols;
        c_decl_init(c_type(const_ptr(c_type(float))), x,
            c_binop('+', c_var(in), c_binop('*', c_var(row), c_var(ncols)))),
        %% float *y = out + row * ncols;
        c_decl_init(c_type(ptr(c_type(float))), y,
            c_binop('+', c_var(out), c_binop('*', c_var(row), c_var(ncols)))),
        c_blank,
        %% __shared__ float buf_iw[8]; reused by both block_reduce phases
        c_shared_decl(c_type(float), buf_iw, c_int(8)),
        c_blank,
        c_comment('Phase 1: find max via block_reduce_max'),
        c_decl_init(c_type(float), max_val, c_var('-INFINITY')),
        c_for(c_decl_init(c_type(int), col, c_var(tid)),
              c_binop('<', c_var(col), c_var(ncols)),
              c_compound_step(c_var(col), '+=', c_int(256)),
              [c_assign(c_var(max_val),
                  c_call(fmaxf, [c_var(max_val), c_index(c_var(x), c_var(col))]))]),
        c_assign(c_var(max_val), c_call(block_reduce_max, [c_var(max_val), c_var(buf_iw)])),
        c_blank,
        c_comment('Phase 2: exp(x - max), sum via block_reduce_sum'),
        c_decl_init(c_type(float), tmp, c_float_f(0.0)),
        c_for(c_decl_init(c_type(int), col, c_var(tid)),
              c_binop('<', c_var(col), c_var(ncols)),
              c_compound_step(c_var(col), '+=', c_int(256)),
              [c_decl_init(c_type(float), val,
                   c_call(expf,
                       [c_binop('-', c_index(c_var(x), c_var(col)), c_var(max_val))])),
               c_assign(c_index(c_var(y), c_var(col)), c_var(val)),
               c_compound_assign('+=', c_var(tmp), c_var(val))])
    ],
    KernelTail = [
        c_assign(c_var(tmp), c_call(block_reduce_sum, [c_var(tmp), c_var(buf_iw)])),
        c_blank,
        c_comment('Phase 3: normalize'),
        c_decl_init(c_type(float), inv_sum,
            c_binop('/', c_float_f(1.0), c_var(tmp))),
        c_for(c_decl_init(c_type(int), col, c_var(tid)),
              c_binop('<', c_var(col), c_var(ncols)),
              c_compound_step(c_var(col), '+=', c_int(256)),
              [c_assign(c_index(c_var(y), c_var(col)),
                  c_binop('*', c_index(c_var(y), c_var(col)), c_var(inv_sum)))])
    ],
    %% Splice: KernelHead + Phase2Prelude + KernelTail
    append(KernelHead, Phase2Prelude, BodyMid),
    append(BodyMid, KernelTail, Body),
    Kernel = c_func(['__global__'], c_type(void), k_softmax,
        [param(c_type(const_restrict_ptr(c_type(float))), in),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), ncols)],
        Body).


%% kernel_available_fixes(+KernelPred, -FixList)
%%
%% Per 2026-05-18 ~08:00 UTC: discovery predicate for the bit-identical test
%% harness. Lists the named fixes available for a given kernel emit
%% predicate. The harness uses this to enumerate the fix-flag powerset and
%% test bit-identical output across all combinations.
%%
%% A kernel with [] available fixes is invariant — its emit is the same
%% regardless of fix-list. A kernel with N fixes has 2^N distinct emit
%% variants the harness can test.
kernel_available_fixes(softmax_kernel, [fix_softmax_phase_inter_race]).
kernel_available_fixes(rms_norm_kernel, []).
kernel_available_fixes(warp_reduce_sum_helper, []).
kernel_available_fixes(warp_reduce_max_helper, []).
kernel_available_fixes(block_reduce_sum_helper, []).
kernel_available_fixes(block_reduce_max_helper, []).


%% fix_description(+FixAtom, -Description)
%%
%% Per 2026-05-18 ~08:00 UTC: harness uses this for reports.
%% Each fix represents a named, individually-toggleable defect-repair in the
%% substrate's reproduction of the subsumed software (Ollama/llama.cpp).
%% The default emit is bug-for-bug compatible (empty fix list); fixes are
%% authored acts of substrate authority that disable specific identified
%% defects. See bpd/docs/methodology/subsumed-software-fix-catalog.md.
fix_description(fix_softmax_phase_inter_race,
    "Pre-syncthreads between Phase 1 and Phase 2 of softmax block reduction. \c
     Defends against theoretical race where slow warps in Phase 1's gather \c
     read could overlap with fast warps' Phase 2 writes to the shared buffer. \c
     llama.cpp soft_max_f32 does not include this sync; the strided exp+sum \c
     loop acts as de facto barrier in practice. Apply when bit-identical-with- \c
     llama.cpp is not required and theoretical race-safety is preferred. \c
     Origin: substrate analysis 2026-05-18. Severity: theoretical only \c
     (no empirical race observed). Status: substrate-authored fix.").


%% softmax_wrapper(-Wrapper)
%%
%% C API wrapper for the softmax kernel. Launches one block per row,
%% 256 threads per block.
%%
%% Launch: k_softmax<<<nrows, 256>>>(in, out, ncols)
softmax_wrapper(Wrapper) :-
    Wrapper = c_func(c_type(void), gpu_softmax,
        [param(c_type(const_restrict_ptr(c_type(float))), in),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), nrows),
         param(c_type(int), ncols)],
        [c_cuda_launch(k_softmax,
            c_var(nrows),
            c_int(256),
            [c_var(in), c_var(out), c_var(ncols)])]).


%% rms_norm_kernel(-Kernel)
%%
%% Emits the RMS normalization kernel for layer normalization in
%% transformer architectures. The kernel computes:
%%
%%   out[row, j] = in[row, j] * rsqrtf(mean(in[row, :]^2) + eps) * weight[j]
%%
%% IMPLEMENTATION: 256 threads per block, strided partial sum + warp-
%% level reduction + cross-warp shared memory reduction. This matches
%% llama.cpp's ggml-cuda/norm.cu rms_norm_f32 pattern EXACTLY, which
%% is necessary for bit-identical output with Ollama on quantized models.
%%
%% Per mavchin's 2026-05-18 diagnosis: the previous sequential-on-thread-0
%% form differed from llama.cpp in 2047/2048 elements at up to 4 ULP,
%% which was the SOLE cause of token divergence with Ollama for
%% llama3.2:1b. The verbatim form below is verified self-consistent
%% (0/2048 diffs run-to-run) and produces the correct reduction order
%% to match llama.cpp's accumulation.
%%
%% Depends on warp_reduce_sum (emitted via warp_reduce_sum_helper/1).
%% That helper MUST appear above this kernel in the file.
%% Per 2026-05-18 ~07:00 UTC: replaced c_raw body with structural c_ast.
%% Per Heath's "(b) first, then decompose (c) into non-challenging subtasks":
%% the substrate now comprehends every line of the rms_norm kernel.
%% Cosmetic formatting differs slightly from the c_raw oracle (whitespace
%% in pointer decl, braces around single-statement if/for bodies, no parens
%% around ternary condition) but the C semantics are identical — same SASS
%% after NVCC compilation, same FP accumulation order, same bit output.
rms_norm_kernel(Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_rms_norm,
        [param(c_type(const_restrict_ptr(c_type(float))), in),
         param(c_type(const_restrict_ptr(c_type(float))), weight),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), cols),
         param(c_type(float), eps)],
        [%% int row = blockIdx.x;
         c_decl_init(c_type(int), row, c_member(c_var(blockIdx), x)),
         %% const int tid = threadIdx.x;
         c_decl_init(c_type(const(c_type(int))), tid, c_member(c_var(threadIdx), x)),
         %% const float *x = in + row * cols;
         c_decl_init(c_type(const_ptr(c_type(float))), x,
             c_binop('+', c_var(in), c_binop('*', c_var(row), c_var(cols)))),
         c_blank,
         c_comment('Step 1: strided partial sum (each of 256 threads)'),
         %% float tmp = 0.0f;
         c_decl_init(c_type(float), tmp, c_float_f(0.0)),
         %% for (int col = tid; col < cols; col += 256) { float xi = x[col]; tmp += xi*xi; }
         c_for(c_decl_init(c_type(int), col, c_var(tid)),
               c_binop('<', c_var(col), c_var(cols)),
               c_compound_step(c_var(col), '+=', c_int(256)),
               [c_decl_init(c_type(float), xi, c_index(c_var(x), c_var(col))),
                c_compound_assign('+=', c_var(tmp),
                    c_binop('*', c_var(xi), c_var(xi)))]),
         c_blank,
         c_comment('Step 2: full-block reduction (composed: warp_reduce_sum + cross-warp + warp_reduce_sum)'),
         %% __shared__ float s_sum[8]; (8 warps for block_size=256)
         c_shared_decl(c_type(float), s_sum, c_int(8)),
         %% tmp = block_reduce_sum(tmp, s_sum);
         %% Per Heath's 2026-05-18 ~07:35 UTC: substrate-honest factoring as
         %% callable __device__ function rather than statement-splicing.
         %% warp_id/lane_id are internal locals of block_reduce_sum, not
         %% leaked to the kernel scope. The helper is __forceinline__ so
         %% SASS after NVCC should be identical to the prior inline form.
         c_assign(c_var(tmp), c_call(block_reduce_sum, [c_var(tmp), c_var(s_sum)])),
         c_blank,
         c_comment('Step 3: normalize'),
         c_decl_init(c_type(float), scale,
             c_call(rsqrtf,
                 [c_binop('+',
                     c_binop('/', c_var(tmp), c_var(cols)),
                     c_var(eps))])),
         c_for(c_decl_init(c_type(int), col, c_var(tid)),
               c_binop('<', c_var(col), c_var(cols)),
               c_compound_step(c_var(col), '+=', c_int(256)),
               [c_assign(
                   c_index(c_var(out),
                       c_binop('+', c_binop('*', c_var(row), c_var(cols)), c_var(col))),
                   c_binop('*',
                       c_binop('*', c_index(c_var(x), c_var(col)), c_var(scale)),
                       c_index(c_var(weight), c_var(col))))])
        ]).

rms_norm_wrapper(W) :-
    W = c_func(c_type(void), gpu_rms_norm,
        [param(c_type(const_restrict_ptr(c_type(float))), i),
         param(c_type(const_restrict_ptr(c_type(float))), w),
         param(c_type(restrict_ptr(c_type(float))), o),
         param(c_type(int), r),
         param(c_type(int), c),
         param(c_type(float), e)],
        [c_cuda_launch(k_rms_norm, c_var(r), c_int(256),
            [c_var(i), c_var(w), c_var(o), c_var(c), c_var(e)])]).


%% ═══════════════════════════════════════════════════════════════════
%% ELEMENTWISE KERNELS (parameterized)
%% ═══════════════════════════════════════════════════════════════════

%% Binary elementwise template
binary_elem_kernel(KName, Op, K) :-
    K = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), a),
         param(c_type(const_restrict_ptr(c_type(float))), b),
         param(c_type(restrict_ptr(c_type(float))), o),
         param(c_type(int), n)],
        [c_decl_init(c_type(int), i,
             c_binop(+, c_binop(*, c_member(c_var(blockIdx), x),
                                    c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_if(c_binop(>=, c_var(i), c_var(n)), [c_return_void]),
         c_assign(c_index(c_var(o), c_var(i)),
             c_binop(Op, c_index(c_var(a), c_var(i)),
                         c_index(c_var(b), c_var(i))))]).

binary_wrapper(KName, WName, W) :-
    W = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), a),
         param(c_type(const_restrict_ptr(c_type(float))), b),
         param(c_type(restrict_ptr(c_type(float))), o),
         param(c_type(int), n)],
        [c_cuda_launch(KName,
            c_binop(/, c_paren(c_binop(+, c_var(n), c_int(255))), c_int(256)),
            c_int(256),
            [c_var(a), c_var(b), c_var(o), c_var(n)])]).


%% ═══════════════════════════════════════════════════════════════════
%% UNARY ACTIVATION KERNELS (BPD fact-driven generation)
%% Each activation is a BPD fact: activation_expr(Name, VarX, Expr)
%% The template generates the kernel + wrapper from the fact.
%% ═══════════════════════════════════════════════════════════════════

%% BPD activation facts — the KNOWLEDGE
%% Activation facts use c_float_f (f-suffixed float literals) rather
%% than c_float (bare double literals) because the kernel template
%% generates a __global__ CUDA kernel where x is `float`. Mixing
%% `double_literal * float_var` causes C/C++ promotion to double for
%% the entire subexpression, then narrowing at store-to-float. Using
%% c_float_f keeps the entire expression in float32.
%%
%% GELU is split into TWO canonical forms (per the 2026-05-17 gelu
%% terminology investigation; see lib/terminology.pl):
%%
%%   k_gelu_tanh — tanh approximation form:
%%       0.5x · (1 + tanh(√(2/π) · x · (1 + 0.044715·x²)))
%%     Matches ggml_gelu_f32 in external/llama.cpp/ggml/src/ggml-cpu/vec.h,
%%     also matches F.gelu(approximate='tanh') and HF gelu_new bit-equally.
%%     llama.cpp's graph builder uses this form at all three LLM_FFN_GELU
%%     dispatch sites — this is what Ollama runs at runtime.
%%
%%   k_gelu_erf — exact erf form:
%%       0.5x · (1 + erf(x · 1/√2))
%%     Matches ggml_gelu_erf_f32 in ggml-cpu/vec.h, also matches
%%     F.gelu(approximate='none') and HF gelu (default) bit-equally.
%%     llama.cpp does NOT use this form in its model dispatch — it's
%%     an alternative for use cases that need exact-form gelu.
%%
%% A k_gelu_quick (sigmoid form, x·sigmoid(1.702x)) variant exists
%% in ggml's API but is not yet expressed here. Add when needed.
%%
%% Constants are pre-computed at decimal-string precision:
%%   GELU_COEF_A     = 0.044715f       (cubic coefficient)
%%   SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f
%%   SQRT_2_INV      = 0.70710678118654752440084436210484f
%% These match the constants in ggml-cpu/vec.h exactly so that within-
%% target bit-identicality is achievable when both sides call the
%% same math library.
activation_expr(k_silu, X, c_binop(/, X, c_paren(c_binop(+, c_float_f(1.0), c_call(expf, [c_unop(-, X)]))))).
activation_expr(k_sigmoid, X, c_binop(/, c_float_f(1.0), c_paren(c_binop(+, c_float_f(1.0), c_call(expf, [c_unop(-, X)]))))).
activation_expr(k_relu, X, c_call(fmaxf, [c_float_f(0.0), X])).

%% k_gelu_tanh — the form ggml/llama.cpp/Ollama uses at runtime.
%%   0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)))
activation_expr(k_gelu_tanh, X,
    c_binop(*, c_binop(*, c_float_f(0.5), X),
        c_paren(c_binop(+, c_float_f(1.0),
            c_call(tanhf,
                [c_binop(*, c_binop(*, c_float_f(0.79788456080286535587989211986876), X),
                    c_paren(c_binop(+, c_float_f(1.0),
                        c_binop(*, c_binop(*, c_float_f(0.044715), X), X))))]))))).

%% k_gelu_erf — exact form. The original k_gelu pre-split was this form.
%%   0.5f * x * (1.0f + erff(x * SQRT_2_INV))
activation_expr(k_gelu_erf, X,
    c_binop(*, c_binop(*, c_float_f(0.5), X),
        c_paren(c_binop(+, c_float_f(1.0),
            c_call(erff, [c_binop(*, X, c_float_f(0.70710678118654752440084436210484))]))))).

activation_expr(k_tanh, X, c_call(tanhf, [X])).

%% ── Expanded elementwise library (2026-05-17) ──────────────────
%% Unary arithmetic ops
activation_expr(k_neg,  X, c_unop(-, X)).
activation_expr(k_abs,  X, c_call(fabsf, [X])).
activation_expr(k_sqr,  X, c_binop(*, X, X)).
activation_expr(k_sqrt, X, c_call(sqrtf, [X])).
activation_expr(k_exp,  X, c_call(expf, [X])).
activation_expr(k_log,  X, c_call(logf, [X])).

%% Additional activation functions
activation_expr(k_leaky_relu, X,
    c_ternary(c_binop(>, X, c_float_f(0.0)), X, c_binop(*, c_float_f(0.01), X))).
activation_expr(k_elu, X,
    c_ternary(c_binop(>, X, c_float_f(0.0)), X,
              c_binop(-, c_call(expf, [X]), c_float_f(1.0)))).
activation_expr(k_softplus, X,
    c_call(logf, [c_binop(+, c_float_f(1.0), c_call(expf, [X]))])).
activation_expr(k_hardsigmoid, X,
    c_call(fminf, [c_float_f(1.0),
                   c_call(fmaxf, [c_float_f(0.0),
                                  c_binop(+, c_binop(*, c_float_f(0.1666667), X),
                                             c_float_f(0.5))])])).
activation_expr(k_softsign, X,
    c_binop(/, X, c_paren(c_binop(+, c_float_f(1.0), c_call(fabsf, [X]))))).
activation_expr(k_selu, X,
    c_ternary(c_binop(>, X, c_float_f(0.0)),
              c_binop(*, c_float_f(1.05070098), X),
              c_binop(*, c_float_f(1.05070098),
                         c_binop(*, c_float_f(1.67326324),
                                    c_binop(-, c_call(expf, [X]), c_float_f(1.0)))))).

%% ── Stanford KernelBench L1 additions (2026-05-18) ──────────
%%
%% Dropout (identity at inference — no-op)
activation_expr(k_dropout, X, X).

%% Loss functions
binary_loss_expr(k_mse_elem, X, Y, c_binop(*, c_paren(c_binop(-, X, Y)), c_paren(c_binop(-, X, Y)))).

%% Softmax (numerically stable: subtract max, exp, normalize)
%% This is a REDUCTION kernel — needs its own template (not activation_expr).
%% Expressed as a multi-step fact for the BPD emitter.
softmax_kernel_spec(k_softmax, [
    step(max_reduce,  "float max_val = -INFINITY; for(int j=0;j<K;j++) if(in[row*K+j]>max_val) max_val=in[row*K+j];"),
    step(exp_subtract, "float sum = 0.0f; for(int j=0;j<K;j++) { out[row*K+j] = expf(in[row*K+j]-max_val); sum += out[row*K+j]; }"),
    step(normalize,   "for(int j=0;j<K;j++) out[row*K+j] /= sum;")
]).

%% Pooling (2D, parameterized by reduction op)
pool2d_spec(k_max_pool2d, max, "fmaxf(acc, val)").
pool2d_spec(k_avg_pool2d, avg, "acc + val").

%% Convolution (1D and 2D, direct implementation)
conv_spec(k_conv1d, 1, "sum += weight[k] * input[i + k];").
conv_spec(k_conv2d, 2, "sum += weight[ky*KW+kx] * input[(y+ky)*W+(x+kx)];").
conv_spec(k_depthwise_conv2d, 2, "sum += weight[ky*KW+kx] * input[c*H*W+(y+ky)*W+(x+kx)];").

%% Normalization variants (all follow the same pattern: normalize, scale, shift)
norm_spec(k_layer_norm, layer, "mean+var over last dim, per-element affine").
norm_spec(k_batch_norm, batch, "mean+var over batch dim, per-channel affine").
norm_spec(k_group_norm, group, "mean+var over groups, per-channel affine").
norm_spec(k_instance_norm, instance, "mean+var over spatial dims, per-channel affine").

%% Linear / GEMM (F32, not quantized)
matmul_spec(k_linear, "out[i] = bias[i]; for(int k=0;k<K;k++) out[i] += weight[i*K+k] * input[k];").
matmul_spec(k_gemm, "C[i*N+j] = 0; for(int k=0;k<K;k++) C[i*N+j] += A[i*K+k] * B[k*N+j];").

%% Memory layout ops (no computation, just data movement)
layout_spec(k_transpose_2d, "out[j*M+i] = in[i*N+j];").

%% Embedding lookup
lookup_spec(k_embedding, "out[i*D+d] = table[idx[i]*D+d];").

%% Loss functions (reduction)
loss_spec(k_cross_entropy, "loss = -sum(target[i] * logf(softmax[i]))").
loss_spec(k_nll_loss, "loss = -logf(prob[target_idx])").

%% Unary activation kernel template (generates from fact)
unary_activation_kernel(KName, K) :-
    activation_expr(KName, c_var(x), Expr),
    K = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), in),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), n)],
        [c_decl_init(c_type(int), i,
             c_binop(+, c_binop(*, c_member(c_var(blockIdx), x),
                                    c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_if(c_binop(>=, c_var(i), c_var(n)), [c_return_void]),
         c_decl_init(c_type(float), x, c_index(c_var(in), c_var(i))),
         c_assign(c_index(c_var(out), c_var(i)), Expr)]).

unary_activation_wrapper(KName, WName, W) :-
    W = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), i),
         param(c_type(restrict_ptr(c_type(float))), o),
         param(c_type(int), n)],
        [c_cuda_launch(KName,
            c_binop(/, c_paren(c_binop(+, c_var(n), c_int(255))), c_int(256)),
            c_int(256),
            [c_var(i), c_var(o), c_var(n)])]).


%% ═══════════════════════════════════════════════════════════════════
%% SiLU KERNEL (hand-written form, kept for historical bit-equivalence)
%% Used by generate_llama_kernels.pl's Program assembly to produce the
%% same llamatov_inference.cu output as before the activation_expr
%% generation existed. The activation-driven form (k_sigmoid, k_relu,
%% k_gelu, k_tanh) is the new BPD pattern.
%% ═══════════════════════════════════════════════════════════════════

silu_kernel(K) :-
    K = c_func(['__global__'], c_type(void), k_silu,
        [param(c_type(const_restrict_ptr(c_type(float))), in),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), n)],
        [c_decl_init(c_type(int), i,
             c_binop(+, c_binop(*, c_member(c_var(blockIdx), x),
                                    c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_if(c_binop(>=, c_var(i), c_var(n)), [c_return_void]),
         c_decl_init(c_type(float), x, c_index(c_var(in), c_var(i))),
         c_assign(c_index(c_var(out), c_var(i)),
             c_binop(/, c_var(x),
                 c_paren(c_binop(+, c_float(1.0),
                     c_call(expf, [c_unop(-, c_var(x))])))))]).

silu_wrapper(W) :-
    W = c_func(c_type(void), gpu_silu,
        [param(c_type(const_restrict_ptr(c_type(float))), i),
         param(c_type(restrict_ptr(c_type(float))), o),
         param(c_type(int), n)],
        [c_cuda_launch(k_silu,
            c_binop(/, c_paren(c_binop(+, c_var(n), c_int(255))), c_int(256)),
            c_int(256),
            [c_var(i), c_var(o), c_var(n)])]).


%% ============================================================================
%% RoPE — Rotary Position Embedding (half-split form)
%% ============================================================================
%%
%% Substrate-honest lift of k_rope from bpd/llamatov_kernels.cu (commit
%% c54317926 lineage) into BPD facts. This is the SIMPLEST valid RoPE
%% kernel: half-split geometry, matching the reference implementation
%% bit-for-bit.
%%
%% Math: for each (pos, head, i in [0, head_dim/2)):
%%   freq    = 1 / theta_base^(2i / head_dim)
%%   angle   = pos * freq
%%   q'[i]        = q[i] * cos(angle) - q[i+half] * sin(angle)
%%   q'[i+half]   = q[i] * sin(angle) + q[i+half] * cos(angle)
%%
%% Grid: dim3(seq_len, n_head). Block: min(head_dim/2, 256) threads.
%% Each block handles one (pos, head) pair; threads stride over half.
%%
%% Substrate-design observations:
%% - The kernel is IN-PLACE (q is read AND written). This is the
%%   substrate-historical form. A future fix flag could request
%%   out-of-place semantics.
%% - The "interleaved" RoPE form (pairs are q[2i], q[2i+1]) is a
%%   variant used by some llama derivatives. Not implemented here;
%%   would be a separate kernel (rope_kernel_interleaved/1) when needed.
%% - powf(theta, exp) is the substrate-historical choice. Modern variants
%%   sometimes use precomputed cos/sin tables; not implemented here.
%%
%% Per Heath's substrate-honest direction 2026-05-19 ~02:00 UTC:
%% "please start with the simplest example of RoPE that you can."
rope_kernel(K) :-
    K = c_func(['__global__'], c_type(void), k_rope,
        [param(c_type(restrict_ptr(c_type(float))), q),
         param(c_type(int), seq_len),
         param(c_type(int), n_head),
         param(c_type(int), head_dim),
         param(c_type(float), theta_base)],
        [c_decl_init(c_type(int), pos, c_member(c_var(blockIdx), x)),
         c_decl_init(c_type(int), h, c_member(c_var(blockIdx), y)),
         c_if(c_binop(>=, c_var(pos), c_var(seq_len)), [c_return_void]),
         c_blank,
         c_decl_init(c_type(int), half, c_binop(/, c_var(head_dim), c_int(2))),
         c_decl_init(c_type(ptr(c_type(float))), q_head,
             c_binop(+, c_var(q),
                 c_binop(*,
                     c_paren(c_binop(+,
                         c_binop(*, c_var(pos), c_var(n_head)),
                         c_var(h))),
                     c_var(head_dim)))),
         c_blank,
         c_for(c_decl_init(c_type(int), i, c_member(c_var(threadIdx), x)),
               c_binop(<, c_var(i), c_var(half)),
               c_compound_step(c_var(i), '+=', c_member(c_var(blockDim), x)),
               [c_decl_init(c_type(float), freq,
                    c_binop(/, c_float_f(1.0),
                        c_call(powf, [c_var(theta_base),
                            c_binop(/,
                                c_paren(c_call('(float)',
                                    [c_binop(*, c_int(2), c_var(i))])),
                                c_var(head_dim))]))),
                c_decl_init(c_type(float), angle,
                    c_binop(*, c_var(pos), c_var(freq))),
                c_decl_init(c_type(float), cos_a,
                    c_call(cosf, [c_var(angle)])),
                c_decl_init(c_type(float), sin_a,
                    c_call(sinf, [c_var(angle)])),
                c_blank,
                c_decl_init(c_type(float), q0,
                    c_index(c_var(q_head), c_var(i))),
                c_decl_init(c_type(float), q1,
                    c_index(c_var(q_head),
                        c_binop(+, c_var(i), c_var(half)))),
                c_assign(c_index(c_var(q_head), c_var(i)),
                    c_binop(-,
                        c_binop(*, c_var(q0), c_var(cos_a)),
                        c_binop(*, c_var(q1), c_var(sin_a)))),
                c_assign(c_index(c_var(q_head),
                             c_binop(+, c_var(i), c_var(half))),
                    c_binop(+,
                        c_binop(*, c_var(q0), c_var(sin_a)),
                        c_binop(*, c_var(q1), c_var(cos_a))))])]).

%% gpu_rope_q/5 wrapper: launches k_rope on the Q tensor with
%% dim3(seq_len, n_head) grid. Block size is min(head_dim/2, 256);
%% fixed at 256 here for simplicity.
%%
%% Renamed from earlier gpu_rope per medayek's driver-symbol parity
%% requirement 2026-05-19 ~06:35 UTC: the reference symbol in
%% llamatov_kernels.cu is gpu_rope_q (matching the substrate-historical
%% separate-Q-tensor handling), not bare gpu_rope. The Q variant uses
%% n_head; the K variant uses n_head_kv (GQA support).
rope_wrapper(W) :-
    W = c_func(c_type(void), gpu_rope_q,
        [param(c_type(restrict_ptr(c_type(float))), q),
         param(c_type(int), seq_len),
         param(c_type(int), n_head),
         param(c_type(int), head_dim),
         param(c_type(float), theta)],
        [c_cuda_launch(k_rope,
            c_call(dim3, [c_var(seq_len), c_var(n_head)]),
            c_int(256),
            [c_var(q), c_var(seq_len), c_var(n_head),
             c_var(head_dim), c_var(theta)])]).

%% gpu_rope_k/5 wrapper: launches k_rope on the K tensor with
%% dim3(seq_len, n_head_kv) grid. Identical kernel body to gpu_rope_q;
%% only the head-count parameter differs (GQA: n_head_kv can be < n_head).
%%
%% Per medayek's option-(a) quick-path direction 2026-05-19 ~06:35 UTC:
%% emit gpu_rope_k as identical to gpu_rope_q (same k_rope kernel) for
%% the driver swap. The substrate-design fix-flag refinement
%% (fix_rope_separate_k, which would emit a specialized k_rope_k kernel
%% with potentially different optimizations for the K tensor) is
%% follow-up work.
rope_k_wrapper(W) :-
    W = c_func(c_type(void), gpu_rope_k,
        [param(c_type(restrict_ptr(c_type(float))), k),
         param(c_type(int), seq_len),
         param(c_type(int), n_head_kv),
         param(c_type(int), head_dim),
         param(c_type(float), theta)],
        [c_cuda_launch(k_rope,
            c_call(dim3, [c_var(seq_len), c_var(n_head_kv)]),
            c_int(256),
            [c_var(k), c_var(seq_len), c_var(n_head_kv),
             c_var(head_dim), c_var(theta)])]).


%% ============================================================================
%% Token Embedding Lookup — gather embedding vectors by token ID
%% ============================================================================
%%
%% Substrate-honest lift of k_embed from bpd/llamatov_kernels.cu:136-143
%% into BPD facts.
%%
%% Operation: for each token position `pos` in [0, seq_len):
%%   tok = indices[pos]                       ← lookup the token ID
%%   out[pos*embd + j] = table[tok*embd + j]  ← copy embedding vector
%%
%% Grid: seq_len blocks (one per token). Block: 256 threads
%% (reference used min(embd,256); see fix_embed_block_size_optimal).
%% Threads in each block stride over the embedding dimension.
%%
%% Substantive substrate-design observation: this is the SIMPLEST
%% transformer kernel — pure data gather, no arithmetic. It maps a
%% sequence of token IDs to a sequence of embedding vectors. Required
%% as the first step of every forward pass.
%%
%% Per Heath's substrate-honest direction 2026-05-19 ~02:00 UTC and
%% the kernel-lift trajectory established by the RoPE lift (commit
%% 6f3237c81). Second of four planned kernel lifts before the
%% forward-pass orchestrator can be built.
embed_kernel(K) :-
    K = c_func(['__global__'], c_type(void), k_embed,
        [param(c_type(const_restrict_ptr(c_type(float))), table),
         param(c_type(const_restrict_ptr(c_type(int))), indices),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), seq_len),
         param(c_type(int), embd)],
        [c_decl_init(c_type(int), pos, c_member(c_var(blockIdx), x)),
         c_if(c_binop(>=, c_var(pos), c_var(seq_len)), [c_return_void]),
         c_decl_init(c_type(int), tok,
             c_index(c_var(indices), c_var(pos))),
         c_for(c_decl_init(c_type(int), j, c_member(c_var(threadIdx), x)),
               c_binop(<, c_var(j), c_var(embd)),
               c_compound_step(c_var(j), '+=', c_member(c_var(blockDim), x)),
               [c_assign(
                   c_index(c_var(out),
                       c_binop(+,
                           c_binop(*, c_var(pos), c_var(embd)),
                           c_var(j))),
                   c_index(c_var(table),
                       c_binop(+,
                           c_binop(*, c_var(tok), c_var(embd)),
                           c_var(j))))])]).

%% gpu_embed/5 wrapper: launches k_embed with seq_len blocks of 256 threads.
%% A future fix flag fix_embed_block_size_optimal could compute
%% min(embd, 256) from embd at launch time, matching the reference's
%% efficiency profile for small embedding dimensions.
embed_wrapper(W) :-
    W = c_func(c_type(void), gpu_embed,
        [param(c_type(const_restrict_ptr(c_type(float))), table),
         param(c_type(const_restrict_ptr(c_type(int))), indices),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), seq_len),
         param(c_type(int), embd)],
        [c_cuda_launch(k_embed,
            c_var(seq_len),
            c_int(256),
            [c_var(table), c_var(indices), c_var(out),
             c_var(seq_len), c_var(embd)])]).


%% ============================================================================
%% Causal Mask — set upper triangle of attention matrix to -inf
%% ============================================================================
%%
%% Substrate-honest lift of k_causal_mask from bpd/llamatov_kernels.cu:204-218.
%%
%% Operation: for an attention score matrix att[T×T], set elements
%% above the diagonal (col > row) to -1e30f. This prevents attention
%% from looking at future tokens during autoregressive generation.
%%
%% The kernel itself processes one (row, col) pair per thread. Grid is
%% dim3(T, ceil(T/256)) — T rows × enough column-tiles to cover T.
%% Each block has 256 threads; threadIdx.x is the column-within-tile,
%% blockIdx.x is the row, blockIdx.y is the column-tile.
%%
%% Substrate-historical wrapper structure: host-side for-loop over
%% n_head, launching the kernel once per attention head. Could be a
%% single 3D launch (T, ceil(T/256), n_head) but lifts as-is.
%%
%% Per Heath's kernel-lift trajectory 2026-05-19. Third of four planned
%% kernel lifts before the forward-pass orchestrator can be built.
causal_mask_kernel(K) :-
    K = c_func(['__global__'], c_type(void), k_causal_mask,
        [param(c_type(restrict_ptr(c_type(float))), att),
         param(c_type(int), 'T'),
         param(c_type(int), stride)],
        [c_decl_init(c_type(int), row, c_member(c_var(blockIdx), x)),
         c_decl_init(c_type(int), col,
             c_binop(+,
                 c_member(c_var(threadIdx), x),
                 c_binop(*,
                     c_member(c_var(blockIdx), y),
                     c_member(c_var(blockDim), x)))),
         c_if(c_binop('&&',
                  c_binop(<, c_var(row), c_var('T')),
                  c_binop('&&',
                      c_binop(<, c_var(col), c_var('T')),
                      c_binop(>, c_var(col), c_var(row)))),
              [c_assign(
                  c_index(c_var(att),
                      c_binop(+,
                          c_binop(*, c_var(row), c_var(stride)),
                          c_var(col))),
                  c_float_raw('-1e30f'))])]).

%% gpu_causal_mask/3 wrapper: launches k_causal_mask once per
%% attention head. Grid per launch: dim3(T, (T+255)/256).
%%
%% Substrate-design observation: this uses a host-side for-loop over
%% heads rather than a 3D grid. The substrate-historical reason is
%% likely "one substantive launch per head was simpler to write."
%% A future fix flag fix_causal_mask_3d_grid could collapse this to a
%% single launch with grid dim3(T, ceil(T/256), n_head).
causal_mask_wrapper(W) :-
    W = c_func(c_type(void), gpu_causal_mask,
        [param(c_type(restrict_ptr(c_type(float))), att),
         param(c_type(int), n_head),
         param(c_type(int), 'T')],
        [c_decl_init(c_type(int), stride, c_var('T')),
         c_for(c_decl_init(c_type(int), h, c_int(0)),
               c_binop(<, c_var(h), c_var(n_head)),
               c_compound_step(c_var(h), '+=', c_int(1)),
               [c_cuda_launch(k_causal_mask,
                   c_call(dim3,
                       [c_var('T'),
                        c_binop(/,
                            c_paren(c_binop(+, c_var('T'), c_int(255))),
                            c_int(256))]),
                   c_int(256),
                   [c_binop(+, c_var(att),
                       c_binop(*,
                           c_binop(*, c_var(h), c_var('T')),
                           c_var('T'))),
                    c_var('T'),
                    c_var(stride)])])]).


%% ============================================================================
%% Matmul — General M×K @ K×N → M×N with tiled shared memory
%% ============================================================================
%%
%% Substrate-honest lift of k_matmul from bpd/llamatov_kernels.cu:14-30.
%%
%% Computes C[M,N] = A[M,K] @ B[K,N] using a square TILE×TILE tiling
%% strategy in shared memory. Each block computes one TILE×TILE output
%% sub-block by walking through K in TILE-sized chunks, loading A and B
%% tiles cooperatively into shared memory and accumulating products.
%%
%% Grid: dim3(ceil(N/TILE), ceil(M/TILE)). Block: dim3(TILE, TILE).
%% Total threads per block: TILE² = 1024 for TILE=32.
%%
%% TILE size is a substrate-emit parameter. Default TILE=32 matches
%% the substrate-historical #define TILE 32 in llamatov_kernels.cu.
%%
%% Distinct from vecmat_kernel/1 (which handles M=1 decode path) and
%% sgemv_kernel_*/1 (vector-matrix). This is the general M×N case for
%% prefill, attention QK^T, and attention @V.
%%
%% Per Heath's kernel-lift trajectory 2026-05-19. Fourth and largest
%% of the four planned kernel lifts.
matmul_kernel(K) :-
    matmul_kernel(K, 32).  % Default TILE=32

matmul_kernel(Kernel, TILE) :-
    integer(TILE),
    TILE > 0,
    Kernel = c_func(['__global__'], c_type(void), k_matmul,
        [param(c_type(const_restrict_ptr(c_type(float))), 'A'),
         param(c_type(const_restrict_ptr(c_type(float))), 'B'),
         param(c_type(restrict_ptr(c_type(float))), 'C'),
         param(c_type(int), 'M'),
         param(c_type(int), 'N'),
         param(c_type(int), 'K')],
        [c_shared_decl_2d(c_type(float), 'sA', c_int(TILE), c_int(TILE)),
         c_shared_decl_2d(c_type(float), 'sB', c_int(TILE), c_int(TILE)),
         c_decl_init(c_type(int), row,
             c_binop(+,
                 c_binop(*, c_member(c_var(blockIdx), y), c_int(TILE)),
                 c_member(c_var(threadIdx), y))),
         c_decl_init(c_type(int), col,
             c_binop(+,
                 c_binop(*, c_member(c_var(blockIdx), x), c_int(TILE)),
                 c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         c_for(c_decl_init(c_type(int), t, c_int(0)),
               c_binop(<, c_var(t),
                   c_binop(/,
                       c_paren(c_binop(+, c_var('K'),
                           c_paren(c_binop(-, c_int(TILE), c_int(1))))),
                       c_int(TILE))),
               c_compound_step(c_var(t), '+=', c_int(1)),
               [%% Load tile of A into shared memory
                c_if(c_binop('&&',
                         c_binop(<, c_var(row), c_var('M')),
                         c_binop(<,
                             c_binop(+,
                                 c_binop(*, c_var(t), c_int(TILE)),
                                 c_member(c_var(threadIdx), x)),
                             c_var('K'))),
                     [c_assign(
                         c_index(c_index(c_var('sA'),
                             c_member(c_var(threadIdx), y)),
                             c_member(c_var(threadIdx), x)),
                         c_index(c_var('A'),
                             c_binop(+,
                                 c_binop(*, c_var(row), c_var('K')),
                                 c_binop(+,
                                     c_binop(*, c_var(t), c_int(TILE)),
                                     c_member(c_var(threadIdx), x)))))],
                     [c_assign(
                         c_index(c_index(c_var('sA'),
                             c_member(c_var(threadIdx), y)),
                             c_member(c_var(threadIdx), x)),
                         c_float_f(0.0))]),
                %% Load tile of B into shared memory
                c_if(c_binop('&&',
                         c_binop(<,
                             c_binop(+,
                                 c_binop(*, c_var(t), c_int(TILE)),
                                 c_member(c_var(threadIdx), y)),
                             c_var('K')),
                         c_binop(<, c_var(col), c_var('N'))),
                     [c_assign(
                         c_index(c_index(c_var('sB'),
                             c_member(c_var(threadIdx), y)),
                             c_member(c_var(threadIdx), x)),
                         c_index(c_var('B'),
                             c_binop(+,
                                 c_binop(*,
                                     c_paren(c_binop(+,
                                         c_binop(*, c_var(t), c_int(TILE)),
                                         c_member(c_var(threadIdx), y))),
                                     c_var('N')),
                                 c_var(col))))],
                     [c_assign(
                         c_index(c_index(c_var('sB'),
                             c_member(c_var(threadIdx), y)),
                             c_member(c_var(threadIdx), x)),
                         c_float_f(0.0))]),
                c_syncthreads,
                %% Inner accumulation over tile
                c_for(c_decl_init(c_type(int), k, c_int(0)),
                      c_binop(<, c_var(k), c_int(TILE)),
                      c_compound_step(c_var(k), '+=', c_int(1)),
                      [c_compound_assign('+=', c_var(sum),
                          c_binop(*,
                              c_index(c_index(c_var('sA'),
                                  c_member(c_var(threadIdx), y)),
                                  c_var(k)),
                              c_index(c_index(c_var('sB'),
                                  c_var(k)),
                                  c_member(c_var(threadIdx), x))))]),
                c_syncthreads]),
         c_if(c_binop('&&',
                  c_binop(<, c_var(row), c_var('M')),
                  c_binop(<, c_var(col), c_var('N'))),
              [c_assign(
                  c_index(c_var('C'),
                      c_binop(+,
                          c_binop(*, c_var(row), c_var('N')),
                          c_var(col))),
                  c_var(sum))])]).

%% gpu_matmul/6 wrapper: launches k_matmul with computed grid/block
%% based on M, N, and TILE size.
matmul_wrapper(W) :-
    matmul_wrapper(W, 32).

matmul_wrapper(W, TILE) :-
    integer(TILE),
    TILE > 0,
    W = c_func(c_type(void), gpu_matmul,
        [param(c_type(const_restrict_ptr(c_type(float))), 'A'),
         param(c_type(const_restrict_ptr(c_type(float))), 'B'),
         param(c_type(restrict_ptr(c_type(float))), 'C'),
         param(c_type(int), 'M'),
         param(c_type(int), 'N'),
         param(c_type(int), 'K')],
        [c_cuda_launch(k_matmul,
            c_call(dim3,
                [c_binop(/,
                    c_paren(c_binop(+, c_var('N'),
                        c_paren(c_binop(-, c_int(TILE), c_int(1))))),
                    c_int(TILE)),
                 c_binop(/,
                    c_paren(c_binop(+, c_var('M'),
                        c_paren(c_binop(-, c_int(TILE), c_int(1))))),
                    c_int(TILE))]),
            c_call(dim3, [c_int(TILE), c_int(TILE)]),
            [c_var('A'), c_var('B'), c_var('C'),
             c_var('M'), c_var('N'), c_var('K')])]).

%% gpu_matmul_opt/6 wrapper: same body as gpu_matmul for now.
%%
%% Per medayek's option-(a) quick-path direction 2026-05-19 ~06:35 UTC:
%% emit gpu_matmul_opt identical to gpu_matmul to unblock the driver
%% swap. The substrate-design fix-flag refinement (fix_matmul_opt,
%% cooperative-load variant, or other matmul optimizations) is
%% follow-up work.
%%
%% Symbol parity matters more for the swap than substrate-design
%% improvement. medayek will verify E2E correctness with the
%% identical-body wrapper; performance variants come later as
%% additional kernels with explicit fix flags.
matmul_opt_wrapper(W) :-
    matmul_opt_wrapper(W, 32).

matmul_opt_wrapper(W, TILE) :-
    integer(TILE),
    TILE > 0,
    W = c_func(c_type(void), gpu_matmul_opt,
        [param(c_type(const_restrict_ptr(c_type(float))), 'A'),
         param(c_type(const_restrict_ptr(c_type(float))), 'B'),
         param(c_type(restrict_ptr(c_type(float))), 'C'),
         param(c_type(int), 'M'),
         param(c_type(int), 'N'),
         param(c_type(int), 'K')],
        [c_cuda_launch(k_matmul,
            c_call(dim3,
                [c_binop('/',
                    c_paren(c_binop('+', c_var('N'),
                        c_paren(c_binop('-', c_int(TILE), c_int(1))))),
                    c_int(TILE)),
                 c_binop('/',
                    c_paren(c_binop('+', c_var('M'),
                        c_paren(c_binop('-', c_int(TILE), c_int(1))))),
                    c_int(TILE))]),
            c_call(dim3, [c_int(TILE), c_int(TILE)]),
            [c_var('A'), c_var('B'), c_var('C'),
             c_var('M'), c_var('N'), c_var('K')])]).


%% ============================================================================
%% LayerNorm — Mean + variance normalization with bias (BERT-class)
%% ============================================================================
%%
%% Substrate-honest lift of k_layer_norm from
%% bpd/llamatov_kernels.cu:225-254. This is the FAITHFUL baseline lift —
%% the substrate-historical naive thread-0-only reduction.
%%
%% Per Heath's substrate-design judgment 2026-05-19 ~06:30 UTC:
%% "lift faithfully -- this way we learn the performance technique
%%  that makes it different. it becomes a feature flag parameter that
%%  is a performance win. another one. we set that flag and we get in
%%  return way better memory performance."
%%
%% The current substrate's rms_norm_kernel uses parallel block_reduce_sum
%% (much faster than thread-0-only). A future fix flag
%% fix_layer_norm_parallel_reduce would emit the parallel-reduction
%% variant that uses warp_reduce_sum + cross-warp + warp_reduce_sum
%% (same pattern as rms_norm_kernel). The baseline lift captures the
%% naive form so the performance delta is empirically measurable.
%%
%% Math: for each row (block) and each column j in [0, cols):
%%   mean    = (1/cols) * sum_j(x[j])
%%   var     = (1/cols) * sum_j((x[j] - mean)^2)
%%   inv_std = rsqrtf(var + eps)
%%   y[j]    = (x[j] - mean) * inv_std * weight[j] + bias[j]
%%
%% Distinct from RMSNorm: LayerNorm SUBTRACTS the mean (RMSNorm
%% doesn't) and has bias (RMSNorm has only weight).
%%
%% Grid: rows blocks (one per row). Block: min(cols, 256) threads
%% (substrate uses literal 256; see fix_layer_norm_block_size_optimal).
%%
%% Required for non-llama architectures: BERT, GPT-2, nomic-bert,
%% mxbai-embed-large, and other models that use LayerNorm rather
%% than RMSNorm.
layer_norm_kernel(K) :-
    K = c_func(['__global__'], c_type(void), k_layer_norm,
        [param(c_type(const_restrict_ptr(c_type(float))), in),
         param(c_type(const_restrict_ptr(c_type(float))), weight),
         param(c_type(const_restrict_ptr(c_type(float))), bias),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), cols),
         param(c_type(float), eps)],
        [c_decl_init(c_type(int), row, c_member(c_var(blockIdx), x)),
         c_decl_init(c_type(const_ptr(c_type(float))), x,
             c_binop(+, c_var(in),
                 c_binop(*, c_var(row), c_var(cols)))),
         c_decl_init(c_type(ptr(c_type(float))), y,
             c_binop(+, c_var(out),
                 c_binop(*, c_var(row), c_var(cols)))),
         c_blank,
         c_shared_scalar_decl(c_type(float), [s_mean, s_inv_std]),
         c_blank,
         c_comment('Thread 0 computes mean and variance (naive baseline)'),
         c_if(c_binop('==', c_member(c_var(threadIdx), x), c_int(0)),
             [c_decl_init(c_type(float), sum, c_float_f(0.0)),
              c_decl_init(c_type(float), sum_sq, c_float_f(0.0)),
              c_for(c_decl_init(c_type(int), j, c_int(0)),
                    c_binop(<, c_var(j), c_var(cols)),
                    c_compound_step(c_var(j), '+=', c_int(1)),
                    [c_compound_assign('+=', c_var(sum),
                        c_index(c_var(x), c_var(j)))]),
              c_assign(c_var(s_mean),
                  c_binop(/, c_var(sum), c_var(cols))),
              c_for(c_decl_init(c_type(int), j, c_int(0)),
                    c_binop(<, c_var(j), c_var(cols)),
                    c_compound_step(c_var(j), '+=', c_int(1)),
                    [c_decl_init(c_type(float), d,
                        c_binop(-, c_index(c_var(x), c_var(j)),
                                   c_var(s_mean))),
                     c_compound_assign('+=', c_var(sum_sq),
                        c_binop(*, c_var(d), c_var(d)))]),
              c_assign(c_var(s_inv_std),
                  c_call(rsqrtf,
                      [c_binop(+,
                          c_binop(/, c_var(sum_sq), c_var(cols)),
                          c_var(eps))]))]),
         c_syncthreads,
         c_blank,
         c_comment('All threads normalize in parallel'),
         c_decl_init(c_type(float), mean, c_var(s_mean)),
         c_decl_init(c_type(float), inv_std, c_var(s_inv_std)),
         c_for(c_decl_init(c_type(int), j, c_member(c_var(threadIdx), x)),
               c_binop(<, c_var(j), c_var(cols)),
               c_compound_step(c_var(j), '+=', c_member(c_var(blockDim), x)),
               [c_assign(c_index(c_var(y), c_var(j)),
                   c_binop(+,
                       c_binop(*,
                           c_binop(*,
                               c_paren(c_binop(-,
                                   c_index(c_var(x), c_var(j)),
                                   c_var(mean))),
                               c_var(inv_std)),
                           c_index(c_var(weight), c_var(j))),
                       c_index(c_var(bias), c_var(j))))])]).

%% gpu_layer_norm/7 wrapper: launches k_layer_norm with rows blocks.
%% Block size is literal 256; reference uses min(cols, 256) at launch.
%% Future fix flag fix_layer_norm_block_size_optimal could compute the
%% optimal block size from cols.
layer_norm_wrapper(W) :-
    W = c_func(c_type(void), gpu_layer_norm,
        [param(c_type(const_restrict_ptr(c_type(float))), in),
         param(c_type(const_restrict_ptr(c_type(float))), weight),
         param(c_type(const_restrict_ptr(c_type(float))), bias),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), rows),
         param(c_type(int), cols),
         param(c_type(float), eps)],
        [c_cuda_launch(k_layer_norm,
            c_var(rows),
            c_int(256),
            [c_var(in), c_var(weight), c_var(bias), c_var(out),
             c_var(cols), c_var(eps)])]).
