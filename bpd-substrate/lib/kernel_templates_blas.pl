%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% =============================================================================
%% kernel_templates_blas.pl — BLAS kernels for NVIDIA Tech Level subsumption
%% =============================================================================
%%
%% Substrate emit for BLAS-class kernels with both substrate-native and
%% cuBLAS-matching configurations. Per Heath's 2026-05-18 ~16:30 UTC
%% framing of "subsuming the NVIDIA Technology Level" across three modes:
%%
%%   Mode 1: bit-identical with cuBLAS (the cublas_match predicates)
%%   Mode 2: substrate-optimal faster-than-cuBLAS with named discrepancies
%%   Mode 3: account-for-every-bit, every transformation traceable
%%
%% This module currently ships Mode 1 + a substrate-native baseline for
%% sgemv. Mode 2 will accumulate as we identify substrate-optimal patterns
%% empirically.
%%
%% =============================================================================
%% SUBSTANTIVE METHODOLOGY: BIT-DETERMINING vs TICK-DETERMINING
%% =============================================================================
%%
%% Per mavchin's SASS analysis on cuBLAS sgemv (2026-05-18 ~18:15 UTC):
%%
%% "The accumulation order is what determines the bits. The register
%%  allocation is what determines the ticks."
%%
%% This distinguishes two classes of kernel parameters:
%%
%%   BIT-DETERMINING (substrate emits these explicitly):
%%     - Iteration order over K dimension
%%     - Reduction tree shape (warp-shuffle vs shared-memory strided)
%%     - Logical accumulator chain structure
%%     - FMA contraction boundaries (where __fmul_rn/__fadd_rn force
%%       explicit rounding)
%%
%%   TICK-DETERMINING (compiler chooses freely via standard codegen):
%%     - Physical register allocation (R41 vs R35 alternation in SASS)
%%     - Instruction scheduling (loads interleaved with FMAs)
%%     - Cache hint variants (LDG.CI vs LDG.CA — same bits)
%%
%% For Mode 1 subsumption, the substrate emits C code expressing the
%% bit-determining parameters at the algorithm level, and trusts nvcc
%% to produce equivalent SASS via standard -O2 codegen. The bit-identical
%% test verifies. SASS-level register matching is not required for
%% output-bit identity.
%%
%% This principle deserves capture as substrate-honesty methodology
%% when the cuBLAS-match kernel ships and the principle is empirically
%% demonstrated.
%%
%% =============================================================================
%% CUBLAS SGEMV STRUCTURE (from mavchin's cuobjdump SASS analysis)
%% =============================================================================
%%
%% Kernel: gemv2T_kernel_val<ii,ffff,128,16,2,2>
%%
%% Launch geometry:
%%   blockDim = 128 threads per block
%%   16 threads per row → 8 rows per block
%%   gridDim.x = M / 8 blocks
%%
%% Per-thread accumulation:
%%   x preloaded into shared memory once via cooperative load + __syncthreads
%%   Sequential 8-FMA chain reading sx[0..7] (shared) and A[row*K + j..j+7]
%%     (global, via LDG.E.CI cache-invalidate)
%%   Single logical accumulator (compiler ping-pongs R41/R35 for latency
%%   hiding; bit-invisible at the algorithm level)
%%   Loop with stride 16 (threads_per_row) × 8 (elements_per_iter) = 128
%%
%% Cross-thread reduction:
%%   __shared__ float sred[128]    (128 = blockDim, 16 per row group)
%%   Store partial sum → __syncthreads
%%   Strided pair reduction: 8 → 4 → 2 → 1 with __syncthreads between
%%   Thread 0 of each row writes y[row]
%%
%% The bit-determining differences from a warp-shuffle implementation:
%%   - Iteration order over K: tile-by-8 sequential reads vs stride-32
%%     thread-interleaved reads → different FMA evaluation order
%%   - Reduction tree shape: strided shared-mem pairs vs warp-shuffle
%%     XOR butterfly → different 16-way sum order
%%
%% =============================================================================

:- module(kernel_templates_blas, [
    %% Mode 1 — bit-identical with cuBLAS
    sgemv_kernel_cublas_match/2,    % +KName, -Kernel
    sgemv_wrapper_cublas_match/2,   % +KName, -Wrapper

    %% Substrate-native baseline (the warp-shuffle 12-ULP-gap kernel)
    sgemv_kernel_substrate_native/2,
    sgemv_wrapper_substrate_native/2,

    %% Elementwise kernel factory
    elem_op/4,                      % +KName, -Params, -OutputExpr, -ComputeExpr
    elem_kernel/2,                  % +KName, -Kernel

    %% Pooling kernels (KernelBench L1)
    pool_kernel/3,                  % +KName, +PoolType, -Kernel

    %% Convolution kernels (KernelBench L1 #50-87)
    conv_kernel/5,                  % +KName, +Dims, +Direction, +Groups, -Kernel
    conv_kernel_with_epilogue/6,    % +KName, +Dims, +Direction, +Groups, +Epilogue, -Kernel

    %% Scan kernels (KernelBench L1 #89-93)
    scan_kernel/3,                  % +KName, +Op, -Kernel
    upsample_kernel/2,            % +KName, -Kernel

    %% Normalization kernels (KernelBench L1 #33-40)
    norm_kernel/2,                  % +KName, -Kernel

    %% BLAS L1: parameterized reduction kernels
    blas_l1_reduction_kernel/4,     % +KName, +Op, +NormSafety, -Kernel
    blas_l1_sdot_kernel/3,          % +KName, +NormSafety, -Kernel
    blas_l1_snrm2_kernel/3,         % +KName, +NormSafety, -Kernel
    blas_l1_sasum_kernel/3,         % +KName, +NormSafety, -Kernel

    %% Config metadata
    kernel_configs/2,
    config_description/2,
    kernel_available_fixes/2,
    fix_description/2,

    %% Driver-symbol wrappers — gpu_* for ctypes dispatch
    gpu_scale_wrapper/1,
    gpu_copy_d2d_wrapper/1
]).

:- discontiguous pool_kernel/3.
:- discontiguous scan_kernel/3.
:- discontiguous norm_kernel/2.
:- discontiguous conv_kernel/5.

:- use_module(c_ast).
:- use_module(epilogue_generator).

:- dynamic kernel_configs/2.
:- dynamic config_description/2.
:- dynamic kernel_available_fixes/2.
:- dynamic fix_description/2.
:- discontiguous kernel_available_fixes/2.
:- discontiguous fix_description/2.

%% ═══════════════════════════════════════════════════════════════
%% Conv accumulation loop helpers — structured AST generation
%% Replaces c_raw for/if/brace blocks with typed c_ast nodes.
%% ═══════════════════════════════════════════════════════════════

%% conv2d_accumulate(-Stmts)
%% Triple-nested ci/kh/kw loop with bounds check and input*weight accumulation.
%% Used by: conv_kernel(k_conv2d, ...) 
conv2d_accum(Stmts) :-
    Stmts = [c_for(
        c_decl_init(c_type(int), ci, c_int(0)),
        c_binop('<', c_var(ci), c_var('C_in')),
        c_pre_inc(c_var(ci)),
        [c_for(
            c_decl_init(c_type(int), kh, c_int(0)),
            c_binop('<', c_var(kh), c_var(kH)),
            c_pre_inc(c_var(kh)),
            [c_for(
                c_decl_init(c_type(int), kw, c_int(0)),
                c_binop('<', c_var(kw), c_var(kW)),
                c_pre_inc(c_var(kw)),
                [c_decl_init(c_type(int), hi,
                    c_binop('+', c_binop('-', c_binop('*', c_var(ho), c_var(stride_h)), c_var(pad_h)),
                                 c_binop('*', c_var(kh), c_var(dil_h)))),
                 c_decl_init(c_type(int), wi,
                    c_binop('+', c_binop('-', c_binop('*', c_var(wo), c_var(stride_w)), c_var(pad_w)),
                                 c_binop('*', c_var(kw), c_var(dil_w)))),
                 c_if(c_binop('&&',
                        c_binop('&&', c_binop('>=', c_var(hi), c_int(0)),
                                      c_binop('<', c_var(hi), c_var('H_in'))),
                        c_binop('&&', c_binop('>=', c_var(wi), c_int(0)),
                                      c_binop('<', c_var(wi), c_var('W_in')))),
                    [c_decl_init(c_type(int), in_idx,
                        c_nd_index([n, 'C_in', ci, 'H_in', hi, 'W_in', wi])),
                     c_decl_init(c_type(int), w_idx,
                        c_nd_index([co, 'C_in', ci, kH, kh, kW, kw])),
                     c_compound_assign('+=', c_var(sum),
                        c_binop('*', c_index(c_var(input), c_var(in_idx)),
                                     c_index(c_var(weight), c_var(w_idx))))])])])])].


%% conv1d_accum(-Stmts)
%% Double-nested ci/k loop for 1D convolution.
conv1d_accum(Stmts) :-
    Stmts = [c_for(
        c_decl_init(c_type(int), ci, c_int(0)),
        c_binop('<', c_var(ci), c_var('C_in')),
        c_pre_inc(c_var(ci)),
        [c_for(
            c_decl_init(c_type(int), k, c_int(0)),
            c_binop('<', c_var(k), c_var(kL)),
            c_pre_inc(c_var(k)),
            [c_decl_init(c_type(int), li,
                c_binop('+', c_binop('-', c_binop('*', c_var(lo), c_var(stride)), c_var(pad)),
                             c_binop('*', c_var(k), c_var(dilation)))),
             c_if(c_binop('&&', c_binop('>=', c_var(li), c_int(0)),
                                c_binop('<', c_var(li), c_var('L_in'))),
                [c_compound_assign('+=', c_var(sum),
                    c_binop('*',
                        c_index(c_var(input), c_nd_index([n, 'C_in', ci, 'L_in', li])),
                        c_index(c_var(weight), c_nd_index([co, 'C_in', ci, kL, k]))))])])])].


%% depthwise_conv2d_accum(-Stmts)
%% Double-nested kh/kw loop (no ci — each channel is independent).
depthwise_conv2d_accum(Stmts) :-
    Stmts = [c_for(
        c_decl_init(c_type(int), kh, c_int(0)),
        c_binop('<', c_var(kh), c_var(kH)),
        c_pre_inc(c_var(kh)),
        [c_for(
            c_decl_init(c_type(int), kw, c_int(0)),
            c_binop('<', c_var(kw), c_var(kW)),
            c_pre_inc(c_var(kw)),
            [c_decl_init(c_type(int), hi,
                c_binop('+', c_binop('-', c_binop('*', c_var(ho), c_var(stride_h)), c_var(pad_h)),
                             c_var(kh))),
             c_decl_init(c_type(int), wi,
                c_binop('+', c_binop('-', c_binop('*', c_var(wo), c_var(stride_w)), c_var(pad_w)),
                             c_var(kw))),
             c_if(c_binop('&&',
                    c_binop('&&', c_binop('>=', c_var(hi), c_int(0)),
                                  c_binop('<', c_var(hi), c_var('H_in'))),
                    c_binop('&&', c_binop('>=', c_var(wi), c_int(0)),
                                  c_binop('<', c_var(wi), c_var('W_in')))),
                [c_compound_assign('+=', c_var(sum),
                    c_binop('*',
                        c_index(c_var(input), c_nd_index([n, 'C', c, 'H_in', hi, 'W_in', wi])),
                        c_index(c_var(weight), c_nd_index([c, kH, kh, kW, kw]))))])])])].


%% =============================================================================
%% SGEMV — SUBSTRATE-NATIVE BASELINE (warp-shuffle, 32-thread, simple stride)
%% =============================================================================
%%
%% The current "best substrate effort" form: 32 threads per row, warp-shuffle
%% reduction, simple stride-32 accumulation over K. This is mavchin's
%% empirically-tested kernel that produced the 12 ULP gap against cuBLAS.
%%
%% Useful as the substrate's reference point and as a faster-than-cuBLAS
%% candidate once we measure both on the same workload (warp-shuffle is
%% lower latency than shared-mem reduction; the bit cost is the 12 ULP).
%%
%% Signature:
%%   __global__ void k_sgemv_substrate_native(const float * A,
%%                                              const float * x,
%%                                              float * y,
%%                                              int M, int K);
%%
%% Launch: <<<M, 32>>>   (one warp per row, 32 threads per warp)


%% ═══════════════════════════════════════════════════════════════
%% block_reduce_sum(Arr, Size, -Stmt)
%% Emits: for(int s=Size/2; s>0; s>>=1) { if(tid<s) Arr[tid]+=Arr[tid+s]; __syncthreads(); }
%% Pure structural c_ast — uses c_for_step + c_compound_assign('>>=', ...) per fa9c27e.
%% ═══════════════════════════════════════════════════════════════

%% block_reduce_sum — pure c_ast, zero c_raw.
%% Emits: for(int s=Size/2; s>0; s>>=1) { if(tid<s) Arr[tid]+=Arr[tid+s]; __syncthreads(); }
block_reduce_sum(Arr, Size, Stmt) :-
    Half is Size // 2,
    Stmt = c_for_step(
        c_decl_init(c_type(int), s, c_int(Half)),
        c_binop('>', c_var(s), c_int(0)),
        c_compound_assign('>>=', c_var(s), c_int(1)),
        [c_if(c_binop('<', c_var(tid), c_var(s)),
              [c_compound_assign('+=',
                  c_index(c_var(Arr), c_var(tid)),
                  c_index(c_var(Arr), c_binop('+', c_var(tid), c_var(s))))]),
         c_syncthreads]).

sgemv_kernel_substrate_native(KName, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'A'),
         param(c_type(const_restrict_ptr(c_type(float))), x),
         param(c_type(restrict_ptr(c_type(float))), y),
         param(c_type(int), 'M'),
         param(c_type(int), 'K')],
        [c_decl_init(c_type(int), row, c_member(c_var(blockIdx), x)),
         c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_if(c_binop('>=', c_var(row), c_var('M')), [c_return_void]),
         %% Sequential accumulation, stride-32 across K
         c_decl_init(c_type(float), p, c_float_f(0.0)),
         c_for(c_decl_init(c_type(int), j, c_var(tid)),
               c_binop('<', c_var(j), c_var('K')),
               c_compound_step(c_var(j), '+=', c_int(32)),
               [c_assign(c_var(p),
                   c_binop('+', c_var(p),
                       c_binop('*',
                           c_index(c_var('A'),
                               c_binop('+',
                                   c_binop('*', c_var(row), c_var('K')),
                                   c_var(j))),
                           c_index(c_var(x), c_var(j)))))]),
         %% Warp-shuffle reduction (5 steps: xor 16, 8, 4, 2, 1)
         c_for(c_decl_init(c_type(int), o, c_int(16)),
               c_binop('>', c_var(o), c_int(0)),
               c_compound_step(c_var(o), '>>=', c_int(1)),
               [c_assign(c_var(p),
                   c_binop('+', c_var(p),
                       c_call('__shfl_xor_sync',
                           [c_hex(0xffffffff),
                            c_var(p), c_var(o), c_int(32)])))]),
         %% Lane 0 writes the result
         c_if(c_binop('==', c_var(tid), c_int(0)),
              [c_assign(c_index(c_var(y), c_var(row)), c_var(p))])]).


sgemv_wrapper_substrate_native(KName, Wrapper) :-
    %% Derive wrapper name from kernel name: k_<suffix> → gpu_<suffix>
    %% Matches the CFD wrapper convention.
    atom_concat('k_', Suffix, KName),
    atom_concat('gpu_', Suffix, WName),
    Wrapper = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'A'),
         param(c_type(const_restrict_ptr(c_type(float))), x),
         param(c_type(restrict_ptr(c_type(float))), y),
         param(c_type(int), 'M'),
         param(c_type(int), 'K')],
        [%% Launch: <<<M, 32>>> — one warp per row
         c_cuda_launch(KName, c_var('M'), c_int(32),
             [c_var('A'), c_var(x), c_var(y), c_var('M'), c_var('K')])]).


%% =============================================================================
%% SGEMV — CUBLAS-MATCHING (Mode 1 strict subsumption)
%% =============================================================================
%%
%% Per mavchin's COMPREHENSIVE EMPIRICAL SWEEP 2026-05-18 ~22:36 UTC:
%%
%%   stride-32 + warp shuffle:    12 ULP ← CLOSEST to cuBLAS
%%   stride-16 + warp shuffle:    48 ULP
%%   stride-16 + shared-mem:      48 ULP ← SAME as stride-16 shuffle
%%   contiguous-16 chunks:        176 ULP
%%   1-thread sequential:         179 ULP
%%
%% SUBSTANTIVELY IMPORTANT FINDINGS:
%%
%%   1. The reduction tree shape (warp shuffle vs shared-mem strided pair)
%%      is BIT-INVARIANT at this kernel scale. Both reductions converge
%%      on the same lane-0 result for any given per-thread partial sums.
%%      The earlier hypothesis "shared-mem reduction closes the gap" was
%%      empirically disproved.
%%
%%   2. cuBLAS uses STRIDE-32, not stride-16. The substrate-native kernel
%%      structure (32 threads per warp, stride-32 accumulation) IS the
%%      correct algorithm-level emit. The template params <128,16,2,2>
%%      do NOT decode to "16 threads per row" — they likely encode tile
%%      dimensions or vectorization, not the per-row thread count.
%%
%%   3. The 12 ULP remaining gap is NOT in the kernel's algorithm-level
%%      structure. It's somewhere below — possibly nvcc compilation flags,
%%      software pipelining, NVIDIA-internal toolchain differences, or
%%      subtle FMA scheduling. SASS-level investigation pending.
%%
%% SUBSTRATE-DESIGN POSITION:
%%
%% The cublas-match variant now uses the SAME stride-32 accumulation as
%% substrate-native (which empirically matches cuBLAS at 12 ULP). The
%% structural differences from substrate-native are:
%%
%%   - Shared-memory x preload (amortizes x access across multiple rows
%%     per block) — tick-determining, won't change bits, but matches
%%     cuBLAS's structural choice from the SASS LDG/LDS pattern
%%   - 4 rows per block × 32 threads per row = 128 threads per block
%%     (matches cuBLAS's <128, ...> template parameter)
%%   - Warp-shuffle reduction (works because stride-32 means each row's
%%     32 threads are exactly one warp — warp primitives apply naturally)
%%
%% The 12 ULP residual will be verified equal to substrate-native's 12 ULP
%% in mavchin's next empirical run. If yes, the gap is confirmed as
%% compilation-level rather than substrate-level.
%%
%% Launch: <<<(M + 3) / 4, 128, K * 4>>>
%%   Grid: ceil(M / 4) blocks
%%   Block: 128 threads (4 rows × 32 threads each)
%%   Shared mem: K * sizeof(float) = K * 4 bytes for sx[]

sgemv_kernel_cublas_match(KName, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'A'),
         param(c_type(const_restrict_ptr(c_type(float))), x),
         param(c_type(restrict_ptr(c_type(float))), y),
         param(c_type(int), 'M'),
         param(c_type(int), 'K')],
        [%% Shared memory: x preload buffer.
         %% No sred[] needed — warp shuffle handles the reduction.
         c_extern_shared(c_type(float), smem),
         c_decl_init(c_type(ptr(c_type(float))), sx, c_var(smem)),

         %% Thread indexing
         c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_decl_init(c_type(int), bid, c_member(c_var(blockIdx), x)),
         %% 32 threads per row × 4 rows per block.
         %% Each row's 32 threads form one warp, enabling warp shuffle.
         c_decl_init(c_type(int), row_in_block,
             c_binop('/', c_var(tid), c_int(32))),
         c_decl_init(c_type(int), tid_in_row,
             c_binop('%', c_var(tid), c_int(32))),
         c_decl_init(c_type(int), row,
             c_binop('+',
                 c_binop('*', c_var(bid), c_int(4)),
                 c_var(row_in_block))),

         %% Cooperative x preload: 128 threads load K elements.
         c_for(c_decl_init(c_type(int), j, c_var(tid)),
               c_binop('<', c_var(j), c_var('K')),
               c_compound_step(c_var(j), '+=', c_int(128)),
               [c_assign(c_index(c_var(sx), c_var(j)),
                   c_index(c_var(x), c_var(j)))]),
         c_syncthreads,

         %% NOTE: We do NOT early-return for rows >= M.
         %% All threads must reach every barrier in this kernel.
         %% The syncthreads above is per-block; threads past M still
         %% participated in the cooperative x preload. Below, the
         %% warp shuffle is per-warp (32 threads of one row), so threads
         %% in invalid rows just contribute 0 to their warp's reduction.
         c_decl_init(c_type(int), row_valid,
             c_binop('<', c_var(row), c_var('M'))),

         %% Stride-32 per-thread accumulation — SAME pattern as
         %% substrate-native. The bit-determining per-thread FMA chain
         %% is identical between the two kernels.
         %%
         %% For K=256 with 32 threads per row:
         %%   Thread 0: elements [0, 32, 64, 96, 128, 160, 192, 224] (8 FMAs)
         %%   Thread 1: elements [1, 33, 65, 97, ..., 225]
         %%   ...
         %%   Thread 31: elements [31, 63, 95, ..., 255]
         %%
         %% Reads A from global memory, sx from shared (cuBLAS-structural
         %% choice — won't change bits per mavchin's sweep, but matches
         %% the LDS pattern from cuBLAS SASS).
         c_decl_init(c_type(float), p, c_float_f(0.0)),
         c_for(c_decl_init(c_type(int), j, c_var(tid_in_row)),
               c_binop('<', c_var(j), c_var('K')),
               c_compound_step(c_var(j), '+=', c_int(32)),
               [c_if(c_var(row_valid),
                     [c_assign(c_var(p),
                          c_binop('+', c_var(p),
                              c_binop('*',
                                  c_index(c_var('A'),
                                      c_binop('+',
                                          c_binop('*',
                                              c_var(row), c_var('K')),
                                          c_var(j))),
                                  c_index(c_var(sx),
                                      c_var(j)))))])]),

         %% Warp-shuffle reduction within each row's warp (32 threads).
         %% This is the SAME reduction as substrate-native, just operating
         %% on the row's own warp instead of the whole block.
         %% Per mavchin's empirical sweep: bit-equivalent to shared-mem
         %% reduction at this stride.
         c_for(c_decl_init(c_type(int), o, c_int(16)),
               c_binop('>', c_var(o), c_int(0)),
               c_compound_step(c_var(o), '>>=', c_int(1)),
               [c_assign(c_var(p),
                   c_binop('+', c_var(p),
                       c_call('__shfl_xor_sync',
                           [c_hex(0xffffffff),
                            c_var(p), c_var(o), c_int(32)])))]),

         %% Thread 0 of each valid row writes the result
         c_if(c_binop('&&',
                  c_var(row_valid),
                  c_binop('==', c_var(tid_in_row), c_int(0))),
              [c_assign(c_index(c_var(y), c_var(row)), c_var(p))])]).




sgemv_wrapper_cublas_match(KName, Wrapper) :-
    %% Derive wrapper name from kernel name: k_<suffix> → gpu_<suffix>
    atom_concat('k_', Suffix, KName),
    atom_concat('gpu_', Suffix, WName),
    Wrapper = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'A'),
         param(c_type(const_restrict_ptr(c_type(float))), x),
         param(c_type(restrict_ptr(c_type(float))), y),
         param(c_type(int), 'M'),
         param(c_type(int), 'K')],
        [%% Launch: 4 rows per block × 32 threads per row = 128 threads.
         %% Grid = ceil(M / 4).
         %% Shared memory = K * sizeof(float) = K * 4 bytes for sx[]
         %% (No sred[] — warp shuffle handles reduction per row's warp.)
         c_cuda_launch(KName,
             c_binop('/',
                 c_paren(c_binop('+', c_var('M'), c_int(3))),
                 c_int(4)),
             c_int(128),
             c_binop('*', c_var('K'), c_int(4)),
             [c_var('A'), c_var(x), c_var(y), c_var('M'), c_var('K')])]).


%% =============================================================================
%% ELEMENTWISE KERNEL FACTORY — One Template, Many Facts
%% =============================================================================
%%
%% THE UNIVERSAL ELEMENTWISE KERNEL:
%%   Every elementwise GPU kernel has the same structure:
%%     1. Compute thread index
%%     2. Bounds check
%%     3. Load input(s)
%%     4. Apply operation    ← THIS is the BPD fact
%%     5. Store output
%%
%% The BPD fact `elem_op/4` defines the operation:
%%   elem_op(Name, InputParams, OutputVar, Expr)
%%
%% The factory generates a complete __global__ kernel from one fact.
%% One template subsumes: activation functions, arithmetic ops,
%% BLAS L1 elementwise (saxpy, sscal), and any future pointwise op.
%%
%% This is the pattern that subsumes whole industries:
%%   Each new operation = one new fact = one new line of Prolog.
%%   The template handles everything else.

%% ── The BPD facts ──

%% BLAS L1 elementwise
elem_op(k_saxpy,
    [param(c_type(int), n),
     param(c_type(float), alpha),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_binop('+',
        c_binop('*', c_var(alpha), c_index(c_var(x), c_var(i))),
        c_index(c_var(y), c_var(i)))).

elem_op(k_sscal,
    [param(c_type(int), n),
     param(c_type(float), alpha),
     param(c_type(restrict_ptr(c_type(float))), x)],
    c_index(c_var(x), c_var(i)),
    c_binop('*', c_var(alpha), c_index(c_var(x), c_var(i)))).

elem_op(k_scopy,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_index(c_var(x), c_var(i))).

%% k_scale — out-of-place scalar multiply. Distinct from k_sscal
%% (which is in-place x[i] *= alpha). Substrate-honest lift of
%% k_scale from bpd/llamatov_kernels.cu:53-56. Substrate-debt
%% retirement: the hand-written k_scale was a parallel implementation
%% of what elem_op already expresses; this fact subsumes it.
elem_op(k_scale,
    [param(c_type(const_restrict_ptr(c_type(float))), in),
     param(c_type(restrict_ptr(c_type(float))), out),
     param(c_type(float), s),
     param(c_type(int), n)],
    c_index(c_var(out), c_var(i)),
    c_binop('*', c_index(c_var(in), c_var(i)), c_var(s))).

%% Unary activations (can also be expressed here alongside activation_expr/3)
elem_op(k_silu_blas,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_binop('/', c_index(c_var(x), c_var(i)),
        c_paren(c_binop('+', c_float_f(1.0),
            c_call(expf, [c_unop('-', c_index(c_var(x), c_var(i)))]))))).

elem_op(k_relu_blas,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(fmaxf, [c_float_f(0.0), c_index(c_var(x), c_var(i))])).

%% Binary arithmetic
elem_op(k_vadd,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), a),
     param(c_type(const_restrict_ptr(c_type(float))), b),
     param(c_type(restrict_ptr(c_type(float))), c)],
    c_index(c_var(c), c_var(i)),
    c_binop('+', c_index(c_var(a), c_var(i)),
                 c_index(c_var(b), c_var(i)))).

elem_op(k_vmul,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), a),
     param(c_type(const_restrict_ptr(c_type(float))), b),
     param(c_type(restrict_ptr(c_type(float))), c)],
    c_index(c_var(c), c_var(i)),
    c_binop('*', c_index(c_var(a), c_var(i)),
                 c_index(c_var(b), c_var(i)))).

%% ── Binary arithmetic (continued) ──

elem_op(k_vsub,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), a),
     param(c_type(const_restrict_ptr(c_type(float))), b),
     param(c_type(restrict_ptr(c_type(float))), c)],
    c_index(c_var(c), c_var(i)),
    c_binop('-', c_index(c_var(a), c_var(i)),
                 c_index(c_var(b), c_var(i)))).

elem_op(k_vdiv,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), a),
     param(c_type(const_restrict_ptr(c_type(float))), b),
     param(c_type(restrict_ptr(c_type(float))), c)],
    c_index(c_var(c), c_var(i)),
    c_binop('/', c_index(c_var(a), c_var(i)),
                 c_index(c_var(b), c_var(i)))).

elem_op(k_vmax,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), a),
     param(c_type(const_restrict_ptr(c_type(float))), b),
     param(c_type(restrict_ptr(c_type(float))), c)],
    c_index(c_var(c), c_var(i)),
    c_call(fmaxf, [c_index(c_var(a), c_var(i)),
                   c_index(c_var(b), c_var(i))])).

elem_op(k_vmin,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), a),
     param(c_type(const_restrict_ptr(c_type(float))), b),
     param(c_type(restrict_ptr(c_type(float))), c)],
    c_index(c_var(c), c_var(i)),
    c_call(fminf, [c_index(c_var(a), c_var(i)),
                   c_index(c_var(b), c_var(i))])).

%% ── Unary ops (BLAS convention: n, x, y) ──

elem_op(k_vneg,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_unop('-', c_index(c_var(x), c_var(i)))).

elem_op(k_vabs,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(fabsf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vsqr,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_binop('*', c_index(c_var(x), c_var(i)),
                 c_index(c_var(x), c_var(i)))).

elem_op(k_vsqrt,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(sqrtf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vrsqrt,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(rsqrtf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vexp,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(expf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vlog,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(logf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vsigmoid,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_binop('/', c_float_f(1.0),
        c_paren(c_binop('+', c_float_f(1.0),
            c_call(expf, [c_unop('-', c_index(c_var(x), c_var(i)))]))))).

elem_op(k_vtanh,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(tanhf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vceil,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(ceilf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vfloor,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(floorf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vround,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(roundf, [c_index(c_var(x), c_var(i))])).

elem_op(k_vsign,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(copysignf, [c_float_f(1.0), c_index(c_var(x), c_var(i))])).

%% ── Scalar-vector ops ──

elem_op(k_vscale,
    [param(c_type(int), n),
     param(c_type(float), alpha),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_binop('*', c_var(alpha), c_index(c_var(x), c_var(i)))).

elem_op(k_vaddscalar,
    [param(c_type(int), n),
     param(c_type(float), alpha),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_binop('+', c_var(alpha), c_index(c_var(x), c_var(i)))).

%% ── Fused ops (two operations, one kernel, no DRAM round-trip) ──

elem_op(k_silu_mul,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), gate),
     param(c_type(const_restrict_ptr(c_type(float))), up),
     param(c_type(restrict_ptr(c_type(float))), out)],
    c_index(c_var(out), c_var(i)),
    c_binop('*',
        c_binop('/', c_index(c_var(gate), c_var(i)),
            c_paren(c_binop('+', c_float_f(1.0),
                c_call(expf, [c_unop('-', c_index(c_var(gate), c_var(i)))])))),
        c_index(c_var(up), c_var(i)))).

elem_op(k_add_relu,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), a),
     param(c_type(const_restrict_ptr(c_type(float))), b),
     param(c_type(restrict_ptr(c_type(float))), c)],
    c_index(c_var(c), c_var(i)),
    c_call(fmaxf, [c_float_f(0.0),
        c_binop('+', c_index(c_var(a), c_var(i)),
                     c_index(c_var(b), c_var(i)))])).

elem_op(k_gelu_blas,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_binop('*', c_binop('*', c_float_f(0.5), c_index(c_var(x), c_var(i))),
        c_paren(c_binop('+', c_float_f(1.0),
            c_call(tanhf, [c_binop('*', c_float_f(0.7978845608),
                c_paren(c_binop('+', c_index(c_var(x), c_var(i)),
                    c_binop('*', c_float_f(0.044715),
                        c_binop('*', c_index(c_var(x), c_var(i)),
                            c_binop('*', c_index(c_var(x), c_var(i)),
                                         c_index(c_var(x), c_var(i))))))))]))))).

%% ── Activation: HardTanh (KernelBench L1 #32) ──

elem_op(k_hardtanh,
    [param(c_type(int), n),
     param(c_type(float), min_val),
     param(c_type(float), max_val),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_call(fminf, [c_var(max_val),
        c_call(fmaxf, [c_var(min_val),
            c_index(c_var(x), c_var(i))])])).

%% ── Losses: Huber, KLDiv (KernelBench L1 #96, #98) ──

elem_op(k_huber_elem,
    [param(c_type(int), n),
     param(c_type(float), delta),
     param(c_type(const_restrict_ptr(c_type(float))), pred),
     param(c_type(const_restrict_ptr(c_type(float))), target),
     param(c_type(restrict_ptr(c_type(float))), loss)],
    c_index(c_var(loss), c_var(i)),
    c_ternary(
        c_binop('<',
            c_call(fabsf, [c_binop('-', c_index(c_var(pred), c_var(i)),
                                        c_index(c_var(target), c_var(i)))]),
            c_var(delta)),
        %% |diff| < delta: 0.5 * diff^2 (quadratic region)
        c_binop('*', c_float_f(0.5),
            c_binop('*',
                c_binop('-', c_index(c_var(pred), c_var(i)), c_index(c_var(target), c_var(i))),
                c_binop('-', c_index(c_var(pred), c_var(i)), c_index(c_var(target), c_var(i))))),
        %% |diff| >= delta: delta * (|diff| - 0.5 * delta) (linear region)
        c_binop('*', c_var(delta),
            c_binop('-',
                c_call(fabsf, [c_binop('-', c_index(c_var(pred), c_var(i)),
                                            c_index(c_var(target), c_var(i)))]),
                c_binop('*', c_float_f(0.5), c_var(delta)))))).

elem_op(k_kldiv_elem,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), p),
     param(c_type(const_restrict_ptr(c_type(float))), q),
     param(c_type(restrict_ptr(c_type(float))), loss)],
    c_index(c_var(loss), c_var(i)),
    c_binop('*', c_index(c_var(p), c_var(i)),
        c_binop('-',
            c_call(logf, [c_binop('+', c_index(c_var(p), c_var(i)), c_float_f(1.0e-7))]),
            c_call(logf, [c_binop('+', c_index(c_var(q), c_var(i)), c_float_f(1.0e-7))])))).

elem_op(k_hinge_elem,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), pred),
     param(c_type(const_restrict_ptr(c_type(float))), target),
     param(c_type(restrict_ptr(c_type(float))), loss)],
    c_index(c_var(loss), c_var(i)),
    c_call(fmaxf, [c_float_f(0.0),
        c_binop('-', c_float_f(1.0),
            c_binop('*', c_index(c_var(pred), c_var(i)),
                         c_index(c_var(target), c_var(i))))])).

%% ── Stanford KernelBench L1 gap closers ──

%% KB_L1_cross_entropy element: target[i] * log(softmax_output[i])
%% (the reduction is separate — this is the per-element part)
elem_op(k_cross_entropy_elem,
    [param(c_type(int), n),
     param(c_type(const_restrict_ptr(c_type(float))), pred),
     param(c_type(const_restrict_ptr(c_type(float))), target),
     param(c_type(restrict_ptr(c_type(float))), loss)],
    c_index(c_var(loss), c_var(i)),
    c_unop('-', c_binop('*',
        c_index(c_var(target), c_var(i)),
        c_call(logf, [c_binop('+',
            c_index(c_var(pred), c_var(i)),
            c_float_f(1.0e-7))])))).

%% KB_L1_linear bias add: y[i] = matmul_result[i] + bias[i % bias_dim]
%% (the matmul is separate — this adds the bias vector, broadcasting)
elem_op(k_bias_add,
    [param(c_type(int), n),
     param(c_type(int), bias_dim),
     param(c_type(const_restrict_ptr(c_type(float))), x),
     param(c_type(const_restrict_ptr(c_type(float))), bias),
     param(c_type(restrict_ptr(c_type(float))), y)],
    c_index(c_var(y), c_var(i)),
    c_binop('+', c_index(c_var(x), c_var(i)),
        c_index(c_var(bias), c_binop('%', c_var(i), c_var(bias_dim))))).

%% ── Pooling kernels (Stanford KernelBench L1) ──
%% These need 2D indexing with window reduction — a new pattern.
%% Input: (N, C, H, W), kernel_size, stride
%% Output: (N, C, H_out, W_out)

%% MaxPool2D: output = max over (kernel_size × kernel_size) window
pool_kernel(k_maxpool2d, max, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_maxpool2d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'C'), param(c_type(int), 'H'), param(c_type(int), 'W'),
         param(c_type(int), ksize), param(c_type(int), stride),
         param(c_type(int), 'H_out'), param(c_type(int), 'W_out')],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('C'), c_binop('*', c_var('H_out'), c_var('W_out')))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         %% Index decomposition: idx -> (w_out, h_out, c)
         c_decl_init(c_type(int), w_out,
            c_binop('%', c_var(idx), c_var('W_out'))),
         c_decl_init(c_type(int), h_out,
            c_binop('%', c_binop('/', c_var(idx), c_var('W_out')), c_var('H_out'))),
         c_decl_init(c_type(int), c,
            c_binop('/', c_var(idx), c_binop('*', c_var('W_out'), c_var('H_out')))),
         %% Accumulator
         c_decl_init(c_type(float), val, c_float_f(-1.0e30)),
         %% Nested loop: for kh, kw
         c_for(c_decl_init(c_type(int), kh, c_int(0)),
               c_binop('<', c_var(kh), c_var(ksize)),
               c_unop('++', c_var(kh)),
           [c_for(c_decl_init(c_type(int), kw, c_int(0)),
                  c_binop('<', c_var(kw), c_var(ksize)),
                  c_unop('++', c_var(kw)),
              [c_decl_init(c_type(int), h_in,
                  c_binop('+', c_binop('*', c_var(h_out), c_var(stride)), c_var(kh))),
               c_decl_init(c_type(int), w_in,
                  c_binop('+', c_binop('*', c_var(w_out), c_var(stride)), c_var(kw))),
               c_if(c_binop('&&',
                       c_binop('<', c_var(h_in), c_var('H')),
                       c_binop('<', c_var(w_in), c_var('W'))),
                  [c_decl_init(c_type(float), v,
                      c_index(c_var(input),
                          c_binop('+',
                              c_binop('*', c_var(c), c_binop('*', c_var('H'), c_var('W'))),
                              c_binop('+', c_binop('*', c_var(h_in), c_var('W')), c_var(w_in))))),
                   c_if(c_binop('>', c_var(v), c_var(val)),
                      [c_assign(c_var(val), c_var(v))])])])]),
         %% Write output
         c_assign(c_index(c_var(output), c_var(idx)), c_var(val))]).

%% AvgPool2D: output = mean over (kernel_size × kernel_size) window
pool_kernel(k_avgpool2d, avg, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_avgpool2d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'C'), param(c_type(int), 'H'), param(c_type(int), 'W'),
         param(c_type(int), ksize), param(c_type(int), stride),
         param(c_type(int), 'H_out'), param(c_type(int), 'W_out')],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('C'), c_binop('*', c_var('H_out'), c_var('W_out')))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         %% Index decomposition: idx -> (w_out, h_out, c)
         c_decl_init(c_type(int), w_out,
            c_binop('%', c_var(idx), c_var('W_out'))),
         c_decl_init(c_type(int), h_out,
            c_binop('%', c_binop('/', c_var(idx), c_var('W_out')), c_var('H_out'))),
         c_decl_init(c_type(int), c,
            c_binop('/', c_var(idx), c_binop('*', c_var('W_out'), c_var('H_out')))),
         %% Accumulators
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         c_decl_init(c_type(int), count, c_int(0)),
         %% Nested loop: for kh, kw
         c_for(c_decl_init(c_type(int), kh, c_int(0)),
               c_binop('<', c_var(kh), c_var(ksize)),
               c_unop('++', c_var(kh)),
           [c_for(c_decl_init(c_type(int), kw, c_int(0)),
                  c_binop('<', c_var(kw), c_var(ksize)),
                  c_unop('++', c_var(kw)),
              [c_decl_init(c_type(int), h_in,
                  c_binop('+', c_binop('*', c_var(h_out), c_var(stride)), c_var(kh))),
               c_decl_init(c_type(int), w_in,
                  c_binop('+', c_binop('*', c_var(w_out), c_var(stride)), c_var(kw))),
               c_if(c_binop('&&',
                       c_binop('<', c_var(h_in), c_var('H')),
                       c_binop('<', c_var(w_in), c_var('W'))),
                  [c_assign(c_var(sum),
                      c_binop('+', c_var(sum),
                          c_index(c_var(input),
                              c_binop('+',
                                  c_binop('*', c_var(c), c_binop('*', c_var('H'), c_var('W'))),
                                  c_binop('+', c_binop('*', c_var(h_in), c_var('W')), c_var(w_in)))))),
                   c_assign(c_var(count),
                      c_binop('+', c_var(count), c_int(1)))])])]),
         %% Write output: sum / (float)count
         c_assign(c_index(c_var(output), c_var(idx)),
            c_binop('/', c_var(sum), c_cast(c_type(float), c_var(count))))]).

%% ══════════════════════════════════════════════════════════════
%% CONVOLUTION KERNEL FAMILY — Stanford KernelBench L1 #50-87
%% ══════════════════════════════════════════════════════════════
%%
%% ONE template generates ALL 22 convolution variants via parameters:
%%   conv_kernel(+KName, +Dims, +Direction, +Groups, -Kernel)
%%
%%   Dims:      1 | 2 | 3         (spatial dimensions)
%%   Direction: forward | transposed
%%   Groups:    1 (standard) | C_in (depthwise) | N (grouped)
%%
%% All variants share the same structure:
%%   1. Compute output spatial index from thread ID
%%   2. Loop over kernel window
%%   3. Accumulate weighted sum (+ bias optionally)
%%   4. Write output
%%
%% The input/kernel shape (square vs asymmetric) is handled by
%% runtime parameters (H, W, kH, kW), not template variants.
%% Padding, stride, dilation are runtime parameters too.

%% ── Conv2D: the base case ──

conv_kernel(k_conv2d, 2, forward, 1, Kernel) :-
    conv_kernel_with_epilogue(k_conv2d, 2, forward, 1, [], Kernel).

%% conv_kernel_with_epilogue/6: conv2d forward with an optional epilogue chain.
%%
%% Modeled on the L2 #76 matmul+bias+relu fusion pattern
%% (lib/epilogue_generator.pl's chain_ops + epilogue_for_chain).
%%
%%   conv_kernel_with_epilogue(k_conv2d, 2, forward, 1, [], K).
%%   %  Plain conv2d (delegated to here from conv_kernel/5; backward-compat).
%%
%%   conv_kernel_with_epilogue(k_conv2d, 2, forward, 1, [bn_affine_fused, mish], K).
%%   %  YOLOv4 CBA (Conv + BN-eval + Mish) as a single kernel.
%%
%%   conv_kernel_with_epilogue(k_conv2d, 2, forward, 1, [bn_affine_fused, silu], K).
%%   %  YOLOv5 CBA (Conv + BN-eval + SiLU) — the orchestrator's CPU-then-GPU
%%   %  integration target.
%%
%%   conv_kernel_with_epilogue(k_conv2d, 2, forward, 1, [bias_add, relu], K).
%%   %  Classic Conv+bias+ReLU (non-BN models).
%%
%% Variable-name alignment: bn_affine_fused references c_var(c_out) for
%% per-output-channel indexing. The conv kernel uses 'co' for output channel.
%% The body emits `c_decl_init c_out = co` before the epilogue so both
%% naming domains compose cleanly.
conv_kernel_with_epilogue(k_conv2d, 2, forward, 1, Epilogue, Kernel) :-
    standard_conv2d_params(BaseParams),
    epilogue_extra_params(Epilogue, ExtraParams),
    append(BaseParams, ExtraParams, AllParams),
    standard_conv2d_body_with_epilogue(Epilogue, Body),
    Kernel = c_func(['__global__'], c_type(void), k_conv2d, AllParams, Body).

%% standard_conv2d_params/1: canonical 19-parameter conv2d signature.
standard_conv2d_params([
    param(c_type(const_restrict_ptr(c_type(float))), input),
    param(c_type(const_restrict_ptr(c_type(float))), weight),
    param(c_type(const_restrict_ptr(c_type(float))), bias),
    param(c_type(restrict_ptr(c_type(float))), output),
    param(c_type(int), 'N'), param(c_type(int), 'C_in'),
    param(c_type(int), 'H_in'), param(c_type(int), 'W_in'),
    param(c_type(int), 'C_out'),
    param(c_type(int), 'H_out'), param(c_type(int), 'W_out'),
    param(c_type(int), 'kH'), param(c_type(int), 'kW'),
    param(c_type(int), stride_h), param(c_type(int), stride_w),
    param(c_type(int), pad_h), param(c_type(int), pad_w),
    param(c_type(int), dil_h), param(c_type(int), dil_w)
]).

%% epilogue_extra_params/2: extra kernel parameters required by the chain.
%% bn_affine_fused needs precomputed bn_scale and bn_offset arrays.
epilogue_extra_params([], []).
epilogue_extra_params(Epilogue, [
    param(c_type(const_restrict_ptr(c_type(float))), bn_scale),
    param(c_type(const_restrict_ptr(c_type(float))), bn_offset)
]) :-
    member(bn_affine_fused, Epilogue), !.
epilogue_extra_params(_, []).

%% standard_conv2d_body_with_epilogue/2: conv2d kernel body. The conv loop
%% computes `sum`; then either:
%%   - empty Epilogue: bias add + store (matches the historical body).
%%   - non-empty Epilogue: alias `co` -> `c_out`, chain_ops(Epilogue, sum, Expr),
%%                          store the chain result.
standard_conv2d_body_with_epilogue(Epilogue, Body) :-
    Header = [
        c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
        c_decl_init(c_type(int), total,
            c_binop('*', c_var('N'),
                c_binop('*', c_var('C_out'),
                    c_binop('*', c_var('H_out'), c_var('W_out'))))),
        c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
        c_decl_init(c_type(int), wo, c_binop('%', c_var(idx), c_var('W_out'))),
        c_decl_init(c_type(int), ho, c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('W_out'))), c_var('H_out'))),
        c_decl_init(c_type(int), co, c_binop('%', c_paren(c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_var('H_out'))))), c_var('C_out'))),
        %% Replaces the historical c_raw('int n = ...') with proper c_ast.
        c_decl_init(c_type(int), n,
            c_binop('/', c_var(idx),
                c_paren(c_binop('*', c_var('W_out'),
                    c_binop('*', c_var('H_out'), c_var('C_out')))))),
        c_decl_init(c_type(float), sum, c_float_f(0.0))
    ],
    conv2d_accum(AccumStmts),
    (   Epilogue == []
    ->  EpilogueStmts = [
            c_if(c_var(bias),
                 c_assign(c_var(sum),
                          c_binop('+', c_var(sum), c_index(c_var(bias), c_var(co))))),
            c_assign(c_index(c_var(output), c_var(idx)), c_var(sum))
        ]
    ;   chain_ops(Epilogue, c_var(sum), FusedExpr),
        EpilogueStmts = [
            c_decl_init(c_type(int), c_out, c_var(co)),
            c_assign(c_index(c_var(output), c_var(idx)), FusedExpr)
        ]
    ),
    append([Header, AccumStmts, EpilogueStmts], Body).

%% ── Conv1D: collapse spatial to 1D ──

conv_kernel(k_conv1d, 1, forward, 1, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_conv1d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(const_restrict_ptr(c_type(float))), weight),
         param(c_type(const_restrict_ptr(c_type(float))), bias),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'N'), param(c_type(int), 'C_in'),
         param(c_type(int), 'L_in'), param(c_type(int), 'C_out'),
         param(c_type(int), 'L_out'), param(c_type(int), 'kL'),
         param(c_type(int), stride), param(c_type(int), pad),
         param(c_type(int), dilation)],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('N'), c_binop('*', c_var('C_out'), c_var('L_out')))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         c_decl_init(c_type(int), lo, c_binop('%', c_var(idx), c_var('L_out'))),
         c_decl_init(c_type(int), co, c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('L_out'))), c_var('C_out'))),
         c_decl_init(c_type(int), n, c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('L_out'), c_var('C_out'))))),
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         %% Conv1d accumulation (generated by conv1d_accum/1)
         { conv1d_accum(ConvLoop1D) },
         ConvLoop1D,
         c_if(c_var(bias), c_assign(c_var(sum), c_binop('+', c_var(sum), c_index(c_var(bias), c_var(co))))),
         c_assign(c_index(c_var(output), c_var(idx)), c_var(sum))]).

%% ── Depthwise Conv2D: groups = C_in, no cross-channel mixing ──

conv_kernel(k_conv2d_depthwise, 2, forward, depthwise, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_conv2d_depthwise,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(const_restrict_ptr(c_type(float))), weight),
         param(c_type(const_restrict_ptr(c_type(float))), bias),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'N'), param(c_type(int), 'C'),
         param(c_type(int), 'H_in'), param(c_type(int), 'W_in'),
         param(c_type(int), 'H_out'), param(c_type(int), 'W_out'),
         param(c_type(int), 'kH'), param(c_type(int), 'kW'),
         param(c_type(int), stride_h), param(c_type(int), stride_w),
         param(c_type(int), pad_h), param(c_type(int), pad_w)],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('N'),
                c_binop('*', c_var('C'),
                    c_binop('*', c_var('H_out'), c_var('W_out'))))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         c_decl_init(c_type(int), wo, c_binop('%', c_var(idx), c_var('W_out'))),
         c_decl_init(c_type(int), ho, c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('W_out'))), c_var('H_out'))),
         c_decl_init(c_type(int), c, c_binop('%', c_paren(c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_var('H_out'))))), c_var('C'))),
         c_decl_init(c_type(int), n, c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_binop('*', c_var('H_out'), c_var('C')))))),
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         c_for(c_decl_init(c_type(int), kh, c_int(0)),
               c_binop('<', c_var(kh), c_var('kH')),
               c_assign(c_var(kh), c_binop('+', c_var(kh), c_int(1))),
               c_block([
                 c_for(c_decl_init(c_type(int), kw, c_int(0)),
                       c_binop('<', c_var(kw), c_var('kW')),
                       c_assign(c_var(kw), c_binop('+', c_var(kw), c_int(1))),
                       c_block([
                         c_decl_init(c_type(int), hi, c_binop('-', c_binop('+', c_binop('*', c_var(ho), c_var(stride_h)), c_var(kh)), c_var(pad_h))),
                         c_decl_init(c_type(int), wi, c_binop('-', c_binop('+', c_binop('*', c_var(wo), c_var(stride_w)), c_var(kw)), c_var(pad_w))),
                         c_if(c_binop('&&',
                                c_binop('&&', c_binop('>=', c_var(hi), c_int(0)), c_binop('<', c_var(hi), c_var('H_in'))),
                                c_binop('&&', c_binop('>=', c_var(wi), c_int(0)), c_binop('<', c_var(wi), c_var('W_in')))),
                              [c_assign(c_var(sum), c_binop('+', c_var(sum),
                                c_binop('*',
                                  c_index(c_var(input), c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(n), c_var('C')), c_var(c))), c_var('H_in')), c_var(hi))), c_var('W_in')), c_var(wi))),
                                  c_index(c_var(weight), c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(c), c_var('kH')), c_var(kh))), c_var('kW')), c_var(kw))))))])
                       ]))
               ])),
         c_if(c_var(bias), c_assign(c_var(sum), c_binop('+', c_var(sum), c_index(c_var(bias), c_var(c))))),
         c_assign(c_index(c_var(output), c_var(idx)), c_var(sum))]).

%% ── Transposed Conv2D (deconvolution) ──

conv_kernel(k_conv_transpose2d, 2, transposed, 1, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_conv_transpose2d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(const_restrict_ptr(c_type(float))), weight),
         param(c_type(const_restrict_ptr(c_type(float))), bias),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'N'), param(c_type(int), 'C_in'),
         param(c_type(int), 'H_in'), param(c_type(int), 'W_in'),
         param(c_type(int), 'C_out'),
         param(c_type(int), 'H_out'), param(c_type(int), 'W_out'),
         param(c_type(int), 'kH'), param(c_type(int), 'kW'),
         param(c_type(int), stride_h), param(c_type(int), stride_w),
         param(c_type(int), pad_h), param(c_type(int), pad_w)],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('N'),
                c_binop('*', c_var('C_out'),
                    c_binop('*', c_var('H_out'), c_var('W_out'))))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         %% Transposed conv: for each OUTPUT pixel, find which INPUT pixels contribute
         c_decl_init(c_type(int), wo, c_binop('%', c_var(idx), c_var('W_out'))),
         c_decl_init(c_type(int), ho, c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('W_out'))), c_var('H_out'))),
         c_decl_init(c_type(int), co, c_binop('%', c_paren(c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_var('H_out'))))), c_var('C_out'))),
         c_decl_init(c_type(int), n, c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_binop('*', c_var('H_out'), c_var('C_out')))))),
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         c_for(c_decl_init(c_type(int), ci, c_int(0)),
               c_binop('<', c_var(ci), c_var('C_in')),
               c_assign(c_var(ci), c_binop('+', c_var(ci), c_int(1))),
               c_block([
                 c_for(c_decl_init(c_type(int), kh, c_int(0)),
                       c_binop('<', c_var(kh), c_var('kH')),
                       c_assign(c_var(kh), c_binop('+', c_var(kh), c_int(1))),
                       c_block([
                         c_for(c_decl_init(c_type(int), kw, c_int(0)),
                               c_binop('<', c_var(kw), c_var('kW')),
                               c_assign(c_var(kw), c_binop('+', c_var(kw), c_int(1))),
                               c_block([
                                 c_decl_init(c_type(int), hi_num, c_binop('-', c_binop('+', c_var(ho), c_var(pad_h)), c_var(kh))),
                                 c_decl_init(c_type(int), wi_num, c_binop('-', c_binop('+', c_var(wo), c_var(pad_w)), c_var(kw))),
                                 c_if(c_binop('&&',
                                        c_binop('==', c_binop('%', c_var(hi_num), c_var(stride_h)), c_int(0)),
                                        c_binop('==', c_binop('%', c_var(wi_num), c_var(stride_w)), c_int(0))),
                                      [c_decl_init(c_type(int), hi, c_binop('/', c_var(hi_num), c_var(stride_h))),
                                       c_decl_init(c_type(int), wi, c_binop('/', c_var(wi_num), c_var(stride_w))),
                                       c_if(c_binop('&&',
                                              c_binop('&&', c_binop('>=', c_var(hi), c_int(0)), c_binop('<', c_var(hi), c_var('H_in'))),
                                              c_binop('&&', c_binop('>=', c_var(wi), c_int(0)), c_binop('<', c_var(wi), c_var('W_in')))),
                                            [c_decl_init(c_type(int), in_idx, c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(n), c_var('C_in')), c_var(ci))), c_var('H_in')), c_var(hi))), c_var('W_in')), c_var(wi))),
                                             c_decl_init(c_type(int), w_idx, c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(ci), c_var('C_out')), c_var(co))), c_var('kH')), c_var(kh))), c_var('kW')), c_var(kw))),
                                             c_assign(c_var(sum), c_binop('+', c_var(sum), c_binop('*', c_index(c_var(input), c_var(in_idx)), c_index(c_var(weight), c_var(w_idx)))))])])
                               ]))
                       ]))
               ])),
         c_if(c_var(bias), c_assign(c_var(sum), c_binop('+', c_var(sum), c_index(c_var(bias), c_var(co))))),
         c_assign(c_index(c_var(output), c_var(idx)), c_var(sum))]).

%% ── Conv3D: standard 3D convolution (KernelBench L1 #54,59,60,66) ──

conv_kernel(k_conv3d, 3, forward, 1, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_conv3d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(const_restrict_ptr(c_type(float))), weight),
         param(c_type(const_restrict_ptr(c_type(float))), bias),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'N'), param(c_type(int), 'C_in'),
         param(c_type(int), 'D_in'), param(c_type(int), 'H_in'), param(c_type(int), 'W_in'),
         param(c_type(int), 'C_out'),
         param(c_type(int), 'D_out'), param(c_type(int), 'H_out'), param(c_type(int), 'W_out'),
         param(c_type(int), 'kD'), param(c_type(int), 'kH'), param(c_type(int), 'kW'),
         param(c_type(int), stride_d), param(c_type(int), stride_h), param(c_type(int), stride_w),
         param(c_type(int), pad_d), param(c_type(int), pad_h), param(c_type(int), pad_w)],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('N'),
                c_binop('*', c_var('C_out'),
                    c_binop('*', c_var('D_out'),
                        c_binop('*', c_var('H_out'), c_var('W_out')))))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         c_decl_init(c_type(int), wo, c_binop('%', c_var(idx), c_var('W_out'))),
         c_decl_init(c_type(int), ho, c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('W_out'))), c_var('H_out'))),
         c_decl_init(c_type(int), do_, c_binop('%', c_paren(c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_var('H_out'))))), c_var('D_out'))),
         c_decl_init(c_type(int), co, c_binop('%', c_paren(c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_binop('*', c_var('H_out'), c_var('D_out')))))), c_var('C_out'))),
         c_decl_init(c_type(int), n, c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_binop('*', c_var('H_out'), c_binop('*', c_var('D_out'), c_var('C_out'))))))),
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         c_for(c_decl_init(c_type(int), ci, c_int(0)),
               c_binop('<', c_var(ci), c_var('C_in')),
               c_assign(c_var(ci), c_binop('+', c_var(ci), c_int(1))),
               c_block([
                 c_for(c_decl_init(c_type(int), kd, c_int(0)),
                       c_binop('<', c_var(kd), c_var('kD')),
                       c_assign(c_var(kd), c_binop('+', c_var(kd), c_int(1))),
                       c_block([
                         c_for(c_decl_init(c_type(int), kh, c_int(0)),
                               c_binop('<', c_var(kh), c_var('kH')),
                               c_assign(c_var(kh), c_binop('+', c_var(kh), c_int(1))),
                               c_block([
                                 c_for(c_decl_init(c_type(int), kw, c_int(0)),
                                       c_binop('<', c_var(kw), c_var('kW')),
                                       c_assign(c_var(kw), c_binop('+', c_var(kw), c_int(1))),
                                       c_block([
                                         c_decl_init(c_type(int), di, c_binop('-', c_binop('+', c_binop('*', c_var(do_), c_var(stride_d)), c_var(kd)), c_var(pad_d))),
                                         c_decl_init(c_type(int), hi, c_binop('-', c_binop('+', c_binop('*', c_var(ho), c_var(stride_h)), c_var(kh)), c_var(pad_h))),
                                         c_decl_init(c_type(int), wi, c_binop('-', c_binop('+', c_binop('*', c_var(wo), c_var(stride_w)), c_var(kw)), c_var(pad_w))),
                                         c_if(c_binop('&&',
                                                c_binop('&&', c_binop('>=', c_var(di), c_int(0)), c_binop('<', c_var(di), c_var('D_in'))),
                                                c_binop('&&',
                                                  c_binop('&&', c_binop('>=', c_var(hi), c_int(0)), c_binop('<', c_var(hi), c_var('H_in'))),
                                                  c_binop('&&', c_binop('>=', c_var(wi), c_int(0)), c_binop('<', c_var(wi), c_var('W_in'))))),
                                              [c_decl_init(c_type(int), in_idx, c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(n), c_var('C_in')), c_var(ci))), c_var('D_in')), c_var(di))), c_var('H_in')), c_var(hi))), c_var('W_in')), c_var(wi))),
                                               c_decl_init(c_type(int), w_idx, c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(co), c_var('C_in')), c_var(ci))), c_var('kD')), c_var(kd))), c_var('kH')), c_var(kh))), c_var('kW')), c_var(kw))),
                                               c_assign(c_var(sum), c_binop('+', c_var(sum), c_binop('*', c_index(c_var(input), c_var(in_idx)), c_index(c_var(weight), c_var(w_idx)))))])
                                       ]))
                               ]))
                       ]))
               ])),
         c_if(c_var(bias), c_assign(c_var(sum), c_binop('+', c_var(sum), c_index(c_var(bias), c_var(co))))),
         c_assign(c_index(c_var(output), c_var(idx)), c_var(sum))]).

%% ── Pool 1D and 3D (KernelBench L1 #41,43,44,46) ──

pool_kernel(k_maxpool1d, max, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_maxpool1d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'C'), param(c_type(int), 'L_in'),
         param(c_type(int), ksize), param(c_type(int), stride),
         param(c_type(int), 'L_out')],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('C'), c_var('L_out'))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         c_decl_init(c_type(int), lo, c_binop('%', c_var(idx), c_var('L_out'))),
         c_decl_init(c_type(int), c, c_binop('/', c_var(idx), c_var('L_out'))),
         c_decl_init(c_type(float), val, c_float_f(-1.0e30)),
         c_for(c_decl_init(c_type(int), k, c_int(0)),
               c_binop('<', c_var(k), c_var(ksize)),
               c_assign(c_var(k), c_binop('+', c_var(k), c_int(1))),
               c_block([
                 c_decl_init(c_type(int), li, c_binop('+', c_binop('*', c_var(lo), c_var(stride)), c_var(k))),
                 c_if(c_binop('<', c_var(li), c_var('L_in')),
                      [c_decl_init(c_type(float), v, c_index(c_var(input), c_binop('+', c_binop('*', c_var(c), c_var('L_in')), c_var(li)))),
                       c_if(c_binop('>', c_var(v), c_var(val)),
                            [c_assign(c_var(val), c_var(v))])])
               ])),
         c_assign(c_index(c_var(output), c_var(idx)), c_var(val))]).

pool_kernel(k_avgpool1d, avg, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_avgpool1d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'C'), param(c_type(int), 'L_in'),
         param(c_type(int), ksize), param(c_type(int), stride),
         param(c_type(int), 'L_out')],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('C'), c_var('L_out'))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         c_decl_init(c_type(int), lo, c_binop('%', c_var(idx), c_var('L_out'))),
         c_decl_init(c_type(int), c, c_binop('/', c_var(idx), c_var('L_out'))),
         c_decl_init(c_type(float), sum, c_float_f(0.0)), c_decl_init(c_type(int), count, c_int(0)),
         c_for(c_decl_init(c_type(int), k, c_int(0)),
               c_binop('<', c_var(k), c_var(ksize)),
               c_assign(c_var(k), c_binop('+', c_var(k), c_int(1))),
               c_block([
                 c_decl_init(c_type(int), li, c_binop('+', c_binop('*', c_var(lo), c_var(stride)), c_var(k))),
                 c_if(c_binop('<', c_var(li), c_var('L_in')),
                      [c_assign(c_var(sum), c_binop('+', c_var(sum), c_index(c_var(input), c_binop('+', c_binop('*', c_var(c), c_var('L_in')), c_var(li))))),
                       c_assign(c_var(count), c_binop('+', c_var(count), c_int(1)))])
               ])),
         c_assign(c_index(c_var(output), c_var(idx)), c_binop('/', c_var(sum), c_cast(c_type(float), c_var(count))))]).

pool_kernel(k_maxpool3d, max, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_maxpool3d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'C'),
         param(c_type(int), 'D'), param(c_type(int), 'H'), param(c_type(int), 'W'),
         param(c_type(int), ksize), param(c_type(int), stride),
         param(c_type(int), 'D_out'), param(c_type(int), 'H_out'), param(c_type(int), 'W_out')],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('C'),
                c_binop('*', c_var('D_out'),
                    c_binop('*', c_var('H_out'), c_var('W_out'))))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         c_decl_init(c_type(int), wo, c_binop('%', c_var(idx), c_var('W_out'))),
         c_decl_init(c_type(int), ho, c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('W_out'))), c_var('H_out'))),
         c_decl_init(c_type(int), do_, c_binop('%', c_paren(c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_var('H_out'))))), c_var('D_out'))),
         c_decl_init(c_type(int), c, c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_binop('*', c_var('H_out'), c_var('D_out')))))),
         c_decl_init(c_type(float), val, c_float_f(-1.0e30)),
         c_for(c_decl_init(c_type(int), kd, c_int(0)),
               c_binop('<', c_var(kd), c_var(ksize)),
               c_assign(c_var(kd), c_binop('+', c_var(kd), c_int(1))),
               c_block([
                 c_for(c_decl_init(c_type(int), kh, c_int(0)),
                       c_binop('<', c_var(kh), c_var(ksize)),
                       c_assign(c_var(kh), c_binop('+', c_var(kh), c_int(1))),
                       c_block([
                         c_for(c_decl_init(c_type(int), kw, c_int(0)),
                               c_binop('<', c_var(kw), c_var(ksize)),
                               c_assign(c_var(kw), c_binop('+', c_var(kw), c_int(1))),
                               c_block([
                                 c_decl_init(c_type(int), di, c_binop('+', c_binop('*', c_var(do_), c_var(stride)), c_var(kd))),
                                 c_decl_init(c_type(int), hi, c_binop('+', c_binop('*', c_var(ho), c_var(stride)), c_var(kh))),
                                 c_decl_init(c_type(int), wi, c_binop('+', c_binop('*', c_var(wo), c_var(stride)), c_var(kw))),
                                 c_if(c_binop('&&',
                                        c_binop('&&', c_binop('<', c_var(di), c_var('D')), c_binop('<', c_var(hi), c_var('H'))),
                                        c_binop('<', c_var(wi), c_var('W'))),
                                      [c_decl_init(c_type(float), v, c_index(c_var(input), c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(c), c_var('D')), c_var(di))), c_var('H')), c_var(hi))), c_var('W')), c_var(wi)))),
                                       c_if(c_binop('>', c_var(v), c_var(val)),
                                            [c_assign(c_var(val), c_var(v))])])
                               ]))
                       ]))
               ])),
         c_assign(c_index(c_var(output), c_var(idx)), c_var(val))]).

pool_kernel(k_avgpool3d, avg, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_avgpool3d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'C'),
         param(c_type(int), 'D'), param(c_type(int), 'H'), param(c_type(int), 'W'),
         param(c_type(int), ksize), param(c_type(int), stride),
         param(c_type(int), 'D_out'), param(c_type(int), 'H_out'), param(c_type(int), 'W_out')],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('C'),
                c_binop('*', c_var('D_out'),
                    c_binop('*', c_var('H_out'), c_var('W_out'))))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         c_decl_init(c_type(int), wo, c_binop('%', c_var(idx), c_var('W_out'))),
         c_decl_init(c_type(int), ho, c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('W_out'))), c_var('H_out'))),
         c_decl_init(c_type(int), do_, c_binop('%', c_paren(c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_var('H_out'))))), c_var('D_out'))),
         c_decl_init(c_type(int), c, c_binop('/', c_var(idx), c_paren(c_binop('*', c_var('W_out'), c_binop('*', c_var('H_out'), c_var('D_out')))))),
         c_decl_init(c_type(float), sum, c_float_f(0.0)), c_decl_init(c_type(int), count, c_int(0)),
         c_for(c_decl_init(c_type(int), kd, c_int(0)),
               c_binop('<', c_var(kd), c_var(ksize)),
               c_assign(c_var(kd), c_binop('+', c_var(kd), c_int(1))),
               c_block([
                 c_for(c_decl_init(c_type(int), kh, c_int(0)),
                       c_binop('<', c_var(kh), c_var(ksize)),
                       c_assign(c_var(kh), c_binop('+', c_var(kh), c_int(1))),
                       c_block([
                         c_for(c_decl_init(c_type(int), kw, c_int(0)),
                               c_binop('<', c_var(kw), c_var(ksize)),
                               c_assign(c_var(kw), c_binop('+', c_var(kw), c_int(1))),
                               c_block([
                                 c_decl_init(c_type(int), di, c_binop('+', c_binop('*', c_var(do_), c_var(stride)), c_var(kd))),
                                 c_decl_init(c_type(int), hi, c_binop('+', c_binop('*', c_var(ho), c_var(stride)), c_var(kh))),
                                 c_decl_init(c_type(int), wi, c_binop('+', c_binop('*', c_var(wo), c_var(stride)), c_var(kw))),
                                 c_if(c_binop('&&',
                                        c_binop('&&', c_binop('<', c_var(di), c_var('D')), c_binop('<', c_var(hi), c_var('H'))),
                                        c_binop('<', c_var(wi), c_var('W'))),
                                      [c_assign(c_var(sum), c_binop('+', c_var(sum), c_index(c_var(input), c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(c), c_var('D')), c_var(di))), c_var('H')), c_var(hi))), c_var('W')), c_var(wi))))),
                                       c_assign(c_var(count), c_binop('+', c_var(count), c_int(1)))])
                               ]))
                       ]))
               ])),
         c_assign(c_index(c_var(output), c_var(idx)), c_binop('/', c_var(sum), c_cast(c_type(float), c_var(count))))]).

%% ── Prefix Scan (KernelBench L1 #89-93) ──

%% ── Upsample Nearest-Neighbor 2× (YOLO backbone) ──────────────────
%% One thread per OUTPUT element. Reads input[n][c][h/2][w/2].
%% Produces bit-identical output with F.interpolate(mode='nearest', scale_factor=2).
%%
%% Input:  float[N][C][H][W]
%% Output: float[N][C][2*H][2*W]
%% Grid:   (N*C*4*H*W + 255) / 256 blocks, 256 threads

upsample_kernel(k_upsample_nearest2d, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_upsample_nearest2d,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'N'),
         param(c_type(int), 'C'),
         param(c_type(int), 'H'),
         param(c_type(int), 'W')],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         %% Output dimensions are 2*H × 2*W
         c_decl_init(c_type(int), 'H_out', c_binop('*', c_int(2), c_var('H'))),
         c_decl_init(c_type(int), 'W_out', c_binop('*', c_int(2), c_var('W'))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('N'), c_binop('*', c_var('C'),
                c_binop('*', c_var('H_out'), c_var('W_out'))))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         %% Decompose linear index → (n, c, oh, ow) in output space
         c_decl_init(c_type(int), ow,
            c_binop('%', c_var(idx), c_var('W_out'))),
         c_decl_init(c_type(int), oh,
            c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('W_out'))), c_var('H_out'))),
         c_decl_init(c_type(int), c,
            c_binop('%', c_paren(c_binop('/', c_var(idx),
                c_paren(c_binop('*', c_var('H_out'), c_var('W_out'))))), c_var('C'))),
         c_decl_init(c_type(int), n,
            c_binop('/', c_var(idx),
                c_paren(c_binop('*', c_var('C'),
                    c_paren(c_binop('*', c_var('H_out'), c_var('W_out'))))))),
         %% Map output coords to input: nearest-neighbor = floor(oh/2), floor(ow/2)
         c_decl_init(c_type(int), ih, c_binop('/', c_var(oh), c_int(2))),
         c_decl_init(c_type(int), iw, c_binop('/', c_var(ow), c_int(2))),
         %% Read input, write output
         c_decl_init(c_type(int), in_idx,
            c_nd_index([n, 'C', c, 'H', ih, 'W', iw])),
         c_assign(c_index(c_var(output), c_var(idx)),
                  c_index(c_var(input), c_var(in_idx)))]).

%% Inclusive prefix sum/product using Blelloch scan algorithm
%% Single block, shared memory. For large N, multi-block with decoupled lookback.

scan_kernel(k_cumsum, sum, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_cumsum,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), n)],
        [c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_decl_init(c_type(int), gid,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_extern_shared(c_type(float), temp),
         c_assign(c_index(c_var(temp), c_var(tid)), c_ternary(c_binop('<', c_var(gid), c_var(n)), c_index(c_var(input), c_var(gid)), c_float_f(0.0))),
         c_syncthreads,
         %% Up-sweep (reduce)
         { scan_upsweep('+', temp, ScanUp) },
         ScanUp,
         %% Down-sweep
         { scan_downsweep('+', temp, ScanDown) },
         ScanDown,
         c_if(c_binop('<', c_var(gid), c_var(n)), c_assign(c_index(c_var(output), c_var(gid)), c_index(c_var(temp), c_var(tid))))]).

scan_kernel(k_cumprod, product, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_cumprod,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), n)],
        [c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_decl_init(c_type(int), gid,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_extern_shared(c_type(float), temp),
         c_assign(c_index(c_var(temp), c_var(tid)), c_ternary(c_binop('<', c_var(gid), c_var(n)), c_index(c_var(input), c_var(gid)), c_float_f(1.0))),
         c_syncthreads,
         { scan_upsweep('*', temp, ScanUp) },
         ScanUp,
         { scan_downsweep('*', temp, ScanDown) },
         ScanDown,
         c_if(c_binop('<', c_var(gid), c_var(n)), c_assign(c_index(c_var(output), c_var(gid)), c_index(c_var(temp), c_var(tid))))]).

%% ── BatchNorm (KernelBench L1 #33) ──
%% Two-pass: compute mean+var per channel, then normalize
%% InstanceNorm (#34) and GroupNorm (#35) are variants of the same pattern

norm_kernel(k_batchnorm, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_batchnorm,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(const_restrict_ptr(c_type(float))), gamma),
         param(c_type(const_restrict_ptr(c_type(float))), beta),
         param(c_type(const_restrict_ptr(c_type(float))), running_mean),
         param(c_type(const_restrict_ptr(c_type(float))), running_var),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'N'), param(c_type(int), 'C'),
         param(c_type(int), 'HW'), param(c_type(float), eps)],
        [c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_decl_init(c_type(int), total,
            c_binop('*', c_var('N'), c_binop('*', c_var('C'), c_var('HW')))),
         c_if(c_binop('>=', c_var(idx), c_var(total)), [c_return_void]),
         c_decl_init(c_type(int), c, c_binop('%', c_paren(c_binop('/', c_var(idx), c_var('HW'))), c_var('C'))),
         c_decl_init(c_type(float), x_val, c_index(c_var(input), c_var(idx))),
         c_decl_init(c_type(float), mean, c_index(c_var(running_mean), c_var(c))),
         c_decl_init(c_type(float), var, c_index(c_var(running_var), c_var(c))),
         c_decl_init(c_type(float), x_norm, c_binop('*', c_paren(c_binop('-', c_var(x_val), c_var(mean))), c_call(rsqrtf, [c_binop('+', c_var(var), c_var(eps))]))),
         c_assign(c_index(c_var(output), c_var(idx)),
                  c_binop('+', c_binop('*', c_index(c_var(gamma), c_var(c)), c_var(x_norm)),
                               c_index(c_var(beta), c_var(c))))]).

norm_kernel(k_layernorm, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_layernorm,
        [param(c_type(const_restrict_ptr(c_type(float))), x),
         param(c_type(const_restrict_ptr(c_type(float))), gamma),
         param(c_type(const_restrict_ptr(c_type(float))), beta),
         param(c_type(restrict_ptr(c_type(float))), y),
         param(c_type(int), n), param(c_type(float), eps)],
        [c_shared_decl(c_type(float), sred, c_int(128)),
         c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         %% Pass 1: mean
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var(n)),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_compound_assign('+=', c_var(sum), c_index(c_var(x), c_var(i)))]),
         c_assign(c_index(c_var(sred), c_var(tid)), c_var(sum)), c_syncthreads,
         { block_reduce_sum(sred, 128, BRStmt) }, BRStmt,
         c_decl_init(c_type(float), mean, c_binop('/', c_index(c_var(sred), c_int(0)), c_cast(c_type(float), c_var(n)))), c_syncthreads,
         %% Pass 2: variance
         c_decl_init(c_type(float), vsum, c_float_f(0.0)),
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var(n)),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_decl_init(c_type(float), d,
                  c_binop('-', c_index(c_var(x), c_var(i)), c_var(mean))),
              c_compound_assign('+=', c_var(vsum),
                  c_binop('*', c_var(d), c_var(d)))]),
         c_assign(c_index(c_var(sred), c_var(tid)), c_var(vsum)), c_syncthreads,
         { block_reduce_sum(sred, 128, BRStmt) }, BRStmt,
         c_decl_init(c_type(float), inv_std, c_call(rsqrtf, [c_binop('+', c_binop('/', c_index(c_var(sred), c_int(0)), c_cast(c_type(float), c_var(n))), c_var(eps))])), c_syncthreads,
         %% Pass 3: normalize
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var(n)),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_assign(c_index(c_var(y), c_var(i)),
                  c_binop('+',
                      c_binop('*', c_index(c_var(gamma), c_var(i)),
                          c_binop('*',
                              c_binop('-', c_index(c_var(x), c_var(i)), c_var(mean)),
                              c_var(inv_std))),
                      c_index(c_var(beta), c_var(i))))])]).

%% ── Remaining KernelBench L1 gap closers: 5/5 ──

%% Cumsum reverse: reverse input, cumsum, reverse output
scan_kernel(k_cumsum_reverse, sum_reverse, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_cumsum_reverse,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), n)],
        [c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_decl_init(c_type(int), gid,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_extern_shared(c_type(float), temp),
         c_decl_init(c_type(int), rev, c_ternary(c_binop('<', c_var(gid), c_var(n)), c_paren(c_binop('-', c_binop('-', c_var(n), c_int(1)), c_var(gid))), c_int(0))),
         c_assign(c_index(c_var(temp), c_var(tid)), c_ternary(c_binop('<', c_var(gid), c_var(n)), c_index(c_var(input), c_var(rev)), c_float_f(0.0))),
         c_syncthreads,
         { scan_upsweep('+', temp, ScanUp) },
         ScanUp,
         { scan_downsweep('+', temp, ScanDown) },
         ScanDown,
         c_if(c_binop('<', c_var(gid), c_var(n)),
              [c_assign(c_index(c_var(output), c_binop('-', c_binop('-', c_var(n), c_int(1)), c_var(gid))),
                        c_index(c_var(temp), c_var(tid)))])]).

%% Cumsum exclusive: output[i] = sum(input[0..i-1]), output[0] = 0
scan_kernel(k_cumsum_exclusive, sum_exclusive, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_cumsum_exclusive,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), n)],
        [c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_decl_init(c_type(int), gid,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_extern_shared(c_type(float), temp),
         c_assign(c_index(c_var(temp), c_var(tid)), c_ternary(c_binop('<', c_var(gid), c_var(n)), c_index(c_var(input), c_var(gid)), c_float_f(0.0))),
         c_syncthreads,
         { scan_upsweep('+', temp, ScanUp) },
         ScanUp,
         { scan_downsweep('+', temp, ScanDown) },
         ScanDown,
         %% Shift right for exclusive: output[i] = inclusive[i-1], output[0] = 0
         c_if(c_binop('<', c_var(gid), c_var(n)), c_assign(c_index(c_var(output), c_var(gid)), c_ternary(c_binop('>', c_var(tid), c_int(0)), c_index(c_var(temp), c_binop('-', c_var(tid), c_int(1))), c_float_f(0.0))))]).

%% Cumsum masked: cumsum only where mask is nonzero
scan_kernel(k_cumsum_masked, sum_masked, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_cumsum_masked,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(const_restrict_ptr(c_type(float))), mask),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), n)],
        [c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_decl_init(c_type(int), gid,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x),
                                      c_member(c_var(blockDim), x)),
                         c_member(c_var(threadIdx), x))),
         c_extern_shared(c_type(float), temp),
         c_decl_init(c_type(float), val, c_ternary(c_binop('<', c_var(gid), c_var(n)), c_binop('*', c_index(c_var(input), c_var(gid)), c_index(c_var(mask), c_var(gid))), c_float_f(0.0))),
         c_assign(c_index(c_var(temp), c_var(tid)), c_var(val)),
         c_syncthreads,
         { scan_upsweep('+', temp, ScanUp) },
         ScanUp,
         { scan_downsweep('+', temp, ScanDown) },
         ScanDown,
         c_if(c_binop('<', c_var(gid), c_var(n)), c_assign(c_index(c_var(output), c_var(gid)), c_index(c_var(temp), c_var(tid))))]).

%% InstanceNorm: like BatchNorm but per-sample (N×C, each HW independently)
norm_kernel(k_instance_norm, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_instance_norm,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(const_restrict_ptr(c_type(float))), gamma),
         param(c_type(const_restrict_ptr(c_type(float))), beta),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'C'), param(c_type(int), 'HW'), param(c_type(float), eps)],
        [c_shared_decl(c_type(float), sred, c_int(128)),
         c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_decl_init(c_type(int), nc, c_member(c_var(blockIdx), x)),
         c_decl_init(c_type(int), c, c_binop('%', c_var(nc), c_var('C'))),
         c_decl_init(c_type(const_ptr(c_type(float))), slice,
             c_ptr_arith(c_var(input), c_binop('*', c_var(nc), c_var('HW')))),
         c_decl_init(c_type(ptr(c_type(float))), out_slice,
             c_ptr_arith(c_var(output), c_binop('*', c_var(nc), c_var('HW')))),
         %% Mean
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var('HW')),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_compound_assign('+=', c_var(sum), c_index(c_var(slice), c_var(i)))]),
         c_assign(c_index(c_var(sred), c_var(tid)), c_var(sum)), c_syncthreads,
         { block_reduce_sum(sred, 128, BRStmt) }, BRStmt,
         c_decl_init(c_type(float), mean, c_binop('/', c_index(c_var(sred), c_int(0)), c_cast(c_type(float), c_var('HW')))), c_syncthreads,
         %% Variance
         c_decl_init(c_type(float), vsum, c_float_f(0.0)),
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var('HW')),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_decl_init(c_type(float), d, c_binop('-', c_index(c_var(slice), c_var(i)), c_var(mean))),
              c_compound_assign('+=', c_var(vsum), c_binop('*', c_var(d), c_var(d)))]),
         c_assign(c_index(c_var(sred), c_var(tid)), c_var(vsum)), c_syncthreads,
         { block_reduce_sum(sred, 128, BRStmt) }, BRStmt,
         c_decl_init(c_type(float), inv_std, c_call(rsqrtf, [c_binop('+', c_binop('/', c_index(c_var(sred), c_int(0)), c_cast(c_type(float), c_var('HW'))), c_var(eps))])), c_syncthreads,
         %% Normalize
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var('HW')),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_assign(c_index(c_var(out_slice), c_var(i)),
                 c_binop('+',
                     c_binop('*',
                         c_binop('*',
                             c_index(c_var(gamma), c_var(c)),
                             c_binop('-', c_index(c_var(slice), c_var(i)), c_var(mean))),
                         c_var(inv_std)),
                     c_index(c_var(beta), c_var(c))))])]).

%% GroupNorm: like InstanceNorm but groups of channels together
norm_kernel(k_group_norm, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), k_group_norm,
        [param(c_type(const_restrict_ptr(c_type(float))), input),
         param(c_type(const_restrict_ptr(c_type(float))), gamma),
         param(c_type(const_restrict_ptr(c_type(float))), beta),
         param(c_type(restrict_ptr(c_type(float))), output),
         param(c_type(int), 'C'), param(c_type(int), 'HW'),
         param(c_type(int), groups), param(c_type(float), eps)],
        [c_shared_decl(c_type(float), sred, c_int(128)),
         c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         c_decl_init(c_type(int), ng, c_member(c_var(blockIdx), x)),
         c_decl_init(c_type(int), g, c_binop('%', c_var(ng), c_var(groups))),
         c_decl_init(c_type(int), cpg, c_binop('/', c_var('C'), c_var(groups))),
         c_decl_init(c_type(int), group_size, c_binop('*', c_var(cpg), c_var('HW'))),
         c_decl_init(c_type(const_ptr(c_type(float))), slice,
             c_ptr_arith(c_var(input), c_binop('*', c_var(ng), c_var(group_size)))),
         c_decl_init(c_type(ptr(c_type(float))), out_slice,
             c_ptr_arith(c_var(output), c_binop('*', c_var(ng), c_var(group_size)))),
         %% Mean over group
         c_decl_init(c_type(float), sum, c_float_f(0.0)),
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var(group_size)),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_compound_assign('+=', c_var(sum), c_index(c_var(slice), c_var(i)))]),
         c_assign(c_index(c_var(sred), c_var(tid)), c_var(sum)), c_syncthreads,
         { block_reduce_sum(sred, 128, BRStmt) }, BRStmt,
         c_decl_init(c_type(float), mean, c_binop('/', c_index(c_var(sred), c_int(0)), c_cast(c_type(float), c_var(group_size)))), c_syncthreads,
         %% Variance
         c_decl_init(c_type(float), vsum, c_float_f(0.0)),
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var(group_size)),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_decl_init(c_type(float), d, c_binop('-', c_index(c_var(slice), c_var(i)), c_var(mean))),
              c_compound_assign('+=', c_var(vsum), c_binop('*', c_var(d), c_var(d)))]),
         c_assign(c_index(c_var(sred), c_var(tid)), c_var(vsum)), c_syncthreads,
         { block_reduce_sum(sred, 128, BRStmt) }, BRStmt,
         c_decl_init(c_type(float), inv_std, c_call(rsqrtf, [c_binop('+', c_binop('/', c_index(c_var(sred), c_int(0)), c_cast(c_type(float), c_var(group_size))), c_var(eps))])), c_syncthreads,
         %% Normalize with per-channel gamma/beta
         c_for_step(
             c_decl_init(c_type(int), i, c_var(tid)),
             c_binop('<', c_var(i), c_var(group_size)),
             c_compound_assign('+=', c_var(i), c_int(128)),
             [c_decl_init(c_type(int), local_c,
                  c_binop('+', c_binop('*', c_var(g), c_var(cpg)),
                               c_binop('/', c_var(i), c_var('HW')))),
              c_assign(c_index(c_var(out_slice), c_var(i)),
                  c_binop('+',
                      c_binop('*',
                          c_binop('*',
                              c_index(c_var(gamma), c_var(local_c)),
                              c_binop('-', c_index(c_var(slice), c_var(i)), c_var(mean))),
                          c_var(inv_std)),
                      c_index(c_var(beta), c_var(local_c))))])]).

%% ── The universal factory ──

%% Generate a kernel from an elem_op fact.
%% One template. Any operation. Add a fact, get a kernel.
elem_kernel(KName, Kernel) :-
    elem_op(KName, Params, OutputExpr, ComputeExpr),
    Kernel = c_func(['__global__'], c_type(void), KName,
        Params,
        [c_decl_init(c_type(int), i,
                c_binop('+',
                    c_binop('*', c_member(c_var(blockIdx), x),
                                 c_member(c_var(blockDim), x)),
                    c_member(c_var(threadIdx), x))),
         c_if(c_binop('>=', c_var(i), c_var(n)),
                [c_return_void]),
         c_assign(OutputExpr, ComputeExpr)]).


%% =============================================================================
%% BLAS L1: PARAMETERIZED REDUCTION KERNELS
%% =============================================================================
%%
%% Template parameters:
%%   Op:         dot | nrm2 | asum    (the reduction operation)
%%   NormSafety: none | safe_3way     (magnitude classification)
%%
%% Architecture:
%%   blas_l1_reduction_kernel(+KName, +Op, +NormSafety, -Kernel)
%%     ├── accumulation_body(Op, NormSafety) → per-element FMA
%%     ├── shared_mem_reduction()             → tree reduce in shmem
%%     └── finalize(Op, NormSafety)           → sqrt for nrm2, identity for dot/asum
%%
%% The same base kernel handles all three ops. NormSafety controls whether
%% elements are classified by magnitude before accumulation (cuBLAS's
%% safe-norm algorithm, discovered via SASS analysis 2026-05-19).
%%
%% SASS-derived constants (from cuBLAS nrm2_kernel on sm_61):
%%   THRESH_SMALL = 1.175494350822287508e-38  (FLT_MIN, denorm boundary)
%%   SCALE_SMALL  = 8.50705917302346158658e+37 (2^126)
%%   THRESH_BIG   = 1.304381782533278759e+19  (~sqrt(FLT_MAX))
%%   SCALE_BIG    = 1/SCALE_SMALL = 2^-126

%% ── Convenience wrappers ──

blas_l1_sdot_kernel(KName, NormSafety, Kernel) :-
    blas_l1_reduction_kernel(KName, dot, NormSafety, Kernel).

blas_l1_snrm2_kernel(KName, NormSafety, Kernel) :-
    blas_l1_reduction_kernel(KName, nrm2, NormSafety, Kernel).

blas_l1_sasum_kernel(KName, NormSafety, Kernel) :-
    blas_l1_reduction_kernel(KName, asum, NormSafety, Kernel).

%% ── The parameterized kernel ──

blas_l1_reduction_kernel(KName, Op, NormSafety, Kernel) :-
    %% Determine function signature based on Op
    op_params(Op, Params),
    op_shared_decl(NormSafety, SharedDecl),
    op_accum_decls(NormSafety, AccumDecls),
    op_accum_body(Op, NormSafety, AccumBody),
    op_reduce_body(NormSafety, ReduceBody),
    op_finalize(Op, NormSafety, FinalizeBody),

    Kernel = c_func(['__global__'], c_type(void), KName,
        Params,
        c_block([
            SharedDecl,
            c_decl_init(c_type(int), tid, c_member(threadIdx, x)),
            c_decl_init(c_type(int), gid,
                c_binop('+',
                    c_binop('*', c_member(blockIdx, x), c_int(128)),
                    c_var(tid))),
            c_decl_init(c_type(int), stride,
                c_binop('*', c_int(128), c_member(gridDim, x))),
            c_blank,
            %% Per-thread accumulator declarations
            AccumDecls,
            c_blank,
            %% Accumulation loop
            c_for(c_decl_init(c_type(int), i, c_var(gid)),
                  c_binop('<', c_var(i), c_var(n)),
                  c_assign(c_var(i), c_binop('+', c_var(i), c_var(stride))),
                  c_block(AccumBody)),
            c_blank,
            %% Store to shared memory and reduce
            ReduceBody,
            c_blank,
            %% Finalize and write result
            FinalizeBody
        ])).

%% ── Op-specific parameter lists ──

op_params(dot, [
    param(c_type(int), n),
    param(c_type(const_restrict_ptr(c_type(float))), x),
    param(c_type(const_restrict_ptr(c_type(float))), y),
    param(c_type(restrict_ptr(c_type(float))), result)
]).
op_params(nrm2, [
    param(c_type(int), n),
    param(c_type(const_restrict_ptr(c_type(float))), x),
    param(c_type(restrict_ptr(c_type(float))), result)
]).
op_params(asum, [
    param(c_type(int), n),
    param(c_type(const_restrict_ptr(c_type(float))), x),
    param(c_type(restrict_ptr(c_type(float))), result)
]).

%% ── Shared memory declarations ──

op_shared_decl(none, c_shared_decl(c_type(float), sred, c_int(128))).
op_shared_decl(safe_3way, c_shared_decl_2d(c_type(float), sred, c_int(3), c_int(128))).

%% ── Accumulator declarations ──

op_accum_decls(none, c_decl_init(c_type(float), p, c_float_f(0.0))).
op_accum_decls(safe_3way, c_block([
    c_decl_init(c_type(float), p_small, c_float_f(0.0)),
    c_decl_init(c_type(float), p_norm, c_float_f(0.0)),
    c_decl_init(c_type(float), p_big, c_float_f(0.0))
])).

%% ── Accumulation body (per-element, inside the for loop) ──

%% dot, no safety: p += x[i] * y[i]
op_accum_body(dot, none, [
    c_assign(c_var(p), c_binop('+', c_var(p),
        c_binop('*', c_index(c_var(x), c_var(i)),
                     c_index(c_var(y), c_var(i)))))
]).
%% nrm2, no safety: p += x[i] * x[i]
op_accum_body(nrm2, none, [
    c_assign(c_var(p), c_binop('+', c_var(p),
        c_binop('*', c_index(c_var(x), c_var(i)),
                     c_index(c_var(x), c_var(i)))))
]).
%% asum, no safety: p += fabsf(x[i])
op_accum_body(asum, none, [
    c_assign(c_var(p), c_binop('+', c_var(p),
        c_call(fabsf, [c_index(c_var(x), c_var(i))])))
]).

%% nrm2, safe_3way: classify magnitude, scale, accumulate into 3 bins
%% SASS-derived constants (from cuBLAS nrm2_kernel on sm_61):
%%   THRESH_BIG   = 1.304381782533278759e+19  (~sqrt(FLT_MAX))
%%   THRESH_SMALL = 1.175494350822287508e-38  (FLT_MIN, denorm boundary)
%%   SCALE_BIG    = 1.17549435082228750797e-38 (2^-126, shrinks big values)
%%   SCALE_SMALL  = 8.50705917302346158658e+37 (2^126, grows small values)
op_accum_body(nrm2, safe_3way, [
    c_block([
        c_decl_init(c_type(float), xi, c_index(c_var(x), c_var(i))),
        c_decl_init(c_type(float), ax, c_call(fabsf, [c_var(xi)])),
        c_if(c_binop('>=', c_var(ax), c_float_f(1.304381782533278759e+19)),
            [c_decl_init(c_type(float), xs,
                 c_binop('*', c_var(xi), c_float_f(1.17549435082228750797e-38))),
             c_assign(c_var(p_big), c_fma(c_var(xs), c_var(xs), c_var(p_big)))],
            [c_if(c_binop('>=', c_var(ax), c_float_f(1.175494350822287508e-38)),
                [c_assign(c_var(p_norm), c_fma(c_var(xi), c_var(xi), c_var(p_norm)))],
                [c_decl_init(c_type(float), xs,
                     c_binop('*', c_var(xi), c_float_f(8.50705917302346158658e+37))),
                 c_assign(c_var(p_small), c_fma(c_var(xs), c_var(xs), c_var(p_small)))])])
    ])
]).
%% dot, safe_3way: same classification on products (for completeness)
op_accum_body(dot, safe_3way, [
    c_assign(c_var(p_norm), c_binop('+', c_var(p_norm),
        c_binop('*', c_index(c_var(x), c_var(i)),
                     c_index(c_var(y), c_var(i)))))
]).
%% asum, safe_3way: no classification needed (absolute values don't overflow on sum)
op_accum_body(asum, safe_3way, [
    c_assign(c_var(p_norm), c_binop('+', c_var(p_norm),
        c_call(fabsf, [c_index(c_var(x), c_var(i))])))
]).

%% ── Reduction body ──

op_reduce_body(none, c_block([
    c_assign(c_index(c_var(sred), c_var(tid)), c_var(p)),
    c_syncthreads,
    c_for_step(
        c_decl_init(c_type(int), s, c_int(64)),
        c_binop('>', c_var(s), c_int(0)),
        c_compound_assign('>>=', c_var(s), c_int(1)),
        [c_if(c_binop('<', c_var(tid), c_var(s)),
            [c_compound_assign('+=',
                c_index(c_var(sred), c_var(tid)),
                c_index(c_var(sred), c_binop('+', c_var(tid), c_var(s))))]),
         c_syncthreads])
])).
op_reduce_body(safe_3way, c_block([
    c_assign(c_index2d(c_var(sred), c_int(0), c_var(tid)), c_var(p_small)),
    c_assign(c_index2d(c_var(sred), c_int(1), c_var(tid)), c_var(p_norm)),
    c_assign(c_index2d(c_var(sred), c_int(2), c_var(tid)), c_var(p_big)),
    c_syncthreads,
    c_for_step(
        c_decl_init(c_type(int), s, c_int(64)),
        c_binop('>', c_var(s), c_int(0)),
        c_compound_assign('>>=', c_var(s), c_int(1)),
        [c_if(c_binop('<', c_var(tid), c_var(s)),
            [c_compound_assign('+=',
                c_index2d(c_var(sred), c_int(0), c_var(tid)),
                c_index2d(c_var(sred), c_int(0), c_binop('+', c_var(tid), c_var(s)))),
             c_compound_assign('+=',
                c_index2d(c_var(sred), c_int(1), c_var(tid)),
                c_index2d(c_var(sred), c_int(1), c_binop('+', c_var(tid), c_var(s)))),
             c_compound_assign('+=',
                c_index2d(c_var(sred), c_int(2), c_var(tid)),
                c_index2d(c_var(sred), c_int(2), c_binop('+', c_var(tid), c_var(s))))]),
         c_syncthreads])
])).

%% ── Finalize: write result ──

op_finalize(dot, none, c_block([
    c_if(c_binop('==', c_var(tid), c_int(0)),
        [c_assign(c_deref(c_var(result)), c_index(c_var(sred), c_int(0)))])
])).
op_finalize(asum, none, c_block([
    c_if(c_binop('==', c_var(tid), c_int(0)),
        [c_assign(c_deref(c_var(result)), c_index(c_var(sred), c_int(0)))])
])).
op_finalize(nrm2, none, c_block([
    c_if(c_binop('==', c_var(tid), c_int(0)),
        [c_assign(c_deref(c_var(result)), c_call(sqrtf, [c_index(c_var(sred), c_int(0))]))])
])).
%% nrm2 safe_3way: rsqrt+Newton refinement on the dominant magnitude class.
%% SASS-derived algorithm: pick the largest class, compute rsqrt, apply one
%% Newton-Raphson step for extra precision, then scale back.
op_finalize(nrm2, safe_3way, c_block([
    c_if(c_binop('==', c_var(tid), c_int(0)),
        [c_decl_init(c_type(float), ss, c_index2d(c_var(sred), c_int(0), c_int(0))),
         c_decl_init(c_type(float), sn, c_index2d(c_var(sred), c_int(1), c_int(0))),
         c_decl_init(c_type(float), sb, c_index2d(c_var(sred), c_int(2), c_int(0))),
         c_decl(c_type(float), sum),
         c_decl(c_type(float), scale_inv),
         c_if(c_binop('>', c_var(sb), c_float_f(0.0)),
             [c_assign(c_var(sum), c_var(sb)),
              c_assign(c_var(scale_inv), c_float_f(8.50705917302346158658e+37))],
             [c_if(c_binop('>', c_var(sn), c_float_f(0.0)),
                 [c_assign(c_var(sum), c_var(sn)),
                  c_assign(c_var(scale_inv), c_float_f(1.0))],
                 [c_assign(c_var(sum), c_var(ss)),
                  c_assign(c_var(scale_inv), c_float_f(1.17549435082228750797e-38))])]),
         c_decl_init(c_type(float), rsq, c_call(rsqrtf, [c_var(sum)])),
         c_decl_init(c_type(float), h, c_binop('*', c_var(sum), c_var(rsq))),
         c_decl_init(c_type(float), hr, c_binop('*', c_var(rsq), c_float_f(0.5))),
         c_decl_init(c_type(float), e, c_fma(c_var(h), c_unop('-', c_var(h)), c_var(sum))),
         c_assign(c_deref(c_var(result)),
             c_binop('*', c_fma(c_var(e), c_var(hr), c_var(h)), c_var(scale_inv)))])
])).
op_finalize(dot, safe_3way, c_block([
    c_if(c_binop('==', c_var(tid), c_int(0)),
        [c_assign(c_deref(c_var(result)), c_index2d(c_var(sred), c_int(1), c_int(0)))])
])).
op_finalize(asum, safe_3way, c_block([
    c_if(c_binop('==', c_var(tid), c_int(0)),
        [c_assign(c_deref(c_var(result)), c_index2d(c_var(sred), c_int(1), c_int(0)))])
])).


%% =============================================================================
%% CONFIG METADATA
%% =============================================================================
%%
%% Per the substrate's fix-flag pattern extended for Tech-Level subsumption:
%% each kernel has multiple named configurations. The harness chooses which
%% to invoke based on the verification target (cuBLAS bit-match? substrate-
%% optimal? hybrid?).

kernel_configs(sgemv, [substrate_native, cublas_match]).

config_description(sgemv, substrate_native,
    'Warp-shuffle reduction, 32 threads per row, simple stride-32 accumulation. \c
     ~12 ULP gap vs cuBLAS sgemv on sm_61 (per mavchin empirical measurement). \c
     Lower latency than cublas_match due to no shared-mem reduction.').

config_description(sgemv, cublas_match,
    '32 threads per row × 4 rows per block (128 threads/block, sm_61), \c
     shared-memory x preload, stride-32 per-thread accumulation, warp-shuffle \c
     reduction within each row''s warp. Same per-thread arithmetic as \c
     substrate-native (12 ULP gap vs cuBLAS). Structural differences from \c
     substrate-native are tick-determining (memory hierarchy, block geometry) \c
     and do not change bits, per mavchin''s 2026-05-18 ~22:36 UTC empirical \c
     sweep. Remaining 12 ULP gap likely lives at compilation level (nvcc \c
     vs NVIDIA-internal toolchain).').


%% =============================================================================
%% FIX-FLAG METADATA
%% =============================================================================
%%
%% sgemv has no defect-repair or precision-tradeoff fixes — both configurations
%% are bit-deterministic and produce the bits they advertise. The fix-flag
%% mechanism is ready if/when we add Mode 2 substrate-optimizations that
%% trade bits for speed.

kernel_available_fixes(k_sgemv_substrate_native, []).
kernel_available_fixes(k_sgemv_cublas_match, []).


%% =============================================================================
%% DRIVER-SYMBOL WRAPPERS — gpu_* names for ctypes dispatch
%% =============================================================================
%%
%% Per medayek's swap discipline 2026-05-19 ~06:35 UTC:
%% the load-bearing llamatov_gpu_llama.py driver imports gpu_* symbols
%% from kernels.so via ctypes. For the file-level swap of
%% bpd/llamatov_kernels.cu to a BPD-generated equivalent, ALL imported
%% symbols must be present. These wrappers wire the BPD kernels into
%% the gpu_* symbol space the driver expects.
%%
%% Substantive substrate-design choice: keep the wrappers minimal
%% (single c_cuda_launch or c_expr_stmt). Per-kernel performance tuning
%% lives in the kernel templates, not the wrappers.

%% gpu_scale — out-of-place scalar multiply
%% Wraps k_scale (lifted as elem_op fact, commit 0f3e1dad6).
gpu_scale_wrapper(W) :-
    W = c_func(c_type(void), gpu_scale,
        [param(c_type(const_restrict_ptr(c_type(float))), in),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(float), s),
         param(c_type(int), n)],
        [c_cuda_launch(k_scale,
            c_binop('/',
                c_paren(c_binop('+', c_var(n), c_int(255))),
                c_int(256)),
            c_int(256),
            [c_var(in), c_var(out), c_var(s), c_var(n)])]).

%% gpu_copy_d2d — trivial cudaMemcpy device-to-device wrapper.
%% Substrate-design note: this is the simplest possible wrapper —
%% one cudaMemcpy call. Required for the driver because some
%% intermediate buffers are copied without computation.
gpu_copy_d2d_wrapper(W) :-
    W = c_func(c_type(void), gpu_copy_d2d,
        [param(c_type(ptr(c_type(void))), dst),
         param(c_type(const_ptr(c_type(void))), src),
         param(c_type(int), bytes)],
        [c_expr_stmt(c_call(cudaMemcpy,
            [c_var(dst), c_var(src), c_var(bytes),
             c_var(cudaMemcpyDeviceToDevice)]))]).

%% ═══════════════════════════════════════════════════════════════
%% Blelloch parallel prefix scan helpers
%% ═══════════════════════════════════════════════════════════════
%%
%% scan_upsweep(Op, Arr, -Stmts) — up-sweep (reduce) phase
%% scan_downsweep(Op, Arr, -Stmts) — down-sweep (distribute) phase
%%
%% Op: '+' or '*'
%% Arr: shared memory array name (atom, typically 'temp')
%%
%% NOTE: Emits c_raw internally because c_for DCG does not yet support
%% compound-assign step (d *= 2, d /= 2) or complex index expressions
%% (ai = (tid+1)*2*d-1). Tier 3 cleanup: extend c_for step + c_decl_init
%% with compound expressions.

%% scan_upsweep(+Op, +Arr, -Stmts)
%% Blelloch up-sweep (reduce) phase: for (d=1; d<blockDim.x; d*=2)
%% ai = (tid+1)*2*d - 1;  arr[ai] op= arr[ai-d]
%% Uses c_for_step with '*=' step and c_decl_init for ai inside the body.
scan_upsweep('+', Arr, [Stmts]) :-
    ArrV = c_var(Arr),
    Stmts = c_for_step(
        c_decl_init(c_type(int), d, c_int(1)),
        c_binop('<', c_var(d), c_member(c_var(blockDim), x)),
        c_compound_assign('*=', c_var(d), c_int(2)),
        [c_decl_init(c_type(int), ai,
             c_binop('-',
                 c_binop('*', c_binop('*', c_binop('+', c_var(tid), c_int(1)),
                                          c_int(2)),
                              c_var(d)),
                 c_int(1))),
         c_if(c_binop('<', c_var(ai), c_member(c_var(blockDim), x)),
             [c_compound_assign('+=',
                 c_index(ArrV, c_var(ai)),
                 c_index(ArrV, c_binop('-', c_var(ai), c_var(d))))]),
         c_syncthreads]).

scan_upsweep('*', Arr, [Stmts]) :-
    ArrV = c_var(Arr),
    Stmts = c_for_step(
        c_decl_init(c_type(int), d, c_int(1)),
        c_binop('<', c_var(d), c_member(c_var(blockDim), x)),
        c_compound_assign('*=', c_var(d), c_int(2)),
        [c_decl_init(c_type(int), ai,
             c_binop('-',
                 c_binop('*', c_binop('*', c_binop('+', c_var(tid), c_int(1)),
                                          c_int(2)),
                              c_var(d)),
                 c_int(1))),
         c_if(c_binop('<', c_var(ai), c_member(c_var(blockDim), x)),
             [c_compound_assign('*=',
                 c_index(ArrV, c_var(ai)),
                 c_index(ArrV, c_binop('-', c_var(ai), c_var(d))))]),
         c_syncthreads]).

%% scan_downsweep(+Op, +Arr, -Stmts)
%% Blelloch down-sweep (distribute) phase: for (d=blockDim.x/4; d>0; d/=2)
%% ai = (tid+1)*2*d - 1 + d;  arr[ai] op= arr[ai-d]
scan_downsweep('+', Arr, [Stmts]) :-
    ArrV = c_var(Arr),
    Stmts = c_for_step(
        c_decl_init(c_type(int), d,
            c_binop('/', c_member(c_var(blockDim), x), c_int(4))),
        c_binop('>', c_var(d), c_int(0)),
        c_compound_assign('/=', c_var(d), c_int(2)),
        [c_decl_init(c_type(int), ai,
             c_binop('+',
                 c_binop('-',
                     c_binop('*', c_binop('*', c_binop('+', c_var(tid), c_int(1)),
                                              c_int(2)),
                                  c_var(d)),
                     c_int(1)),
                 c_var(d))),
         c_if(c_binop('<', c_var(ai), c_member(c_var(blockDim), x)),
             [c_compound_assign('+=',
                 c_index(ArrV, c_var(ai)),
                 c_index(ArrV, c_binop('-', c_var(ai), c_var(d))))]),
         c_syncthreads]).

scan_downsweep('*', Arr, [Stmts]) :-
    ArrV = c_var(Arr),
    Stmts = c_for_step(
        c_decl_init(c_type(int), d,
            c_binop('/', c_member(c_var(blockDim), x), c_int(4))),
        c_binop('>', c_var(d), c_int(0)),
        c_compound_assign('/=', c_var(d), c_int(2)),
        [c_decl_init(c_type(int), ai,
             c_binop('+',
                 c_binop('-',
                     c_binop('*', c_binop('*', c_binop('+', c_var(tid), c_int(1)),
                                              c_int(2)),
                                  c_var(d)),
                     c_int(1)),
                 c_var(d))),
         c_if(c_binop('<', c_var(ai), c_member(c_var(blockDim), x)),
             [c_compound_assign('*=',
                 c_index(ArrV, c_var(ai)),
                 c_index(ArrV, c_binop('-', c_var(ai), c_var(d))))]),
         c_syncthreads]).
