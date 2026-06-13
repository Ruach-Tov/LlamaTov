%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% generate_fused_kernels.pl — FUSED CUDA kernels via Prolog AST emission
%%
%% Per Heath's directive (via medayek 2026-05-16 00:31): beat cuBLAS via
%% kernel fusion. cuBLAS cannot fuse custom epilogues; we can.
%%
%% Strategy: extend mavchin's vecmat structure (the 100%-thread-utilization
%% M=1 decode kernel from commit 89d726216) with a register-resident
%% epilogue. The activation applies on the matmul accumulator BEFORE
%% writing to DRAM — eliminating one DRAM round-trip per fused operation.
%%
%% Uses the same AST DSL vocabulary mavchin established in commit 0a3b5abf5:
%%   c_extern_shared (statement, not type)
%%   c_compound_step (for += updates)
%%   c_postfix (for ++ post-increment)
%%
%% Composition: matches mavchin's vecmat_kernel/1 pattern in
%% generate_llama_kernels.pl. Adds an EpilogueExpr parameter that
%% transforms `sum` register before the final DRAM write.
%%
%% Author: metayen 2026-05-16
%% Per Heath's directive + corrections-11-resolution unblocking my work.

:- use_module(lib/c_ast).

%% vecmat_with_epilogue_kernel(+KernelName, +EpilogueExpr, -KernelAST)
%%
%% Identical structure to mavchin's k_vecmat but with an extra step:
%% before writing to C[col], transform `sum` via EpilogueExpr in registers.
%%
%% EpilogueExpr is an AST expression that references c_var(sum) and
%% produces the new value to assign back to sum.

vecmat_with_epilogue_kernel(KernelName, EpilogueExpr, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), KernelName,
        [param(c_type(const_ptr(c_type(float))), a),
         param(c_type(const_ptr(c_type(float))), b),
         param(c_type(ptr(c_type(float))), c),
         param(c_type(int), k_dim),
         param(c_type(int), n_dim)],
        [
         %% Dynamically-sized shared memory for vector A
         c_extern_shared(c_type(float), sA),
         %% Cooperative load of A into shared memory
         c_for(c_decl_init(c_type(int), i, c_member(c_var(threadIdx), x)),
               c_binop(<, c_var(i), c_var(k_dim)),
               c_compound_step(c_var(i), '+=', c_member(c_var(blockDim), x)),
               [c_assign(c_index(c_var(sA), c_var(i)),
                          c_index(c_var(a), c_var(i)))]),
         c_syncthreads,
         %% Each thread computes one output column
         c_decl_init(c_type(int), col,
             c_binop('+',
                 c_binop('*',
                     c_member(c_var(blockIdx), x),
                     c_member(c_var(blockDim), x)),
                 c_member(c_var(threadIdx), x))),
         c_if(c_binop('>=', c_var(col), c_var(n_dim)), [c_return_void]),
         %% Dot product: sum = sum sA[k] * b[k*n_dim + col]
         c_decl_init(c_type(float), sum, c_float(0.0)),
         c_for(c_decl_init(c_type(int), k, c_int(0)),
               c_binop('<', c_var(k), c_var(k_dim)),
               c_postfix('++', c_var(k)),
               [c_assign(c_var(sum),
                   c_binop('+',
                       c_var(sum),
                       c_binop('*',
                           c_index(c_var(sA), c_var(k)),
                           c_index(c_var(b),
                               c_binop('+',
                                   c_binop('*', c_var(k), c_var(n_dim)),
                                   c_var(col))))))]),
         %% ★ FUSED EPILOGUE: activation applied to `sum` in registers
         %% This is the substantive optimization vs cuBLAS (which can't fuse)
         c_assign(c_var(sum), EpilogueExpr),
         %% Single DRAM write (one write instead of two)
         c_assign(c_index(c_var(c), c_var(col)), c_var(sum))
        ]).

%% ═════════════════════════════════════════════════════════════════════
%% Specific fused kernels per activation
%% ═════════════════════════════════════════════════════════════════════

%% Fused vecmat + SiLU: sum = sum / (1.0 + expf(-sum))
%% The most common fusion in Llama FFN gate path.
vecmat_silu_kernel(K) :-
    Epilogue = c_binop('/',
                    c_var(sum),
                    c_paren(c_binop('+',
                        c_float(1.0),
                        c_call(expf, [c_unop('-', c_var(sum))])))),
    vecmat_with_epilogue_kernel(k_vecmat_silu, Epilogue, K).

%% Fused vecmat + GELU (exact via erff)
%% GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
%% sqrt(2) reciprocal ≈ 0.7071067812
vecmat_gelu_kernel(K) :-
    Epilogue = c_binop('*',
                    c_binop('*', c_float(0.5), c_var(sum)),
                    c_paren(c_binop('+',
                        c_float(1.0),
                        c_call(erff, [c_binop('*',
                            c_var(sum),
                            c_float(0.70710678118654752440))])))),
    vecmat_with_epilogue_kernel(k_vecmat_gelu, Epilogue, K).

%% Fused vecmat + ReLU
vecmat_relu_kernel(K) :-
    Epilogue = c_ternary(c_binop('>', c_var(sum), c_float(0.0)),
                          c_var(sum),
                          c_float(0.0)),
    vecmat_with_epilogue_kernel(k_vecmat_relu, Epilogue, K).

%% ═════════════════════════════════════════════════════════════════════
%% Full program assembly (for nvcc compilation)
%% ═════════════════════════════════════════════════════════════════════

generate_all(Code) :-
    vecmat_silu_kernel(SiluK),
    vecmat_gelu_kernel(GeluK),
    vecmat_relu_kernel(ReluK),
    Program = [
        c_include_sys('cuda_runtime.h'),
        c_include_sys('math.h'),
        c_blank,
        c_comment('BPD-generated FUSED CUDA kernels for LlamaTov'),
        c_comment('Per Heath directive: beat cuBLAS via fusion'),
        c_comment('Each kernel: matmul + activation in ONE launch, ONE DRAM write'),
        c_comment('100% AST-generated via Prolog c_ast DCG emitter'),
        c_blank,
        c_extern_c_open,
        c_blank,
        SiluK, c_blank,
        GeluK, c_blank,
        ReluK, c_blank,
        c_extern_c_close
    ],
    emit_program(Program, Code).

test :-
    generate_all(Code),
    write(Code), nl.

%% Only run as standalone (don't fire when consulted from test files)
:- initialization((
    catch(current_prolog_flag(associated_file, F), _, F=''),
    ( atom_concat(_, '/generate_fused_kernels.pl', F)
    -> ( test -> halt(0) ; (write('FAILED'), nl, halt(1)) )
    ;  true
    )
)).
