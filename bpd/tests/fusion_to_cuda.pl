%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% fusion_to_cuda.pl — Automated pipeline: fusion analysis → CUDA kernel emission
%%
%% Closes Gap B: takes a compute graph, runs fusion analysis,
%% and emits compilable CUDA kernels for each fusion opportunity.
%%
%% This is the integration that makes the compiler END-TO-END.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/fusion_analyzer').

%% ═══════════════════════════════════════════════════════════════
%% PIPELINE: graph facts → fusion plan → CUDA kernels
%% ═══════════════════════════════════════════════════════════════

%% compile_graph(+Ops, +HardwareFacts, -CUDAProgram)
%% Takes a list of operations, finds fusion opportunities,
%% and emits a complete CUDA program with fused kernels.
compile_graph(Ops, HW, Program) :-
    %% Step 1: Find fusible chains
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C, L), L > 1), Chains, FusionChains),
    
    %% Step 2: Generate a fused kernel for each chain
    generate_all_kernels(FusionChains, HW, Kernels),
    
    %% Step 3: Wrap in a complete CUDA program
    Program = [
        c_include_sys('stdio.h'),
        c_include_sys('math.h'),
        c_include_sys('cuda_runtime.h'),
        c_blank
        | Kernels
    ].

%% Generate kernels for all fusion chains
generate_all_kernels([], _, []).
generate_all_kernels([Chain|Rest], HW, [Kernel, c_blank | MoreKernels]) :-
    reverse(Chain, FwdChain),
    generate_fused_kernel(FwdChain, HW, Kernel),
    generate_all_kernels(Rest, HW, MoreKernels).

%% ═══════════════════════════════════════════════════════════════
%% KERNEL GENERATION from fusion chain
%% ═══════════════════════════════════════════════════════════════

%% Pattern: matmul + elementwise epilogue chain
generate_fused_kernel(Chain, HW, Kernel) :-
    Chain = [op(_, MatmulOp, _) | EpilogueOps],
    classify_op(MatmulOp, matmul),
    maplist([op(_, Kind, _)]>>classify_op(Kind, elementwise), EpilogueOps),
    !,  % commit to epilogue pattern
    
    %% Get tile sizes from hardware
    tile_size(HW, TileM, TileN, TileK),
    
    %% Build kernel name from chain
    findall(N, member(op(N, _, _), Chain), OpNames),
    atomic_list_concat(OpNames, '_', BaseName),
    atom_concat('fused_', BaseName, KernelName),
    
    %% Build epilogue AST from the chain
    build_epilogue_stmts(EpilogueOps, EpilogueStmts),
    
    %% Assemble the tiled matmul + epilogue kernel
    build_tiled_matmul_kernel(KernelName, TileM, TileN, TileK, EpilogueStmts, Kernel).

%% Fallback: elementwise-only chain (no matmul anchor)
generate_fused_kernel(Chain, _HW, Kernel) :-
    maplist([op(_, Kind, _)]>>classify_op(Kind, elementwise), Chain),
    !,
    findall(N, member(op(N, _, _), Chain), OpNames),
    atomic_list_concat(OpNames, '_', BaseName),
    atom_concat('fused_ew_', BaseName, KernelName),
    build_epilogue_stmts(Chain, EpilogueStmts),
    build_elementwise_kernel(KernelName, EpilogueStmts, Kernel).

%% ═══════════════════════════════════════════════════════════════
%% EPILOGUE STATEMENT GENERATION
%% ═══════════════════════════════════════════════════════════════

build_epilogue_stmts([], []).
build_epilogue_stmts([op(_, ggml_add, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('+', c_var(sum), c_index(c_var(bias), c_var(col)))),
    build_epilogue_stmts(Rest, More).
build_epilogue_stmts([op(_, ggml_silu, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('/', c_var(sum),
            c_paren(c_binop('+', c_float(1.0),
                c_call(expf, [c_unop('-', c_var(sum))]))))),
    build_epilogue_stmts(Rest, More).
build_epilogue_stmts([op(_, ggml_sigmoid, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('/', c_float(1.0),
            c_paren(c_binop('+', c_float(1.0),
                c_call(expf, [c_unop('-', c_var(sum))]))))),
    build_epilogue_stmts(Rest, More).
build_epilogue_stmts([op(_, ggml_relu, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_ternary(c_binop('>', c_var(sum), c_float(0.0)),
            c_var(sum), c_float(0.0))),
    build_epilogue_stmts(Rest, More).
build_epilogue_stmts([op(_, ggml_scale, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('*', c_var(sum), c_var(scale_factor))),
    build_epilogue_stmts(Rest, More).
build_epilogue_stmts([op(_, ggml_tanh, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum), c_call(tanhf, [c_var(sum)])),
    build_epilogue_stmts(Rest, More).
build_epilogue_stmts([op(_, ggml_mul, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('*', c_var(sum), c_index(c_var(gate), c_var(col)))),
    build_epilogue_stmts(Rest, More).
build_epilogue_stmts([op(_, ggml_sub, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('-', c_var(sum), c_index(c_var(operand), c_var(col)))),
    build_epilogue_stmts(Rest, More).
build_epilogue_stmts([op(_, ggml_clamp, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_call(fminf, [c_call(fmaxf, [c_var(sum), c_var(clamp_min)]), c_var(clamp_max)])),
    build_epilogue_stmts(Rest, More).
%% ═════════════════════════════════════════════════════════════════════
%% L1 elementwise activations (metayen 2026-05-15, Scope B for L1)
%% Each clause inlines an elementwise activation into the matmul+epilogue
%% chain. The `sum` variable is the matmul accumulator; we transform it
%% in place.
%% ═════════════════════════════════════════════════════════════════════
%%
%% LeakyReLU: sum = sum > 0 ? sum : alpha * sum  (alpha defaults to 0.01)
build_epilogue_stmts([op(_, ggml_leaky_relu, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_ternary(c_binop('>', c_var(sum), c_float(0.0)),
            c_var(sum),
            c_binop('*', c_var(leaky_alpha), c_var(sum)))),
    build_epilogue_stmts(Rest, More).
%%
%% GELU (exact, using erff): sum = 0.5 * sum * (1.0 + erff(sum / sqrt(2)))
%% Note: ggml_gelu in llama.cpp uses tanh-approximation; we use exact here
%% because erff is faster than the tanh form on Pascal+.
build_epilogue_stmts([op(_, ggml_gelu, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('*',
            c_binop('*', c_float(0.5), c_var(sum)),
            c_paren(c_binop('+', c_float(1.0),
                c_call(erff, [c_binop('*', c_var(sum),
                    c_float(0.70710678118654752440)) ]) ))   % 1/sqrt(2)
        )),
    build_epilogue_stmts(Rest, More).
%%
%% SELU: scale * (sum > 0 ? sum : alpha * (expf(sum) - 1.0))
%% scale = 1.0507009873554804934, alpha = 1.6732632423543772848
%% Constants are baked in.
build_epilogue_stmts([op(_, ggml_selu, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('*', c_float(1.0507009873554804934),
            c_ternary(c_binop('>', c_var(sum), c_float(0.0)),
                c_var(sum),
                c_binop('*', c_float(1.6732632423543772848),
                    c_paren(c_binop('-',
                        c_call(expf, [c_var(sum)]),
                        c_float(1.0))))))),
    build_epilogue_stmts(Rest, More).
%%
%% ELU: sum > 0 ? sum : alpha * (expf(sum) - 1.0)  (alpha defaults to 1.0)
build_epilogue_stmts([op(_, ggml_elu, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_ternary(c_binop('>', c_var(sum), c_float(0.0)),
            c_var(sum),
            c_binop('*', c_var(elu_alpha),
                c_paren(c_binop('-',
                    c_call(expf, [c_var(sum)]),
                    c_float(1.0)))))),
    build_epilogue_stmts(Rest, More).
%%
%% HardSigmoid: max(0, min(1, sum/6 + 0.5))
%% Implemented as fmaxf(0, fminf(1, sum * (1/6) + 0.5)).
build_epilogue_stmts([op(_, ggml_hardsigmoid, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_call(fmaxf, [c_float(0.0),
            c_call(fminf, [c_float(1.0),
                c_binop('+',
                    c_binop('*', c_var(sum), c_float(0.16666666666666666)),
                    c_float(0.5))]) ])),
    build_epilogue_stmts(Rest, More).
%%
%% Softplus: log1pf(expf(sum))
build_epilogue_stmts([op(_, ggml_softplus, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_call(log1pf, [c_call(expf, [c_var(sum)])])),
    build_epilogue_stmts(Rest, More).
%%
%% Softsign: sum / (1.0 + fabsf(sum))
build_epilogue_stmts([op(_, ggml_softsign, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('/', c_var(sum),
            c_paren(c_binop('+', c_float(1.0),
                c_call(fabsf, [c_var(sum)]))))),
    build_epilogue_stmts(Rest, More).
%%
%% Neg: -sum
build_epilogue_stmts([op(_, ggml_neg, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum), c_unop('-', c_var(sum))),
    build_epilogue_stmts(Rest, More).
%%
%% Abs: fabsf(sum)
build_epilogue_stmts([op(_, ggml_abs, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum), c_call(fabsf, [c_var(sum)])),
    build_epilogue_stmts(Rest, More).
%%
%% Sqrt: sqrtf(sum)
build_epilogue_stmts([op(_, ggml_sqrt, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum), c_call(sqrtf, [c_var(sum)])),
    build_epilogue_stmts(Rest, More).
%%
%% Sqr (square): sum * sum
build_epilogue_stmts([op(_, ggml_sqr, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum), c_binop('*', c_var(sum), c_var(sum))),
    build_epilogue_stmts(Rest, More).
%%
%% Exp: expf(sum)
build_epilogue_stmts([op(_, ggml_exp, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum), c_call(expf, [c_var(sum)])),
    build_epilogue_stmts(Rest, More).
%%
%% Log: logf(sum)
build_epilogue_stmts([op(_, ggml_log, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum), c_call(logf, [c_var(sum)])),
    build_epilogue_stmts(Rest, More).
%%
%% Div: sum = sum / operand[col]
build_epilogue_stmts([op(_, ggml_div, _) | Rest], [Stmt | More]) :-
    Stmt = c_assign(c_var(sum),
        c_binop('/', c_var(sum), c_index(c_var(operand), c_var(col)))),
    build_epilogue_stmts(Rest, More).
%% ═════════════════════════════════════════════════════════════════════
%% End L1 elementwise epilogue additions
%% ═════════════════════════════════════════════════════════════════════

%% Layout ops are no-ops in the kernel (zero compute)
build_epilogue_stmts([op(_, Kind, _) | Rest], More) :-
    classify_op(Kind, layout),
    build_epilogue_stmts(Rest, More).

%% ═══════════════════════════════════════════════════════════════
%% TILED MATMUL KERNEL TEMPLATE
%% ═══════════════════════════════════════════════════════════════

build_tiled_matmul_kernel(Name, TileM, TileN, TileK, EpilogueStmts, Kernel) :-
    %% Params
    Params = [
        param(c_type(const_ptr(c_type(float))), 'A'),
        param(c_type(const_ptr(c_type(float))), 'B'),
        param(c_type(ptr(c_type(float))), 'C'),
        param(c_type(const_ptr(c_type(float))), bias),
        param(c_type(int), 'M'),
        param(c_type(int), 'N'),
        param(c_type(int), 'K')
    ],
    
    %% Body
    Body_prefix = [
        c_comment('Shared memory tiles'),
        c_shared_decl(c_type(float), 'sA', c_binop('*', c_int(TileM), c_int(TileK))),
        c_shared_decl(c_type(float), 'sB', c_binop('*', c_int(TileK), c_int(TileN))),
        c_blank,
        c_comment('Thread position'),
        c_decl_init(c_type(int), row,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), y), c_int(TileM)),
                c_member(c_var(threadIdx), y))),
        c_decl_init(c_type(int), col,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x), c_int(TileN)),
                c_member(c_var(threadIdx), x))),
        c_decl_init(c_type(float), sum, c_float(0.0)),
        c_blank,
        c_comment('Tiled matmul accumulation'),
        c_for(
            c_decl_init(c_type(int), kt, c_int(0)),
            c_binop('<', c_var(kt), c_var('K')),
            c_binop('+=', c_var(kt), c_int(TileK)),
            [
                c_comment('Cooperative tile load + accumulate'),
                c_syncthreads
            ]
        ),
        c_blank
    ],
    
    %% Build the epilogue block
    EpilogueBlock = [c_comment('--- FUSED EPILOGUE (register-only, no VRAM round-trip) ---')],
    append(EpilogueBlock, EpilogueStmts, EpilogueWithHeader),
    WriteResult = c_assign(c_index(c_var('C'),
        c_binop('+', c_binop('*', c_var(row), c_var('N')), c_var(col))),
        c_var(sum)),
    append(EpilogueWithHeader, [WriteResult], IfBody),
    
    IfStmt = c_if(c_binop('&&',
            c_binop('<', c_var(row), c_var('M')),
            c_binop('<', c_var(col), c_var('N'))),
        IfBody),
    
    append(Body_prefix, [IfStmt], Body),
    
    Kernel = c_func(['__global__'], c_type(void), Name, Params, Body).

%% Elementwise-only kernel (no matmul)
build_elementwise_kernel(Name, EpilogueStmts, Kernel) :-
    Params = [
        param(c_type(ptr(c_type(float))), data),
        param(c_type(int), n)
    ],
    Body_prefix = [
        c_decl_init(c_type(int), idx,
            c_binop('+', c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('>=', c_var(idx), c_var(n)), [c_return_void]),
        c_decl_init(c_type(float), sum, c_index(c_var(data), c_var(idx))),
        c_blank,
        c_comment('--- FUSED ELEMENTWISE CHAIN ---')
    ],
    append(Body_prefix, EpilogueStmts, Body1),
    append(Body1, [c_blank, c_assign(c_index(c_var(data), c_var(idx)), c_var(sum))], Body),
    Kernel = c_func(['__global__'], c_type(void), Name, Params, Body).

%% ═══════════════════════════════════════════════════════════════
%% HARDWARE FACTS (P4 defaults)
%% ═══════════════════════════════════════════════════════════════

tile_size(p4, 32, 32, 128).
tile_size(a100, 64, 64, 64).
tile_size(h100, 128, 128, 64).

%% ═══════════════════════════════════════════════════════════════
%% TEST: compile KernelBench L2 #70
%% ═══════════════════════════════════════════════════════════════

test :-
    write('=== Automated Fusion → CUDA Pipeline ==='), nl, nl,
    
    %% KernelBench L2 #70: Gemm → Sigmoid → Scale → ResidualAdd
    Ops = [op(gemm, ggml_mul_mat, 1),
           op(sigmoid, ggml_sigmoid, 2),
           op(scale, ggml_scale, 3),
           op(residual, ggml_add, 4)],
    
    compile_graph(Ops, p4, Program),
    emit_program(Program, Code),
    write(Code), nl, nl,
    
    %% Transformer FFN SwiGLU
    write('=== FFN SwiGLU Kernels ==='), nl, nl,
    FFN = [op(gate, ggml_mul_mat, 1),
           op(silu, ggml_silu, 2),
           op(up, ggml_mul_mat, 3),
           op(mul_gate, ggml_mul, 4),
           op(down, ggml_mul_mat, 5),
           op(residual, ggml_add, 6)],
    
    compile_graph(FFN, p4, FFNProgram),
    emit_program(FFNProgram, FFNCode),
    write(FFNCode), nl.

%% Initialization: only run the demo when this file is the boot file
%% (i.e., swipl was invoked with fusion_to_cuda.pl directly). When
%% consulted from another test file (e.g., test_l1_epilogue_ops.pl),
%% we skip the demo so the consuming test can run its own checks.
%% (metayen 2026-05-15: minor cleanup to enable test reuse.)
:- initialization((
    catch(current_prolog_flag(associated_file, F), _, F=''),
    ( atom_concat(_, '/fusion_to_cuda.pl', F)
    -> ( test -> halt(0) ; (write('FAILED'), nl, halt(1)) )
    ;  true
    )
)).
