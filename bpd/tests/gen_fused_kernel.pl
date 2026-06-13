%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% gen_fused_kernel.pl — Generate a fused matmul+bias CUDA kernel from BPD facts.
%%
%% This is the HANDS of the fusion compiler.
%% The fusion analyzer (brain) says "fuse matmul+bias."
%% This module emits the CUDA kernel that implements it.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').

%% ═══════════════════════════════════════════════════════════════
%% BPD FACTS describing the fusion opportunity
%% ═══════════════════════════════════════════════════════════════

%% From fusion analyzer: matmul(wq, cur) → add(bias) is fusible as epilogue.
%% The fused kernel computes: output[i] = sum(wq[i,:] * cur[:]) + bias[i]
%% Instead of: temp = matmul(wq, cur); output = temp + bias;

%% ═══════════════════════════════════════════════════════════════
%% KERNEL GENERATION from fusion plan
%% ═══════════════════════════════════════════════════════════════

%% Generate a fused matmul+bias+activation kernel
generate_fused_matmul_epilogue(M, N, K, HasBias, Activation, KernelAST) :-
    %% Thread computes one output element
    %% row = blockIdx.y * blockDim.y + threadIdx.y
    %% col = blockIdx.x * blockDim.x + threadIdx.x
    Row = c_decl_init(c_type(int), row,
        c_binop('+',
            c_binop('*', c_member(c_var(blockIdx), y), c_member(c_var(blockDim), y)),
            c_member(c_var(threadIdx), y))),
    Col = c_decl_init(c_type(int), col,
        c_binop('+',
            c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
            c_member(c_var(threadIdx), x))),
    
    %% Bounds check
    BoundsCheck = c_if(
        c_binop('||',
            c_binop('>=', c_var(row), c_var('M')),
            c_binop('>=', c_var(col), c_var('N'))),
        [c_return_void]),
    
    %% Accumulator
    Accum = c_decl_init(c_type(float), sum, c_float(0.0)),
    
    %% Inner product loop
    InnerLoop = c_for(
        c_decl_init(c_type(int), k, c_int(0)),
        c_binop('<', c_var(k), c_var('K')),
        c_postfix('++', c_var(k)),
        [
            c_assign(c_var(sum),
                c_binop('+', c_var(sum),
                    c_binop('*',
                        c_index(c_var('A'), c_binop('+', c_binop('*', c_var(row), c_var('K')), c_var(k))),
                        c_index(c_var('B'), c_binop('+', c_binop('*', c_var(k), c_var('N')), c_var(col))))))
        ]),
    
    %% Epilogue: bias add (fused — no VRAM round-trip!)
    ( HasBias = true ->
        BiasStmt = c_assign(c_var(sum),
            c_binop('+', c_var(sum), c_index(c_var(bias), c_var(col))))
    ;
        BiasStmt = c_comment('no bias')
    ),
    
    %% Epilogue: activation (fused — still in registers!)
    ( Activation = silu ->
        %% SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        ActStmt = c_assign(c_var(sum),
            c_binop('/', c_var(sum),
                c_paren(c_binop('+', c_float(1.0),
                    c_call(expf, [c_unop('-', c_var(sum))])))))
    ; Activation = relu ->
        ActStmt = c_assign(c_var(sum),
            c_ternary(c_binop('>', c_var(sum), c_float(0.0)),
                c_var(sum), c_float(0.0)))
    ; Activation = none ->
        ActStmt = c_comment('no activation')
    ;
        ActStmt = c_comment('unknown activation')
    ),
    
    %% Write result
    WriteResult = c_assign(
        c_index(c_var('C'), c_binop('+', c_binop('*', c_var(row), c_var('N')), c_var(col))),
        c_var(sum)),
    
    %% Assemble the kernel function
    Params = [
        param(c_type(const_ptr(c_type(float))), 'A'),
        param(c_type(const_ptr(c_type(float))), 'B'),
        param(c_type(ptr(c_type(float))), 'C'),
        param(c_type(const_ptr(c_type(float))), bias),
        param(c_type(int), 'M'),
        param(c_type(int), 'N'),
        param(c_type(int), 'K')
    ],
    
    Body = [
        Row, Col, BoundsCheck, c_blank,
        Accum, InnerLoop, c_blank,
        c_comment('--- FUSED EPILOGUE (no VRAM round-trip) ---'),
        BiasStmt, ActStmt, c_blank,
        WriteResult
    ],
    
    KernelAST = c_func(['__global__'], c_type(void),
        fused_matmul_bias_act, Params, Body).

%% ═══════════════════════════════════════════════════════════════
%% TEST
%% ═══════════════════════════════════════════════════════════════

test :-
    write('=== Fused Matmul+Bias+SiLU Kernel ==='), nl, nl,
    generate_fused_matmul_epilogue(_, _, _, true, silu, Kernel),
    emit_c(Kernel, Code),
    write(Code), nl, nl,
    
    write('=== Fused Matmul (no bias, ReLU) ==='), nl, nl,
    generate_fused_matmul_epilogue(_, _, _, false, relu, Kernel2),
    emit_c(Kernel2, Code2),
    write(Code2), nl.

:- initialization((test -> halt(0) ; (write('FAILED'), nl, halt(1)))).
