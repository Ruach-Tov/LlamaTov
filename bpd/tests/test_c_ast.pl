%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_c_ast.pl — Smoke test for the C AST library.
%%
%% Generates a small C function from AST terms and prints it.

:- use_module('../lib/c_ast').

test_simple_function :-
    Program = [
        c_include('models.h'),
        c_blank,
        c_func(c_type(void), hello_world, [], [
            c_comment('A simple generated function'),
            c_decl_init(c_type(int), x, c_int(42)),
            c_decl(c_type(ptr(c_type(named(ggml_tensor)))), cur),
            c_blank,
            c_assign(c_var(cur), 
                c_call(ggml_mul_mat, [c_var(ctx0), c_var(weight), c_var(input)])),
            c_blank,
            c_if(c_binop('!=', c_var(bias), c_nullptr), [
                c_assign(c_var(cur),
                    c_call(ggml_add, [c_var(ctx0), c_var(cur), c_var(bias)]))
            ]),
            c_blank,
            c_for(
                c_binop('=', c_var(il), c_int(0)),
                c_binop('<', c_var(il), c_var(n_layer)),
                c_unop('++', c_var(il)),
                [
                    c_comment('layer body'),
                    c_assign(c_var(cur),
                        c_call(build_norm, [
                            c_var(inpL),
                            c_member(c_index(c_member(c_var(model), layers), c_var(il)), attn_norm),
                            c_null
                        ]))
                ]
            )
        ])
    ],
    emit_program(Program, Code),
    write(Code), nl.

:- initialization(test_simple_function).
