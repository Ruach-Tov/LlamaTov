%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').

test_chained_access :-
    % ctx->params.n_embd
    Expr = c_member(c_arrow(c_var(ctx), params), n_embd),
    phrase(c_ast:emit_expr(Expr), Codes),
    atom_codes(S, Codes),
    write('Chained access: '), write(S), nl,
    % model.layers[il].wq
    Expr2 = c_member(c_index(c_member(c_var(model), layers), c_var(il)), wq),
    phrase(c_ast:emit_expr(Expr2), Codes2),
    atom_codes(S2, Codes2),
    write('Array member: '), write(S2), nl.

test_struct_def :-
    Node = c_typedef_struct(gguf_header_t, [
        field(c_type(uint32_t), magic),
        field(c_type(uint32_t), version),
        field(c_type(uint64_t), tensor_count),
        field(c_type(uint64_t), metadata_kv_count)
    ]),
    emit_c(Node, S),
    write(S), nl.

test_switch :-
    Node = c_func(c_type(void), dispatch, [param(c_type(int), arch)], [
        c_switch(c_var(arch), [
            c_case(c_int(0), [
                c_expr_stmt(c_call(build_llama, [])),
                c_expr_stmt(c_call(break, []))
            ]),
            c_case(c_int(1), [
                c_expr_stmt(c_call(build_qwen2, [])),
                c_expr_stmt(c_call(break, []))
            ]),
            c_default([
                c_expr_stmt(c_call(abort, []))
            ])
        ])
    ]),
    emit_c(Node, S),
    write(S), nl.

test_addr_deref :-
    Stmts = [
        c_decl_init(c_type(ptr(c_type(int))), p, c_addr(c_var(x))),
        c_assign(c_deref(c_var(p)), c_int(42))
    ],
    phrase(c_ast:emit_stmts(Stmts, 0), Codes),
    atom_codes(S, Codes),
    write(S), nl.

test_check_magic :-
    Program = [
        c_include_sys('string.h'),
        c_include_sys('stdint.h'),
        c_blank,
        c_func([static], c_type(int), check_magic,
            [param(c_type(const_ptr(c_type(uint8_t))), raw)],
            [
                c_if(
                    c_binop('!=',
                        c_call(memcmp, [c_var(raw), c_string('GGUF'), c_int(4)]),
                        c_int(0)),
                    [c_return(c_int(-1))]
                ),
                c_return(c_int(0))
            ])
    ],
    emit_program(Program, Code),
    write(Code), nl.

run_all :-
    write('=== Chained Access ==='), nl,
    test_chained_access, nl,
    write('=== Struct Def ==='), nl,
    test_struct_def,
    write('=== Switch ==='), nl,
    test_switch,
    write('=== Addr/Deref ==='), nl,
    test_addr_deref,
    write('=== check_magic ==='), nl,
    test_check_magic,
    write('ALL TESTS PASSED'), nl.

:- initialization((run_all -> halt(0) ; (write('FAILED'), nl, halt(1)))).
