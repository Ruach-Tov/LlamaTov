%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').

test :-
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

:- initialization((test -> halt(0) ; (write('FAILED'), nl, halt(1)))).
