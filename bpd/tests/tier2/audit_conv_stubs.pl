%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_module('lib/c_ast').
:- use_module('lib/kernel_templates_blas',
              except([kernel_available_fixes/2, fix_description/2])).
:- use_module('lib/kernel_templates').

audit_case(OpKind, Dims, Direction) :-
    catch(
        ( generate_kernel_im2col(OpKind, Dims, Direction, K),
          K = c_func(_, _, KernelName, Params, Body),
          length(Params, NumParams),
          length(Body, BodyLen),
          format("OK     ~w / dims=~w / dir=~w: kernel=~w params=~w body_stmts=~w~n",
                 [OpKind, Dims, Direction, KernelName, NumParams, BodyLen])
        ),
        Err,
        format("ERR    ~w / dims=~w / dir=~w: ~q~n",
               [OpKind, Dims, Direction, Err])
    ).

main :-
    audit_case(ggml_conv_1d, 1, forward),
    audit_case(ggml_conv_2d, 2, forward),
    audit_case(ggml_conv_3d, 3, forward),
    audit_case(ggml_conv_transpose_1d, 1, transpose),
    audit_case(ggml_conv_transpose_2d, 2, transpose),
    audit_case(ggml_conv_transpose_3d, 3, transpose),
    halt(0).

:- initialization(main).
