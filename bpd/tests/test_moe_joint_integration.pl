%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
:- use_module('kernel_templates_moe').
:- initialization(main).
main :-
  ( generate_kernel_top_k(2, n_experts, A1), A1=c_func(_,N1,_,_)
    -> format("PASS top_k -> ~w~n",[N1]) ; writeln('FAIL top_k') ),
  ( generate_kernel_get_rows(rows, cols, A2), A2=c_func(_,N2,_,_)
    -> format("PASS get_rows -> ~w~n",[N2]) ; writeln('FAIL get_rows') ),
  ( generate_kernel_weighted_scatter_add(2, vec_len, A3), A3=c_func(_,N3,_,_)
    -> format("PASS wsa -> ~w~n",[N3]) ; writeln('FAIL wsa') ),
  halt.
