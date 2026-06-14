%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
:- use_module('moe_dispatch').
:- use_module('bpd/tests/synth_moe_graph.pl').
:- initialization(main).
main :-
  findall(op(I,K,In,O), model_graph:op(I,K,In,O), G),
  % the glue: every MoE op -> its kernel, de-duped
  moe_dispatch_graph(G, Kernels),
  length(Kernels, N),
  findall(Name, (member(c_func(_,Name,_,_), Kernels)), Names), sort(Names, NS),
  format("dispatched ~w distinct MoE kernels: ~w~n", [N, NS]),
  ( NS == [get_rows, top_k_2, weighted_scatter_add_k2]
  -> format("PASS: deriver op_kinds -> dispatch -> boneh templates -> 3 kernels (k-atom k2 extracted as integer 2)~n")
  ;  format("FAIL: got ~w~n", [NS]) ),
  % also verify k_atom_int directly
  ( k_atom_int(k2, 2), k_atom_int(k8, 8), k_atom_int(5, 5)
  -> format("PASS: k_atom_int k2->2, k8->8, 5->5~n") ; format("FAIL k_atom_int~n") ),
  halt.
