%% ─────────────────────────────────────────────────────────────────────────────
%% moe_dispatch.pl — the glue: map a deriver MoE op (op/4 fact) to its kernel-template call.
%%
%% gguf_to_graph.py emits MoE op_kinds (ggml_top_k, ggml_get_rows, weighted_scatter_add);
%% kernel_templates_moe.pl (boneh) generates the kernels. This bridges them: given an op/4 fact,
%% extract the template parameters from the op's args and invoke the right generate_kernel_*.
%%
%% Resolves the one contract detail: the deriver encodes the static top-k as an atom (e.g. k2 in
%% ggml_top_k(router_logits, k2)); the template wants k as an integer (generate_kernel_top_k(2, ...)).
%% k_atom_int/2 extracts it. n_experts / rows / cols / vec_len stay symbolic (generic kernels).
%%
%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ─────────────────────────────────────────────────────────────────────────────
:- module(moe_dispatch, [
    moe_op_kernel/2,        % moe_op_kernel(+Op, -KernelAST)  — dispatch one op/4 to its kernel
    moe_dispatch_graph/2,   % moe_dispatch_graph(+Graph, -Kernels) — all MoE kernels for a graph
    k_atom_int/2            % k_atom_int(+Atom, -Int)  — extract integer k from the deriver's k-atom
]).
:- use_module(kernel_templates_moe).

%% k_atom_int(+KAtom, -K): the deriver emits the static top-k as an atom like 'k2'. Strip the 'k'
%% prefix to recover the integer. (If already an integer, pass through.)
k_atom_int(K, K) :- integer(K), !.
k_atom_int(KAtom, K) :-
    atom(KAtom),
    atom_concat(k, NumAtom, KAtom),       % 'k2' -> NumAtom='2'
    atom_number(NumAtom, K), integer(K).

%% moe_op_kernel(+Op, -KernelAST): dispatch a single deriver op/4 to its generated kernel.
%% Only MoE op_kinds are handled; others fail (so a caller can findall the MoE ones).

%% ggml_top_k(router_logits, KAtom) -> top_k kernel. n_experts stays symbolic.
moe_op_kernel(op(_Id, ggml_top_k, [_RouterLogits, KAtom], _Out), KernelAST) :-
    k_atom_int(KAtom, K),
    generate_kernel_top_k(K, n_experts, KernelAST).

%% ggml_get_rows(stacked_weight, index) -> gather kernel. rows/cols symbolic (kernel is generic
%% over the gathered slice dims; resolved at C-compile from the tensor shape).
moe_op_kernel(op(_Id, ggml_get_rows, [_Stacked, _Index], _Out), KernelAST) :-
    generate_kernel_get_rows(rows, cols, KernelAST).

%% weighted_scatter_add([eout_0..eout_{k-1}, expert_sel]) -> weighted accumulate kernel.
%% k = number of expert outputs being combined = (length of inputs) - 1 (the last is expert_sel).
moe_op_kernel(op(_Id, weighted_scatter_add, Ins, _Out), KernelAST) :-
    length(Ins, NIns), K is NIns - 1, K >= 1,
    generate_kernel_weighted_scatter_add(K, vec_len, KernelAST).

%% moe_dispatch_graph(+Graph, -Kernels): generate a kernel for every MoE op in the graph.
%% De-duplicates by kernel function name (top_k_2 / get_rows / weighted_scatter_add_k2 are emitted
%% once per distinct (op_kind, k), not once per layer).
moe_dispatch_graph(Graph, Kernels) :-
    findall(K, ( member(Op, Graph), moe_op_kernel(Op, K) ), Ks0),
    dedup_by_funcname(Ks0, Kernels).

dedup_by_funcname(Ks, Unique) :-
    findall(Name-K, ( member(K, Ks), K = c_func(_, Name, _, _) ), Pairs),
    keysort(Pairs, Sorted),
    dedup_pairs(Sorted, Unique).
dedup_pairs([], []).
dedup_pairs([N-K|Rest], [K|Out]) :-
    exclude([N2-_]>>(N2 == N), Rest, Rest2),
    dedup_pairs(Rest2, Out).
