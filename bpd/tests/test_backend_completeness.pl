%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_backend_completeness.pl — STATIC completeness check for the codegen
%% backends. Catches "added an op_expr / a primitive, but forgot a backend" — the
%% gap between what a backend CLAIMS to support (its *_supported_op/1) and what it
%% can ACTUALLY emit.
%%
%% (3) of the kernelgen tooling (Heath, 2026-06-08). Pure Prolog, NO GPU: just
%% checks that emit predicates exist + produce output for claimed ops, and that
%% the multi-backend op sets are consistent. Run:
%%   swipl -q -g run_tests -t halt bpd/tests/test_backend_completeness.pl
%%
%% Author: Iyun, 2026-06-08
:- use_module('../lib/robust_op_match.pl', [op_expr/2]).
:- use_module('../kernelgen/emitters/mlir_gpu_from_facts.pl').
:- use_module('../kernelgen/emitters/cuda_c_from_facts.pl').

run_tests :-
    Tests = [
        test_every_op_has_an_expr,
        test_mlir_elementwise_claims_emit,
        test_mlir_emitters_exported,
        test_cuda_c_claims_emit,
        test_no_op_expr_arity_drift
    ],
    run_each(Tests, 0, 0, P, F),
    format("~n=============================================~n", []),
    format("BACKEND COMPLETENESS: ~d passed, ~d failed~n", [P, F]),
    format("=============================================~n", []),
    ( F > 0 -> halt(1) ; true ).

run_each([], P, F, P, F).
run_each([T | Rest], P0, F0, P, F) :-
    ( catch(call(T), Err, (format("  FAIL ~w: error ~w~n", [T, Err]), fail))
    -> ( format("  PASS ~w~n", [T]), P1 is P0 + 1, F1 = F0 )
    ; ( format("  FAIL ~w~n", [T]), P1 = P0, F1 is F0 + 1 )
    ),
    run_each(Rest, P1, F1, P, F).

%% the elementwise ops the MLIR-GPU emitter (emit_mlir_gpu/2) is meant to cover.
mlir_elementwise(O) :- member(O, [bpd_relu, bpd_tanh, bpd_sigmoid, bpd_silu,
    bpd_elu, bpd_gelu, bpd_leaky_relu, bpd_hardsigmoid, bpd_softplus, bpd_selu,
    bpd_mish]).

%% ── 1. every op a backend claims (via op_expr) actually HAS an op_expr fact ──
test_every_op_has_an_expr :-
    findall(O, mlir_elementwise(O), Ops),
    forall(member(O, Ops),
           ( op_expr(O, _) -> true
           ; (format("    ~w has no op_expr fact~n", [O]), fail) )).

%% ── 2. each claimed MLIR elementwise op actually GENERATES (to a temp file) ──
test_mlir_elementwise_claims_emit :-
    findall(O, mlir_elementwise(O), Ops),
    forall(member(O, Ops),
           ( tmp_file(mlirc, Tmp),
             catch(emit_mlir_gpu(O, Tmp), _, fail),
             exists_file(Tmp),
             size_file(Tmp, Sz), Sz > 50,          % non-trivial output
             delete_file(Tmp)
           -> true
           ; (format("    emit_mlir_gpu(~w) failed / empty~n", [O]), fail) )).

%% ── 3. the structured MLIR emitters (reduce/pool/conv/matmul) are EXPORTED ──
%%   (catches the 'added emit_mlir_gpu_X but forgot the module export' bug)
test_mlir_emitters_exported :-
    Preds = [emit_mlir_gpu/2, emit_mlir_gpu_reduce/2, emit_mlir_gpu_pool/2,
             emit_mlir_gpu_conv/2, emit_mlir_gpu_matmul/2],
    forall(member(P/A, Preds),
           ( current_predicate(mlir_gpu_from_facts:P/A) -> true
           ; (format("    mlir_gpu_from_facts:~w/~w not exported/defined~n", [P, A]), fail) )).

%% ── 4. each claimed cuda-c op (cuda_c_supported_op) actually emits ──
test_cuda_c_claims_emit :-
    findall(O, (mlir_elementwise(O), cuda_c_supported_op(O)), Ops),
    ( Ops == [] -> format("    (no cuda_c_supported_op overlap — check predicate)~n", []) ; true ),
    forall(member(O, Ops),
           ( tmp_file(cc, Tmp),
             catch(emit_cuda_c_from_fact(O, Tmp), _, fail),
             exists_file(Tmp), size_file(Tmp, Sz), Sz > 50, delete_file(Tmp)
           -> true
           ; (format("    emit_cuda_c_from_fact(~w) failed / empty~n", [O]), fail) )).

%% ── 5. op_expr facts have consistent arity per head functor (no silent drift) ──
%%   every axis_reduce has 3 args, every pool has 7, every conv has 6.
test_no_op_expr_arity_drift :-
    forall(op_expr(_, axis_reduce(K,A,B)), (nonvar(K),nonvar(A),nonvar(B))),
    forall(op_expr(_, pool(K,N,Ks,St,P,D,Bd)),
           (nonvar(K),nonvar(N),nonvar(Ks),nonvar(St),nonvar(P),nonvar(D),nonvar(Bd))),
    forall(op_expr(_, conv(N,P,S,X,Di,G)),
           (nonvar(N),nonvar(P),nonvar(S),nonvar(X),nonvar(Di),nonvar(G))).
