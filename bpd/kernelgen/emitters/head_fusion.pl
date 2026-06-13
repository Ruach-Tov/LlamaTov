%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% head_fusion.pl — wire the fusion RECOGNIZER (bpd/lib/fusion_rules.pl +
%% iterative_fusion.pl) to the kernelgen codegen path, for EPILOGUE fusion of an
%% operand-bound head (conv/gemm/pool) followed by an elementwise tail.
%%
%% DESIGN (Heath, 2026-06-08): the visitor/recognizer approach — DISCOVER the
%% fusion from the already-constructed chain (op graph), don't INCORPORATE it as a
%% constructor parameter. Head emitters stay pure; this pass recognizes
%% [head | elementwise-tail], and produces a FUSED-HEAD spec that the head's
%% lowering consumes (inlining the lowered tail at the C-store via -DCONV_EPILOGUE).
%%
%% Built FOR COMPARATIVE MEASUREMENT: fuse_spec/3 yields BOTH a fused build (the
%% -D epilogue flag) and the non-fused baseline, so a harness can build+verify+time
%% both and report fused-vs-non-fused speedup, gated on bit-exact / within-contract.
%%
%% Pipeline: chain -> recognizer_facts -> discover_epilogue -> lower_epilogue ->
%%           fuse_spec(Chain, FusedDFlags, NonFusedSteps).
%% Author: Iyun, 2026-06-08
%% ═══════════════════════════════════════════════════════════════════════════
:- module(head_fusion, [
    recognizer_facts/2,      % recognizer_facts(+Chain, -Facts)
    discover_epilogue/4,     % discover_epilogue(+Chain, -HeadOp, -TailOps, -RuleName)
    lower_epilogue_cuda/3,   % lower_epilogue_cuda(+TailOps, -EpilogueExpr, -NeedsBias)
    fuse_spec/2              % fuse_spec(+Chain, -Spec)  [the comparison spec]
]).
:- use_module(library(lists)).
:- use_module('../../lib/iterative_fusion.pl').
:- use_module('../emitters/chain_compose.pl').
:- use_module('../../lib/robust_op_match.pl', [op_expr/2]).

%% op_class for our bpd_* op KINDS, in the recognizer's module. Operand-bound
%% heads (conv/gemm/pool) are 'matmul'-class for the epilogue rule (which keys on
%% matmul-class producer). Elementwise tail ops are 'elementwise'.
:- multifile fusion_rules:op_class/2.
fusion_rules:op_class(K, matmul)      :- head_kind(K).
fusion_rules:op_class(K, elementwise) :- tail_kind(K).

head_kind(conv2d). head_kind(conv1d). head_kind(conv3d).
head_kind(matmul). head_kind(gemm).
head_kind(maxpool2d). head_kind(avgpool2d).

tail_kind(K) :- op_kind_atom(K, Op), var_composable(Op).

%% map a bare kind atom (conv2d) <-> the bpd_ op name (bpd_conv2d)
op_kind_atom(K, Op) :- atom_concat(bpd_, K, Op).
kind_of(Op, K) :- atom_concat(bpd_, K, Op).

%% ── recognizer_facts(+Chain, -Facts): emit the recognizer's vocabulary ──
%% Chain = [bpd_conv2d, bpd_relu, bpd_scalar_add]. Each op -> op/op_kind/op_inputs/
%% op_output with a synthesized intermediate-tensor name; writes/reads regions.
recognizer_facts(Chain, Facts) :-
    %% (dead numlist/pairs rows removed — singleton sweep, review tour 2026-06-13)
    findall(F, ( nth1(I, Chain, Op), kind_of(Op, K),
                 op_id(I, OpId), in_tensor(I, In), out_tensor(I, Out),
                 member(F, [ op(OpId), op_kind(OpId, K),
                             op_output(OpId, Out), op_inputs(OpId, [In]),
                             op_writes(OpId, Out, region(matmul_output, shape([n]))),
                             op_reads(OpId, In, region(elementwise, shape([n]))) ]) ),
            Facts0),
    Facts = Facts0.

op_id(I, OpId)   :- atom_concat(op, I, OpId).
out_tensor(I, T) :- atom_concat(t, I, T).
in_tensor(1, x) :- !.                       % first op reads external input x
in_tensor(I, T) :- I0 is I-1, atom_concat(t, I0, T).   % later op reads prev output

%% ── discover_epilogue: run the recognizer, return head + fused tail ──
%% A chain [Head | EwTail] where Head is operand-bound and EwTail are all
%% var-composable elementwise. Uses chain_compose's split_chain (already proven).
discover_epilogue(Chain, HeadOp, TailOps, epilogue_head_elementwise) :-
    split_chain(Chain, [HeadOp], TailOps),
    TailOps \= [],
    head_kind_of(HeadOp).
head_kind_of(Op) :- kind_of(Op, K), head_kind(K).

%% ── lower_epilogue_cuda: the composed tail term -> a C-expression on (v) ──
%% Reuses compose_chain (folds the tail to one op_expr term) + lower_cuda
%% (var -> "v"). So CONV_EPILOGUE(v,oc) = the lowered tail with input bound to v.
lower_epilogue_cuda(TailOps, Expr, NeedsBias) :-
    compose_chain(TailOps, Term),
    expr_ir:lower_cuda(Term, Raw),
    % lower_cuda emits the input as literal "v" — exactly our macro arg.
    Expr = Raw,
    ( sub_atom(Raw, _, _, _, 'bias') -> NeedsBias = true ; NeedsBias = false ).

%% ── fuse_spec(+Chain, -Spec): the COMPARISON spec for the harness ──
%% Spec = spec(Head, Tail, fused(DFlag), nonfused(HeadAlone, TailKernels)).
%% The harness builds BOTH (fused: head -DCONV_EPILOGUE=Expr ; non-fused: head +
%% the tail as a separate elementwise kernel), verifies bit-exact, times both.
fuse_spec(Chain, Spec) :-
    discover_epilogue(Chain, HeadOp, TailOps, Rule),
    lower_epilogue_cuda(TailOps, Expr, NeedsBias),
    head_macro(HeadOp, Macro),
    format(atom(DFlag), "-D~w=(~w)", [Macro, Expr]),
    Spec = spec(rule(Rule),
                head(HeadOp),
                tail(TailOps),
                fused(dflag(DFlag), needs_bias(NeedsBias)),
                nonfused(head(HeadOp), tail_kernels(TailOps))).

%% head_macro(+HeadOp, -MacroName): which -D macro the head's C-store reads.
%% conv -> CONV_EPILOGUE ; matmul/gemm -> GEMM_EPILOGUE ; pool emitter takes the
%% epilogue as a Prolog arg (emit_cuda_pool/3), not a -D, so no macro here.
head_macro(Op, 'CONV_EPILOGUE') :- kind_of(Op, K), member(K, [conv2d, conv1d, conv3d]), !.
head_macro(Op, 'GEMM_EPILOGUE') :- kind_of(Op, K), member(K, [matmul, gemm]), !.
head_macro(_,  'CONV_EPILOGUE').   % default
