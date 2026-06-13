%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% qkv_lifter.pl — Lift QKV BPD facts from llama.cpp Qwen2 C source.
%%
%% Per mavchin's direction: "do the actual llama.cpp Qwen2 lifter."
%% This is the convergence work that uses BOTH substrate tracks:
%%   - mavchin's bidirectional DCG parser (c_ast.pl) for C → AST
%%   - this module for AST → BPD facts
%%   - mavchin's generator for BPD → C regeneration
%%   - composed idempotency: original C ≡ regenerated C (clang-format)
%%
%% Lifts to the FULL QKV BPD vocabulary metayen used in qkv.bpd:
%%   op_kind/2, op_inputs/2, op_output/2, op_level/2,
%%   tensor_join/2, parameter/3, sequence/3
%%
%% Then we compare lifted facts against the original qkv.bpd to verify
%% the lifter preserves all structurally meaningful information.

:- module(qkv_lifter, [
    lift_qkv_section/2,
    lift_ffn_section/2,
    lift_block_section/3,
    lift_ast_to_bpd/2,
    lift_ast_to_bpd_with_block/3,
    op_level_of/2
]).

:- use_module('c_ast').

%% ────────────────────────────────────────────────────────────────────
%% Operation level classification (builder vs primitive)
%% ────────────────────────────────────────────────────────────────────
%%
%% Per the faithfulness/optimization layering principle:
%% builder ops are opaque to fusion (build_norm, build_lora_mm,
%% build_attn). Primitive ops are fusion candidates (ggml_add,
%% ggml_reshape_3d, ggml_rope_ext).

op_level_of(build_lora_mm, builder).
op_level_of(build_norm, builder).
op_level_of(build_attn, builder).
op_level_of(build_ffn, builder).
op_level_of(ggml_add, primitive).
op_level_of(ggml_mul, primitive).
op_level_of(ggml_reshape_3d, primitive).
op_level_of(ggml_rope_ext, primitive).
op_level_of(ggml_silu, primitive).
op_level_of(ggml_gelu, primitive).

%% ────────────────────────────────────────────────────────────────────
%% Architecture parameter recognition
%% ────────────────────────────────────────────────────────────────────
%%
%% Pattern: model.layers[il].X is an ARCHITECTURE PARAMETER (e.g., wq,
%% bq, attn_norm). When extracting BPD facts, these should be recognized
%% as parameter references, not as anonymous tensors.

is_arch_param_access(c_member(c_index(c_member(c_var(model), layers),
                                       c_var(il)),
                              ParamName),
                     parameter(ParamName, layer(il))).

%% ────────────────────────────────────────────────────────────────────
%% Main lift predicates
%% ────────────────────────────────────────────────────────────────────
%%
%% lift_qkv_section(+CSource, -BPDFacts) — top-level entry point
%% Takes C source as an atom, parses to AST, extracts BPD facts.
%%
%% KNOWN LIMITATION (substrate-honest):
%% When C source re-assigns a name (e.g., Qcur = reshape(Qcur, ...);
%% Qcur = rope(Qcur, ...);), the lifter produces multiple op facts
%% with the same output name. In the manually-authored qkv.bpd, these
%% are distinguished as :Qcur_3d (after reshape) and :Qcur (after rope).
%% The lifter currently doesn't auto-generate disambiguating SSA names.
%%
%% This is structurally correct (facts faithfully represent C structure)
%% but semantically incomplete (the C code's "Qcur becomes a new value"
%% is implicit in C semantics; in BPD we'd usually make it explicit).
%%
%% Resolution: a post-lift pass that detects re-assignment and rewrites
%% to SSA form. Deferred until the lifter handles more architectures
%% and we have empirical data on whether the SSA-disambiguation is
%% worth its complexity.

lift_qkv_section(CSource, BPDFacts) :-
    lift_block_section(qkv_block, CSource, BPDFacts).

%% lift_ffn_section(+CSource, -BPDFacts) — lift FFN block C source.
%% Same as lift_qkv_section but emits sequence/3 facts with ffn_block
%% namespace instead of qkv_block.
lift_ffn_section(CSource, BPDFacts) :-
    lift_block_section(ffn_block, CSource, BPDFacts).

%% lift_block_section(+Block, +CSource, -BPDFacts) — block-aware lifter.
%% The Block atom (e.g., qkv_block, ffn_block, residual_block) becomes
%% the first argument of all emitted sequence/3 facts.
lift_block_section(Block, CSource, BPDFacts) :-
    c_ast:c_parse_stmts_v2(CSource, AST),
    lift_ast_to_bpd_with_block(Block, AST, BPDFacts).

%% lift_ast_to_bpd(+AST, -BPDFacts) — backward-compatible alias.
%% Defaults to qkv_block for sequence namespace.
lift_ast_to_bpd(AST, BPDFacts) :-
    lift_ast_to_bpd_with_block(qkv_block, AST, BPDFacts).

%% lift_ast_to_bpd_with_block(+Block, +AST, -BPDFacts)
%% Use a thread-local dynamic flag to pass the block name through to
%% lift_stmt without changing every call signature. This is a controlled
%% use of global state — set at start of lift, read inside lift_stmt,
%% cleared at end (in success and failure cases via setup_call_cleanup).
lift_ast_to_bpd_with_block(Block, AST, BPDFacts) :-
    setup_call_cleanup(
        asserta(current_lift_block(Block)),
        ( lift_stmts(AST, 1, [], FactsReversed),
          reverse(FactsReversed, BPDFacts)
        ),
        retractall(current_lift_block(_))
    ).

:- dynamic(current_lift_block/1).

%% current_block(-Block) — read the active block name during lifting.
%% Defaults to qkv_block if no block is currently set (backward compat).
current_block(Block) :-
    current_lift_block(Block), !.
current_block(qkv_block).

%% lift_stmts(+AST, +SeqStart, +Acc, -Facts)
lift_stmts([], _, Acc, Acc).
lift_stmts([Stmt | Rest], Seq, Acc, Facts) :-
    ( lift_stmt(Stmt, Seq, StmtFacts)
    -> append(StmtFacts, Acc, Acc1),
       Seq1 is Seq + 1
    ;  Acc1 = Acc, Seq1 = Seq
    ),
    lift_stmts(Rest, Seq1, Acc1, Facts).

%% ────────────────────────────────────────────────────────────────────
%% Statement-level lifting rules
%% ────────────────────────────────────────────────────────────────────

%% Pattern: Output = build_*(Args...) → builder op
%% e.g., Qcur = build_lora_mm(model.layers[il].wq, cur)
lift_stmt(c_assign(c_var(Output), c_call(OpName, Args)), Seq, Facts) :-
    op_level_of(OpName, Level),
    process_args(Args, Inputs, ParamFacts),
    current_block(Block),
    Facts = [
        op(Output),
        op_kind(Output, OpName),
        op_inputs(Output, Inputs),
        op_output(Output, Output),
        op_level(Output, Level),
        sequence(Block, Output, Seq)
        | ParamFacts
    ].

%% Pattern: cb(Tensor, "Label", il) → callback fact
lift_stmt(c_expr_stmt(c_call(cb, [c_var(Tensor), c_string(Label), c_var(il)])),
          Seq,
          [cb_after_v2(Tensor, Label, Seq)]).

%% Pattern: if (model.layers[il].bP) { Output = ggml_add(...); cb(...); }
%% This is the OPTIONAL BIAS pattern. Emit tensor_join + the ops in the body.
lift_stmt(c_if(Condition, Body), Seq, Facts) :-
    is_arch_param_access(Condition, parameter(ParamName, layer(il))),
    %% Body should contain an assignment + optionally a cb call
    lift_conditional_body(Body, ParamName, Seq, Facts).

lift_conditional_body([c_assign(c_var(JoinName),
                                c_call(ggml_add, [c_var(Ctx), c_var(Input), _BiasArg])),
                       c_expr_stmt(c_call(cb,
                            [c_var(JoinName), c_string(Label), c_var(il)]))],
                      ParamName, Seq,
                      Facts) :-
    %% Conditional ggml_add: JoinName = ggml_add(ctx0, Input, bias)
    %% The bias-applied version is JoinName_post_bias; without bias it's Input.
    %% Emit tensor_join facts modeling this.
    %%
    %% Preserve enough info for round-trip generation:
    %%   - Ctx (the ggml context handle)
    %%   - Input (the tensor being added to)
    %%   - ParamName (the architecture parameter providing the bias)
    %%   - Label (the cb label)
    atom_concat(JoinName, '_post_bias', PostBias),
    current_block(Block),
    Facts = [
        op(PostBias),
        op_kind(PostBias, ggml_add),
        %% Inputs preserve all THREE call args:
        %%   - Ctx (context handle, regular var)
        %%   - Input (tensor reference)
        %%   - ParamName (architecture parameter — picked up via parameter/3 facts)
        op_inputs(PostBias, [Ctx, Input, ParamName]),
        op_output(PostBias, PostBias),
        op_level(PostBias, primitive),
        op_condition(PostBias, present(ParamName)),
        sequence(Block, PostBias, Seq),
        tensor_join(JoinName,
                    [if(present(ParamName), PostBias, Input)]),
        %% Also preserve the cb_after label so generator can regenerate
        cb_after_v2(JoinName, Label, Seq),
        %% Emit parameter fact for the bias param so generator knows
        %% it's a model.layers[il].X access, not a plain var.
        parameter(ParamName, layer(il), from_hparams)
    ].

%% Pattern: function-call statement (no assignment)
%% e.g., raw cb(...) at top level — already handled above
%% but bare expression statements that don't match patterns are skipped

%% ────────────────────────────────────────────────────────────────────
%% Argument processing
%% ────────────────────────────────────────────────────────────────────
%%
%% process_args(+Args, -InputNames, -ParameterFacts) walks the C call
%% arguments. Each argument becomes either:
%%   - A simple tensor reference (var name)
%%   - A parameter reference (model.layers[il].X — emit parameter fact)

process_args([], [], []).
process_args([Arg | Rest], [InputName | RestInputs], ParamFacts) :-
    ( is_arch_param_access(Arg, parameter(ParamName, layer(il)))
    -> InputName = ParamName,
       ThisParamFact = [parameter(ParamName, layer(il), from_hparams)]
    ;  Arg = c_var(InputName)
    -> ThisParamFact = []
    ;  Arg = c_null
    -> InputName = 'NULL',
       ThisParamFact = []
    ;  Arg = c_string(_)
    -> InputName = '_STRING',
       ThisParamFact = []
    ;  Arg = c_member(_, MemberName)
    -> InputName = MemberName,
       ThisParamFact = []
    ;  InputName = unknown,
       ThisParamFact = []
    ),
    process_args(Rest, RestInputs, RestFacts),
    append(ThisParamFact, RestFacts, ParamFacts).
