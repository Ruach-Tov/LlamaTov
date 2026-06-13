%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_ffn_lifter.pl — verify FFN block lifting from C source
%%
%% Per Heath's directive: extend the lifter to FFN section.
%% Empirical finding from earlier work: the lifter ALREADY handles
%% build_ffn calls (via the generic c_assign + c_call pattern).
%% The substantive change is BLOCK NAMESPACING — sequence facts
%% must use ffn_block, not qkv_block.

:- use_module('../lib/qkv_lifter').

run_tests :-
    Tests = [
        test_lift_ffn_opaque_call,
        test_lift_ffn_uses_ffn_block_namespace,
        test_lift_ffn_extracts_parameters,
        test_qkv_lifter_still_uses_qkv_block,
        test_block_namespacing_isolated_between_calls,
        test_lift_ffn_preserves_op_level_as_builder,
        test_lift_block_section_explicit_namespace
    ],
    run_each(Tests, 0, 0, P, F),
    format("~n=============================================~n", []),
    format("RESULTS: ~d passed, ~d failed~n", [P, F]),
    format("=============================================~n", []),
    ( F > 0 -> halt(1) ; true ).

run_each([], P, F, P, F).
run_each([T | Rest], P0, F0, P, F) :-
    ( catch(call(T), Err, (format("  FAIL ~w: error ~w~n", [T, Err]), fail))
    -> ( format("  PASS ~w~n", [T]), P1 is P0 + 1, F1 = F0 )
    ; ( format("  FAIL ~w~n", [T]), P1 = P0, F1 is F0 + 1 )
    ),
    run_each(Rest, P1, F1, P, F).

%% Real Qwen2 FFN C call (with build_ffn opaque builder)
ffn_source('cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);').

%% Real Qwen2 QKV bias-add pattern (for backward-compat verification)
qkv_source('Qcur = build_lora_mm(model.layers[il].wq, cur);').

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

test_lift_ffn_opaque_call :-
    ffn_source(Source),
    qkv_lifter:lift_ffn_section(Source, Facts),
    %% The build_ffn op should be lifted as opaque builder
    member(op_kind(cur, build_ffn), Facts),
    member(op_level(cur, builder), Facts).

test_lift_ffn_uses_ffn_block_namespace :-
    ffn_source(Source),
    qkv_lifter:lift_ffn_section(Source, Facts),
    %% Sequence facts should use ffn_block, NOT qkv_block
    member(sequence(ffn_block, cur, _), Facts),
    \+ member(sequence(qkv_block, cur, _), Facts).

test_lift_ffn_extracts_parameters :-
    ffn_source(Source),
    qkv_lifter:lift_ffn_section(Source, Facts),
    %% All three FFN weight tensors should be lifted as parameters
    member(parameter(ffn_up, layer(il), from_hparams), Facts),
    member(parameter(ffn_gate, layer(il), from_hparams), Facts),
    member(parameter(ffn_down, layer(il), from_hparams), Facts).

test_qkv_lifter_still_uses_qkv_block :-
    %% Backward compat: existing lift_qkv_section must still emit qkv_block
    qkv_source(Source),
    qkv_lifter:lift_qkv_section(Source, Facts),
    member(sequence(qkv_block, _, _), Facts),
    \+ member(sequence(ffn_block, _, _), Facts).

test_block_namespacing_isolated_between_calls :-
    %% After calling lift_ffn_section, a subsequent lift_qkv_section
    %% should not "leak" ffn_block. Verifies setup_call_cleanup hygiene.
    ffn_source(FfnSrc),
    qkv_source(QkvSrc),
    qkv_lifter:lift_ffn_section(FfnSrc, _),       % first call (ffn_block)
    qkv_lifter:lift_qkv_section(QkvSrc, Facts2),  % second call must be qkv_block
    member(sequence(qkv_block, _, _), Facts2),
    \+ member(sequence(ffn_block, _, _), Facts2).

test_lift_ffn_preserves_op_level_as_builder :-
    %% build_ffn is in op_level_of/2 as builder; verify preserved
    ffn_source(Source),
    qkv_lifter:lift_ffn_section(Source, Facts),
    member(op_level(cur, Level), Facts),
    Level == builder.

test_lift_block_section_explicit_namespace :-
    %% lift_block_section/3 allows arbitrary block namespace.
    %% Verify by using a custom block name (residual_block).
    ffn_source(Source),
    qkv_lifter:lift_block_section(residual_block, Source, Facts),
    member(sequence(residual_block, cur, _), Facts),
    \+ member(sequence(qkv_block, _, _), Facts),
    \+ member(sequence(ffn_block, _, _), Facts).

:- initialization(run_tests, main).
