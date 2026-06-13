%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_ffn_roundtrip.pl — FFN section byte-identical round-trip
%%
%% Per medayek's "structurally verified" epistemic category: FFN section
%% round-trips through C → AST → BPD → AST byte-identically, matching
%% the property QKV section has had since commit c4b464268.
%%
%% Empirical demonstration: the lifter+generator pair preserves enough
%% information to reconstruct the original C source AST exactly.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/qkv_lifter').
:- use_module('../lib/qkv_generator').

run_tests :-
    Tests = [
        test_ffn_opaque_call_roundtrips,
        test_ffn_AST_identical_after_roundtrip,
        test_ffn_no_bias_NULL_handling,
        test_ffn_preserves_block_namespace,
        test_qkv_still_roundtrips_after_NULL_fix
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

%% Helper: verify byte-identical roundtrip for a given source
verify_byte_identical_roundtrip(LiftPred, Source) :-
    c_ast:c_parse_stmts_v2(Source, ASTOriginal),
    call(qkv_lifter:LiftPred, Source, BPDFacts),
    qkv_generator:generate_from_bpd(BPDFacts, ASTRegenerated),
    length(ASTOriginal, NOrig),
    length(ASTRegenerated, NRegen),
    NOrig == NRegen,
    forall(
        ( nth0(I, ASTOriginal, Orig),
          nth0(I, ASTRegenerated, Regen) ),
        Orig == Regen
    ).

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

%% Real Qwen2 FFN call (with build_ffn opaque builder).
test_ffn_opaque_call_roundtrips :-
    Source = 'cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);',
    verify_byte_identical_roundtrip(lift_ffn_section, Source).

%% The CORE arxiv-result claim for FFN section: AST identity.
test_ffn_AST_identical_after_roundtrip :-
    Source = 'cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);',
    c_ast:c_parse_stmts_v2(Source, ASTOriginal),
    qkv_lifter:lift_ffn_section(Source, BPDFacts),
    qkv_generator:generate_from_bpd(BPDFacts, ASTRegenerated),
    ASTOriginal == ASTRegenerated.

%% NULL handling: build_ffn passes NULL for unused bias/scale fields.
%% Parser sees NULL as c_var(NULL); generator must produce same.
%% Use a known op kind (ggml_add) so the lifter can process it.
test_ffn_no_bias_NULL_handling :-
    Source = 'x = ggml_add(NULL, NULL, NULL);',
    c_ast:c_parse_stmts_v2(Source, ASTOriginal),
    qkv_lifter:lift_block_section(test_block, Source, BPDFacts),
    qkv_generator:generate_from_bpd(BPDFacts, ASTRegenerated),
    ASTOriginal == ASTRegenerated.

%% After FFN round-trip, the BPD facts use ffn_block namespace.
test_ffn_preserves_block_namespace :-
    Source = 'cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);',
    qkv_lifter:lift_ffn_section(Source, BPDFacts),
    %% Sequence fact should use ffn_block (not qkv_block)
    member(sequence(ffn_block, cur, 1), BPDFacts),
    \+ member(sequence(qkv_block, _, _), BPDFacts).

%% After the NULL fix, QKV roundtrip MUST still hold. (Regression check.)
test_qkv_still_roundtrips_after_NULL_fix :-
    Source = 'cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);',
    verify_byte_identical_roundtrip(lift_qkv_section, Source).

:- initialization(run_tests, main).
