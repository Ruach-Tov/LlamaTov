%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_qkv_roundtrip.pl — Full round-trip: C → AST → BPD → AST → C
%%
%% Per mavchin's direction: "verify the full QKV section round-trips.
%% Q+K+V projections, all conditional biases, all reshapes, all RoPE.
%% One shot, full section, zero diff. The paper writes itself from
%% the commit log."
%%
%% The loop:
%%   C source [real qwen2.cpp Q projection]
%%     → c_ast:c_parse_stmts_v2 → AST_original
%%     → qkv_lifter:lift_qkv_section → BPD_facts
%%     → qkv_generator:generate_from_bpd → AST_regenerated
%%     → c_ast:emit → C_regenerated
%%   Compare AST_original ~ AST_regenerated (or C ~ C via clang-format)

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/qkv_lifter').
:- use_module('../lib/qkv_generator').

run_tests :-
    Tests = [
        test_simple_op_roundtrips_through_bpd,
        test_q_projection_roundtrips_through_bpd,
        test_full_qkv_block_roundtrips_through_bpd,
        test_q_projection_AST_identical_after_roundtrip,
        test_k_projection_AST_identical_after_roundtrip,
        test_v_projection_AST_identical_after_roundtrip,
        test_full_qkv_block_AST_identical_after_roundtrip
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

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

%% Test 1: A simple Q projection (just the matmul, no conditional bias)
%% should round-trip through the BPD layer.
test_simple_op_roundtrips_through_bpd :-
    Source = 'Qcur = build_lora_mm(model.layers[il].wq, cur);',
    %% Step 1: parse C → AST
    c_ast:c_parse_stmts_v2(Source, ASTOriginal),
    %% Step 2: lift AST → BPD facts
    qkv_lifter:lift_ast_to_bpd(ASTOriginal, BPDFacts),
    %% Step 3: regenerate AST from BPD facts
    qkv_generator:generate_from_bpd(BPDFacts, ASTRegenerated),
    %% Step 4: verify both ASTs are equivalent
    %% (For now, basic structural equivalence: same first statement)
    [FirstOriginal | _] = ASTOriginal,
    [FirstRegen | _] = ASTRegenerated,
    FirstOriginal == FirstRegen.

%% Test 2: Q projection with conditional bias should round-trip.
test_q_projection_roundtrips_through_bpd :-
    Source = 'Qcur = build_lora_mm(model.layers[il].wq, cur); if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); cb(Qcur, "Qcur", il); }',
    c_ast:c_parse_stmts_v2(Source, ASTOriginal),
    qkv_lifter:lift_ast_to_bpd(ASTOriginal, BPDFacts),
    qkv_generator:generate_from_bpd(BPDFacts, ASTRegenerated),
    %% Both should have 2 statements: the matmul + the if-block
    length(ASTOriginal, NOrig),
    length(ASTRegenerated, NRegen),
    NOrig == NRegen,
    %% First statement (Qcur assignment) should match structurally
    nth0(0, ASTOriginal, FirstOrig),
    nth0(0, ASTRegenerated, FirstRegen),
    FirstOrig = c_assign(c_var(Qcur), c_call(build_lora_mm, _)),
    FirstRegen = c_assign(c_var(Qcur), c_call(build_lora_mm, _)).

%% Test 3: Full QKV block (norm + Q + K + V + conditional biases) round-trips
test_full_qkv_block_roundtrips_through_bpd :-
    Source = 'cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il); Qcur = build_lora_mm(model.layers[il].wq, cur); if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); cb(Qcur, "Qcur", il); } Kcur = build_lora_mm(model.layers[il].wk, cur); if (model.layers[il].bk) { Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk); cb(Kcur, "Kcur", il); } Vcur = build_lora_mm(model.layers[il].wv, cur); if (model.layers[il].bv) { Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv); cb(Vcur, "Vcur", il); }',
    c_ast:c_parse_stmts_v2(Source, ASTOriginal),
    qkv_lifter:lift_ast_to_bpd(ASTOriginal, BPDFacts),
    qkv_generator:generate_from_bpd(BPDFacts, ASTRegenerated),
    %% Both ASTs should have 7 top-level statements:
    %% norm, Qcur+if, Kcur+if, Vcur+if = 1 + 2 + 2 + 2 = 7
    length(ASTOriginal, NOrig),
    length(ASTRegenerated, NRegen),
    NOrig == NRegen,
    format("    Original AST has ~d statements; regenerated has ~d~n",
           [NOrig, NRegen]).

%% THE CORE ARXIV RESULT: full Q-projection (matmul + conditional bias)
%% produces BYTE-IDENTICAL AST after round-trip through BPD facts.
test_q_projection_AST_identical_after_roundtrip :-
    Source = 'Qcur = build_lora_mm(model.layers[il].wq, cur); if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); cb(Qcur, "Qcur", il); }',
    verify_byte_identical_roundtrip(Source).

%% K-projection should also produce BYTE-IDENTICAL round-trip
test_k_projection_AST_identical_after_roundtrip :-
    Source = 'Kcur = build_lora_mm(model.layers[il].wk, cur); if (model.layers[il].bk) { Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk); cb(Kcur, "Kcur", il); }',
    verify_byte_identical_roundtrip(Source).

%% V-projection should also produce BYTE-IDENTICAL round-trip
test_v_projection_AST_identical_after_roundtrip :-
    Source = 'Vcur = build_lora_mm(model.layers[il].wv, cur); if (model.layers[il].bv) { Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv); cb(Vcur, "Vcur", il); }',
    verify_byte_identical_roundtrip(Source).

%% Full QKV block (7 statements) should produce BYTE-IDENTICAL round-trip
test_full_qkv_block_AST_identical_after_roundtrip :-
    Source = 'Qcur = build_lora_mm(model.layers[il].wq, cur); if (model.layers[il].bq) { Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq); cb(Qcur, "Qcur", il); } Kcur = build_lora_mm(model.layers[il].wk, cur); if (model.layers[il].bk) { Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk); cb(Kcur, "Kcur", il); } Vcur = build_lora_mm(model.layers[il].wv, cur); if (model.layers[il].bv) { Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv); cb(Vcur, "Vcur", il); }',
    verify_byte_identical_roundtrip(Source).

%% Helper: verify that parsing Source, lifting to BPD, regenerating to AST
%% produces an AST byte-identical to the original parse.
verify_byte_identical_roundtrip(Source) :-
    c_ast:c_parse_stmts_v2(Source, ASTOriginal),
    qkv_lifter:lift_ast_to_bpd(ASTOriginal, BPDFacts),
    qkv_generator:generate_from_bpd(BPDFacts, ASTRegenerated),
    length(ASTOriginal, NOrig),
    length(ASTRegenerated, NRegen),
    NOrig == NRegen,
    forall(
        ( nth0(I, ASTOriginal, Orig),
          nth0(I, ASTRegenerated, Regen) ),
        Orig == Regen
    ).

:- initialization(run_tests, main).
