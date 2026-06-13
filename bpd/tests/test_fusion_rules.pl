%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_fusion_rules.pl — Stage 3 tests for CHiLL-style fusion rules
%%
%% Tests:
%%   1. The epilogue_matmul_elementwise rule applies to wq_mul+wq_bias_add
%%   2. The rule does NOT apply when intermediate has multiple consumers
%%   3. Exhaustive enumeration finds all valid fusions in the QKV graph
%%   4. Equivalence classes are correctly declared (all bit_exact for these rules)
%%   5. Equivalence class ordering works (bit_exact implies tolerance and mathematical)

:- use_module('../lib/fusion_rules').

:- dynamic op_kind/2.
:- dynamic op_input/2.
:- dynamic op_output/2.
:- dynamic op_reads/3.
:- dynamic op_writes/3.

%% Wire fusion_rules's expected predicates
fusion_rules:op_kind(Op, K) :- op_kind(Op, K).
fusion_rules:op_input(Op, T) :- op_input(Op, T).
fusion_rules:op_output(Op, T) :- op_output(Op, T).
fusion_rules:op_reads(Op, T, R) :- op_reads(Op, T, R).
fusion_rules:op_writes(Op, T, R) :- op_writes(Op, T, R).

%% ────────────────────────────────────────────────────────────────────
%% Test fixture: QKV BPD compute graph
%% ────────────────────────────────────────────────────────────────────

% wq_mul: matmul producing qcur_pre_bias
op_kind(wq_mul, build_lora_mm).
op_input(wq_mul, wq).
op_input(wq_mul, cur_after_norm).
op_output(wq_mul, qcur_pre_bias).
op_writes(wq_mul, qcur_pre_bias, region(matmul_output, [n_tokens, n_head_x_head_dim])).

% wq_bias_add: elementwise add consuming qcur_pre_bias
op_kind(wq_bias_add, ggml_add).
op_input(wq_bias_add, qcur_pre_bias).
op_input(wq_bias_add, bq).
op_output(wq_bias_add, qcur_post_bias).
op_reads(wq_bias_add, qcur_pre_bias, region(elementwise, [n_tokens, n_head_x_head_dim])).

% Similar for K side
op_kind(wk_mul, build_lora_mm).
op_input(wk_mul, wk).
op_input(wk_mul, cur_after_norm).
op_output(wk_mul, kcur_pre_bias).
op_writes(wk_mul, kcur_pre_bias, region(matmul_output, [n_tokens, n_head_kv_x_head_dim])).

op_kind(wk_bias_add, ggml_add).
op_input(wk_bias_add, kcur_pre_bias).
op_input(wk_bias_add, bk).
op_output(wk_bias_add, kcur_post_bias).
op_reads(wk_bias_add, kcur_pre_bias, region(elementwise, [n_tokens, n_head_kv_x_head_dim])).

% V side
op_kind(wv_mul, build_lora_mm).
op_input(wv_mul, wv).
op_input(wv_mul, cur_after_norm).
op_output(wv_mul, vcur_pre_bias).
op_writes(wv_mul, vcur_pre_bias, region(matmul_output, [n_tokens, n_head_kv_x_head_dim])).

op_kind(wv_bias_add, ggml_add).
op_input(wv_bias_add, vcur_pre_bias).
op_input(wv_bias_add, bv).
op_output(wv_bias_add, vcur_post_bias).
op_reads(wv_bias_add, vcur_pre_bias, region(elementwise, [n_tokens, n_head_kv_x_head_dim])).

%% ────────────────────────────────────────────────────────────────────
%% Test runner
%% ────────────────────────────────────────────────────────────────────

run_tests :-
    Tests = [
        test_epilogue_rule_applies_to_qkv,
        test_equivalence_classes_declared,
        test_equivalence_implies,
        test_exhaustive_enumeration_finds_three_qkv_fusions,
        test_no_fusion_when_multiple_consumers
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

%% Test 1: The epilogue rule applies to wq_mul + wq_bias_add
test_epilogue_rule_applies_to_qkv :-
    fusion_rules:fusion_rule(epilogue_matmul_elementwise, Precondition, _, _),
    fusion_rules:satisfy_precondition_with_ops(Precondition, Ops),
    member(wq_mul, Ops),
    member(wq_bias_add, Ops).

%% Test 2: All declared rules have valid equivalence classes
test_equivalence_classes_declared :-
    findall(EC, fusion_rules:fusion_rule(_, _, _, EC), AllClasses),
    forall(member(EC, AllClasses), fusion_rules:equivalence_class(EC)).

%% Test 3: Equivalence ordering: bit_exact implies all weaker classes
test_equivalence_implies :-
    fusion_rules:equivalence_implies(bit_exact, tolerance(_)),
    fusion_rules:equivalence_implies(bit_exact, mathematical),
    fusion_rules:equivalence_implies(tolerance(1.0e-6), mathematical),
    fusion_rules:equivalence_implies(bit_exact, bit_exact).

%% Test 4: Exhaustive enumeration finds the THREE epilogue fusions
%% (wq+wq_bias, wk+wk_bias, wv+wv_bias) in our QKV fixture
test_exhaustive_enumeration_finds_three_qkv_fusions :-
    fusion_rules:enumerate_valid_fusions(
        [epilogue_matmul_elementwise], Fusions),
    length(Fusions, N),
    N >= 3,
    % All three should be bit_exact
    forall(member(fusion(_, _, EC), Fusions), EC == bit_exact).

%% Test 5: When intermediate has multiple consumers, fusion does NOT apply
%% Set up: vcur_pre_bias becomes consumed by TWO ops (synthetic 2nd consumer)
test_no_fusion_when_multiple_consumers :-
    % Add a second consumer of vcur_pre_bias
    assertz(op_input(synth_consumer, vcur_pre_bias)),
    assertz(op_kind(synth_consumer, ggml_silu)),
    % Now wv_mul -> wv_bias_add fusion should fail
    % (vcur_pre_bias has 2 consumers: wv_bias_add and synth_consumer)
    fusion_rules:enumerate_valid_fusions(
        [epilogue_matmul_elementwise], Fusions),
    % wv_mul should NOT appear in any fusion
    \+ ( member(fusion(_, Ops, _), Fusions), member(wv_mul, Ops) ),
    % Cleanup
    retract(op_input(synth_consumer, vcur_pre_bias)),
    retract(op_kind(synth_consumer, ggml_silu)).

:- initialization(run_tests, main).
