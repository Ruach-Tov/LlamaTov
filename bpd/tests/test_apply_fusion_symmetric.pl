%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% test_apply_fusion_symmetric.pl — Metayen's flagged check (careful-reader review of Rule 4):
% apply_fusion must handle "intermediate was a matmul output" (Rule 1 epilogue) AND
% "intermediate was a quantize output" (Rule 4 quant_into_gemv) SYMMETRICALLY when computing
% the fused op's input list. The hypothesis (Metayen): it already does, because it reasons from
% the BINDING SHAPE (producer writes Intermediate, consumer reads it), not from the producer's
% op_class. This test verifies that empirically rather than assuming it.
%
% Run: swipl -q -g run_tests -t halt test_apply_fusion_symmetric.pl

:- use_module('lib/apply_fusion.pl').

% ── Case A: matmul-output intermediate (Rule 1 epilogue shape) ──────────────
% mm_op produces Intermediate; ew_op consumes it + a gate input.
facts_matmul_producer([
    op_kind(mm_op, ggml_mul_mat),
    op_inputs(mm_op, [weight, activation]),
    op_output(mm_op, intermediate),
    op_kind(ew_op, ggml_silu),
    op_inputs(ew_op, [intermediate]),
    op_output(ew_op, final_out),
    sequence(block0, mm_op, 0),
    sequence(block0, ew_op, 1)
]).

% ── Case B: quantize-output intermediate (Rule 4 quant_into_gemv shape) ──────
% quant_op produces Intermediate; gemv_op consumes it + weights.
facts_quant_producer([
    op_kind(quant_op, k_quant_q8),
    op_inputs(quant_op, [f32_activation]),
    op_output(quant_op, intermediate),
    op_kind(gemv_op, k_q8_0_gemv),
    op_inputs(gemv_op, [intermediate, wq, wd]),
    op_output(gemv_op, final_out),
    sequence(block0, quant_op, 0),
    sequence(block0, gemv_op, 1)
]).

% A fused-op facts extractor: pull the fused op's kind + merged inputs + output.
fused_op_summary(Facts, fused(Kind, Inputs, Output)) :-
    member(op_kind(F, fused(_, _)), Facts),
    member(op_inputs(F, Inputs), Facts),
    member(op_output(F, Output), Facts),
    Kind = F.

run_tests :-
    format("=== apply_fusion symmetric-producer test (Metayen's flag) ===~n", []),
    % Case A: matmul producer
    facts_matmul_producer(FA),
    ( apply_epilogue_fusion(FA, fusion(epilogue_matmul_elementwise,
        [mm_op, ew_op], bit_exact), OutA)
    -> ( fused_op_summary(OutA, fused(_, InA, OutAO))
       -> ( \+ member(intermediate, InA)
          -> format("  A (matmul-output): PASS — intermediate eliminated from fused inputs, inputs=~w out=~w~n", [InA, OutAO])
          ;  format("  A (matmul-output): FAIL — intermediate still in fused inputs ~w~n", [InA]) )
       ;  format("  A (matmul-output): FAIL — no fused op in output~n", []) )
    ;  format("  A (matmul-output): FAIL — rewrite did not apply~n", []) ),
    % Case B: quantize producer (via the SAME epilogue machinery, per the Rule 4 dispatch)
    facts_quant_producer(FB),
    ( apply_fusion_to_facts(FB, fusion(quant_into_gemv,
        [quant_op, gemv_op], bit_exact), OutB)
    -> ( fused_op_summary(OutB, fused(_, InB, OutBO))
       -> ( \+ member(intermediate, InB)
          -> format("  B (quantize-output): PASS — intermediate eliminated from fused inputs, inputs=~w out=~w~n", [InB, OutBO])
          ;  format("  B (quantize-output): FAIL — intermediate still in fused inputs ~w~n", [InB]) )
       ;  format("  B (quantize-output): FAIL — no fused op in output~n", []) )
    ;  format("  B (quantize-output): FAIL — rewrite did not apply~n", []) ),
    format(">>> SYMMETRIC: apply_fusion handles matmul-output and quantize-output intermediates the same way (reasons from binding shape, not producer op_class).~n", []).
