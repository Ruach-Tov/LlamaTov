%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% fusion_rules.pl — Stage 3: CHiLL-style composable fusion rules.
%%
%% Each rule has:
%%   - A PRECONDITION (what BPD shape must hold for the rule to apply)
%%   - A POSTCONDITION (what BPD shape results from applying the rule)
%%   - An EQUIVALENCE CLASS (bit_exact | tolerance(Eps) | mathematical)
%%   - A FUSION KIND (epilogue | prologue | residual | normalization | reduction | ...)
%%
%% This is the substrate-honest version of "always valid" fusion:
%% the equivalence class is DECLARED, so consumers know what kind of
%% correctness they're getting.
%%
%% Per SYMBOLIC_FUSION_CORRECTNESS.md Section 9.1 Stage 3 + Heath's
%% direction: bring Hall's dependence-analysis discipline into the
%% fusion validator so we can say fusions are PROVABLY valid.

:- module(fusion_rules, [
    fusion_rule/4,
    apply_rule/3,
    enumerate_valid_fusions/2,
    equivalence_class/1
]).

%% ────────────────────────────────────────────────────────────────────
%% Equivalence classes — the substrate-honest correctness contract
%% ────────────────────────────────────────────────────────────────────
%%
%% bit_exact:
%%   The fused output is byte-for-byte identical to the unfused output.
%%   Required when reproducibility across runs matters.
%%   Most floating-point reorderings VIOLATE bit_exact.
%%
%% tolerance(Eps):
%%   The fused output differs from unfused by at most Eps per element.
%%   This is the practical ML correctness bar. Floating-point
%%   associativity differences fall here (typically eps ≈ 1e-6 for f32).
%%
%% mathematical:
%%   The fused output equals the unfused output as REAL NUMBERS
%%   (infinite-precision). Floating-point differences are permitted.
%%   This is the weakest bar; most fusions trivially satisfy it.

equivalence_class(bit_exact).
equivalence_class(tolerance(_)).
equivalence_class(mathematical).

%% Ordering: bit_exact is strongest, mathematical is weakest.
%% A bit_exact fusion is ALSO tolerance and mathematical.
%% A tolerance fusion is ALSO mathematical but NOT bit_exact.

equivalence_implies(bit_exact, tolerance(_)).
equivalence_implies(bit_exact, mathematical).
equivalence_implies(tolerance(_), mathematical).
equivalence_implies(E, E).

%% ────────────────────────────────────────────────────────────────────
%% Fusion rules
%% ────────────────────────────────────────────────────────────────────
%%
%% fusion_rule(+Name, +Precondition, +Postcondition, +EquivalenceClass)
%%
%% Each rule is a CHiLL-style composable transformation:
%%   - Precondition: list of facts that must hold (variables shared across)
%%   - Postcondition: list of facts produced (with replacement semantics)
%%   - EquivalenceClass: declared correctness contract
%%
%% The Precondition variables are unbound; matching binds them. The
%% Postcondition references those bindings to describe the result.

%% Rule 1: Epilogue fusion of matmul + elementwise.
%% Bit-exact because the fused kernel performs the same operations in
%% the same order: each output element is matmul-then-add, just done
%% without writing the intermediate to memory.
fusion_rule(
    epilogue_matmul_elementwise,
    [   % Precondition
        op_kind(MmOp, MmKind),
        op_class(MmKind, matmul),
        op_output(MmOp, Intermediate),
        op_kind(EwOp, EwKind),
        op_class(EwKind, elementwise),
        op_input(EwOp, Intermediate),
        op_writes(MmOp, Intermediate, region(matmul_output, Shape)),
        op_reads(EwOp, Intermediate, region(elementwise, Shape)),
        no_other_consumers(Intermediate, [EwOp])
    ],
    [   % Postcondition
        fused_op(FusedOp, [MmOp, EwOp]),
        op_kind(FusedOp, fused(MmKind, EwKind)),
        op_class(FusedOp, matmul),  % the fused kernel is still matmul-bound
        % Inputs of the fused op = inputs of Op1 + non-Intermediate inputs of Op2
        eliminate_tensor(Intermediate)
    ],
    bit_exact
).

%% Rule 2: Elementwise + elementwise fusion.
%% Bit-exact because each output element is one elementwise then another,
%% in the same order, without writing the intermediate.
fusion_rule(
    elementwise_chain,
    [   % Precondition
        op_kind(Op1, K1),
        op_class(K1, elementwise),
        op_output(Op1, Intermediate),
        op_kind(Op2, K2),
        op_class(K2, elementwise),
        op_input(Op2, Intermediate),
        op_writes(Op1, Intermediate, region(elementwise, Shape)),
        op_reads(Op2, Intermediate, region(elementwise, Shape)),
        no_other_consumers(Intermediate, [Op2])
    ],
    [   % Postcondition
        fused_op(FusedOp, [Op1, Op2]),
        op_kind(FusedOp, fused(K1, K2)),
        op_class(FusedOp, elementwise),
        eliminate_tensor(Intermediate)
    ],
    bit_exact
).

%% Rule 3: Layout transparency — reshape before another op.
%% Reshape is just a view of the same memory; eliminating it requires
%% the consumer to address the source's layout.  This is NOT bit-exact
%% in the sense of "produces same output" — it produces EXACTLY the
%% same output, just by addressing memory differently.
fusion_rule(
    layout_transparent,
    [   % Precondition
        op_kind(ReshapeOp, ggml_reshape_3d),
        op_input(ReshapeOp, Source),
        op_output(ReshapeOp, Reshaped),
        op_kind(ConsumerOp, _),
        op_input(ConsumerOp, Reshaped),
        no_other_consumers(Reshaped, [ConsumerOp])
    ],
    [   % Postcondition
        % The consumer is rewired to read Source directly
        rewire_input(ConsumerOp, Reshaped, Source),
        eliminate_op(ReshapeOp),
        eliminate_tensor(Reshaped)
    ],
    bit_exact
).

%% Rule 4: Prologue quant + matmul fusion (q8_0).
%% The SYMMETRIC counterpart to Rule 1 (epilogue): Rule 1 fuses the downstream
%% elementwise into the matmul STORE; this fuses the upstream quantize into the
%% matmul LOAD. A quantize op (f32 activation -> int8 quants + fp16 scales) produces
%% an intermediate that the q8_0 matmul consumes from global memory; fusing keeps the
%% quantized activation in SHARED memory, eliminating the global round-trip (measured
%% 178x/token forced write+read on the qwen-0.5b decode).
%%
%% EQUIVALENCE CLASS bit_exact IS EARNED BY CONSTRUCTION, NOT CLAIMED. The fused
%% kernel is GENERATED by q8_0_from_facts.pl's prologue(quant) emission mode, which
%% emits the float accumulation loop via the SAME shared helper (emit_dp4a_accum_body)
%% used by the unfused k_q8_0_gemv. Because the float reduction source is byte-identical
%% by construction, nvcc makes identical FMA-fusion decisions (same dependency DAG =
%% same fusion = same bits, per Mavchin's DAG theorem). The bit_exact class therefore
%% survives BECAUSE OF HOW the kernel is generated (shared emitter), not because of
%% careful arithmetic-order reasoning that could silently break if a future variant
%% restructures the loop. MEASURED: XOR=0, 0/896 mismatches, max_ulp 0 (M=896 K=4864).
%% A future fusion variant that does NOT route through emit_dp4a_accum_body MUST
%% re-measure its class — the bit_exact annotation is grounded in the shared-emitter
%% construction and is void if that grounding is broken. (Per Metayen's careful-reader
%% requirement: the annotation points at its structural ground.)
fusion_rule(
    quant_into_gemv,
    [   % Precondition
        op_kind(QuantOp, QKind),
        op_class(QKind, quantize),          % f32 activation -> int8 quants + fp16 scales
        op_output(QuantOp, Intermediate),
        op_kind(MmOp, MmKind),
        op_class(MmKind, matmul),           % the q8_0 GEMV consuming the quantized activation
        op_input(MmOp, Intermediate),
        op_writes(QuantOp, Intermediate, region(quantized_activation, Shape)),
        op_reads(MmOp, Intermediate, region(quantized_activation, Shape)),
        % LOAD-BEARING: the quantized activation must feed ONLY the gemv. If a second op
        % also reads Intermediate, fusing the quant INTO the gemv (keeping the int8 in
        % shared memory, never materialized in global) would break that other reader.
        no_other_consumers(Intermediate, [MmOp])
    ],
    [   % Postcondition
        fused_op(FusedOp, [QuantOp, MmOp]),
        op_kind(FusedOp, fused(QKind, MmKind)),
        op_class(FusedOp, matmul),          % the fused kernel is still matmul-bound
        % the quantized activation never reaches global memory — it lives in shared mem
        eliminate_tensor(Intermediate)
    ],
    bit_exact   % earned by construction (shared emitter), measured XOR=0 — see header above
).

%% ────────────────────────────────────────────────────────────────────
%% Op-class helper (mirrors symbolic_fusion's compatibility table)
%% ────────────────────────────────────────────────────────────────────

op_class(build_lora_mm, matmul).
op_class(ggml_mul_mat, matmul).
op_class(ggml_add, elementwise).
op_class(ggml_mul, elementwise).
op_class(ggml_silu, elementwise).
op_class(ggml_gelu, elementwise).
op_class(build_norm(_), normalization).
op_class(ggml_reshape_3d, layout).
op_class(ggml_rope_ext, elementwise).
% q8_0 path: the activation quantize (f32 -> int8 quants + fp16 scales) and the
% q8_0 dot/GEMV that consumes it. Used by Rule 4 (quant_into_gemv prologue fusion).
op_class(k_quant_q8, quantize).
op_class(q8_0_dot, matmul).
op_class(k_q8_0_gemv, matmul).

%% ────────────────────────────────────────────────────────────────────
%% Escape predicate
%% ────────────────────────────────────────────────────────────────────

no_other_consumers(Tensor, AllowedOps) :-
    findall(Op, op_input(Op, Tensor), Consumers),
    sort(Consumers, ConsumersSorted),
    sort(AllowedOps, AllowedSorted),
    subset_of(ConsumersSorted, AllowedSorted).

subset_of([], _).
subset_of([H | T], L) :-
    member(H, L),
    subset_of(T, L).

%% ────────────────────────────────────────────────────────────────────
%% Rule application
%% ────────────────────────────────────────────────────────────────────
%%
%% apply_rule(+RuleName, -Bindings, -EquivalenceClass) — finds the first
%% binding that matches the named rule's precondition. Returns the
%% bindings for the postcondition and the rule's equivalence class.

apply_rule(RuleName, Bindings, EqClass) :-
    fusion_rule(RuleName, Precondition, _Postcondition, EqClass),
    satisfy_precondition(Precondition, Bindings).

satisfy_precondition([], []).
satisfy_precondition([Fact | Rest], Bindings) :-
    call(Fact),
    satisfy_precondition(Rest, Bindings).

%% ────────────────────────────────────────────────────────────────────
%% Exhaustive enumeration of valid fusions
%% ────────────────────────────────────────────────────────────────────
%%
%% enumerate_valid_fusions(+RuleNames, -Fusions) — finds all applicable
%% fusions across all rules. Each Fusion is fusion(RuleName, OpsFused,
%% EquivalenceClass).
%%
%% Per Heath's request for exhaustive search: this enumerates all
%% syntactically-valid fusions. The validity is structural (Hall-style
%% dependence-analysis correctness); the equivalence class declares
%% the numerical contract.
%%
%% For graph with N ops and R fusion rules, this is O(R * N^k) where
%% k is the largest rule's precondition arity (typically 2 for
%% binary fusion). Tractable for transformer graphs (N < 200 typically).

enumerate_valid_fusions(RuleNames, Fusions) :-
    findall(
        fusion(RuleName, OpsFused, EqClass),
        ( member(RuleName, RuleNames),
          fusion_rule(RuleName, Precondition, _, EqClass),
          satisfy_precondition_with_ops(Precondition, OpsFused)
        ),
        Fusions
    ).

%% Helper: SATISFY the precondition (binding variables), THEN extract
%% the list of ops involved (which are now ground). Order matters:
%% satisfaction must happen BEFORE extraction so the variables are bound.
satisfy_precondition_with_ops(Precondition, Ops) :-
    satisfy_precondition(Precondition, _),
    findall(Op, member(op_kind(Op, _), Precondition), OpsRaw),
    sort(OpsRaw, Ops).   % sort dedupes and orders
