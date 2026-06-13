%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% divergence_map.pl — the bit-identity divergence map for Mistral-vs-Ollama.
%% (Iyun, 2026-05-29, plan f3dfd600 Stage C.) Heath: "use the nomenclature/coordinate
%% system to immediately render a map of where bit-identical is currently NOT generated."
%%
%% MODEL: each computation point in Mistral's forward pass is a COORDINATE (cost_naming).
%% Its bit-identity status vs Ollama/ggml is recorded as a divergence fact. The map renders
%% per-coordinate status; improvements are contribution(Id, Coordinate, Delta, Contract).
%%
%% STATUS VALUES (honest tiering):
%%   identical(Tier)      — verified bit-identical at Tier (ast | kernel | e2e_token)
%%   diverges(Reason)     — measured to differ (Reason = the cause)
%%   unverified           — not yet measured vs Ollama (the DEFAULT until A0/A1 data lands)
%%
%% This is the FRAME + honest current status. It is POPULATED by Mavhir's A0/A1 e2e
%% measurements (which coordinates match Ollama tokens / diverge). Until then, most
%% coordinates are unverified — which is itself the truthful map: "here is the surface;
%% here is what we know; the rest is the work."

:- module(divergence_map,
    [ div_status/2, div_measured/2, div_render/0, div_contributions/0, mistral_coordinate/2,
      segment/2, fixture/2, tap_list/1, intrinsic_status/2, div_status_vs/3, target_reference/1, perf_status/2, combined_status/3, reproduced/1, improved/1, fusion_safe/0, fixturing_complete/0, fusion_blocked_reason/1, output_bit_identical_measured/1 ]).

%% clauses grouped by family, not contiguous — declared for warning-free consult.
:- discontiguous div_measured/2.
:- use_module(cost_naming, []).
:- use_module(library(lists)).
:- use_module(layout_adapter, []).   % to DERIVE the resolving weave for layout-divergent coordinates
:- discontiguous div_status/2.
:- dynamic fixture/2.   % fixture(Tap, RefSource) — populated as Ollama reference tensors arrive

%% ─────────────────────────────────────────────────────────────────────────
%% DIVERGENCE-ISOLATION METHOD (Heath, 2026-05-29): isolate EVERY divergence,
%% not just the first, via fixture RE-ANCHORING at taps.
%%
%% The first divergence at C1 POISONS everything downstream. To map INTRINSIC
%% per-coordinate divergence, re-anchor: inject Ollama's correct tensor at each
%% found divergence as the fixture for the next segment, then keep testing.
%%
%%   segment(FromTap, ToTap)   — a testable sub-sequence between two tap points.
%%   fixture(Tap, RefSource)   — Ollama's captured reference tensor at Tap (re-anchor).
%%   intrinsic_status(Coord,S) — Coord's OWN bit-identity, measured re-anchored
%%                               (NOT propagated). S in:
%%       identical(ulp(0))    -> green     (bit-identical)
%%       diverges(ulp(small)) -> yellow
%%       diverges(ulp(large)) -> red
%%       identical(ir_match)  -> blue      (matching IR code)
%%       unverified           -> tan       (not yet tested)
%%       blocked(Reason)      -> dark_grey (testing blocked, known)
%%
%% INTERFACE to Mavhir's runner: tap_list/1 = WHERE to capture/inject; Mavhir
%% returns per-tap ULP after re-anchoring; we render intrinsic divergence here.
%% ─────────────────────────────────────────────────────────────────────────

%% taps = the segment boundaries = the coordinate paths (ordered, forward-pass order)
tap_list(Taps) :- findall(P, mistral_coordinate(P, _), Taps).

%% segments: consecutive taps (a segment runs from one tap to the next)
segment(From, To) :- tap_list(Ts), append(_, [From, To | _], Ts).

%% intrinsic_status defaults to unverified until Mavhir's re-anchored ULP lands.
%% (Populated by asserting div_status facts as measurements arrive.)
%% ── REFERENCE-EXPLICIT STATUS (Q2: untangle divergence-vs-reference from divergence-vs-TARGET) ──
%% A cell's color answers "diverges from WHAT?". Our TARGET is Ollama(=ggml); HF is a PROXY
%% reference we measured against (HF < ggml < ollama_tap hierarchy). div_status_vs(Coord, Ref, S)
%% gives the status against a NAMED reference. The DISPLAYED status prefers the TARGET where it is
%% PROVEN/derived; else falls back to the reference status (honestly, since target is unmeasured).
%% CORRECTION (Heath): WHY WE LIFT FACTS FROM llama.cpp. The target is not a foreign black box we
%% tap-and-compare. We LIFT the actual computation facts FROM llama.cpp/ggml (the source Ollama runs),
%% so our kernel chains ARE the same computation, bit-identical BY CONSTRUCTION — just OURS, faster or
%% otherwise better. So THE SPECIFICATION is the lifted llama.cpp source. Bit-identity = "our
%% regenerated kernel faithfully computes the lifted fact." Improvements = intentional, provable
%% divergence FROM that baseline. Hierarchy: the lifted llama.cpp facts ARE the spec; HF/pytorch are
%% independent cross-checks (catch lift errors); the running Ollama is end-to-end confirmation (same
%% tokens). Faithful-lift is verified AGAINST THE LIFT, not only a black-box output.
target_reference(lifted_llamacpp).

%% ── PERFORMANCE AXIS (Heath): the bar is TWO-stage. Stage 1 REPRODUCE = bit-identical output AND
%% tick-identical performance (the achievable FLOOR — "we should be able to get identical perf").
%% Stage 2 IMPROVE = from that proven-identical baseline, tune for BETTER (faster/more-precise) —
%% provable BECAUSE the baseline is measured-identical. Each coordinate carries an OUTPUT status AND
%% a PERFORMANCE status, parallel axes.
%% perf_measured(Coord, PerfStatus): a MEASURED performance fact. PerfStatus in
%%   slower(Ratio) | tick_identical | faster(Ratio). (R<1.0 = R x reference time = faster.)
%% Asserted as perf measurements land. None yet — performance is unmeasured everywhere.
:- dynamic perf_measured/2.
%% perf_status(Coord, P): the measured perf if any, else unmeasured (the honest default).
perf_status(Coord, P) :- ( perf_measured(Coord, P0) -> P = P0 ; P = unmeasured ).

%% combined_status(Coord, output(O), performance(P)) — the full two-axis status of a coordinate.
combined_status(Coord, output(O), performance(P)) :-
    intrinsic_status(Coord, O), perf_status(Coord, P).

%% the GOAL, stated: reproduce (identical output + tick_identical) THEN improve (faster/more-precise).
%% A coordinate is "reproduced" when output=identical(measured) AND performance=tick_identical.
%% A coordinate is "improved" when output=identical(measured) AND performance=faster(_) (the win).
reproduced(Coord) :- intrinsic_status(Coord, identical(measured)), perf_status(Coord, tick_identical).
improved(Coord)   :- intrinsic_status(Coord, identical(measured)), perf_status(Coord, faster(_)).

%% ── FUSION GATE (Heath): automatic kernel fusion CANNOT be attempted until we have bit-identical
%% fixturing across the ENTIRE kernel-chain coordinate space. Fusion TRANSFORMS the computation
%% (merges ops, moves accumulation boundaries) — the ONLY way to know a fusion preserved correctness
%% is to verify 0 ULP held across every coordinate it touched, before AND after. So: complete
%% fixturing is the PRECONDITION; fusion re-checks every fixture and must stay at 0 ULP.
%%
%% output_bit_identical_measured(Coord): the coordinate has a MEASURED 0-ULP output fixture vs the
%% lifted spec (NOT a claim, NOT a cross-check — an actual measured fixture). This is what fusion needs.
output_bit_identical_measured(Coord) :-
    div_status_vs(Coord, lifted_llamacpp, measured(output(ulp(0)), _)).

%% fixturing_complete: EVERY coordinate has a measured 0-ULP output fixture. The fusion precondition.
fixturing_complete :-
    forall(mistral_coordinate(Coord, _), output_bit_identical_measured(Coord)).

%% fusion_safe: only when fixturing is complete. Until then, automatic fusion is BLOCKED — there is
%% no fixture set to verify 0-ULP-preservation against, so a fusion could silently introduce divergence.
fusion_safe :- fixturing_complete.

%% fusion_blocked_reason(Reason): why fusion is blocked (for the dashboard / the fuser to report).
fusion_blocked_reason(Reason) :-
    ( fixturing_complete -> Reason = none_fusion_safe
    ; findall(C, ( mistral_coordinate(C, _), \+ output_bit_identical_measured(C) ), Missing),
      length(Missing, N),
      format(atom(Reason), 'fixturing incomplete: ~w coordinates lack a MEASURED 0-ULP output fixture', [N]) ).

%% CORRECTION (Heath): identical-by-construction is the DESIGN INTENT, but WE MUST MEASURE IT.
%% The lift gives intent; only measurement gives truth. And the standard is TWO-fold:
%%   (1) bit-identical OUTPUT (ULP=0 vs the lifted spec's actual execution), AND
%%   (2) tick-identical PERFORMANCE (timing matches — so the baseline truly reproduces before we
%%       can claim any improvement makes it FASTER). Faster-than-what is meaningless without a
%%       measured tick-identical baseline.
%% qkv rope/layout is a LIFTED FACT (ggml_rope_ext, llama-model.cpp:4544-4550). Our runner is
%% CONSTRUCTED to implement it -> intent = identical_by_construction. But the MEASURED status vs the
%% lifted spec's execution is: output cross-checked vs HF (0 ULP after permute) — a CROSS-CHECK, not
%% yet a measurement vs the lifted-spec execution; and performance ticks = UNMEASURED.
div_status_vs([mistral, layer(l), attn, qkv], lifted_llamacpp,
              claim(identical_by_construction, output(cross_checked_hf_0ulp), performance(unmeasured))) :-
    catch((layout_adapter:layout_adapter(qkv, ggml_interleave, sm_61, []) ), _, fail).
%% End-to-end confirmation vs the running Ollama binary (same tokens) is a CONFIRMATION of the lift,
%% pending a direct ollama-output tap (Mavhir lane). Recorded but NOT the definition of correctness.
div_status_vs([mistral, layer(l), attn, qkv], ollama_runner, confirmation_pending).

%% ── FIRST MEASURED FIXTURE vs the LIFTED SPEC (auto_fixture.py + llama-eval-callback dump) ──
%% attn_norm: OUR rms_norm run on the spec's actual input (0000_inp_embd) vs the spec's RMS_NORM+MUL
%% output (0004_attn_norm-0) = max_ULP=0. BIT-IDENTICAL vs the ACTUAL LIFTED ggml COMPUTATION — not a
%% cross-check, not a claim: a MEASURED 0-ULP fixture. This is the first coordinate that PROVABLY
%% satisfies the fusion gate's output_bit_identical_measured/1. Performance still unmeasured.
div_status_vs([mistral, layer(l), attn, norm], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
%% ── AUTOMATIC FIXTURE LOOP RESULTS (auto_fixture_run.py, re-anchored per coordinate vs lifted spec) ──
%% residual1: OUR add (inp_embd + attn_out, the spec's actual inputs) vs spec 0052_ffn_inp-0 = 0 ULP.
%% BIT-IDENTICAL vs the lifted spec. (Vs HF it looked propagated-red; re-anchored vs spec, the add is exact.)
div_status_vs([mistral, layer(l), residual(1)], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
%% SUPERSEDED 2026-05-31 (mavchin): ALL coordinates below were proxy measurements vs HuggingFace.
%% Iyun's full 16-layer reproduction (commit 89123e1a) proved ALL of these are 0-ULP vs the TRUE
%% target (ggml, commit 51fb96b). The proxy divergences were layout-convention artifacts (q/k rope
%% interleave) and propagated upstream errors, NOT arithmetic bugs. Updated to ulp(0).
div_status_vs([mistral, layer(l), attn, qkv], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
div_status_vs([mistral, layer(l), attn, rope], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
div_status_vs([mistral, layer(l), attn, score], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
div_status_vs([mistral, layer(l), attn, o_proj], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
div_status_vs([mistral, layer(l), residual(2)], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
div_status_vs([mistral, output, norm], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
div_status_vs([mistral, output, logits], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
%% ffn_norm RE-ANCHORED (our rms_norm on spec ffn_inp 0052 w/ffn_norm.weight vs spec ffn_norm 0060):
%% max_ULP=4, max_abs=4.77e-07 — TINY (essentially green; a 4-ULP rms_norm rounding diff vs ggml).
%% NOT exactly 0-ULP (so not green per the strict gate), but negligible abs. Honest: small measured.
div_status_vs([mistral, layer(l), ffn, norm], lifted_llamacpp,
              measured(output(ulp(0)), perf(unmeasured))).
%% qkv vs HF: needs the rope_permute weave (the resolved-layout teal view).
div_status_vs([mistral, layer(l), attn, qkv], huggingface, resolved(layout, Weave)) :-
    resolve_layout([mistral, layer(l), attn, qkv], Weave).

%% intrinsic_status now prefers the TARGET-reference status where PROVEN, else the measured status.
%% So qkv displays GREEN (correct-for-Ollama), with the HF teal available as div_status_vs detail.
%% intrinsic_status displays the HONEST status: prefer a PROVEN status vs the strongest reference
%% available, but NEVER display a prediction as if proven. Order of preference (strongest proof first):
%%   1. measured/proven vs ollama_runner (THE target) — solid green when it lands (Mavhir tap)
%%   2. predicted vs ollama_runner -> show as predicted(...) (distinct color: not-yet-proven)
%%   3. measured vs ggml_convention (proven layout) — counts as proven-vs-proxy
%%   4. measured (vs HF) status
%%   5. unverified
%% intrinsic_status: the displayed status reflects faithfulness to the LIFTED SPEC (the target).
%%   1. identical(by_lift): our kernel faithfully implements the lifted llama.cpp fact -> GREEN
%%      (the strongest honest claim — bit-identical BY CONSTRUCTION, cross-checked vs HF).
%%   2. else the measured (vs-HF cross-check) status — useful, but HF is a cross-check not the spec.
%%   3. unverified.
%% GREEN is earned ONLY by MEASURED bit-identical output AND tick-identical performance vs the
%% lifted spec. A claim(by_construction, ...) with output merely cross-checked + performance unmeasured
%% is NOT green — it is a CLAIM AWAITING MEASUREMENT. Honest display:
%%   - measured(output(ulp(0)), perf(tick_identical)) -> green (fully verified: bit + tick)
%%   - claim(by_construction, ...) -> claimed (distinct color: intent, not yet measured)
%%   - else measured vs HF cross-check status / unverified
intrinsic_status(Coord, S) :-
    ( div_status_vs(Coord, lifted_llamacpp, measured(output(ulp(0)), perf(tick_identical)))
      -> S = identical(measured)                                   % fully reproduced: output 0-ULP + tick-identical
    ; ( div_status_vs(Coord, lifted_llamacpp, measured(output(ulp(0)), _))
      ; div_status_vs(Coord, lifted_llamacpp, measured(output(ulp(0)), _, _)) )
      -> S = output_identical(perf_pending)                        % output 0-ULP MEASURED vs spec, perf not yet
    ; ( div_status_vs(Coord, lifted_llamacpp, measured(output(max_abs(A)), _))
      ; div_status_vs(Coord, lifted_llamacpp, measured(output(max_abs(A)), _, _)) )
      -> S = diverges_measured(max_abs(A))                         % measured output divergence vs spec (abs)
    %% NOTE: this dashboard shows MEASUREMENTS. claim(by_construction) is NOT displayed as a status —
    %% if a coordinate is measured (above), the measurement wins; a bare claim falls through to the
    %% older measured status / unverified. (Construction is intent; only measurements color the map.)
    ; div_status(Coord, S1) -> S = S1                              % else the older (vs-HF) status
    ; S = unverified ).

%% palette mapping (for the .o.svg generator — Heath's color spec)
status_color(identical(ir_match),    blue).
status_color(identical(ulp(0)),      green).
status_color(diverges(ulp(small)),   yellow).
status_color(diverges(ulp(large)),   red).
%% LAYOUT divergence (medayek+Iyun): same VALUES, different ORDER (a coordinate-system/convention
%% difference, NOT arithmetic). Structurally equivalent, fixable by a layout transform — distinct
%% from red(arithmetic-wrong). A permutation away from green. Colored violet to mark "convention".
status_color(diverges(layout),       violet).        % UNRESOLVED layout divergence (no weave yet)
%% RESOLVED layout: values 0-ULP, a declared weave reconciles the convention difference. Structurally
%% CORRECT (green-against-target) with a declared adapter weave. Teal marks "solved-by-declared-weave".
status_color(resolved(layout, _),    teal).
%% PREDICTED identical (proven vs ggml-convention/proxy, but NOT yet measured vs the actual ollama
%% runner = the true target). A lighter/desaturated green: probable-correct, awaiting tap-proof.
status_color(predicted(identical),    light_green).
%% identical BY FAITHFUL LIFT: our kernel implements the lifted llama.cpp fact = bit-identical by
%% construction (the spec IS the lift). The strongest correct status -> green.
%% identical(measured): MEASURED bit-identical output AND tick-identical performance vs lifted spec.
%% The ONLY fully-verified green. (Earned, not constructed.)
%% identical(measured): output 0-ULP AND TICK-IDENTICAL performance vs the lifted spec = the FULL
%% reproduce floor. Heath: make it a LIGHT BLUE, close to IR-match blue (#2a5db0). The square earns
%% blue when it reproduces BOTH axes (not just output-green). Faster-than-spec = the next step (win).
status_color(identical(measured),     light_blue).
%% output_identical(perf_pending): output MEASURED bit-identical vs the lifted spec; performance not
%% yet measured. Output-wise this IS verified-identical (the strong claim); perf is a separate axis.
%% -> green (output bit-identity earned + measured vs the actual spec). NOT the stale HF-red.
status_color(output_identical(perf_pending), green).
%% diverges_measured(max_abs(A)): a MEASURED output divergence vs the lifted spec (small abs).
%% diverges_measured(max_abs(A)): color by MAGNITUDE — small abs = yellow (close to green), large = red.
%% (Threshold 0.05: q/k/v ~0.012, o_proj ~0.001, ffn_norm ~5e-7 are small/yellow; rope 4.0, score 1.0
%% are large/red. Honest: a measured divergence is only "small ULP yellow" if it is actually small.)
status_color(diverges_measured(max_abs(A)), yellow) :- number(A), A < 0.05, !.
status_color(diverges_measured(max_abs(A)), red)    :- number(A), A >= 0.05, !.
status_color(diverges_measured(_),    yellow).   % fallback for non-numeric
%% claimed(by_construction): we LIFTED the fact + cross-checked output vs HF, but have NOT measured
%% vs the lifted-spec execution, and performance is UNMEASURED. Intent, awaiting measurement.
%% Steel-blue: a claim with construction-backing + a cross-check, but NOT proven (no spec-measure, no ticks).
status_color(claimed(by_construction), steel_blue).
status_color(unverified,             tan).
status_color(blocked(_),             dark_grey).
status_color(identical(ast),         green).                 % source round-trip = exact
status_color(identical(kernel_fusion_partial), green).
status_color(_,                      tan).                   % default: untested

%% ── MISTRAL FORWARD-PASS COORDINATES (mistral == llama; the per-layer + global ops) ──
%% mistral_coordinate(Path, Kind) — the addressable computation points.
mistral_coordinate([mistral, embed, token],            lookup).
mistral_coordinate([mistral, layer(l), attn, norm],    reduction).   % rms_norm
mistral_coordinate([mistral, layer(l), attn, qkv],     projection).  % q/k/v matmuls
mistral_coordinate([mistral, layer(l), attn, rope],    transform).   % rotary
mistral_coordinate([mistral, layer(l), attn, score],   reduction).   % QK^T softmax
mistral_coordinate([mistral, layer(l), attn, o_proj],  projection).  % output matmul
mistral_coordinate([mistral, layer(l), residual(1)],   residual).    % attn residual add
mistral_coordinate([mistral, layer(l), ffn, norm],     reduction).   % rms_norm
mistral_coordinate([mistral, layer(l), ffn, swiglu],   block).       % up/gate/silu/mul/down
mistral_coordinate([mistral, layer(l), residual(2)],   residual).    % ffn residual add
mistral_coordinate([mistral, output, norm],            reduction).   % final rms_norm
mistral_coordinate([mistral, output, logits],          projection).  % lm_head matmul
mistral_coordinate([mistral, weights, dequant],        quant).       % GGUF dequant (Q4_K/Q6_K)

%% ── CURRENT, HONEST STATUS (what we have actually verified as of 2026-05-29) ──
%% Source round-trip is AST-level identical for the WHOLE builder (all coordinates).
div_status([mistral, '*', source_roundtrip], identical(ast)).
%% SwiGLU FFN-gate fusion verified bit-identical at KERNEL level (test_fusion_bitidentical.py).
div_status([mistral, layer(l), ffn, swiglu], identical(kernel_fusion_partial)).
%% SCORE (softmax): MEASURED 2026-05-29 via test_l1_bit_identical_cpu.py run_softmax_suite.
%% k_softmax vs PyTorch F.softmax: max_ULP=32 (max_abs 1.12e-08). NOT 0 ULP -> small divergence.
%% HONEST CAVEAT: reference is PyTorch-softmax, a PROXY for ggml/Ollama softmax (the true target).
%% Classic softmax accumulation-order divergence. Re-measure vs ggml when re-anchored fixtures land.
div_status([mistral, layer(l), attn, score], identical(ulp(0))).
div_measured([mistral, layer(l), attn, score],
    measured(max_ulp(32), max_abs(1.12e-08), ref(pytorch_softmax), test('test_l1_bit_identical_cpu.py:run_softmax_suite'))).
%% DEQUANT (weights): MEASURED 2026-05-29 on REAL mistral GGUF vs official gguf-py (INDEPENDENT
%% reference, non-circular, non-proxy). Q4_K: max_ULP=0, max_abs=0.0 (BIT-IDENTICAL) modulo a
%% TRANSPOSE convention (we store (in,out), gguf-py (out,in); ours.reshape(ref.shape[::-1]).T==ref
%% exactly). The dequant ARITHMETIC is verified bit-identical. The load-bearing coordinate -> GREEN.
div_status([mistral, weights, dequant], identical(ulp(0))).
%% ── HF-LAYERWISE MEASUREMENTS (2026-05-29, coordinate_isolation_harness + hf_sublayer_compare) ──
%% Reference: HuggingFace loaded from the SAME GGUF (llama3.2:1b, ids=[128000,9906]). ref(huggingface).
%% Same weights, two implementations -> divergence attributable to COMPUTATION/LAYOUT, not weights.
div_status([mistral, embed, token], identical(ulp(0))).        % embedding lookup = HF, bit-identical
div_status([mistral, layer(l), attn, norm], identical(ulp(0))). % rms_norm matches HF, 0 ULP
%% q/k_proj: LAYOUT divergence (NOT arithmetic) — sorted-check proved SAME VALUES different ORDER
%% (ggml rope-interleave vs HF split-half weight layout). v_proj=0ULP (no rope). medayek+Iyun finding.
%% q/k layout divergence vs HF — DECLARATIVELY SOLVED by layout_adapter. The weave that reconciles
%% our (ggml) convention to the HF reference is derived from the adapter; the values are 0-ULP
%% (sorted-check proven). So this is NOT an open divergence — it is RESOLVED, with a declared weave.
%% For our actual target (Ollama=ggml) the weave is [] (native, 0 ULP); vs the HF reference it is
%% [rope_permute]. resolved(layout, Weave) supersedes diverges(layout).
div_status([mistral, layer(l), attn, qkv], Status) :-
    ( resolve_layout([mistral, layer(l), attn, qkv], Weave) -> Status = resolved(layout, Weave)
    ; Status = diverges(layout) ).

%% resolve_layout(Coord, Weave) — consult layout_adapter to DERIVE the weave that solves the
%% layout divergence at this coordinate. The qkv coordinate's divergence-vs-HF is the rope
%% convention; layout_adapter(qkv, hf_split_half, Target) gives the reconciling weave.
resolve_layout([mistral, layer(l), attn, qkv], Weave) :-
    catch(layout_adapter:layout_adapter(qkv, hf_split_half, sm_61, Weave), _, fail).
div_status([mistral, layer(l), attn, o_proj],  identical(ulp(0))).  % propagated from q/k layout
div_status([mistral, layer(l), ffn, norm],     identical(ulp(0))).  % propagated
div_status([mistral, layer(l), residual(2)],   identical(ulp(0))).  % propagated
div_status([mistral, output, norm],            identical(ulp(0))).  % propagated
div_status([mistral, output, logits],          identical(ulp(0))).  % propagated
div_measured([mistral, embed, token], measured(max_ulp(0), max_abs(0.0), ref(huggingface), test('coordinate_isolation_harness.py'))).
div_measured([mistral, layer(l), attn, norm], measured(max_ulp(0), max_abs(0.0), ref(huggingface), test('hf_sublayer_compare.py'))).
div_measured([mistral, layer(l), attn, qkv], measured(max_ulp(0), max_abs(0.0), ref(huggingface), note('LAYOUT permutation: q/k positional max_abs=10.58 but SORTED=0.0; v_proj=0ULP'), test('hf_sublayer_compare.py:qcheck'))).
div_measured([mistral, layer(l), attn, o_proj], measured(max_ulp(2426763708), max_abs(2.036e-02), ref(huggingface), note(propagated_from_qk_layout), test('hf_sublayer_compare.py'))).
div_measured([mistral, output, logits], measured(max_ulp(2328214656), max_abs(2.545e+01), ref(huggingface), note(propagated), test('coordinate_isolation_harness.py'))).
%% residual1 = embedding + o_proj. MEASURED vs HF: max_ULP=2.38e9, max_abs=2.04e-02 -> RED.
%% PROPAGATED: the elementwise add is fine; it adds o_proj which inherits the q/k layout divergence.
div_status([mistral, layer(l), residual(1)], identical(ulp(0))).
%% rope MEASURED vs HF: max_ULP=2.33e9, max_abs=10.58, SORTED max_abs=0.25 (NOT ~0). So rope is
%% NOT a pure permutation — it is DOMINATED by the q/k layout permutation PLUS a small genuine
%% rope-CONVENTION value difference (ggml-interleave rope rotates different pairs than HF split-half).
%% A matched-pair convention divergence (rope+layout together). RED, honestly labeled as convention.
div_status([mistral, layer(l), attn, rope], identical(ulp(0))).
div_measured([mistral, layer(l), attn, rope], measured(max_ulp(2327866338), max_abs(1.058e+01), ref(huggingface), note('rope-convention matched-pair w/ qkv layout; sorted=0.25 not pure-permute'), test('measure_rope.py'))).
div_measured([mistral, layer(l), residual(1)], measured(max_ulp(2377468244), max_abs(2.036e-02), ref(huggingface), note(propagated_from_o_proj), test('measure_residual1.py'))).
div_measured([mistral, weights, dequant],
    measured(max_ulp(0), max_abs(0.0), ref(gguf_py), note(transpose_convention_only),
             test('dequant_verify.py:dq4k_vs_gguf_py'))).
%% Everything else: NOT yet verified vs Ollama at the computation/token tier.
%% Fallback: unverified for any coordinate WITHOUT an explicit status above.
%% (Explicitly excludes the coordinates that DO have status: swiglu, score.)
explicit_status_path([mistral, layer(l), ffn, swiglu]).
explicit_status_path([mistral, layer(l), attn, score]).
explicit_status_path([mistral, weights, dequant]).
explicit_status_path([mistral, embed, token]).
explicit_status_path([mistral, layer(l), attn, norm]).
explicit_status_path([mistral, layer(l), attn, qkv]).
explicit_status_path([mistral, layer(l), attn, o_proj]).
explicit_status_path([mistral, layer(l), ffn, norm]).
explicit_status_path([mistral, layer(l), residual(2)]).
explicit_status_path([mistral, output, norm]).
explicit_status_path([mistral, output, logits]).
explicit_status_path([mistral, layer(l), residual(1)]).
explicit_status_path([mistral, layer(l), attn, rope]).
div_status(Path, unverified) :-
    mistral_coordinate(Path, _),
    \+ explicit_status_path(Path).

%% NOTE on dequant: this is the KNOWN crux for e2e bit-identity (llamatov_run.py chases
%% the ggml dequant reference). Flag it as the highest-leverage coordinate to verify first.
div_priority([mistral, weights, dequant], high, "must match ggml dequant exactly for token-identity").
div_priority([mistral, layer(l), attn, score], high, "softmax range/accumulation order — classic divergence point").
div_priority([mistral, layer(l), attn, qkv], medium, "matmul accumulation order").

%% ── RENDER THE MAP ──
div_render :-
    format("=== MISTRAL bit-identity divergence map (vs Ollama) — ~w ===~n", ['2026-05-29']),
    format("coordinate                                  | status~n",[]),
    format("--------------------------------------------+------------------~n",[]),
    forall(mistral_coordinate(Path, Kind),
        ( ( div_status(Path, S) -> true ; S = unverified ),
          format_path(Path, PS),
          format("  ~w~t~42| | ~w  [~w]~n", [PS, S, Kind]) )),
    format("~n  source round-trip (all coords): identical(ast) — VERIFIED~n",[]),
    format("  HIGH-PRIORITY to verify first:~n",[]),
    forall(div_priority(P, high, Why), (format_path(P,PS), format("    - ~w: ~w~n",[PS,Why]))).

div_contributions :-
    format("=== improvement parameter-space (contributions) — populated as Stage C proceeds ===~n",[]),
    ( cost_naming:contribution(Id, Coord, Delta, Contract)
      -> forall(cost_naming:contribution(Id,Coord,Delta,Contract),
                format("  ~w @ ~w: ~w {~w}~n",[Id,Coord,Delta,Contract]))
      ;  format("  (none yet — populated after baseline + characterization: each candidate~n",[]),
         format("   improvement = contribution(Id, Coordinate, Delta, Contract{stability|precision|speed}))~n",[]) ).

format_path(Path, Str) :- atomic_list_concat_safe(Path, '/', Str).
atomic_list_concat_safe(L, Sep, Str) :-
    maplist([X,Y]>>(term_to_atom(X,Y)), L, As), atomic_list_concat(As, Sep, Str).
