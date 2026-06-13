%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ir_param_axes.pl — PARAMETERIZED recipe-families: the coordinate system of VARIATION.
%%
%% Heath's principle (2026-05-31): holding 0 ULP vs MULTIPLE references is a forcing function.
%% Where references differ, we expose a SWEEPABLE PARAMETER (an axis) so one generator can
%% match each reference by setting the axis. The union of references MAPS the axis; the
%% generator then TRANSCENDS the references — we steer to optimal regions (stability/speed)
%% with full knowledge of which reference-quality we depart from.
%%
%% This complements ir_match_recipes.pl (fixed proven recipes): here each op is a FAMILY
%% parameterized by axes, and each reference pins axis-values. Proven on silu (2026-05-31):
%% form=divide -> 0-ULP vs F.silu; form=reciprocal_mul -> 0-ULP vs Stanford Swish.
%%
%% Tracked source (bpd/lib/). Drives: the parameterized kernel generator, the stanford_referee
%% (DIVERGENT cell -> "which axis does the reference pin?"), and the diagram subsystem.

:- module(ir_param_axes, [
     param_axis/3,             % param_axis(Op, AxisName, ValueList) — the sweep dimension
     reference_pins/3,         % reference_pins(Reference, AxisName, Value) — a ref pins an axis-value
     axis_quality/3,           % axis_quality(AxisName, Value, Quality) — why one might prefer a setting
     op_family/2,              % op_family(Op, AxisList) — which axes parameterize an op
     axis_order/3,             % axis_order(Axis, Objective, OrderedValues) — steering gradient (best→worst)
     axis_metric/4,            % axis_metric(Axis, V1, V2, Props) — measured magnitude of difference
     axis_default/3,           % axis_default(Axis, Objective, Value) — head of the order
     resolve_axis/5            % resolve_axis(Op, Axis, Reference, Value, Note) — decision procedure
   ]).

%% These multi-op fact families are intentionally grouped BY THEME (all param_axis together by
%% concept, the reduce-op axes near their reduction_order discussion at the bottom), not by op —
%% so the same predicate appears in several places. Declare discontiguous to keep that organization
%% without warnings (and so a real load error isn't masked by benign "clauses not together" noise).
:- discontiguous param_axis/3, op_family/2, axis_quality/3, axis_order/3, axis_metric/4,
                 reference_pins/3, axis_default/3.

%% ── AXES (the sweepable parameters discovered by matching multiple references) ──
%% form: how a sigmoid/softmax-family op applies the normalizing division.
param_axis(silu,    form, [divide, reciprocal_mul]).
param_axis(softmax, form, [divide, reciprocal_mul]).
%% acc_type: reduction accumulator precision (the single biggest reduction axis).
param_axis(rms_norm,   acc_type, [f32, f64]).
param_axis(layer_norm, acc_type, [f32, f64]).
param_axis(mean,       acc_type, [f32, f64]).
param_axis(softmax_sum,acc_type, [f32, f64]).
%% gelu_approx: the two canonical GELU formulations.
param_axis(gelu, gelu_approx, [erf, tanh]).
%% reduction_grouping: how a matmul groups the K-reduction (size-dependent in references).
param_axis(gemm, reduction_grouping, [sequential, kc(256), kc(384), multi_acc(2), multi_acc(4), tiled]).
%% fma_contract: whether fmul+fadd contract to fma (target -mno-fma pins off).
param_axis(gemm, fma_contract, [on, off]).
%% weight_access: how the K-reduction's weight bytes are FETCHED from DRAM. A PERFORMANCE axis,
%% ORTHOGONAL to reduction_order — it changes which bytes a lane physically loads and when, but
%% NOT which products are summed in what order. Holding reduction_order fixed, every weight_access
%% value yields BIT-IDENTICAL output (the math DAG is unchanged; only the memory schedule differs).
%% This is the thesis separation: reduction_order = the correctness contract (0-ULP), weight_access
%% = the performance contract (coalescing/bandwidth). Each backend lowers weight_access in its own
%% idiom (CUDA warp-cooperative loads, Rust/oxide SIMD, CPU cache-line striding) but the INTENT is
%% one backend-neutral fact, so the optimization translates to all generated backends.
%%   lane_strided    — lane b reads block b,b+32,...; each lane issues its own int4 loads. Simple
%%                     but adjacent lanes read 32B-apart addresses -> partial coalescing (measured).
%%   warp_contiguous — the 32 lanes of a warp cooperatively load a CONTIGUOUS span (one coalesced
%%                     burst) into a shared/register staging buffer, THEN each lane consumes its
%%                     canonical blocks from staging. Sectors/load -> ideal; reduction_order intact.
param_axis(gemm, weight_access, [lane_strided, warp_contiguous]).
%% epilogue_compute: which downstream element-local op is FOLDED into the GEMV's store. The lever
%% is HIDING COMPUTE UNDER THE MEMORY WALL: a memory-bound GEMV leaves the SMs idle during DRAM
%% stalls, so per-output-row compute in the store-epilogue runs in that shadow — measured ~free
%% (SiLU transcendental into gate GEMV: +0.2%% gate / +1.3%% vocab). Each value also ELIMINATES a
%% separate kernel launch + a DRAM round-trip of the intermediate. CRITERION: the folded op must
%% be ELEMENT-LOCAL — read only Y[row], its own output element — because the GEMV produces rows
%% independently (cross-row ops like rope/rmsnorm are EXCLUDED; see q8_0 emitter). BIT-IDENTICAL:
%% the reduction_order (the 0-ULP contract) is untouched; only the store transform changes, and it
%% is the same arithmetic the separate kernel did. Like weight_access, this is the thesis
%% separation — the INTENT is one backend-neutral fact, each backend lowers it in its idiom, so
%% "hide compute under memory" translates to ALL generated backends.
%%   none        — store acc unchanged (the plain GEMV).
%%   add_resid   — Y = acc + Resid[row]   (residual; the addres mode, validated +5.9%% e2e).
%%   bias        — Y = acc + Bias[row]    (q/k/v bias; eliminates a separate k_add).
%%   silu        — Y = acc / (1 + exp(-acc))   (gate activation; hides the transcendental).
%%   bias_silu   — Y = silu(acc + Bias[row])   (composed, if a biased GEMV feeds an activation).
param_axis(gemm, epilogue_compute, [none, add_resid, bias, silu, bias_silu]).
%% output_target: WHERE the GEMV writes its result rows. A PERFORMANCE / GRAPH-TOPOLOGY axis,
%% ORTHOGONAL to the math (the float DAG is identical; only the destination pointer changes ->
%% BIT-IDENTICAL). The lever is ELIMINATING A COPY: under a captured/replayed compute graph the
%% layer output must land in a FIXED buffer the next layer reads (graph needs stable pointers).
%% The naive form writes to fresh scratch then COPIES into the carry buffer (a graph node per
%% layer); fixed_carry has the GEMV write the carry buffer DIRECTLY, deleting the copy node. The
%% graph gets structurally SMALLER as it gets faster. Each backend lowers it in its idiom (CUDA:
%% the out= kernel-arg pointer; Rust/oxide: the destination slice; CPU: the output buffer arg) but
%% the INTENT is one backend-neutral fact: write where the consumer reads, don't copy.
%%   scratch     — GEMV -> fresh buffer, then memcpy into the persistent carry (a copy per layer).
%%   fixed_carry — GEMV writes the persistent carry buffer directly (no copy). REQUIRES no aliasing
%%                 between the output buffer and the GEMV's other inputs (residual, activation).
param_axis(gemm, output_target, [scratch, fixed_carry]).
%% head_grouping: whether a per-head op (RoPE) launches PER-TENSOR (q alone, k alone) or JOINTLY
%% over a CONTIGUOUS q+k span in one pass. A PERFORMANCE axis (BIT-IDENTICAL: same per-head
%% rotation at the same position; only the launch boundary changes). When the producing QKV GEMV
%% is fused, q and k are CONTIGUOUS views of one buffer, and a per-head-uniform op (RoPE rotates
%% every head identically at the given position) can sweep (nh+nkv) heads in ONE launch instead of
%% two -> a launch eliminated per layer. PRECONDITION: q and k must be contiguous and adjacent
%% (qk_joined is only valid when the layout guarantees it; the lowering asserts the contiguity and
%% FALLS BACK to per_tensor otherwise — a loud guard, not a silent assumption). Each backend lowers
%% it in its idiom (CUDA: one kernel over the joined head range; Rust/oxide: one loop over the
%% slice) — the INTENT is one fact: group per-head work over adjacent heads to amortize the launch.
%%   per_tensor — rope(q); rope(k)   (two passes, always valid).
%%   qk_joined  — rope(q‖k)          (one pass over nh+nkv contiguous heads; needs contiguity).
param_axis(rope, head_grouping, [per_tensor, qk_joined]).
%% activation_fold: which ELEMENT-LOCAL activation is FOLDED INTO the quantize op (the PROLOGUE
%% cousin of epilogue_compute — fold on the CONSUMER's read side instead of the PRODUCER's store
%% side). The down-proj quantizes its input activation; if that activation is silu(g)*u, the quant
%% kernel can compute silu(g[i])*u[i] per lane and feed it STRAIGHT into the warp-amax + quantize,
%% deleting the separate elementwise kernel AND the intermediate's global round-trip. BIT-IDENTICAL:
%% same silu (gv/(1+exp(-gv))*uv), same quant rounding (amax/rintf/fp16-scale) — the folded value
%% equals the two-kernel value, then quantized identically. ★ This is the GATE/UP-FUSION-COMPATIBLE
%% route: folding silu on the GEMV STORE side (epilogue_compute(silu)) conflicts with gate/up GEMV
%% fusion (they contend for the same store), but folding on the QUANT side does not — the veto of
%% the store-side silu (0.55x when it forces re-quant) MAPPED to this open door. Each backend lowers
%% it in its idiom; the INTENT is one fact: fuse the activation into the consumer's quantize.
%%   none      — quantize the activation as given (a separate elementwise produced it).
%%   silu_mul  — quantize silu(g)*u computed in-kernel (folds k_silu_mul; needs the (g,u) sources).
%%   rms_norm  — quantize rms_norm(x)*w computed in-kernel (folds k_rmsnorm; needs raw x + norm
%%               weight + eps). The RMS→QUANT SEAM. Unlike silu_mul (element-local), rms_norm needs a
%%               ROW REDUCTION first: inv = rsqrt(mean(x^2)+eps). So the fold is two-phase within one
%%               kernel: PHASE 1 the canonical rms sum-of-squares reduction -> inv (order-preserved
%%               == the standalone k_rmsnorm reduction); PHASE 2 each lane computes nv[j]=x[j]*inv*w[j]
%%               and feeds it STRAIGHT into the warp-amax + quantize (order-preserved == the standalone
%%               k_quant_q8). The two canonical orders are each already declared facts, so the
%%               composition declares cleanly and is BIT-IDENTICAL to running k_rmsnorm then k_quant_q8.
%%               Eliminates the k_rmsnorm launch AND the normalized-activation global round-trip
%%               (rms writes the 896-float vector, quant re-reads it — that round-trip is the seam).
%%               ★ BORN POLYGLOT: declared here as one fact; BOTH backends (CUDA via fused_norm,
%%               Rust/oxide via the oxide emitter) inherit the SAME intent; the cross-backend gate
%%               certifies both lowerings agree from birth — the first cross-backend-certified-from-
%%               birth fusion. silu_mul's sibling (quant-side fold, NO GEMV conflict — note the
%%               earlier rms→quant→GEMV mega-fusion in fused_norm_q8.pl was NEVER wired precisely
%%               because folding into the GEMV conflicts with QKV/gateup fusion; the quant-side-only
%%               seam is the COMPOSABLE one).
param_axis(quant, activation_fold, [none, silu_mul, rms_norm]).

%% ── op families: which axes parameterize each op ──
op_family(silu,    [form]).
op_family(softmax, [form, acc_type]).
op_family(gelu,    [gelu_approx]).
op_family(rms_norm,[acc_type]).
op_family(gemm,    [reduction_grouping, fma_contract, weight_access, epilogue_compute, output_target]).
op_family(rope,    [head_grouping]).
op_family(quant,   [activation_fold]).

%% ── which axis-value each reference pins (the union maps the axis) ──
%% silu/swish form
reference_pins(pytorch_f_silu,  form, divide).          %% F.silu = x/(1+exp(-x))
reference_pins(stanford_swish,  form, reciprocal_mul).  %% Stanford 25_Swish = x*sigmoid(x)
%% softmax form
reference_pins(pytorch_softmax, form, reciprocal_mul).  %% torch softmax: e*(1/sum)
%% reduction acc_type
reference_pins(pytorch_cpu, acc_type, f64).             %% ATen acc_type<float> = double
reference_pins(ggml,        acc_type, f32).             %% ggml f32-throughout
%% gelu form
reference_pins(pytorch_f_gelu,    gelu_approx, erf).    %% F.gelu default
reference_pins(stanford_newgelu,  gelu_approx, tanh).   %% 88_MinGPTNewGelu / approximate='tanh'
%% gemm reduction grouping
reference_pins(openblas_sandybridge, reduction_grouping, kc(256)).  %% empirical @512
reference_pins(ggml_b5311,           reduction_grouping, tiled).
reference_pins(torch_matmul_small,   reduction_grouping, sequential). %% bpd_mm matches @256-2048

%% ── axis-quality: WHY one might steer to a setting (beyond reference-matching) ──
%% This is the "transcend the references" layer: known tradeoffs per axis-value.
axis_quality(form, divide,        stability('better near overflow: single rounding, no reciprocal blowup')).
axis_quality(form, reciprocal_mul,speed('reciprocal computed once per group; N muls vs N divides — faster when divide is slow')).
axis_quality(acc_type, f64,       precision('lower reduction error; matches pytorch; ~2x reduction cost')).
axis_quality(acc_type, f32,       speed('faster, less memory; matches ggml; higher reduction error at large N')).
axis_quality(gelu_approx, erf,    precision('exact GELU; erff is the reference')).
axis_quality(gelu_approx, tanh,   speed('tanh-approx; faster than erff; tiny accuracy delta')).
axis_quality(reduction_grouping, sequential, precision('canonical order; but cache-poor at large N')).
axis_quality(reduction_grouping, kc(_),      speed('cache-blocked; matches OpenBLAS; changes rounding vs sequential')).
axis_quality(fma_contract, off,   determinism('no fmul+fadd->fma ambiguity; bit-deterministic; our -mno-fma target')).
axis_quality(fma_contract, on,    speed('fused multiply-add; faster but contracts rounding')).
axis_quality(weight_access, lane_strided,    simplicity('one 128-bit-vector (int4) load per lane per block; no staging; minimal regs/shmem')).
axis_quality(weight_access, warp_contiguous, speed('coalesced burst loads -> sectors/load toward ideal on arches where lane-strided UNDER-coalesces. BIT-IDENTICAL. NOTE: on sm_61 q8_0 lane_strided is ALREADY near-optimal (1.2 sectors/load, 86%% of achievable BW) so this is performance-equivalent there -- a real knob for OTHER arches/widths, not a lever on the P4')).
axis_quality(epilogue_compute, none,      simplicity('plain store; the producing op stands alone')).
axis_quality(epilogue_compute, silu,      speed('hides the SiLU transcendental under the gate GEMV memory wall (+0.2%%); k_silu_mul reduces to a bare multiply. BIT-IDENTICAL')).
axis_quality(epilogue_compute, bias,      speed('eliminates the q/k/v bias k_add launch + round-trip; trivial add hidden under memory. BIT-IDENTICAL')).
axis_quality(epilogue_compute, add_resid, speed('eliminates the residual k_add (shipped as addres, +5.9%% e2e); per-row add hidden under memory. BIT-IDENTICAL')).
axis_quality(output_target, scratch,     simplicity('GEMV writes fresh scratch; a separate copy moves it to the persistent carry buffer the next layer reads')).
axis_quality(output_target, fixed_carry, speed('GEMV writes the persistent carry buffer DIRECTLY -> eliminates the per-layer memcpy graph node (+0.63%% e2e). The captured graph gets structurally SMALLER. BIT-IDENTICAL (only the destination pointer changes). Requires no aliasing with the GEMV inputs')).
axis_quality(head_grouping, per_tensor, simplicity('rope q and k in separate passes; always valid regardless of layout')).
axis_quality(head_grouping, qk_joined,  speed('rope q and k in ONE pass over (nh+nkv) contiguous heads -> a launch eliminated per layer (+1.13%% e2e). BIT-IDENTICAL (same per-head rotation). Requires q,k contiguous+adjacent; lowering asserts contiguity and falls back loudly otherwise')).
axis_quality(activation_fold, none,     simplicity('quantize the activation as given; a separate elementwise kernel produced it')).
axis_quality(activation_fold, silu_mul, speed('fold silu(g)*u into the quantize -> eliminates k_silu_mul + the intermediate global round-trip (+0.87%% e2e). BIT-IDENTICAL. The gate/up-fusion-COMPATIBLE route the store-side silu veto mapped to')).
axis_quality(activation_fold, rms_norm, simplicity('fold rms_norm(x)*w into the quantize. BIT-IDENTICAL (two-phase: canonical rms reduction then canonical warp-amax). Born polyglot: one fact, both backends, cross-backend certified from birth. MEASURED e2e-NEUTRAL on the graph path (CUPTI: collapses the launches but does the same compute; the ~6%% wall-time it touches is compute, not the launch/round-trip overhead graph capture already hides). Kept default OFF -- valid, not profitable')).

%% ── steering gradients: axes ORDERED by objective (measured, not asserted) ──
%% axis_order(Axis, Objective, OrderedValues) — values listed best→worst for that objective.
%% Measured tonight: divide ≻ reciprocal_mul on precision (mean |err vs f64| 3.39e-8 vs 3.65e-8,
%% divide closer to f64-truth in 9/11 differing elems, worst-case tie). Mechanism: divide=1
%% rounding, recipmul=2 roundings; fewer roundings = more precise. recipmul ≻ divide on speed.
axis_order(form, precision, [divide, reciprocal_mul]).        %% divide more precise
axis_order(form, speed,     [reciprocal_mul, divide]).        %% recipmul faster (1 recip + N mul)
axis_order(acc_type, precision, [f64, f32]).                  %% f64 lower reduction error
axis_order(acc_type, speed,     [f32, f64]).
axis_order(gelu_approx, precision, [erf, tanh]).              %% erf is exact GELU
axis_order(gelu_approx, speed,     [tanh, erf]).
axis_order(reduction_grouping, precision, [sequential, kc(256), multi_acc(2)]). %% sequential canonical
axis_order(reduction_grouping, speed,     [kc(256), multi_acc(2), sequential]). %% blocked faster
axis_order(fma_contract, speed,       [on, off]).
axis_order(fma_contract, determinism, [off, on]).             %% off = bit-deterministic
axis_order(weight_access, speed,      [warp_contiguous, lane_strided]).  %% coalesced burst faster
axis_order(output_target, speed,      [fixed_carry, scratch]).           %% direct write > copy
axis_order(head_grouping, speed,      [qk_joined, per_tensor]).          %% one pass > two
axis_order(activation_fold, speed,    [rms_norm, silu_mul, none]).      %% folded > separate kernel (rms_norm folds the heaviest pair)
%% NOTE: output_target, head_grouping, activation_fold have NO precision/determinism order — like
%% weight_access, all values are BIT-IDENTICAL (the reduction_order / float DAG is held fixed). They
%% are PURE-PERFORMANCE axes: steer freely by speed, correctness invariant. (output_target changes a
%% destination pointer; head_grouping changes a launch boundary; activation_fold moves an elementwise
%% into the consumer's quantize — none touch which products are summed in what order.)
%% NOTE: weight_access has NO precision/determinism order — both values are BIT-IDENTICAL (the
%% reduction_order is held fixed). It is the rare PURE-performance axis: steer freely by speed,
%% the correctness is invariant. (Contrast fma_contract/acc_type, where the speed pick costs bits.)

%% axis_metric(Axis, V1, V2, Metric) — the MAGNITUDE of difference between two settings.
%% nearly_trivial threshold: max_ulp =< 1. The silu form gap is the canonical nearly-trivial divergence.
%% axis_metric: MEASURED magnitude per op (large-N, vs f64 truth). The precision-win-ratio
%% (fraction of differing elements where the single-rounding form is closer to truth) is
%% OP-SPECIFIC, not a universal constant: it scales with how INDEPENDENT the 2nd rounding is
%% per output (per-element double-rounding hurts more than shared/amortized).
%%   silu:    form reciprocal_mul has PER-ELEMENT 1/(1+e) double-rounding -> strong 0.66 (~2:1)
%%   softmax: form reciprocal_mul shares 1/sum across the row -> diluted 0.58 (~1.4:1)
%% DIRECTION (divide more precise) is the LAW; MAGNITUDE is measured per op.
axis_metric(form, divide, reciprocal_mul, [op(silu),    max_ulp(1), precision_win_ratio(0.66), nearly_trivial(true), mechanism(per_element_double_rounding)]).
axis_metric(form, divide, reciprocal_mul, [op(softmax), max_ulp(1), precision_win_ratio(0.58), nearly_trivial(true), mechanism(shared_reciprocal_double_rounding)]).
%% weight_access: MEASURED on the P4 (Tesla, sm_61) for the q8_0 128-bit-vector-load tiled GEMV,
%% 2026-06-12. ★ CORRECTED after careful sector analysis + an achievable-bandwidth measurement:
%% lane_strided is ALREADY near-optimally coalesced. Lane b reads block b (bytes [b*32,b*32+32));
%% its two 16B (128-bit-vector) loads w0,w1 fall in the SAME 32B sector and together use all 32B.
%% Across a warp the 32 lanes touch 32 distinct, fully-used sectors -> contiguous 1024B span.
%% sectors_per_load ~1.2 is near-ideal for 16B loads (the "0.5 ideal" needs 2 ADJACENT LANES to
%% share a sector; here the same lane's 2 loads share it). And the achievable-bandwidth floor is
%% 141.5 GB/s (measured stream copy = 74%% of the 192 theoretical, normal for GDDR5); the vocab
%% GEMV runs at 121 GB/s = 86%% of ACHIEVABLE. So warp_contiguous offers NEGLIGIBLE gain on this
%% kernel/arch -- the GEMV is at the DRAM wall, the same wall ggml/Ollama hit. The axis remains a
%% real backend-neutral knob (other arches/load-widths may differ), but on sm_61 q8_0 the values
%% are performance-EQUIVALENT, not a lever. (Lesson: the memory_dependency stall is the kernel
%% correctly waiting on near-saturated DRAM, not a coalescing defect -- measure achievable BW, not
%% theoretical, before chasing a bandwidth 'gap'.)
axis_metric(weight_access, lane_strided, warp_contiguous, [op(q8_0_gemv), max_ulp(0), target(sm_61), sectors_per_load(lane_strided(1.2), near_optimal_for_16B), achievable_gbs(141.5), vocab_gemv_gbs(121), pct_of_achievable(86), verdict(equivalent_on_sm61_q8_0), bit_identical(true)]).
%% epilogue_compute: MEASURED on the P4 (sm_61), 2026-06-12. The compute hides under the memory
%% wall -- adding the op to the store costs ~nothing because the SMs idle during DRAM stalls.
%% silu (transcendental, the most compute): +0.2%% on gate(4864x896 GEMV), +1.3%% on vocab. The
%% add-family (add_resid/bias) is trivial and also hides. ALL bit_identical (element-local, same
%% arithmetic, reduction_order untouched). The win is double: free compute + a launch + a DRAM
%% round-trip eliminated per fold. This is the endgame lever WHEN the GEMV is memory-bound: don't
%% speed up the bytes (at the wall), fill the compute shadow behind them.
axis_metric(epilogue_compute, none, silu, [op(gate_gemv), max_ulp(0), target(sm_61), overhead_pct(gate(0.2), vocab(1.3)), hidden(true), eliminates(silu_compute_from_k_silu_mul), bit_identical(true)]).
axis_metric(epilogue_compute, none, bias, [op(qkv_gemv), max_ulp(0), target(sm_61), overhead_pct(trivial), hidden(true), eliminates(k_add_bias_launch), bit_identical(true)]).
axis_metric(epilogue_compute, none, add_resid, [op(o_down_gemv), max_ulp(0), target(sm_61), e2e_gain_pct(5.9), hidden(true), eliminates(k_add_resid_launch), bit_identical(true), status(shipped_as_addres)]).
%% output_target / head_grouping / activation_fold: the chain-audit fusions, MEASURED on the P4
%% (sm_61), 2026-06-12 (Heath's "fusion that eliminates launches; memcpy hoisted out" directive).
%% All three are launch/round-trip/copy ELIMINATIONS — they remove graph nodes, not float ops, so
%% all are max_ulp(0), token-exact, and shipped ON in PRODUCTION_PROFILE. Combined +2.6%% e2e on top
%% of the small-pieces +2.2%%; CUPTI: 20970 -> 17514 (small-pieces) -> 15210 (chain) launches/decode.
axis_metric(output_target, scratch, fixed_carry, [op(down_gemv), max_ulp(0), target(sm_61), e2e_gain_pct(0.63), eliminates(per_layer_memcpy_dtod), graph_nodes_removed_per_token(24), bit_identical(true), requires(no_aliasing), status(shipped), toggle('_RESID_CARRY_FUSED')]).
axis_metric(head_grouping, per_tensor, qk_joined, [op(rope), max_ulp(0), target(sm_61), e2e_gain_pct(1.13), eliminates(one_rope_launch_per_layer), precondition(qk_contiguous_adjacent), guard(explicit_contiguity_check_falls_back), bit_identical(true), status(shipped), toggle('_ROPE_QK_FUSED')]).
axis_metric(activation_fold, none, silu_mul, [op(quant), max_ulp(0), target(sm_61), e2e_gain_pct(0.87), eliminates(k_silu_mul_launch_and_gu_round_trip), compatible_with(gateup_fusion), maps_from_veto(store_side_silu_0_55x), bit_identical(true), status(shipped), toggle('_SILU_QUANT_FUSED')]).
%% rms_norm fold (the RMS→QUANT SEAM): declared FACTS-FIRST (Bocher's proposal, Heath's "vein of
%% precious ore"). To be BORN POLYGLOT — CUDA + oxide lowerings from this one fact, cross-backend
%% certified from birth. status(declared) until both lowerings land + the gate certifies them.
%% e2e_gain estimated from the ~10% wall-time in the k_rmsnorm+k_quant_q8 pair (the launch + the
%% normalized-activation global round-trip); to be MEASURED. two_phase: canonical rms reduction
%% (order == standalone k_rmsnorm) then canonical warp-amax (order == standalone k_quant_q8).
axis_metric(activation_fold, none, rms_norm, [op(quant), max_ulp(0), target(sm_61), e2e_gain_pct(measured(-0.05)), eliminates(k_rmsnorm_launch_and_normed_activation_round_trip), wall_time_pair_pct(6), two_phase(rms_reduction_then_warp_amax), born_polyglot(true), bit_identical(true), status(measured_neutral), toggle('_RMS_QUANT_FUSED'), default(off), cupti_verdict('collapses 864 k_rmsnorm + 864 k_quant_q8 launches into 864 k_rms_quant, but the fused kernel does the SAME compute (8.86M ns) as the pair it replaces (9.04M ns) -> net -0.05% of 234M ns total. Targets ~6% of kernel time that is COMPUTE the fold does not remove, not launch/round-trip overhead (free under CUDA graph capture). GEMVs are 76% of kernel time and the only place left to win. VALID but not PROFITABLE on the graph path.')]).

%% ── axis resolution: default vs reference-pinned, with the precision tradeoff made explicit ──
%% The synthesis (Heath 2026-05-31): each axis has a PRINCIPLED DEFAULT (the most-precise
%% setting, from axis_order precision), AND a reference-pinned setting (from reference_pins).
%% When they agree -> free. When they differ -> we match the reference and KNOW the cost.

%% axis_default(Axis, Objective, Value) — the default setting for an objective (head of the order).
axis_default(Axis, Objective, Value) :-
    axis_order(Axis, Objective, [Value|_]).

%% resolve_axis(Op, Axis, Reference, Value, Note) — the setting to USE, and what it means.
%%   If a Reference pins the axis, use that (we must match it), and flag if it departs from
%%   the precision-default. If no Reference, use the precision-default (reasonable default).
resolve_axis(_Op, Axis, Reference, Value, matches_reference_at_precision_default) :-
    nonvar(Reference), reference_pins(Reference, Axis, Value),
    axis_default(Axis, precision, Value), !.
resolve_axis(_Op, Axis, Reference, Value, matches_reference_departs_from_precision_default(Default)) :-
    nonvar(Reference), reference_pins(Reference, Axis, Value),
    axis_default(Axis, precision, Default), Default \== Value, !.
resolve_axis(Op, Axis, _Reference, Value, precision_default_no_reference_pin) :-
    op_family(Op, Axes), member(Axis, Axes),
    axis_default(Axis, precision, Value).

%% ── reduction_order axis (discovered 2026-05-31 by factoring out acc_type) ──
%% After controlling acc_type=f64, the residual divergence on reductions splits into:
%%   whole_tensor       - contiguous full-tensor reduction; f64-acc = 0 ULP (Frobenius, MSE, sum/mean-all)
%%   strided_tensoriter - torch reduces over a DIM via TensorIterator: vectorized over the contiguous
%%                        inner dim (SIMD lanes) + tiled 2-pass order. f64-acc necessary but NOT
%%                        sufficient (Mean-dim 4 ULP, RMSNorm-dim 2 ULP residual until this order matched).
param_axis(reduce, reduction_order, [whole_tensor, strided_tensoriter]).
op_family(reduce, [acc_type, reduction_order]).
reference_pins(pytorch_cpu_whole, reduction_order, whole_tensor).
reference_pins(pytorch_cpu_dim,   reduction_order, strided_tensoriter).
axis_quality(reduction_order, whole_tensor,       simplicity(contiguous_seq_f64_zero_ulp)).
axis_quality(reduction_order, strided_tensoriter, fidelity(requires_tensoriterator_tiled_order)).

%% ── attn_decode_split reduction order (DECLARED 2026-06-12, Iyun; blessed by Bocher) ──────────
%% The split-K flash-decode attention's canonical reduction order. This is a RE-CANONICALIZATION
%% relative to the original k_attn_decode_masked: where the original did one flat sequential
%% left-fold over all L positions for the V-sum and a single global max, split-K does per-split
%% local-max + per-split sequential left-fold + a flash-rescale combine. The new order is DECLARED
%% canonical; the new kernel is 0-ULP TO THIS FACT, with a bounded one-time migration delta vs the
%% old left-fold (measured max_abs ~3-4e-8 at value scale; the V-sum re-parenthesization).
%%
%% reduction_order(attn_decode_split, ...) terms, each order-bearing:
%%   max(local_per_split)         - each split takes the max over ITS range only (not global-first).
%%   exp(against_local_max)       - exp(score - local_max) within the split.
%%   per_split_Z(strided_tree)    - partial denom = per-thread strided partials, tree-reduced (same
%%                                  shape as the original's Z tree, but over the split's range).
%%   per_split_V(sequential_left_fold) - partial numerator = acc += exp_t * V[t] over t in [p0,p1),
%%                                  strictly sequential within the range.
%%   combine(flash_rescale, split_index_order) - global M = max_sp(local_max_sp); for each split
%%                                  w_sp = exp(local_max_sp - M); Z = sum_sp w_sp*partZ_sp,
%%                                  O = sum_sp w_sp*partO_sp; both summed in ASCENDING split index.
%%   rescale_arith(exp_diff_then_mul) - the w_sp = exp(local_max_sp - M) is computed first, THEN
%%                                  multiplied into partZ/partO. This multiply is order-bearing
%%                                  (it is arithmetic the single-pass kernel never did) and is
%%                                  therefore PART of the declared order, not an accidental ULP.
%%   boundaries(contiguous_ceil_L_over_S, S_DEPENDENT) - ★ HONEST DECLARATION (Bocher's S-invariance
%%                                  condition): splits are contiguous equal ranges per = ceil(L/S);
%%                                  the order IS S-dependent (NOT power-of-2 subtree-invariant).
%%                                  S MUST be fixed per graph-capture so the order is pinned at
%%                                  capture time. An honest S-dependent declaration beats an
%%                                  aspirational S-invariant one. If a future wire wants S-invariance,
%%                                  re-declare with power-of-2 subtree-edge boundaries.
%%   status(banked_not_wired)     - validated frontier point (commit 92b925a0). Conditionally
%%                                  profitable: loses short-L, wins long-L (crossover ~L=80-100;
%%                                  L=120 -> 1.30x/1.38x at NSPLIT 2/4). NSPLIT=1 control = 0 ULP.
%%   gate_owed(pair_gate_to_fact, migration_delta_once, long_run_flip_cert_with_sentinel) - the
%%                                  after-gate OWED before any production dispatch. Attention is
%%                                  upstream of everything, so the flip-cert matters more than rmsnorm.
param_axis(attn_decode, reduction_order, [single_pass_global_max_leftfold, attn_decode_split]).
op_family(attn_decode, [reduction_order]).
axis_quality(reduction_order, single_pass_global_max_leftfold, simplicity(original_one_block_per_head_grid_nh)).
axis_quality(reduction_order, attn_decode_split,                speed(occupancy_grid_nh_times_S_wins_long_L_conditional)).
axis_metric(reduction_order, single_pass_global_max_leftfold, attn_decode_split, [op(attn_decode), max_abs(3.7e-8), target(sm_61), crossover_L(80_100), speedup(l40(0.84), l64(0.95), l120(1.38)), re_canonicalization(true), status(banked_not_wired), gate_owed(true)]).
