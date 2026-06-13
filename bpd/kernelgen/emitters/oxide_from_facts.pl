%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% oxide_from_facts.pl — generate Rust/CUDA-oxide kernels DIRECTLY from the
%% canonical robust_op_match.pl facts.
%%
%% The canonical fact carries a backend-agnostic C-style `formulation` (+ pinned
%% FP coordinates). This emitter:
%%   1. loads robust_op_match/5 facts
%%   2. reads the formulation for an op
%%   3. translates the C formulation -> a Rust per-element expression over `x`
%%   4. emits the complete cuda-oxide example (kernel + host + bit-identity check)
%%
%% Single source of truth: robust_op_match.pl. No per-backend duplicate facts.
%% The same translator pattern will drive the C++/CUDA backend (it keeps the C
%% form ~as-is) and others.
%%
%% Author: Iyun, 2026-06-06 (wire the cuda-oxide emitter to the canonical facts)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(oxide_from_facts, [emit_oxide_from_fact/2, emit_oxide_from_fact_mode/3, oxide_supported_op/1, emit_oxide_dump/4, emit_oxide_dump_mode/5, emit_oxide_q8_gemv/1]).


:- use_module(library(lists)).

%% Import the shared expression IR (lower_rust + op_expr) so cuda-oxide generates
%% from the SAME neutral AST as cuda_c/mlir/llvm/torch. Resolve relative to here.
:- ( prolog_load_context(directory, ED2),
     atomic_list_concat([ED2, '/expr_ir.pl'], EP2), exists_file(EP2)
   -> use_module(EP2, [lower_rust/2, op_expr/2])
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> use_module('kernelgen/emitters/expr_ir.pl', [lower_rust/2, op_expr/2])
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> use_module('kernelgen/emitters/expr_ir.pl', [lower_rust/2, op_expr/2])
   ;  true ).

%% op_rust_expr(+Op,+NanMode,-RustExpr): UNIFIED body source — prefer the shared
%% AST (op_expr -> lower_rust), fall back to the legacy formulation-string
%% translate for ops without an expr term. Mirrors cuda_c's op_cuda_expr.
op_rust_expr(Op, NanMode, RustExpr) :-
    ( op_expr(Op, Expr)
    -> lower_rust(Expr, Raw)
    ;  robust_op_match(unary_elementwise, Op, _, _, Ev),
       member(formulation(F), Ev), formulation_to_rust(F, Raw) ),
    apply_nan_mode(NanMode, Raw, RustExpr).

expr_has_erf(E) :- sub_string(E, _, _, _, "__nv_erff").

%% Load the canonical facts (robust_op_match/5).
%% Load the canonical facts (robust_op_match/5), resolved relative to this
%% emitter's location (bpd/kernelgen/emitters/ -> ../../lib/robust_op_match.pl),
%% with absolute fallbacks for the enclave/laptop checkouts.
:- ( prolog_load_context(directory, Dir),
     atomic_list_concat([Dir, '/../../lib/robust_op_match.pl'], RelPath),
     exists_file(RelPath)
   -> consult(RelPath)
   ;  exists_file('lib/robust_op_match.pl')
   -> consult('lib/robust_op_match.pl')
   ;  exists_file('lib/robust_op_match.pl')
   -> consult('lib/robust_op_match.pl')
   ;  exists_file('robust_op_match.pl')
   -> consult('robust_op_match.pl')
   ;  true ).

%% ── C-formulation -> Rust-expression translation ───────────────────────────
%% Translate a robust_op_match formulation string (over variable x) into a Rust
%% per-element expression over `v: f32`. Handles ternary, transcendental fns,
%% and f32 literal suffixes. The translation PRESERVES the FP form (divide vs
%% reciprocal, literal-subtract vs expm1) so bit-identity is by construction.

%% -- nan_mode: transform the per-element expr per the chosen NaN policy --
%% propagate (default): keep the fact's form (carries the x!=x guard for
%% flush-prone ops). fast: strip the guard (NaN unspecified, faster).
apply_nan_mode(propagate, Expr, Expr).
apply_nan_mode(fast, Expr, Out) :-
    ( strip_nan_guard(Expr, Out) -> true ; Out = Expr ).

%% strip a leading Rust NaN guard "if v != v { v } else <REST>" -> <REST>
strip_nan_guard(Expr, Rest) :-
    atom_string(Expr, S),
    string_concat("if v != v { v } else ", Rest, S).

nan_policy_note(propagate, "propagate (IEEE: NaN flows through; matches torch; no nnan fast-math)").
nan_policy_note(fast, "fast (assume-no-NaN: NaN unspecified; guard stripped; nnan fast-math allowed)").

formulation_to_rust(CForm, RustExpr) :-
    atom_string(CForm, S),
    rust_translate(S, RustExpr).

%% Op-specific translations, keyed on the canonical formulation string.
%% (Explicit per-form rules — a parser would generalize, but explicit rules
%%  keep the FP-knob translation auditable, which is the whole point.)
rust_translate("x != x ? x : (x >= 0 ? x : 0.0f)",
    "if v != v { v } else if v >= 0.0 { v } else { 0.0 }").
rust_translate("tanhf(x)",
    "v.tanh()").
rust_translate("x * 0.5f * (1.0f + erff(x * 0.7071067811865476f))",
    "v * 0.5 * (1.0 + unsafe { __nv_erff(v * 0.7071067811865476) })").
rust_translate("x <= 0 ? expf(x) - 1.0f : x",
    "if v <= 0.0 { v.exp() - 1.0 } else { v }").
rust_translate("(scale*alpha) * (expf(x) - 1.0f) for x<=0, else scale*x",
    "{ let scale=1.0507009873_f32; let alpha=1.6732632423_f32; if v <= 0.0 { (scale*alpha)*(v.exp()-1.0) } else { scale*v } }").
%% silu/sigmoid: not in robust_op_match yet, but the divide-form recipe is known
%% (swiglu_fused_emitter.pl). Added here so the fact->rust path covers them.
rust_translate("x / (1 + expf(-x))",
    "v / (1.0 + (-v).exp())").
rust_translate("1 / (1 + expf(-x))",
    "1.0 / (1.0 + (-v).exp())").

%% Which ops can we currently translate (have a rust_translate rule)?
oxide_supported_op(Op) :-
    robust_op_match(unary_elementwise, Op, _, _, Ev),
    member(formulation(F), Ev),
    formulation_to_rust(F, _).

%% ── emit a cuda-oxide kernel for an op, reading its canonical fact ──────────
emit_oxide_from_fact(Op, OutFile) :-
    default_nan_mode(Op, Mode),
    emit_oxide_from_fact_mode(Op, Mode, OutFile).

default_nan_mode(Op, propagate) :-
    robust_op_match(unary_elementwise, Op, _, _, Ev),
    member(nan_propagation(ieee), Ev), !.
default_nan_mode(_, propagate).

emit_oxide_from_fact_mode(Op, NanMode, OutFile) :-
    robust_op_match(unary_elementwise, Op, Ref, Tier, Ev),
    ( member(formulation(F), Ev) -> true ; F = '(unspecified)' ),
    op_rust_expr(Op, NanMode, RustExpr),  %% unified: shared AST -> lower_rust
    %% strip the bpd_ prefix for the kernel name
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, '/* GENERATED from robust_op_match(~w) — canonical fact.~n', [Op]),
    format(S, ' * formulation: ~w~n', [F]),
    format(S, ' * reference: ~w  tier: ~w~n', [Ref, Tier]),
    format(S, ' * Rust/cuda-oxide backend. Rust->PTX(sm_61) on the P4.~n */~n', []),
    format(S, 'use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};~n', []),
    format(S, 'use cuda_device::{DisjointSlice, cuda_module, kernel, thread};~n~n', []),
    format(S, '#[cuda_module]~nmod kernels {~n    use super::*;~n    #[kernel]~n', []),
    format(S, '    pub fn ~w(x: &[f32], mut c: DisjointSlice<f32>) {~n', [Name]),
    format(S, '        let idx = thread::index_1d();~n        let idx_raw = idx.get();~n', []),
    format(S, '        if let Some(ce) = c.get_mut(idx) {~n            let v = x[idx_raw];~n', []),
    format(S, '            *ce = ~w;~n        }~n    }~n}~n~n', [RustExpr]),
    format(S, 'fn ~w_ref(v: f32) -> f32 { ~w }~n~n', [Name, RustExpr]),
    emit_host(S, Name),
    close(S),
    format("Generated cuda-oxide ~w from canonical fact -> ~w~n", [Name, OutFile]).

emit_host(S, Name) :-
    format(S, 'fn main() {~n', []),
    format(S, '    println!("=== GENERATED-FROM-FACT cuda-oxide ~w on Tesla P4 ===\\n");~n', [Name]),
    format(S, '    let ctx = CudaContext::new(0).expect("ctx");~n    let stream = ctx.default_stream();~n', []),
    format(S, '    let mut x_host: Vec<f32> = vec![0.0, -0.0, f32::NAN, -1.0, 1.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 3.4e38, -3.4e38];~n', []),
    format(S, '    for i in 0..1015 { let t=(i as f32)*0.013-6.6; x_host.push(t*(if i%3==0 {-1.0} else {1.0})); }~n', []),
    format(S, '    let n = x_host.len();~n', []),
    format(S, '    let xd = DeviceBuffer::from_host(&stream,&x_host).unwrap();~n', []),
    format(S, '    let mut cd = DeviceBuffer::<f32>::zeroed(&stream,n).unwrap();~n', []),
    format(S, '    let m = kernels::load(&ctx).expect("load");~n', []),
    format(S, '    m.~w(&stream, LaunchConfig::for_num_elems(n as u32), &xd, &mut cd).expect("launch");~n', [Name]),
    format(S, '    let ch = cd.to_host_vec(&stream).unwrap();~n', []),
    format(S, '    let mut diffs=0; let mut first=None;~n', []),
    format(S, '    for i in 0..n { let want=~w_ref(x_host[i]); let got=ch[i];~n', [Name]),
    format(S, '        let eq = want.to_bits()==got.to_bits() || (want.is_nan()&&got.is_nan());~n', []),
    format(S, '        if !eq { if first.is_none() { first=Some((i,x_host[i],want,got)); } diffs+=1; } }~n', []),
    format(S, '    if diffs==0 { println!("  *** 0 differences — BIT-IDENTICAL ({} elems) ***", n); }~n', []),
    format(S, '    else { let (i,xi,w,g)=first.unwrap();~n', []),
    format(S, '        println!("  {} differ. first idx {} x={} want={:#010x} got={:#010x}", diffs,i,xi,w.to_bits(),g.to_bits());~n', []),
    format(S, '        std::process::exit(1); }~n}~n', []).

%% ── dump-mode: cuda-oxide runner reading input.bin -> <op>_oxide.bin ──
emit_oxide_dump(Op, InBin, OutBin, OutFile) :-
    default_nan_mode(Op, Mode),
    emit_oxide_dump_mode(Op, Mode, InBin, OutBin, OutFile).

emit_oxide_dump_mode(Op, NanMode, InBin, OutBin, OutFile) :-
    robust_op_match(unary_elementwise, Op, _, _, Ev),
    member(formulation(F), Ev), formulation_to_rust(F, RustExpr0),
    apply_nan_mode(NanMode, RustExpr0, RustExpr),
    (atom_concat("bpd_", Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, "use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};~n", []),
    format(S, "use cuda_device::{DisjointSlice, cuda_module, kernel, thread};~n", []),
    format(S, "use std::io::{Read, Write};~n", []),
    ( expr_has_erf(RustExpr) ->
        format(S, "#[cuda_device::device]~nunsafe extern \"C\" { fn __nv_erff(x: f32) -> f32; }~n~n", [])
    ; format(S, "~n", []) ),
    format(S, "#[cuda_module]~nmod kernels {~n    use super::*;~n    #[kernel]~n", []),
    format(S, "    pub fn ~w(x: &[f32], mut c: DisjointSlice<f32>) {~n", [Name]),
    format(S, "        let idx=thread::index_1d(); let i=idx.get();~n", []),
    format(S, "        if let Some(ce)=c.get_mut(idx){ let v=x[i]; *ce=~w; }~n    }~n}~n~n", [RustExpr]),
    format(S, "fn main() {~n", []),
    format(S, "    let mut buf=Vec::new(); std::fs::File::open(\"~w\").unwrap().read_to_end(&mut buf).unwrap();~n", [InBin]),
    format(S, "    let x: Vec<f32> = buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();~n", []),
    format(S, "    let n=x.len();~n", []),
    format(S, "    let ctx=CudaContext::new(0).unwrap(); let stream=ctx.default_stream();~n", []),
    format(S, "    let xd=DeviceBuffer::from_host(&stream,&x).unwrap();~n", []),
    format(S, "    let mut cd=DeviceBuffer::<f32>::zeroed(&stream,n).unwrap();~n", []),
    format(S, "    let m=kernels::load(&ctx).unwrap();~n", []),
    format(S, "    m.~w(&stream, LaunchConfig::for_num_elems(n as u32), &xd, &mut cd).unwrap();~n", [Name]),
    format(S, "    let ch=cd.to_host_vec(&stream).unwrap();~n", []),
    format(S, "    let mut o=std::fs::File::create(\"~w\").unwrap();~n", [OutBin]),
    format(S, "    for v in &ch { o.write_all(&v.to_le_bytes()).unwrap(); }~n", []),
    format(S, "    println!(\"~w_oxide dumped {} elems\", n);~n}~n", [Name]),
    close(S),
    format("oxide dump-runner ~w -> ~w~n", [Name, OutFile]).


%% ═══════════════════════════════════════════════════════════════════════════
%% q8_0 dp4a GEMV — fact-driven Rust/cuda-oxide emission (the dominant op).
%% Reads the q8_0_op_expr fact (q8_0_dot(block(32), scale(fp16), quant(int8))) and emits the
%% canonical_serial reduction order — BIT-IDENTICAL to the CUDA emitter's emit_q8_0_gemv_canonical_
%% serial (both render from reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma),
%% tree(shfl_down,5))). Same fact, two backends, 0 ULP (proven: oxide_cuda_xcheck.py).
%% The KERNEL ONLY (host harness is the example's main.rs). One thread per row: 32 fma-folded lane
%% partials over strided blocks, then the 5-level shfl-down tree merge — in-thread (no warp shuffle,
%% so it lowers on sm_61 where .sync shuffles do not). dp4a = exact integer dot (== __dp4a).
%% ═══════════════════════════════════════════════════════════════════════════
emit_oxide_q8_gemv(OutFile) :-
    %% derive the block size from the fact (block(B)); honors "read the fact, don't hardcode"
    ( q8_0_op_expr(q8_0_dot(block(B), scale(fp16), quant(int8))) -> true ; B = 32 ),
    open(OutFile, write, S),
    format(S, "// q8_0 dp4a GEMV — FACT-DERIVED Rust/cuda-oxide kernel (canonical_serial order).~n", []),
    format(S, "// reduction_order(q8_gemv_dp4a, lanes(~w), strided, accum(fma), tree(shfl_down,5))~n", [B]),
    format(S, "// 0-ULP cross-backend to emit_q8_0_gemv_canonical_serial (CUDA). Generated, not copied.~n", []),
    format(S, "#[kernel]~n", []),
    format(S, "pub fn q8_gemv(wq: &[i8], wd: &[f32], xq: &[i8], xd: &[f32],~n", []),
    format(S, "               mut y: cuda_device::DisjointSlice<f32>, k: u32) {~n", []),
    format(S, "    let gid = cuda_device::thread::index_1d();~n", []),
    format(S, "    let row = gid.get();~n", []),
    format(S, "    let m = y.len();~n", []),
    format(S, "    if row >= m { return; }~n", []),
    format(S, "    let k_size = k as usize;~n", []),
    format(S, "    let nb = k_size / ~w;~n", [B]),
    format(S, "    let mut lane_acc = [0.0f32; ~w];~n", [B]),
    %% 1) lane partials, fma-contracted strided fold
    format(S, "    let mut lane = 0usize;~n", []),
    format(S, "    while lane < ~w {~n", [B]),
    format(S, "        let mut acc = 0.0f32;~n", []),
    format(S, "        let mut b = lane;~n", []),
    format(S, "        while b < nb {~n", []),
    format(S, "            let mut isum: i32 = 0;~n", []),
    format(S, "            let base = b * ~w;~n", [B]),
    format(S, "            let mut j = 0usize;~n", []),
    format(S, "            while j < ~w {~n", [B]),
    format(S, "                isum += (wq[row * k_size + base + j] as i32) * (xq[base + j] as i32);~n", []),
    format(S, "                j += 1;~n", []),
    format(S, "            }~n", []),
    format(S, "            acc = (wd[row * nb + b] * xd[b]).mul_add(isum as f32, acc); // fused == __fmaf_rn~n", []),
    format(S, "            b += ~w;~n", [B]),
    format(S, "        }~n", []),
    format(S, "        lane_acc[lane] = acc;~n", []),
    format(S, "        lane += 1;~n", []),
    format(S, "    }~n", []),
    %% 2) 5-level shfl-down tree merge
    format(S, "    let mut s = ~w / 2usize;~n", [B]),
    format(S, "    while s > 0 {~n", []),
    format(S, "        let mut t = 0usize;~n", []),
    format(S, "        while t < s {~n", []),
    format(S, "            lane_acc[t] += lane_acc[t + s];~n", []),
    format(S, "            t += 1;~n", []),
    format(S, "        }~n", []),
    format(S, "        s >>= 1;~n", []),
    format(S, "    }~n", []),
    format(S, "    unsafe { *y.get_unchecked_mut(row) = lane_acc[0]; }~n", []),
    format(S, "}~n", []),
    close(S),
    format("Generated FACT-DERIVED q8_0 GEMV (oxide/Rust) -> ~w~n", [OutFile]).
