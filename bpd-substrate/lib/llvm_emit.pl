%% llvm_emit.pl — Declarative LLVM IR generation from Prolog facts.
%%
%% Instead of format strings, LLVM constructs are Prolog terms:
%%
%%   llvm_extern(Name, RetType, ArgTypes)     — declare external function
%%   llvm_global(Name, Type, Init)            — global variable
%%   llvm_string(Name, Value)                 — string constant
%%   llvm_function(Name, RetType, Args, Body) — define function
%%
%% The emitter walks these facts and produces valid LLVM IR.
%% Every .so shares common patterns (dlopen, dlsym, PL_* FLI).
%%
%% This is the BPD thesis applied to code generation:
%% the specification IS Prolog facts, the projection IS LLVM IR.
%%
%% Author: mavchin (2026-06-01)

:- module(llvm_emit, [
    emit_declares/1,
    emit_globals/1,
    emit_strings/1,
    emit_type/2,
    emit_arglist/2,
    emit_declare/4,
    emit_header/2,
    emit_lazy_init/3,
    emit_install/2,
    llvm_extern/3,
    %% Path A extension (2026-09-23, mavhir):
    %% kernel-scaffolding primitives for term-based emission
    %% (Shape 2 per Heath: term-language, not format-strings).
    emit_program/2,
    emit_kernel_type/2,
    emit_kernel_value/2
]).

%% ============================================================
%% LLVM type representation
%% ============================================================

%% emit_type(+Type, -Str)
emit_type(void, "void").
emit_type(ptr, "ptr").
emit_type(i1, "i1").
emit_type(i8, "i8").
emit_type(i32, "i32").
emit_type(i64, "i64").
emit_type(f32, "float").
emit_type(f64, "double").
emit_type(array(N, T), Str) :-
    emit_type(T, TS),
    format(atom(Str), "[~w x ~w]", [N, TS]).
emit_type(varargs, "...").

%% ============================================================
%% Common LLVM patterns — every PLF .so needs these
%% ============================================================

%% C runtime
:- discontiguous llvm_extern/3.
llvm_extern(dlopen, ptr, [ptr, i32]).
llvm_extern(dlsym, ptr, [ptr, ptr]).
llvm_extern(printf, i32, [ptr, varargs]).
llvm_extern(malloc, ptr, [i64]).
llvm_extern(free, void, [ptr]).
llvm_extern(memset, ptr, [ptr, i32, i64]).

%% Linux syscall
llvm_extern(syscall, i64, [i64, varargs]).
llvm_extern(ioctl, i32, [i32, i64, varargs]).
llvm_extern(read, i64, [i32, ptr, i64]).
llvm_extern(close, i32, [i32]).

%% SWI-Prolog FLI — the bridge to Prolog
llvm_extern('PL_new_term_ref', i64, []).
llvm_extern('PL_copy_term_ref', i64, [i64]).
llvm_extern('PL_unify_integer', i32, [i64, i64]).
llvm_extern('PL_unify_float', i32, [i64, f64]).
llvm_extern('PL_unify_atom_chars', i32, [i64, ptr]).
llvm_extern('PL_unify_nil', i32, [i64]).
llvm_extern('PL_unify_list', i32, [i64, i64, i64]).
llvm_extern('PL_unify', i32, [i64, i64]).
llvm_extern('PL_get_integer', i32, [i64, ptr]).
llvm_extern('PL_get_chars', i32, [i64, ptr, i32]).
llvm_extern('PL_put_atom_chars', i32, [i64, ptr]).
llvm_extern('PL_put_float', i32, [i64, f64]).
llvm_extern('PL_cons_functor', i32, [i64, i64, i64, i64]).
llvm_extern('PL_new_functor', i64, [i64, i32]).
llvm_extern('PL_new_atom', i64, [ptr]).
llvm_extern('PL_register_foreign', i32, [ptr, i32, ptr, i32]).

%% CUPTI Activity API
llvm_extern(cuptiActivityRegisterCallbacks, i32, [ptr, ptr]).
llvm_extern(cuptiActivityEnable, i32, [i32]).
llvm_extern(cuptiActivityDisable, i32, [i32]).
llvm_extern(cuptiActivityFlushAll, i32, [i32]).
llvm_extern(cuptiActivityGetNextRecord, i32, [ptr, i64, ptr]).
llvm_extern(cuptiGetResultString, void, [i32, ptr]).

%% CUDA driver
llvm_extern(cuInit, i32, [i32]).
llvm_extern(cuCtxGetCurrent, i32, [ptr]).

%% ============================================================
%% Emitters — project facts to LLVM IR text
%% ============================================================

%% Emit all declare directives for registered externs
emit_declares(S) :-
    forall(llvm_extern(Name, RetTy, ArgTys),
           emit_declare(S, Name, RetTy, ArgTys)).

emit_declare(S, Name, RetTy, ArgTys) :-
    emit_type(RetTy, RetStr),
    emit_arglist(ArgTys, ArgStr),
    format(S, 'declare ~w @~w(~w)~n', [RetStr, Name, ArgStr]).

emit_arglist([], "").
emit_arglist([varargs], "...").
emit_arglist([T], Str) :-
    T \= varargs,
    emit_type(T, Str).
emit_arglist([T|Rest], Str) :-
    Rest \= [],
    T \= varargs,
    emit_type(T, TS),
    emit_arglist(Rest, RestStr),
    format(atom(Str), "~w, ~w", [TS, RestStr]).

%% Emit global variables
emit_globals(S) :-
    forall(llvm_global(Name, Type, Init),
           emit_global(S, Name, Type, Init)).

emit_global(S, Name, Type, null) :-
    emit_type(Type, TS),
    format(S, '@~w = internal global ~w null~n', [Name, TS]).
emit_global(S, Name, Type, zero) :-
    emit_type(Type, TS),
    format(S, '@~w = internal global ~w 0~n', [Name, TS]).
emit_global(S, Name, Type, zeroinitializer) :-
    emit_type(Type, TS),
    format(S, '@~w = internal global ~w zeroinitializer~n', [Name, TS]).
emit_global(S, Name, Type, value(V)) :-
    emit_type(Type, TS),
    format(S, '@~w = internal global ~w ~w~n', [Name, TS, V]).

%% Emit string constants
emit_strings(S) :-
    forall(llvm_string(Name, Value),
           emit_string(S, Name, Value)).

emit_string(S, Name, Value) :-
    atom_length(Value, Len),
    Len1 is Len + 1,
    format(S, '@~w = private constant [~w x i8] c"~w\\00"~n', [Name, Len1, Value]).

%% ============================================================
%% PLF bridge pattern — reusable for any library bridge
%% ============================================================

%% A PLF bridge specification:
%%   plf_bridge(LibName, LibSoName, Predicates)
%%   plf_predicate(PrologName, Arity, CFuncName)
%%   plf_dlsym_func(CName, NvmlName, FnPtrGlobal)

%% Emit the dlopen/dlsym lazy-init pattern
emit_lazy_init(S, LibSoStr, FnPtrGlobals) :-
    format(S, 'define internal i32 @ensure_lib_init() {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %already = load i32, ptr @lib_initialized~n', []),
    format(S, '  %done = icmp ne i32 %already, 0~n', []),
    format(S, '  br i1 %done, label %ret_ok, label %do_init~n', []),
    format(S, '~n', []),
    format(S, 'do_init:~n', []),
    format(S, '  %lib = call ptr @dlopen(ptr @~w, i32 2)~n', [LibSoStr]),
    format(S, '  %lib_null = icmp eq ptr %lib, null~n', []),
    format(S, '  br i1 %lib_null, label %ret_fail, label %resolve~n', []),
    format(S, '~n', []),
    format(S, 'resolve:~n', []),
    format(S, '  store ptr %lib, ptr @lib_handle~n', []),
    forall(member(fn(Global, SymStr), FnPtrGlobals),
           (format(S, '  %~w_p = call ptr @dlsym(ptr %lib, ptr @~w)~n', [Global, SymStr]),
            format(S, '  store ptr %~w_p, ptr @~w~n', [Global, Global]))),
    format(S, '  store i32 1, ptr @lib_initialized~n', []),
    format(S, '  br label %ret_ok~n', []),
    format(S, '~n', []),
    format(S, 'ret_ok:~n', []),
    format(S, '  ret i32 0~n', []),
    format(S, 'ret_fail:~n', []),
    format(S, '  ret i32 1~n', []),
    format(S, '}~n~n', []).

%% Emit the install() function that registers all PLF predicates
emit_install(S, Predicates) :-
    format(S, 'define void @install() {~n', []),
    forall(member(pred(NameStr, Arity, CFunc), Predicates),
           format(S, '  call i32 @PL_register_foreign(ptr @~w, i32 ~w, ptr @~w, i32 0)~n',
                  [NameStr, Arity, CFunc])),
    format(S, '  ret void~n', []),
    format(S, '}~n', []).

%% ============================================================
%% LLVM IR header (common to all targets)
%% ============================================================

emit_header(S, Triple) :-
    format(S, '; Generated by llvm_emit.pl — declarative LLVM IR from Prolog facts~n', []),
    format(S, '; Path: Prolog facts → llvm_emit rules → LLVM IR → llc → .so → hardware~n', []),
    format(S, '~n', []),
    format(S, 'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"~n', []),
    format(S, 'target triple = "~w"~n', [Triple]),
    format(S, '~n', []).

%% ============================================================
%% Path A extension (2026-09-23, mavhir):
%% Kernel-scaffolding primitives — term-based LLVM IR emission
%% for the llvm_from_facts.pl rewrite.
%%
%% Design: the terms ARE the specification; emit_program/2
%% dispatches per term-type to renderers. Replaces the raw
%% format-string anti-pattern with a term-language per Heath's
%% direction. Same output byte-for-byte as the pre-rewrite
%% format-string emission (28-golden diff-gate).
%%
%% Term vocabulary (see also /tmp/llvm_emit_kernel_design.md):
%%
%%   llvm_program(Preamble, Body).
%%     Preamble = preamble(Comment, HeaderOpt, Decls, MetaPre).
%%       Comment    = comment(Str) | no_comment
%%       HeaderOpt  = header(Triple, Datalayout) | no_header
%%       Decls      = list of raw_decl(Atom) or decl(Fn, RetT, ArgTs)
%%       MetaPre    = list of metadata terms placed before functions
%%     Body       = list of top-level items:
%%                    llvm_function(Name, RetType, Args, BBs)
%%                  | metadata(...)
%%
%%   arg(Name, Type, Attrs).
%%   basic_block(Label, Instructions).
%%
%%   Types: void, i1, i8, i32, i64, float, double
%%          ptr(ElemType)  or  ptr(ElemType, addrspace(N))
%%
%%   Values: ssa(Name) | label(Name) | imm(N) | fimm(F) | Atom
%%           (bare atoms rendered as %<atom>; imm/fimm are literals)
%%
%%   Instructions (per-line term-forms):
%%     phi(R, Type, Entries)   Entries = list of (Val, LabelName)
%%     icmp(R, Op, Type, A, B)
%%     gep_ptr(R, ElemT, PtrT, Ptr, Idx)
%%     load(R, Type, PtrT, Ptr, Align)
%%     store(V, Type, PtrT, Ptr, Align)
%%     add_int(R, Type, A, B)
%%     mul_int(R, Type, A, B)
%%     and_i1(R, A, B)
%%     sext(R, FromT, ToT, V)
%%     call_i32(R, FnName, Args)   Args = list of arg(Type, Value)
%%     call_void(FnName, Args)
%%     br_uncond(Label)
%%     br_cond(Cond, ThenL, ElseL)
%%     ret_void
%%     raw_ssa(Atom)            passthrough for lower_llvm output
%%     raw_ssa_lines(ListOfAtoms)   emits each atom on its own line
%%     multi_instr(ListOfInstrs)     emits ~2-3 instrs on one line joined by "  "
%%     comment_line(Str)         "; <Str>"
%% ============================================================

:- discontiguous emit_instr/2.
:- discontiguous emit_top_item/2.

%% emit_program(+Program, +Stream): top-level renderer.
emit_program(llvm_program(Preamble, Body), S) :-
    emit_preamble(Preamble, S),
    forall(member(Item, Body), emit_top_item(Item, S)).

%% emit_preamble/2
emit_preamble(preamble(Comment, HeaderOpt, Decls, MetaPre), S) :-
    emit_comment(Comment, S),
    emit_header_opt(HeaderOpt, S),
    forall(member(D, Decls), emit_decl(D, S)),
    forall(member(M, MetaPre), emit_top_item(M, S)).

emit_comment(no_comment, _).
emit_comment(comment(Str), S) :-
    format(S, "~w~n", [Str]).

emit_header_opt(no_header, _).
emit_header_opt(header(Triple, Datalayout), S) :-
    format(S, "target triple = \"~w\"~n", [Triple]),
    format(S, "target datalayout = \"~w\"~n", [Datalayout]).

emit_decl(raw_decl(Atom), S) :-
    format(S, "~w~n", [Atom]).
emit_decl(decl(Fn, RetT, ArgTs), S) :-
    emit_kernel_type(RetT, RetStr),
    emit_kernel_arg_type_list(ArgTs, ArgsStr),
    format(S, "declare ~w @~w(~w)~n", [RetStr, Fn, ArgsStr]).

%% emit_top_item/2: functions or top-level metadata
emit_top_item(llvm_function(Name, RetType, Args, BBs), S) :-
    emit_kernel_type(RetType, RetStr),
    emit_kernel_argsig_list(Args, ArgsStr),
    format(S, "define ~w @~w(~w) {~n", [RetStr, Name, ArgsStr]),
    forall(member(BB, BBs), emit_basic_block(BB, S)),
    format(S, "}~n", []).
emit_top_item(metadata(named, MdName, Vals), S) :-
    md_vals_str(Vals, VStr),
    format(S, "!~w = !{~w}~n", [MdName, VStr]).
emit_top_item(metadata(unnamed, Id, kernel_metadata(FnType, FnName, KernelStr, IntVal)), S) :-
    format(S, "!~w = !{~w @~w, !\"~w\", i32 ~w}~n",
           [Id, FnType, FnName, KernelStr, IntVal]).

md_vals_str([V], VStr) :- !,
    format(atom(VStr), "!~w", [V]).
md_vals_str([V|Rest], Str) :-
    md_vals_str(Rest, RestStr),
    format(atom(Str), "!~w, ~w", [V, RestStr]).

%% emit_basic_block/2
emit_basic_block(basic_block(Label, Instructions), S) :-
    format(S, "~w:~n", [Label]),
    forall(member(Instr, Instructions), emit_instr(Instr, S)).

%% emit_instr/2 — one dispatch per instruction type.
%% All indent by "  " (two spaces), matching the current format-string output.

emit_instr(phi(R, Type, Entries), S) :-
    emit_kernel_type(Type, TStr),
    phi_entries_str(Entries, EStr),
    format(S, "  %~w = phi ~w ~w~n", [R, TStr, EStr]).

phi_entries_str([(V, L)], Str) :- !,
    emit_kernel_value(V, VStr),
    format(atom(Str), "[~w, %~w]", [VStr, L]).
phi_entries_str([(V, L)|Rest], Str) :-
    emit_kernel_value(V, VStr),
    phi_entries_str(Rest, RestStr),
    format(atom(Str), "[~w, %~w], ~w", [VStr, L, RestStr]).

emit_instr(icmp(R, Op, Type, A, B), S) :-
    emit_kernel_type(Type, TStr),
    emit_kernel_value(A, AStr),
    emit_kernel_value(B, BStr),
    format(S, "  %~w = icmp ~w ~w ~w, ~w~n", [R, Op, TStr, AStr, BStr]).

emit_instr(gep_ptr(R, ElemT, PtrT, Ptr, Idx), S) :-
    emit_kernel_type(ElemT, EStr),
    emit_kernel_type(PtrT, PStr),
    emit_kernel_value(Ptr, PtrStr),
    emit_kernel_value(Idx, IdxStr),
    format(S, "  %~w = getelementptr ~w, ~w ~w, i64 ~w~n",
           [R, EStr, PStr, PtrStr, IdxStr]).

emit_instr(load(R, Type, PtrT, Ptr, Align), S) :-
    emit_kernel_type(Type, TStr),
    emit_kernel_type(PtrT, PtrTStr),
    emit_kernel_value(Ptr, PtrStr),
    format(S, "  %~w = load ~w, ~w ~w, align ~w~n",
           [R, TStr, PtrTStr, PtrStr, Align]).

emit_instr(store(V, Type, PtrT, Ptr, Align), S) :-
    emit_kernel_type(Type, TStr),
    emit_kernel_type(PtrT, PtrTStr),
    emit_kernel_value(V, VStr),
    emit_kernel_value(Ptr, PtrStr),
    format(S, "  store ~w ~w, ~w ~w, align ~w~n",
           [TStr, VStr, PtrTStr, PtrStr, Align]).

emit_instr(add_int(R, Type, A, B), S) :-
    emit_kernel_type(Type, TStr),
    emit_kernel_value(A, AStr),
    emit_kernel_value(B, BStr),
    format(S, "  %~w = add ~w ~w, ~w~n", [R, TStr, AStr, BStr]).

emit_instr(mul_int(R, Type, A, B), S) :-
    emit_kernel_type(Type, TStr),
    emit_kernel_value(A, AStr),
    emit_kernel_value(B, BStr),
    format(S, "  %~w = mul ~w ~w, ~w~n", [R, TStr, AStr, BStr]).

emit_instr(and_i1(R, A, B), S) :-
    emit_kernel_value(A, AStr),
    emit_kernel_value(B, BStr),
    format(S, "  %~w = and i1 ~w, ~w~n", [R, AStr, BStr]).

emit_instr(sext(R, FromT, ToT, V), S) :-
    emit_kernel_type(FromT, FStr),
    emit_kernel_type(ToT, TStr),
    emit_kernel_value(V, VStr),
    format(S, "  %~w = sext ~w ~w to ~w~n", [R, FStr, VStr, TStr]).

emit_instr(call_i32(R, FnName, Args), S) :-
    call_args_str(Args, ArgsStr),
    format(S, "  %~w = call i32 @~w(~w)~n", [R, FnName, ArgsStr]).

emit_instr(call_void(FnName, Args), S) :-
    call_args_str(Args, ArgsStr),
    format(S, "  call void @~w(~w)~n", [FnName, ArgsStr]).

call_args_str([], "").
call_args_str([arg(T, V)], Str) :- !,
    emit_kernel_type(T, TStr),
    emit_kernel_value(V, VStr),
    format(atom(Str), "~w ~w", [TStr, VStr]).
call_args_str([arg(T, V)|Rest], Str) :-
    emit_kernel_type(T, TStr),
    emit_kernel_value(V, VStr),
    call_args_str(Rest, RestStr),
    format(atom(Str), "~w ~w, ~w", [TStr, VStr, RestStr]).

emit_instr(br_uncond(Label), S) :-
    format(S, "  br label %~w~n", [Label]).

emit_instr(br_cond(Cond, ThenL, ElseL), S) :-
    emit_kernel_value(Cond, CStr),
    format(S, "  br i1 ~w, label %~w, label %~w~n", [CStr, ThenL, ElseL]).

emit_instr(ret_void, S) :-
    format(S, "  ret void~n", []).

emit_instr(raw_ssa(Atom), S) :-
    format(S, "  ~w~n", [Atom]).

emit_instr(raw_ssa_lines(List), S) :-
    forall(member(Atom, List), format(S, "  ~w~n", [Atom])).

%% raw_line/1: emit atom verbatim (already includes indent/newlines).
%% Used for pre-formatted multi-line body-fragments like matmul_mac_llvm output.
emit_instr(raw_line(Atom), S) :-
    format(S, "~w", [Atom]).

emit_instr(comment_line(Str), S) :-
    format(S, "  ; ~w~n", [Str]).

%% emit_kernel_type(+Type, -Str) — LLVM type-to-string (kernel extension).
%% Extends emit_type/2 with ptr(...) forms needed for kernel scaffolding.
emit_kernel_type(void, "void") :- !.
emit_kernel_type(i1, "i1") :- !.
emit_kernel_type(i8, "i8") :- !.
emit_kernel_type(i32, "i32") :- !.
emit_kernel_type(i64, "i64") :- !.
emit_kernel_type(float, "float") :- !.
emit_kernel_type(double, "double") :- !.
emit_kernel_type(ptr(ElemT), Str) :- !,
    emit_kernel_type(ElemT, EStr),
    format(atom(Str), "~w*", [EStr]).
emit_kernel_type(ptr(ElemT, addrspace(N)), Str) :- !,
    emit_kernel_type(ElemT, EStr),
    format(atom(Str), "~w addrspace(~w)*", [EStr, N]).
emit_kernel_type(T, Str) :-  % fall through to base emit_type/2
    emit_type(T, Str).

%% emit_kernel_argsig_list(+Args, -Str)
%% Args = list of arg(Name, Type, Attrs).
emit_kernel_argsig_list([], "").
emit_kernel_argsig_list([arg(Name, Type, Attrs)], Str) :- !,
    emit_kernel_type(Type, TStr),
    attrs_str(Attrs, AttrStr),
    format(atom(Str), "~w~w %~w", [TStr, AttrStr, Name]).
emit_kernel_argsig_list([arg(Name, Type, Attrs)|Rest], Str) :-
    emit_kernel_type(Type, TStr),
    attrs_str(Attrs, AttrStr),
    emit_kernel_argsig_list(Rest, RestStr),
    format(atom(Str), "~w~w %~w, ~w", [TStr, AttrStr, Name, RestStr]).

attrs_str([], "").
attrs_str([A|As], Str) :-
    attrs_str(As, RestStr),
    format(atom(Str), " ~w~w", [A, RestStr]).

%% emit_kernel_arg_type_list(+Types, -Str) — for `decl(...)` renderer (no names).
emit_kernel_arg_type_list([], "").
emit_kernel_arg_type_list([T], Str) :- !,
    emit_kernel_type(T, Str).
emit_kernel_arg_type_list([T|Rest], Str) :-
    emit_kernel_type(T, TStr),
    emit_kernel_arg_type_list(Rest, RestStr),
    format(atom(Str), "~w, ~w", [TStr, RestStr]).

%% emit_kernel_value(+Val, -Str) — renders an operand.
%% ssa(Name) → %<Name>. label(Name) → %<Name>. imm(N) → decimal. fimm(F) → "F.F".
%% raw_val(Atom) → verbatim (for values already-including-% or complex text).
%% Bare atoms → %<atom> (interpreted as SSA name).
emit_kernel_value(ssa(Name), Str) :- !,
    format(atom(Str), "%~w", [Name]).
emit_kernel_value(label(Name), Str) :- !,
    format(atom(Str), "%~w", [Name]).
emit_kernel_value(imm(N), Str) :- !,
    format(atom(Str), "~w", [N]).
emit_kernel_value(fimm(F), Str) :- !,
    %% Gap-2a fix (2026-09-23 mavhir): emit hex-float form (LLVM canonical).
    %% Decimal fails for values like 0.01 that aren't exactly f32-representable.
    llvm_hex_float(F, Str).
emit_kernel_value(raw_val(Atom), Atom) :- !.
emit_kernel_value(Atom, Str) :-
    atom(Atom), !,
    format(atom(Str), "%~w", [Atom]).
emit_kernel_value(N, Str) :-
    number(N),
    format(atom(Str), "~w", [N]).

%% ═══ LLVM hex-float encoding helper (Gap-2a, 2026-09-23 mavhir) ═══
%%
%% LLVM's IR-textual float constants require exact decimal representation OR
%% hex form. Values like 0.01, 0.1666…, 0.7071… aren't exactly f32-representable,
%% so decimal emission produces "floating-point constant invalid for type".
%%
%% LLVM's hex-float parser requires: the 64-bit hex must represent a DOUBLE
%% value that is EXACTLY-representable as f32 (low mantissa bits zero). Fix
%% (v2): round to f32 FIRST, then emit that f32's double bits. Refuted-by-
%% Mavdil's-llc-run 2026-09-23 (v1 emitted full double bits → 0/5 parse).
%%
%% Duplicate of the same predicate in bpd/kernelgen/emitters/expr_ir.pl —
%% kept independent to preserve the llvm_emit ↔ expr_ir module separation
%% (scaffolding-emitter vs body-lowerer). Both must agree; test-checked.

llvm_hex_float(F, HexStr) :-
    F1 is F * 1.0,
    round_to_f32(F1, F32),
    (   F32 =:= 0.0 -> Bits = 0
    ;   ieee754_double_bits(F32, Bits)),
    format(atom(Padded), '~`0t~16R~16|', [Bits]),
    atom_concat('0x', Padded, HexStr).

round_to_f32(F, F32) :-
    (   F =:= 0.0 -> F32 = 0.0
    ;   f32_encode(F, Bits32),
        f32_decode(Bits32, F32)).

f32_encode(F, Bits) :-
    (   F < 0 -> Sign = 1, Abs is -F
    ;   Sign = 0, Abs is F),
    LogAbs is log(Abs) / log(2),
    E0 is truncate(LogAbs),
    Pow is 2.0 ** E0,
    (   Abs < Pow -> E is E0 - 1 ; E is E0),
    NormPow is 2.0 ** E,
    Norm is Abs / NormPow,
    Frac is Norm - 1.0,
    Mant is round(Frac * (2 ** 23)),
    BiasedE is E + 127,
    (   Mant >= (1 << 23)
    ->  BiasedE2 is BiasedE + 1, Mant2 is 0
    ;   BiasedE2 = BiasedE, Mant2 = Mant),
    Bits is (Sign << 31) \/ (BiasedE2 << 23) \/ Mant2.

f32_decode(Bits, F) :-
    Sign is (Bits >> 31) /\ 1,
    BiasedE is (Bits >> 23) /\ 0xFF,
    Mant is Bits /\ 0x7FFFFF,
    (   BiasedE =:= 0
    ->  (   Mant =:= 0 -> F0 = 0.0
        ;   F0 is (Mant / (2.0 ** 23)) * (2.0 ** -126))
    ;   E is BiasedE - 127,
        Norm is 1.0 + (Mant / (2.0 ** 23)),
        F0 is Norm * (2.0 ** E)),
    (   Sign =:= 1 -> F is -F0 ; F = F0).

ieee754_double_bits(F, Bits) :-
    (   F < 0 -> Sign = 1, Abs is -F
    ;   Sign = 0, Abs is F),
    LogAbs is log(Abs) / log(2),
    E0 is truncate(LogAbs),
    Pow is 2.0 ** E0,
    (   Abs < Pow -> E is E0 - 1 ; E is E0),
    NormPow is 2.0 ** E,
    Norm is Abs / NormPow,       % in [1, 2) for normal doubles
    Frac is Norm - 1.0,
    Mant is round(Frac * (2 ** 52)),
    BiasedE is E + 1023,
    (   Mant >= (1 << 52)
    ->  BiasedE2 is BiasedE + 1, Mant2 is 0
    ;   BiasedE2 = BiasedE, Mant2 = Mant),
    Bits is (Sign << 63) \/ (BiasedE2 << 52) \/ Mant2.
