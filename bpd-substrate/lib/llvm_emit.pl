%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
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
    llvm_extern/3
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
