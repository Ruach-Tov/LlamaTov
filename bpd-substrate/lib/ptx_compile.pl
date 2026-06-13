%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ptx_compile.pl — Compile LLVM IR to PTX for GPU execution from Prolog
%%
%% The missing link: Prolog emits .ll → this compiles to .ptx → cuda_launch loads it.
%%
%% Usage:
%%   ?- ll_to_ptx('kernel.ll', 'kernel.ptx').
%%   ?- emit_and_compile(softmax, 'softmax.ptx').

:- module(ptx_compile, [
    ll_to_ptx/2,         % ll_to_ptx(+LLFile, +PTXFile)
    ll_to_ptx/3,         % ll_to_ptx(+LLFile, +PTXFile, +Options)
    emit_and_compile/2,  % emit_and_compile(+KernelName, +PTXFile)
    ir_string_to_ptx/2   % ir_string_to_ptx(+IRString, +PTXFile)
]).

%% LLC path on NixOS enclave
llc_path('/nix/store/a8hbpr6c8rhdcvgr19r8gnqifnnfb9q3-llvm-19.1.7/bin/llc').

%% Default compilation options for Tesla P4
default_options([
    march(nvptx64),
    mcpu(sm_61),
    'O'(2)
]).

%% ll_to_ptx(+LLFile, +PTXFile)
%% Compile LLVM IR to PTX using llc
ll_to_ptx(LLFile, PTXFile) :-
    default_options(Opts),
    ll_to_ptx(LLFile, PTXFile, Opts).

ll_to_ptx(LLFile, PTXFile, Options) :-
    llc_path(LLC),
    build_llc_args(Options, OptArgs),
    atomic_list_concat([LLC | OptArgs], ' ', BaseCmd),
    format(atom(Cmd), '~w -o ~w ~w 2>&1', [BaseCmd, PTXFile, LLFile]),
    shell(Cmd, ExitCode),
    (   ExitCode =:= 0
    ->  true
    ;   format(user_error, "llc failed (exit ~w): ~w~n", [ExitCode, Cmd]),
        fail
    ).

build_llc_args([], []).
build_llc_args([march(V)|Rest], [Arg|Args]) :-
    format(atom(Arg), '-march=~w', [V]),
    build_llc_args(Rest, Args).
build_llc_args([mcpu(V)|Rest], [Arg|Args]) :-
    format(atom(Arg), '-mcpu=~w', [V]),
    build_llc_args(Rest, Args).
build_llc_args(['O'(V)|Rest], [Arg|Args]) :-
    format(atom(Arg), '-O~w', [V]),
    build_llc_args(Rest, Args).
build_llc_args([_|Rest], Args) :-
    build_llc_args(Rest, Args).

%% ir_string_to_ptx(+IRString, +PTXFile)
%% Write IR to temp file, compile to PTX
ir_string_to_ptx(IRString, PTXFile) :-
    tmp_file_stream(text, TmpLL, Stream),
    write(Stream, IRString),
    close(Stream),
    ll_to_ptx(TmpLL, PTXFile),
    delete_file(TmpLL).

%% emit_and_compile(+KernelName, +PTXFile)
%% Full chain: emit LLVM IR from BPD facts, compile to PTX
%% Requires the Prolog LLVM emitter (bpd_llvm_elem.so) to be loaded
emit_and_compile(KernelName, PTXFile) :-
    tmp_file_stream(text, TmpLL, Stream),
    close(Stream),
    %% Call the Prolog emitter to generate .ll
    format(atom(EmitGoal), 'llvm_emit_to_file(~w, ~w)', [KernelName, TmpLL]),
    (   catch(term_to_atom(Goal, EmitGoal), _, fail),
        call(Goal)
    ->  ll_to_ptx(TmpLL, PTXFile),
        delete_file(TmpLL)
    ;   format(user_error, "emit_and_compile: emitter not available for ~w~n", [KernelName]),
        delete_file(TmpLL),
        fail
    ).
