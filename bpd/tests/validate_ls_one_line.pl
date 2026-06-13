%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% validate_ls_one_line.pl — Validates ls-one-line.pl format string safety & conformance.
%%
%% Heath's directive: automatic validation that format strings returned from
%% ls-one-line.pl rulesets are 'safe', 'valid', and have matching signatures.
%%
%% Usage:
%%   swipl -g "validate_file('/path/to/ls-one-line.pl')" validate_ls_one_line.pl
%%   swipl -g "validate_file('/path/to/ls-one-line.pl', quiet)" validate_ls_one_line.pl
%%
%% Author: medayek, 2026-05-31

:- module(validate_ls_one_line, [
    validate_file/1,
    validate_file/2,
    validate_file_gate/2,
    validate_module/1
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% ─── Public entry points ───

%% validate_file(+Path) — load and validate, print report, succeed on PASS.
validate_file(Path) :- validate_file(Path, verbose).

%% validate_file_gate(+Path, -Result) — programmatic API for consumer integration.
%%   Result = gate(Status, Passes, Warns, Fails)
%%   Status = pass | warn | fail
%%   Passes/Warns/Fails = list of result(Category, Status, Message)
validate_file_gate(Path, gate(Status, Passes, Warns, Fails)) :-
    ( exists_file(Path) -> true ; Status = fail, Passes = [], Warns = [], Fails = [result(file, fail, "file not found")], ! ),
    read_module_name(Path, Module),
    catch(
        load_files(Path, [if(true)]),
        _Error,
        ( Status = fail, Passes = [], Warns = [], Fails = [result(load, fail, "load error")], ! )
    ),
    validate_module_impl(Module, quiet, Results),
    include([R]>>(R = result(_, fail, _)), Results, Fails),
    include([R]>>(R = result(_, warn, _)), Results, Warns),
    include([R]>>(R = result(_, pass, _)), Results, Passes),
    ( Fails \= [] -> Status = fail
    ; Warns \= [] -> Status = warn
    ; Status = pass
    ).

validate_file(Path, Verbosity_) :-
    ( exists_file(Path) ->
        true
    ;
        format("FAIL: file not found: ~w~n", [Path]),
        fail
    ),
    % Read the module name from the file's :- module(...) declaration
    read_module_name(Path, Module),
    catch(
        load_files(Path, [if(true)]),
        Error,
        ( format("FAIL: load error: ~w~n", [Error]), fail )
    ),
    validate_module_impl(Module, Verbosity_, Results),
    report_results(Results, Verbosity_).

%% validate_module(+Module) — validate an already-loaded module.
validate_module(Module) :-
    validate_module_impl(Module, verbose, Results),
    report_results(Results, verbose).

%% ─── Validation checks ───

validate_module_impl(Module, Verbosity, Results) :-
    check_required_exports(Module, R1),
    check_namespace(Module, R2),
    check_templates(Module, R3),
    check_template_vars_covered(Module, R4),
    check_no_dead_fields(Module, R5),
    check_extractor_safety(Module, R6),
    append([R1, R2, R3, R4, R5, R6], Results).

%% 1. Required predicates exported
check_required_exports(Module, Results) :-
    Required = [namespace/2, one_line/2, field/3],
    findall(
        result(required_export, Status, Msg),
        ( member(Pred/Arity, Required),
          functor(Term, Pred, Arity),
          ( predicate_property(Module:Term, defined) ->
              Status = pass,
              format(atom(Msg), "~w/~w exported", [Pred, Arity])
          ;
              Status = fail,
              format(atom(Msg), "~w/~w MISSING — required predicate not defined", [Pred, Arity])
          )
        ),
        Results
    ).

%% 2. namespace/2 well-formed
check_namespace(Module, Results) :-
    ( Module:namespace(Tag, Desc) ->
        ( atom(Tag), (atom(Desc) ; string(Desc)) ->
            Results = [result(namespace, pass, "namespace/2 well-formed")]
        ;
            Results = [result(namespace, fail, "namespace/2 args must be atom, atom|string")]
        )
    ;
        Results = [result(namespace, fail, "no namespace/2 clause found")]
    ).

%% 3. one_line/2 templates well-formed (pattern + format string)
check_templates(Module, Results) :-
    findall(
        result(template, Status, Msg),
        ( Module:one_line(Pattern, Format),
          ( (string(Pattern) ; atom(Pattern)),
            (string(Format) ; atom(Format)) ->
              % Check pattern has at least one ~VAR
              atom_string(Pattern, PS),
              ( extract_vars(PS, PVars), PVars \= [] ->
                  Status = pass,
                  format(atom(Msg), "template ~w — ~w pattern vars", [Pattern, PVars])
              ;
                  Status = warn,
                  format(atom(Msg), "template ~w — no ~VAR in pattern (static match only)", [Pattern])
              )
          ;
              Status = fail,
              format(atom(Msg), "template args must be strings: ~w", [Pattern])
          )
        ),
        Results
    ),
    ( Results = [] ->
        Results = [result(template, fail, "no one_line/2 clauses found")]
    ;
        true
    ).

%% 4. Every ~VAR in templates has a matching field/3 declaration
check_template_vars_covered(Module, Results) :-
    findall(Var, (
        Module:one_line(Pattern, Format),
        atom_string(Pattern, PS), atom_string(Format, FS),
        extract_vars(PS, PVars),
        extract_vars(FS, FVars),
        append(PVars, FVars, AllV),
        member(Var, AllV)
    ), AllVarsRaw),
    sort(AllVarsRaw, AllVars),
    % Get all declared field names
    findall(FName, Module:field(FName, _, _), DeclaredRaw),
    sort(DeclaredRaw, Declared),
    % Check coverage
    findall(
        result(var_coverage, Status, Msg),
        ( member(Var, AllVars),
          normalize_var(Var, VarNorm),
          ( member(VarNorm, Declared) ->
              Status = pass,
              format(atom(Msg), "~w → ~w has field/3 declaration", [Var, VarNorm])
          ;
              Status = fail,
              format(atom(Msg), "~w (normalized: ~w) — DANGLING: used in template but no field/3 declaration", [Var, VarNorm])
          )
        ),
        Results
    ).

%% 5. No dead fields (declared but never used in any template)
check_no_dead_fields(Module, Results) :-
    % Get all vars used in templates
    findall(Var, (
        Module:one_line(Pattern, Format),
        atom_string(Pattern, PS), atom_string(Format, FS),
        extract_vars(PS, PVars),
        extract_vars(FS, FVars),
        append(PVars, FVars, AllV),
        member(Var, AllV)
    ), UsedRaw),
    sort(UsedRaw, Used),
    % Lowercase version for case-insensitive match
    maplist([V,VL]>>normalize_var(V, VL), Used, UsedLower),
    % Get all declared field names
    findall(FName, Module:field(FName, _, _), DeclaredRaw),
    sort(DeclaredRaw, Declared),
    findall(
        result(dead_field, Status, Msg),
        ( member(F, Declared),
          downcase_atom(F, FL),
          ( member(FL, UsedLower) ->
              Status = pass,
              format(atom(Msg), "field ~w is used", [F])
          ;
              Status = warn,
              format(atom(Msg), "field ~w DECLARED but never referenced in any template", [F])
          )
        ),
        Results
    ).

%% 6. Extractor safety checks
check_extractor_safety(Module, Results) :-
    findall(
        result(extractor_safety, Status, Msg),
        ( Module:field(Name, Source, Extractor),
          check_one_extractor(Name, Source, Extractor, Status, Msg)
        ),
        Results
    ).

check_one_extractor(Name, Source, Extractor, Status, Msg) :-
    % Source must be a known type
    known_source(Source), !,
    % Extractor must be well-formed for its source type
    ( valid_extractor(Source, Extractor) ->
        Status = pass,
        format(atom(Msg), "field ~w: ~w/~w — safe", [Name, Source, Extractor])
    ;
        Status = fail,
        format(atom(Msg), "field ~w: extractor ~w not valid for source ~w", [Name, Extractor, Source])
    ).
check_one_extractor(Name, Source, _Extractor, warn, Msg) :-
    format(atom(Msg), "field ~w: unknown source type ~w (known: filename, json, req_line, literal)", [Name, Source]).

known_source(filename).
known_source(json).
known_source(req_line).
known_source(literal).
known_source(resp_line).

valid_extractor(filename, regex(Pattern)) :- string(Pattern) ; atom(Pattern).
valid_extractor(json, path(Path)) :- string(Path) ; atom(Path).
valid_extractor(req_line, word(N)) :- integer(N), N > 0.
valid_extractor(resp_line, word(N)) :- integer(N), N > 0.
valid_extractor(literal, Value) :- string(Value) ; atom(Value) ; number(Value).

%% ─── Module discovery ───

%% read_module_name(+Path, -Module) — extract module name from :- module(Name, ...) directive
read_module_name(Path, Module) :-
    setup_call_cleanup(
        open(Path, read, Stream),
        read_module_from_stream(Stream, Module),
        close(Stream)
    ).

read_module_from_stream(Stream, Module) :-
    read_term(Stream, Term, []),
    ( Term = (:- module(Module, _)) ->
        true
    ; Term = end_of_file ->
        fail
    ;
        read_module_from_stream(Stream, Module)
    ).

%% ─── Helpers ───

%% extract_vars(+String, -Vars) — find all ~VAR references in a string
extract_vars(String, Vars) :-
    atom_string(String, S),
    string_codes(S, Codes),
    extract_vars_codes(Codes, Vars).

extract_vars_codes([], []).
extract_vars_codes([0'~|Rest], [Var|Vars]) :- !,
    take_var_chars(Rest, VarCodes, Remaining),
    ( VarCodes \= [] ->
        atom_codes(Var, VarCodes),
        extract_vars_codes(Remaining, Vars)
    ;
        extract_vars_codes(Rest, [Var|Vars])
    ).
extract_vars_codes([_|Rest], Vars) :-
    extract_vars_codes(Rest, Vars).

take_var_chars([], [], []).
take_var_chars([C|Rest], [C|VCs], Rem) :-
    var_char(C), !,
    take_var_chars(Rest, VCs, Rem).
take_var_chars(Cs, [], Cs).

var_char(C) :- C >= 0'A, C =< 0'Z, !.
var_char(C) :- C >= 0'a, C =< 0'z, !.
var_char(C) :- C >= 0'0, C =< 0'9, !.
var_char(0'_).

%% normalize_var(+Raw, -Normalized) — strip trailing underscores, lowercase
normalize_var(Raw, Normalized) :-
    atom_string(Raw, S),
    string_codes(S, Codes),
    reverse(Codes, Rev),
    drop_underscores(Rev, Trimmed),
    reverse(Trimmed, Clean),
    atom_codes(CleanAtom, Clean),
    downcase_atom(CleanAtom, Normalized).

drop_underscores([0'_|Rest], Result) :- !, drop_underscores(Rest, Result).
drop_underscores(X, X).

%% ─── Reporting ───

report_results(Results, Verbosity) :-
    include([R]>>(R = result(_, fail, _)), Results, Fails),
    include([R]>>(R = result(_, warn, _)), Results, Warns),
    include([R]>>(R = result(_, pass, _)), Results, Passes),
    length(Fails, NF), length(Warns, NW), length(Passes, NP),
    Total is NF + NW + NP,
    nl,
    format("═══ ls-one-line.pl Validation ═══~n"),
    format("  ~w checks: ~w pass, ~w warn, ~w FAIL~n~n", [Total, NP, NW, NF]),
    ( Verbosity = verbose ->
        ( Fails \= [] ->
            format("FAILURES:~n"),
            forall(member(result(Cat, fail, Msg), Fails),
                format("  ✗ [~w] ~w~n", [Cat, Msg]))
        ; true ),
        ( Warns \= [] ->
            format("WARNINGS:~n"),
            forall(member(result(Cat, warn, Msg), Warns),
                format("  ⚠ [~w] ~w~n", [Cat, Msg]))
        ; true ),
        ( Fails = [] ->
            format("GATE: PASS ✓~n")
        ;
            format("GATE: FAIL ✗~n")
        )
    ;
        true
    ),
    ( Fails = [] -> true ; fail ).

%% ─── CLI entry ───

:- initialization((
    current_prolog_flag(argv, Args),
    ( Args = [Path|_] ->
        ( validate_file(Path) ->
            halt(0)
        ;
            halt(1)
        )
    ;
        true  % loaded as module, no CLI
    )
), main).
