%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% extract_kernel_signatures.pl — Tier 2 subtask 2b.
%%
%% For each validation case in test_kernelbench_l1_cuda.pl, call the kernel
%% generator and extract the c_func term's parameter list. Emit
%% kernel_signature(CaseName, KernelName, ParamSpec) facts that the generic
%% harness can consume.

:- use_module('/tmp/bpd-substrate/lib/c_ast').
:- use_module('/tmp/bpd-substrate/lib/kernel_templates_blas',
              except([kernel_available_fixes/2, fix_description/2])).
:- use_module('/tmp/bpd-substrate/lib/kernel_templates').
:- use_module('/tmp/bpd-substrate/lib/kernel_templates_llama',
              except([kernel_available_fixes/2, fix_description/2])).

%% Re-declare the validation cases here. Source: bpd/tests/test_kernelbench_l1_cuda.pl
%% These are the 28 cases the substrate emits.
validation_case('conv_2d_forward',           generate_kernel_im2col(ggml_conv_2d, 2, forward, _)).
validation_case('conv_1d_forward',           generate_kernel_im2col(ggml_conv_1d, 1, forward, _)).
validation_case('conv_3d_forward',           generate_kernel_im2col(ggml_conv_3d, 3, forward, _)).
validation_case('conv_transpose_2d',         generate_kernel_im2col(ggml_conv_transpose_2d, 2, transpose, _)).

validation_case('reduce_sum_rows',  generate_kernel_reduction(ggml_sum_rows, 2, axis_inner, _)).
validation_case('reduce_mean',      generate_kernel_reduction(ggml_mean,     2, axis_inner, _)).
validation_case('reduce_max',       generate_kernel_reduction(ggml_max,      2, axis_inner, _)).
validation_case('reduce_min',       generate_kernel_reduction(ggml_min,      2, axis_inner, _)).
validation_case('reduce_argmax',    generate_kernel_reduction(ggml_argmax,   2, axis_inner, _)).
validation_case('reduce_argmin',    generate_kernel_reduction(ggml_argmin,   2, axis_inner, _)).
validation_case('cumsum',           generate_kernel_reduction(ggml_cumsum,   2, axis_inner, _)).
validation_case('cumprod',          generate_kernel_reduction(ggml_cumprod,  2, axis_inner, _)).

validation_case('norm_layer_plain', generate_kernel_norm(ggml_norm, 2, false, _)).
validation_case('norm_layer_affine', generate_kernel_norm(ggml_norm, 2, true, _)).
validation_case('norm_rms_plain',   generate_kernel_norm(ggml_rms_norm, 2, false, _)).
validation_case('norm_rms_affine',  generate_kernel_norm(ggml_rms_norm, 2, true, _)).
validation_case('norm_l2_plain',    generate_kernel_norm(ggml_l2_norm, 2, false, _)).
validation_case('norm_l2_affine',   generate_kernel_norm(ggml_l2_norm, 2, true, _)).
validation_case('norm_group_plain', generate_kernel_norm(ggml_group_norm, 2, false, _)).

validation_case('loss_mse_mean',         generate_kernel_loss(ggml_mse_loss,           mean, [], _)).
validation_case('loss_mse_sum',          generate_kernel_loss(ggml_mse_loss,           sum,  [], _)).
validation_case('loss_cross_entropy',    generate_kernel_loss(ggml_cross_entropy_loss, mean, [], _)).
validation_case('loss_huber',            generate_kernel_loss(ggml_huber_loss,         mean, [], _)).
validation_case('loss_kl_div',           generate_kernel_loss(ggml_kl_div_loss,        sum,  [], _)).
validation_case('loss_hinge',            generate_kernel_loss(ggml_hinge_loss,         mean, [], _)).
validation_case('loss_triplet_margin',   generate_kernel_loss(ggml_triplet_margin_loss, mean, [], _)).

validation_case('pool_2d_max',  generate_kernel_pool(ggml_pool_2d, 2, max, [], _)).
validation_case('pool_2d_avg',  generate_kernel_pool(ggml_pool_2d, 2, avg, [], _)).

%% extract_signature(+CaseName, -KernelName, -Params) — invoke the generator,
%% pluck the c_func term, return (kernel_name, parameter list).
extract_signature(CaseName, KernelName, Params) :-
    validation_case(CaseName, GeneratorCall),
    %% The generator's last argument is the output kernel term — bind it
    GeneratorCall =.. [Generator | Args],
    %% Replace the last (variable) arg with our local Kernel
    append(FixedArgs, [_LastVar], Args),
    append(FixedArgs, [Kernel], NewArgs),
    NewGoal =.. [Generator | NewArgs],
    call(NewGoal),
    %% Kernel = c_func(Attrs, RetType, Name, Params, Body)
    Kernel = c_func(_Attrs, _RetType, KernelName, Params, _Body).

%% Pretty-print a parameter list as a Python-friendly type spec
print_param(param(c_type(const_ptr(c_type(T))), Name)) :-
    format("    (~q, c_const_ptr(~q))", [Name, T]).
print_param(param(c_type(const_restrict_ptr(c_type(T))), Name)) :-
    format("    (~q, c_const_ptr(~q))", [Name, T]).
print_param(param(c_type(ptr(c_type(T))), Name)) :-
    format("    (~q, c_ptr(~q))", [Name, T]).
print_param(param(c_type(restrict_ptr(c_type(T))), Name)) :-
    format("    (~q, c_ptr(~q))", [Name, T]).
print_param(param(c_type(T), Name)) :-
    atom(T),
    format("    (~q, c_scalar(~q))", [Name, T]).
print_param(P) :-
    format("    %% UNHANDLED PARAM SHAPE: ~q~n", [P]).

emit_signature(CaseName) :-
    catch(
        ( extract_signature(CaseName, KernelName, Params),
          format("kernel_signature('~w', '~w', [~n", [CaseName, KernelName]),
          forall(member(P, Params),
                 (print_param(P), format(",~n"))),
          format("    ]).~n~n")
        ),
        Err,
        format("%% ERROR extracting ~w: ~q~n~n", [CaseName, Err])
    ).

main :-
    format("%% Auto-generated kernel signature catalog~n"),
    format("%% Subtask 2b — Tier 2 plan 8d65ba1c-5782-47f4-82a3-fa017b727e96~n"),
    format("%% Reflectively extracted from substrate's emit predicates.~n~n"),
    forall(validation_case(Name, _), emit_signature(Name)),
    halt(0).

:- initialization(main).
