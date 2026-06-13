%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% kernel_emit_param.pl — PARAMETERIZED kernel emitter (step 4 of the L1 0-ULP mapping pass).
%%
%% The load-bearing artifact: takes an axis-setting (from resolve_axis/5 in ir_param_axes)
%% and emits a c_func/5 AST whose formulation is determined by the axis. One generator,
%% the axis-value selects the opcode sequence -> 0-ULP vs whichever reference pinned it.
%%
%% Proven (2026-05-31): form=divide -> 0-ULP vs F.silu; form=reciprocal_mul -> 0-ULP vs
%% Stanford Swish(x*sigmoid). This module makes that emission declarative + AST-based,
%% fitting kernel_templates.pl's c_func convention (emitted to C/CUDA by c_ast.pl).
%%
%% Pipeline: stanford_referee finds DIVERGENT -> resolve_axis(Op,Axis,Ref,Val) -> here:
%%   generate_kernel_activation(Op, [form(Val)], -KernelAST) -> emit_program -> compile -> 0-ULP.
%%
%% Tracked source (bpd/lib/).

:- module(kernel_emit_param, [
     generate_kernel_activation/3,   % generate_kernel_activation(+Op, +AxisSettings, -KernelAST)
     activation_body/4,              % activation_body(+Op, +Form, +VarIn, -BodyStmts)
     emit_activation_for_reference/4, % emit_activation_for_reference(+Op, +Reference, -KernelAST, -Note)
     generate_kernel_rmsnorm/2,
     emit_rmsnorm_for_reference/3
   ]).

:- use_module(ir_param_axes).        % resolve_axis/5, reference_pins/3, op_family/2

%% ── activation kernel skeleton: void NAME(const float* X, float* Y, int N) ──
%% Body computes Y[i] = f(X[i]) per the form axis. Elementwise loop.
generate_kernel_activation(Op, AxisSettings, KernelAST) :-
    ( member(form(Form), AxisSettings) -> true ; Form = divide ),  % default divide
    activation_name(Op, Name),
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'N')
    ],
    activation_body(Op, Form, c_index(c_var('X'), c_var(i)), Compute),
    Body = [
        c_comment('=== parameterized activation kernel ==='),
        c_for(
            c_decl_init(c_type(int), i, c_int(0)),
            c_binop('<', c_var(i), c_var('N')),
            c_unop('++', c_var(i)),
            [ c_decl_init(c_type(float), xi, c_index(c_var('X'), c_var(i)))
            | Compute ]
        )
    ],
    KernelAST = c_func([], c_type(void), Name, Params, Body).

activation_name(silu, bpd_silu_param).
activation_name(swish, bpd_silu_param).   %% same generator, form axis distinguishes
activation_name(gelu, bpd_gelu_param).

%% ── activation_body: the FORM AXIS selects the opcode sequence ──
%% silu, form=divide:        Y[i] = xi / (1.0f + expf(-xi))         [1 rounding]
%% silu, form=reciprocal_mul:Y[i] = xi * (1.0f / (1.0f + expf(-xi)))[2 roundings = Stanford Swish]
%% NOTE: c_ast emit_program does NOT auto-parenthesize binops — explicit c_paren/1 is required
%% wherever math precedence demands grouping. Getting this wrong silently breaks 0-ULP.
activation_body(silu, divide, _XiExpr, [
    c_assign(c_index(c_var('Y'), c_var(i)),
      c_binop('/', c_var(xi),
        c_paren(c_binop('+', c_float_f(1.0),
          c_call(expf, [c_paren(c_unop('-', c_var(xi)))])))))
  ]).
activation_body(silu, reciprocal_mul, _XiExpr, [
    c_decl_init(c_type(float), s,
      c_binop('/', c_float_f(1.0),
        c_paren(c_binop('+', c_float_f(1.0),
          c_call(expf, [c_paren(c_unop('-', c_var(xi)))]))))),
    c_assign(c_index(c_var('Y'), c_var(i)),
      c_binop('*', c_var(xi), c_var(s)))
  ]).
%% gelu, form=erf:   Y[i] = 0.5f*xi*(1.0f+erff(xi*0.70710678f))   [exact GELU]
activation_body(gelu, erf, _XiExpr, [
    c_assign(c_index(c_var('Y'), c_var(i)),
      c_binop('*',
        c_paren(c_binop('*', c_float_f(0.5), c_var(xi))),
        c_paren(c_binop('+', c_float_f(1.0),
          c_call(erff, [c_paren(c_binop('*', c_var(xi), c_float_f(0.70710678)))])))))
  ]).

%% ── emit FOR a reference: resolve the axis the reference pins, emit that form ──
%% This is the climb's inner loop: given a Stanford reference, emit the 0-ULP-matching kernel.
emit_activation_for_reference(Op, Reference, KernelAST, Note) :-
    op_family(Op, Axes),
    member(form, Axes),
    resolve_axis(Op, form, Reference, FormVal, Note),
    generate_kernel_activation(Op, [form(FormVal)], KernelAST).

%% ── reduction/norm emitter parameterized by the acc_type axis ──
%% RMSNorm over dim=1 (features): rms = sqrt(mean(x^2, dim=1)+eps); y = x/rms.
%% acc_type=f64 -> accumulate sum-of-squares in DOUBLE (matches pytorch torch.mean acc_type),
%% acc_type=f32 -> accumulate in float (matches ggml). The axis selects the accumulator type.
%% Signature: void NAME(const float* X, float* Y, int N, int C, int HW, float eps)
generate_kernel_rmsnorm(AxisSettings, KernelAST) :-
    ( member(acc_type(Acc), AxisSettings) -> true ; Acc = f64 ),  %% default f64 (precision)
    acc_ctype(Acc, AccT),
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'N'),
        param(c_type(int), 'C'),
        param(c_type(int), 'HW'),
        param(c_type(float), eps)
    ],
    acc_zero(Acc, Zero),
    acc_square(Acc, c_var(v), SqExpr),
    acc_count(Acc, CountExpr),
    Body = [
      c_comment('rmsnorm over dim=1 features C, acc_type parameterized'),
      c_for(c_decl_init(c_type(int), n, c_int(0)), c_binop('<', c_var(n), c_var('N')), c_unop('++', c_var(n)), [
        c_for(c_decl_init(c_type(int), hw, c_int(0)), c_binop('<', c_var(hw), c_var('HW')), c_unop('++', c_var(hw)), [
          c_decl_init(c_type(AccT), acc, Zero),
          c_for(c_decl_init(c_type(int), c, c_int(0)), c_binop('<', c_var(c), c_var('C')), c_unop('++', c_var(c)), [
            c_decl_init(c_type(int), idx, c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(n), c_var('C')), c_var(c))), c_var('HW')), c_var(hw))),
            c_decl_init(c_type(float), v, c_index(c_var('X'), c_var(idx))),
            c_assign(c_var(acc), c_binop('+', c_var(acc), SqExpr))
          ]),
          c_decl_init(c_type(float), ms, c_cast(c_type(float), c_paren(c_binop('/', c_var(acc), CountExpr)))),
          c_decl_init(c_type(float), r, c_call(sqrtf, [c_paren(c_binop('+', c_var(ms), c_var(eps)))])),
          c_for(c_decl_init(c_type(int), c2, c_int(0)), c_binop('<', c_var(c2), c_var('C')), c_unop('++', c_var(c2)), [
            c_decl_init(c_type(int), idx2, c_binop('+', c_binop('*', c_paren(c_binop('+', c_binop('*', c_var(n), c_var('C')), c_var(c2))), c_var('HW')), c_var(hw))),
            c_assign(c_index(c_var('Y'), c_var(idx2)), c_binop('/', c_index(c_var('X'), c_var(idx2)), c_var(r)))
          ])
        ])
      ])
    ],
    KernelAST = c_func([], c_type(void), bpd_rmsnorm_param, Params, Body).

acc_ctype(f32, float).
acc_ctype(f64, double).
acc_zero(f32, c_float_f(0.0)).
acc_zero(f64, c_float(0.0)).
%% square term: f64 path promotes v to double before squaring (matches pytorch f64-acc mean(x^2))
acc_square(f32, V, c_binop('*', V, V)).
acc_square(f64, V, c_binop('*', c_cast(c_type(double), V), c_cast(c_type(double), V))).
acc_count(f32, c_cast(c_type(float), c_var('C'))).
acc_count(f64, c_cast(c_type(double), c_var('C'))).

%% emit rmsnorm for a reference (resolve the acc_type the reference pins)
emit_rmsnorm_for_reference(Reference, KernelAST, Note) :-
    resolve_axis(rms_norm, acc_type, Reference, AccVal, Note),
    generate_kernel_rmsnorm([acc_type(AccVal)], KernelAST).
