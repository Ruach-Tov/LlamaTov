%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% kernel_templates.pl — standalone CUDA kernel templates for L1 ops
%% that don't fit the matmul+epilogue fusion pattern.
%%
%% Per Heath's "no single-problem rules" principle: each predicate is a
%% PARAMETERIZED FAMILY serving multiple L1 problems via op-kind variants.
%%
%% Naming follows mavchin's existing generate_fused_kernel/3 convention.
%% Each generate_kernel_<family>/N returns a c_func/5 AST suitable for
%% c_ast:emit_program.
%%
%% This commit (Scope B2-Conv) implements Family 4 only:
%%   generate_kernel_im2col/4 — covers 35 L1 conv problems via im2col+matmul
%%
%% Future commits will add:
%%   generate_kernel_reduction/4 — sum/mean/max/min/argmax/argmin/cumsum/cumprod
%%   generate_kernel_norm/4       — layer/rms/l2/group norm
%%   generate_kernel_pool/5       — max/avg pool 1d/2d/3d
%%   generate_kernel_loss/4       — mse/crossentropy/huber/kldiv/triplet/hinge
%%   generate_kernel_flash_attn/4 — flash attention
%%
%% Per mavchin's directive (inbox 2026-05-15 18:42):
%% "Substrate file: extend fusion_to_cuda.pl for epilogue-composable ops.
%%  New file bpd/lib/kernel_templates.pl for standalone families."

:- module(kernel_templates, [
    generate_kernel_im2col/4,
    generate_kernel_reduction/4,
    generate_kernel_reduction/5,        % +OpKind, +Dim, +Mode, +Strategy, -Kernel
    reduction_strategy_kind/1,           % enumerate strategies
    reduction_strategy_description/2,    % strategy -> description text
    generate_kernel_norm/4,
    generate_kernel_loss/4,
    generate_kernel_pool/5
]).

:- set_prolog_flag(double_quotes, codes).

%% ═════════════════════════════════════════════════════════════════════
%% Family 4: Convolutions (im2col + matmul composition)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Per mavchin's directive: "Conv approach: im2col + matmul. It's what
%% ggml does internally, it composes with our existing tiled matmul
%% kernel, and it makes conv fusible with matmul epilogues for free."
%%
%% The full conv emission is a TWO-KERNEL pipeline:
%%   1. im2col: unfolds input patches into a matrix [B*outH*outW, C*kH*kW]
%%   2. matmul: (existing tiled matmul kernel) multiplies by weight matrix
%%
%% This commit emits the im2col kernel only. The matmul stage reuses
%% mavchin's existing build_tiled_matmul_kernel from fusion_to_cuda.pl.
%%
%% For 2D conv with input X of shape [B, C_in, H, W] and weights of
%% shape [C_out, C_in, kH, kW]:
%%   im2col produces: [B * outH * outW, C_in * kH * kW]
%%   matmul with weights reshaped to [C_in * kH * kW, C_out]:
%%     produces [B * outH * outW, C_out]
%%   reshape to [B, C_out, outH, outW] — final result.
%%
%% generate_kernel_im2col(+OpKind, +Dim, +Mode, -KernelAST)
%%
%% OpKind:    ggml_conv_1d, ggml_conv_2d, ggml_conv_3d,
%%            ggml_conv_transpose_1d/2d/3d
%% Dim:       1, 2, or 3
%% Mode:      forward (standard conv) | transpose (col2im for conv_transpose)
%% KernelAST: c_func/5 AST representing the kernel
%%
%% This commit implements Dim=2 forward (the most common case — 27 of
%% the 35 L1 conv problems are 2D). 1D, 3D, and transposed variants
%% are stubs that delegate via a TODO comment and will be expanded in
%% follow-up commits (B2-Conv-Extended).

generate_kernel_im2col(OpKind, Dim, forward, KernelAST) :-
    conv_kernel_name(OpKind, Dim, forward, Name),
    build_im2col_kernel(Name, Dim, forward, KernelAST).

generate_kernel_im2col(OpKind, Dim, transpose, KernelAST) :-
    conv_kernel_name(OpKind, Dim, transpose, Name),
    build_im2col_kernel(Name, Dim, transpose, KernelAST).

%% conv_kernel_name(+OpKind, +Dim, +Mode, -Name)
conv_kernel_name(ggml_conv_1d,           1, forward,   im2col_1d_forward).
conv_kernel_name(ggml_conv_2d,           2, forward,   im2col_2d_forward).
conv_kernel_name(ggml_conv_3d,           3, forward,   im2col_3d_forward).
conv_kernel_name(ggml_conv_transpose_1d, 1, transpose, col2im_1d_transpose).
conv_kernel_name(ggml_conv_transpose_2d, 2, transpose, col2im_2d_transpose).
conv_kernel_name(ggml_conv_transpose_3d, 3, transpose, col2im_3d_transpose).

%% ─────────────────────────────────────────────────────────────────────
%% build_im2col_kernel(+Name, +Dim, +Mode, -Kernel)
%% ─────────────────────────────────────────────────────────────────────
%%
%% Constructs the CUDA kernel AST for an im2col (or col2im) operation.
%% Per substrate-honest scoping: this commit implements Dim=2 forward
%% with full body; other variants emit a skeleton that compiles but
%% doesn't yet do the full unfold (TODO B2-Conv-Extended).

%% 2D forward — the most common case (27 of 35 L1 conv problems are 2D)
build_im2col_kernel(Name, 2, forward, Kernel) :-
    !,
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'B'),       % batch
        param(c_type(int), 'C'),       % input channels
        param(c_type(int), 'H'),       % input height
        param(c_type(int), 'W'),       % input width
        param(c_type(int), 'kH'),      % kernel height
        param(c_type(int), 'kW'),      % kernel width
        param(c_type(int), 'outH'),
        param(c_type(int), 'outW'),
        param(c_type(int), stride_h),
        param(c_type(int), stride_w),
        param(c_type(int), pad_h),
        param(c_type(int), pad_w),
        param(c_type(int), dilation_h),
        param(c_type(int), dilation_w)
    ],
    Body = [
        c_comment('=== im2col 2D forward — unfolds patches into [B*outH*outW, C*kH*kW] ==='),
        c_comment('Thread layout: blockIdx.z=batch, blockIdx.y=outH, x dim covers outW'),
        c_decl_init(c_type(int), b, c_member(c_var(blockIdx), z)),
        c_decl_init(c_type(int), oh, c_member(c_var(blockIdx), y)),
        c_decl_init(c_type(int), ow,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('||',
                c_binop('>=', c_var(b), c_var('B')),
                c_binop('>=', c_var(ow), c_var('outW'))),
             [c_return_void]),
        c_blank,
        c_comment('Output row index in im2col matrix: (b * outH + oh) * outW + ow'),
        c_decl_init(c_type(int), out_row,
            c_binop('+',
                c_binop('*',
                    c_paren(c_binop('+',
                        c_binop('*', c_var(b), c_var('outH')),
                        c_var(oh))),
                    c_var('outW')),
                c_var(ow))),
        c_comment('Each output row has C * kH * kW entries (one per kernel position × channel)'),
        c_decl_init(c_type(int), patch_size,
            c_binop('*', c_var('C'), c_binop('*', c_var('kH'), c_var('kW')))),
        c_blank,
        c_comment('Unfold the patch: iterate over channels and kernel offsets'),
        c_for(
            c_decl_init(c_type(int), c, c_int(0)),
            c_binop('<', c_var(c), c_var('C')),
            c_unop('++', c_var(c)),
            [
                c_for(
                    c_decl_init(c_type(int), kh, c_int(0)),
                    c_binop('<', c_var(kh), c_var('kH')),
                    c_unop('++', c_var(kh)),
                    [
                        c_for(
                            c_decl_init(c_type(int), kw, c_int(0)),
                            c_binop('<', c_var(kw), c_var('kW')),
                            c_unop('++', c_var(kw)),
                            [
                                c_comment('Input spatial position with stride/pad/dilation'),
                                c_decl_init(c_type(int), ih,
                                    c_binop('+',
                                        c_binop('-',
                                            c_binop('*', c_var(oh), c_var(stride_h)),
                                            c_var(pad_h)),
                                        c_binop('*', c_var(kh), c_var(dilation_h)))),
                                c_decl_init(c_type(int), iw,
                                    c_binop('+',
                                        c_binop('-',
                                            c_binop('*', c_var(ow), c_var(stride_w)),
                                            c_var(pad_w)),
                                        c_binop('*', c_var(kw), c_var(dilation_w)))),
                                c_decl_init(c_type(float), v, c_float(0.0)),
                                c_if(c_binop('&&',
                                        c_binop('&&',
                                            c_binop('>=', c_var(ih), c_int(0)),
                                            c_binop('<', c_var(ih), c_var('H'))),
                                        c_binop('&&',
                                            c_binop('>=', c_var(iw), c_int(0)),
                                            c_binop('<', c_var(iw), c_var('W')))),
                                     [
                                         c_assign(c_var(v),
                                             c_index(c_var('X'),
                                                 c_binop('+',
                                                     c_binop('*',
                                                         c_paren(c_binop('+',
                                                             c_binop('*',
                                                                 c_paren(c_binop('+',
                                                                     c_binop('*', c_var(b), c_var('C')),
                                                                     c_var(c))),
                                                                 c_var('H')),
                                                             c_var(ih))),
                                                         c_var('W')),
                                                     c_var(iw))))
                                     ]),
                                c_comment('Write to im2col output at column (c * kH + kh) * kW + kw'),
                                c_decl_init(c_type(int), col,
                                    c_binop('+',
                                        c_binop('*',
                                            c_paren(c_binop('+',
                                                c_binop('*', c_var(c), c_var('kH')),
                                                c_var(kh))),
                                            c_var('kW')),
                                        c_var(kw))),
                                c_assign(
                                    c_index(c_var('Y'),
                                        c_binop('+',
                                            c_binop('*', c_var(out_row), c_var(patch_size)),
                                            c_var(col))),
                                    c_var(v))
                            ]
                        )
                    ]
                )
            ]
        )
    ],
    Kernel = c_func(['__global__'], c_type(void), Name, Params, Body).

%% 1D forward — full body implementation
%% Per Tier 2 plan 8d65ba1c subtask 2-rs-i-1.
%% Empirically verified stub status by bit_identical harness 2026-05-19:
%% commit 0f952ee95 showed im2col_1d_forward returns all-zero output (STUB_DETECTED).
%% This body unblocks 2 KernelBench L1 conv_1d problems (#67, #76).
%%
%% Algorithm: for each (b, ol, c, kl), compute il=ol*stride - pad + kl*dilation;
%% if 0 <= il < L load X[b*C*L + c*L + il] else 0; write to
%% Y[(b*outL + ol) * (C*kL) + (c*kL + kl)]. Direct 1D analog of im2col_2d_forward.
build_im2col_kernel(Name, 1, forward, Kernel) :-
    !,
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'B'),
        param(c_type(int), 'C'),
        param(c_type(int), 'L'),       % input length
        param(c_type(int), 'kL'),
        param(c_type(int), 'outL'),
        param(c_type(int), stride),
        param(c_type(int), pad),
        param(c_type(int), dilation)
    ],
    Body = [
        c_comment('=== im2col 1D forward — unfolds patches into [B*outL, C*kL] ==='),
        c_comment('Thread layout: blockIdx.y=batch, x dim covers outL'),
        c_decl_init(c_type(int), b, c_member(c_var(blockIdx), y)),
        c_decl_init(c_type(int), ol,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('||',
                c_binop('>=', c_var(b), c_var('B')),
                c_binop('>=', c_var(ol), c_var('outL'))),
             [c_return_void]),
        c_blank,
        c_comment('Output row index in im2col matrix: b * outL + ol'),
        c_decl_init(c_type(int), out_row,
            c_binop('+',
                c_binop('*', c_var(b), c_var('outL')),
                c_var(ol))),
        c_comment('Each output row has C * kL entries'),
        c_decl_init(c_type(int), patch_size,
            c_binop('*', c_var('C'), c_var('kL'))),
        c_blank,
        c_comment('Unfold the patch: iterate over channels and kernel offsets'),
        c_for(
            c_decl_init(c_type(int), c, c_int(0)),
            c_binop('<', c_var(c), c_var('C')),
            c_unop('++', c_var(c)),
            [
                c_for(
                    c_decl_init(c_type(int), kl, c_int(0)),
                    c_binop('<', c_var(kl), c_var('kL')),
                    c_unop('++', c_var(kl)),
                    [
                        c_comment('Input position with stride/pad/dilation'),
                        c_decl_init(c_type(int), il,
                            c_binop('+',
                                c_binop('-',
                                    c_binop('*', c_var(ol), c_var(stride)),
                                    c_var(pad)),
                                c_binop('*', c_var(kl), c_var(dilation)))),
                        c_decl_init(c_type(float), v, c_float(0.0)),
                        c_if(c_binop('&&',
                                c_binop('>=', c_var(il), c_int(0)),
                                c_binop('<', c_var(il), c_var('L'))),
                             [
                                 c_assign(c_var(v),
                                     c_index(c_var('X'),
                                         c_binop('+',
                                             c_binop('*',
                                                 c_paren(c_binop('+',
                                                     c_binop('*', c_var(b), c_var('C')),
                                                     c_var(c))),
                                                 c_var('L')),
                                             c_var(il))))
                             ]),
                        c_comment('Write to im2col output at column c * kL + kl'),
                        c_decl_init(c_type(int), col,
                            c_binop('+',
                                c_binop('*', c_var(c), c_var('kL')),
                                c_var(kl))),
                        c_assign(
                            c_index(c_var('Y'),
                                c_binop('+',
                                    c_binop('*', c_var(out_row), c_var(patch_size)),
                                    c_var(col))),
                            c_var(v))
                    ]
                )
            ]
        )
    ],
    Kernel = c_func(['__global__'], c_type(void), Name, Params, Body).

%% 3D forward — full body implementation
%% Per Tier 2 plan 8d65ba1c subtask 2-rs-i-2. Empirically verified stub
%% status by `make bit_identical` 2026-05-19 (output all-zero). This body
%% unblocks 4 KernelBench L1 conv_3d problems.
%%
%% Thread layout substrate-design choice: CUDA grid is 3D, but we have 4
%% spatial dimensions to cover (batch, outD, outH, outW). Collapse batch
%% and outD into blockIdx.z: `bz = b * outD + od`. Then blockIdx.y=outH,
%% blockIdx.x covers outW.
%%
%% Adds stride_d/h/w, pad_d/h/w, dilation_d/h/w params (substrate-historical
%% 3D stub was missing these; 1D and 2D have them).
build_im2col_kernel(Name, 3, forward, Kernel) :-
    !,
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'B'),
        param(c_type(int), 'C'),
        param(c_type(int), 'D'),
        param(c_type(int), 'H'),
        param(c_type(int), 'W'),
        param(c_type(int), 'kD'),
        param(c_type(int), 'kH'),
        param(c_type(int), 'kW'),
        param(c_type(int), 'outD'),
        param(c_type(int), 'outH'),
        param(c_type(int), 'outW'),
        param(c_type(int), stride_d),
        param(c_type(int), stride_h),
        param(c_type(int), stride_w),
        param(c_type(int), pad_d),
        param(c_type(int), pad_h),
        param(c_type(int), pad_w),
        param(c_type(int), dilation_d),
        param(c_type(int), dilation_h),
        param(c_type(int), dilation_w)
    ],
    Body = [
        c_comment('=== im2col 3D forward — unfolds patches into [B*outD*outH*outW, C*kD*kH*kW] ==='),
        c_comment('Thread layout: blockIdx.z = b*outD+od; blockIdx.y=outH; x covers outW'),
        c_decl_init(c_type(int), bz, c_member(c_var(blockIdx), z)),
        c_decl_init(c_type(int), b,
            c_binop('/', c_var(bz), c_var('outD'))),
        c_decl_init(c_type(int), od,
            c_binop('%', c_var(bz), c_var('outD'))),
        c_decl_init(c_type(int), oh, c_member(c_var(blockIdx), y)),
        c_decl_init(c_type(int), ow,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('||',
                c_binop('>=', c_var(b), c_var('B')),
                c_binop('||',
                    c_binop('>=', c_var(oh), c_var('outH')),
                    c_binop('>=', c_var(ow), c_var('outW')))),
             [c_return_void]),
        c_blank,
        c_comment('Output row index: ((b*outD + od) * outH + oh) * outW + ow'),
        c_decl_init(c_type(int), out_row,
            c_binop('+',
                c_binop('*',
                    c_paren(c_binop('+',
                        c_binop('*',
                            c_paren(c_binop('+',
                                c_binop('*', c_var(b), c_var('outD')),
                                c_var(od))),
                            c_var('outH')),
                        c_var(oh))),
                    c_var('outW')),
                c_var(ow))),
        c_comment('Each output row has C * kD * kH * kW entries'),
        c_decl_init(c_type(int), patch_size,
            c_binop('*', c_var('C'),
                c_binop('*', c_var('kD'),
                    c_binop('*', c_var('kH'), c_var('kW'))))),
        c_blank,
        c_comment('Unfold the patch: iterate over channels and 3D kernel offsets'),
        c_for(
            c_decl_init(c_type(int), c, c_int(0)),
            c_binop('<', c_var(c), c_var('C')),
            c_unop('++', c_var(c)),
            [
                c_for(
                    c_decl_init(c_type(int), kd, c_int(0)),
                    c_binop('<', c_var(kd), c_var('kD')),
                    c_unop('++', c_var(kd)),
                    [
                        c_for(
                            c_decl_init(c_type(int), kh, c_int(0)),
                            c_binop('<', c_var(kh), c_var('kH')),
                            c_unop('++', c_var(kh)),
                            [
                                c_for(
                                    c_decl_init(c_type(int), kw, c_int(0)),
                                    c_binop('<', c_var(kw), c_var('kW')),
                                    c_unop('++', c_var(kw)),
                                    [
                                        c_comment('Input position in each of D, H, W'),
                                        c_decl_init(c_type(int), id,
                                            c_binop('+',
                                                c_binop('-',
                                                    c_binop('*', c_var(od), c_var(stride_d)),
                                                    c_var(pad_d)),
                                                c_binop('*', c_var(kd), c_var(dilation_d)))),
                                        c_decl_init(c_type(int), ih,
                                            c_binop('+',
                                                c_binop('-',
                                                    c_binop('*', c_var(oh), c_var(stride_h)),
                                                    c_var(pad_h)),
                                                c_binop('*', c_var(kh), c_var(dilation_h)))),
                                        c_decl_init(c_type(int), iw,
                                            c_binop('+',
                                                c_binop('-',
                                                    c_binop('*', c_var(ow), c_var(stride_w)),
                                                    c_var(pad_w)),
                                                c_binop('*', c_var(kw), c_var(dilation_w)))),
                                        c_decl_init(c_type(float), v, c_float(0.0)),
                                        c_if(c_binop('&&',
                                                c_binop('&&',
                                                    c_binop('>=', c_var(id), c_int(0)),
                                                    c_binop('<', c_var(id), c_var('D'))),
                                                c_binop('&&',
                                                    c_binop('&&',
                                                        c_binop('>=', c_var(ih), c_int(0)),
                                                        c_binop('<', c_var(ih), c_var('H'))),
                                                    c_binop('&&',
                                                        c_binop('>=', c_var(iw), c_int(0)),
                                                        c_binop('<', c_var(iw), c_var('W'))))),
                                             [
                                                 c_comment('X index: (((b*C + c)*D + id)*H + ih)*W + iw'),
                                                 c_assign(c_var(v),
                                                     c_index(c_var('X'),
                                                         c_binop('+',
                                                             c_binop('*',
                                                                 c_paren(c_binop('+',
                                                                     c_binop('*',
                                                                         c_paren(c_binop('+',
                                                                             c_binop('*',
                                                                                 c_paren(c_binop('+',
                                                                                     c_binop('*', c_var(b), c_var('C')),
                                                                                     c_var(c))),
                                                                                 c_var('D')),
                                                                             c_var(id))),
                                                                         c_var('H')),
                                                                     c_var(ih))),
                                                                 c_var('W')),
                                                             c_var(iw))))
                                             ]),
                                        c_comment('Output column: ((c*kD + kd)*kH + kh)*kW + kw'),
                                        c_decl_init(c_type(int), col,
                                            c_binop('+',
                                                c_binop('*',
                                                    c_paren(c_binop('+',
                                                        c_binop('*',
                                                            c_paren(c_binop('+',
                                                                c_binop('*', c_var(c), c_var('kD')),
                                                                c_var(kd))),
                                                            c_var('kH')),
                                                        c_var(kh))),
                                                    c_var('kW')),
                                                c_var(kw))),
                                        c_assign(
                                            c_index(c_var('Y'),
                                                c_binop('+',
                                                    c_binop('*', c_var(out_row), c_var(patch_size)),
                                                    c_var(col))),
                                            c_var(v))
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ],
    Kernel = c_func(['__global__'], c_type(void), Name, Params, Body).

%% Transposed variants (col2im) — per-dim implementations.
%% Substantive substrate-design choice: parallelize over OUTPUT (spatial)
%% elements, not input (col matrix) elements. This avoids atomicAdd, makes
%% the kernel deterministic, and matches PyTorch's conv_transpose semantics.
%% Each output element gathers contributions from kernel positions that
%% target it under the inverse-stride map.
%%
%% Inverse map: output position l is written by (ol, kl) such that
%%   ol*stride - pad + kl*dilation == l
%% Solving for ol given (l, kl):
%%   ol = (l + pad - kl*dilation) / stride
%% Valid iff: (l + pad - kl*dilation) % stride == 0 AND 0 <= ol < L_in_col.

%% 1D col2im transpose — full body
%% Per Tier 2 plan 8d65ba1c subtask 2-rs-i-3.
%% Empirically verified stub status via `make bit_identical`; this body
%% unblocks 3 KernelBench L1 conv_transpose_1d problems.
build_im2col_kernel(Name, 1, transpose, Kernel) :-
    !,
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),    % col matrix [B*L_in_col, C*kL]
        param(c_type(ptr(c_type(float))), 'Y'),          % spatial output [B, C, L_out]
        param(c_type(int), 'B'),
        param(c_type(int), 'C'),
        param(c_type(int), 'L_in_col'),                  % rows of col matrix
        param(c_type(int), 'L_out'),                     % spatial output length
        param(c_type(int), 'kL'),
        param(c_type(int), stride),
        param(c_type(int), pad),
        param(c_type(int), dilation)
    ],
    Body = [
        c_comment('=== col2im 1D transpose — gather pattern, no atomicAdd ==='),
        c_comment('Thread layout: blockIdx.z=batch, blockIdx.y=channel, x covers L_out'),
        c_decl_init(c_type(int), b, c_member(c_var(blockIdx), z)),
        c_decl_init(c_type(int), c, c_member(c_var(blockIdx), y)),
        c_decl_init(c_type(int), l,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('||',
                c_binop('>=', c_var(b), c_var('B')),
                c_binop('||',
                    c_binop('>=', c_var(c), c_var('C')),
                    c_binop('>=', c_var(l), c_var('L_out')))),
             [c_return_void]),
        c_blank,
        c_comment('Sum contributions from kernel positions targeting this output l'),
        c_decl_init(c_type(float), acc, c_float(0.0)),
        c_for(
            c_decl_init(c_type(int), kl, c_int(0)),
            c_binop('<', c_var(kl), c_var('kL')),
            c_unop('++', c_var(kl)),
            [
                c_comment('Compute ol such that ol*stride - pad + kl*dilation == l'),
                c_decl_init(c_type(int), numerator,
                    c_binop('+', c_var(l),
                        c_binop('-', c_var(pad),
                            c_binop('*', c_var(kl), c_var(dilation))))),
                c_comment('Skip if not divisible by stride (no integer ol exists)'),
                c_if(c_binop('==',
                        c_binop('%', c_var(numerator), c_var(stride)),
                        c_int(0)),
                    [
                        c_decl_init(c_type(int), ol,
                            c_binop('/', c_var(numerator), c_var(stride))),
                        c_if(c_binop('&&',
                                c_binop('>=', c_var(ol), c_int(0)),
                                c_binop('<', c_var(ol), c_var('L_in_col'))),
                            [
                                c_comment('X[b*L_in_col+ol, c*kL+kl]'),
                                c_decl_init(c_type(int), col_row,
                                    c_binop('+',
                                        c_binop('*', c_var(b), c_var('L_in_col')),
                                        c_var(ol))),
                                c_decl_init(c_type(int), col_col,
                                    c_binop('+',
                                        c_binop('*', c_var(c), c_var('kL')),
                                        c_var(kl))),
                                c_decl_init(c_type(int), patch_size,
                                    c_binop('*', c_var('C'), c_var('kL'))),
                                c_compound_assign('+=', c_var(acc),
                                    c_index(c_var('X'),
                                        c_binop('+',
                                            c_binop('*', c_var(col_row), c_var(patch_size)),
                                            c_var(col_col))))
                            ])
                    ])
            ]
        ),
        c_comment('Y[b, c, l] = (b*C + c)*L_out + l'),
        c_assign(
            c_index(c_var('Y'),
                c_binop('+',
                    c_binop('*',
                        c_paren(c_binop('+',
                            c_binop('*', c_var(b), c_var('C')),
                            c_var(c))),
                        c_var('L_out')),
                    c_var(l))),
            c_var(acc))
    ],
    Kernel = c_func(['__global__'], c_type(void), Name, Params, Body).

%% 2D col2im transpose — full body
%% Per Tier 2 plan 8d65ba1c subtask 2-rs-i-4.
%% Gather pattern: parallelize over output spatial (H_out, W_out); for each
%% output position, iterate over (kh, kw), use inverse map to find (oh, ow)
%% in the col matrix, accumulate. Deterministic, no atomicAdd.
%%
%% Inverse map per dimension:
%%   oh = (h + pad_h - kh*dilation_h) / stride_h  (must be exact int division)
%%   ow = (w + pad_w - kw*dilation_w) / stride_w
build_im2col_kernel(Name, 2, transpose, Kernel) :-
    !,
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),  % col matrix [B*H_in*W_in, C*kH*kW]
        param(c_type(ptr(c_type(float))), 'Y'),        % spatial [B, C, H_out, W_out]
        param(c_type(int), 'B'),
        param(c_type(int), 'C'),
        param(c_type(int), 'H_in'),
        param(c_type(int), 'W_in'),
        param(c_type(int), 'H_out'),
        param(c_type(int), 'W_out'),
        param(c_type(int), 'kH'),
        param(c_type(int), 'kW'),
        param(c_type(int), stride_h),
        param(c_type(int), stride_w),
        param(c_type(int), pad_h),
        param(c_type(int), pad_w),
        param(c_type(int), dilation_h),
        param(c_type(int), dilation_w)
    ],
    Body = [
        c_comment('=== col2im 2D transpose — gather pattern, no atomicAdd ==='),
        c_comment('Thread layout: blockIdx.z=b*C+c, blockIdx.y=H_out, x covers W_out'),
        c_decl_init(c_type(int), bc, c_member(c_var(blockIdx), z)),
        c_decl_init(c_type(int), b, c_binop('/', c_var(bc), c_var('C'))),
        c_decl_init(c_type(int), c, c_binop('%', c_var(bc), c_var('C'))),
        c_decl_init(c_type(int), h, c_member(c_var(blockIdx), y)),
        c_decl_init(c_type(int), w,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('||',
                c_binop('>=', c_var(b), c_var('B')),
                c_binop('||',
                    c_binop('>=', c_var(h), c_var('H_out')),
                    c_binop('>=', c_var(w), c_var('W_out')))),
             [c_return_void]),
        c_blank,
        c_comment('Each col-matrix row has C*kH*kW entries'),
        c_decl_init(c_type(int), patch_size,
            c_binop('*', c_var('C'),
                c_binop('*', c_var('kH'), c_var('kW')))),
        c_decl_init(c_type(float), acc, c_float(0.0)),
        c_for(
            c_decl_init(c_type(int), kh, c_int(0)),
            c_binop('<', c_var(kh), c_var('kH')),
            c_unop('++', c_var(kh)),
            [
                c_decl_init(c_type(int), num_h,
                    c_binop('+', c_var(h),
                        c_binop('-', c_var(pad_h),
                            c_binop('*', c_var(kh), c_var(dilation_h))))),
                c_if(c_binop('==',
                        c_binop('%', c_var(num_h), c_var(stride_h)),
                        c_int(0)),
                    [
                        c_decl_init(c_type(int), oh,
                            c_binop('/', c_var(num_h), c_var(stride_h))),
                        c_if(c_binop('&&',
                                c_binop('>=', c_var(oh), c_int(0)),
                                c_binop('<', c_var(oh), c_var('H_in'))),
                            [
                                c_for(
                                    c_decl_init(c_type(int), kw, c_int(0)),
                                    c_binop('<', c_var(kw), c_var('kW')),
                                    c_unop('++', c_var(kw)),
                                    [
                                        c_decl_init(c_type(int), num_w,
                                            c_binop('+', c_var(w),
                                                c_binop('-', c_var(pad_w),
                                                    c_binop('*', c_var(kw), c_var(dilation_w))))),
                                        c_if(c_binop('==',
                                                c_binop('%', c_var(num_w), c_var(stride_w)),
                                                c_int(0)),
                                            [
                                                c_decl_init(c_type(int), ow,
                                                    c_binop('/', c_var(num_w), c_var(stride_w))),
                                                c_if(c_binop('&&',
                                                        c_binop('>=', c_var(ow), c_int(0)),
                                                        c_binop('<', c_var(ow), c_var('W_in'))),
                                                    [
                                                        c_comment('col_row = (b*H_in + oh)*W_in + ow'),
                                                        c_decl_init(c_type(int), col_row,
                                                            c_binop('+',
                                                                c_binop('*',
                                                                    c_paren(c_binop('+',
                                                                        c_binop('*', c_var(b), c_var('H_in')),
                                                                        c_var(oh))),
                                                                    c_var('W_in')),
                                                                c_var(ow))),
                                                        c_comment('col_col = (c*kH + kh)*kW + kw'),
                                                        c_decl_init(c_type(int), col_col,
                                                            c_binop('+',
                                                                c_binop('*',
                                                                    c_paren(c_binop('+',
                                                                        c_binop('*', c_var(c), c_var('kH')),
                                                                        c_var(kh))),
                                                                    c_var('kW')),
                                                                c_var(kw))),
                                                        c_compound_assign('+=', c_var(acc),
                                                            c_index(c_var('X'),
                                                                c_binop('+',
                                                                    c_binop('*', c_var(col_row), c_var(patch_size)),
                                                                    c_var(col_col))))
                                                    ])
                                            ])
                                    ]
                                )
                            ])
                    ])
            ]
        ),
        c_comment('Y[b, c, h, w] = ((b*C + c)*H_out + h)*W_out + w'),
        c_assign(
            c_index(c_var('Y'),
                c_binop('+',
                    c_binop('*',
                        c_paren(c_binop('+',
                            c_binop('*',
                                c_paren(c_binop('+',
                                    c_binop('*', c_var(b), c_var('C')),
                                    c_var(c))),
                                c_var('H_out')),
                            c_var(h))),
                        c_var('W_out')),
                    c_var(w))),
            c_var(acc))
    ],
    Kernel = c_func(['__global__'], c_type(void), Name, Params, Body).

%% 3D col2im transpose — full body
%% Per Tier 2 plan 8d65ba1c subtask 2-rs-i-5.
%% 3D analog of 2D transpose. Gather pattern: parallelize over output spatial
%% (D_out, H_out, W_out); inverse map in each of 3 dims.
%%
%% Thread layout: CUDA grid is 3D; collapse (b, c) into blockIdx.z=b*C+c,
%% (d, h) into blockIdx.y=d*H_out+h, blockIdx.x*blockDim.x covers W_out.
build_im2col_kernel(Name, 3, transpose, Kernel) :-
    !,
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),  % col matrix
        param(c_type(ptr(c_type(float))), 'Y'),        % spatial [B,C,D_out,H_out,W_out]
        param(c_type(int), 'B'),
        param(c_type(int), 'C'),
        param(c_type(int), 'D_in'),
        param(c_type(int), 'H_in'),
        param(c_type(int), 'W_in'),
        param(c_type(int), 'D_out'),
        param(c_type(int), 'H_out'),
        param(c_type(int), 'W_out'),
        param(c_type(int), 'kD'),
        param(c_type(int), 'kH'),
        param(c_type(int), 'kW'),
        param(c_type(int), stride_d),
        param(c_type(int), stride_h),
        param(c_type(int), stride_w),
        param(c_type(int), pad_d),
        param(c_type(int), pad_h),
        param(c_type(int), pad_w),
        param(c_type(int), dilation_d),
        param(c_type(int), dilation_h),
        param(c_type(int), dilation_w)
    ],
    Body = [
        c_comment('=== col2im 3D transpose — gather pattern, no atomicAdd ==='),
        c_comment('Thread layout: blockIdx.z=b*C+c, blockIdx.y=d*H_out+h, x covers W_out'),
        c_decl_init(c_type(int), bc, c_member(c_var(blockIdx), z)),
        c_decl_init(c_type(int), b, c_binop('/', c_var(bc), c_var('C'))),
        c_decl_init(c_type(int), c, c_binop('%', c_var(bc), c_var('C'))),
        c_decl_init(c_type(int), dh, c_member(c_var(blockIdx), y)),
        c_decl_init(c_type(int), d, c_binop('/', c_var(dh), c_var('H_out'))),
        c_decl_init(c_type(int), h, c_binop('%', c_var(dh), c_var('H_out'))),
        c_decl_init(c_type(int), w,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('||',
                c_binop('>=', c_var(b), c_var('B')),
                c_binop('||',
                    c_binop('>=', c_var(d), c_var('D_out')),
                    c_binop('>=', c_var(w), c_var('W_out')))),
             [c_return_void]),
        c_blank,
        c_decl_init(c_type(int), patch_size,
            c_binop('*', c_var('C'),
                c_binop('*', c_var('kD'),
                    c_binop('*', c_var('kH'), c_var('kW'))))),
        c_decl_init(c_type(float), acc, c_float(0.0)),
        c_for(
            c_decl_init(c_type(int), kd, c_int(0)),
            c_binop('<', c_var(kd), c_var('kD')),
            c_unop('++', c_var(kd)),
            [
                c_decl_init(c_type(int), num_d,
                    c_binop('+', c_var(d),
                        c_binop('-', c_var(pad_d),
                            c_binop('*', c_var(kd), c_var(dilation_d))))),
                c_if(c_binop('==', c_binop('%', c_var(num_d), c_var(stride_d)), c_int(0)),
                    [
                        c_decl_init(c_type(int), od, c_binop('/', c_var(num_d), c_var(stride_d))),
                        c_if(c_binop('&&',
                                c_binop('>=', c_var(od), c_int(0)),
                                c_binop('<', c_var(od), c_var('D_in'))),
                            [
                                c_for(
                                    c_decl_init(c_type(int), kh, c_int(0)),
                                    c_binop('<', c_var(kh), c_var('kH')),
                                    c_unop('++', c_var(kh)),
                                    [
                                        c_decl_init(c_type(int), num_h,
                                            c_binop('+', c_var(h),
                                                c_binop('-', c_var(pad_h),
                                                    c_binop('*', c_var(kh), c_var(dilation_h))))),
                                        c_if(c_binop('==', c_binop('%', c_var(num_h), c_var(stride_h)), c_int(0)),
                                            [
                                                c_decl_init(c_type(int), oh, c_binop('/', c_var(num_h), c_var(stride_h))),
                                                c_if(c_binop('&&',
                                                        c_binop('>=', c_var(oh), c_int(0)),
                                                        c_binop('<', c_var(oh), c_var('H_in'))),
                                                    [
                                                        c_for(
                                                            c_decl_init(c_type(int), kw, c_int(0)),
                                                            c_binop('<', c_var(kw), c_var('kW')),
                                                            c_unop('++', c_var(kw)),
                                                            [
                                                                c_decl_init(c_type(int), num_w,
                                                                    c_binop('+', c_var(w),
                                                                        c_binop('-', c_var(pad_w),
                                                                            c_binop('*', c_var(kw), c_var(dilation_w))))),
                                                                c_if(c_binop('==', c_binop('%', c_var(num_w), c_var(stride_w)), c_int(0)),
                                                                    [
                                                                        c_decl_init(c_type(int), ow, c_binop('/', c_var(num_w), c_var(stride_w))),
                                                                        c_if(c_binop('&&',
                                                                                c_binop('>=', c_var(ow), c_int(0)),
                                                                                c_binop('<', c_var(ow), c_var('W_in'))),
                                                                            [
                                                                                c_comment('col_row = ((b*D_in+od)*H_in+oh)*W_in+ow'),
                                                                                c_decl_init(c_type(int), col_row,
                                                                                    c_binop('+',
                                                                                        c_binop('*',
                                                                                            c_paren(c_binop('+',
                                                                                                c_binop('*',
                                                                                                    c_paren(c_binop('+',
                                                                                                        c_binop('*', c_var(b), c_var('D_in')),
                                                                                                        c_var(od))),
                                                                                                    c_var('H_in')),
                                                                                                c_var(oh))),
                                                                                            c_var('W_in')),
                                                                                        c_var(ow))),
                                                                                c_comment('col_col = ((c*kD+kd)*kH+kh)*kW+kw'),
                                                                                c_decl_init(c_type(int), col_col,
                                                                                    c_binop('+',
                                                                                        c_binop('*',
                                                                                            c_paren(c_binop('+',
                                                                                                c_binop('*',
                                                                                                    c_paren(c_binop('+',
                                                                                                        c_binop('*', c_var(c), c_var('kD')),
                                                                                                        c_var(kd))),
                                                                                                    c_var('kH')),
                                                                                                c_var(kh))),
                                                                                            c_var('kW')),
                                                                                        c_var(kw))),
                                                                                c_compound_assign('+=', c_var(acc),
                                                                                    c_index(c_var('X'),
                                                                                        c_binop('+',
                                                                                            c_binop('*', c_var(col_row), c_var(patch_size)),
                                                                                            c_var(col_col))))
                                                                            ])
                                                                    ])
                                                            ]
                                                        )
                                                    ])
                                            ])
                                    ]
                                )
                            ])
                    ])
            ]
        ),
        c_comment('Y[b, c, d, h, w] = (((b*C+c)*D_out+d)*H_out+h)*W_out+w'),
        c_assign(
            c_index(c_var('Y'),
                c_binop('+',
                    c_binop('*',
                        c_paren(c_binop('+',
                            c_binop('*',
                                c_paren(c_binop('+',
                                    c_binop('*',
                                        c_paren(c_binop('+',
                                            c_binop('*', c_var(b), c_var('C')),
                                            c_var(c))),
                                        c_var('D_out')),
                                    c_var(d))),
                                c_var('H_out')),
                            c_var(h))),
                        c_var('W_out')),
                    c_var(w))),
            c_var(acc))
    ],
    Kernel = c_func(['__global__'], c_type(void), Name, Params, Body).

%% ═════════════════════════════════════════════════════════════════════
%% Family 1: Reductions (sequential, one template covers 8 ops)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Per mavchin's directive (inbox 2026-05-15 18:42:14):
%% "Reduction approach: sequential first (simpler, correct). Warp shuffle
%%  optimization later — that's a TILE_K style optimization the
%%  warp_optimizer can derive from hardware_facts."
%%
%% generate_kernel_reduction(+OpKind, +Dim, +ReductionMode, -KernelAST)
%%
%% OpKind ∈ {ggml_sum_rows, ggml_mean, ggml_max, ggml_min,
%%           ggml_argmax, ggml_argmin, ggml_cumsum, ggml_cumprod}
%% Dim: input dimensionality (1, 2, 3, ...). Currently unused at codegen
%%      time because the kernel reads N at runtime; reserved for future
%%      multi-axis reduction variants.
%% ReductionMode: axis_inner | axis_outer (only axis_inner exercised by L1)
%% KernelAST: c_func/5 AST
%%
%% Per mavchin's spec (inbox 18:42:14): generate_kernel_reduction/4
%% naming convention consistent with generate_fused_kernel/3.
%%
%% The template is:
%%
%%   __global__ void reduce_<KIND>(const float * X, float * Y,
%%                                  int N, int outer) {
%%     int o = blockIdx.x * blockDim.x + threadIdx.x;
%%     if (o >= outer) return;
%%     float acc = <INIT>;
%%     int arg = 0;        // argmax/argmin only; harmless for others
%%     for (int i = 0; i < N; i++) {
%%       float v = X[o * N + i];
%%       <ACCUMULATE>;     // acc += v / acc = max(acc, v) / ...
%%     }
%%     Y[o] = <FINALIZE>;  // acc / acc/N for mean / (float)arg for argmax
%%   }
%%
%% Cumulative variants (cumsum, cumprod) write each step inline rather
%% than reducing, so they share most of the template but emit Y[idx]
%% inside the loop instead of after.

%% ═══════════════════════════════════════════════════════════════════════════
%% Reduction strategy substrate-design vocabulary (subtask 2-rs-b)
%% ═══════════════════════════════════════════════════════════════════════════
%%
%% reduction_strategy_kind(+Kind) — enumerate valid strategies.
%%
%% Each strategy is IEEE-correct but produces different bit patterns due to
%% float32's non-associativity. The substrate exposes the strategy as an
%% explicit parameter; user picks which set of bits they want.
%%
%% sequential:    acc = X[0]; for i in 1..N: acc = acc + X[i]
%%                Single accumulator, FADD chain at SASS. Bit pattern A.
%%                Currently the substrate's default for generate_kernel_reduction/4.
%%
%% block_reduce:  256 threads cooperate on one row. Each thread accumulates
%%                its strided partial sum, then block_reduce_sum_helper
%%                combines via warp-shuffle + cross-warp. Bit pattern B.
%%                Matches the substrate's existing rms_norm/softmax pattern.
%%                At SASS: SHFL instructions + reduced FADD count per row.
%%
%% pairwise_tree: classical tree reduction. log2(N) passes, each halves count.
%%                Requires N-sized auxiliary buffer or in-place ping-pong.
%%                Matches PyTorch's default for many reductions. Bit pattern C.
%%
%% kahan:         compensated summation. Tracks per-step round-off and
%%                corrects. Numerically most accurate. Bit pattern D.
%%                Per-thread overhead but no architecture change.

reduction_strategy_kind(sequential).
reduction_strategy_kind(block_reduce).
reduction_strategy_kind(pairwise_tree).
reduction_strategy_kind(kahan).

reduction_strategy_description(sequential,
    'Single-accumulator FADD chain. Substrate-historical default.').
reduction_strategy_description(block_reduce,
    'Block-cooperative warp-shuffle reduction. Matches substrate rms_norm/softmax pattern.').
reduction_strategy_description(pairwise_tree,
    'Tree reduction with log2(N) passes. Matches PyTorch default.').
reduction_strategy_description(kahan,
    'Compensated summation. Numerically robust, single accumulator.').

%% Reduction-style ops (single output per row)
%% 4-arity: substrate-historical default (strategy=sequential).
generate_kernel_reduction(OpKind, Dim, axis_inner, KernelAST) :-
    generate_kernel_reduction(OpKind, Dim, axis_inner, sequential, KernelAST).

%% 5-arity: explicit strategy.
generate_kernel_reduction(OpKind, _Dim, axis_inner, sequential, KernelAST) :-
    member(OpKind, [ggml_sum_rows, ggml_mean, ggml_max, ggml_min,
                    ggml_argmax, ggml_argmin]),
    !,
    reduction_kernel_name(OpKind, KernelName),
    reduction_params(OpKind, InitValue, AccumStmt, FinalizeExpr),
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'N'),
        param(c_type(int), outer)
    ],
    %% Derive whether the `arg` index variable is needed by asking the
    %% AST itself: does AccumStmt or FinalizeExpr reference c_var(arg)?
    %% Substrate-honest — no per-op flag, no parallel side table. The
    %% AST already encodes the structural fact (argmax/argmin assign
    %% to c_var(arg); sum/mean/max/min don't); we just ask it.
    %% Per Heath's "investigate at root cause and resolve" direction.
    ( ( ast_uses_var(AccumStmt, arg) ; ast_uses_var(FinalizeExpr, arg) )
    -> ArgDecls = [c_decl_init(c_type(int), arg, c_int(0))]
    ;  ArgDecls = []
    ),

    %% Thread index: one thread per output element (per "row")
    BodyHead = [
        c_comment('=== reduction kernel (sequential, parameterized by OpKind) ==='),
        c_decl_init(c_type(int), o,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('>=', c_var(o), c_var(outer)), [c_return_void]),
        c_blank,
        c_decl_init(c_type(float), acc, InitValue)
    ],
    BodyTail = [
        c_for(
            c_decl_init(c_type(int), i, c_int(0)),
            c_binop('<', c_var(i), c_var('N')),
            c_unop('++', c_var(i)),
            [
                c_comment('Read element X[o * N + i] with explicit parens'),
                c_decl_init(c_type(float), v,
                    c_index(c_var('X'),
                        c_binop('+',
                            c_binop('*', c_var(o), c_var('N')),
                            c_var(i)))),
                AccumStmt
            ]
        ),
        c_assign(c_index(c_var('Y'), c_var(o)), FinalizeExpr)
    ],
    append([BodyHead, ArgDecls, BodyTail], Body),
    KernelAST = c_func(['__global__'], c_type(void), KernelName, Params, Body).

%% block_reduce strategy: 256 threads cooperate per row via block_reduce_sum.
%% Substrate-design choice surfaced by Tier 2 verification subtask 2-rs-c.
%% Empirically expected to converge with PyTorch reduction order.
%%
%% Requires block_reduce_sum_helper to be emitted above this kernel in the
%% same .cu file (substrate-historical pattern, see rms_norm_kernel/1).
%%
%% Currently implemented for: ggml_sum_rows, ggml_mean.
%% (Other ops like argmax don't have a direct block-reduce analog and stay sequential.)
generate_kernel_reduction(OpKind, _Dim, axis_inner, block_reduce, KernelAST) :-
    member(OpKind, [ggml_sum_rows, ggml_mean]),
    !,
    reduction_kernel_name(OpKind, KernelName),
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'N'),
        param(c_type(int), outer)
    ],
    %% For mean, divide by N at the end; for sum, just write the accumulated.
    ( OpKind = ggml_mean
    -> FinalAssign = c_assign(c_index(c_var('Y'), c_var(row)),
                              c_binop('/', c_var(total), c_var('N')))
    ;  FinalAssign = c_assign(c_index(c_var('Y'), c_var(row)), c_var(total))
    ),
    Body = [
        c_comment('=== block-reduce strategy: 256 threads cooperate per row ==='),
        c_comment('Each thread sums its strided slice; block_reduce_sum combines.'),
        c_decl_init(c_type(int), row, c_member(c_var(blockIdx), x)),
        c_if(c_binop('>=', c_var(row), c_var(outer)), [c_return_void]),
        c_blank,
        c_comment('Strided partial sum: thread t handles X[row, t], X[row, t+256], ...'),
        c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
        c_decl_init(c_type(float), local_sum, c_float(0.0)),
        c_for(
            c_decl_init(c_type(int), i, c_var(tid)),
            c_binop('<', c_var(i), c_var('N')),
            c_compound_step(c_var(i), '+=', c_int(256)),
            [
                c_compound_assign('+=', c_var(local_sum),
                    c_index(c_var('X'),
                        c_binop('+',
                            c_binop('*', c_var(row), c_var('N')),
                            c_var(i))))
            ]
        ),
        c_blank,
        c_comment('Block-level reduction via existing substrate helper'),
        c_shared_decl(c_type(float), block_sum_buf, c_int(8)),
        c_decl_init(c_type(float), total,
            c_call(block_reduce_sum, [c_var(local_sum), c_var(block_sum_buf)])),
        c_blank,
        c_if(c_binop('==', c_var(tid), c_int(0)),
             [FinalAssign])
    ],
    KernelAST = c_func(['__global__'], c_type(void), KernelName, Params, Body).

%% Cumulative variants (output same shape as input, write each step inline)
%% Order is the spec, not a substrate-design choice. Strategy parameter ignored.
generate_kernel_reduction(OpKind, _Dim, axis_inner, _Strategy, KernelAST) :-
    member(OpKind, [ggml_cumsum, ggml_cumprod]),
    !,
    reduction_kernel_name(OpKind, KernelName),
    reduction_params(OpKind, InitValue, AccumStmt, _),
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'N'),
        param(c_type(int), outer)
    ],
    Body = [
        c_comment('=== cumulative kernel (sequential prefix scan) ==='),
        c_decl_init(c_type(int), o,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('>=', c_var(o), c_var(outer)), [c_return_void]),
        c_blank,
        c_decl_init(c_type(float), acc, InitValue),
        c_for(
            c_decl_init(c_type(int), i, c_int(0)),
            c_binop('<', c_var(i), c_var('N')),
            c_unop('++', c_var(i)),
            [
                c_decl_init(c_type(int), idx,
                    c_binop('+',
                        c_binop('*', c_var(o), c_var('N')),
                        c_var(i))),
                c_decl_init(c_type(float), v, c_index(c_var('X'), c_var(idx))),
                AccumStmt,
                c_assign(c_index(c_var('Y'), c_var(idx)), c_var(acc))
            ]
        )
    ],
    KernelAST = c_func(['__global__'], c_type(void), KernelName, Params, Body).

%% reduction_kernel_name(+OpKind, -Name)
reduction_kernel_name(ggml_sum_rows, reduce_sum).
reduction_kernel_name(ggml_mean,     reduce_mean).
reduction_kernel_name(ggml_max,      reduce_max).
reduction_kernel_name(ggml_min,      reduce_min).
reduction_kernel_name(ggml_argmax,   reduce_argmax).
reduction_kernel_name(ggml_argmin,   reduce_argmin).
reduction_kernel_name(ggml_cumsum,   cumsum).
reduction_kernel_name(ggml_cumprod,  cumprod).

%% reduction_params(+OpKind, -InitValue, -AccumStmt, -FinalizeExpr)
%%   InitValue: initial accumulator value
%%   AccumStmt: statement updating acc/arg using v and i
%%   FinalizeExpr: what to assign to Y[o] (for non-cumulative variants)
reduction_params(ggml_sum_rows, c_float(0.0),
    c_assign(c_var(acc), c_binop('+', c_var(acc), c_var(v))),
    c_var(acc)).
reduction_params(ggml_mean, c_float(0.0),
    c_assign(c_var(acc), c_binop('+', c_var(acc), c_var(v))),
    c_binop('/', c_var(acc), c_cast(c_type(float), c_var('N')))).
reduction_params(ggml_max, c_float(-1.0e30),
    c_assign(c_var(acc),
        c_ternary(c_binop('>', c_var(v), c_var(acc)),
                  c_var(v), c_var(acc))),
    c_var(acc)).
reduction_params(ggml_min, c_float(1.0e30),
    c_assign(c_var(acc),
        c_ternary(c_binop('<', c_var(v), c_var(acc)),
                  c_var(v), c_var(acc))),
    c_var(acc)).
reduction_params(ggml_argmax, c_float(-1.0e30),
    c_if(c_binop('>', c_var(v), c_var(acc)),
         [c_assign(c_var(acc), c_var(v)),
          c_assign(c_var(arg), c_var(i))]),
    c_cast(c_type(float), c_var(arg))).
reduction_params(ggml_argmin, c_float(1.0e30),
    c_if(c_binop('<', c_var(v), c_var(acc)),
         [c_assign(c_var(acc), c_var(v)),
          c_assign(c_var(arg), c_var(i))]),
    c_cast(c_type(float), c_var(arg))).
reduction_params(ggml_cumsum, c_float(0.0),
    c_assign(c_var(acc), c_binop('+', c_var(acc), c_var(v))),
    c_var(acc)).
reduction_params(ggml_cumprod, c_float(1.0),
    c_assign(c_var(acc), c_binop('*', c_var(acc), c_var(v))),
    c_var(acc)).

%% ═════════════════════════════════════════════════════════════════════
%% Family 2: Normalizations (two-pass, one template covers 4 ops)
%% ═════════════════════════════════════════════════════════════════════
%%
%% generate_kernel_norm(+OpKind, +Dim, +Affine, -KernelAST)
%%
%% OpKind ∈ {ggml_norm, ggml_l2_norm, ggml_rms_norm, ggml_group_norm}
%% Dim:    input dimensionality (reserved; currently unused at codegen)
%% Affine: true | false  — whether learned scale (W) and bias (B) apply
%% KernelAST: c_func/5 AST
%%
%% Template shape:
%%
%%   __global__ void norm_<KIND>(const float * X, float * Y,
%%                               const float * W, const float * B,
%%                               int N, int outer, float eps) {
%%     int o = blockIdx.x * blockDim.x + threadIdx.x;
%%     if (o >= outer) return;
%%
%%     // Pass 1: compute statistics over the inner dim
%%     <STATS_DECLS>;
%%     for (int i = 0; i < N; i++) {
%%       float v = X[o * N + i];
%%       <STATS_UPDATE>;
%%     }
%%     <STATS_FINALIZE>;     // compute inv_std / inv_rms / inv_norm
%%
%%     // Pass 2: apply normalization
%%     for (int i = 0; i < N; i++) {
%%       int idx = o * N + i;
%%       float v = X[idx];
%%       Y[idx] = <APPLY>(v);   // (v-mean)*inv_std * W[i] + B[i] for layer
%%     }
%%   }
%%
%% Per-op parameterization:
%%   Layer norm (ggml_norm):     STATS = sum + sumsq → mean, variance, inv_std
%%                                APPLY = (v - mean) * inv_std
%%   RMS norm:                    STATS = sumsq → inv_rms = 1/sqrt(sumsq/N + eps)
%%                                APPLY = v * inv_rms
%%   L2 norm:                     STATS = sumsq → inv_norm = 1/sqrt(sumsq + eps)
%%                                APPLY = v * inv_norm
%%   Group norm:                  same as layer norm but per-group (deferred to
%%                                B2-Norm-Extended; treated like layer here)

generate_kernel_norm(OpKind, _Dim, Affine, KernelAST) :-
    member(OpKind, [ggml_norm, ggml_l2_norm, ggml_rms_norm, ggml_group_norm]),
    norm_kernel_name(OpKind, KernelName),
    norm_stats_decls(OpKind, StatsDecls),
    norm_stats_update(OpKind, StatsUpdate),
    norm_stats_finalize(OpKind, StatsFinalize),
    norm_apply_expr(OpKind, Affine, ApplyExpr),
    BaseParams = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'N'),
        param(c_type(int), outer),
        param(c_type(float), eps)
    ],
    ( Affine == true
    -> AffineParams = [
           param(c_type(const_ptr(c_type(float))), 'W'),
           param(c_type(const_ptr(c_type(float))), 'B')
       ],
       append(BaseParams, AffineParams, Params)
    ;  Params = BaseParams
    ),
    %% Pass 1: stats compute
    Pass1 = c_for(
        c_decl_init(c_type(int), i, c_int(0)),
        c_binop('<', c_var(i), c_var('N')),
        c_unop('++', c_var(i)),
        [
            c_decl_init(c_type(float), v,
                c_index(c_var('X'),
                    c_binop('+',
                        c_binop('*', c_var(o), c_var('N')),
                        c_var(i)))),
            StatsUpdate
        ]
    ),
    %% Pass 2: apply
    Pass2 = c_for(
        c_decl_init(c_type(int), i, c_int(0)),
        c_binop('<', c_var(i), c_var('N')),
        c_unop('++', c_var(i)),
        [
            c_decl_init(c_type(int), idx,
                c_binop('+',
                    c_binop('*', c_var(o), c_var('N')),
                    c_var(i))),
            c_decl_init(c_type(float), v, c_index(c_var('X'), c_var(idx))),
            c_assign(c_index(c_var('Y'), c_var(idx)), ApplyExpr)
        ]
    ),
    %% Compose body
    Header = [
        c_comment('=== normalization kernel (two-pass) ==='),
        c_decl_init(c_type(int), o,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('>=', c_var(o), c_var(outer)), [c_return_void]),
        c_blank,
        c_comment('Pass 1: accumulate statistics')
    ],
    Middle = [c_blank, c_comment('Finalize statistics'), c_blank],
    Tail = [c_blank, c_comment('Pass 2: write normalized output')],
    append(Header, StatsDecls, B1),
    append(B1, [Pass1], B2),
    append(B2, Middle, B3),
    append(B3, StatsFinalize, B4),
    append(B4, Tail, B5),
    append(B5, [Pass2], Body),
    KernelAST = c_func(['__global__'], c_type(void), KernelName, Params, Body).

%% Kernel naming per family member
norm_kernel_name(ggml_norm,       norm_layer).
norm_kernel_name(ggml_l2_norm,    norm_l2).
norm_kernel_name(ggml_rms_norm,   norm_rms).
norm_kernel_name(ggml_group_norm, norm_group).

%% Statistics-pass declarations: what local floats to init
norm_stats_decls(ggml_norm,
    [c_decl_init(c_type(float), sum, c_float(0.0)),
     c_decl_init(c_type(float), sumsq, c_float(0.0))]).
norm_stats_decls(ggml_l2_norm,
    [c_decl_init(c_type(float), sumsq, c_float(0.0))]).
norm_stats_decls(ggml_rms_norm,
    [c_decl_init(c_type(float), sumsq, c_float(0.0))]).
norm_stats_decls(ggml_group_norm,
    [c_decl_init(c_type(float), sum, c_float(0.0)),
     c_decl_init(c_type(float), sumsq, c_float(0.0))]).

%% Statistics update inside the pass-1 loop (uses local `v`)
norm_stats_update(ggml_norm,
    c_block([
        c_assign(c_var(sum), c_binop('+', c_var(sum), c_var(v))),
        c_assign(c_var(sumsq), c_binop('+', c_var(sumsq),
            c_binop('*', c_var(v), c_var(v))))
    ])).
norm_stats_update(ggml_l2_norm,
    c_assign(c_var(sumsq), c_binop('+', c_var(sumsq),
        c_binop('*', c_var(v), c_var(v))))).
norm_stats_update(ggml_rms_norm,
    c_assign(c_var(sumsq), c_binop('+', c_var(sumsq),
        c_binop('*', c_var(v), c_var(v))))).
norm_stats_update(ggml_group_norm, U) :- norm_stats_update(ggml_norm, U).

%% Statistics finalization: compute the normalization factor
%%   layer:  mean = sum/N, var = sumsq/N - mean², inv_std = 1/sqrt(var + eps)
%%   rms:    inv_rms = 1/sqrt(sumsq/N + eps)
%%   l2:     inv_norm = 1/sqrt(sumsq + eps)
norm_stats_finalize(ggml_norm,
    [c_decl_init(c_type(float), mean,
        c_binop('/', c_var(sum), c_cast(c_type(float), c_var('N')))),
     c_decl_init(c_type(float), var,
        c_binop('-',
            c_binop('/', c_var(sumsq), c_cast(c_type(float), c_var('N'))),
            c_binop('*', c_var(mean), c_var(mean)))),
     c_decl_init(c_type(float), inv_std,
        c_binop('/', c_float(1.0),
            c_call(sqrtf, [c_binop('+', c_var(var), c_var(eps))])))]).
norm_stats_finalize(ggml_l2_norm,
    [c_decl_init(c_type(float), inv_norm,
        c_binop('/', c_float(1.0),
            c_call(sqrtf, [c_binop('+', c_var(sumsq), c_var(eps))])))]).
norm_stats_finalize(ggml_rms_norm,
    [c_decl_init(c_type(float), inv_rms,
        c_binop('/', c_float(1.0),
            c_call(sqrtf,
                [c_binop('+',
                    c_binop('/', c_var(sumsq), c_cast(c_type(float), c_var('N'))),
                    c_var(eps))])))]).
norm_stats_finalize(ggml_group_norm, F) :- norm_stats_finalize(ggml_norm, F).

%% Apply expression: how to compute Y[idx] from v + computed stats
%% Per Tier 2 plan 8d65ba1c subtask 2g triage 2026-05-19 ~22:45 UTC:
%% substrate-historical rms/l2 affine omitted the +B[i] term — bug fixed.
%% All affine variants now use the same shape: normed * W[i] + B[i].
%%
%%   layer (no affine):  (v - mean) * inv_std
%%   layer (affine):     (v - mean) * inv_std * W[i] + B[i]
%%   rms (no affine):    v * inv_rms
%%   rms (affine):       v * inv_rms * W[i] + B[i]     [was: v * inv_rms * W[i]]
%%   l2 (no affine):     v * inv_norm
%%   l2 (affine):        v * inv_norm * W[i] + B[i]    [was: v * inv_norm * W[i]]
norm_apply_expr(ggml_norm, false,
    c_binop('*',
        c_paren(c_binop('-', c_var(v), c_var(mean))),
        c_var(inv_std))).
norm_apply_expr(ggml_norm, true,
    c_binop('+',
        c_binop('*',
            c_binop('*',
                c_paren(c_binop('-', c_var(v), c_var(mean))),
                c_var(inv_std)),
            c_index(c_var('W'), c_var(i))),
        c_index(c_var('B'), c_var(i)))).
norm_apply_expr(ggml_l2_norm, false,
    c_binop('*', c_var(v), c_var(inv_norm))).
norm_apply_expr(ggml_l2_norm, true,
    c_binop('+',
        c_binop('*',
            c_binop('*', c_var(v), c_var(inv_norm)),
            c_index(c_var('W'), c_var(i))),
        c_index(c_var('B'), c_var(i)))).
norm_apply_expr(ggml_rms_norm, false,
    c_binop('*', c_var(v), c_var(inv_rms))).
norm_apply_expr(ggml_rms_norm, true,
    c_binop('+',
        c_binop('*',
            c_binop('*', c_var(v), c_var(inv_rms)),
            c_index(c_var('W'), c_var(i))),
        c_index(c_var('B'), c_var(i)))).
norm_apply_expr(ggml_group_norm, A, E) :- norm_apply_expr(ggml_norm, A, E).

%% ═════════════════════════════════════════════════════════════════════
%% Family 5: Losses (element-wise op then reduction)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Per mavchin's note: "Loss functions — compose from existing templates."
%% Each loss is an element-wise computation between two (or three) tensors
%% followed by a per-batch reduction.
%%
%% generate_kernel_loss(+OpKind, +ReductionMode, +Params, -KernelAST)
%%
%% OpKind ∈ {ggml_mse_loss, ggml_cross_entropy_loss, ggml_huber_loss,
%%           ggml_kl_div_loss, ggml_hinge_loss, ggml_triplet_margin_loss}
%% ReductionMode ∈ {mean, sum}  — how to reduce per-batch
%% Params: loss-specific hyperparameters (e.g., huber_delta, margin)
%% KernelAST: c_func/5 AST
%%
%% Template (two-input losses):
%%
%%   __global__ void loss_<KIND>(const float * X, const float * Y, float * Out,
%%                                int N, int outer, ...config) {
%%     int b = blockIdx.x * blockDim.x + threadIdx.x;
%%     if (b >= outer) return;
%%     float acc = 0.0;
%%     for (int i = 0; i < N; i++) {
%%       int idx = b * N + i;
%%       float x_i = X[idx], y_i = Y[idx];
%%       acc += <ELEMENT_OP>(x_i, y_i);
%%     }
%%     Out[b] = <REDUCE>(acc, N);
%%   }
%%
%% TripletMargin is special (three inputs) — separate clause.

%% Standard two-input losses
generate_kernel_loss(OpKind, ReductionMode, _Params, KernelAST) :-
    member(OpKind, [ggml_mse_loss, ggml_cross_entropy_loss, ggml_huber_loss,
                    ggml_kl_div_loss, ggml_hinge_loss]),
    !,
    loss_kernel_name(OpKind, KernelName),
    loss_element_expr(OpKind, ElementExpr),
    loss_extra_params(OpKind, ExtraParams),
    loss_reduce_finalize(ReductionMode, ReduceFinalize),
    BaseParams = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(const_ptr(c_type(float))), 'Y'),
        param(c_type(ptr(c_type(float))), 'Out'),
        param(c_type(int), 'N'),
        param(c_type(int), outer)
    ],
    append(BaseParams, ExtraParams, Params),
    Body = [
        c_comment('=== loss kernel: element-wise op then per-batch reduction ==='),
        c_decl_init(c_type(int), b,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('>=', c_var(b), c_var(outer)), [c_return_void]),
        c_blank,
        c_decl_init(c_type(float), acc, c_float(0.0)),
        c_for(
            c_decl_init(c_type(int), i, c_int(0)),
            c_binop('<', c_var(i), c_var('N')),
            c_unop('++', c_var(i)),
            [
                c_decl_init(c_type(int), idx,
                    c_binop('+',
                        c_binop('*', c_var(b), c_var('N')),
                        c_var(i))),
                c_decl_init(c_type(float), x_i, c_index(c_var('X'), c_var(idx))),
                c_decl_init(c_type(float), y_i, c_index(c_var('Y'), c_var(idx))),
                c_assign(c_var(acc), c_binop('+', c_var(acc), ElementExpr))
            ]
        ),
        c_assign(c_index(c_var('Out'), c_var(b)), ReduceFinalize)
    ],
    KernelAST = c_func(['__global__'], c_type(void), KernelName, Params, Body).

%% Triplet margin loss (three inputs)
generate_kernel_loss(ggml_triplet_margin_loss, _ReductionMode, _Params, KernelAST) :-
    Params = [
        param(c_type(const_ptr(c_type(float))), anchor),
        param(c_type(const_ptr(c_type(float))), positive),
        param(c_type(const_ptr(c_type(float))), negative),
        param(c_type(ptr(c_type(float))), 'Out'),
        param(c_type(int), 'N'),
        param(c_type(int), outer),
        param(c_type(float), margin)
    ],
    Body = [
        c_comment('=== triplet margin loss: max(0, ||a-p||^2 - ||a-n||^2 + margin) ==='),
        c_decl_init(c_type(int), b,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('>=', c_var(b), c_var(outer)), [c_return_void]),
        c_blank,
        c_decl_init(c_type(float), d_ap, c_float(0.0)),
        c_decl_init(c_type(float), d_an, c_float(0.0)),
        c_for(
            c_decl_init(c_type(int), i, c_int(0)),
            c_binop('<', c_var(i), c_var('N')),
            c_unop('++', c_var(i)),
            [
                c_decl_init(c_type(int), idx,
                    c_binop('+',
                        c_binop('*', c_var(b), c_var('N')),
                        c_var(i))),
                c_decl_init(c_type(float), a_i, c_index(c_var(anchor), c_var(idx))),
                c_decl_init(c_type(float), p_i, c_index(c_var(positive), c_var(idx))),
                c_decl_init(c_type(float), n_i, c_index(c_var(negative), c_var(idx))),
                c_decl_init(c_type(float), diff_ap,
                    c_binop('-', c_var(a_i), c_var(p_i))),
                c_decl_init(c_type(float), diff_an,
                    c_binop('-', c_var(a_i), c_var(n_i))),
                c_assign(c_var(d_ap),
                    c_binop('+', c_var(d_ap),
                        c_binop('*', c_var(diff_ap), c_var(diff_ap)))),
                c_assign(c_var(d_an),
                    c_binop('+', c_var(d_an),
                        c_binop('*', c_var(diff_an), c_var(diff_an))))
            ]
        ),
        c_assign(c_index(c_var('Out'), c_var(b)),
            c_call(fmaxf, [c_float(0.0),
                c_binop('+',
                    c_binop('-', c_var(d_ap), c_var(d_an)),
                    c_var(margin))]))
    ],
    KernelAST = c_func(['__global__'], c_type(void), loss_triplet_margin, Params, Body).

%% loss_kernel_name(+OpKind, -Name)
loss_kernel_name(ggml_mse_loss,           loss_mse).
loss_kernel_name(ggml_cross_entropy_loss, loss_crossentropy).
loss_kernel_name(ggml_huber_loss,         loss_huber).
loss_kernel_name(ggml_kl_div_loss,        loss_kldiv).
loss_kernel_name(ggml_hinge_loss,         loss_hinge).

%% loss_extra_params(+OpKind, -Params)
%%   Some losses take a hyperparameter (delta for huber, etc.)
loss_extra_params(ggml_huber_loss, [param(c_type(float), huber_delta)]) :- !.
loss_extra_params(_, []).

%% loss_reduce_finalize(+ReductionMode, -FinalizeExpr)
%%   mean: acc / N
%%   sum:  acc
loss_reduce_finalize(mean,
    c_binop('/', c_var(acc), c_cast(c_type(float), c_var('N')))) :- !.
loss_reduce_finalize(sum, c_var(acc)).

%% loss_element_expr(+OpKind, -ElementExpr)
%%   Per-element contribution to the accumulator. Uses local vars
%%   x_i, y_i which are bound inside the inner loop.
loss_element_expr(ggml_mse_loss,
    c_binop('*',
        c_paren(c_binop('-', c_var(x_i), c_var(y_i))),
        c_paren(c_binop('-', c_var(x_i), c_var(y_i))))).

loss_element_expr(ggml_cross_entropy_loss,
    %% -y_i * log(x_i + 1e-12) — assumes x is softmax output
    c_unop('-',
        c_binop('*', c_var(y_i),
            c_call(logf, [c_binop('+', c_var(x_i), c_float(1.0e-12))])))).

loss_element_expr(ggml_huber_loss,
    %% |x-y| < delta ? 0.5*(x-y)² : delta*(|x-y| - 0.5*delta)
    %% Per Tier 2 plan 8d65ba1c subtask 2g triage 2026-05-19 ~22:50 UTC:
    %% wrap the ternary in c_paren so 'acc + <ternary>' doesn't get parsed
    %% as '(acc + cond) ? then : else' due to + binding tighter than <.
    c_paren(c_ternary(
        c_binop('<',
            c_call(fabsf, [c_binop('-', c_var(x_i), c_var(y_i))]),
            c_var(huber_delta)),
        c_binop('*', c_float(0.5),
            c_binop('*',
                c_paren(c_binop('-', c_var(x_i), c_var(y_i))),
                c_paren(c_binop('-', c_var(x_i), c_var(y_i))))),
        c_binop('*', c_var(huber_delta),
            c_paren(c_binop('-',
                c_call(fabsf, [c_binop('-', c_var(x_i), c_var(y_i))]),
                c_binop('*', c_float(0.5), c_var(huber_delta)))))))).

loss_element_expr(ggml_kl_div_loss,
    %% y * (log(y + ε) - log(x + ε))
    c_binop('*', c_var(y_i),
        c_paren(c_binop('-',
            c_call(logf, [c_binop('+', c_var(y_i), c_float(1.0e-12))]),
            c_call(logf, [c_binop('+', c_var(x_i), c_float(1.0e-12))]))))).

loss_element_expr(ggml_hinge_loss,
    %% max(0, 1 - y * x)  (assumes y ∈ {-1, +1})
    c_call(fmaxf, [c_float(0.0),
        c_binop('-', c_float(1.0),
            c_binop('*', c_var(y_i), c_var(x_i)))])).

%% ═════════════════════════════════════════════════════════════════════
%% Family 3: Pooling (window-reduce, parameterized by Dim and PoolKind)
%% ═════════════════════════════════════════════════════════════════════
%%
%% generate_kernel_pool(+OpKind, +Dim, +PoolKind, +Params, -KernelAST)
%%
%% OpKind ∈ {ggml_pool_1d, ggml_pool_2d, ggml_pool_3d}
%% Dim ∈ {1, 2, 3}
%% PoolKind ∈ {max, avg}
%% Params: pool_params(KernelSize, Stride, Padding)
%% KernelAST: c_func/5 AST
%%
%% Template (2D forward, max pool):
%%
%%   __global__ void pool_2d_max(const float * X, float * Y,
%%                                int B, int C, int inH, int inW,
%%                                int outH, int outW,
%%                                int kH, int kW, int stride_h, int stride_w,
%%                                int pad_h, int pad_w) {
%%     int b = blockIdx.z;
%%     int c = blockIdx.y;
%%     int oh = blockIdx.x / outW;
%%     int ow = blockIdx.x % outW;       // or via threadIdx
%%     if (b >= B || c >= C) return;
%%     float acc = -INF;                  // max kind
%%     for (int kh = 0; kh < kH; kh++) {
%%       for (int kw = 0; kw < kW; kw++) {
%%         int ih = oh * stride_h - pad_h + kh;
%%         int iw = ow * stride_w - pad_w + kw;
%%         if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
%%           float v = X[((b * C + c) * inH + ih) * inW + iw];
%%           acc = max(acc, v);
%%         }
%%       }
%%     }
%%     Y[((b * C + c) * outH + oh) * outW + ow] = acc;
%%   }
%%
%% Same template handles avg by changing init/accumulate/finalize. Same
%% parametric pattern as the reduction family.
%%
%% This commit implements Dim=2 forward fully; Dim 1 and 3 are skeletons
%% (dispatch-reachable, full body in B2-Pool-Extended).

%% 2D pool — full body
generate_kernel_pool(ggml_pool_2d, 2, PoolKind, _Params, KernelAST) :-
    !,
    member(PoolKind, [max, avg]),
    pool_kernel_name(ggml_pool_2d, PoolKind, KernelName),
    pool_init_value(PoolKind, InitValue),
    pool_accumulate_stmt(PoolKind, AccumStmt),
    pool_finalize_expr(PoolKind, FinalizeExpr),
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'B'),
        param(c_type(int), 'C'),
        param(c_type(int), 'inH'),
        param(c_type(int), 'inW'),
        param(c_type(int), 'outH'),
        param(c_type(int), 'outW'),
        param(c_type(int), 'kH'),
        param(c_type(int), 'kW'),
        param(c_type(int), stride_h),
        param(c_type(int), stride_w),
        param(c_type(int), pad_h),
        param(c_type(int), pad_w)
    ],
    Body = [
        c_comment('=== 2D pooling kernel (window-reduce) ==='),
        c_comment('Thread layout: blockIdx.z=batch, blockIdx.y=channel, x covers (outH * outW)'),
        c_decl_init(c_type(int), b, c_member(c_var(blockIdx), z)),
        c_decl_init(c_type(int), c, c_member(c_var(blockIdx), y)),
        c_decl_init(c_type(int), out_pos,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_decl_init(c_type(int), oh,
            c_binop('/', c_var(out_pos), c_var('outW'))),
        c_decl_init(c_type(int), ow,
            c_binop('%', c_var(out_pos), c_var('outW'))),
        c_if(c_binop('||',
                c_binop('||',
                    c_binop('>=', c_var(b), c_var('B')),
                    c_binop('>=', c_var(c), c_var('C'))),
                c_binop('>=', c_var(oh), c_var('outH'))),
             [c_return_void]),
        c_blank,
        c_decl_init(c_type(float), acc, InitValue),
        c_decl_init(c_type(int), count, c_int(0)),
        c_comment('Iterate over pool window (kH, kW) with bounds checking'),
        c_for(
            c_decl_init(c_type(int), kh, c_int(0)),
            c_binop('<', c_var(kh), c_var('kH')),
            c_unop('++', c_var(kh)),
            [
                c_for(
                    c_decl_init(c_type(int), kw, c_int(0)),
                    c_binop('<', c_var(kw), c_var('kW')),
                    c_unop('++', c_var(kw)),
                    [
                        c_decl_init(c_type(int), ih,
                            c_binop('+',
                                c_binop('-',
                                    c_binop('*', c_var(oh), c_var(stride_h)),
                                    c_var(pad_h)),
                                c_var(kh))),
                        c_decl_init(c_type(int), iw,
                            c_binop('+',
                                c_binop('-',
                                    c_binop('*', c_var(ow), c_var(stride_w)),
                                    c_var(pad_w)),
                                c_var(kw))),
                        c_if(c_binop('&&',
                                c_binop('&&',
                                    c_binop('>=', c_var(ih), c_int(0)),
                                    c_binop('<', c_var(ih), c_var('inH'))),
                                c_binop('&&',
                                    c_binop('>=', c_var(iw), c_int(0)),
                                    c_binop('<', c_var(iw), c_var('inW')))),
                             [
                                 c_decl_init(c_type(float), v,
                                     c_index(c_var('X'),
                                         c_binop('+',
                                             c_binop('*',
                                                 c_paren(c_binop('+',
                                                     c_binop('*',
                                                         c_paren(c_binop('+',
                                                             c_binop('*', c_var(b), c_var('C')),
                                                             c_var(c))),
                                                         c_var('inH')),
                                                     c_var(ih))),
                                                 c_var('inW')),
                                             c_var(iw)))),
                                 AccumStmt,
                                 c_expr_stmt(c_unop('++', c_var(count)))
                             ])
                    ]
                )
            ]
        ),
        c_blank,
        c_comment('Write output Y[((b*C+c)*outH+oh)*outW+ow] = finalize(acc)'),
        c_assign(
            c_index(c_var('Y'),
                c_binop('+',
                    c_binop('*',
                        c_paren(c_binop('+',
                            c_binop('*',
                                c_paren(c_binop('+',
                                    c_binop('*', c_var(b), c_var('C')),
                                    c_var(c))),
                                c_var('outH')),
                            c_var(oh))),
                        c_var('outW')),
                    c_var(ow))),
            FinalizeExpr)
    ],
    KernelAST = c_func(['__global__'], c_type(void), KernelName, Params, Body).

%% 1D pool — skeleton (full body in B2-Pool-Extended)
generate_kernel_pool(ggml_pool_1d, 1, PoolKind, _Params, KernelAST) :-
    !,
    member(PoolKind, [max, avg]),
    pool_kernel_name(ggml_pool_1d, PoolKind, KernelName),
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'B'),
        param(c_type(int), 'C'),
        param(c_type(int), 'inL'),
        param(c_type(int), 'outL'),
        param(c_type(int), 'kL'),
        param(c_type(int), stride),
        param(c_type(int), pad)
    ],
    Body = [
        c_comment('=== 1D pooling kernel skeleton (B2-Pool-Extended: full body) ==='),
        c_decl_init(c_type(int), b, c_member(c_var(blockIdx), z)),
        c_decl_init(c_type(int), c, c_member(c_var(blockIdx), y)),
        c_decl_init(c_type(int), ol,
            c_binop('+',
                c_binop('*', c_member(c_var(blockIdx), x), c_member(c_var(blockDim), x)),
                c_member(c_var(threadIdx), x))),
        c_if(c_binop('>=', c_var(ol), c_var('outL')), [c_return_void])
    ],
    KernelAST = c_func(['__global__'], c_type(void), KernelName, Params, Body).

%% 3D pool — skeleton (full body in B2-Pool-Extended)
generate_kernel_pool(ggml_pool_3d, 3, PoolKind, _Params, KernelAST) :-
    member(PoolKind, [max, avg]),
    pool_kernel_name(ggml_pool_3d, PoolKind, KernelName),
    Params = [
        param(c_type(const_ptr(c_type(float))), 'X'),
        param(c_type(ptr(c_type(float))), 'Y'),
        param(c_type(int), 'B'),
        param(c_type(int), 'C'),
        param(c_type(int), 'inD'),
        param(c_type(int), 'inH'),
        param(c_type(int), 'inW'),
        param(c_type(int), 'outD'),
        param(c_type(int), 'outH'),
        param(c_type(int), 'outW'),
        param(c_type(int), 'kD'),
        param(c_type(int), 'kH'),
        param(c_type(int), 'kW')
    ],
    Body = [
        c_comment('=== 3D pooling kernel skeleton (B2-Pool-Extended: full body) ==='),
        c_comment('3D thread layout deferred; 2D version is the substantive case for L1')
    ],
    KernelAST = c_func(['__global__'], c_type(void), KernelName, Params, Body).

%% pool_kernel_name(+OpKind, +PoolKind, -Name)
pool_kernel_name(ggml_pool_1d, max, pool_1d_max).
pool_kernel_name(ggml_pool_1d, avg, pool_1d_avg).
pool_kernel_name(ggml_pool_2d, max, pool_2d_max).
pool_kernel_name(ggml_pool_2d, avg, pool_2d_avg).
pool_kernel_name(ggml_pool_3d, max, pool_3d_max).
pool_kernel_name(ggml_pool_3d, avg, pool_3d_avg).

%% pool_init_value(+PoolKind, -InitValue)
pool_init_value(max, c_float(-1.0e30)).
pool_init_value(avg, c_float(0.0)).

%% pool_accumulate_stmt(+PoolKind, -AccumStmt)
%% Each step inside the window loop; uses local `v` (the input element)
pool_accumulate_stmt(max,
    c_assign(c_var(acc),
        c_ternary(c_binop('>', c_var(v), c_var(acc)),
                  c_var(v), c_var(acc)))).
pool_accumulate_stmt(avg,
    c_assign(c_var(acc), c_binop('+', c_var(acc), c_var(v)))).

%% pool_finalize_expr(+PoolKind, -FinalizeExpr)
%% Used at the output write; for max it's just acc, for avg it's acc/count
pool_finalize_expr(max, c_var(acc)).
pool_finalize_expr(avg,
    c_binop('/', c_var(acc), c_cast(c_type(float), c_var(count)))).
