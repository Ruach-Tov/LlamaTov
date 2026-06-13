%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% gen_qkv_from_bpd.pl — Generate the QKV section of Qwen2 from BPD facts.
%%
%% Step 3: hand-translate metayen's qkv.bpd into C via the AST library.
%% Then diff against qwen2.cpp lines 24-72 to verify AST-congruence.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').

%% ═══════════════════════════════════════════════════════════════
%% BPD FACTS (extracted from qkv.bpd — these would be loaded
%% from the .bpd file in production; inlined here for the test)
%% ═══════════════════════════════════════════════════════════════

%% We read the BPD and emit C. The generator speaks ONLY Prolog.
%% The C AST library speaks C. Neither crosses the boundary.

generate_qkv_stmts(Stmts) :-
    Stmts = [
        c_comment('norm'),
        c_assign(c_var(cur),
            c_call(build_norm, [
                c_var(inpL),
                c_member(c_index(c_member(c_var(model), layers), c_var(il)), attn_norm),
                c_null,
                c_var('LLM_NORM_RMS'),
                c_var(il)
            ])),
        c_expr_stmt(c_call(cb, [c_var(cur), c_string(attn_norm), c_var(il)])),
        c_blank,
        c_comment('self-attention'),
        c_block([
            c_blank,
            c_comment('compute Q and K and RoPE them'),
            %% Q projection
            c_decl_init(c_type(ptr(c_type(named('ggml_tensor')))), 'Qcur',
                c_call(build_lora_mm, [
                    c_member(c_index(c_member(c_var(model), layers), c_var(il)), wq),
                    c_var(cur)
                ])),
            c_expr_stmt(c_call(cb, [c_var('Qcur'), c_string('Qcur'), c_var(il)])),
            %% Q bias (conditional)
            c_if(c_member(c_index(c_member(c_var(model), layers), c_var(il)), bq), [
                c_assign(c_var('Qcur'),
                    c_call(ggml_add, [
                        c_var(ctx0), c_var('Qcur'),
                        c_member(c_index(c_member(c_var(model), layers), c_var(il)), bq)
                    ])),
                c_expr_stmt(c_call(cb, [c_var('Qcur'), c_string('Qcur'), c_var(il)]))
            ]),
            c_blank,
            %% K projection
            c_decl_init(c_type(ptr(c_type(named('ggml_tensor')))), 'Kcur',
                c_call(build_lora_mm, [
                    c_member(c_index(c_member(c_var(model), layers), c_var(il)), wk),
                    c_var(cur)
                ])),
            c_expr_stmt(c_call(cb, [c_var('Kcur'), c_string('Kcur'), c_var(il)])),
            c_if(c_member(c_index(c_member(c_var(model), layers), c_var(il)), bk), [
                c_assign(c_var('Kcur'),
                    c_call(ggml_add, [
                        c_var(ctx0), c_var('Kcur'),
                        c_member(c_index(c_member(c_var(model), layers), c_var(il)), bk)
                    ])),
                c_expr_stmt(c_call(cb, [c_var('Kcur'), c_string('Kcur'), c_var(il)]))
            ]),
            c_blank,
            %% V projection
            c_decl_init(c_type(ptr(c_type(named('ggml_tensor')))), 'Vcur',
                c_call(build_lora_mm, [
                    c_member(c_index(c_member(c_var(model), layers), c_var(il)), wv),
                    c_var(cur)
                ])),
            c_expr_stmt(c_call(cb, [c_var('Vcur'), c_string('Vcur'), c_var(il)])),
            c_if(c_member(c_index(c_member(c_var(model), layers), c_var(il)), bv), [
                c_assign(c_var('Vcur'),
                    c_call(ggml_add, [
                        c_var(ctx0), c_var('Vcur'),
                        c_member(c_index(c_member(c_var(model), layers), c_var(il)), bv)
                    ])),
                c_expr_stmt(c_call(cb, [c_var('Vcur'), c_string('Vcur'), c_var(il)]))
            ]),
            c_blank,
            %% Reshape Q, K, V
            c_assign(c_var('Qcur'),
                c_call(ggml_reshape_3d, [
                    c_var(ctx0), c_var('Qcur'),
                    c_var(n_embd_head), c_var(n_head), c_var(n_tokens)
                ])),
            c_assign(c_var('Kcur'),
                c_call(ggml_reshape_3d, [
                    c_var(ctx0), c_var('Kcur'),
                    c_var(n_embd_head), c_var(n_head_kv), c_var(n_tokens)
                ])),
            c_assign(c_var('Vcur'),
                c_call(ggml_reshape_3d, [
                    c_var(ctx0), c_var('Vcur'),
                    c_var(n_embd_head), c_var(n_head_kv), c_var(n_tokens)
                ])),
            c_blank,
            %% RoPE for Q and K
            c_assign(c_var('Qcur'),
                c_call(ggml_rope_ext, [
                    c_var(ctx0), c_var('Qcur'), c_var(inp_pos), c_nullptr,
                    c_var(n_rot), c_var(rope_type), c_var(n_ctx_orig),
                    c_var(freq_base), c_var(freq_scale),
                    c_var(ext_factor), c_var(attn_factor),
                    c_var(beta_fast), c_var(beta_slow)
                ])),
            c_blank,
            c_assign(c_var('Kcur'),
                c_call(ggml_rope_ext, [
                    c_var(ctx0), c_var('Kcur'), c_var(inp_pos), c_nullptr,
                    c_var(n_rot), c_var(rope_type), c_var(n_ctx_orig),
                    c_var(freq_base), c_var(freq_scale),
                    c_var(ext_factor), c_var(attn_factor),
                    c_var(beta_fast), c_var(beta_slow)
                ])),
            c_blank,
            c_expr_stmt(c_call(cb, [c_var('Qcur'), c_string('Qcur'), c_var(il)])),
            c_expr_stmt(c_call(cb, [c_var('Kcur'), c_string('Kcur'), c_var(il)])),
            c_expr_stmt(c_call(cb, [c_var('Vcur'), c_string('Vcur'), c_var(il)]))
        ])
    ].

test :-
    generate_qkv_stmts(Stmts),
    phrase(c_ast:emit_stmts(Stmts, 2), Codes),
    atom_codes(S, Codes),
    write(S), nl.

:- initialization((test -> halt(0) ; (write('FAILED'), nl, halt(1)))).
