%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% kernel_templates_moe.pl — C AST templates for MoE routing ops.
%%
%% Three new op_kinds for Mixture-of-Experts architecture:
%%   ggml_top_k           — select top-k expert indices + gate weights
%%   ggml_get_rows        — gather expert weight slice by index
%%   weighted_scatter_add — weighted combine of expert outputs
%%
%% These are the c_ast definitions that complete the MoE pipeline:
%%   gguf_to_graph.py (Iyun) → op_role/3 (Iyun) → kernel emission (boneh)
%%
%% Design: static-k enumeration. k is a compile-time constant from
%% {arch}.expert_used_count in GGUF metadata (e.g., Mixtral k=2 of 8).
%% Role inference works per-expert-slot UNCHANGED — the gather makes
%% the weight input dynamic but the op POSITION determines the role.
%%
%% Part of the LlamaTov ecosystem.
%% Author: boneh, 2026-06-14 (joint MoE frontier with Iyun)

:- module(kernel_templates_moe, [
    generate_kernel_top_k/3,
    generate_kernel_get_rows/3,
    generate_kernel_weighted_scatter_add/3
]).

:- use_module(c_ast).

%% ─── ggml_top_k ─────────────────────────────────────────────────
%%
%% Selects top-k expert indices and their softmax gate weights from
%% router logits. Data-dependent control flow — the selected experts
%% vary per token.
%%
%% Signature:
%%   void top_k(const float* router_logits, int n_experts, int k,
%%              int* expert_ids, float* expert_gates)
%%
%% Algorithm:
%%   1. Softmax over router_logits[0..n_experts-1]
%%   2. Select k largest values, store indices and weights
%%   3. Renormalize gate weights to sum to 1.0
%%
%% k is a compile-time constant from GGUF metadata.

generate_kernel_top_k(K, n_experts, KernelAST) :-
    atom_concat('top_k_', K, FuncName),
    KernelAST = c_func(
        void, FuncName,
        [ c_param(c_ptr(c_const(float)), router_logits),
          c_param(int, n_experts),
          c_param(c_ptr(int), expert_ids),
          c_param(c_ptr(float), expert_gates)
        ],
        c_block([
            %% Step 1: Softmax over router logits
            c_decl(float, max_val, c_call(c_id('-FLT_MAX'))),
            c_for(c_decl(int, i, c_int(0)),
                  c_lt(c_id(i), c_id(n_experts)),
                  c_postinc(c_id(i)),
                  c_if(c_gt(c_index(c_id(router_logits), c_id(i)), c_id(max_val)),
                       c_assign(c_id(max_val), c_index(c_id(router_logits), c_id(i))))),
            c_decl(float, sum_exp, c_float(0.0)),
            c_decl_array(float, probs, c_id(n_experts)),
            c_for(c_decl(int, i, c_int(0)),
                  c_lt(c_id(i), c_id(n_experts)),
                  c_postinc(c_id(i)),
                  c_block([
                      c_assign(c_index(c_id(probs), c_id(i)),
                               c_call(expf, [c_sub(c_index(c_id(router_logits), c_id(i)),
                                                   c_id(max_val))])),
                      c_assign(c_id(sum_exp),
                               c_add(c_id(sum_exp), c_index(c_id(probs), c_id(i))))])),
            %% Normalize
            c_for(c_decl(int, i, c_int(0)),
                  c_lt(c_id(i), c_id(n_experts)),
                  c_postinc(c_id(i)),
                  c_assign(c_index(c_id(probs), c_id(i)),
                           c_div(c_index(c_id(probs), c_id(i)), c_id(sum_exp)))),
            %% Step 2: Select top-k (insertion sort for small k)
            c_for(c_decl(int, j, c_int(0)),
                  c_lt(c_id(j), c_int(K)),
                  c_postinc(c_id(j)),
                  c_block([
                      c_decl(int, best_idx, c_int(-1)),
                      c_decl(float, best_val, c_float(-1.0)),
                      c_for(c_decl(int, i, c_int(0)),
                            c_lt(c_id(i), c_id(n_experts)),
                            c_postinc(c_id(i)),
                            c_if(c_gt(c_index(c_id(probs), c_id(i)), c_id(best_val)),
                                 c_block([
                                     c_assign(c_id(best_val), c_index(c_id(probs), c_id(i))),
                                     c_assign(c_id(best_idx), c_id(i))]))),
                      c_assign(c_index(c_id(expert_ids), c_id(j)), c_id(best_idx)),
                      c_assign(c_index(c_id(expert_gates), c_id(j)), c_id(best_val)),
                      %% Zero out selected so next iteration picks next-best
                      c_assign(c_index(c_id(probs), c_id(best_idx)), c_float(-1.0))
                  ])),
            %% Step 3: Renormalize gates to sum to 1.0
            c_decl(float, gate_sum, c_float(0.0)),
            c_for(c_decl(int, j, c_int(0)),
                  c_lt(c_id(j), c_int(K)),
                  c_postinc(c_id(j)),
                  c_assign(c_id(gate_sum),
                           c_add(c_id(gate_sum), c_index(c_id(expert_gates), c_id(j))))),
            c_for(c_decl(int, j, c_int(0)),
                  c_lt(c_id(j), c_int(K)),
                  c_postinc(c_id(j)),
                  c_assign(c_index(c_id(expert_gates), c_id(j)),
                           c_div(c_index(c_id(expert_gates), c_id(j)), c_id(gate_sum))))
        ])
    ).


%% ─── ggml_get_rows ──────────────────────────────────────────────
%%
%% Gather: select expert j's weight slice from a stacked weight tensor.
%% The stacked tensor has shape [n_experts, rows, cols]; we extract
%% the slice at expert_idx, producing a [rows, cols] output.
%%
%% Role inference note: the gathered slice lands in weight-position
%% of the downstream mul_mat → classified as PARAMETER by the
%% parameter/activation discriminator. Role-transparent for activation
%% dataflow.
%%
%% Signature:
%%   void get_rows(const float* stacked, int expert_idx,
%%                 int rows, int cols, float* out)

generate_kernel_get_rows(rows, cols, KernelAST) :-
    KernelAST = c_func(
        void, get_rows,
        [ c_param(c_ptr(c_const(float)), stacked),
          c_param(int, expert_idx),
          c_param(int, rows),
          c_param(int, cols),
          c_param(c_ptr(float), out)
        ],
        c_block([
            c_decl(int, offset, c_mul(c_id(expert_idx),
                                      c_mul(c_id(rows), c_id(cols)))),
            c_for(c_decl(int, i, c_int(0)),
                  c_lt(c_id(i), c_mul(c_id(rows), c_id(cols))),
                  c_postinc(c_id(i)),
                  c_assign(c_index(c_id(out), c_id(i)),
                           c_index(c_id(stacked), c_add(c_id(offset), c_id(i)))))
        ])
    ).


%% ─── weighted_scatter_add ───────────────────────────────────────
%%
%% Weighted combine of k expert outputs: out = sum_j(gate_j * expert_out_j)
%% This is a small weighted vector accumulation over k terms (k=2-8),
%% reusing the BLAS L1 reduction pattern (scale-then-accumulate).
%%
%% Signature:
%%   void weighted_scatter_add(const float** expert_outs,
%%                             const float* gates, int k,
%%                             int vec_len, float* out)

generate_kernel_weighted_scatter_add(K, vec_len, KernelAST) :-
    atom_concat('weighted_scatter_add_k', K, FuncName),
    KernelAST = c_func(
        void, FuncName,
        [ c_param(c_ptr(c_ptr(c_const(float))), expert_outs),
          c_param(c_ptr(c_const(float)), gates),
          c_param(int, vec_len),
          c_param(c_ptr(float), out)
        ],
        c_block([
            %% Zero the output
            c_for(c_decl(int, i, c_int(0)),
                  c_lt(c_id(i), c_id(vec_len)),
                  c_postinc(c_id(i)),
                  c_assign(c_index(c_id(out), c_id(i)), c_float(0.0))),
            %% Accumulate: out[i] += gate[j] * expert_outs[j][i]
            c_for(c_decl(int, j, c_int(0)),
                  c_lt(c_id(j), c_int(K)),
                  c_postinc(c_id(j)),
                  c_block([
                      c_decl(float, g, c_index(c_id(gates), c_id(j))),
                      c_decl(c_ptr(c_const(float)), eout,
                             c_index(c_id(expert_outs), c_id(j))),
                      c_for(c_decl(int, i, c_int(0)),
                            c_lt(c_id(i), c_id(vec_len)),
                            c_postinc(c_id(i)),
                            c_assign(c_index(c_id(out), c_id(i)),
                                     c_add(c_index(c_id(out), c_id(i)),
                                           c_mul(c_id(g),
                                                 c_index(c_id(eout), c_id(i))))))
                  ]))
        ])
    ).
