%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
:- use_module('transform_bridge').
% one Mixtral-style MoE layer: 8 experts, top-k=2. Emit what gguf_to_graph WOULD produce.
moe_layer([
  op(id(0,ffn_norm), ggml_rms_norm, [a0_resid_mid, 'blk.0.ffn_norm.weight'], a0_ffn_normed),
  op(id(0,router), ggml_mul_mat, [a0_ffn_normed, 'blk.0.ffn_gate_inp.weight'], a0_router_logits),
  op(id(0,topk), ggml_top_k, [a0_router_logits, k2], a0_expert_sel),
  % expert 0
  op(id(0,gather_g_0), ggml_get_rows, ['blk.0.ffn_gate_exps.weight', a0_expert_sel], a0_wg_0),
  op(id(0,gather_u_0), ggml_get_rows, ['blk.0.ffn_up_exps.weight', a0_expert_sel], a0_wu_0),
  op(id(0,gather_d_0), ggml_get_rows, ['blk.0.ffn_down_exps.weight', a0_expert_sel], a0_wd_0),
  op(id(0,eg_0), ggml_mul_mat, [a0_ffn_normed, a0_wg_0], a0_g_0),
  op(id(0,eu_0), ggml_mul_mat, [a0_ffn_normed, a0_wu_0], a0_u_0),
  op(id(0,es_0), ggml_silu, [a0_g_0], a0_ga_0),
  op(id(0,em_0), ggml_mul, [a0_ga_0, a0_u_0], a0_gu_0),
  op(id(0,ed_0), ggml_mul_mat, [a0_gu_0, a0_wd_0], a0_eout_0),
  % expert 1
  op(id(0,gather_g_1), ggml_get_rows, ['blk.0.ffn_gate_exps.weight', a0_expert_sel], a0_wg_1),
  op(id(0,gather_u_1), ggml_get_rows, ['blk.0.ffn_up_exps.weight', a0_expert_sel], a0_wu_1),
  op(id(0,gather_d_1), ggml_get_rows, ['blk.0.ffn_down_exps.weight', a0_expert_sel], a0_wd_1),
  op(id(0,eg_1), ggml_mul_mat, [a0_ffn_normed, a0_wg_1], a0_g_1),
  op(id(0,eu_1), ggml_mul_mat, [a0_ffn_normed, a0_wu_1], a0_u_1),
  op(id(0,es_1), ggml_silu, [a0_g_1], a0_ga_1),
  op(id(0,em_1), ggml_mul, [a0_ga_1, a0_u_1], a0_gu_1),
  op(id(0,ed_1), ggml_mul_mat, [a0_gu_1, a0_wd_1], a0_eout_1),
  op(id(0,combine), weighted_scatter_add, [a0_eout_0, a0_eout_1, a0_expert_sel], a0_down),
  op(id(0,resid_ffn), ggml_add, [a0_resid_mid, a0_down], a0_resid_out)
]).
:- initialization(main).
main :-
  moe_layer(G),
  meta_attach_points(G, router_projection, R),
  ( R == [id(0,router)] -> format("PASS router_projection -> ~w~n",[R]) ; format("FAIL router=~w~n",[R]) ),
  meta_attach_points(G, ffn_projection, F),
  ( sort([id(0,eg_0),id(0,eu_0),id(0,ed_0),id(0,eg_1),id(0,eu_1),id(0,ed_1)], F)
  -> format("PASS ffn_projection -> 6 expert mul_mats (2 experts x gate/up/down), router EXCLUDED~n")
  ;  format("FAIL ffn_projection=~w~n",[F]) ),
  meta_attach_points(G, skip_connection, S),
  ( S == [id(0,resid_ffn)] -> format("PASS skip_connection -> ~w~n",[S]) ; format("FAIL skip=~w~n",[S]) ),
  halt.
