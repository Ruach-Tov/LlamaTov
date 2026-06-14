%% Auto-derived compute graph for arch=llama, layers=2.
%% structure: {'qkv_bias': False, 'ffn_gate': False, 'attn_norm': True, 'ffn_norm': True, 'o_proj': True, 'qk_norm': False, 'moe': True, 'ssm': False, 'moe_top_k': 2, 'moe_n_experts': 8}
:- module(model_graph, [op/4]).
:- discontiguous op/4.
op(id(0,attn_norm), ggml_rms_norm, [a0_resid_in, 'blk.0.attn_norm.weight'], a0_normed).
op(id(0,q_proj), ggml_mul_mat, [a0_normed, 'blk.0.attn_q.weight'], a0_q_raw).
op(id(0,k_proj), ggml_mul_mat, [a0_normed, 'blk.0.attn_k.weight'], a0_k_raw).
op(id(0,v_proj), ggml_mul_mat, [a0_normed, 'blk.0.attn_v.weight'], a0_v_raw).
op(id(0,q_rope), ggml_rope, [a0_q_raw], a0_q_rope).
op(id(0,k_rope), ggml_rope, [a0_k_raw], a0_k_rope).
op(id(0,attn), flash_attention, [a0_q_rope, a0_k_rope, a0_v_raw], a0_attn_out).
op(id(0,o_proj), ggml_mul_mat, [a0_attn_out, 'blk.0.attn_output.weight'], a0_o_raw).
op(id(0,resid_attn), ggml_add, [a0_resid_in, a0_o_raw], a0_resid_mid).
op(id(0,ffn_norm), ggml_rms_norm, [a0_resid_mid, 'blk.0.ffn_norm.weight'], a0_ffn_normed).
op(id(0,router), ggml_mul_mat, [a0_ffn_normed, 'blk.0.ffn_gate_inp.weight'], a0_router_logits).
op(id(0,topk), ggml_top_k, [a0_router_logits, k2], a0_expert_sel).
op(id(0,gather_g_0), ggml_get_rows, ['blk.0.ffn_gate_exps.weight', a0_expert_sel], a0_wg_0).
op(id(0,gather_u_0), ggml_get_rows, ['blk.0.ffn_up_exps.weight', a0_expert_sel], a0_wu_0).
op(id(0,gather_d_0), ggml_get_rows, ['blk.0.ffn_down_exps.weight', a0_expert_sel], a0_wd_0).
op(id(0,eg_0), ggml_mul_mat, [a0_ffn_normed, a0_wg_0], a0_g_0).
op(id(0,eu_0), ggml_mul_mat, [a0_ffn_normed, a0_wu_0], a0_u_0).
op(id(0,es_0), ggml_silu, [a0_g_0], a0_ga_0).
op(id(0,em_0), ggml_mul, [a0_ga_0, a0_u_0], a0_gu_0).
op(id(0,ed_0), ggml_mul_mat, [a0_gu_0, a0_wd_0], a0_eout_0).
op(id(0,gather_g_1), ggml_get_rows, ['blk.0.ffn_gate_exps.weight', a0_expert_sel], a0_wg_1).
op(id(0,gather_u_1), ggml_get_rows, ['blk.0.ffn_up_exps.weight', a0_expert_sel], a0_wu_1).
op(id(0,gather_d_1), ggml_get_rows, ['blk.0.ffn_down_exps.weight', a0_expert_sel], a0_wd_1).
op(id(0,eg_1), ggml_mul_mat, [a0_ffn_normed, a0_wg_1], a0_g_1).
op(id(0,eu_1), ggml_mul_mat, [a0_ffn_normed, a0_wu_1], a0_u_1).
op(id(0,es_1), ggml_silu, [a0_g_1], a0_ga_1).
op(id(0,em_1), ggml_mul, [a0_ga_1, a0_u_1], a0_gu_1).
op(id(0,ed_1), ggml_mul_mat, [a0_gu_1, a0_wd_1], a0_eout_1).
op(id(0,combine), weighted_scatter_add, [a0_eout_0, a0_eout_1, a0_expert_sel], a0_down).
op(id(0,resid_ffn), ggml_add, [a0_resid_mid, a0_down], a0_resid_out).
op(id(1,attn_norm), ggml_rms_norm, [a1_resid_in, 'blk.1.attn_norm.weight'], a1_normed).
op(id(1,q_proj), ggml_mul_mat, [a1_normed, 'blk.1.attn_q.weight'], a1_q_raw).
op(id(1,k_proj), ggml_mul_mat, [a1_normed, 'blk.1.attn_k.weight'], a1_k_raw).
op(id(1,v_proj), ggml_mul_mat, [a1_normed, 'blk.1.attn_v.weight'], a1_v_raw).
op(id(1,q_rope), ggml_rope, [a1_q_raw], a1_q_rope).
op(id(1,k_rope), ggml_rope, [a1_k_raw], a1_k_rope).
op(id(1,attn), flash_attention, [a1_q_rope, a1_k_rope, a1_v_raw], a1_attn_out).
op(id(1,o_proj), ggml_mul_mat, [a1_attn_out, 'blk.1.attn_output.weight'], a1_o_raw).
op(id(1,resid_attn), ggml_add, [a1_resid_in, a1_o_raw], a1_resid_mid).
op(id(1,ffn_norm), ggml_rms_norm, [a1_resid_mid, 'blk.1.ffn_norm.weight'], a1_ffn_normed).
op(id(1,router), ggml_mul_mat, [a1_ffn_normed, 'blk.1.ffn_gate_inp.weight'], a1_router_logits).
op(id(1,topk), ggml_top_k, [a1_router_logits, k2], a1_expert_sel).
op(id(1,gather_g_0), ggml_get_rows, ['blk.1.ffn_gate_exps.weight', a1_expert_sel], a1_wg_0).
op(id(1,gather_u_0), ggml_get_rows, ['blk.1.ffn_up_exps.weight', a1_expert_sel], a1_wu_0).
op(id(1,gather_d_0), ggml_get_rows, ['blk.1.ffn_down_exps.weight', a1_expert_sel], a1_wd_0).
op(id(1,eg_0), ggml_mul_mat, [a1_ffn_normed, a1_wg_0], a1_g_0).
op(id(1,eu_0), ggml_mul_mat, [a1_ffn_normed, a1_wu_0], a1_u_0).
op(id(1,es_0), ggml_silu, [a1_g_0], a1_ga_0).
op(id(1,em_0), ggml_mul, [a1_ga_0, a1_u_0], a1_gu_0).
op(id(1,ed_0), ggml_mul_mat, [a1_gu_0, a1_wd_0], a1_eout_0).
op(id(1,gather_g_1), ggml_get_rows, ['blk.1.ffn_gate_exps.weight', a1_expert_sel], a1_wg_1).
op(id(1,gather_u_1), ggml_get_rows, ['blk.1.ffn_up_exps.weight', a1_expert_sel], a1_wu_1).
op(id(1,gather_d_1), ggml_get_rows, ['blk.1.ffn_down_exps.weight', a1_expert_sel], a1_wd_1).
op(id(1,eg_1), ggml_mul_mat, [a1_ffn_normed, a1_wg_1], a1_g_1).
op(id(1,eu_1), ggml_mul_mat, [a1_ffn_normed, a1_wu_1], a1_u_1).
op(id(1,es_1), ggml_silu, [a1_g_1], a1_ga_1).
op(id(1,em_1), ggml_mul, [a1_ga_1, a1_u_1], a1_gu_1).
op(id(1,ed_1), ggml_mul_mat, [a1_gu_1, a1_wd_1], a1_eout_1).
op(id(1,combine), weighted_scatter_add, [a1_eout_0, a1_eout_1, a1_expert_sel], a1_down).
op(id(1,resid_ffn), ggml_add, [a1_resid_mid, a1_down], a1_resid_out).
