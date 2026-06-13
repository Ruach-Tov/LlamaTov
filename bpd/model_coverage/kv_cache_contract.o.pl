%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% AUTO-LIFTED from llama.cpp source by kv_cache_contract.py — the KV-cache contract.
%% These facts define what our launcher's cache MUST match for bit-identical operation.

kv_cache_dtype(k, 'GGML_TYPE_F16').   %% default
kv_cache_dtype(v, 'GGML_TYPE_F16').   %% default
kv_cache_alloc(k, ndim(1), type(type_k), size('n_embd_k_gqa * kv_size')).
kv_cache_alloc(v, ndim(1), type(type_v), size('n_embd_v_gqa * kv_size')).
kv_cache_dim(n_embd_k_gqa, 'head_dim * n_kv_heads').
kv_cache_write_view(dims(['n_embd_head_k', 'n_head_kv', 'size']), stride1('row_size(type, n_embd_head_k)'), stride2('row_size(type, n_embd_k_gqa)')).

%% The bit-identity claim: our launcher's cache satisfies ALL of the above + the
%% multi-turn eval-callback verification (cache_k/cache_v writes+reads 0-ULP across turns).
