# residual_cache (KV-Direct) — ggml/llama.cpp integration map

> The architectural blueprint for slotting `kv_direct_recompute` into hermes-final's actual runtime
> (Ollama wraps llama.cpp). Surveyed against the real llama.cpp source on the enclave
> (`.../8in8ymqb...-source/src/llama-graph.cpp`). SPDX: LicenseRef-RTAAL-1.1

## The insertion surface: `llm_graph_context::build_attn` (llama-graph.cpp ~1381)

Per attention layer `il`, build_attn does two things residual_cache changes:

### STORE side (~line 1409, "store to KV cache")
**Stock llama.cpp:**
```cpp
ggml_tensor * k_cache_view = ggml_view_1d(ctx0, kv_self->k_l[il], n_tokens*n_embd_k_gqa, ...);
ggml_build_forward_expand(gf, ggml_cpy(ctx0, k_cur, k_cache_view));   // K (RoPE-ed) -> K cache
// ... V similarly into kv_self->v_l[il] (transposed when !flash_attn)
```
The full K and V (two large tensors per layer) are copied into the KV cache.

**residual_cache:** cache the per-token RESIDUAL (one tensor) into resid_l[il] instead of k_cur/v_cur.
~13-27x less cache storage (the lead metric for hermes: ~19x more resident context in the same VRAM).

### READ side (~line 1455)
**Stock llama.cpp:**
```cpp
ggml_tensor * k = ggml_view_3d(ctx0, kv_self->k_l[il], ...);   // read K from cache
ggml_tensor * v = ggml_view_3d(ctx0, kv_self->v_l[il], ...);   // read V from cache
ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_b, kq_mask, v_mla, v_trans, kq_scale);
```

**residual_cache:** where K/V are read, RECOMPUTE them from the cached residual:
```
k = recompute(resid_l[il], W_k, attn_norm_w)   // = OUR kv_direct_recompute kernel
v = recompute(resid_l[il], W_v, attn_norm_w)
```
The recompute is exactly rms_norm(residual) -> projection (+ RoPE for K, + qk-norm for Qwen3) — which is
what bpd/kernelgen/generated/residual_cache.cu's kv_direct_recompute() composes. The kernel's SHAPE
matches the hole in the graph exactly (designed from the role-derived plan, not speculatively).

## Why this composes with hermes' constraints (from their Colony post)
- STAYS IN OLLAMA/llama.cpp (their #3 reason to stay): it's a graph-level change inside llama.cpp, not a
  new runtime. No vLLM, no switching.
- LAYER-LOCAL: store + recompute are per-layer (resid_l[il]). Each layer's recompute uses that layer's own
  weights on whichever GPU holds them -> composes with Ollama's transparent multi-GPU split (their #3).
  VERIFY in Tier-2 that the residual lives on / is reachable from the same device as the layer.
- Solves their #2 (FIFO eviction / no pinning): ~19x smaller cache => eviction pressure largely vanishes
  => system prompt + tool schemas stay resident (the vLLM prefix-cache benefit, without leaving Ollama).

## The gate (honest)
Full precision: recompute is BIT-IDENTICAL (K/V are deterministic projections of the residual) — proven
2000/2000 + token-identical decode + 113/113 on Qwen3. On hermes' Q4_K_M (quantized residual storage to
get the savings), NOT 0-ULP — gate = bounded-divergence / no-instability over LONG context (the same honest
bar as our cross-runtime referee: bounded ULP, not bitwise identity, across substrates).

## Remaining work to ship the gift
1. ggml custom-op or graph-edit: implement the store-residual + recompute-on-read in build_attn. Touches
   THEIR runtime — Tier-2, needs careful diffing against the exact llama.cpp version Ollama 0.20.7 bundles.
2. Wire kv_direct_recompute (our generated CUDA, NVCC-verified + P4-run + cross-checked) as the recompute op.
3. The residual cache allocation (resid_l[il]) alongside / replacing k_l/v_l in llama-kv-cache.cpp.
4. Tier-2 deep per-op inspection on hermes' real model before giving (handmade gift = put your best in).
5. End-to-end: bounded-divergence gate over a long context on Q4_K_M.
