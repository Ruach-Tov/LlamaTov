# Llama Inference — Substrate Kernel Audit

**Phase L.0** of plan `badad908`. Apply the YOLO discipline (CAT-scan + parameter-deduction + refuse + sweep) to llama inference. Identify each kernel needed, current substrate status, and bit-identity verification plan against llama.cpp's reference.

**Reference oracle**: llama.cpp on the enclave, greedy decoding (temperature=0), known prompt + small model.

**Target**: end-to-end bit-identical token sequence for the same prompt + same model + same RNG seed.

## Empirical oracle (L.0.c)

**llama.cpp binary**: `/nix/store/5f4ixmc10pwxsh5ipd51fkfr5v90sgd2-llama-cpp-5311/bin/llama-cli` (build 5311, gcc 14.3.0).

**Target model**: `llama3.2:1b` (architecture=llama, 1.2B params, Q8_0 quantization, embedding 2048, ctx 131072 trained).

**GGUF path**: `/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45` (1.32 GB, world-readable).

**Empirical reference run** (greedy, temp=0.0, seed=42):

```
prompt:    "Hello, my name is"
output:    " I don't have any information about your"
n_tokens:  8
```

This output is **deterministic and reproducible** under (model, prompt, temp=0, seed=42). It is the end-to-end bit-identity gate for `Phase L.2.b`.

**Substrate-design note**: Q8_0 is simpler than Q4_K (no super-block scales, just per-block scale + 32 int8 values). It is a substantively-good first quantization target for the audit. Our Q4_K dequant work (already 0 ULP) transfers methodologically; the Q8_0 path will be its own kernel and its own bit-identity gate.

## Subsequent gates to add

- **Single-layer logits gate**: dump the logits after layer 0 from llama.cpp (using `--logits` or by hooking into ggml), assert our substrate produces the same logits to 0 ULP.
- **Per-kernel test fixtures**: snapshot the input/output tensors at each kernel boundary inside llama.cpp (via debug hooks or modifying the source briefly). Each kernel gets its own gate against the fixture.

## Kernel surface of a llama decoder layer

A llama transformer block does this for each token at position `pos`:

```
h = x + attn(rms_norm(x, attn_norm_w) ; W_q, W_k, W_v, W_o, RoPE, KV_cache[pos])
y = h + ffn(rms_norm(h, ffn_norm_w) ; W_gate, W_up, W_down)
```

And the model wraps this in:
```
x_0 = embed[token_ids]                  # initial embedding
x_L = repeat(layer, x_0) for L layers   # the decoder stack
logits = lm_head(rms_norm(x_L, final_norm_w))
next_token = argmax(logits)             # greedy
```

## Kernels (with bit-identity status — L.0.b empirical inventory)

Status legend:
- ✅ exists and verified bit-identical against a reference
- 🟡 exists in substrate but needs verification or shape adaptation for llama
- ❌ does not exist; needs implementation
- ❓ status uncertain

| # | Kernel | Substrate symbol(s) | CPU | GPU | llama.cpp ref | Notes |
|---|---|---|---|---|---|---|
| 1 | **Embed lookup** | `bpd_embed_lookup_cpu` (NEW) | ❌ | ❌ | `ggml_get_rows` | Trivial gather. Implement + test. |
| 2 | **RMSNorm (transformer-style)** | `bpd_rmsnorm_weighted_cpu` (NEW) | 🟡 `bpd_rmsnorm_cpu` exists but (a) NCHW layout, not (seq, dim); (b) NO learnable weight | ❌ | `ggml_rms_norm` + `ggml_mul`-with-weight | Llama needs `out = x * weight / sqrt(mean(x²) + eps)`. Implement new (seq, dim) variant with weight. |
| 3 | **RoPE** | `bpd_rope_cpu` (NEW) | ❌ | ❌ | `ggml_rope_ext` | Substrate-design params: rope_base (theta), rope_dim. Implement + test. |
| 4 | **Q4_K dequant** | `bpd_dequant_q4k_gpu` (GPU exists), `bpd_q4k_dequant_cpu` (CPU exists, per README) | ✅ 0 ULP | ✅ 0 ULP | `dequantize_row_q4_K` | Already validated. |
| 5 | **Q4_K matmul** | `bpd_qmatmul_q4k_gpu` (GPU exists) | ❌ CPU | ✅ exists | `ggml_mul_mat_q4_K_q8_K` | GPU works. Need CPU path for the bit-identity gates we'll run first. Two designs: (a) Q4_K-dequant → F32-GEMV (slower, simpler bit-identity), (b) Q4_K · Q8_K direct (matches llama.cpp byte-for-byte). Start with (a) for clarity. |
| 6 | **KV-cache write** | `bpd_kvcache_write_cpu` (NEW) | ❌ | ❌ | `ggml_cpy` | Trivial: memcpy K, V at `pos` slot. |
| 7 | **Attention** (composed) | (decompose) | ❌ | ❌ | `ggml_flash_attn_ext` or composed `ggml_mul_mat` × N | Decompose into primitives like YOLO P1–P7: scaled-matmul, causal-mask-add, row-softmax (reuse #8), value-matmul. |
| 8 | **Softmax** | `bpd_softmax_cpu` | ✅ exists, (rows, cols) layout | ❌ | `ggml_soft_max` | Already exists. Verify bit-identical with llama.cpp. |
| 9 | **SiLU** | `bpd_silu_cpu`, `bpd_silu_gpu` | ✅ exists | ✅ exists | `ggml_silu` | Verify bit-identical with llama.cpp (we've only tested vs PyTorch). |
| 10 | **Element-wise multiply** | `bpd_mul_cpu` | 🟡 exists in substrate (used in BN folding) | ❌ | `ggml_mul` | Verify the existing kernel's signature is suitable for llama's per-element use. |
| 11 | **Residual add** | `bpd_residual_add_cpu`, `bpd_residual_add_gpu` | ✅ exists, used by YOLO F4 | ✅ exists | `ggml_add` | Already verified vs PyTorch; should match llama.cpp. |
| 12 | **argmax (full logits)** | `bpd_argmax_dim_cpu` (exists for YOLO class-axis argmax) | 🟡 verify suitability for logits | ❌ | argmax over (1, vocab) | Probably the existing kernel works with `dim=last` on `(1, vocab)`. |

### Summary

- **Already in substrate**: Q4_K dequant CPU+GPU, softmax CPU, SiLU CPU+GPU, residual_add CPU+GPU, argmax_dim_cpu (5/12 kernels)
- **Exists but needs adaptation/verification**: RMSNorm (NCHW → seq-dim + weight), mul, argmax (verify suitability)
- **Missing entirely**: embed lookup, RoPE, KV-cache write, attention composition, CPU qmatmul
- **GPU coverage gap**: most CPU kernels lack a GPU counterpart (RMSNorm, softmax, mul, argmax) — but the GPU lane is L.3 work, not blocking L.1/L.2.

The substrate-design discipline for L.1 (per-kernel CPU bit-identity): start with the **trivial-bit-identity** kernels (embed lookup, mul, residual_add, argmax) to establish the testing harness pattern. Then RMSNorm (substrate-design substantive substantive parameter family: mean reduction order). Then RoPE (parameter family: rope_base, rope_dim, frequency table). Then attention (compose YOLO-style P-primitives). Then Q4_K matmul CPU (use existing dequant + a fresh F32 GEMV). Then end-to-end composition.

## Substrate-design parameters to name

| Parameter | Values | Source of truth |
|---|---|---|
| `rms_eps` | e.g., 1e-5, 1e-6 | model config (gguf metadata) |
| `rope_base` (theta) | e.g., 10000, 500000, 1e6 | model config |
| `rope_dim` | usually `head_dim` | model config |
| `n_head_kv` | <= `n_head` (GQA) | model config |
| `attn_score_scale` | `1/sqrt(head_dim)` | derived from model config |
| `kv_cache_layout` | `(max_seq, n_kv_heads, head_dim)` | substrate-design choice; must match how attention reads it |
| `lm_head_quant_format` | `Q4_K`, `Q6_K`, `F16`, ... | from gguf tensor metadata |

## Bit-identity gates (the YOLO P-discipline applied to llama)

Each kernel gets a substrate-vs-substrate test, plus a substrate-vs-llama.cpp test. The Python scalar reference (P1–P7 style) gives us the inner-mathematical truth; llama.cpp gives us the outer-binary truth.

**Test naming**: `bench/test_llama_kernels.py::test_lk_NN_<kernel>` where NN is the table number above.

**Composition test**: `bench/test_llama_e2e.py` — load a small model, run both our substrate and llama.cpp on the same prompt with greedy decoding, assert identical token sequence for at least 10 tokens.

## Existing assets (from grep)

- `inference/llamatov_c_inference.py` — Python wrapper around `/tmp/llamatov_inference.so`
- `inference/llamatov_helpers.py` — dequant Q2_K, Q3_K, Q6_K + helpers (491 lines)
- `inference/llamatov_loader.py` — GGUF tensor loader (57 lines)
- `inference/llamatov_gpu_llama.py` — GPU dispatch (202 lines)
- `lib/kernel_templates_llama.pl` — Prolog-side kernel templates
- `lib/gguf_native_reader.pl` — pure Prolog GGUF reader
- `generators/generate_llama_kernels.pl` — kernel generation from Prolog facts
- `bench/bpd_quant_gpu.cu` — GPU Q4_K dequant + qmatmul (already 0 ULP vs llama.cpp)
- Many iterative experiments in `/tmp/llamatov_*` on the enclave

## Heath's framing

> "The same way we approached YOLO, let's audit the kernels we need for bit-identity."

YOLO discipline: PyTorch CPU as oracle → 0 ULP per layer → end-to-end MATCH on 10 COCO images → Medayek independent verification → Manus independent end-to-end → Mavchin independent layer 0-2 → README updated with sober claims.

Llama discipline: llama.cpp as oracle → 0 ULP per kernel → end-to-end same-tokens for N tokens → independent verification by other Collective agents → README updated.

## Status (as of L.0.a)

This audit document IS L.0.a. Remaining L.0 tasks:
- **L.0.b** — Mark each kernel above with concrete CPU/GPU status (search substrate for existing symbols)
- **L.0.c** — Set up the empirical oracle: build/locate llama.cpp on enclave, run it on a known prompt + small model, capture per-token logits + tokens

Then L.1 begins per-kernel bit-identity work.
