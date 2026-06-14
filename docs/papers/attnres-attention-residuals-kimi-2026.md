# AttnRes — Attention Residuals (Kimi, 2026)

**arXiv:** [2603.15031](https://arxiv.org/abs/2603.15031) · Submitted 16 Mar 2026 · cs.CL
**Archived:** `attnres-attention-residuals-kimi-2026.pdf` (verified: PDF 1.7, 9pp, 1065095 bytes)
**Referenced by:** `bpd/lib/transform_attnres.pl` (the `attnres` model transformation)

## What it is

Standard LLM residuals (with PreNorm) accumulate **all** layer outputs with **fixed unit
weights**. This uniform aggregation causes uncontrolled hidden-state growth with depth,
progressively diluting each layer's contribution.

**AttnRes** replaces the fixed accumulation with **softmax attention over preceding layer
outputs** — each layer selectively aggregates earlier representations with *learned,
input-dependent* weights.

**Block AttnRes** partitions layers into blocks and attends over block-level representations,
reducing memory/communication overhead — a practical drop-in for standard residuals.

## Why it matters for `model_transform(Model, attnres)`

- **It is an ARCHITECTURE change, not a quantization.** Unlike `kv_quantize_q8`/turboquant (which
  re-encode existing data, verifiable by a bounded error), AttnRes introduces **new learned
  parameters** (the softmax-attention weights over layer history).
- **Therefore "applying" AttnRes requires TRAINING / fine-tuning** to recover quality — there is no
  data-oblivious "apply + verify within tolerance" path. The paper integrates AttnRes into Kimi
  Linear (48B/3B-activated) and **pre-trains on 1.4T tokens**.
- The role bridge (`transform_bridge.pl`) already finds AttnRes's attach points correctly: the 48
  `role(skip_connection)` true residuals on qwen2 (the q/k/v bias adds correctly excluded). What is
  missing is (a) the `attn_residual` op's real math + kernel, and (b) the *trained* attention weights.

## Key result (from the abstract)

Scaling-law experiments show consistent improvement across model sizes; integrated into Kimi Linear
and pre-trained on 1.4T tokens, AttnRes mitigates PreNorm dilution (more uniform output magnitudes +
gradient distribution across depth) and improves downstream performance across all evaluated tasks.
