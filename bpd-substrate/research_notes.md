# Research Notes: TurboQuant and AttnRes

## TurboQuant (Google Research, ICLR 2026)

**Core idea**: Near-optimal vector quantization for KV cache compression.

**Two-stage algorithm**:
1. **PolarQuant** (the main compression): Randomly rotate input vectors (precondition with random matrix S), 
   which makes coordinates behave as i.i.d. Gaussians. Then recursively convert pairs of coordinates to 
   polar form (radius, angle). The angles concentrate tightly around π/4 due to the Gaussian property.
   This means angles can be quantized with very few bits and NO per-block scale factors needed.

2. **QJL** (residual correction): Apply 1-bit Johnson-Lindenstrauss to the quantization error.
   Zero overhead, eliminates bias in attention scores.

**Key properties for BPD transformation**:
- Applied post-training (no fine-tuning needed)
- Operates on KV cache during inference (not weights per se, but the same math applies to weights)
- The random rotation matrix S is fixed and shared — can be fused into the projection
- Replaces standard dequant with: rotate → polar → quantize angles → store
- At inference: load quantized angles → reconstruct → inverse rotate → compute attention

**As a BPD model transform**:
- Pattern: any matmul that produces K or V vectors for attention
- Transform: insert rotation (can be fused into weight matrix), replace KV storage with polar-quantized form
- The attention score computation changes: instead of Q·K^T, compute via the polar representation
- This is a SEMANTIC transform — it changes how the model stores intermediate state

## AttnRes (Kimi/Moonshot AI, March 2026, arXiv:2603.15031)

**Core idea**: Replace fixed residual connections (x + layer(x)) with learned, input-dependent 
softmax attention over ALL preceding layer outputs.

**Standard residual**: h_l = h_{l-1} + f_l(h_{l-1})
  - Fixed unit weights
  - Causes uncontrolled hidden-state growth with depth
  - Progressively dilutes each layer's contribution

**AttnRes**: h_l = softmax(q_l · [h_0, h_1, ..., h_{l-1}]^T) · [h_0, h_1, ..., h_{l-1}]
  - Learned, content-dependent aggregation over depth
  - Each layer selectively attends to earlier representations
  - Prevents dilution, more uniform gradient distribution

**Block AttnRes** (practical variant):
  - Partition L layers into N blocks
  - Attend over block-level representations (not all L layers)
  - Reduces memory from O(L) to O(N) where N << L
  - "Drop-in replacement" for standard residuals
  - ~2% memory overhead
  - 90% communication reduction vs full AttnRes

**As a BPD model transform**:
- Pattern: any sequence of transformer blocks with residual connections
- Transform: replace `add(x, layer_output)` with `attn_residual(x, [all_prev_outputs])`
- The Block variant: group layers into blocks, maintain block-level representations
- This is a STRUCTURAL transform — it changes the model's depth-wise aggregation

**Key insight for implementation**:
- AttnRes adds a small attention computation per layer (query over depth)
- But it can be FUSED with the existing attention mechanism
- The "attention over depth" is itself fusible as an elementwise-like operation
  (it's just a small softmax + weighted sum over a fixed-size set)

## Implementation Strategy

Both transforms follow the same pattern:
1. Pattern-match on a subgraph in BPD facts
2. Check preconditions (quantization type, residual structure)
3. Replace matched subgraph with transformed subgraph
4. The transformed subgraph is then subject to normal fusion optimization

The model_transform/3 framework:
```prolog
model_transform(TransformName, InputFacts, OutputFacts) :-
    transform_precondition(TransformName, InputFacts),
    transform_apply(TransformName, InputFacts, OutputFacts).
```
