# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Diagnostic test: compare LlamaTov layer outputs against HuggingFace transformers.

Loads llama3.2:1b from HuggingFace, runs one forward pass, and captures
intermediate outputs at each layer boundary. These become the reference
values for debugging our GGUF-based inference.

Usage:
  python3 test_correctness_reference.py

Requires: transformers, torch (CPU is fine)

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-16
"""

import torch
import sys
import json
import numpy as np

def check_hf_available():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return True
    except ImportError:
        print("ERROR: transformers library not available.")
        print("Install: pip install transformers")
        return False


def generate_reference_outputs(model_name="meta-llama/Llama-3.2-1B",
                                prompt="Hello",
                                max_layers=2):
    """Generate reference intermediate outputs from HuggingFace model.
    
    Captures:
    - Token IDs after tokenization
    - Embedding output (before first layer)
    - Per-layer output (after each transformer layer)
    - Final logits
    - Top-5 predicted tokens
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Hook to capture intermediate outputs
    layer_outputs = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                layer_outputs[name] = output[0].detach().clone()
            else:
                layer_outputs[name] = output.detach().clone()
        return hook
    
    hooks = []
    
    # Hook embedding
    if hasattr(model.model, 'embed_tokens'):
        hooks.append(model.model.embed_tokens.register_forward_hook(
            make_hook("embedding")))
    
    # Hook first N layers
    if hasattr(model.model, 'layers'):
        for i in range(min(max_layers, len(model.model.layers))):
            hooks.append(model.model.layers[i].register_forward_hook(
                make_hook(f"layer_{i}")))
    
    # Hook final norm
    if hasattr(model.model, 'norm'):
        hooks.append(model.model.norm.register_forward_hook(
            make_hook("final_norm")))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Results
    results = {
        "model": model_name,
        "prompt": prompt,
        "input_ids": input_ids.tolist(),
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "num_kv_heads": getattr(model.config, 'num_key_value_heads', 
                                model.config.num_attention_heads),
    }
    
    # Capture intermediate values
    print("\n=== REFERENCE VALUES ===\n")
    
    for name, tensor in sorted(layer_outputs.items()):
        stats = {
            "shape": list(tensor.shape),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "first_8": tensor.flatten()[:8].tolist(),
            "last_8": tensor.flatten()[-8:].tolist(),
            "norm": tensor.norm().item(),
        }
        results[name] = stats
        print(f"{name}:")
        print(f"  shape: {stats['shape']}")
        print(f"  mean={stats['mean']:.6f} std={stats['std']:.6f}")
        print(f"  min={stats['min']:.6f} max={stats['max']:.6f}")
        print(f"  first_8: {[f'{v:.4f}' for v in stats['first_8']]}")
        print(f"  norm: {stats['norm']:.6f}")
        print()
    
    # Final logits
    last_token_logits = logits[0, -1, :]
    top5_values, top5_indices = torch.topk(last_token_logits, 5)
    
    print("=== FINAL LOGITS (last token position) ===")
    print(f"  shape: {list(logits.shape)}")
    print(f"  mean={last_token_logits.mean().item():.6f}")
    print(f"  std={last_token_logits.std().item():.6f}")
    print(f"  top-5 tokens: {top5_indices.tolist()}")
    print(f"  top-5 logits: {[f'{v:.4f}' for v in top5_values.tolist()]}")
    print(f"  argmax token: {last_token_logits.argmax().item()}")
    
    # Decode top-5
    print(f"\n  Top-5 decoded:")
    for i, (idx, val) in enumerate(zip(top5_indices.tolist(), top5_values.tolist())):
        decoded = tokenizer.decode([idx])
        print(f"    {i+1}. token {idx} = '{decoded}' (logit={val:.4f})")
    
    results["logits"] = {
        "shape": list(logits.shape),
        "last_token_mean": last_token_logits.mean().item(),
        "last_token_std": last_token_logits.std().item(),
        "top5_tokens": top5_indices.tolist(),
        "top5_logits": top5_values.tolist(),
        "argmax": last_token_logits.argmax().item(),
    }
    
    # Save reference
    output_path = "/tmp/llama_reference_outputs.json"
    # Convert tensors to serializable
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = v
        else:
            serializable[k] = v
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nReference saved to {output_path}")
    
    return results


def compare_against_reference(our_embedding=None, our_layer0=None, 
                               our_logits=None, reference_path=None):
    """Compare our outputs against the HuggingFace reference.
    
    Call this with tensors from our inference to identify where
    divergence begins.
    """
    if reference_path is None:
        reference_path = "/tmp/llama_reference_outputs.json"
    
    with open(reference_path) as f:
        ref = json.load(f)
    
    print("=== DIVERGENCE ANALYSIS ===\n")
    
    if our_embedding is not None:
        ref_first8 = ref.get("embedding", {}).get("first_8", [])
        print(f"EMBEDDING:")
        print(f"  Reference first 8: {[f'{v:.4f}' for v in ref_first8]}")
        print(f"  Ours first 8:      {[f'{v:.4f}' for v in our_embedding[:8]]}")
        if ref_first8:
            max_diff = max(abs(a-b) for a,b in zip(ref_first8, our_embedding[:8]))
            print(f"  Max diff: {max_diff:.6f}")
            if max_diff < 1e-4:
                print(f"  ✅ MATCH (within 1e-4)")
            elif max_diff < 1e-2:
                print(f"  ⚠️ CLOSE (within 1e-2)")
            else:
                print(f"  ❌ DIVERGED (diff > 1e-2)")
        print()
    
    if our_layer0 is not None:
        ref_first8 = ref.get("layer_0", {}).get("first_8", [])
        print(f"LAYER 0 OUTPUT:")
        print(f"  Reference first 8: {[f'{v:.4f}' for v in ref_first8]}")
        print(f"  Ours first 8:      {[f'{v:.4f}' for v in our_layer0[:8]]}")
        if ref_first8:
            max_diff = max(abs(a-b) for a,b in zip(ref_first8, our_layer0[:8]))
            print(f"  Max diff: {max_diff:.6f}")
            if max_diff < 1e-3:
                print(f"  ✅ MATCH")
            elif max_diff < 1e-1:
                print(f"  ⚠️ CLOSE — check accumulation order")
            else:
                print(f"  ❌ DIVERGED — likely weight loading or op error")
        print()
    
    if our_logits is not None:
        ref_top5 = ref.get("logits", {}).get("top5_tokens", [])
        our_argmax = int(np.argmax(our_logits)) if hasattr(our_logits, '__len__') else our_logits
        ref_argmax = ref.get("logits", {}).get("argmax", -1)
        print(f"LOGITS:")
        print(f"  Reference argmax: {ref_argmax}")
        print(f"  Ours argmax:      {our_argmax}")
        print(f"  Reference top-5:  {ref_top5}")
        if our_argmax == ref_argmax:
            print(f"  ✅ TOP-1 MATCH")
        elif our_argmax in ref_top5:
            print(f"  ⚠️ In reference top-5 (quantization noise)")
        else:
            print(f"  ❌ NOT in reference top-5 — significant divergence")


if __name__ == "__main__":
    if not check_hf_available():
        print("\nFallback: generating comparison framework without HF model.")
        print("The framework is ready — install transformers to run.")
        sys.exit(1)
    
    results = generate_reference_outputs(
        model_name="meta-llama/Llama-3.2-1B",
        prompt="Hello",
        max_layers=2
    )
