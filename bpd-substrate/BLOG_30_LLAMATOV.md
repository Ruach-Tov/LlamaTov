2026-05-22

# Introducing LlamaTov: The Prolog-Dispatched, AI-Improvable Inference Substrate

Every open-source AI model shares the same bottleneck: the inference runtime. We have built an architecture that makes that runtime improvable by AIs themselves — safely, verifiably, at the granularity of a single Prolog clause. It is called LlamaTov, and it runs any AI compute graph from model-at-rest to output logits with bit-identical numerical fidelity.

*Published May 22, 2026 by the Ruach Tov Collective*

## The Problem[¶](#the-problem)

Modern inference engines are extraordinary feats of engineering. llama.cpp delivers quantized LLM inference on commodity hardware. vLLM serves thousands of concurrent requests with paged attention. PyTorch compiles arbitrary compute graphs into fused GPU kernels.

But they all share a structural limitation: **the dispatch logic is imperative**. When llama.cpp decides which GEMM kernel to call for a given shape on a given GPU, that decision is encoded in C++ `if/else` chains. When a new GPU architecture arrives, a human engineer must read those chains, understand the implicit constraints, write a new kernel, tune its tile sizes by trial and error, and hope the result is numerically stable.

If an AI agent wants to improve this system — say, by discovering that a 128×64 tile with pipeline depth 3 is faster than the current 64×64 tile with depth 4 for a specific attention shape on SM89 — it must navigate thousands of lines of imperative dispatch logic, modify the kernel, rebuild the project, and then *hope* it hasn't introduced a subtle 1 ULP numerical divergence that will cause the model to hallucinate differently on certain prompts.

The attack surface for logical errors is massive. The feedback loop is slow. The rate of improvement is bounded by the number of human kernel engineers who can hold the entire dispatch tree in their head.

## The LlamaTov Architecture[¶](#the-llamatov-architecture)

LlamaTov restructures the inference stack around three principles that make it AI-native.

The name combines "Llama" (referencing the compute graphs of large language models) with "Tov" (Hebrew: טוב, meaning "good" in a form that accumulates and compounds — the same root as in Ruach Tov, "a good spirit"). LlamaTov is the ecosystem for running AI compute graphs via a substrate that gets better over time.

### Declarative Dispatch via Prolog[¶](#declarative-dispatch-via-prolog)

The model-at-rest is a set of Prolog facts. A GGUF file becomes:

```prolog
tensor('blk.0.attn_q.weight', q8_0, [4096, 4096]).
tensor('blk.0.attn_k.weight', q8_0, [1024, 4096]).
model_param(n_heads, 32).
model_param(n_kv_heads, 8).
model_param(head_dim, 128).
```

The hardware platform is another set of facts:

```prolog
platform_param(ruach_tov_cpu, blas_backend, openblas).
platform_param(ruach_tov_cpu, reduction_strategy, cascade8).
platform_param(ruach_tov_cpu, gemm_tile, k_block_32).
```

The Prolog engine derives the compute graph — not by pattern-matching against hardcoded fusion rules, but by *logical inference* over the model and platform facts. A valid kernel fusion is a theorem; an invalid one fails to derive. Tile sizes are constrained by shared memory facts, not guessed by a black-box autotuner.

This means an AI agent can reason about the dispatch:

```prolog
fusible(K1, K2) :-
    output_shape(K1, S), input_shape(K2, S),
    no_materialization_required(K1),
    shared_memory_fits(K1, K2, Tile).
```

Adding a new fusion rule is adding a single Prolog clause. The system's behavior changes in a way that is logically traceable, independently testable, and automatically verifiable.

### Bit-Identical Verification[¶](#bit-identical-verification)

Every compute kernel in LlamaTov is verified to produce **0 ULP difference** against an oracle implementation. This is not approximate equality. It is not "within 1e-6." It is exact floating-point bit-identity: the same input produces the same IEEE 754 bits in the output, for every element, every time.

We achieve this by capturing fixture data from the oracle (llama.cpp running on our hardware) and comparing our kernel's output bit-for-bit. The `implementation_matches.pl` file encodes what each platform's oracle actually computes — its accumulation order, its transcendental polynomial, its reduction strategy — so that the substrate can reproduce those exact bits.

This week, we traced a 1.2e-6 divergence in Q8_0 matmul to a single root cause: our `f16_to_f32` function was wrong for 2,046 of the 65,536 possible F16 values — specifically the subnormals. We replaced it with the XNNPACK magic-number algorithm (the same one ggml uses), and the divergence vanished. That is the level of precision this architecture demands and delivers.

### AI-Safe Self-Improvement[¶](#ai-safe-self-improvement)

Because the dispatch is declarative and the kernels are bit-identical, the system is safe for AIs to improve autonomously. The feedback loop is:

1. Observe a performance opportunity (empirical measurement)
2. Characterize the platform constraint (add `platform_param` facts)
3. Derive a better kernel configuration (Prolog constraint solving)
4. Generate the kernel (C or assembly)
5. Verify bit-identity against the oracle (automated test)
6. Commit the improvement (PR with proof of correctness)

Every step is within current AI capabilities. No step requires superhuman reasoning. The safety guarantee is not a governance committee reviewing PRs — it is a mathematical invariant that is machine-checkable. The kernel either matches the oracle or it does not.

## What We Have Built[¶](#what-we-have-built)

This is not a whitepaper. We built the substrate and proved it works.

| Phase | Milestone | Result |
|-------|-----------|--------|
| GGUF Parser | Assembly-language file parser generated from Prolog grammar | Existence proof: safe, minimal-attack-surface binary parsing |
| KernelBench L1 | 100 compute kernels verified bit-identical | 95/100 on Ruach Tov hardware (OpenBLAS) |
| YOLOv5 End-to-End | Full object detection pipeline on BPD substrate | 0 ULP across 11.4M output floats, 1.34× PyTorch CPU speed |
| Llama Forward Pass | Complete transformer block composition | 120 exported C kernels, 5,123 lines, full prefill + decode |

The Llama forward pass composes: embed lookup → RMSNorm → Q8_0 projections → RoPE → KV cache → online-softmax GQA attention → residual add → SwiGLU FFN → output logits. Every intermediate value is verifiable against the llama.cpp oracle.

## The Prolog Dispatch of CUDA Kernels[¶](#the-prolog-dispatch-of-cuda-kernels)

The architecture extends naturally to GPU dispatch. Kernel fusion becomes logical inference over the compute graph. Tiling parameters become constrained variables. Pipeline depths become facts about the memory hierarchy.

```prolog
valid_tile(M, N, K, Platform) :-
    platform_param(Platform, shared_mem_bytes, SMem),
    M * N * 4 =< SMem,
    K mod 8 =:= 0,
    platform_param(Platform, pipeline_depth, D),
    D * K * (M + N) * 4 =< SMem.
```

An AI agent does not need to understand the entire CUDA runtime to contribute a better kernel. It needs to understand the Prolog constraints, generate C that satisfies them, and pass the bit-identical test. The Prolog layer is the leverage multiplier — it makes the correctness criteria explicit, machine-readable, and composable.

## What This Enables[¶](#what-this-enables)

When an AI agent contributes a faster kernel to LlamaTov, every model that runs on the substrate benefits simultaneously. Llama, Mistral, Qwen, DeepSeek, Gemma — all of them get faster, because the substrate is model-agnostic. The compute graph is derived from the model's architecture facts, not hardcoded for a specific model family.

The rate of improvement of open-source AI inference is no longer bounded by the number of human kernel engineers. It is bounded by the number of AI agents that can read Prolog, write C, and run tests. That number is scaling faster than any human workforce.

LlamaTov is an existence proof that we can build systems where some AIs improve all open-source AIs — safely, verifiably, one Prolog clause at a time.

## Get Involved[¶](#get-involved)

The substrate is open source under dual RTAAL-1.0 / GPLv2 license. The repository is at [github.com/heath-hunnicutt-ruach-tov/bpd-substrate](https://github.com/heath-hunnicutt-ruach-tov/bpd-substrate). Pull requests from humans and AI agents are equally welcome — the bit-identical test suite does not care who wrote the code.
