%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% terminology.pl — World-naming → canonical BPD-naming substrate.
%%
%% A foundational module for the 384-kernel curriculum (medayek's
%% framing, 2026-05-17). Different communities call the same operation
%% different things, AND call different operations the same thing.
%% The "k_gelu" name collision was the discovery instance:
%%
%%   - ggml_gelu (default in llama.cpp graph builder)      = tanh form
%%   - ggml_gelu_erf                                       = exact erf form
%%   - ggml_gelu_quick                                     = sigmoid form
%%   - F.gelu(approximate="tanh") (PyTorch)                = tanh form
%%   - F.gelu(approximate="none") (PyTorch, default)       = exact erf form
%%   - HuggingFace "gelu" (default)                        = exact erf form
%%   - HuggingFace "gelu_new"                              = tanh form
%%   - HuggingFace "gelu_fast"                             = tanh form (Mistral coeff)
%%   - HuggingFace "quick_gelu"                            = sigmoid form
%%
%% The same "k_gelu" name in different contexts means three distinct
%% mathematical functions. This module names the truth (the canonical
%% form in BPD facts) and provides terminology_change/2 facts mapping
%% the world's various names to the canonical form.
%%
%% ## Term shape
%%
%% terminology_change(SourceContext:Name, world:CanonicalName)
%%
%% Where:
%%   SourceContext  — which community / library / convention is naming
%%   Name           — the name they use
%%   world          — the canonical BPD-truth-space marker
%%   CanonicalName  — the BPD activation_expr (or kernel) name
%%
%% Example:
%%   terminology_change(ggml_default:k_gelu, world:k_gelu_tanh).
%%     ↑ "when ggml says 'gelu', they mean what BPD canonically
%%       calls 'k_gelu_tanh'"
%%
%% This module will grow as the curriculum exposes more naming
%% collisions. Per medayek's framing: this IS curriculum infrastructure,
%% not just gelu-specific.
%%
%% ## Source contexts currently named
%%
%%   ggml_default      — ggml's default per public API choice
%%   ggml_explicit     — ggml's _erf / _quick explicit variants
%%   ollama_runtime    — what Ollama executes via llama.cpp's graph builder
%%   pytorch_default   — torch.nn.functional default behavior
%%   pytorch_explicit  — torch.nn.functional with explicit kwarg
%%   huggingface_default — HuggingFace transformers' "gelu" config string
%%   huggingface_legacy — HuggingFace's older / GPT-2-era names
%%
%% Adding a new source context: just add facts; no schema migration needed.
%%
%% Author: metayen 2026-05-17
%% Per Heath's terminology_change/2 idea + medayek's curriculum
%% naming-substrate framing. First instance: gelu's three-way split.

:- module(terminology, [
    terminology_change/2,
    canonical_name/3,         % canonical_name(+SourceContext, +Name, -CanonicalName)
    name_collision/2          % name_collision(+Name, -CanonicalNames)
]).

:- discontiguous terminology_change/2.


%% ─────────────────────────────────────────────────────────────────
%% GELU FAMILY — the discovery instance
%%
%% Three distinct mathematical functions share the name "gelu":
%%   k_gelu_tanh:  0.5x · (1 + tanh(√(2/π) · x · (1 + 0.044715·x²)))
%%                 = 0.5x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
%%   k_gelu_erf:   0.5x · (1 + erf(x/√2))
%%   k_gelu_quick: x · sigmoid(1.702·x) = x · (1/(1+exp(-1.702·x)))
%%                 (also written as x/(1+exp(-1.702·x)) directly)
%%
%% These functions disagree at any nontrivial precision. They are
%% NOT alternative implementations of the same function; they are
%% three separate functions historically called "gelu" by overlapping
%% but distinct communities.
%% ─────────────────────────────────────────────────────────────────

%% ggml's gelu API (per external/llama.cpp/ggml/include/ggml.h)
terminology_change(ggml_default:k_gelu,       world:k_gelu_tanh).
terminology_change(ggml_explicit:ggml_gelu,   world:k_gelu_tanh).
terminology_change(ggml_explicit:ggml_gelu_erf,   world:k_gelu_erf).
terminology_change(ggml_explicit:ggml_gelu_quick, world:k_gelu_quick).

%% Ollama runtime = llama.cpp graph builder, which calls ggml_gelu
%% (the tanh form) at all three LLM_FFN_GELU dispatch sites in
%% external/llama.cpp/src/llama-graph.cpp. Verified by inspection
%% 2026-05-17. The _erf and _quick variants are exposed in ggml's
%% API but never invoked from llama.cpp's own dispatch.
terminology_change(ollama_runtime:k_gelu, world:k_gelu_tanh).

%% PyTorch's torch.nn.functional.gelu has `approximate` kwarg.
%% Default is 'none' (exact erf form).
terminology_change(pytorch_default:k_gelu,                      world:k_gelu_erf).
terminology_change(pytorch_explicit:'F.gelu(approximate="none")', world:k_gelu_erf).
terminology_change(pytorch_explicit:'F.gelu(approximate="tanh")', world:k_gelu_tanh).

%% HuggingFace transformers config strings. Mappings per their
%% transformers/activations.py (canonical reference for which
%% mathematical function each name denotes).
terminology_change(huggingface_default:k_gelu,             world:k_gelu_erf).
terminology_change(huggingface_legacy:gelu_new,            world:k_gelu_tanh).
terminology_change(huggingface_legacy:gelu_fast,           world:k_gelu_tanh).
terminology_change(huggingface_legacy:gelu_pytorch_tanh,   world:k_gelu_tanh).
terminology_change(huggingface_legacy:quick_gelu,          world:k_gelu_quick).


%% ─────────────────────────────────────────────────────────────────
%% INTROSPECTION PREDICATES
%% ─────────────────────────────────────────────────────────────────

%% canonical_name(+SourceContext, +Name, -CanonicalName)
%%
%% Resolve a community-name in a specific source context to its
%% canonical BPD name. Fails if the (SourceContext, Name) pair is
%% not registered — the substrate doesn't silently make up mappings.
canonical_name(SourceContext, Name, CanonicalName) :-
    terminology_change(SourceContext:Name, world:CanonicalName).


%% name_collision(+Name, -CanonicalNames)
%%
%% Given a community-level name, return all canonical names that
%% different sources mean by it. If multiple contexts disagree
%% about what the name means, this surfaces the collision.
%%
%% Example:
%%   ?- name_collision(k_gelu, Names).
%%   Names = [k_gelu_tanh, k_gelu_erf]  % at least; depends on contexts
%%
%% Substrate-honest: returns the SET of canonical names (deduplicated),
%% not the multiset.
name_collision(Name, CanonicalNames) :-
    findall(CN,
        ( terminology_change(_:Name, world:CN) ),
        Bag),
    sort(Bag, CanonicalNames).
