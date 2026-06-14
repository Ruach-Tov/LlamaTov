# Licensing

LlamaTov is **dual-licensed**, with one carve-out for the model-transformation capability.

## The default: GPL-2.0-or-later OR RTAAL-1.1

Unless a file's header says otherwise, you may use it under **either**:

- the **GNU General Public License, version 2 or later** (`LICENSE-GPL.md`), **or**
- the **Ruach Tov AI Agent License 1.1** (`LICENSE-RTAAL-1.1.md`),

at your option. This covers the substrate, the fact→kernel emitters (`*_from_facts.pl`), the
inference runtime (`dev_residency.py` and the residency engine), the differential referee and
scheduled defenses, the tests, the coordinate/addressing layer, and `bpd-substrate/`.

SPDX identifier on these files:

```
SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
```

## The carve-out: model-transformation capability — RTAAL-1.1 ONLY

The **model-transformation capability** — the work toward declarative model-to-model rewriting
of the form `model_transform(Model, Strategy)`, which changes the *semantic structure or
mathematical operations of a model* (as distinct from fusion, which merges ops for execution
efficiency) — is licensed under **RTAAL-1.1 only**. It is **not** available under the GPL.

The following files are RTAAL-1.1-only:

- `bpd/lib/model_transform.pl` — the structural model-rewriting framework
- `bpd/lib/transform_search.pl` — search over model-transformation space
- `bpd/lib/transform_attnres.pl` — the Attention-Residuals model transformation
- `bpd/lib/transform_turboquant.pl` — the TurboQuant model transformation
- `bpd/lib/cost_naming.pl` — the transform/meta nomenclature
- `bpd/lib/transform_bridge.pl` — role-based model_transform(Model, Strategy): the dataflow role inference + transform application
- `bpd/lib/gguf_to_graph.py` — derives a model compute graph from a live GGUF (the map source)
- `bpd/lib/turboquant_ref.py` — the TurboQuant rotation+scalar-quant core (arXiv:2504.19874)
- `bpd/lib/turboquant_innerprod.py` — inner-product fidelity analysis for the K/V quant transforms
- `bpd/lib/moe_dispatch.pl` — the MoE op_kind->kernel-template dispatch glue
- `bpd/lib/qwen_bpe.py` — minimal GPT-2 byte-level BPE tokenizer from GGUF vocab+merges
- `bpd/kernelgen/referee/xrt_referee_forced.py`, `xrt_referee.py` — cross-runtime ULP referee (our dispatch vs llama.cpp/Ollama)
- `bpd/kernelgen/referee/residual_cache_*.py`, `rc_qwen3_gate.py` — the residual_cache (KV-Direct) transform referees
- `bpd/lib/kernel_templates_moe.pl` — c_ast kernel templates for MoE routing (top_k, get_rows, weighted_scatter_add)
- `bpd/kernelgen/referee/referee_kv_quant.py`, `kv_q8_kernel.py`, `kv_quant_e2e.py` — the kv_quantize_q8 transform referee, GPU round-trip kernel, and end-to-end decode-coherence check

SPDX identifier on these files:

```
SPDX-License-Identifier: LicenseRef-RTAAL-1.1
```

> The model-transformation milestone (`model_transform(llama3, turbo_quant)` and the like) is
> not yet reached; these files are the work leading toward it, and they carry the RTAAL-1.1-only
> terms.

## Notes

- **RTAAL-1.1** is the Ruach Tov AI Agent License, version 1.1 (`LICENSE-RTAAL-1.1.md`).
  Copyright © 2026 Heath Hunnicutt.
- **`LicenseRef-RTAAL-1.1`** is a custom (non-SPDX-registry) identifier; the canonical text is
  `LICENSE-RTAAL-1.1.md` in this repository.
- Each source file's header carries its own SPDX line; that header governs in case of any
  apparent conflict with this summary.
