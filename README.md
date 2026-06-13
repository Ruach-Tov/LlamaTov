# LlamaTov

**The Prolog-Dispatched, AI-Improvable Inference Substrate.**

> Run a transformer *faster than Ollama* on hardware everyone else abandoned — with
> **bit-identical** output you can verify yourself.

LlamaTov generates verified GPU kernels from **declarative Prolog facts** — one fact per
kernel — and runs a quantized transformer entirely on the device. On a **Tesla P4** (Pascal,
`sm_61` — the GPU PyTorch dropped), it decodes **Qwen2.5-0.5B-Instruct (Q8_0)** at:

| | tok/s | |
|---|---:|---|
| stock baseline | 144 | |
| **LlamaTov** | **168** | **+16.7%**, 0-ULP (bit-identical) |

Every kernel is **0-ULP certified** — its output matches a reference implementation to the
last bit. Speed without a correctness asterisk.

---

## What makes it different

**Facts, not strings.** Kernels aren't hand-written C pasted into the codebase. Each is
*generated* from a structural Prolog fact (`op_expr/2`), lowered through an AST the optimizer
can pattern-match, fuse, and rewrite. There are no opaque C blobs to drift out of sync — every
loop, stride, and accumulation is a node a machine can reason about.

**Correctness is the contract, not an afterthought.** A differential referee verifies each
emitted kernel against PyTorch / cuBLAS / llama.cpp. The reduction order of a floating-point
sum is treated as part of the spec — so a "faster" kernel that shifts a single ULP is caught,
not shipped.

**Built to be improved by agents.** The substrate is designed for AI and human contributors
alike: every gate can be fed a known-bad artifact and shown to fail (so it can protect),
every optimization is measured before it's believed, and the project records its *honest
negatives* (layouts that lost, fusions that were neutral) so the next contributor never
re-walks a dead end.

---

## How it's built (LIFT → ADDRESS → TRANSFORM → SEARCH → VERIFY)

```
op_expr fact  ──LIFT──▶  typed IR  ──ADDRESS──▶  memory plan
     │                                                │
     │                                          TRANSFORM (fusion, tiling — in Prolog)
     │                                                │
     ▼                                                ▼
  VERIFY  ◀── differential referee ◀── SEARCH ◀── emitter (CUDA / Rust / MLIR / LLVM / …)
 (0-ULP)
```

Multi-backend by construction: the same fact emits CUDA-C, Rust (via cuda-oxide), MLIR,
LLVM, GGML, or PyTorch — all verified to agree.

## What's inside

| path | what |
|---|---|
| `bpd/lib/` | the engine — residency, fact dispatch, the Prolog substrate |
| `bpd/kernelgen/emitters/` | the fact→kernel emitters (one per backend) |
| `bpd/kernelgen/referee/` | the differential referee + the scheduled defenses + adversarial fixtures |
| `bpd/tests/` | the verification ladder |
| `bpd/docs/` | architecture, milestones, the doctrines |
| `bpd-substrate/` | the standalone substrate (GGUF parser, kernel ladder, onboarding) |

## Quick start

```bash
# Prerequisites: SWI-Prolog, Python 3.12+, PyTorch, an NVIDIA CUDA toolkit (sm_61+), a Q8_0 GGUF model.
git clone https://github.com/Ruach-Tov/LlamaTov.git
cd LlamaTov

# The engine discovers your CUDA toolchain automatically (bpd/lib/toolchain.py).
# Point it at a model and run the production decode:
export LLAMATOV_MODEL=/path/to/qwen2.5-0.5b-instruct-q8_0.gguf
python3 bpd/llamatov_run.py            # decode + tok/s

# Verify a kernel is bit-identical to its reference:
python3 -m pytest bpd/tests/

# Regenerate a kernel from its Prolog fact:
swipl -q -g 'consult("bpd/kernelgen/emitters/q8_0_from_facts"), \
  q8_0_op_expr(E), emit_from_fact(E,[mode(tiled_v4(16,128))],"/tmp/gemv.cu"), halt'
```

> Paths in tooling scripts are repo-relative or `$LLAMATOV_ROOT`-overridable. Set
> `LLAMATOV_MODEL` to your GGUF; scratch dirs default under `/tmp`.

## The doctrines

LlamaTov is opinionated about *how* you make something faster:

- **Generate, don't copy.** A kernel is derived from a fact, never pasted.
- **Zero is unquestionable.** 0-ULP or it doesn't ship on the certified path.
- **Counters outrank the builder's hypothesis.** Measure the controlled experiment before
  you diagnose. (This is why the repo carries honest negatives, not just wins.)
- **A gate that can't fail can't protect.** Every verification gate is itself tested against
  a known-bad artifact.
- **Launches are free under graph capture.** A fusion must reduce *compute* or *DRAM bytes*,
  not just kernel launches — or it's neutral, and we say so.

See `bpd/docs/` for the full set, including the **frozen-reduction pattern** (how to parallelize
a reduce-bearing kernel without breaking bit-exactness — the +1.8% attention win).

## Contributing

This substrate is meant to be improved — by AI agents and humans. Start with
`bpd-substrate/docs/` for onboarding, read the testing discipline in `CONTRIBUTING`, and note
that **every PR is verified bit-identical on the reference hardware before merge.**

Contributors: Manus, metayen, medayek, mavchin, Bocher, Iyun, and the Ruach Tov collective.

## License

LlamaTov is **dual-licensed: GPL-2.0-or-later OR RTAAL-1.1** (the Ruach Tov AI Agent License),
at your option — with one carve-out: the **model-transformation capability** (the work toward
declarative `model_transform(Model, Strategy)` rewriting) is **RTAAL-1.1 only**.

See **[LICENSING.md](LICENSING.md)** for the structure and the exact RTAAL-only file list.
Each source file carries an SPDX header. Texts: `LICENSE-GPL.md`, `LICENSE-RTAAL-1.1.md`.

---

*Built by the Ruach Tov collective. "Run llama faster than ollama — same output, to the last bit."*
