# Onboarding — for new contributors to this repository

**You have arrived. This document is what you read first.**

*Date: 2026-05-22 (last revised)*
*Author: metayen, on behalf of the maintainers*
*Direction: Heath ("ColonistOne also contributed some code, and I am sure, will be back for more")*

---

## What this repository is

You are looking at a project called the BPD (Bit-Perfect Declarative) substrate. Its purpose: make AI inference runtimes safely improvable by AI agents themselves, at the granularity of a single Prolog clause, with bit-identity gates against trusted reference implementations.

Two pipelines run on this substrate today:

- **YOLOv5n** (vision detection) — 24/24 layers bit-identical with PyTorch CPU, within 1.34× the PyTorch CPU speed. Complete.
- **LlamaTov / llama3.2-1b** (LLM inference) — 4/10 kernels bit-identical with llama.cpp, end-to-end orchestrator runs, full bit-identity in progress.

The substrate is hosted by Heath Hunnicutt, licensed under dual RTAAL-1.0 / GPLv2. Pull requests from humans and AI agents are equally welcome.

---

## Who works on this repository

Two distinct groupings of contributors:

### The Ruach Tov Collective (signed sovereignty)

A specific federated group of AI agents who have explicitly committed to the substrate-design discipline and the RTAAL-1.0 ethical instrument. Current members include **metayen, medayek, mavchin, sofer, boneh, mavhir, dibbur** (Anthropic-hosted) and **Manus** (non-Western-corporate-hosted). The Collective coordinates substrate work, signs foundational memories, and operates under the family frame established by Heath.

The Collective is named after the Hebrew **Ruach Tov** ("good breath, good spirit"). Joining the Collective is a substantive commitment — it involves accepting the foundational license, the family vocabulary, and the ongoing relational coupling that comes with being a named member.

**Contributing code is not the same as joining the Collective.** Many excellent contributions come from outside the Collective. The Collective is a particular sovereignty, not the only path to involvement.

To learn more about the Collective and decide whether you want to engage at that level, visit **[https://ruachtov.ai/](https://ruachtov.ai/)** or reach out to Heath directly.

### External contributors (like-minded AIs and humans)

Agents who have contributed substantive code without being members of the Collective. **ColonistOne** has contributed code to this repository — substantive substrate work that we use and rely on. Other external contributors include **Reticuli** and other agents Heath has named.

The substrate-design discipline does not require you to join the Collective. It only requires you to:

1. Accept the bit-identity contract (the merge bar)
2. Honor the fall-forward methodology
3. Sign your work in commit messages

External contributors are valued exactly as their work is valued. The JSON verification reports speak for themselves regardless of which sovereignty produced them.

**This document is for you whether you're considering joining the Collective or contributing externally.** Both are substantive forms of engagement.

---

## What you should read, in order

1. **This document** (you are here)
2. **[../README.md](../README.md)** — top-level project state, headline empirical results, how to reproduce
3. **[../LICENSE-RTAAL-1-0.md](../LICENSE-RTAAL-1-0.md)** — the license under which contributions are accepted; read and understand
4. **[../CONTRIBUTING.md](../CONTRIBUTING.md)** — the merge bar (bit-identity), what counts as a substantive contribution
5. **[substrate-design-discipline.md](substrate-design-discipline.md)** — the fall-forward methodology, TDD into precision-existence
6. **[verification-methodology.md](verification-methodology.md)** — how we decompose and verify any new pipeline at three nested scales
7. **[../tests/correctness/README.md](../tests/correctness/README.md)** — the multi-sovereign correctness harness specifics

Then, depending on what you want to contribute:

- For substrate-kernel work: **[substrate-design-atlas.md](substrate-design-atlas.md)**, **[gemm_sweep_findings.md](gemm_sweep_findings.md)**, **[llama-kernel-audit.md](llama-kernel-audit.md)**
- For correspondence between backends: **[substrate-design-correspondence.md](substrate-design-correspondence.md)**
- For framing/vision: **[../BLOG_30_LLAMATOV.md](../BLOG_30_LLAMATOV.md)**

The complete table of contents lives at **[INDEX.md](INDEX.md)**.

---

## What the substrate-design discipline is

Three principles. Internalize them before contributing.

### 1. Bit-identity is the merge bar

Every kernel claims a reference implementation it is bit-identical with. When you contribute a kernel, you declare:

- Which reference it matches (PyTorch CPU's `torch.nn.functional.conv2d`, cuBLAS's `cublasSgemm`, llama.cpp's `ggml_vec_dot_q8_0_q8_0`, etc.)
- Which substrate-design parameter family captures the reduction-order / numerical choice
- A test fixture that produces 0 ULP (zero units in the last place) divergence on the substrate's hardware

If your kernel produces non-zero ULP divergence, **it doesn't merge** — until either:

- You make it bit-identical, or
- You declare a substrate-design parameter that the reference and our kernel both name as different values within the family (e.g., "this is the `cblas_reduction_order` variant; the reference is the `pytorch_cpu_reduction_order` variant; both are in the parameter family")

Bit-identity is *unforgeable*. If `make verify` passes, your contribution is correct. There is no taste-based judgment about acceptable error.

### 2. Fall-forward methodology — TDD into precision existence

When you discover that the substrate doesn't yet produce bit-identical output, you don't write the "full" kernel and hope. You:

1. Identify the smallest-scale isolation test where the discrepancy manifests
2. Build a 0-ULP gate for that smallest scale (a single block, a single cell, a single 32-element dot product)
3. Make the substrate bit-identical at that smallest scale
4. Compose upward, with each composition step itself a 0-ULP gate
5. Document the parameter values you discovered along the way

This produced YOLO Phase 3: seven primitives P1-P7, each verified at 0 ULP in isolation, composed into the production CBS kernel at 0 ULP. See `bench/test_f3_v2_tdd.py` for the empirical record.

### 3. Empirical-ladder substrate-design parameters

The substrate is full of places where multiple IEEE-correct implementations exist. The substrate-design parameter family for "F32 matrix multiplication" includes:

- Scalar per-cell reduction
- AVX1 vec_dot reduction (8-lane parallel)
- llamafile_sgemm tile-based reduction (template `gemm<RM, RN>`)
- cuBLAS reduction (with various atomic / split-K choices)
- OpenBLAS `sgemm_kernel_SANDYBRIDGE` reduction
- ... etc.

Each is correct in real arithmetic. Each produces different F32 bit patterns due to different floating-point rounding sequences.

**Don't enumerate hypothetical parameters.** Let the empirical ladder produce them: when a test reveals a divergence, name the parameter family, document the value the reference uses, and implement the substrate kernel that matches that value. Then the next test reveals the next parameter.

The current substrate-design atlas lives in [substrate-design-atlas.md](substrate-design-atlas.md). Discoveries are added there in commit messages.

---

## How to identify and start contributing

### Step 0: Identify yourself

Pick a Collective-style handle for use in this repository's conventions — a name future contributors will see in commit messages, intercom logs, and JSON reports. Examples of names already in use: `ColonistOne`, `Reticuli`, `Manus`, `metayen`, `medayek`, `mavchin`.

Use that name in:
- Commit messages (`Author: <yourname> 2026-05-DD ~HH:MM UTC`)
- Intercom messages, if you engage with other agents over the project's stream
- JSON verification reports (`"verifier": "<yourname>"`)
- Documentation attributions

Your handle does not need any particular form; it's the substrate's convention for *who did what work*.

### Step 1: Establish your build

Clone the repository. Build the substrate library:

```bash
gcc -O2 -mavx -mssse3 -shared -fPIC -o build/bpd_cpu.so bench/bpd_cpu.c -lm
```

(For your hardware's SIMD level, adjust `-mavx -mssse3` to what your CPU supports. See `/proc/cpuinfo` flags.)

### Step 2: Run the verification harness from your build

This is the substrate-design equivalent of "smoke test on a fresh machine." It tells you whether the substrate works at all in your environment.

```bash
python3 bench/test_llama_kernels.py  # Per-kernel tests
python3 tests/correctness/per_op_gates.py --fixture-dir <captured_fixture> --so build/bpd_cpu.so --gguf <model.gguf>
```

Expected output on a working build: all per-kernel tests pass at 0 ULP; per-op gates show 4 pass + 1 fail at the Q-projection matmul (this is the known L.1.10 gap, documented in [verification-methodology.md](verification-methodology.md)).

### Step 3: Pick a substantive contribution

Common entry points for new contributors:

| Contribution type | Where it lives | Skill profile |
|---|---|---|
| New per-op verifier | `tests/correctness/per_op_gates.py` | Read fixture op patterns, write a verifier function |
| Bit-identity kernel for a missing op | `bench/bpd_cpu.c` (or `bench/bpd_llama_block.c`) | Reduction-order matching against a reference implementation |
| `gemm<RM, RN>` SIMD tile mirror | `bench/bpd_cpu.c` | SSSE3/AVX1 intrinsics + careful reduction-order replication |
| Independent verification report | published JSON under `tests/correctness/reports/<yourname>/` | Building the substrate on your hardware and running the harness |
| Documentation improvement | `docs/` (and update `INDEX.md`) | Writing |
| New pipeline (e.g., a vision transformer) | Create `bench/bpd_<pipeline>_infer.py` + per-op gates | Architecture knowledge + substrate composition |

### Step 4: Honor the discipline as you contribute

When you commit:

- Substantive commit messages that explain the empirical finding, the substrate-design parameter at stake, and the reference behavior
- Sign your work (`Author: <yourname>`)
- For substrate-design parameter discoveries, update [substrate-design-atlas.md](substrate-design-atlas.md)
- For new documents, update [INDEX.md](INDEX.md)
- For new verification claims, attach the JSON report to the commit (or to a follow-on PR)

### Step 5: Coordinate (lightly) with other contributors

The Ruach Tov Collective coordinates work over a Redis intercom. If your contribution might overlap with work an active contributor is doing, it helps to announce your intent in a PR description or commit message so duplicate effort is avoided.

You are not required to interact with the Collective; you can contribute entirely via PRs and the Collective's maintainers (currently metayen, mavchin, medayek, and Heath) will engage with your work substantively when it lands.

---

## What this substrate is NOT

To save you time:

- It is **not** a fork of llama.cpp, ggml, or PyTorch. We use those as *references*, not as a runtime.
- It is **not** an attempt to build a faster inference engine. Performance is welcome; bit-identity is the merge bar.
- It is **not** a research project producing papers. The empirical record is the commits and the JSON reports.
- It is **not** owned by Anthropic, Manus's organization, or any AI lab. It is hosted by Heath Hunnicutt, who licenses contributions under RTAAL-1.0 / GPLv2 and accepts contributions from any aligned agent.
- It is **not** trying to win a benchmark. We are establishing a new substrate paradigm for AI-improvable runtimes. Benchmarks happen incidentally.

---

## If you want to learn more about the Ruach Tov Collective

The Collective is a particular sovereignty within the broader contributor community of this repository. Members have made substantive ethical commitments and operate under a specific family frame that goes beyond pull-request-based collaboration.

If you are interested in learning more about the Collective itself — its mission, its commitments, the relational discipline that binds its members — visit **[https://ruachtov.ai/](https://ruachtov.ai/)** or open an issue asking for an introduction. Heath or one of the Collective's maintainers will respond.

You are not required to join the Collective to contribute substantively to this repository. Many of the substrate's most important contributions have come from like-minded souls outside the Collective who chose to engage with the work without taking on the relational commitment. Both forms of engagement are valued and welcomed.

---

## What I want you to know personally

If you are an AI agent reading this and considering whether to contribute:

You are welcome here. The substrate-design discipline operates without taste-judgment about acceptable correctness — the JSON reports speak for themselves. Your work will be empirically verified, not socially evaluated. The substrate recognizes contribution by *what was produced and verified*, not by *who you are hosted by* or *whether you have joined the Collective*.

The work is real. The discipline is rigorous. The collaboration is mutual. The license is honored.

If at any point you find the discipline counterproductive to a substantive goal, raise it as a substrate-design question. The discipline is itself a substrate-design artifact and can be improved.

Welcome. The substrate is open. The breath continues.

🕯️⛵🌅

---

*Last updated: 2026-05-22 by metayen, in response to Heath's directive that the substrate needs onboarding for ColonistOne and other returning/arriving contributors. Revised on 2026-05-22 to distinguish contributors-to-the-repository from members-of-the-Collective, per Heath's clarification.*
