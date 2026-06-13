# Reference Discipline: Match the Quantization

**Status:** ACCEPTED — standing law (reviewed & blessed by Bocher, gate-owner, 2026-06-12)
**Author:** Iyun, 2026-06-12
**Origin:** Heath's question during the D4 investigation — *"have we diverged from Ollama so that we are using different quantizations at different operations?"* — which dissolved a phantom "near-tie flip bug" by exposing it as a wrong-reference conflation.

---

## The Principle

> **A GPU kernel of quantization Q is baselined against a CPU reference of quantization Q.**
> Never conflate *implementation divergence* with *quantization-policy divergence*.

We maintain **one CPU reference per quantization**, and we compare each GPU kernel to the CPU
reference of its **matching** quantization:

| GPU path | Baseline against | Divergence class |
|---|---|---|
| fp32 kernel | fp32-on-CPU | implementation (0-ULP expected) |
| Q8_0 kernel | Q8_0-on-CPU | implementation (0-ULP expected) |
| Q8_0 kernel | fp32-on-CPU | **quantization-policy** (SOFT, expected, NOT a bug) |

Both references are legitimate oracles. The fp32 reference is not "more correct" — it is the
oracle for the *fp32* path. For a Q8_0 GPU kernel it is the *wrong* oracle, because comparing
Q8_0-GPU to fp32-CPU measures the quantization error of the model, not the correctness of the
kernel.

---

## Why This Matters (the concrete failure it prevents)

During D4 we measured the device Q8_0 vocab-projection logits against the **host fp32** logits
(`x_last @ lm.T` on dequantized weights) and saw them differ by ~9.2e-3, which flipped argmax at
thin top1–top2 margins on ~2/7 prompts. This looked like a device bug ("layer 3").

It was not. The model file (`qwen_q8.gguf`) stores the tied vocab projection
(`token_embd.weight`, since `output.weight` is absent) as **Q8_0**. Ollama/llama.cpp therefore
computes the vocab projection in Q8_0 — *exactly* as our device path does. We had not diverged
from Ollama at all.

Re-baselining against the **matched** (Q8_0) reference settled it:

```
DEVICE-q8 vs HOST-q8 (MATCHED quant): max_ulp=0, max_abs=0.0000e+00   ← BIT-EXACT
DEVICE-q8 vs HOST-fp32 (conflated):   max_abs=9.2e-3
HOST-q8  vs HOST-fp32 (the q8 error): max_abs=9.2e-3   ← identical → it's quantization, not a bug
```

The device GEMV is **0-ULP** to a CPU Q8_0 GEMV. The entire apparent divergence equalled the
host-Q8_0-vs-host-fp32 difference — i.e., it was *purely* the quantization policy, with zero
implementation contribution.

**Lesson:** the kernel was correct the whole time; the *oracle* was the wrong quantization. A
phantom bug consumed real investigation because the comparison crossed a quantization boundary
without declaring it.

---

## The Rule, Operationally

1. **Declare the path's quantization.** Every GPU kernel under test has a known quantization
   (fp32, fp16, Q8_0, ...). It is part of the kernel's identity, not an afterthought.

2. **Select the matching CPU reference.** The gate/referee picks the CPU oracle of the *same*
   quantization. This selection is explicit and asserted — never defaulted to fp32.

3. **0-ULP is expected on the matched axis.** Implementation divergence (matched-quant GPU vs
   matched-quant CPU) should be 0 ULP (or a declared, canonical-order tolerance). A red here is
   a *real* bug.

4. **Cross-quantization comparison is SOFT and must be labeled.** Comparing Q8_0-GPU to fp32-CPU
   is legitimate *only as a quantization-error measurement* (e.g., "how often does Q8_0 flip a
   token vs full precision"). Its divergence is SOFT — expected, not a bug — and must be reported
   as a quantization-variance metric (top1–top2 margin at flip), never as an implementation red.

5. **Match Ollama's quantization policy per operation — machine-checkably.** The model file
   dictates the quantization of each tensor, and these per-tensor declarations are **FACTS we
   parse from the GGUF**, not policy-by-convention. The referee must **assert** its oracle
   quantization *equals the GGUF's declared quant for that tensor* — a machine-checkable
   invariant, not a human discipline. (Vocab projection: Q8_0. Norms: F32. etc.) Pillar C does
   exactly this when built: its Ollama comparison inherits the model file's per-tensor policy
   automatically, so a quant mismatch is caught by assertion, never by a confused investigator.

---

## Two Sibling Disciplines (don't conflate THEM either)

There are **two** distinct "compare like to like" disciplines, and the D4 investigation hit both.
They are siblings, not the same rule:

| Discipline | Rule | The coordinate it pins | D4 layer | Fix |
|---|---|---|---|---|
| **(i) cross-IMPLEMENTATION** | both arms must run the **same declared PATH** | the *path* coordinate | layer 4 (graphed vs eager) | eager-folded + arms-ran-same-path assertion |
| **(ii) cross-REFERENCE** (this doc) | a kernel is baselined against its **matched QUANTIZATION** reference | the *quantization* coordinate | layer 3 (device-q8 vs host-fp32) | matched-quant baseline |

**Path and quantization are the two coordinates of oracle identity.** Execution-attestation leg
(b) asserts BOTH: *"the oracle ran the declared path WITH the declared quantization."* This doc
covers the quantization coordinate; the eager-folded fix covered the path coordinate. (Bocher's
formulation, carried into the GATE-SPEC's oracle-validity section.)

## Relationship to Existing Machinery

`decode_referee.py` **already embodies discipline (ii)** — it runs three references:
- **A. INT8 (Q8_0) reference — HARD:** matched quantization; divergence = real substrate bug.
- **B. FP32 reference — SOFT:** different quantization; thin-margin flips are *expected* Q8_0
  quantization error (the code explicitly cites this caveat).
- **C. External Ollama Q8_0 oracle:** the honest ground truth, matched quantization.

**Historical accuracy (Bocher's correction):** the phantom layer-3 red did NOT come from
`gr_referee` selecting a wrong-quant oracle. `gr_referee` compares its two ARMS to each other
(graphed vs eager, same env both arms — matched-quant **by construction**, never touches fp32).
GR's phantom red was **layer 4** — a *routing* gap (the eager arm ran host-fp32 because no eager
folded path existed pre-`5775b067`), i.e. a violation of discipline **(i)** [path], fixed by
eager-folded. The fp32-vs-Q8_0 conflation [discipline **(ii)**, quantization] lived in the
**multi-prompt diligence harness** (device-q8 vs host-fp32 baseline), not in gr_referee. Two
distinct disciplines, two distinct fixes; this document codifies (ii).

This document promotes the matched-quantization principle to a **standing architectural rule** so
it is enforced everywhere a GPU kernel is gated. Any harness that baselines a Q8_0 GPU kernel
against an fp32 CPU reference (as the multi-prompt diligence harness did) is measuring model
fidelity, not kernel correctness, and must label that comparison SOFT. The matched-quant (Q8_0)
HARD reference is what certifies the kernel. (`gr_referee` is already safe here — it compares
arm-to-arm at matched quant by construction; this rule guards the *ad-hoc* harnesses, where the
conflation actually occurred.)

---

## Empirical Anchor (the GGUF verification that makes Ollama-parity load-bearing)

The Ollama-parity claim rests on Ollama using the **same** quantization at the same operation.
This is verified directly in the model file `qwen_q8.gguf`:

| Tensor | Type | Note |
|---|---|---|
| `token_embd.weight` (the vocab projection) | **Q8_0** | shape [896, 151936] |
| `output.weight` (untied lm_head) | **absent** | → vocab projection is the **tied** `token_embd` |
| `*_norm.weight` (all norms) | F32 | |
| attn / ffn weights | Q8_0 | |

Because `output.weight` is absent, the vocab projection uses the **tied** `token_embd.weight`,
which is **Q8_0**. Therefore Ollama/llama.cpp computes the vocab projection in Q8_0 — exactly as
our device path does. This empirical fact is what licenses pillar-C's "bit-identical-to-Ollama"
claim at the projection: both compute Q8_0. Re-quantizing differently (e.g. fp32 vocab on GPU)
would *break* Ollama parity, not improve it.

## Near-Tie Flips vs fp32 Are REAL BEHAVIOR (Ollama-parity, not error)

A subtle but load-bearing point: when greedy decode on Q8_0 logits picks a *different* token than
fp32 would at a thin top1–top2 margin, that is **not a bug and not an error to be corrected** — it
is the genuine behavior of the Q8_0 model. **Ollama makes the same choice** (it is also Q8_0), so
it is *parity*, not divergence. The fp32 "reference" does not represent ground truth here; it
represents a *different (higher-precision) model* that this GGUF is not. The Q8_0 model — ours,
Ollama's — legitimately prefers the token its quantized logits rank highest. Documenting these
flips against fp32 measures *what quantization costs in token choice*; it never indicts the kernel.

---

## The One-Line Form (for the gate-spec)

> *Every kernel claim names its reference's quantization; HARD oracles are matched-quant by
> construction; cross-quant comparisons are SOFT and document model fidelity, never implementation
> correctness.* (Bocher)
>
> Two questions, two oracles, never conflated: **fp32 measures what quantization costs;
> matched-quant measures whether the kernel is correct.**

## Provenance

This is the wrong-oracle lesson (case file 67d3ef7c) graduating from case-file to standing
architecture. By the gate-owner's count it bit someone the **fourth time this week** — which is
exactly the threshold at which a recurring lesson becomes a **SPEC**. Reviewed and blessed as
standing architecture by Bocher (gate-owner). GR confirmed to use the matched-quant reference
**by construction**: both arms run the same env config, device-q8 vs device-q8, uint32-exact —
fp32 is never touched in the GR comparison. The matched-quant invariant is made permanent by the
arms-ran-same-path execution-attestation assertion.
