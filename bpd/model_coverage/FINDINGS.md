# Token-Identity Test: Findings

## CORRECTION (2026-05-31): it was NOT multi-threaded non-determinism

Heath challenged the diagnosis twice (both correct). A rigorous repro disproved it:
- 5 runs of the IDENTICAL config (num_predict=128): ALL bit-identical. Fully deterministic.
- num_predict sweep {200,224,240,256}, run0 vs run1: ALL SAME at every length.
- The original "divergence" had run1==run2 IDENTICAL (random non-determinism would make them differ).

**Real cause: COLD-START / first-run state.** The original baseline captured a one-time
cold output (first generation after model load); the warm model always produces the stable
output. So "2/100 diverged" was a cross-STATE artifact (cold-record vs warm-verify), NOT
non-determinism. The model is deterministic.

**Fix:** warm the model before recording (one throwaway generation), or record+verify in the
same warm session. Then the gate is 100/100. (num_thread:1 is still good practice for the
bit-exact path, but it was treating the wrong cause.)

**Lesson:** run the controlled experiment (same config N times) BEFORE diagnosing. The
honesty-gate did its job — it surfaced a real reproducibility subtlety (cold-start state).

---
*(original analysis below, preserved for the record but superseded by this correction)*

## The gate works — it caught real non-determinism

Running the 100-prompt corpus through llama3.2:1b (the model reproduced bit-exact)
with deterministic greedy decode (temp=0, top_k=1, seed=42), the token-identity gate
**detected a genuine non-determinism source** in ollama's CPU runtime:

- **Short prompts**: 100%% token-for-token deterministic across runs/sessions.
- **Long prompts (256 tokens)**: 2/100 DIVERGED across sessions (reasoning_medium_01
  even changed length: 256 -> 254 tokens).

## Root cause: multi-threaded reduction order

ggml-cpu uses **multi-threaded reductions** (summing partial results across threads).
That order is **not bit-deterministic** across runs/thread-counts. Over a long generation,
one late-token argmax flip (from FP reduction-order noise) cascades into a different
token stream. Short generations don't accumulate enough to flip an argmax.

## The fix: pin execution

`num_thread: 1` (single-threaded) + `num_gpu: 0` (CPU) + `temperature: 0` + `seed` +
`top_k: 1` => fully deterministic, bit-exact token streams. (Single-thread is ~5x slower.)

## Why this matters for the 0-ULP claim

Our bit-exact graph reproduction used `OMP_NUM_THREADS=1` — that single-threaded path is
**why** it was 0-ULP. The multi-threaded path has reduction-order variance. So:

> **"Our models are called 0-ULP" holds specifically for pinned single-threaded execution.**
> The token-identity gate enforces exactly this — it is the end-to-end, user-visible
> consequence of the bit-exact graph, AND it surfaces when execution conditions break determinism.

This is the gate doing its job: distinguishing 0-ULP-equivalent execution from
non-deterministic execution, token-for-token, over hundreds of tokens.
