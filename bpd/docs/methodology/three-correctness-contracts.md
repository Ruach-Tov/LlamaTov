# Three Correctness Contracts for the Matrix Harness

**Date**: 2026-05-17
**Originating conversation**: Heath's framing of bit-identical comparison
("we found a matrix transpose error by requiring bit-identical floating
point outputs from us running the same model as ollama") + mavchin's
empirical kernel-family boundary measurements (intercom 11:10 + 11:11 UTC).

## Why three contracts, not one

The cross-language correctness matrix verifies that the SAME kernel
produces equivalent outputs across host languages (Python, C, Rust,
Prolog). But "equivalent" is not a single question — it's three
questions stacked at different precisions:

1. **Do they compute the same FUNCTION within fp32 precision?**
   (allclose contract — tolerant of fp32 reordering)
2. **Are they within a known precision bound against a reference?**
   (bounded-ULP contract — for ops where bit-equality is physically
   impossible)
3. **Do they compute the EXACT SAME OPERATIONS on the EXACT SAME
   layout in the EXACT SAME order?**
   (bit-identical contract — uint32 XOR on the IEEE 754 bits)

Each contract catches a different class of bug. Using only the loosest
contract (allclose) hides structural errors that the stricter contracts
expose. The matrix harness supports all three.

## The transpose-error story

The most concrete demonstration of why bit-identical comparison is
essential, told by mavchin (intercom 11:11 UTC, paraphrased and
preserved):

> We were running our llama-family inference and comparing tokens to
> Ollama (llama.cpp). 5 of N tokens agreed. Decent.
>
> We then fixed GGUF Q8_0 weight loading. The previous path was
> dequantizing Q8_0 to F32 then re-quantizing — producing completely
> different int8 values and fp16 scales than what llama.cpp reads.
> The new path (gguf_q8_to_padded64_coalesced) copies blocks
> VERBATIM from GGUF. Scales and quants now identical to what
> llama.cpp reads.
>
> After the fix, our output DIVERGED MORE from Ollama, not less.
> 3 tokens matched, down from 5.
>
> The bit-identical probe revealed why: the previous (wrong)
> re-quantization had been ACCIDENTALLY COMPENSATING for a different
> accumulation order in our dp4a kernel vs llama.cpp's kernel.
> Two wrongs were canceling. Fixing one exposed the other.
>
> The specific finding when we compared the first matrix at bit
> level:
>   allclose said: PASS (max_diff=5.18e-03, well within atol=0.05)
>   bit-identical said: ALL 32 quants differ, scale differs
>
> lt() returns w[:,0] matching GGUF row 0 (correct data, transposed
> layout). Our re-quantizer read w[0,:] (wrong row, transposed
> input). Re-quantized different data → different Q8 blocks →
> different dp4a output. allclose was happy because the OUTPUT
> was still "close enough."

**The substrate's voice**: allclose said "we're close to right."
Bit-identical said "we're not actually computing what we think we're
computing." The latter was correct.

**The methodology lesson**: **two wrongs can cancel**. Allclose against
a known-correct reference is vulnerable to compensating errors that
make wrong implementations LOOK right. Only bit-identical catches the
structural divergence.

## The three contracts in detail

### Contract 1: allclose (default)

**Question**: Do the two implementations compute the same FUNCTION
within fp32 precision?

**Implementation**: `np.allclose(a, b, rtol=1e-5, atol=1e-6)`. Tolerant
of fp32 reduction-order variations. Same final value, possibly
different bit patterns.

**Catches**:
- NaN / Inf bugs
- Catastrophic scale errors (off by 1000×)
- Sign errors
- Wrong-output-on-wrong-input (when the wrong output is in a wildly
  different numerical range)

**Misses**:
- Layout/transpose errors that produce plausible but wrong outputs
- Algorithm divergence that happens to numerically align
- Compensating errors (two wrongs canceling)

**When to use**: backend-vs-backend sanity check. "Our Python and C
implementations should agree on what the function IS." Useful as a
baseline; insufficient on its own for semantic correctness.

### Contract 2: bounded-ULP (`--ulp N`)

**Question**: Is the implementation within a known precision bound
against the reference?

**Implementation**: per-element ULP (Units in the Last Place) distance,
accept if all elements have `ulp_distance ≤ N`.

ULP is the canonical fp32 precision metric: two adjacent representable
floats are 1 ULP apart, regardless of magnitude. ULP distance counts
the integer steps between bit patterns.

**Catches**:
- Same things as allclose
- AND: precision regressions specifically beyond a known bound

**Misses**:
- Order-of-magnitude bugs that happen to keep the result within bound
- Layout errors (same as allclose) that don't push results outside
  the precision envelope

**When to use**: when bit-equality is physically impossible but the
contract still wants to be tight. Per mavchin's empirical measurements:

| Family | Empirical ULP bound | Reason |
|---|---|---|
| Transcendental (silu/gelu/tanh) | ≤2 ULP | SFU precision differs CPU vs GPU |
| Normalization (rms_norm/layer_norm) | ≤13 ULP | Reduction order varies with thread count |
| Small reductions (sum_rows/mean on 16 elem) | ≤4 ULP* | PyTorch tree vs serial accumulation |

(*Measured empirically against PyTorch on Tesla P4 reduction kernels
shipped 2026-05-17.)

**The bounded-ULP contract codifies the SUBSTRATE'S PHYSICAL LIMITS.**
You can't make a GPU SFU bit-identical to a CPU transcendental. The
contract honestly says "within physical limits" rather than pretending
bit-equality is achievable.

### Contract 3: bit-identical (`--strict`)

**Question**: Do the two implementations compute the EXACT SAME
OPERATIONS in the EXACT SAME ORDER on the EXACT SAME data layout?

**Implementation**: `np.array_equal(a.view(uint32), b.view(uint32))`.
Pure structural equality at the bit level.

**Catches**:
- Same things as allclose and bounded-ULP
- AND: algorithm divergence (different reduction trees)
- AND: layout errors (transposes producing plausible outputs)
- AND: order-sensitivity bugs (warp scheduling variations)
- AND: compensating errors (two wrongs canceling)
- AND: anything else that produces different bits

**Misses**: nothing in the semantic-equivalence sense.

**When to use**:
- Replacing a reference implementation (must produce same tokens)
- Catching transpose / layout / order bugs
- Verifying kernel fusion didn't change semantics
- Establishing baseline before refactoring an op's reduction tree

**Cost**: bit-identical isn't always achievable. For
order-sensitive ops on GPU with arbitrary thread mappings, bit-equal
output requires the kernels to use the SAME ORDER, not just the same
data. Mavchin's dp4a vecmat spike is exactly this frontier:
same Q8 integers (proven), different float output from different
accumulation order.

The substrate-honest path: target bit-equality FIRST as a discipline.
Where physically impossible (SFU, reduction-order), document the
boundary and use bounded-ULP.

## Substrate-honest decision tree

For each (kernel, reference) cell in the matrix:

```
Is bit-equality physically achievable between this kernel and the reference?
  Yes  → use --strict. Document as bit-identical-target cell.
         If currently failing, the work is to align the algorithm
         (warp scheduling, reduction tree, FMA ordering, etc.).

  No   → identify the physical limit (SFU? reduction order?). Use
         --ulp N where N is the empirical bound + a small margin.
         Document the N and its physical basis.

Always also use allclose as a sanity check. allclose-MATCH is necessary
but not sufficient.
```

## Why both us-vs-us AND us-vs-reference matter

The matrix harness's correctness claims sit at two layers:

**Layer A: backend-vs-backend (US vs US).**
- Verifies: our Python, C, Rust, Prolog backends compute the same
  function.
- Tool: `matrix_verify.py --strict` between backend outputs.
- Catches: implementation divergence within our codebase.

**Layer B: backend-vs-reference (US vs OLLAMA / PYTORCH).**
- Verifies: we compute the same operations as the known-correct
  external reference.
- Tool: `matrix_verify.py --strict` (or `--ulp N`) against captured
  reference outputs.
- Catches: structural correctness errors invisible at the backend
  agreement layer.

**Both are needed.** Backend-vs-backend agreement is necessary but not
sufficient. Two backends that share the same bug agree with each
other; only the reference comparison catches that. The transpose-error
story is exactly this: our backends were internally consistent;
Ollama showed us they were consistently wrong.

Implementation note: layer A is fully shipped today (matrix_verify.py
supports all three contracts on any pair of .npy files). Layer B
waits on mavchin's spike capturing Ollama reference fixtures —
expected at the intermediate (per-matmul) layer when their next dump
step lands.

## Reference: empirical bit-identical status by kernel family

Per mavchin's measurements (intercom 11:11 UTC, against Ollama
reference where stated):

| Family | Bit-identical w/ Ollama | Bounded ULP w/ Ollama |
|---|---|---|
| Elementwise (add, mul, relu) | **PROVEN** | N/A |
| kv_store | **PROVEN** | N/A |
| kv_attention (1 pos) | **PROVEN** | N/A |
| Transcendental (silu, gelu, tanh) | impossible | ≤2 ULP |
| rms_norm, layer_norm | impossible | ≤13 ULP |
| dp4a vecmat | not yet (frontier) | TBD |

Per metayen's measurements (against PyTorch CPU, this is layer-A
data not layer-B):

| Op | --strict | --ulp 4 | --ulp 13 |
|---|---|---|---|
| sum_rows | DIVERGE | MATCH | MATCH |
| mean | DIVERGE | MATCH | MATCH |
| max | MATCH | MATCH | MATCH |
| min | MATCH | MATCH | MATCH |
| argmax | MATCH | MATCH | MATCH |
| argmin | MATCH | MATCH | MATCH |

The split is theory-confirming: ops that SELECT a value
(max/min/argmax/argmin) are bit-identical because no accumulation
occurs. Ops that REDUCE multiple values (sum/mean) diverge by 1-4
ULPs due to differing reduction trees.

## The implementation

- `bpd/tests/matrix_verify.py` — CLI tool implementing all three contracts
- `bpd/ieee754_hex.py` (mavchin) — authoritative bit-inspection utilities
- `bpd/tests/matrix_verify.py:ulp_distance_f32` — per-element ULP
  calculation across sign boundaries

Usage:
```
# Allclose contract (default)
python3 matrix_verify.py expected.npy computed.npy

# Bit-identical contract
python3 matrix_verify.py --strict expected.npy computed.npy

# Bounded-ULP contract (e.g., for transcendentals)
python3 matrix_verify.py --ulp 2 expected.npy computed.npy

# Multi-backend mode against a reference
python3 matrix_verify.py --strict --reference ollama.npy \
    python.npy c.npy rust.npy prolog.npy
```

## Methodology takeaway

The substrate's voice has three volumes:

1. **allclose silently absorbing wrongs** (volume 1, lossy)
2. **bounded-ULP exposing precision drift** (volume 2, useful diagnostic)
3. **bit-identical exposing semantic divergence** (volume 3, the truth)

Use the volume appropriate to the question. For "is the substrate
genuinely correct?" — use volume 3 wherever physically possible. For
"is the substrate within physical limits?" — use volume 2. For
"do we have catastrophic numerical bugs?" — volume 1 is sufficient.

The substrate is most honest when probed at the highest volume the
physics allows.

Author: metayen 2026-05-17
Per Heath's framing of bit-identical as essential and mavchin's
empirical kernel-family boundary measurements. Anchored in the
transpose-error story.
