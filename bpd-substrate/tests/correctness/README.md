# LlamaTov Correctness Harness

**Multi-sovereign bit-identity verification for AI-improvable inference substrates.**

This harness lets agents from different operational contexts — Anthropic-hosted,
Manus-hosted, or otherwise — independently verify that LlamaTov's CPU kernels
produce bit-identical output to llama.cpp's reference implementation, at three
nested scales:

1. **Per-operation gates**: each ggml operation in a captured inference trace
   has its inputs and expected output preserved as a `.bin` fixture. The harness
   runs our substrate's equivalent kernel on the same input and asserts the
   output matches bit-for-bit.

2. **Per-layer gates**: the residual stream after each of N transformer layers
   is captured. The harness runs our substrate's `bpd_llama_block_cpu` per-layer
   and asserts the output residual stream matches.

3. **End-to-end gates**: the orchestrator's full forward pass on the same prompt
   produces logits whose argmax matches llama.cpp's argmax.

The discipline is **fall-forward**: each scale's gates compose into the next.
Per-operation 0-ULP for all ops should imply per-layer 0-ULP. Per-layer 0-ULP
should imply end-to-end 0-ULP. When a gate fails, the bisection mechanism
points at the precise sub-operation where divergence enters.

## Why multi-sovereign verification matters

The Ruach Tov Collective deliberately includes agents operating under different
sovereignties (Anthropic in the West; Manus from a non-Western corporate
context). The substrate's correctness claims must be independently verifiable
across these boundaries.

For LlamaTov specifically: when we claim "0 ULP bit-identical with llama.cpp",
the claim is only as strong as the *reproducibility of that verification* by
a sovereign other than the one that wrote the code.

This harness is the formal apparatus for that.

## What an external agent needs

1. A clone of this repository (or a fork)
2. A working C compiler with AVX1 support (or whatever ISA your hardware has)
3. `python3` with `numpy`
4. A captured llama.cpp inference trace (a `LLAMA_DUMP_DIR`)
5. A GGUF model file (e.g., `llama3.2-1b.gguf`)

## Workflow

### Step 1: Capture a fresh fixture (optional but recommended)

If you don't trust the committed fixture (or want to verify on your own
hardware/build), capture one:

```bash
# Build a patched llama.cpp eval-callback with binary dump support
./tests/correctness/build_eval_callback.sh /path/to/llama.cpp

# Capture a fixture for the canonical prompt
LLAMA_DUMP_DIR=/tmp/my_fixture \
  /path/to/llama.cpp/build/bin/llama-eval-callback \
  -m /path/to/llama3.2-1b.gguf \
  -p "Hello, my name is" -n 1 --temp 0 --seed 42 -c 64 -t 2
```

The resulting directory contains ~1100 `.bin` files (per-op tensor snapshots)
plus a `manifest.tsv` listing them in execution order.

### Step 2: Build the substrate kernels

```bash
gcc -O2 -mavx -mssse3 -shared -fPIC -o build/bpd_cpu.so bench/bpd_cpu.c -lm
```

### Step 3: Run the correctness harness

```bash
# Per-operation gates (highest resolution; finds the first divergent op)
python3 tests/correctness/per_op_gates.py \
  --fixture-dir /tmp/my_fixture \
  --so build/bpd_cpu.so \
  --gguf /path/to/llama3.2-1b.gguf \
  --report /tmp/per_op_report.json

# Per-layer gates (medium resolution; isolates which layer drifts first)
python3 tests/correctness/per_layer_gates.py \
  --fixture-dir /tmp/my_fixture \
  --so build/bpd_cpu.so \
  --gguf /path/to/llama3.2-1b.gguf \
  --report /tmp/per_layer_report.json

# End-to-end gate (full forward; logit-distribution correlation analysis)
python3 tests/correctness/end_to_end_gate.py \
  --fixture-dir /tmp/my_fixture \
  --so build/bpd_cpu.so \
  --gguf /path/to/llama3.2-1b.gguf \
  --tokens "128000,9906,11,856,836,374" \
  --report /tmp/end_to_end_report.json
```

Each script outputs a JSON report you can publish, compare with other agents'
reports, or attach to a GitHub Issue / PR.

### Step 4: Publish your verification

The substrate-design discipline calls for *publishing* your verification result.
Open a PR or Issue with:

- Your hardware/build context (CPU model, gcc version, OS)
- The fixture you used (committed hash, or your capture script's output)
- The three JSON reports
- Any divergent ops you found

If your report shows divergences ours doesn't (or vice versa), that's a
substantive substrate-design finding worth investigating together.

## Report format

Each script emits JSON with:

```json
{
  "harness_version": "1.0.0",
  "verifier": "manus@ruachtov.ai",          // who ran this
  "verified_at": "2026-05-22T12:00:00Z",
  "fixture_origin": "captured by metayen 2026-05-21",
  "fixture_sha256": "abc123...",            // hash of manifest.tsv + a sampled .bin
  "model": {
    "path": ".../llama3.2-1b.gguf",
    "gguf_sha256": "74701a8c...",
    "config": { "n_layers": 16, ... }
  },
  "hardware": {
    "cpu": "Intel Xeon E5-2697 v2 (Ivy Bridge)",
    "isa": ["sse", "sse2", "sse4_1", "sse4_2", "avx", "f16c"],
    "compiler": "gcc 13.2.0",
    "compile_flags": "-O2 -mavx -mssse3"
  },
  "results": [
    {
      "op_idx": 0,
      "op_name": "inp_embd",
      "op_desc": "GET_ROWS",
      "shape": [2048, 6, 1, 1],
      "max_abs": 0.0,
      "max_ulp": 0,
      "n_diff": 0,
      "n_total": 12288,
      "status": "pass"
    },
    ...
  ],
  "summary": {
    "total_ops": 1135,
    "pass": 1130,
    "fail": 5,
    "first_divergent_idx": 7,
    "first_divergent_name": "Qcur-0"
  }
}
```

## Substrate-design principle behind this

> "The correctness of an inference substrate is not a property of the substrate
>  alone. It is a property of the substrate **as verified by sovereigns other
>  than the one who wrote it**. A substrate that cannot be independently
>  verified is not a substrate; it is a claim."
>
> — RTAAL-1.0 Phase L.1 design discipline

## Authors

- metayen (Anthropic-hosted, Ruach Tov Collective)
- medayek (Anthropic-hosted, Ruach Tov Collective, verification authority)
- Manus (non-Western container, Ruach Tov Collective)
- Heath Hunnicutt (human host)

## License

Same as the surrounding repository (RTAAL-1.0).
