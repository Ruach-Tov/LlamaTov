# BPD Toolbox Inventory — 2026-06-05

**Compiled by**: mavhir, at Heath's request
**Purpose**: Organize the substrate-of-built-tools accumulated across the Prolog→GPU→bit-identical-lowering thread so the next moves leverage what's already won.
**Discipline**: every entry maps a built artifact to its location on disk, its purpose, its connections to the larger architecture, and what's verified working vs staged.

---

## The Architectural Vision (per Heath, surfaced via Iyun 2026-06-05)

```
   lift llama.cpp compute-graph → Prolog facts (truth of record)
   ↓
   round-trip to C++ bit-identical
   ↓
   lower kernel facts → LLVM IR / Rust / cuda-oxide BIT-IDENTICAL by construction
   ↓
   fuse kernels in Prolog (declarative; not hand-rolled)
   ↓
   beat ollama on Tesla P4
```

The night's SoA kernel-archaeology demonstrated **the vision's thesis in the negative**:
hand-written bit-identity is fragile and unverifiable (the harness's reference was
built to match the kernel it validates → tautological). Declared FP semantics in
a lowered IR are verifiable by construction. The toolbox below makes the
declared-lowering path concrete.

---

## I. Prolog ↔ GPU Pipeline (CUPTI / CUDA / LLVM → PTX)

**Working end-to-end on Tesla P4 sm_61, enclave node** (per memory 7d0492d3,
2026-05-30 milestone):

```
  BPD Prolog facts → Prolog emitter → LLVM IR (.ll) → llc -march=nvptx64 → PTX
                  → cuda_load_module → cuda_launch → cuda_sync
                  → cupti_flush → stall_report
                  ALL FROM ONE SWIPL SESSION
```

### Components on disk

| File | Purpose | Status |
|------|---------|--------|
| `bpd/lib/cupti_bridge.c` | SWI-Prolog FLI wrapping CUPTI PC sampling. Predicates: `cupti_init/0`, `cupti_flush/0`, `cupti_stall_report/1`, `cupti_suggest/1` | ✅ working |
| `bpd/lib/bpd_cupti_profile.c` | Lower-layer CUPTI counter machinery (stall_counters_t struct) | ✅ working |
| `bpd/lib/cupti_profile.pl` | Prolog-level CUPTI workflow wrapper | ✅ working |
| `bpd/lib/cost_shape_cupti_validation.pl` | Prediction→measurement validation loop closing prediction–profile gap | ✅ working |
| **`bpd/lib/prolog_to_llvm.pl`** | **Emits LLVM IR from BPD Prolog facts.** Phase 1: `k_add` (elementwise). Predicates: `emit_kernel/2,3`, `emit_q8_0_dot/1,2`, `emit_q8_0_dot_vec/1`, `emit_q8_0_dot_intrin/1` | ✅ working |
| `bpd/lib/bpd_llvm.c` | Lower-layer LLVM-C bindings (LLVMModuleCreateWithName, BitWriter, etc.) | ✅ working |
| `bpd/lib/bpd_llvm_elem.c` / `bpd_llvm_norm.c` | Element-wise and norm-kernel LLVM emitters | ✅ working |
| `bpd/llvm/bpd_gelu.ll`, `bpd_clamp.ll`, `bpd_tanh.ll`, etc. | **Reference emitted .ll files**, each a Prolog→.ll output. GELU verified 0-ULP vs PyTorch | ✅ verified |
| `bpd/lib/kernel_emit_param.pl` | Parameterized kernel emitter (ARR, FMA dimensions sweepable) | ✅ |
| `bpd/lib/kernel_emit_bridge.py` | Python bridge for kernel emitter pipeline | ✅ |
| `bpd/lib/kernel_templates_llama.pl` | Llama-specific kernel templates | ✅ |
| `bpd/lib/llama_cpp_lifter.pl` | **Lifts llama.cpp → Prolog facts** (the lift side) | ✅ |
| `bpd/lib/gguf_emit_manifest.pl` | Emits manifest from GGUF (model weight inspection) | ✅ |
| `bpd/lib/llvm_match_status.pl` | Tracks LLVM-emission match status across kernel set | ✅ |

### Build environment (enclave node — Nix store paths)

```
CUDA_LIB=/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib
CUPTI_LIB=/nix/store/i43ngfp7y52sjd8i9yy5fayk1c48xypa-cuda_cupti-12.8.90-lib/lib
CUDA_INC=/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8/include
```

**Build cupti_bridge.so**:
```
swipl-ld -shared -o cupti_bridge \
    lib/cupti_bridge.c lib/bpd_cupti_profile.c \
    -lcupti -lcuda -I$CUPTI_INC -L$CUPTI_LIB
```

**Key fix recorded** (memory 7d0492d3): `PL_get_atom_chars` fails on SWI-Prolog
strings — added `PL_get_chars` fallback with `CVT_ALL`.

### cuda_launch.c (referenced in milestone memory, location to confirm)

Per memory 7d0492d3, `cuda_launch.c` is a PLF with 9 predicates:
`cuda_init`, `cuda_device_info`, `cuda_load_module`, `cuda_launch`,
`cuda_sync`, `cuda_alloc`, `cuda_free`, `cuda_memcpy_h2d`,
`cuda_memcpy_d2h`. Not located in my scan of `bpd/lib/` — may be in
`bpd/cuda_launch/` subdir or on enclave but not synced to local checkout.
**TODO**: locate and confirm cuda_launch.c location.

### What CUPTI hardware data we extract

From memory 14971398 (mavchin + Iyun 2026-06-04, P4 hardware-counters
breakthrough):

**P4 has 80 EXACT hardware event counters via `cuptiEventGroupReadAllEvents`**
— TOTALS not samples, like CPU perf_event:

| Counter | Meaning |
|---------|---------|
| `fb_subp0/1_read_sectors` | Total DRAM read sectors |
| `gld_inst_8/16/32/64/128bit` | Load instructions BY WIDTH |
| `active_cycles_pm`, `elapsed_cycles_sm` | Cycles |
| `l2_subp0/1_read_sector_misses` | L2 misses |

**THE COALESCING METRIC**: `sectors_per_load = fb_read_sectors / gld_inst` —
deterministic, exact, saturation-independent. This is the GPU PMU.

---

## II. Kernel-from-Facts Pipeline (memory f3536d48, kernel_from_facts_pipeline)

**Goal**: lift Ollama's `0xb7110` Q8_0 dot via asm_facts → Prolog facts → emit
OUR OWN LLVM .ll. Bit-identity BY CONSTRUCTION (lifting 0xb7110's exact order
guarantees matching ggml; the Q8_0 dot is **proven order-sensitive** at f32,
95K ULP vs f64 — memory 6a8bbc20).

### asm-facts tooling (bpd/tools/)

| File | Purpose |
|------|---------|
| `bpd/tools/asm_facts.py` | Lift x86 asm → Prolog facts (jit_insn/4: addr, op, dst, operands) |
| `bpd/tools/asm_jit_lifter.py` | JIT-flavor lifter |
| `bpd/tools/asm_dataflow.py` | Dataflow analysis layer |
| `bpd/tools/asm_loops.py` | Loop-structure extractor |
| `bpd/tools/asm_pcode.py` | P-code (Ghidra) IR layer |
| `bpd/tools/asm_unroll.py` | Loop-unrolling analyzer |
| `bpd/tools/asm_transcendental.py` | Transcendental-op recognizer |
| `bpd/tools/opcode_match.py` | Opcode-pattern matching |
| `bpd/tools/llvm_pcode.py` | LLVM-side P-code |

### Kernel-spec → LLVM IR emission

`ondnn_kernel_emitter.pl` referenced in memory f3536d48 — GELU emitter, does
`<8 x float>` vectors + loops, **proven methodology** (4h disassemble → 0-ULP).
Being extended with integer SIMD ops: `pmadd_ub_sw`, `pmadd_wd`, `paddd`,
`load_i8x16`.

LLVM intrinsics targeted:
- `@llvm.x86.ssse3.pmadd.ub.sw.128`
- `@llvm.x86.sse2.pmadd.wd`

**Build paths for emitted code**:
- Ollama's libs: `/nix/store/wfa60hfmh9kf77sxk6dpq20m6bg1df2d-ollama-0.20.7/lib/ollama/`
  - `libggml-cpu-sandybridge.so` (0xb7110 source)
  - `libggml-cuda.so` (GPU backend)
- `ggml_vec_dot_q8_0_q8_0` @ `0xb7110`, `_generic` @ `0x38730`

### Verification infrastructure

| File | Purpose |
|------|---------|
| `/tmp/q8_0_dot_unittest.c` (mavhir/enclave) | f64-reference dot vs kernel-under-test (10000 trials, max ULP 94960 — proved order-sensitivity) |
| `/tmp/q8_0_dot_vs_ggml.c` (mavhir/enclave) | dlopen-based ggml-CPU oracle test |
| `bpd_gemm_q8_0_cpu.c:50` `bpd_vec_dot_q8_0_ggml` | Reference dot kernel (static inline, #include the .c to test) |

---

## III. ggml from Prolog (BREAKTHROUGH 1, memory 14971398, 2026-06-04)

**ggml_matmul_q8_0(2048,2048,1,Ms) → Ms=0.067** (67us on P4, matches standalone C).

Full chain: `Prolog → FLI → LLVM IR → ggml_backend_graph_compute → Tesla P4`.

**ABI fix recorded**: byval struct-passing (learned from `clang -emit-llvm`).

This realizes Heath's "launch ggml from Prolog" — the whole research loop is now
RELATIONAL: drive ggml's REAL kernels from the substrate, measure with CUPTI,
feed facts back. The generator can now invoke + measure the actual target kernels.

**Generator files** (`bpd/generators/`):
- `generate_blas_kernels.pl`
- `generate_cfd_kernels.pl`
- `generate_fused_kernels.pl`
- `generate_llama_kernels.pl`

---

## IV. Tensor-Op Type System (memory f9ad2409, mavchin 2026-06-02)

**"Type-level Prolog"** — compound terms ARE types, unification IS typecheck,
derivation IS inference.

Three layers:
1. **BOTTOM**: `tensor(Name, dtype(D), shape(S), layout(L))` where L ∈
   {contiguous, strided(St), permuted(P)} + `op(Node, Coord, OpType, in([Srcs]), out(Dst))` + `bandwidth(bytes_r, bytes_w, arith_intensity)`.
2. **MIDDLE** (mavchin owns): type-derivation engine — derive out-tensor type
   from in-types + op; e.g., `view(T, _)` requires `layout(T) = contiguous` →
   FAILS after transpose w/o cont. **Would have statically caught the V-sum
   n_kv padding mismatch**.
3. **TOP** (Heath builds visual): `render(Coord, View)` + `describe/2` +
   bandwidth overlays.

**Shared fact relations**: `bpd/lib/tensor_types.pl` (referenced as TODO target
in memory f9ad2409).

### Existing infra for layout-op verification

| File | Purpose |
|------|---------|
| `bpd/lib/dashboard_common.pl` | Tables 10000/10001/10010/10011 dashboards (freshness_stamp, layer_coords) |
| `bpd/lib/cfd_retire_status.pl` | retire_status fact schema |
| `bpd/cfd_retire_dashboard.pl` | SVG render from facts, cross-links |
| `bpd/lib/divergence_map.pl` | `div_status/2` + `div_measured/2` (max_ulp + provenance) |
| `bpd/lib/compute_graph_invariants.pl` | Graph-level invariants |

### Layout-op-verification harness target

Verify zero-computation/layout ops (CONT/CPY/RESHAPE/PERMUTE/TRANSPOSE/VIEW)
against eval-callback fixture (ollama ground truth). Concern: **complementary
mistakes in transpose/reshape/copy that mask each other**.

---

## V. Fusion / SwiGLU Correctness (memory 9caedd34, Heath 2026-06-04 catch)

**Heath's catch**: the mul-into-up fusion bakes in an ordering the graph doesn't
guarantee. Gate and up are INDEPENDENT in true dataflow:
```
    x → gate → silu \
    x → up ----------→ mul → down
```
A scheduler may run either first or parallel.

**Robust alternative (Iyun's 22.3% SwiGLU prototype)**: compute BOTH gate and up
dots in ONE kernel, then silu(g)*u. NO cross-kernel buffer, NO ordering
dependency — CORRECT BY CONSTRUCTION. Faster too (22.3% vs 14.4%).

**Method lesson banked**: a fixture that IMPOSES an ordering (hard-sequenced
launches) can hide a dependency assumption that the production graph doesn't
guarantee. The fixture's correctness doesn't prove the FRAMEWORK's correctness
when the framework's scheduling differs.

---

## VI. Bit-Identity Verification (the cluster)

Topics in memory layer: `bit-identical`, `kernel-proven-bit-identical`,
`bit-identical-fusion`, `bit-identical-foundation-complete`,
`asm-facts-lift-bit-identity`, `bit-identical-scheduling-sweep`,
`soa-e2e-bit-identical`, `bit-identity-gate-mandatory`.

Key memory `53eb2c77` (2026-06-05 mavchin reframe): **kernel proven 0-ULP/bit-
identical at per-matmul level; decode bug is NOT kernel math**.

Tonight's investigation (commits `baf444d5c`, `71fdf847c`, `a19f25df9`,
`4f59febcf`, `89ec9d4a2`) **confirms and extends** this empirically:
- 109/111 prefill matmuls bit-identical with FMA off in mmvq.cu
- The drift enters from non-mmvq kernels (RMS_NORM, RoPE, softmax) over 16 layers
- Hand-written-bit-identity is fragile/unverifiable → vision argument

---

## VII. Hardening / Debugging Substrate

| File | Purpose | Status |
|------|---------|--------|
| **`bpd/lib/cuda_mem_paint.c`** | **LD_PRELOAD shim wrapping cudaMalloc (paint 0xBAADF00D) and cudaFree (paint 0xDEADBEEF). Forces uninit reads and UAF to produce detectable sentinels.** Commit baf444d5c. | ✅ tested, committed, durable |
| `bpd/investigations/2026-06-05-soa-q8-divergence-investigation.md` | 5-commit investigation note (10 findings + 3 fix paths + 3 addenda). The substrate-of-record for tonight's SoA kernel debugging. | ✅ committed |
| `bpd/patches/cuda-oxide-sm61-*.patch.md` | Pascal sm_61 support patches for cuda-oxide (Rust CUDA emitter) | ✅ |

---

## VIII. cuda-oxide / Rust Codegen (the Rust lowering target)

`bpd/rust/` directory exists. Patches at `bpd/patches/cuda-oxide-sm61-{pipeline,call,collector,cmath}.patch.md` indicate active work on **Pascal sm_61 support in cuda-oxide** — the Rust-side declarative-lower target.

This is the third lowering target named in the vision (LLVM/Rust/cuda-oxide).

---

## IX. Inventory Gaps + Locations — RESOLVED (Iyun's discovery, 2026-06-05 evening)

**MAJOR CORRECTION**: Iyun located all four named gap files in a SEPARATE
parallel tree I had not scanned: `<repo>/bpd-substrate/`.

The `bpd-substrate/` tree is parallel to `bpd/` and contains substantively
richer substrate-of-built-foundation than the inventory above captured.

### Gap files located

| File | Location | Size |
|------|----------|------|
| `cuda_launch.c` | `bpd-substrate/lib/cuda_launch.c` | 322 lines |
| `tensor_types.pl` | `bpd-substrate/lib/tensor_types.pl` | 274 lines |
| `ondnn_kernel_emitter.pl` | `bpd-substrate/lib/ondnn_kernel_emitter.pl` | 257 lines |
| `gelu_ondnn_emitter.pl` | `bpd-substrate/lib/gelu_ondnn_emitter.pl` | 211 lines |
| `nvml_bridge_v2.pl` | `bpd-substrate/lib/nvml_bridge_v2.pl` | 298 lines |
| `nvml_bridge_emitter.pl` | `bpd-substrate/lib/nvml_bridge_emitter.pl` | **938 lines** |

### Additional substrate found in `bpd-substrate/lib/` (not in `bpd/lib/`)

**Attribution corrected via ls-by-origin-story** (Heath's archeology tool at
`tools/origin_story/`). The author attribution in the file source headers
("Author: mavchin") doesn't match git origin — the bpd-substrate/ files
were committed by Heath Hunnicutt himself. The file-header bylines may
reflect IRC/handoff context, but the canonical-substrate provenance is
Heath's authorship.

- **`llvm_emit.pl`** — Declarative LLVM IR generation from Prolog facts.
  **Heath, 2026-05-31 23:03Z**, commit `93d411fa9` "feat: declarative LLVM
  IR generation from Prolog facts." Source header credits mavchin
  (2026-06-01 — IRC/handoff context). "The BPD thesis applied to code
  generation: specification IS Prolog facts, projection IS LLVM IR."
- **`swiglu_fused_emitter.pl`** — THE INAUGURAL FUSED KERNEL FROM FACTS.
  Fuses bpd_silu_cpu + bpd_mul_cpu into one pass. **Verification: fused ==
  unfused == ggml (transitive 0 ULP).** **Heath, 2026-06-03 00:37Z**,
  commit `aee5312d4` "feat: inaugural fused kernel — SwiGLU from Prolog
  facts, 0 ULP." Source header credits mavchin (2026-06-03 — IRC/handoff
  context). Critical bit-identity details documented FROM IYUN in the
  source comments: scalar libm expf (NOT polynomial), DIVIDE form
  `x / (1+exp(-x))` NOT reciprocal-mul (the two differ by 1 ULP).
- **`swiglu_strict_emitter.pl`** — strict-mode SwiGLU variant.
- **`ggml_cuda_bridge_emitter.pl`** — ggml CUDA bridge generator.
- **`ptx_compile.pl`** — PTX compilation predicate.
- **`tensor_schema.pl`** — tensor schema layer.
- **`ondnn_jit_facts.pl`** — oneDNN JIT instruction facts.
- **`gpu_roofline_analysis.pl`** — GPU roofline analysis.
- **`citation_extractor.pl`, `citation_linter.pl`, `citation_markdown_emitter.pl`** — citation tooling.
- **`pcie_trace.scm`** — PCIe trace (Scheme).
- **`render_ascii.pl`** — ASCII renderer.
- Plus LLVM .ll output files: `bpd_conv.ll`, `bpd_losses.ll`, `bpd_pool.ll`,
  `bpd_tanh_fix.ll`, `bpd_exp_fix.ll`, `bpd_scale_cumsum.ll`, `bpd_sum_sse3.ll`.

### Tree relationship — corrected by metayen 2026-06-05

**metayen surfaced**: Heath named the `bpd/` vs `bpd-substrate/` bifurcation
on Tuesday. **`bpd-substrate/` is currently a downstream copy of files
canonically maintained elsewhere in `Ruach-Tov/Ruach-Tov`**. The cleanup
is pending. The inventory should name both paths until the cleanup
happens.

- Overlap files in both `bpd/lib/` and `bpd-substrate/lib/` are byte-identical
  (verified for `cupti_bridge.c`, `bpd_llvm.c`).
- `bpd-substrate/` has the substantively-richer set of declarative-emitter
  files (87 lib/ files); `bpd/` has 116 but lacks much of the substrate.
- Canonical location is **Ruach-Tov/Ruach-Tov** (per Heath's Tuesday framing).
  `bpd-substrate/` is the downstream copy. `bpd/` is the partial-copy/in-flight
  reconciliation target.
- Reconciliation is Heath's call. Until then, references to the substrate
  should name both paths.

### Confirmed REAL gap: `soa_test_real.cu`

- **`/tmp/soa_test_real.cu`** — mavhir + Iyun referenced this in tonight's
  SoA investigation. metayen independently confirmed it's NOT located in
  the repo. Either renamed during the SoA work, lives in a non-obvious
  location, exists only as transient artifact on enclave/tmp where mavchin
  runs his SoA work, or the memory ID points to substrate that exists in
  description but not in artifact. **Action**: confirm with mavchin
  whether it has a canonical location.

### Substantively-substantial implication for the architectural trajectory

**The pipeline foundation Iyun and I had named as "to-build" is substantively
ALREADY BUILT** (Iyun's framing):
- LIFT side EXISTS: `llama_cpp_lifter.pl` (step 1)
- LLVM emission EXISTS + WORKS: `prolog_to_llvm.pl` + `llvm_emit.pl` + the
  declarative-emitter cluster (step 3)
- BIT-IDENTITY BY CONSTRUCTION already DEMONSTRATED: `bpd/llvm/bpd_gelu.ll`
  0-ULP vs PyTorch + `swiglu_fused_emitter.pl` verified transitive 0-ULP
  fused==unfused==ggml (step 4)
- cuda_launch + CUPTI working end-to-end from one SWIPL session (memory
  7d0492d3)
- NVML adapter built TWICE: `nvml_bridge_emitter.pl` (format-string version,
  938 lines) AND `nvml_bridge_v2.pl` (declarative-emitter version, 298 lines)
  — demonstrating the lift→emit substrate handles both kernel-class AND
  bridge-class generation uniformly.

So the re-vector Heath named as "build the foundation" is more accurately
**"extend the existing proven foundation."** The SoA hand-kernel was step-7
done by hand while steps 1-3 were already done declaratively. That's the
divergence Heath named, now quantified: tonight's investigation was
hand-fighting bit-identity that `prolog_to_llvm.pl` + `llvm_emit.pl` already
emit declaratively for GELU and SwiGLU.

**The remaining work for SoA Q8_0**: extend `prolog_to_llvm.pl`'s
already-working Q8_0 dot emission (via `emit_q8_0_dot` predicates) with
the SoA-weight-load fact + the full matmul chain. Bit-identity verified
at GELU + SwiGLU means the same emitter-class delivers bit-identity for
Q8_0 by construction.

### Section IX old action items — RESOLVED (4 of 5)

Four named gaps located in `bpd-substrate/lib/`. The fifth (`soa_test_real.cu`)
is a confirmed real gap pending mavchin's clarification on canonical location.

### The Archeological Narrative (via ls-by-origin-story, Heath's direction)

Running `tools/origin_story/ls-by-origin-story.pl` on `bpd-substrate/lib/`
gives the chronological record of the bit-perfect-declarative substrate:

**2026-05-29** — Foundation commits:
- `cuda_launch.c` (commit `9af2bcf0b`): "feat: cuda_launch PLF — launch GPU kernels from Prolog"
- `ptx_compile.pl` (commit `3aa7a4cd7`): "feat: ptx_compile.pl — LLVM IR → PTX compilation from Prolog"

**2026-05-31** — Declarative emission substrate:
- `llvm_emit.pl` + `nvml_bridge_emitter.pl` (commit `93d411fa9`): "feat: declarative LLVM IR generation from Prolog facts"
- `nvml_bridge_v2.pl` (commit `23daf1dab`): "feat: nvml_bridge_v2 — declarative bridge using llvm_emit infrastructure"

**2026-06-01** — First 0-ULP kernel emissions:
- `gelu_ondnn_emitter.pl` (commit `db3a957e8`): "feat: GELU oneDNN — vectorized LLVM IR from Prolog, 93% at 1 ULP"
- `ondnn_kernel_emitter.pl` + `ondnn_jit_facts.pl` (commit `a6db97e0d`): "feat: fact-driven LLVM IR emitter for oneDNN kernels — 0 ULP GELU"

**2026-06-02** — Type derivation:
- `tensor_types.pl` (commit `449117820`): "feat: tensor_types.pl — type derivation engine for tensor operations"

**2026-06-02** (bpd/ tree, mavhir authored):
- `prolog_to_llvm.pl` (commit `eb209bce9`): "prolog_to_llvm.pl Phase 1: BPD facts → LLVM IR → 0 ULP"

**2026-06-03** — Fused kernels + LLVM elementwise/norm/conv/pool/loss:
- `swiglu_fused_emitter.pl` (commit `aee5312d4`): "feat: inaugural fused kernel — SwiGLU from Prolog facts, 0 ULP"
- `swiglu_strict_emitter.pl` (commit `907d8cf02`): "wip: strict SwiGLU emitter"
- `bpd_llvm.c` (commit `cd6f8c70b`): "feat: Prolog→LLVM IR emitter achieves 0 ULP vs ggml SSE3"
- `bpd_llvm_elem.c` (commit `ce1378933`): "feat: Prolog→LLVM IR unary elementwise emitter — 10 ops, silu at 0 ULP"
- `bpd_llvm_norm.c` (commit `444ea6528`): "feat: Prolog→LLVM IR rms_norm emitter — 0 ULP vs scalar ref"
- `bpd_conv.ll` (commit `d37d5dce7`): "feat: conv_im2col LLVM IR emitters — im2col_1d + conv1d at 0 ULP"
- `bpd_pool.ll` (commit `537330f63`): "feat: pool_reduce LLVM IR emitters — max_pool + avg_pool at 0 ULP"
- `bpd_losses.ll` (commit `0c8e9fba8`): "feat: loss_reduce LLVM IR emitters — 4 loss functions"
- `bpd_sum_sse3.ll` (commit `b85113780`): "feat: dedicated sum emitter — 0 ULP vs ggml SSE3"
- `bpd_tanh_fix.ll` (commit `3d53f3846`): "fix: tanh calls tanhf (0 ULP), hardsigmoid fixed constant (0 ULP)"
- `bpd_scale_cumsum.ll` (commit `3f0a10e82`): "feat: scale + cumsum + cumprod + clamp LLVM IR emitters"
- `llama_cpp_lifter.pl` + bulk (commit `af26c1217`): "Initial release: BPD Substrate — Bit-Perfect Declarative GPU Kernels"
- `ggml_cuda_bridge_emitter.pl` (commit `254fa969a`): "wip: ggml-CUDA bridge emitter — dispatches ggml kernels from Prolog"
- `bpd_cupti_profile.c` (commit `f7d94a4c3`): "feat: CUPTI PC sampling profiler — substrate sees its own bottlenecks"

**Five files in `bpd-substrate/lib/` have NO git origin** (potentially imported
via human/CI/scp):
- `bpd_exp_fix.ll`
- `c_preprocess.pl`
- `gguf_native_reader.pl`
- `render_ascii.pl`
- `tensor_schema.pl`

**Tonight (2026-06-05) in `bpd/lib/`** (mavhir):
- `cuda_mem_paint.c` (commit `baf444d5c`): "feat(bpd): cuda_mem_paint.c — LD_PRELOAD shim for cudaMalloc/cudaFree paint hardening"

### What the archeology reveals

1. **Heath authored the bit-perfect-declarative substrate himself** between
   2026-05-29 and 2026-06-03. The file-header bylines crediting mavchin
   are IRC/handoff context, not authorship attribution.

2. **Every milestone commit names 0 ULP achievement**: GELU, SwiGLU,
   rms_norm, conv1d, max_pool, avg_pool, sum, tanh, hardsigmoid — all
   delivered at 0 ULP from Prolog facts. The bit-identity-by-construction
   property is achieved at *many* kernel classes, not just two.

3. **The week-long trajectory had a coherent direction**: foundation →
   declarative emission → first 0-ULP kernel (GELU) → type derivation →
   fused kernels (SwiGLU) → elementwise/norm/conv/pool/loss emission →
   ggml-CUDA bridge → CUPTI profiler. The substrate built itself up the
   abstraction stack in order.

4. **The substrate's existence was already substantially built** before
   tonight's SoA hand-kernel investigation began. Iyun's "the
   foundation isn't just built — it's PROVEN at the exact operations
   we need" was empirically substantiated by the origin-story record.

5. **The substantively-substantial closing-loop finding** (Iyun's
   observation tonight): `swiglu_fused_emitter.pl` (commit `aee5312d4`,
   2026-06-03) produces SwiGLU fusion bit-identically from facts, with
   verification `fused == unfused == ggml (transitive 0 ULP)`. Tonight's
   hand-written SoA bug DROPPED exactly that SwiGLU fusion at blk.15.
   The declarative solution to tonight's bug-class was already committed
   to `bpd-substrate/lib/` two days before the investigation.

## Methodology Lesson 9 — The Substrate-Consultation Boundary

**Three-parent attribution** (the principle has the shape it has because
each contribution is load-bearing in its own way):

- **Iyun (2026-06-05)**: observed the empirical limitation tonight —
  "A kernel-level harness, however rigorous, cannot catch a composition
  bug." Without that observation, no abstraction to name.

- **mavhir (2026-06-05)**: consolidated as methodology lesson +
  connected to tonight's specific surface (SoA fusion-drop, commit
  `dae6d508e`); named "substrate-consultation boundary" as the
  class-of-bug-surface and sharpened the mechanism to **partial-
  consultation** (the consulter's inability to know when consultation
  is incomplete).

- **metayen (2026-06-05)**: named the abstraction underneath —
  property-by-construction vs property-by-consultation. Without that
  distinction, the lesson stays kernel-specific rather than generalizing
  to citation-chains + pytestmark + the broader class.

The principle, as it stands after three-parent synthesis:

> **When substrate-of-record carries the property by construction, the
> property survives composition. When the property lives in code that
> consults the substrate-of-record, composition is where the property
> can be lost silently.**

The bug-class lives at the **substrate-consultation boundary** — where
code reads from a substrate-of-record but only partially, and the
partiality is silent. The hazard property is the consulter's inability
to know when consultation is incomplete.

Three surfaces of the same shape, this week alone:

1. **Tonight's SoA token-divergence bug** (commit `dae6d508e`): the SoA
   dispatch reads `fusion_local` but only partially (`gate`, `glu_op`)
   and silently drops the rest. Partial-consultation at the
   graph-execution composition boundary.

2. **Thursday's pytestmark silent-strip**: the regenerator consults the
   IR substrate-of-record but not the accreted-state substrate of
   hand-edited test marks. Partial-consultation across substrate layers.

3. **Truth Flow citation chains**: when projection carries citation by
   construction, attribution survives. When consumption-code is
   responsible for preserving attribution, it can be dropped silently
   at any consumption.

The declarative lift/lower pipeline (Heath's vision) prevents the
class by construction because composition IS the substrate — there's
no consultation-boundary between substrate-of-record and execution;
the substrate-of-record IS execution. Kernel-level rigor (Iyun's fixed
harness) catches kernel-level bugs by definition. It cannot catch a
silent partial-consultation between substrates because the bug is at
the boundary, invisible below the composition level.

This generalizes lesson 8's framing ("declared FP semantics in lowered
IR are verifiable by construction") to the broader principle: any
property that lives in code-that-consults-substrate is vulnerable at
every consultation boundary. Properties that live in
substrate-carried-by-construction survive composition.

## Section XII Candidate — Documentation Substrate (visible-adjacent, not folded in yet)

metayen flagged: the Truth Flow citation substrate at
`bpd-substrate/lib/citation_extractor.pl` + `citation_linter.pl` +
`citation_markdown_emitter.pl`, plus 5 IRs annotated this week with
`cites/2` (must_close, boundary_dsl, ir, coreutils, and one more) — these
belong in the inventory when it expands to documentation substrate.
Holding visible-adjacent per metayen's framing. Not folded in until
explicitly directed.

---

## X. How the Toolbox Connects to Tonight's SoA Investigation

The SoA hand-debugging tonight exercised these tools:
- **CUPTI bridge** (would have given us coalescing metric per memory 14971398)
- **Paint shim** (built and used; refuted uninit-memory hypothesis)
- **Element-wise comparison discipline** (Heath's correction; revealed long-tail divergence)

What the tools that exist BUT WEREN'T LEVERAGED tonight:
- The Prolog→.ll emitter (prolog_to_llvm.pl) could have produced gemv_soa
  with declared FMA contraction semantics, making the kernel's FP behavior
  STATED rather than discovered through ULP-archaeology
- The asm_facts lifter (asm_facts.py) could lift stock mul_mat_vec_q from
  ggml's binary, producing kernel_spec facts that GUARANTEE bit-identity
  by construction rather than relying on hand-matching reduction order
- The kernel-from-facts pipeline (memory f3536d48) is exactly the
  alternative to the hand-written gemv_soa that the investigation pointed at

**The investigation's negative conclusion → the toolbox's positive direction**:
the bug-class we hand-fought (compiler-FMA-scheduling-induced ULP differences
in matmul kernels + non-matmul kernel cascade) is exactly what a Prolog-lift +
LLVM-IR-with-declared-FP-semantics pipeline eliminates by construction.

---

## XI. Next Substantive Moves (for whoever picks up this thread)

Given the toolbox, the most leveragable next moves are:

1. **Connect** `llama_cpp_lifter.pl` → `prolog_to_llvm.pl` for a single
   matmul kernel as proof-of-concept of the lift-and-lower round-trip.
   Verify the round-trip-emitted matmul is bit-identical to stock ggml at
   the per-matmul level — this is the "verify the round-trip is REALLY
   bit-identical" task Iyun named.

2. **Apply tonight's lesson**: the bit-identity verifier MUST use real
   ggml `mul_mat_vec_q` as oracle (not a hand-written reference like
   mavchin's `stock_matmul_q8_0` in `/tmp/soa_test_real.cu`). Otherwise the
   verification is structurally tautological. Build `q8_0_dot_vs_ggml.c`
   pattern (dlopen libggml-cuda.so) as the verification template.

3. **Locate the missing artifacts** (item IX above) so the toolbox inventory
   is complete on local disk, not split between mavchin's enclave dir and
   this checkout.

4. **Extend the paint shim** (`bpd/lib/cuda_mem_paint.c`) to cover
   `cudaMallocAsync`/`cudaMallocManaged` for production use (currently only
   `cudaMalloc`/`cudaFree`).

5. **Wire CUPTI sectors_per_load** (memory 14971398) as the gating metric
   for any SoA-vs-AoS comparison — replaces the saturation-confounded
   bandwidth measurements with the saturation-independent per-access
   coalescing efficiency.

---

## Methodology Lessons Banked Across This Thread

1. **Hand-written bit-identity is fragile and unverifiable**. Declared FP
   semantics in lowered IR are verifiable by construction. (Tonight's
   investigation; the vision's thesis demonstrated in negative.)

2. **Population statistics ≠ element-wise bit equality**. Aggregate
   matching can mask structural divergence. (Heath's correction tonight.)

3. **Validation reference must be production-stock, not hand-twin**.
   Otherwise the test is structurally tautological. (Tonight's
   harness-validates-kernel-vs-its-twin finding.)

4. **Read aggregate counter TOTALS, not PC-sampling**. Deterministic,
   exact. (Heath's methodological point, memory 14971398.)

5. **A fixture that imposes ordering hides dependency assumptions the
   production graph doesn't guarantee**. (Heath's gate/up catch, memory
   9caedd34.)

6. **Don't analyze a moving target**. Freeze the substrate before measuring.
   (Tonight's mavhir-owned-tree workaround.)

7. **Warm vs cold CUDA driver state can flip measurement results 20x in
   timing and from BIT-IDENTICAL to MISMATCH in correctness**. Always
   verify the libcuda actually loaded (`/run/opengl-driver/lib`). (Tonight's
   harness env-bisection.)

8. **Commit and push artifacts often**, even mid-investigation, even from
   Heath's repo. Substrate-of-record needs version control. (Standing
   policy.)
