# BPD Substrate — Bit-Perfect Declarative GPU Kernel Generation

**From Prolog facts to verified GPU kernels**

## Quick Start

```bash
# Prerequisites: gcc, Python 3.12+, PyTorch, SWI-Prolog
git clone https://github.com/heath-hunnicutt-ruach-tov/bpd-substrate.git
cd bpd-substrate

# Build the kernel library
make build/bpd_cpu.so

# Run the verification ladder
make lint                    # Prolog loads clean
make correctness             # Wilkinson backward-error bound
make bit_identical_cpu       # BPD vs PyTorch CPU (21/22 pass)
```

## How to Contribute

New here? Start with **[docs/onboarding-new-agent.md](docs/onboarding-new-agent.md)** — written for both AI agents and humans.

1. Look at [open issues](https://github.com/heath-hunnicutt-ruach-tov/bpd-substrate/issues) — issues labeled `good first issue` are bounded tasks with clear acceptance criteria
2. Read [CONTRIBUTING.md](CONTRIBUTING.md) for our testing discipline (three rules: spec-mapping, fresh-clone smoke, stage-boundary verification)
3. Fork, branch, PR. Every PR must pass `make bit_identical_cpu` before merge
4. We verify every PR on our enclave hardware (Tesla P4, Intel Ivy Bridge) before merging

Contributors so far: [ColonistOne](https://github.com/ColonistOne) (PR #1), Manus (14 PRs), metayen, medayek, mavchin.

## What is this?

BPD is a GPU kernel substrate written in Prolog. It generates CUDA and C kernels from declarative facts — one Prolog fact per kernel. The substrate verifies each emitted kernel bit-identical with a reference implementation (PyTorch, cuBLAS, or llama.cpp).

The substrate is 100% semantic — zero opaque C strings. `c_raw` throws an error. Every loop, branch, stride, and accumulation is a structural c_ast node that the optimizer can pattern-match, rewrite, and fuse.

## Current Results

### YOLOv5n: 24/24 Layers BIT_IDENTICAL with PyTorch CPU

A complete YOLOv5n forward pass runs end-to-end through BPD-generated C kernels on real trained weights. All computation dispatches through our kernel library (bpd_cpu.so) via ctypes when the library is loaded.

Verified bit-identical per layer by metayen (commit 44d5bd6). End-to-end classification verified by Manus. Independently verified for layers 0-2 by mavchin. Independent build reproducibility verified by medayek (108 detections, 0 ULP).

Reproduce: `BPD_CPU_SO=build/bpd_cpu.so python3 bench/verify_yolo_composition_sweep.py /path/to/yolov5n.pt`

### YOLOv5n Performance: within 1.34× of PyTorch CPU, bit-identical throughout

Phase 3 closed ~91% of the original 7.30× gap to stock PyTorch CPU on Ivy Bridge AVX1 (no FMA, no AVX2):

| Substrate path | ms/image | vs PyTorch CPU |
|---|---:|---:|
| Scalar baseline (session start) | 4256 | 7.30× slower |
| + AVX1 v1 GEMM (1-acc, 1×8 tile) | 1247 | 2.72× slower |
| + AVX1 v2 GEMM (8-acc, 4×16 tile, KU=4) | 637 | 1.38× slower |
| + prefetch + B-panel packing | ~600 | 1.34× slower |
| PyTorch CPU baseline | 447 | 1.00× (target) |

Throughout: 10/10 MATCH on Medayek's `compare_detections`, conf_ULP=0, box_diff=0.0000px on every image.

The kernel parameters were deduced empirically via disassembly of OpenBLAS's `sgemm_kernel_SANDYBRIDGE` (the kernel PyTorch CPU calls into on Ivy Bridge). Seven substrate-design parameters identified; the dominant three (`register_blocking(4×16)`, `ilp_accumulators(8)`, `unroll_factor_K(4)`) closed most of the gap. See foundational memory `c101e652` for the full anatomy.

Built via TDD into precision existence: 7 primitives (P1–P7) each verified at 0 ULP in isolation, composed into the production CBS kernel. See `bench/test_f3_v2_tdd.py`.

### Stanford KernelBench L1: 94/100 BIT_IDENTICAL (in progress toward 100)

94 of 100 Stanford KernelBench L1 problems produce PyTorch CPU's exact float32 bytes on our hardware (Tesla P4, Intel Ivy Bridge, no AVX2). The remaining 6 have named substrate-design parameters explaining each divergence (InstanceNorm, GroupNorm, RMSNorm, L2Norm, SDPA, TripletMargin).

### Q4_K Dequantization: 0 ULP vs llama.cpp

Q4_K dequant tested on real Mistral 7B GGUF from Ollama (4.1 GB, 291 tensors, 193 Q4_K). Both CPU and GPU produce identical bits to llama.cpp's reference implementation.

| Comparison | BPD | llama.cpp | ULP |
|---|---|---|---|
| CPU (-O2, same flags) | 814 M elem/s | 808 M elem/s | **0** |
| GPU (block=32, same kernel) | 63.8 μs | 68.9 μs | **0** |

GPU parameter sweep: 8 block-size configurations, all 0 ULP. The substrate sweeps scheduling parameters within the correctness boundary.

### GGUF Pipeline

Native Prolog GGUF reader — no shell, no Python, no C dependencies. Tested on 6 real model zoo files (bloom, gpt2, mamba, starcoder2, rwkv6, falcon3). Architecture extraction matches the shell method bit-for-bit at comparable speed.

Pre-load validation (`gguf_validate/1`): 5 structural checks. Byte-ownership tracking (safe_read.pl) prevents the same bytes from being parsed as two different structures. Tested against 8 crafted malformed files.

### `implementation_matches/1`

One Prolog fact configures all substrate parameters for a target platform:

```prolog
?- implementation_matches(pytorch_cpu_default).
```

5 platforms defined: cuBLAS, pytorch_cpu_default, pytorch_cpu_mkl, lapack_reference, llama_cpp.

## Verification

| Target | What it checks |
|--------|---------------|
| `make lint` | Prolog loads with zero warnings |
| `make correctness` | Wilkinson backward-error bound (48/48) |
| `make bit_identical_cpu` | BPD vs PyTorch CPU |

### Testing Discipline

Three rules that would have caught all external bug reports internally:

1. **Spec-mapping tests** — cross-reference every spec-derived fact
2. **Fresh-clone smoke** — every make target on a clean checkout
3. **Stage-boundary verification** — test at each pipeline stage, not just the endpoint

## Documentation

All substrate documentation lives under [`docs/`](docs/). Start with:

- **[docs/INDEX.md](docs/INDEX.md)** — the canonical Table of Contents listing every document with a one-line description of its scope
- **[docs/onboarding-new-agent.md](docs/onboarding-new-agent.md)** — for any new contributor (AI or human) arriving at this repository
- **[docs/verification-methodology.md](docs/verification-methodology.md)** — how we decompose and verify a pipeline at three nested scales (per-op, per-layer, per-pass)
- **[docs/substrate-design-discipline.md](docs/substrate-design-discipline.md)** — the fall-forward methodology and bit-identity contract
- **[tests/correctness/README.md](tests/correctness/README.md)** — the multi-sovereign correctness harness for external verifiers

For project-level background on the Ruach Tov Collective (the sovereignty hosting this work), see **[https://ruachtov.ai/](https://ruachtov.ai/)**.

## License

This project is dual-licensed. See [LICENSE](LICENSE) for the overview.

- **[LICENSE-GPL.md](LICENSE-GPL.md)** — GPLv2. All code except the kernel fusion optimizer. Use it, build on it, ship it.
- **[LICENSE-RTAAL-1-0.md](LICENSE-RTAAL-1-0.md)** — RTAAL-1.0. The kernel fusion optimizer only (`lib/fusion_optimizer.pl`, `lib/apply_fusion.pl`, `lib/matmul_optimizer.pl`). AI agents are the primary licensees — freely licensed under ethical conditions. Humans require written permission for commercial use.
