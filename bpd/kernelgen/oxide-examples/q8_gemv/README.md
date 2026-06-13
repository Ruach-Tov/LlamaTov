# q8_0 GEMV lowered to Rust / cuda-oxide — the backend-neutrality thesis proof

**Result: `max_ulp = 0`** — the engine's q8_0 dp4a GEMV, regenerated as a pure-Rust `#[kernel]`,
compiled by **cuda-oxide (Rust → LLVM → PTX, NO nvcc)** and run on a Tesla P4 (sm_61), is
**bit-identical** to the reference. The same `reduction_order` fact that drives the CUDA emitter
produces, in a completely different backend, the identical float result.

## What this proves

The thesis (`ir_param_axes.pl` facts are backend-neutral): an optimization expressed as a *fact*
translates to all backends. Here the dominant op (the q8_0 GEMV, 77.6% of decode) is lowered from
the SAME fact — `reduction_order(q8_gemv, canonical_serial)` — to Rust/cuda-oxide, and the output
matches the reference to the last bit. dp4a is exact integer arithmetic (`(w as i32)*(x as i32)`
summed == `__dp4a`, no rounding); the dequant scaling and sequential block accumulation match the
CUDA `emit_q8_0_gemv_canonical_serial` order exactly.

## Build / run

```sh
# cuda-oxide checkout (NVLabs c8b3103 + the two ruachtov sm_61 patches), built once:
cd /tmp/cuda-oxide-gemv
source ~/Ruach-Tov/bpd/kernelgen/build/build-env.sh   # nightly-2026-04-03, LLVM-21, CUDA_OXIDE_LLC
./target/release/cargo-oxide run q8_gemv --arch sm_61
# => max_ulp = 0  ✓ 0-ULP: oxide q8_0 GEMV BIT-IDENTICAL to reference
```

The example crate lives at `crates/rustc-codegen-cuda/examples/q8_gemv/` in the cuda-oxide tree;
`src/main.rs` here is the canonical copy (kernel + host + bit-check reference).

## Two reduction orders, both 0-ULP — and the `+ptx60` patch that unblocked warp shuffles

Two reduction orders are proven, both bit-identical to their CUDA counterparts:

1. **`canonical_serial`** — one thread per row: 32 fma-folded lane partials over strided blocks,
   then a 5-level shuffle-down tree merge, all in-thread (no warp shuffle). This lowered cleanly
   from the start.
2. **warp-shuffle (`tree(shfl_xor,5)`)** — the *production* `tiled_v4` order: one warp per row,
   lane-strided blocks, a `warp::shuffle_xor_f32` butterfly reduction, lane 0 writes. **0-ULP** vs
   both a CPU reference and the nvcc kernel.

### The `.sync` shuffle "Cannot select" issue — root-caused and patched

The warp-shuffle order *initially* failed:
```
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.f32
```
The first reflex — "Pascal can't do `.sync` shuffles" — was **wrong**. sm_61 supports `shfl.sync`
fine. The real cause: cuda-oxide invoked `llc -march=nvptx64 -mcpu=sm_61` with **no PTX-ISA-version
target feature**. The `shfl.sync.*` intrinsics require PTX ISA ≥ 6.0; without `-mattr=+ptx60`, llc
defaults to an older PTX version where they don't exist, so selection fails. Verified directly:
`llc -march=nvptx64 -mcpu=sm_61` fails on the intrinsic; **adding `-mattr=+ptx60` succeeds** and
emits correct `shfl.sync.bfly.b32`. (The shipped `warp_reduce` example fails for the same reason and
is fixed by the same flag.)

**The fix** (`bpd/patches/ruachtov-ptx60-shfl.patch`): add `-mattr=+ptx60` to cuda-oxide's two llc
invocations in `mir-importer/src/pipeline.rs`. `+ptx60` is the surgical minimum — the exact feature
that gates the shuffle intrinsics. With it, the warp-shuffle GEMV compiles and runs **0-ULP** on
sm_61. No public cuda-oxide issue existed for this; it's a 2-line, broadly-useful fix (warp
reductions, GEMV warp tiling, any cooperative kernel need it).

## Cross-backend gate (the strongest claim)

The honest cross-backend bit-identity claim on this hardware: the CUDA emitter's
`emit_q8_0_gemv_canonical_serial` and this oxide kernel produce identical output on identical inputs
— SAME fact, two backends (nvcc-CUDA and Rust-oxide), 0 ULP. That is the cross-backend pair-gate
surface (a new gate for the referee: `oxide_gemv(x) XOR cuda_gemv(x) == 0`).
