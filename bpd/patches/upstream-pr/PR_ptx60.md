# Fix warp-shuffle on pre-Volta arches: pass `-mattr=+ptx60` to `llc`

## Summary

Warp-shuffle intrinsics (`warp::shuffle_xor_f32`, `shuffle_down_f32`, `shuffle_f32`, and their b32
variants) fail to compile on pre-Volta architectures (`--arch sm_61`) with:

```
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.f32
```

This breaks the shipped `warp_reduce` example and **every** warp-cooperative kernel (reductions,
GEMV warp tiling, scans) on sm_6x. The fix is one flag.

## Root cause

`shfl.sync.*` requires **PTX ISA ≥ 6.0**. The `llc` invocation in
`crates/mir-importer/src/pipeline.rs` passes:

```rust
Command::new(&llc_path)
    .arg("-march=nvptx64")
    .arg(format!("-mcpu={}", target))
    // <-- no PTX-version target feature
    .arg(ll_path) ...
```

`llc`'s default PTX version is **architecture-dependent**: sm_70+ defaults to a PTX version ≥ 6.0
(so `shfl.sync` selects), but pre-Volta targets default to an older PTX version where the intrinsic
does not exist — so selection fails. This is *not* a hardware limitation: sm_61 (and all sm_30+)
support `shfl.sync` fine; it is a missing toolchain flag.

## Fix

```diff
         Command::new(&llc_path)
             .arg("-march=nvptx64")
             .arg(format!("-mcpu={}", target))
+            .arg("-mattr=+ptx60")  // shfl.sync etc. need PTX ISA 6.0+
             .arg(ll_path)
```

applied to **both** `llc` invocations in `pipeline.rs` (the `CUDA_OXIDE_LLC` override path and the
auto-detected path). `+ptx60` is the minimal feature that gates the shuffle intrinsics; it is correct
and harmless on all architectures (it raises only the *minimum* PTX version, which sm_70+ already
exceeds by default).

## Reproduction (before this PR)

```
$ cargo oxide run warp_reduce --arch sm_61
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.idx.f32
error: could not compile (device codegen failed)
```

Direct `llc` confirmation of the cause and the arch-dependence:

```
$ llc -march=nvptx64 -mcpu=sm_61 shfltest.ll -o out.ptx
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.f32
$ llc -march=nvptx64 -mcpu=sm_61 -mattr=+ptx60 shfltest.ll -o out.ptx   # OK
$ llc -march=nvptx64 -mcpu=sm_70 shfltest.ll -o out.ptx                  # OK (no flag needed)
```

## Test

New end-to-end smoke example `examples/shfl_sync_smoke/`: a single-warp reduction over `[0,1,...,31]`
via `shuffle_xor_f32`, asserting every lane ends with the sum `496.0`. Prints `SUCCESS`/`FAILURE`.

```
before this PR:  LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.f32  (won't compile)
after  this PR:  warp sum of [0..32) = 496  (expect 496)
                 SUCCESS: shfl.sync.bfly lowered and executed correctly on this arch
```

## Verification

```
$ cargo oxide run shfl_sync_smoke --arch sm_61
  warp sum of [0..32) = 496  (expect 496)
  SUCCESS: shfl.sync.bfly lowered and executed correctly on this arch
```

Generated PTX contains `shfl.sync.bfly.b32`. Device-verified on a Tesla P4 (sm_61, CUDA 12.8). Found
while lowering a q8_0 dp4a GEMV with a warp-shuffle (`tree(shfl_down,5)`) reduction: the kernel was
0-ULP against the nvcc-generated kernel on identical inputs once the shuffles lowered.

Environment: LLVM 21.1.0-rc1 (NVPTX backend), CUDA Toolkit 12.8, Rust nightly-2026-04-03,
Tesla P4 (Pascal, sm_61).

## DCO

Single change + smoke example, signed off (`git commit -s`).

Fixes #<ISSUE_NUMBER>.
