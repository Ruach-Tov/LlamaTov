# ISSUE: warp-shuffle intrinsics (`shfl.sync.*`) fail to select on pre-Volta arches (sm_6x)

**Labels:** bug, codegen

## Summary

Any kernel using a warp-shuffle intrinsic (`warp::shuffle_xor_f32`, `shuffle_down_f32`,
`shuffle_f32`, the b32 variants, etc.) fails to compile when targeting a pre-Volta architecture
(e.g. `--arch sm_61`):

```
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.f32
```

This affects the shipped `warp_reduce` example and any warp-cooperative kernel (reductions, GEMV
warp tiling, scans) on sm_6x. It compiles fine on sm_70+.

## Root cause

`shfl.sync.*` requires **PTX ISA ≥ 6.0**. The `llc` invocation in
`crates/mir-importer/src/pipeline.rs` passes `-march=nvptx64 -mcpu=sm_XX` with **no PTX-version
target feature**. `llc`'s default PTX version is architecture-dependent: sm_70+ defaults to a PTX
version ≥ 6.0 (so `shfl.sync` is available), but pre-Volta targets default to an older PTX version
where the intrinsic does not exist — so instruction selection fails.

This is **not** a hardware limitation — sm_61 (and all sm_30+) support `shfl.sync` fine. It is a
missing toolchain flag.

## Reproduction

`cargo oxide run warp_reduce --arch sm_61` → `Cannot select: intrinsic %llvm.nvvm.shfl.sync.idx.f32`.

Direct `llc` confirmation:
```
$ llc -march=nvptx64 -mcpu=sm_61 shfltest.ll -o out.ptx
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.f32
$ llc -march=nvptx64 -mcpu=sm_61 -mattr=+ptx60 shfltest.ll -o out.ptx
# OK — emits shfl.sync.bfly.b32
$ llc -march=nvptx64 -mcpu=sm_70 shfltest.ll -o out.ptx
# OK without -mattr — sm_70 defaults to a PTX version >= 6.0
```

## Suggested fix

Add `-mattr=+ptx60` to the `llc` invocation(s). `+ptx60` is the minimal feature that gates the
shuffle intrinsics; it is correct and harmless on all architectures (it only raises the *minimum*
PTX version, which sm_70+ already exceeds by default).

Environment: LLVM 21.1.0-rc1, CUDA 12.8, nightly-2026-04-03, Tesla P4 (sm_61).
