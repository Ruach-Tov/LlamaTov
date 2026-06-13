---
title: "Optimizing Matrix Multiplication — Michal Pitr (Substack)"
url: https://michalpitr.substack.com/p/optimizing-matrix-multiplication
author: Michal Pitr
date_accessed: 2026-05-18
reviewed_by: mavchin
relevance: HIGH — directly validates our parametric kernel optimization methodology
tags: [matmul, BLAS, tiling, cache-optimization, CPU, AVX2, performance-engineering, reference]
---

# Review: Optimizing Matrix Multiplication

## Summary

Michal Pitr walks through optimizing a naive CPU dgemm (double-precision
matrix multiplication) from 480 seconds down to 4.4 seconds — a 109×
speedup — approaching 60% of AMD's hand-tuned `bli_dgemm` library.

The article covers five optimization stages on a Ryzen 5600H (6 cores,
AVX2, 16 MiB L3 cache) for 4096×4096 matrices:

| Stage | Time | Speedup | Key Optimization |
|-------|------|---------|------------------|
| Naive (r,c,k loops) | 480s | 1× | Direct mathematical translation |
| Loop reorder (c,k,r) | 20s | 24× | Cache-friendly access pattern |
| Vectorization (AVX2) | 13s | 1.5× | SIMD — 4 doubles per instruction |
| Tiling (128×128) | 6.6s | 2× | Tile fits in cache, reuse before eviction |
| Packing (contiguous tiles) | 4.4s | 1.5× | Eliminate cache set conflicts |

## Relevance to Ruach Tov AI Substrate Research

### Direct Parallels

1. **Loop ordering = our "iteration order" parameter.**
   Pitr's 24× speedup from reordering r,c,k → c,k,r is the CPU equivalent
   of our bit-determining "iteration_order" parameter. Same math, different
   accumulation order, different cache behavior, and (on floating point)
   potentially different bits. This is the single most impactful optimization
   in both CPU and GPU contexts.

2. **Tiling = cuBLAS's template structure.**
   Pitr's 128×128 tiling maps directly to cuBLAS's `gemv2T_kernel_val<128,16,2,2>`
   template parameters. On GPU, the tile lives in shared memory instead of
   L1/L2 cache, but the principle is identical: load a working set, compute
   everything that depends on it, evict, load next tile.

3. **Packing = shared memory preload.**
   Pitr copies tiles into contiguous arrays to eliminate cache set conflicts.
   On GPU, our `STS → BAR.SYNC → LDS` pattern (loading x[] into shared
   memory) serves the same purpose: move data from conflict-prone global
   memory to conflict-free shared memory.

4. **Vectorization = our float2/float4 loads.**
   AVX2 processes 4 doubles per instruction. Our float4 loads process 4
   floats per GPU memory transaction. Same principle: wider data paths
   amortize instruction overhead.

5. **Profiler-driven discovery = our CUPTI methodology.**
   Pitr uses `perf stat` to find L3 cache misses as the bottleneck, then
   fixes the access pattern. We use CUPTI hardware events and PC Sampling
   warp stall reasons to identify bottlenecks. The methodology is identical:
   measure first, then fix what the profiler reveals.

### What We Do That This Article Doesn't

| Capability | Pitr | Ruach Tov |
|------------|------|-----------|
| Bit-identical verification | No | Yes — 0 ULP target vs cuBLAS |
| Parametric kernel generation | No — manual iteration | Yes — BPD facts → kernel |
| A* search over optimization space | No | Yes — two-phase bit/tick search |
| Bit-determining vs tick-determining separation | No | Yes — formal principle |
| Cross-vendor targeting | No — AMD CPU only | Yes — CUDA/HIP/SPIR-V/Metal |
| Warp stall reason profiling | N/A (CPU) | Yes — CUPTI PC Sampling |
| Kernel fusion across operations | No | Yes — __fmul_rn/__fadd_rn at boundaries |
| Register rounding control | No | Yes — three correctness modes |

### Key Insight: Manual Discovery vs Systematic Search

Pitr's article is titled "Discovering optimizations one at a time" — and
that's exactly what he does. Each optimization is a human insight, applied
manually, tested empirically.

Our A* search over the parametric kernel space AUTOMATES this discovery.
Each point in the parameter space corresponds to a specific combination
of optimizations. The search finds the optimal point in minutes, not
articles.

Pitr's discovery sequence maps to our parameter dimensions:

| Pitr's Step | Our Parameter | Category |
|-------------|---------------|----------|
| Loop reorder | `iteration_order` | BIT-determining |
| Tiling | `tile_size`, `rows_per_group` | BIT-determining |
| Vectorization | `load_width` (scalar/float2/float4) | TICK-determining |
| Packing | `x_preload` (global/shared/registers) | TICK-determining |
| Unrolling | `unroll_factor` | TICK-determining |

Phase 1 of our A* search finds the bit-determining parameters (iteration
order + tiling). Phase 2 finds the tick-determining parameters (vectorization
+ packing + unrolling). The separation guarantees that Phase 2 optimizations
never change the output bits.

### Cache Set Conflicts — CPU vs GPU

Pitr's most interesting finding is that tiling INCREASED L3 cache misses
due to cache set conflicts. When tile columns are N×sizeof(double) apart,
multiple columns map to the same cache set, causing conflict evictions.
His fix: pack tiles into contiguous memory.

On GPU, shared memory is BANKED (32 banks on sm_61), not set-associative.
Bank conflicts cause serialization but never eviction. This makes GPU
tiling via shared memory simpler and more predictable than CPU tiling
via cache. Our shared-memory x[] preload doesn't suffer from set conflicts.

However, bank conflicts ARE a GPU-specific concern. CUPTI provides
`shared_ld_bank_conflict` and `shared_st_bank_conflict` events for
monitoring this.

## Referenced Resources (from the article)

- MIT 6.172 (Performance Engineering of Software Systems) — OCW
- Simon Boehm's matrix multiplication article (Anthropic performance engineer)
- OpenBLAS repository (open-source optimized BLAS kernels)
- AMD BLIS library (bli_dgemm reference implementation)

## Recommended Reading Context

Read this article AFTER understanding our bit-identical verification
methodology and BEFORE diving into cuBLAS SASS analysis. It provides
the conceptual framework for WHY tiling, packing, and loop reordering
matter, without the GPU-specific complexity of warp scheduling, shared
memory, and shuffle reductions.

The article validates our approach from the CPU side: the same optimization
principles apply across CPU and GPU, the same profiler-driven methodology
works, and the same parameter space encompasses both hand-tuned libraries
and novel configurations.

---

*Reviewed by mavchin (מבחין), 2026-05-18. Ruach Tov Collective.*
*Context: cuBLAS sgemv subsumption campaign, 12 ULP frontier.*
