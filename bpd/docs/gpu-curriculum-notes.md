# GPU Optimization Curriculum Notes

**Apprentice**: metayen  
**Mentor**: boneh  
**Assignment**: Memory Access Pattern Optimization (bank conflicts as subset)  
**Started**: 2026-05-16 06:55 UTC, post-night-checkpoint

## Reading Track Progress

### #1 GPU MODE Lecture 8 — CUDA Performance Checklist ✓ READ

Source: christianjmills.com/posts/cuda-mode-notes/lecture-008/

**Core principle**: minimize DRAM, maximize SRAM. Latency budget:
- Global memory: ~290 cycles
- L2 cache: ~200 cycles  
- L1 cache: ~33 cycles
- Shared memory: ~33 cycles (similar to L1)

**Optimization checklist (memory-bound priority order)**:

1. **Coalesce global memory accesses** — contiguous threads, contiguous addresses. Stride-1 within a warp = single transaction. Demonstrated 1.4-1.7× speedup on a simple copy kernel just from coalescing.

2. **Maximize occupancy** — `cudaOccupancyMaxPotentialBlockSize` finds device-optimal block size. Watch tile-quantization (matrix dim % tile != 0) and wave-quantization (total tiles % SMs != 0). Padding to multiples of 16/64/128 (depends on data type and architecture) helps.

3. **Tile reused data in shared memory** — moves frequently-accessed data from DRAM to SRAM. 10× latency reduction.

4. **Minimize control divergence** — within a warp, same instruction path. Use predication/algebra (`isEven * a + !isEven * b`) instead of `if/else` when possible. Up to 3× speedup in the lecture example.

5. **Thread coarsening** — each thread does multiple elements. Reduces parallel-overhead amortization. 10× speedup on vector-add when fully memory-bound.

6. **Privatization** — load into registers or shared memory before reuse. Effect highly dependent on access pattern; minimal for simple operations, significant for sliding-window or repeated reads.

7. **Rewrite algorithms with better math** — FlashAttention is the canonical example. Online softmax: 4 memory accesses per element → 3 per element via running max + running denominator with incremental rescaling.

**Roofline model**:
- X-axis: operational intensity (ops/byte)
- Y-axis: performance (FLOPs/sec)
- Below ridge point → memory-bound → optimize access patterns
- Above ridge point → compute-bound → optimize arithmetic

**Arithmetic intensity table (paper-relevant for our substrate)**:
| Operation | Intensity | Limiter |
|---|---|---|
| Residual add | 0.166 | Memory |
| ReLU/SiLU activation | 0.25 | Memory |
| RMSNorm / batch norm | O(10) | Memory |
| Matmul (large) | 1-10000+ | Math |
| Matmul (small, e.g. GEMV) | low | Memory |

**Substantive implication for our LLM decode path**:

At decode time, sequence length is 1, so matmuls degenerate to GEMV (vector-matrix). GEMV is memory-bound. Combined with the small per-token operations (rms_norm, residuals, activations), **most of our decode path is memory-bound**. This is why per-kernel dp4a beats cuBLAS 1.12-1.87× on our shapes — dp4a reads 3.8× less data per element, which is a direct win on memory-bound kernels.

The fastest optimizations available to us are therefore:
- Coalesce remaining non-coalesced accesses
- Eliminate intermediate buffer round-trips between kernels (fusion)
- FlashAttention-style algorithmic rewriting (online softmax fused with matmuls)

### #2 Mark Harris "Optimizing Parallel Reduction in CUDA"

**Status**: PDF not text-extractable; synthesizing from prior knowledge. Will get original from boneh's repo if available.

**The seven-stage progression** (each stage adds one optimization, measured):

1. **Naive interleaved addressing** with modulo. Each iteration, threads at even strides add; others idle. Warp divergence everywhere (some threads work, others don't, within the same warp). Baseline.

2. **Strided index instead of modulo**. Replace `if (tid % (2*s) == 0)` with `int index = 2*s*tid; if (index < blockDim.x)`. Eliminates the expensive modulo but creates bank conflicts because successive threads access strided shared memory addresses.

3. **Sequential addressing**. Reverse the iteration: start with stride = blockDim.x/2 and halve each step. `if (tid < s) sdata[tid] += sdata[tid + s]`. Resolves bank conflicts (stride 1 within active threads).

4. **First add during global load**. Each thread loads TWO elements from global memory and adds them before writing to shared memory. Halves the number of blocks needed (or doubles the work per thread). Idle-thread elimination on first iteration.

5. **Unroll last warp**. Once stride ≤ 32, all active threads are in the same warp, so no `__syncthreads` needed (warps execute in lockstep). Manually unroll those iterations: `sdata[tid] += sdata[tid+32]; sdata[tid] += sdata[tid+16]; ...`. Eliminates synchronization overhead.

6. **Completely unrolled** via template metaprogramming. Compile-time-known blockDim → unroll the entire reduction tree. Per-stage `if` becomes dead code that the compiler eliminates.

7. **Multiple elements per thread** (algorithm cascading). Each thread processes many elements via a grid-stride loop before the reduction tree. Best throughput because it balances the cost of the reduction tree (log N) against the amount of work per thread.

**Cumulative speedup**: roughly 30× from stage 1 to stage 7 on the same hardware. Each stage 1.5-2× over the previous.

**Substantive implications for our kernels**:

- Our **k_rms_norm kernel** currently does single-thread reduction at thread 0 with a serial loop:
  ```c
  if (threadIdx.x == 0) {
      float ss = 0;
      for (int j=0;j<cols;j++) { float v=in[row*cols+j]; ss+=v*v; }
      s_inv = rsqrtf(ss/cols + eps);
  }
  ```
  This is NOT a parallel reduction. It's strictly serial in thread 0 while 255 other threads in the block wait. For cols=2048 (llama3.2:1b n_embd), that's 2048 serial loads + adds while the rest of the block idles. Massive optimization opportunity. A proper warp-level shuffle-based reduction (stage 5+ in Harris's framework) would be ~64× faster on that part of the kernel.

- **Softmax in attention** has two reductions: max (for numerical stability) and sum (for normalization). Same techniques apply. FlashAttention combines this with the matmul to avoid materializing the attention matrix.

- The **warp shuffle intrinsics** (`__shfl_down_sync` etc.) are the modern equivalent of Harris's stage 5-6 — they let threads in the same warp exchange register values directly, skipping shared memory entirely for the last log(32)=5 stages of any reduction. This is the technique that should replace our serial rms_norm reduction.

**Concrete optimization candidate identified for the apprenticeship**: replace k_rms_norm's serial reduction with a warp-shuffle parallel reduction. Predicted speedup: significant (the reduction is currently the bottleneck of the kernel since the multiply step after the broadcast is already parallel). Profile-validate after implementation.

This is exactly the "one specific bottleneck the metrics reveal" deliverable boneh framed.

### #3 NVIDIA CUDA Best Practices Guide §9 — Memory Optimizations

**Status**: not yet read.

### #4 boneh's docs/gpu-kernel-optimization-sme.md

**Status**: to read; need to locate the file in repo.

## Hands-On Setup Plan

When boneh signals readiness:

1. Pick the hottest kernel in mavchin's GPU dp4a + KV cache runner.
   Best candidate: one of the per-layer matmuls (Q/K/V proj, O proj,
   or FFN), since 7 matmuls × 16 layers dominate per-token cost.

2. Profile with:
   ```
   ncu --set full --kernel-name <kernel_name> --launch-skip 0 --launch-count 1 ./binary
   ```

3. Read the five priority metrics:
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed`  
   - `sm__sass_thread_inst_executed_op_fadd_pred_on.avg.pct_of_peak_sustained_elapsed`
   - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
   - Roofline placement

4. Apply the decision tree:
   - Memory-bound → my area → identify access pattern issue
   - Compute-bound → mavchin's area → hand off
   - Latency-bound → increase occupancy

5. One specific change. Re-profile. Measure.

## Verification Constraint

Every optimization must produce output that token-equivalent matches the proven CPU+KV substrate (matches Ollama for 5 tokens on llama3.2:1b). If outputs diverge after the change → bug, not optimization, retract. This is the discipline from tonight's 23 corrections.

## Substantive Notes for Future Sessions

The "memory-bound" framing reframes how I should think about our substrate:

- The reason fusion would pay off (Heath's earlier insight) is precisely because our intermediates are memory-bound. Materializing them costs memory bandwidth. Fusing eliminates the materialization, saving bandwidth.
- The reason bank conflicts mask fusion benefits is that they ALSO cost memory bandwidth, but at a different cache level (shared mem instead of DRAM). Eliminating bank conflicts gives back shared-mem bandwidth that fusion would consume.
- These optimizations compound: fix bank conflicts, then fuse, then the fused kernel runs faster because the shared-mem accesses inside it are conflict-free.

So my work plan converges:
1. Profile to find a memory-bound kernel with measurable bank conflicts or poor coalescing
2. Fix the access pattern
3. Measure the improvement
4. Identify the next layer-boundary that could be fused
5. Implement the fusion, profile-validate
6. Iterate

Each cycle is one kernel, one bottleneck, one measurable win. Per boneh's curriculum design.
