# CUPTI stall-reason diagnostics — reading the data

A field guide for interpreting `bpd_cupti_profile` PC-sampling stall percentages,
built from real findings on the P4 (sm_61). The stall breakdown tells you *what
to fix* — but only if you read it right. Each pattern below maps a signature to a
cause and a lever.

## The stall reasons (PC-sampling, % of samples)
| Reason | Means |
|---|---|
| **No stall (issuing)** | warp issued an instruction this cycle (good — higher is better) |
| **Execution dependency** | waiting on a prior instruction's result (FFMA latency, ILP-bound) |
| **Memory dependency** | waiting on a load/store (bandwidth/latency-bound) |
| **Instruction fetch** | waiting on the instruction cache (loop overhead, big code) |
| **Not selected** | warp was READY but the scheduler issued a *different* warp |
| **Pipe busy** | the functional unit (FMA/etc) was busy |
| **Synchronization** | waiting at a barrier (`__syncthreads`) |
| **Memory throttle** | the memory subsystem is saturated |

## Diagnostic patterns (signature → cause → lever)

### "Not selected" dominant (>25%), everything else low
**Cause:** the SM has *more eligible warps than it can issue from* — too many
threads doing too little, OR (the trap) an **over-subscribed launch**: a
thread-per-output kernel launched with warp-per-output geometry floods the SM
with warps that immediately early-return.
**Lever:** check the **launch geometry**, NOT occupancy. (2026-06-08: maxpool
launched `total*32` threads for a thread-per-output kernel → 31/32 idle → 31%
Not-selected, ~4x slow. Fix: launch `total` threads. → see the `// LAUNCH:`
contract + `test_launch_contract.py`.)
**How to confirm:** profile fused vs a variant — if stalls + regs are *identical*
across a change that should matter, the bottleneck is structural (launch), not the
kernel's compute.

### "Execution dependency" dominant (>30%)
**Cause:** the FFMA/op dependency chain isn't hidden — too few warps resident to
cover the latency (low occupancy), OR genuinely serial arithmetic.
**Lever:** raise occupancy. Cap registers via `__launch_bounds__(threads, blocks)`
to fit more blocks/SM. (2026-06-08: conv 128 regs → 1 block/SM, 37% exec-dep;
`__launch_bounds__(512,2)` → 64 regs, 2 blocks/SM, 28% exec-dep, 1.2x faster.)
**The floor:** pushing too far (40 regs, 3 blocks) spills registers → memory-dep
explodes. The sweet spot is measured, not assumed.

### "Memory dependency" dominant (>30%)
**Cause:** bandwidth- or load-latency-bound. The kernel is waiting on global memory.
**Lever:** improve coalescing, stage to shared memory, or increase arithmetic
intensity (tile). Pipelining (prefetch) helps *only* here — confirm mem-dep is
high before building it. (Conv was 13% mem-dep → pipelining was correctly rejected.)

### "Synchronization" high
**Cause:** threads idling at `__syncthreads`. Imbalanced work per thread, or too
many barriers.
**Lever:** reduce barrier count, balance the per-thread loop.

## The meta-lesson
**Don't reason about *why* from timing alone — measure the stall breakdown.**
Twice (2026-06-08) a timing result was explained by a *wrong* mechanism until
CUPTI showed the truth: (1) "fusion is neutral because it perturbs codegen" — FALSE,
the kernels were stall-identical, the eliminated work was just ~0; (2) "the pool is
slow because the kernel is latency-bound" — FALSE, it was an over-subscribed launch.
One CUPTI measurement replaced each wrong guess with the real cause and lever.
