# GPU post-kernel fault diagnostic

`kernel_diag_end(name)` — call after a kernel launch+sync. Returns:
- 0 = clean
- 1 = recoverable trap (read from `active_cycles_in_trap` hardware counter)
- 2 = FATAL fault (misaligned load / OOB / div-by-zero — caught via cudaError)

## Honest limitation (tested, not assumed)
`active_cycles_in_trap` only catches RECOVERABLE traps. A FATAL fault (misaligned
128-bit load, OOB, div-by-zero — the SoA-bug class) DESTROYS the CUDA context, so
the post-fault counter read returns 0. The fatal layer (cudaError) catches those,
but to localize WHICH warp/line, use `compute-sanitizer`.

So: trap counter = recoverable-trap health check; cudaError = fatal detection;
compute-sanitizer = fatal localization. `kernel_diag_end` combines the first two.

Events verified present on Tesla P4 (sm_61): active_cycles_in_trap,
inst_executed_in_trap, active_cycles, inst_executed.

## kernel_guard.cuh — robust kernel launch + fault-detecting fixtures

Three layers of defense, distilled from the SoA-vec128 debugging experience:

1. `GUARD_ALIGN(ptr, 16, "what")` — PRE-LAUNCH: reject misaligned pointers for
   vectorized loads BEFORE the launch (instead of a fatal context abort). The
   lesson from the AoS +2-offset bug: a uint4 load needs 16-byte alignment.
2. `GUARD_LAUNCH(kernel<<<...>>>(...), "name")` — catches FATAL faults
   (misaligned/OOB/div0/bad-config) via cudaGetLastError + cudaDeviceSynchronize.
   Points to compute-sanitizer for warp/line localization.
3. `GUARD_VERIFY(out, ref, n, "name")` — catches SILENT WRONG OUTPUT: a kernel
   that runs clean (no fault, no trap) but produces WRONG numbers (e.g. stale
   blocks_per_row reading the wrong layout). NO fault counter sees this — only a
   bit-identity check vs a reference does. THE most dangerous fault class.

Self-test (kernel_guard_test.cu) proves all three: rejects misaligned ptr,
detects silent-wrong (bad_factor=0 -> ndiff=1024/1024), passes the correct kernel.

### The fault taxonomy (learned this session)
| class | example | detected by | NOT by |
|-------|---------|-------------|--------|
| fatal | misaligned uint4, OOB, div0 | cudaError code | trap counter (context dies) |
| recoverable trap | (rare) | active_cycles_in_trap | error code |
| silent wrong | stale param, wrong layout | bit-identity fixture | any fault counter |
