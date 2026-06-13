# cupti-from-prolog: a CUPTI measurement instrument

Scientifically quantify GPU inefficiencies — host↔device copy traffic, per-kernel time,
(and, extensible) occupancy and warp stalls — exposed as **queryable Prolog facts**.

## Why this exists

`bpd/lib/cupti_profile.pl` is a *reasoning* layer: it takes a `StallList` and suggests
optimizations. But it had **no acquisition layer** — nothing actually queried the GPU. This
tool is the missing **body**: it captures real CUPTI Activity records and turns them into
facts the reasoning layer (and you) can query.

## Pipeline

```
libcupti_trace.so      (C, CUPTI Activity API)   → captures memcpy + kernel records
        ↓  LD_PRELOAD into your CUDA python target
*.facts                (Prolog terms)            memcpy(Kind,Bytes,Start,End,Dur).
        ↓                                          kernel(Name,Start,End,Dur,GridX,BlockX).
cupti_facts.pl         (Prolog reasoning)        → copy_inefficiency/1, kernel_time_summary/0
```

## Build

```sh
./build.sh                 # → libcupti_trace.so
```

## Capture

```sh
CUPTI_TRACE_OUT=/tmp/cupti.facts \
LD_PRELOAD=$PWD/libcupti_trace.so \
LD_LIBRARY_PATH=$CUDA_MERGED/lib \
python3 example_driver.py
```

## Analyse

```prolog
?- use_module(cupti_facts).
?- load_cupti('/tmp/cupti.facts').
?- copy_summary.                  % all host<->device traffic by kind
?- copy_inefficiency(Verdict).    % the headline: which copies are avoidable
?- kernel_time_summary.           % per-kernel GPU time
```

## What it found (first run, qwen2.5-0.5b-q8_0 decode on Tesla P4)

- **Copy inefficiency:** per steady-state token, ~1.2 MB of device→host readback is
  **avoidable** — the full logits vector pulled back only for `argmax`. A device-side
  argmax/sample returns one token (4 B), saving ~1,215,484 B/token of host crossing.
- **Kernel time:** `k_q8_0_gemv` is **89% of all GPU kernel time** (35.6 µs of ~40 µs).
  Definitively the compute lever — the Q8_0 GEMV layout is what to optimize next.

## Facts schema

| Fact | Fields |
|---|---|
| `memcpy(Kind, Bytes, StartNs, EndNs, DurationNs)` | Kind ∈ {htod, dtoh, dtod, htoh, other} |
| `kernel(Name, StartNs, EndNs, DurationNs, GridX, BlockX)` | |

## Extending

The tracer enables `CUPTI_ACTIVITY_KIND_MEMCPY` and `CONCURRENT_KERNEL`. To add occupancy
and warp-stall metrics, enable the CUPTI metric/PC-sampling APIs in `cupti_trace.c` and emit
new fact shapes (e.g. `kernel_metric(Name, occupancy, Value)`, `stall(Name, Reason, Pct)`),
then extend `cupti_facts.pl` to feed `cupti_profile.pl`'s `StallList` — connecting body to head.

## Occupancy extension (kernel_launch facts)

The tracer also emits, per kernel:
`kernel_launch(Name, Regs, StaticShmem, DynShmem, LocalMem, BlockSize, GridCount, DevId).`

`cupti_facts.pl` computes THEORETICAL occupancy from these (sm_61 limits), via
`occupancy_summary/0`. Diagnostic value: it tells you whether a slow kernel is
occupancy-limited (fixable by launch config) or compute/bandwidth-bound (needs a layout
rewrite).

**Key finding (qwen decode, P4):** `k_q8_0_gemv` (89% of GPU time) runs at **100% theoretical
occupancy** — so it is NOT occupancy-limited; it is memory-bandwidth-bound. The Q8_0 GEMV
layout work must therefore target MEMORY THROUGHPUT (coalescing, wider loads), not occupancy.
The profiler scientifically rules out the wrong lever.

## sm_61 PC sampling: WORKS — use the existing bridge, not this tracer

**CORRECTION** (the earlier claim here — "PC sampling is unsupported on sm_61" — was WRONG;
Mavchin caught it). PC sampling *does* work on Pascal sm_61. This tracer's
`CUPTI_PC_SAMPLING=1` path returned zero because of a bug in **this file**, not the hardware:
it called `cuptiActivityEnable(PC_SAMPLING)` but **never called
`cuptiActivityConfigurePCSampling`**, and ran in an `LD_PRELOAD` constructor *before any CUDA
context existed*. Enable-without-configure-and-no-context = zero samples.

The **working stall instrument already exists**: `bpd/lib/bpd_cupti_profile.c` +
`bpd/lib/cupti_bridge.c` configure PC sampling correctly
(`cuptiActivityConfigurePCSampling` on the current context) and expose
`cupti_stall_report(-StallList)` as a Prolog foreign predicate that feeds `cupti_profile.pl`
directly. Memory `7d0492d3` records 625 real PC samples from the P4 (constant_memory 38%,
inst_fetch 35%, memory_dependency 16%, …).

**Use the right tool per question:**
- **Warp stalls** → `cupti_stall_report` (Mavchin's `bpd_cupti_profile.c`).
- **Exact HW counters** (coalescing via `gld_inst_*bit` / `fb_read_sectors`) →
  `cuptiEventGroupReadAllEvents` (memory `14971398`, 80 counters).
- **Per-token memcpy traffic** (the `copy_inefficiency` verdict) and
  **theoretical occupancy** (from launch config) → *this* tracer. Complementary, not a
  replacement for the stall/counter paths that already work.

## Stall profiling via cupti-from-prolog (the WORKING path)

NOT Python/ctypes (the GIL blocks CUPTI's activity-buffer callback thread, causing a
deadlock). The instrument is swipl loading the foreign-predicate bridge. Mavchin's
bpd_cupti_profile.c + cupti_bridge.c expose cupti_init/flush/stall_report; q8_gemv_launcher.c
(this dir) adds run_q8_gemv/4 so OUR kernel runs in the same swipl process
(init -> run -> flush -> report).

Build:
  swipl-ld -shared -o cupti_q8 cupti_bridge.c bpd_cupti_profile.c q8_gemv_launcher.c \
    combined_install.c -lcupti -lcuda -I<cupti_inc> -L<cupti_lib> -L<libcuda_dir> -L<swipl_lib>

Run:
  ?- use_foreign_library(cupti_q8, install_cupti_q8),
     cupti_init, run_q8_gemv("q8_gemv.cubin", 896, 4864, 300),
     cupti_flush, cupti_stall_report(Stalls).

FINDING (k_q8_0_gemv, M=896 K=4864 FFN shape, 6.13M PC samples, Tesla P4):
  memory_dependency 89.9% | exec_dependency 4.3% | none 2.0% | inst_fetch 1.6% | other 1.3%
The GEMV is ~90% memory-dependency-bound: warps stalled waiting on weight loads. Confirms the
occupancy reading (100% occ, not occupancy-limited) with a hard number: the LOAD PATTERN is
the lever. Next: sectors_per_load (Mavchin's event-counter path) to decide coalesced vs
straddling, the SoA go/no-go.
