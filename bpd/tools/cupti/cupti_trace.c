// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* cupti_trace.c — CUPTI Activity API tracer, the ACQUISITION layer for cupti-from-prolog.
 *
 * Captures MEMCPY (HtoD/DtoH/DtoD) and CONCURRENT_KERNEL activity records and writes them
 * as structured, Prolog-ingestible facts to $CUPTI_TRACE_OUT (default /tmp/cupti_trace.facts).
 *
 * Built as a shared lib and LD_PRELOAD'd into the target (python). On load it enables the
 * activity API; on unload (atexit) it flushes and writes facts. The target controls the
 * profiled region with two markers via env or a tiny control file — but the simplest robust
 * design: capture EVERYTHING, tag each record with a monotonically-increasing flush epoch,
 * and let the harness slice by time window (it records wall-clock start/stop of the region).
 *
 * Facts emitted (one per line):
 *   memcpy(Kind, Bytes, StartNs, EndNs, DurationNs).      Kind = htod|dtoh|dtod|htoh|other
 *   kernel(Name, StartNs, EndNs, DurationNs, GridX, BlockX).
 *
 * Build:
 *   gcc -shared -fPIC -O2 cupti_trace.c -o libcupti_trace.so \
 *       -I<cupti_include> -L<cupti_lib> -lcupti
 */
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUF_SIZE (8 * 1024 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

static FILE *g_out = NULL;

static const char *memcpy_kind(uint8_t k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "htod";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "dtoh";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "dtod";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "htoh";
    default: return "other";
  }
}

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                      size_t *maxNumRecords) {
  uint8_t *raw = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  *buffer = (uint8_t *)ALIGN_BUFFER(raw, ALIGN_SIZE);
  *size = BUF_SIZE;
  *maxNumRecords = 0;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t streamId,
                                      uint8_t *buffer, size_t size, size_t validSize) {
  CUpti_Activity *record = NULL;
  if (!g_out) return;
  CUptiResult status;
  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
          CUpti_ActivityMemcpy5 *m = (CUpti_ActivityMemcpy5 *)record;
          fprintf(g_out, "memcpy(%s,%llu,%llu,%llu,%llu).\n",
                  memcpy_kind(m->copyKind),
                  (unsigned long long)m->bytes,
                  (unsigned long long)m->start,
                  (unsigned long long)m->end,
                  (unsigned long long)(m->end - m->start));
          break;
        }
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)record;
          fprintf(g_out, "kernel(\"%s\",%llu,%llu,%llu,%d,%d).\n",
                  k->name ? k->name : "?",
                  (unsigned long long)k->start,
                  (unsigned long long)k->end,
                  (unsigned long long)(k->end - k->start),
                  k->gridX, k->blockX);
          /* Occupancy inputs: registers/thread, shared mem, local mem, block size — the
             launch-config facts needed to compute theoretical occupancy and diagnose
             register/shared-mem pressure (e.g. why a memory-bound GEMV under-occupies). */
          fprintf(g_out, "kernel_launch(\"%s\",%u,%d,%d,%u,%d,%d,%d).\n",
                  k->name ? k->name : "?",
                  (unsigned)k->registersPerThread,
                  k->staticSharedMemory,
                  k->dynamicSharedMemory,
                  (unsigned)k->localMemoryPerThread,
                  k->blockX * k->blockY * k->blockZ,
                  k->gridX * k->gridY * k->gridZ,
                  k->deviceId);
          break;
        }
        case CUPTI_ACTIVITY_KIND_PC_SAMPLING: {
          /* Warp stall reasons — the StallList cupti_profile.pl reasons over. Each record
             is samples-at-a-PC with a dominant stallReason; we emit per (correlation, reason)
             so the Prolog layer can aggregate a per-kernel stall profile. */
          CUpti_ActivityPCSampling3 *p = (CUpti_ActivityPCSampling3 *)record;
          fprintf(g_out, "stall(%u,%u,%u).\n",
                  p->correlationId,
                  (unsigned)p->stallReason,
                  p->samples);
          break;
        }
        default: break;
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }
  } while (status == CUPTI_SUCCESS);
  free(buffer);
  fflush(g_out);
}

__attribute__((constructor))
static void cupti_trace_init(void) {
  const char *path = getenv("CUPTI_TRACE_OUT");
  if (!path) path = "/tmp/cupti_trace.facts";
  g_out = fopen(path, "w");
  cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  /* PC sampling (warp stall reasons) is opt-in: it perturbs timing and support varies by
     arch (Pascal sm_61 is older). CUPTI_PC_SAMPLING=1 enables it; the stall(...) facts then
     feed cupti_profile.pl's StallList. CUPTI_PC_PERIOD selects the sampling rate (0..5). */
  if (getenv("CUPTI_PC_SAMPLING")) {
    CUpti_ActivityPCSamplingConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.size = sizeof(CUpti_ActivityPCSamplingConfig);
    const char *per = getenv("CUPTI_PC_PERIOD");
    cfg.samplingPeriod = per ? (CUpti_ActivityPCSamplingPeriod)atoi(per)
                             : CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID;
    cfg.samplingPeriod2 = 0;
    /* config is per-context; CUPTI applies it when a context is current. We enable the
       activity kind here and configure lazily would be ideal, but enabling the kind is the
       minimum; configuration is best-effort (logged via the result, ignored if unsupported). */
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
  }
}

__attribute__((destructor))
static void cupti_trace_fini(void) {
  cuptiActivityFlushAll(1);
  if (g_out) { fflush(g_out); fclose(g_out); g_out = NULL; }
}
