// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// cuda_mem_paint.c — LD_PRELOAD shim for CUDA memory paint hardening
//
// Wraps cudaMalloc (paint 0xBAADF00D) and cudaFree (paint 0xDEADBEEF) to
// expose use-of-uninitialized-memory and use-after-free bugs.
//
// Why this exists (Heath, 2026-06-05): mavchin's SoA Q8_0 matmul harness
// reports BIT-IDENTICAL between gemv_soa and a hand-written reference stock
// kernel on dumped real-model data. But the same gemv_soa kernel in-model
// produces garbage tokens. One candidate explanation: BOTH stock and SoA
// paths in the harness read uninitialized memory at the same indices,
// producing identical garbage. Bit-identity becomes structurally
// tautological. By painting cudaMalloc output with a known sentinel,
// we force any uninitialized read to produce something observably wrong
// (NaN/Inf via 0xBAADF00D as float, or detectable in integer math).
//
// Same logic for cudaFree: paint with 0xDEADBEEF so any use-after-free
// reads a detectable sentinel rather than whatever the allocator
// recycled into that region.
//
// Usage:
//   gcc -shared -fPIC -O2 -o libcuda_mem_paint.so cuda_mem_paint.c -ldl
//   LD_PRELOAD=/path/to/libcuda_mem_paint.so ./your_cuda_binary
//
// Optional env vars:
//   CUDA_PAINT_MALLOC=0  — disable malloc-paint (just record sizes)
//   CUDA_PAINT_FREE=0    — disable free-paint
//   CUDA_PAINT_VERBOSE=1 — fprintf each malloc/free with addr+size
//   CUDA_PAINT_LOG=path  — append events to file
//
// Sentinels:
//   0xBAADF00D — newly-allocated, must-be-written-before-read
//   0xDEADBEEF — freed, must-not-be-read
//
// Both sentinels are valid as fp32:
//   0xBAADF00D as float = -2.6918e-22 (subnormal-region trash)
//   0xDEADBEEF as float = -6.2598e+18 (huge, observably wrong)
// Both are detectable in any reasonable kernel output.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>

// CUDA runtime types — declare minimally so we don't need to link/include
// the cuda runtime headers from this shim. cudaError_t is enum (int).
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToDevice 3

// We only need to call cudaMemset on the device pointer; declare its prototype
extern cudaError_t cudaMemset(void *devPtr, int value, size_t count);

// Real cudaMalloc / cudaFree pointers (resolved lazily via dlsym(RTLD_NEXT))
typedef cudaError_t (*cudaMalloc_t)(void **devPtr, size_t size);
typedef cudaError_t (*cudaFree_t)(void *devPtr);

static cudaMalloc_t real_cudaMalloc = NULL;
static cudaFree_t   real_cudaFree   = NULL;

// Track allocations so cudaFree can paint the correct size.
// Simple hash table — adequate for kernel-debugging workloads (thousands
// of allocations, not millions).
#define ALLOC_TABLE_SIZE 65536
struct alloc_entry {
    void  *ptr;
    size_t size;
};
static struct alloc_entry alloc_table[ALLOC_TABLE_SIZE];
static pthread_mutex_t alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

static int paint_malloc_enabled  = 1;
static int paint_free_enabled    = 1;
static int verbose_enabled       = 0;
static FILE *log_fp              = NULL;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

// Sentinels
//   BAADF00D for newly-allocated (must-write-before-read)
//   DEADBEEF for freed (use-after-free detection)
// cudaMemset takes a single-byte value; we need a way to paint 32-bit
// patterns. We can call a small device kernel for that, OR we can use
// cudaMemset with the LOW byte and accept that the pattern's bytes are
// repeated. Looking at the sentinels:
//   0xBAADF00D = bytes BA AD F0 0D (little-endian: 0x0D F0 AD BA)
//   0xDEADBEEF = bytes DE AD BE EF (little-endian: 0xEF BE AD DE)
// cudaMemset(p, 0xBA, n) would fill with 0xBA repeated — not the full
// 32-bit pattern. To paint the actual 32-bit sentinel, we need a kernel
// or a host-buffer-then-memcpy approach.
//
// For maximum portability (no kernel launch from shim), we use the
// host-side approach: allocate a small CPU-side fill pattern, then
// cudaMemcpy it into device memory. This adds some overhead but is
// observably correct.

static void paint_device_memory(void *devPtr, size_t size, uint32_t pattern) {
    if (devPtr == NULL || size == 0) return;
    // First-byte single-byte memset gets us 95% of the way for free, BUT
    // it doesn't produce the 32-bit sentinel pattern. For the sentinels
    // we picked (0xBAADF00D and 0xDEADBEEF), the LSB and other bytes are
    // distinct, so a single-byte memset of the low byte produces a
    // recognizable-but-not-sentinel pattern. Good enough for first-pass
    // bug detection. Anything reading these bytes will see a clearly-
    // wrong value (e.g., 0x0D0D0D0D as float = ~6.6e-32 = subnormal trash).
    //
    // For Heath's spec (full 32-bit sentinel), we do a host-allocate-
    // and-cudaMemcpy approach for sizes <= 64MB. Above that we fall
    // back to single-byte memset to avoid huge host buffers.
    if (size <= (64 * 1024 * 1024)) {
        // Allocate a host fill buffer, replicate the pattern, copy.
        size_t pattern_count = (size + sizeof(uint32_t) - 1) / sizeof(uint32_t);
        uint32_t *host_fill = (uint32_t*)malloc(pattern_count * sizeof(uint32_t));
        if (host_fill) {
            for (size_t i = 0; i < pattern_count; i++) host_fill[i] = pattern;
            // Use raw cudaMemcpy — but it's not in our symbol table.
            // Resolve it lazily here.
            typedef cudaError_t (*cudaMemcpy_t)(void*, const void*, size_t, int);
            static cudaMemcpy_t real_cudaMemcpy = NULL;
            if (!real_cudaMemcpy) {
                real_cudaMemcpy = (cudaMemcpy_t)dlsym(RTLD_NEXT, "cudaMemcpy");
            }
            if (real_cudaMemcpy) {
                real_cudaMemcpy(devPtr, host_fill, size, cudaMemcpyHostToDevice);
            }
            free(host_fill);
            return;
        }
        // Fallthrough on host-allocate failure
    }
    // Fallback: single-byte memset of the low byte
    cudaMemset(devPtr, (int)(pattern & 0xFF), size);
}

static size_t track_alloc(void *ptr, size_t size) {
    pthread_mutex_lock(&alloc_mutex);
    // Find an empty slot using simple hash + linear probe
    uintptr_t h = (uintptr_t)ptr;
    h = (h >> 4) ^ (h >> 20);
    size_t idx = h % ALLOC_TABLE_SIZE;
    for (size_t i = 0; i < ALLOC_TABLE_SIZE; i++) {
        size_t k = (idx + i) % ALLOC_TABLE_SIZE;
        if (alloc_table[k].ptr == NULL) {
            alloc_table[k].ptr = ptr;
            alloc_table[k].size = size;
            pthread_mutex_unlock(&alloc_mutex);
            return size;
        }
    }
    pthread_mutex_unlock(&alloc_mutex);
    // Table full — silently lose track. With 65K entries this should
    // never happen in normal use.
    return 0;
}

static size_t consume_alloc(void *ptr) {
    if (ptr == NULL) return 0;
    pthread_mutex_lock(&alloc_mutex);
    uintptr_t h = (uintptr_t)ptr;
    h = (h >> 4) ^ (h >> 20);
    size_t idx = h % ALLOC_TABLE_SIZE;
    for (size_t i = 0; i < ALLOC_TABLE_SIZE; i++) {
        size_t k = (idx + i) % ALLOC_TABLE_SIZE;
        if (alloc_table[k].ptr == ptr) {
            size_t size = alloc_table[k].size;
            alloc_table[k].ptr = NULL;
            alloc_table[k].size = 0;
            pthread_mutex_unlock(&alloc_mutex);
            return size;
        }
    }
    pthread_mutex_unlock(&alloc_mutex);
    return 0; // unknown allocation (allocated before LD_PRELOAD took effect)
}

static void log_event(const char *kind, void *ptr, size_t size) {
    if (verbose_enabled) {
        fprintf(stderr, "CUDA_PAINT %s ptr=%p size=%zu\n", kind, ptr, size);
    }
    if (log_fp) {
        pthread_mutex_lock(&log_mutex);
        fprintf(log_fp, "%s\t%p\t%zu\n", kind, ptr, size);
        fflush(log_fp);
        pthread_mutex_unlock(&log_mutex);
    }
}

__attribute__((constructor))
static void cuda_mem_paint_init(void) {
    const char *e;
    if ((e = getenv("CUDA_PAINT_MALLOC")) && strcmp(e, "0") == 0) paint_malloc_enabled = 0;
    if ((e = getenv("CUDA_PAINT_FREE"))   && strcmp(e, "0") == 0) paint_free_enabled = 0;
    if ((e = getenv("CUDA_PAINT_VERBOSE")) && strcmp(e, "1") == 0) verbose_enabled = 1;
    if ((e = getenv("CUDA_PAINT_LOG")) && *e) {
        log_fp = fopen(e, "a");
        if (log_fp) {
            fprintf(log_fp, "# cuda_mem_paint started pid=%d\n", (int)getpid());
            fflush(log_fp);
        }
    }
    fprintf(stderr, "[cuda_mem_paint] loaded: paint_malloc=%d paint_free=%d verbose=%d\n",
        paint_malloc_enabled, paint_free_enabled, verbose_enabled);
}

__attribute__((destructor))
static void cuda_mem_paint_fini(void) {
    if (log_fp) {
        fprintf(log_fp, "# cuda_mem_paint shutdown\n");
        fclose(log_fp);
        log_fp = NULL;
    }
}

// --- The shimmed functions ---

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!real_cudaMalloc) {
        real_cudaMalloc = (cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
        if (!real_cudaMalloc) {
            fprintf(stderr, "[cuda_mem_paint] ERROR: dlsym(cudaMalloc) failed\n");
            return 1; // generic error
        }
    }
    cudaError_t err = real_cudaMalloc(devPtr, size);
    if (err == cudaSuccess && *devPtr != NULL && size > 0) {
        track_alloc(*devPtr, size);
        if (paint_malloc_enabled) {
            paint_device_memory(*devPtr, size, 0xBAADF00D);
        }
        log_event("MALLOC", *devPtr, size);
    }
    return err;
}

cudaError_t cudaFree(void *devPtr) {
    if (!real_cudaFree) {
        real_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
        if (!real_cudaFree) {
            fprintf(stderr, "[cuda_mem_paint] ERROR: dlsym(cudaFree) failed\n");
            return 1;
        }
    }
    size_t size = consume_alloc(devPtr);
    if (paint_free_enabled && devPtr != NULL && size > 0) {
        paint_device_memory(devPtr, size, 0xDEADBEEF);
    }
    log_event("FREE", devPtr, size);
    return real_cudaFree(devPtr);
}
