/* cuda_launch.c — SWI-Prolog Foreign Language Interface for CUDA kernel launch
 *
 * Enables launching GPU kernels from within the same swipl process where
 * CUPTI profiling is active. This closes the measurement loop:
 *   cupti_init → cuda_launch(Kernel, Grid, Block, Args) → cupti_flush → stall_report
 *
 * Uses the CUDA Driver API (cuModuleLoad, cuLaunchKernel) for maximum control.
 *
 * Registered predicates:
 *   cuda_init/0            — initialize CUDA context
 *   cuda_load_module/2     — load a .ptx or .cubin file: cuda_load_module(Path, Handle)
 *   cuda_launch/5          — launch kernel: cuda_launch(Handle, KernelName, Grid, Block, Args)
 *   cuda_launch/6          — launch with sync flag: cuda_launch(Handle, Name, Grid, Block, Args, Sync)
 *   cuda_launch_timed/7    — launch + measure: cuda_launch_timed(H, N, G, B, Args, GpuUs, WallUs)
 *   cuda_sync/0            — synchronize device (wait for kernel completion)
 *   cuda_alloc/2           — allocate device memory: cuda_alloc(Bytes, DevPtr)
 *   cuda_free/1            — free device memory
 *   cuda_memcpy_h2d/3      — host→device: cuda_memcpy_h2d(HostPtr, DevPtr, Bytes)
 *   cuda_memcpy_d2h/3      — device→host: cuda_memcpy_d2h(DevPtr, HostPtr, Bytes)
 *   cuda_device_info/1     — get device name + compute capability
 *   cuda_flush_cache/0     — flush L2 cache (4MB memset, for benchmarking)
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o cuda_launch.so cuda_launch.c \
 *     -I$SWIPL_INC -I$CUDA_INC -L$CUDA_LIB -lcuda -Wl,-rpath,$CUDA_LIB
 *
 * Author: mavchin (2026-05-30)
 */

#include <SWI-Prolog.h>
#include <cuda.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

static int cuda_initialized = 0;
static CUcontext cuda_ctx = NULL;
static CUdevice cuda_dev = 0;

void log_cuda_failure_remark(const char * const szFailedMethod,
                             CUresult r,
                             const char * const szAdditionalRemark)
{
    const char *szCuErrorMsg = NULL;
    cuGetErrorString(r, &szCuErrorMsg);

    if (szCuErrorMsg == NULL) {
        szCuErrorMsg = "(not a CUresult error value)";
    }

    fprintf(stderr,
            "%s failed%s%s: %s (%d)\n",
            szFailedMethod,
            szAdditionalRemark ? " and " : "",
            szAdditionalRemark ? szAdditionalRemark : "",
            szCuErrorMsg,
            r);
}

void log_cuda_failure(const char * const szFailedMethod,
                      CUresult r)
{
    log_cuda_failure_remark(szFailedMethod, r, NULL);
}

/* ============================================================
 * cuda_init/0 — initialize CUDA driver API + create context
 * ============================================================ */
static foreign_t pl_cuda_init(void) {
    CUresult r;
    
    if (cuda_initialized) PL_succeed;
    
    r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuInit", r);
        PL_fail;
    }
    
    r = cuDeviceGet(&cuda_dev, 0);
    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuDeviceGet", r);
        PL_fail;
    }
    
    r = cuCtxCreate(&cuda_ctx, 0, cuda_dev);
    if (r != CUDA_SUCCESS) {
        /* Context might already exist (e.g., from CUPTI) — try to get current */
        r = cuCtxGetCurrent(&cuda_ctx);
        if (r != CUDA_SUCCESS || cuda_ctx == NULL) {
            log_cuda_failure_remark("cuCtxCreate", r, "no existing context");
            PL_fail;
        }
    }
    
    cuda_initialized = 1;
    PL_succeed;
}

/* ============================================================
 * cuda_device_info(-Info) — returns device name as atom
 * ============================================================ */
static foreign_t pl_cuda_device_info(term_t info) {
    if (!cuda_initialized) {
        fprintf(stderr, "cuda_device_info: not initialized\n");
        PL_fail;
    }
    
    char name[256];
    cuDeviceGetName(name, sizeof(name), cuda_dev);
    
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_dev);
    
    char buf[512];
    snprintf(buf, sizeof(buf), "%s (sm_%d%d)", name, major, minor);
    
    return PL_unify_atom_chars(info, buf);
}

/* ============================================================
 * cuda_load_module(+Path, -Handle) — load .ptx/.cubin
 * ============================================================ */
static foreign_t pl_cuda_load_module(term_t path_t, term_t handle_t) {
    char *path;
    if (!PL_get_atom_chars(path_t, &path) && !PL_get_chars(path_t, &path, CVT_ALL)) PL_fail;
    
    if (!cuda_initialized) {
        fprintf(stderr, "cuda_load_module: not initialized\n");
        PL_fail;
    }
    
    CUmodule mod;
    CUresult r = cuModuleLoad(&mod, path);
    if (r != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "cuModuleLoad(%s) failed: %s\n", path, msg);
        PL_fail;
    }
    
    return PL_unify_int64(handle_t, (int64_t)(uintptr_t)mod);
}

/* ============================================================
 * cuda_alloc(+Bytes, -DevPtr) — allocate device memory
 * ============================================================ */
static foreign_t pl_cuda_alloc(term_t bytes_t, term_t ptr_t) {
    int64_t bytes;
    if (!PL_get_int64(bytes_t, &bytes)) PL_fail;
    
    CUdeviceptr dptr;
    CUresult r = cuMemAlloc(&dptr, (size_t)bytes);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc(%lld) failed\n", (long long)bytes);
        PL_fail;
    }
    
    return PL_unify_int64(ptr_t, (int64_t)dptr);
}

/* ============================================================
 * cuda_free(+DevPtr) — free device memory
 * ============================================================ */
static foreign_t pl_cuda_free(term_t ptr_t) {
    int64_t ptr;
    if (!PL_get_int64(ptr_t, &ptr)) PL_fail;
    cuMemFree((CUdeviceptr)ptr);
    PL_succeed;
}

/* ============================================================
 * cuda_memcpy_h2d(+HostFloatList, +DevPtr, +Count) — upload floats
 * ============================================================ */
static foreign_t pl_cuda_memcpy_h2d(term_t list_t, term_t dptr_t, term_t count_t) {
    int64_t dptr, count;
    if (!PL_get_int64(dptr_t, &dptr)) PL_fail;
    if (!PL_get_int64(count_t, &count)) PL_fail;
    
    float *host = malloc(count * sizeof(float));
    if (!host) PL_fail;
    
    term_t head = PL_new_term_ref();
    term_t tail = PL_copy_term_ref(list_t);
    for (int64_t i = 0; i < count && PL_get_list(tail, head, tail); i++) {
        double val;
        PL_get_float(head, &val);
        host[i] = (float)val;
    }
    
    CUresult r = cuMemcpyHtoD((CUdeviceptr)dptr, host, count * sizeof(float));
    free(host);
    
    if (r != CUDA_SUCCESS) PL_fail;
    PL_succeed;
}

/* ============================================================
 * cuda_memcpy_d2h(+DevPtr, -FloatList, +Count) — download floats
 * ============================================================ */
static foreign_t pl_cuda_memcpy_d2h(term_t dptr_t, term_t list_t, term_t count_t) {
    int64_t dptr, count;
    if (!PL_get_int64(dptr_t, &dptr)) PL_fail;
    if (!PL_get_int64(count_t, &count)) PL_fail;
    
    float *host = malloc(count * sizeof(float));
    if (!host) PL_fail;
    
    CUresult r = cuMemcpyDtoH(host, (CUdeviceptr)dptr, count * sizeof(float));
    if (r != CUDA_SUCCESS) {
        free(host);
        PL_fail;
    }
    
    /* Build Prolog list from float array */
    term_t tail = PL_copy_term_ref(list_t);
    term_t head = PL_new_term_ref();
    for (int64_t i = 0; i < count; i++) {
        if (!PL_unify_list(tail, head, tail) ||
            !PL_unify_float(head, (double)host[i])) {
            free(host);
            PL_fail;
        }
    }
    free(host);
    return PL_unify_nil(tail);
}

/* ============================================================
 * cuda_sync/0 — synchronize (wait for all kernels to complete)
 * ============================================================ */
static foreign_t pl_cuda_sync(void) {
    CUresult r = cuCtxSynchronize();
    if (r != CUDA_SUCCESS) PL_fail;
    PL_succeed;
}

/* ============================================================
 * cuda_launch(+ModuleHandle, +KernelName, +Grid, +Block, +Args)
 *   Grid = [gx, gy, gz]
 *   Block = [bx, by, bz]
 *   Args = list of device pointers (int64) or scalar ints/floats
 * ============================================================ */
static foreign_t pl_cuda_launch(term_t mod_t, term_t name_t,
                                 term_t grid_t, term_t block_t,
                                 term_t args_t) {
    int64_t mod_handle;
    char *kernel_name;
    
    if (!PL_get_int64(mod_t, &mod_handle)) PL_fail;
    if (!PL_get_atom_chars(name_t, &kernel_name) && !PL_get_chars(name_t, &kernel_name, CVT_ALL)) PL_fail;
    
    CUmodule mod = (CUmodule)(uintptr_t)mod_handle;
    
    /* Get kernel function */
    CUfunction func;
    CUresult r = cuModuleGetFunction(&func, mod, kernel_name);
    if (r != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "cuModuleGetFunction(%s) failed: %s\n", kernel_name, msg);
        PL_fail;
    }
    
    /* Parse Grid [gx, gy, gz] */
    unsigned int grid[3] = {1, 1, 1};
    term_t gh = PL_new_term_ref();
    term_t gt = PL_copy_term_ref(grid_t);
    for (int i = 0; i < 3 && PL_get_list(gt, gh, gt); i++) {
        int val;
        PL_get_integer(gh, &val);
        grid[i] = (unsigned int)val;
    }
    
    /* Parse Block [bx, by, bz] */
    unsigned int block[3] = {1, 1, 1};
    term_t bh = PL_new_term_ref();
    term_t bt = PL_copy_term_ref(block_t);
    for (int i = 0; i < 3 && PL_get_list(bt, bh, bt); i++) {
        int val;
        PL_get_integer(bh, &val);
        block[i] = (unsigned int)val;
    }
    
    /* Parse Args — each is either an int64 (device pointer) or float (scalar) */
    void *kernel_args[32];
    int64_t int_storage[32];
    float float_storage[32];
    int argc = 0;
    
    term_t ah = PL_new_term_ref();
    term_t at = PL_copy_term_ref(args_t);
    while (PL_get_list(at, ah, at) && argc < 32) {
        double fval;
        int64_t ival;
        
        if (PL_get_int64(ah, &ival)) {
            int_storage[argc] = ival;
            kernel_args[argc] = &int_storage[argc];
        } else if (PL_get_float(ah, &fval)) {
            float_storage[argc] = (float)fval;
            kernel_args[argc] = &float_storage[argc];
        } else {
            /* Try as integer */
            int iv;
            if (PL_get_integer(ah, &iv)) {
                int_storage[argc] = iv;
                kernel_args[argc] = &int_storage[argc];
            } else {
                fprintf(stderr, "cuda_launch: arg %d is not int/float\n", argc);
                PL_fail;
            }
        }
        argc++;
    }
    
    /* Launch! */
    r = cuLaunchKernel(func,
                       grid[0], grid[1], grid[2],
                       block[0], block[1], block[2],
                       0,        /* shared memory bytes */
                       NULL,     /* stream (default) */
                       kernel_args,
                       NULL);    /* extra */
    
    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuLaunchKernel", r);
        PL_fail;
    }
    
    PL_succeed;
}

/* ============================================================
 * cuda_launch/6 — launch kernel with optional sync.
 * 6th arg: atom 'true' or 'false'. When true, calls
 * cuCtxSynchronize before returning to Prolog, eliminating
 * the FLI round-trip jitter between separate launch + sync calls.
 * ============================================================ */
static foreign_t pl_cuda_launch6(term_t mod_t, term_t name_t,
                                  term_t grid_t, term_t block_t,
                                  term_t args_t, term_t sync_t) {
    int64_t mod_handle;
    char *kernel_name;
    
    if (!PL_get_int64(mod_t, &mod_handle)) PL_fail;
    if (!PL_get_atom_chars(name_t, &kernel_name) && !PL_get_chars(name_t, &kernel_name, CVT_ALL)) PL_fail;
    
    CUmodule mod = (CUmodule)(uintptr_t)mod_handle;
    
    CUfunction func;
    CUresult r = cuModuleGetFunction(&func, mod, kernel_name);
    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuModuleGetFunction", r);
        PL_fail;
    }
    
    /* Parse Grid [gx, gy, gz] */
    unsigned int grid[3] = {1, 1, 1};
    term_t gh = PL_new_term_ref();
    term_t gt = PL_copy_term_ref(grid_t);
    for (int i = 0; i < 3 && PL_get_list(gt, gh, gt); i++) {
        int val;
        PL_get_integer(gh, &val);
        grid[i] = (unsigned int)val;
    }
    
    /* Parse Block [bx, by, bz] */
    unsigned int block[3] = {1, 1, 1};
    term_t bh = PL_new_term_ref();
    term_t bt = PL_copy_term_ref(block_t);
    for (int i = 0; i < 3 && PL_get_list(bt, bh, bt); i++) {
        int val;
        PL_get_integer(bh, &val);
        block[i] = (unsigned int)val;
    }
    
    /* Parse Args */
    void *kernel_args[32];
    int64_t int_storage[32];
    float float_storage[32];
    int argc = 0;
    
    term_t ah = PL_new_term_ref();
    term_t at = PL_copy_term_ref(args_t);
    while (PL_get_list(at, ah, at) && argc < 32) {
        double fval;
        int64_t ival;
        
        if (PL_get_int64(ah, &ival)) {
            int_storage[argc] = ival;
            kernel_args[argc] = &int_storage[argc];
        } else if (PL_get_float(ah, &fval)) {
            float_storage[argc] = (float)fval;
            kernel_args[argc] = &float_storage[argc];
        } else {
            int iv;
            if (PL_get_integer(ah, &iv)) {
                int_storage[argc] = iv;
                kernel_args[argc] = &int_storage[argc];
            } else {
                fprintf(stderr, "cuda_launch: arg %d is not int/float\n", argc);
                PL_fail;
            }
        }
        argc++;
    }
    
    /* Launch */
    r = cuLaunchKernel(func,
                       grid[0], grid[1], grid[2],
                       block[0], block[1], block[2],
                       0, NULL, kernel_args, NULL);
    
    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuLaunchKernel", r);
        PL_fail;
    }
    
    /* Sync if requested — no return to Prolog in between */
    static atom_t atom_true = 0;
    if (!atom_true) atom_true = PL_new_atom("true");
    atom_t sync_atom;
    if (PL_get_atom(sync_t, &sync_atom) && sync_atom == atom_true) {
        r = cuCtxSynchronize();
        if (r != CUDA_SUCCESS) {
            log_cuda_failure("cuCtxSynchronize (launch/6)", r);
            PL_fail;
        }
    }
    
    PL_succeed;
}

/* ============================================================
 * L2 cache flush — fills L2 with zeros via cuMemsetD8.
 * 4MB buffer (2× the P4's 2MB L2) guarantees full eviction.
 * The buffer is allocated once and reused across calls.
 * Call before benchmarking to ensure cold-cache start.
 * ============================================================ */
#define FLUSH_SIZE (4 * 1024 * 1024)
static CUdeviceptr flush_buf = 0;

static foreign_t pl_cuda_flush_cache(void) {
    if (!cuda_initialized) {
        fprintf(stderr, "cuda_flush_cache: CUDA not initialized\n");
        PL_fail;
    }

    /* Allocate flush buffer on first call: */
    if (flush_buf == 0) {
        CUresult r = cuMemAlloc(&flush_buf, FLUSH_SIZE);
        if (r != CUDA_SUCCESS) {
            log_cuda_failure("cuMemAlloc (flush_cache)", r);
            PL_fail;
        }
    }

    CUresult r = cuMemsetD8(flush_buf, 0, FLUSH_SIZE);
    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuMemsetD8 (flush_cache)", r);
        PL_fail;
    }

    r = cuCtxSynchronize();
    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuCtxSynchronize (flush_cache)", r);
        PL_fail;
    }

    PL_succeed;
}

/* ============================================================
 * cuda_launch_timed/7 — launch kernel, measure with CUDA Events.
 * Returns both GPU-side time (events) and wall-clock time.
 *
 * cuda_launch_timed(Handle, Name, Grid, Block, Args, GpuTimeUs, WallTimeUs)
 *
 * CUDA Events give GPU-native timing: no host sync overhead,
 * no Prolog FLI jitter. Wall-clock includes everything.
 * Both are useful: GPU time for kernel optimization,
 * wall time for end-to-end reality check.
 * ============================================================ */
static foreign_t pl_cuda_launch_timed(term_t mod_t, term_t name_t,
                                       term_t grid_t, term_t block_t,
                                       term_t args_t,
                                       term_t gpu_time_t, term_t wall_time_t) {
    int64_t mod_handle;
    char *kernel_name;
    struct timespec ts0, ts1;

    if (!PL_get_int64(mod_t, &mod_handle)) PL_fail;
    if (!PL_get_atom_chars(name_t, &kernel_name) && !PL_get_chars(name_t, &kernel_name, CVT_ALL)) PL_fail;

    CUmodule mod = (CUmodule)(uintptr_t)mod_handle;

    CUfunction func;
    CUresult r = cuModuleGetFunction(&func, mod, kernel_name);
    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuModuleGetFunction", r);
        PL_fail;
    }

    /* Parse Grid */
    unsigned int grid[3] = {1, 1, 1};
    term_t gh = PL_new_term_ref();
    term_t gt = PL_copy_term_ref(grid_t);
    for (int i = 0; i < 3 && PL_get_list(gt, gh, gt); i++) {
        int val;
        PL_get_integer(gh, &val);
        grid[i] = (unsigned int)val;
    }

    /* Parse Block */
    unsigned int block[3] = {1, 1, 1};
    term_t bh = PL_new_term_ref();
    term_t bt = PL_copy_term_ref(block_t);
    for (int i = 0; i < 3 && PL_get_list(bt, bh, bt); i++) {
        int val;
        PL_get_integer(bh, &val);
        block[i] = (unsigned int)val;
    }

    /* Parse Args */
    void *kernel_args[32];
    int64_t int_storage[32];
    float float_storage[32];
    int argc = 0;

    term_t ah = PL_new_term_ref();
    term_t at = PL_copy_term_ref(args_t);
    while (PL_get_list(at, ah, at) && argc < 32) {
        double fval;
        int64_t ival;

        if (PL_get_int64(ah, &ival)) {
            int_storage[argc] = ival;
            kernel_args[argc] = &int_storage[argc];
        } else if (PL_get_float(ah, &fval)) {
            float_storage[argc] = (float)fval;
            kernel_args[argc] = &float_storage[argc];
        } else {
            int iv;
            if (PL_get_integer(ah, &iv)) {
                int_storage[argc] = iv;
                kernel_args[argc] = &int_storage[argc];
            } else {
                fprintf(stderr, "cuda_launch_timed: arg %d is not int/float\n", argc);
                PL_fail;
            }
        }
        argc++;
    }

    /* Create events */
    CUevent ev_start, ev_stop;
    r = cuEventCreate(&ev_start, CU_EVENT_DEFAULT);
    if (r != CUDA_SUCCESS) { log_cuda_failure("cuEventCreate(start)", r); PL_fail; }
    r = cuEventCreate(&ev_stop, CU_EVENT_DEFAULT);
    if (r != CUDA_SUCCESS) { log_cuda_failure("cuEventCreate(stop)", r); cuEventDestroy(ev_start); PL_fail; }

    /* Record start event, launch, record stop event */
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    cuEventRecord(ev_start, NULL);
    r = cuLaunchKernel(func,
                       grid[0], grid[1], grid[2],
                       block[0], block[1], block[2],
                       0, NULL, kernel_args, NULL);
    cuEventRecord(ev_stop, NULL);

    if (r != CUDA_SUCCESS) {
        log_cuda_failure("cuLaunchKernel", r);
        cuEventDestroy(ev_start);
        cuEventDestroy(ev_stop);
        PL_fail;
    }

    /* Synchronize on the stop event (waits for kernel completion) */
    cuEventSynchronize(ev_stop);

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* Compute GPU time from events (milliseconds → microseconds) */
    float gpu_ms = 0.0f;
    cuEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    double gpu_us = (double)gpu_ms * 1000.0;

    /* Compute wall time */
    double wall_us = (ts1.tv_sec - ts0.tv_sec) * 1e6 + (ts1.tv_nsec - ts0.tv_nsec) / 1e3;

    cuEventDestroy(ev_start);
    cuEventDestroy(ev_stop);

    /* Unify results */
    if (!PL_unify_float(gpu_time_t, gpu_us)) PL_fail;
    if (!PL_unify_float(wall_time_t, wall_us)) PL_fail;

    PL_succeed;
}

/* ============================================================
 * PLF registration
 * ============================================================ */
install_t install_cuda_launch(void) {
    PL_register_foreign("cuda_init",        0, pl_cuda_init,        0);
    PL_register_foreign("cuda_device_info", 1, pl_cuda_device_info, 0);
    PL_register_foreign("cuda_load_module", 2, pl_cuda_load_module, 0);
    PL_register_foreign("cuda_launch",      5, pl_cuda_launch,      0);
    PL_register_foreign("cuda_launch",      6, pl_cuda_launch6,     0);
    PL_register_foreign("cuda_launch_timed",7, pl_cuda_launch_timed,0);
    PL_register_foreign("cuda_sync",        0, pl_cuda_sync,        0);
    PL_register_foreign("cuda_alloc",       2, pl_cuda_alloc,       0);
    PL_register_foreign("cuda_free",        1, pl_cuda_free,        0);
    PL_register_foreign("cuda_memcpy_h2d",  3, pl_cuda_memcpy_h2d,  0);
    PL_register_foreign("cuda_memcpy_d2h",  3, pl_cuda_memcpy_d2h,  0);
    PL_register_foreign("cuda_flush_cache", 0, pl_cuda_flush_cache, 0);
}
