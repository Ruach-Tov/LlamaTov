// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* harness_reduction.cu — C host program for the reduction-kernel column
 * of the cross-language correctness matrix.
 *
 * The kernel + launcher source (containing `launch_reduce_sum` etc.) is
 * emitted by kernel_emit_bridge.emit_kernel_with_launcher('reduction', ...)
 * and concatenated into the build via the Makefile / build script.
 *
 * Pipeline executed by this harness:
 *   1. Read input fixture .npy via npy_io
 *   2. Allocate output buffer
 *   3. Call launch_<kernel_name>(...)   <-- compiled in from emitted source
 *   4. Write output as .npy via npy_io
 *   5. Exit with nonzero on any error step
 *
 * Argv:
 *   harness_reduction <input.npy> <output.npy> <op_kind>
 *     op_kind selects which launcher symbol to call. For 1.c.ii/b we
 *     hardcode ggml_sum_rows (the anchor). Other ops follow when the
 *     bridge emits their kernel+launcher and we link them in.
 *
 * The same input fixture is read by the Python host (1.b.iv), the
 * Rust host (1.f, pending), and the Prolog host (future). All four
 * should compute outputs that compare allclose-equal via matrix_verify.py.
 *
 * Author: metayen 2026-05-17
 * Per Heath's cross-language correctness matrix vision (1.c.ii).
 */

#include "npy_io.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Declare the launcher we expect to be linked in from the emitted
 * kernel source. The launcher's signature is fixed by the reduction-
 * family template in kernel_emit_bridge.emit_reduction_launcher.
 *
 * The launcher SYMBOL NAME depends on the op_kind: launch_reduce_sum
 * for ggml_sum_rows, launch_reduce_mean for ggml_mean, etc. The build
 * script passes -DREDUCTION_LAUNCHER=<name> at compile time so this
 * harness source is reusable across all reduction op_kinds.
 *
 * The default REDUCTION_LAUNCHER is launch_reduce_sum for backward
 * compatibility with the 1.c.ii/b standalone test.
 *
 * Status codes from the launcher (per emit_reduction_launcher):
 *   0: success
 *   1: cudaMalloc input failed
 *   2: cudaMalloc output failed
 *   3: cudaMemcpy host->device failed
 *   4: kernel launch failed
 *   5: cudaDeviceSynchronize failed
 *   6: cudaMemcpy device->host failed
 */
#ifndef REDUCTION_LAUNCHER
#define REDUCTION_LAUNCHER launch_reduce_sum
#endif

extern "C" int REDUCTION_LAUNCHER(const float *h_X, float *h_Y,
                                   int N, int outer);

static const char *launcher_status_str(int status) {
    switch (status) {
        case 0: return "success";
        case 1: return "cudaMalloc input failed";
        case 2: return "cudaMalloc output failed";
        case 3: return "cudaMemcpy host->device failed";
        case 4: return "kernel launch failed";
        case 5: return "cudaDeviceSynchronize failed";
        case 6: return "cudaMemcpy device->host failed";
        default: return "unknown status";
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
            "usage: %s <input.npy> <output.npy> <op_kind>\n"
            "  op_kind: ggml_sum_rows (anchor for 1.c.ii/b)\n",
            argv[0]);
        return 2;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    const char *op_kind = argv[3];
    /* op_kind is informational. The actual launcher is selected at
     * BUILD time via -DREDUCTION_LAUNCHER=<symbol>. The binary is
     * specific to one op_kind; argv[3] is recorded for diagnostic
     * output but doesn't affect dispatch. */
    (void)op_kind;

    /* Read input fixture: expected shape (outer, N), dtype float32. */
    int ndim;
    size_t shape[4];
    float *h_X = NULL;
    size_t count;
    int rc = npy_read_float32(input_path, &ndim, shape, 4, &h_X, &count);
    if (rc != 0) {
        fprintf(stderr, "npy_read failed for %s (code %d)\n", input_path, rc);
        return 3;
    }
    if (ndim != 2) {
        fprintf(stderr,
            "reduction input must be 2D [outer, N]; got ndim=%d\n", ndim);
        free(h_X);
        return 3;
    }
    int outer = (int)shape[0];
    int N = (int)shape[1];

    printf("Input: shape=(%d, %d), count=%zu\n", outer, N, count);

    /* Allocate output buffer: shape [outer], float32. */
    float *h_Y = (float *)calloc((size_t)outer, sizeof(float));
    if (!h_Y) {
        fprintf(stderr, "calloc failed for output buffer\n");
        free(h_X);
        return 4;
    }

    /* Dispatch through the build-time-selected launcher. */
    int status = REDUCTION_LAUNCHER(h_X, h_Y, N, outer);
    if (status != 0) {
        fprintf(stderr, "launcher returned status %d (%s)\n",
            status, launcher_status_str(status));
        free(h_X); free(h_Y);
        return 5;
    }
    printf("Launcher OK. Y[0..3]: %.6f %.6f %.6f %.6f\n",
        h_Y[0], outer > 1 ? h_Y[1] : 0.0f,
        outer > 2 ? h_Y[2] : 0.0f, outer > 3 ? h_Y[3] : 0.0f);

    /* Write output. */
    size_t out_shape[1] = {(size_t)outer};
    rc = npy_write_float32(output_path, 1, out_shape, h_Y);
    if (rc != 0) {
        fprintf(stderr, "npy_write failed for %s (code %d)\n",
            output_path, rc);
        free(h_X); free(h_Y);
        return 6;
    }
    printf("Wrote: %s\n", output_path);

    free(h_X);
    free(h_Y);
    return 0;
}
