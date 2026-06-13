// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* harness_elementwise.cu — C host program for the elementwise unary
 * kernel column of the cross-language correctness matrix.
 *
 * Sibling to harness_reduction.cu. Same pattern, simpler signature:
 * elementwise ops have 1D input, 1D output (same length), no row
 * structure. The kernel + launcher source (containing
 * `launch_k_silu` etc.) is emitted by
 * kernel_emit_bridge.emit_kernel_with_launcher('activation', ...)
 * and concatenated into the build via the build script.
 *
 * Pipeline:
 *   1. Read input fixture .npy (expected 1D, dtype float32)
 *   2. Allocate output buffer of same length
 *   3. Call launch_<kernel_name>(...)   <-- compiled in from emitted source
 *   4. Write output as .npy
 *   5. Exit with nonzero on any error step
 *
 * Argv:
 *   harness_elementwise <input.npy> <output.npy> <op_kind>
 *     op_kind selects which launcher (k_silu, k_sigmoid, k_relu,
 *     k_gelu, k_tanh). The binary is specific to one op_kind via
 *     -DACTIVATION_LAUNCHER=<symbol> at compile time; argv[3] is
 *     informational.
 *
 * Per mavchin's 7-cell framing (intercom 21:42 UTC): this harness
 * produces cell [2] (C host, GPU dispatch) outputs for the
 * elementwise unary column.
 *
 * Per Heath's --strict-maxxing direction (2026-05-17 A3): bit-equal
 * comparison wherever the physics allows (relu, definitely; the
 * transcendentals when CPU/GPU happens to agree).
 *
 * Author: metayen 2026-05-17
 * Per Heath's cross-language correctness matrix vision (T8.c).
 */

#include "npy_io.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Declare the launcher we expect to be linked in from the emitted
 * kernel source. The launcher's signature is fixed by the activation-
 * family template in kernel_emit_bridge.emit_activation_launcher.
 *
 * The launcher SYMBOL NAME depends on the op_kind: launch_k_silu for
 * k_silu, launch_k_sigmoid for k_sigmoid, etc. The build script passes
 * -DACTIVATION_LAUNCHER=<name> at compile time so this harness source
 * is reusable across all activation op_kinds.
 *
 * The default ACTIVATION_LAUNCHER is launch_k_silu for backward
 * compatibility / convenience.
 *
 * Status codes from the launcher (per emit_activation_launcher):
 *   0: success
 *   1: cudaMalloc input failed
 *   2: cudaMalloc output failed
 *   3: cudaMemcpy host->device failed
 *   4: kernel launch failed
 *   5: cudaDeviceSynchronize failed
 *   6: cudaMemcpy device->host failed
 */
#ifndef ACTIVATION_LAUNCHER
#define ACTIVATION_LAUNCHER launch_k_silu
#endif

extern "C" int ACTIVATION_LAUNCHER(const float *h_X, float *h_Y, int N);

static const char *launcher_status_str(int status) {
    switch (status) {
        case 0: return "success";
        case 1: return "cudaMalloc input";
        case 2: return "cudaMalloc output";
        case 3: return "cudaMemcpy H2D";
        case 4: return "kernel launch";
        case 5: return "cudaDeviceSynchronize";
        case 6: return "cudaMemcpy D2H";
        default: return "unknown";
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
            "usage: %s <input.npy> <output.npy> <op_kind>\n"
            "  op_kind: k_silu, k_sigmoid, k_relu, k_gelu, k_tanh\n",
            argv[0]);
        return 2;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    const char *op_kind = argv[3];
    /* op_kind is informational. Actual launcher selected at BUILD
     * time via -DACTIVATION_LAUNCHER=<symbol>. */
    (void)op_kind;

    /* Read input fixture: expected shape (N,), dtype float32. */
    int ndim;
    size_t shape[4];
    float *h_X = NULL;
    size_t count;
    int rc = npy_read_float32(input_path, &ndim, shape, 4, &h_X, &count);
    if (rc != 0) {
        fprintf(stderr, "npy_read failed for %s (code %d)\n", input_path, rc);
        return 3;
    }
    if (ndim != 1) {
        fprintf(stderr,
            "elementwise input must be 1D [N]; got ndim=%d\n", ndim);
        free(h_X);
        return 3;
    }
    int N = (int)shape[0];

    printf("Input: shape=(%d,), count=%zu\n", N, count);

    /* Allocate output buffer: same shape as input. */
    float *h_Y = (float *)calloc((size_t)N, sizeof(float));
    if (!h_Y) {
        fprintf(stderr, "calloc failed for output buffer\n");
        free(h_X);
        return 4;
    }

    /* Dispatch through the build-time-selected launcher. */
    int status = ACTIVATION_LAUNCHER(h_X, h_Y, N);
    if (status != 0) {
        fprintf(stderr, "launcher returned status %d (%s)\n",
            status, launcher_status_str(status));
        free(h_X); free(h_Y);
        return 5;
    }
    printf("Launcher OK. Y[0..3]: %.6f %.6f %.6f %.6f\n",
        N > 0 ? h_Y[0] : 0.0f,
        N > 1 ? h_Y[1] : 0.0f,
        N > 2 ? h_Y[2] : 0.0f,
        N > 3 ? h_Y[3] : 0.0f);

    /* Write output as 1D .npy of length N. */
    size_t out_shape[1] = {(size_t)N};
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
