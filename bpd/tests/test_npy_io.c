// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* test_npy_io.c — smoke test for npy_io read/write round-trip.
 *
 * Read a fixture, write it back, exit 0 on success. Use the Python
 * matrix_verify.py to confirm the round-trip is byte-identical (or
 * at least allclose-equivalent).
 *
 * Build:
 *   gcc -o test_npy_io test_npy_io.c npy_io.c
 *
 * Run:
 *   ./test_npy_io <input.npy> <output.npy>
 *
 * Then verify with matrix_verify.py:
 *   python3 matrix_verify.py <input.npy> <output.npy>  # should MATCH
 *
 * Author: metayen 2026-05-17
 */

#include "npy_io.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.npy> <output.npy>\n", argv[0]);
        return 2;
    }

    int ndim;
    size_t shape[8];
    float *data = NULL;
    size_t count;

    int rc = npy_read_float32(argv[1], &ndim, shape, 8, &data, &count);
    if (rc != 0) {
        fprintf(stderr, "npy_read failed with code %d\n", rc);
        return 1;
    }

    printf("Read: ndim=%d, shape=(", ndim);
    for (int i = 0; i < ndim; i++) {
        printf("%zu%s", shape[i], (i == ndim - 1) ? "" : ", ");
    }
    printf("), count=%zu\n", count);

    /* Print first 4 and last 4 elements for sanity */
    printf("First 4: ");
    for (size_t i = 0; i < (count < 4 ? count : 4); i++) {
        printf("%.6f ", data[i]);
    }
    printf("\nLast 4:  ");
    for (size_t i = (count >= 4 ? count - 4 : 0); i < count; i++) {
        printf("%.6f ", data[i]);
    }
    printf("\n");

    rc = npy_write_float32(argv[2], ndim, shape, data);
    if (rc != 0) {
        fprintf(stderr, "npy_write failed with code %d\n", rc);
        free(data);
        return 1;
    }
    printf("Wrote: %s\n", argv[2]);

    free(data);
    return 0;
}
