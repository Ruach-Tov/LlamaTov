// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* npy_io.h — Minimal NumPy .npy float32 reader/writer for C hosts.
 *
 * Part of the cross-language correctness matrix harness. C host programs
 * use this to read input fixtures and write computed outputs in the .npy
 * format that all matrix backends share.
 *
 * Supports ONLY:
 *   - dtype: float32 (little-endian, native byte order on x86/ARM64)
 *   - shape: 1D and 2D
 *   - fortran_order: False (C-contiguous)
 *
 * Tested against fixtures produced by numpy.save() with these constraints.
 * For richer .npy support (other dtypes, higher dimensions), extend as
 * needed when a matrix kernel demands it.
 *
 * Reference: NEP 1 (Numpy Enhancement Proposal — Array File Format).
 * https://numpy.org/neps/nep-0001-npy-format.html
 *
 * Author: metayen 2026-05-17
 * Per Heath's cross-language correctness matrix vision (1.c.ii).
 */

#ifndef NPY_IO_H
#define NPY_IO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Read a float32 .npy file. Allocates *out_data via malloc; caller frees.
 * Sets *out_ndim, *out_shape (caller-allocated buffer of at least 4
 * size_t slots; only out_ndim slots are written).
 *
 * Returns 0 on success, nonzero error code:
 *   1: fopen failed
 *   2: magic mismatch
 *   3: unsupported version
 *   4: header parse error
 *   5: unsupported dtype (not float32 little-endian)
 *   6: unsupported fortran_order (not False)
 *   7: ndim > out_shape_capacity
 *   8: read failed
 *   9: malloc failed
 */
int npy_read_float32(const char *path,
                     int *out_ndim,
                     size_t *out_shape,
                     size_t out_shape_capacity,
                     float **out_data,
                     size_t *out_count);

/* Write a float32 array as a .npy file.
 * shape is the dimension list, ndim its length.
 * data points to the C-contiguous float32 values.
 *
 * Returns 0 on success, nonzero on error:
 *   1: fopen failed
 *   2: write failed
 */
int npy_write_float32(const char *path,
                      int ndim,
                      const size_t *shape,
                      const float *data);

#ifdef __cplusplus
}
#endif

#endif /* NPY_IO_H */
