// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* npy_io.c — Minimal NumPy .npy float32 reader/writer for C hosts.
 *
 * Implements the contract in npy_io.h. See header for usage and supported
 * format constraints.
 *
 * Author: metayen 2026-05-17
 */

#include "npy_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* .npy format v1.0:
 *   bytes 0-5:   "\x93NUMPY"
 *   byte 6:      major version (0x01)
 *   byte 7:      minor version (0x00)
 *   bytes 8-9:   little-endian uint16 header length
 *   bytes 10-:   ASCII Python dict literal ending in '\n'
 *                e.g. "{'descr': '<f4', 'fortran_order': False, 'shape': (8, 16), }    \n"
 *                Header is padded with spaces so total preamble (bytes 0..end of header)
 *                aligns to 64 bytes.
 *   bytes after header: raw array data
 */

#define NPY_MAGIC "\x93NUMPY"
#define NPY_MAGIC_LEN 6

static int parse_shape_from_header(const char *header, int *out_ndim,
                                   size_t *out_shape, size_t cap) {
    /* Find "'shape':" */
    const char *p = strstr(header, "'shape'");
    if (!p) return 4;
    p = strchr(p, '(');
    if (!p) return 4;
    p++; /* past '(' */

    int ndim = 0;
    while (*p && *p != ')') {
        /* skip whitespace and commas */
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        if (*p < '0' || *p > '9') return 4;
        size_t v = 0;
        while (*p >= '0' && *p <= '9') {
            v = v * 10 + (size_t)(*p - '0');
            p++;
        }
        if (ndim >= (int)cap) return 7;
        out_shape[ndim++] = v;
    }
    *out_ndim = ndim;
    return 0;
}

int npy_read_float32(const char *path,
                     int *out_ndim,
                     size_t *out_shape,
                     size_t out_shape_capacity,
                     float **out_data,
                     size_t *out_count) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1;

    char magic[NPY_MAGIC_LEN];
    if (fread(magic, 1, NPY_MAGIC_LEN, f) != NPY_MAGIC_LEN) {
        fclose(f); return 2;
    }
    if (memcmp(magic, NPY_MAGIC, NPY_MAGIC_LEN) != 0) {
        fclose(f); return 2;
    }

    uint8_t version[2];
    if (fread(version, 1, 2, f) != 2) { fclose(f); return 3; }
    if (version[0] != 1) { fclose(f); return 3; }

    uint16_t header_len;
    uint8_t header_len_bytes[2];
    if (fread(header_len_bytes, 1, 2, f) != 2) { fclose(f); return 3; }
    header_len = (uint16_t)header_len_bytes[0] |
                 ((uint16_t)header_len_bytes[1] << 8);

    char *header = (char *)malloc(header_len + 1);
    if (!header) { fclose(f); return 9; }
    if (fread(header, 1, header_len, f) != header_len) {
        free(header); fclose(f); return 4;
    }
    header[header_len] = '\0';

    /* Check dtype is float32 little-endian: '<f4' */
    if (!strstr(header, "'<f4'") && !strstr(header, "'<f4\"")) {
        free(header); fclose(f); return 5;
    }
    /* Check fortran_order is False */
    if (!strstr(header, "'fortran_order': False")) {
        free(header); fclose(f); return 6;
    }

    int rc = parse_shape_from_header(header, out_ndim, out_shape,
                                      out_shape_capacity);
    free(header);
    if (rc != 0) { fclose(f); return rc; }

    /* Compute total element count */
    size_t count = 1;
    for (int i = 0; i < *out_ndim; i++) count *= out_shape[i];

    *out_data = (float *)malloc(count * sizeof(float));
    if (!*out_data) { fclose(f); return 9; }

    if (fread(*out_data, sizeof(float), count, f) != count) {
        free(*out_data); fclose(f); return 8;
    }
    *out_count = count;
    fclose(f);
    return 0;
}

int npy_write_float32(const char *path,
                      int ndim,
                      const size_t *shape,
                      const float *data) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;

    /* Construct header: "{'descr': '<f4', 'fortran_order': False, 'shape': (s0, s1, ...), }" */
    char shape_str[256];
    size_t pos = 0;
    pos += (size_t)snprintf(shape_str + pos, sizeof(shape_str) - pos, "(");
    for (int i = 0; i < ndim; i++) {
        pos += (size_t)snprintf(shape_str + pos, sizeof(shape_str) - pos,
                                "%zu%s",
                                shape[i],
                                (i == ndim - 1 && ndim > 1) ? "" : ",");
        if (i < ndim - 1) {
            pos += (size_t)snprintf(shape_str + pos, sizeof(shape_str) - pos, " ");
        }
    }
    snprintf(shape_str + pos, sizeof(shape_str) - pos, ")");

    char header[512];
    int header_body_len = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }",
        shape_str);

    /* Pad header so (10 + header_len) is a multiple of 64, ending with \n */
    size_t total_unpadded = 10 + (size_t)header_body_len + 1;  /* +1 for \n */
    size_t total = ((total_unpadded + 63) / 64) * 64;
    size_t pad = total - total_unpadded;

    for (size_t i = 0; i < pad; i++) {
        header[header_body_len + (int)i] = ' ';
    }
    header[header_body_len + (int)pad] = '\n';
    size_t header_len = (size_t)header_body_len + pad + 1;

    /* Write preamble */
    if (fwrite(NPY_MAGIC, 1, NPY_MAGIC_LEN, f) != NPY_MAGIC_LEN) {
        fclose(f); return 2;
    }
    uint8_t version[2] = {1, 0};
    if (fwrite(version, 1, 2, f) != 2) { fclose(f); return 2; }
    uint8_t header_len_bytes[2] = {
        (uint8_t)(header_len & 0xFF),
        (uint8_t)((header_len >> 8) & 0xFF),
    };
    if (fwrite(header_len_bytes, 1, 2, f) != 2) { fclose(f); return 2; }
    if (fwrite(header, 1, header_len, f) != header_len) {
        fclose(f); return 2;
    }

    /* Write data */
    size_t count = 1;
    for (int i = 0; i < ndim; i++) count *= shape[i];
    if (fwrite(data, sizeof(float), count, f) != count) {
        fclose(f); return 2;
    }

    fclose(f);
    return 0;
}
