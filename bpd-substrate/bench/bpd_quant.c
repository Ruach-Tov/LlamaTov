// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_quant.c — Q4_K_M dequantization + quantized matmul (CPU)
 *
 * Implements llama.cpp-compatible Q4_K dequantization.
 * Block structure: 256 elements per block, 144 bytes per block.
 *
 * Usage:
 *   bpd_dequant_q4k(qdata, output, n_blocks)  — dequant to float32
 *   bpd_qmatmul_q4k(qweight, x, output, M, N, K) — quantized matmul
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Q4_K block layout (144 bytes, 256 elements):
 *   bytes 0-1:   half d (super-block scale)
 *   bytes 2-3:   half dmin (super-block min)
 *   bytes 4-15:  uint8_t scales[12] (sub-block scales+mins, 6-bit each)
 *   bytes 16-143: uint8_t qs[128] (256 nibbles packed into 128 bytes)
 */
#define Q4K_BLOCK_SIZE 256
#define Q4K_BYTES 144

/* Decode half-precision float (IEEE 754 binary16) to float32 */
static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float result;
    if (exp == 0) {
        result = (mant == 0) ? 0.0f : ldexpf((float)mant / 1024.0f, -14);
    } else if (exp == 31) {
        result = (mant == 0) ? INFINITY : NAN;
    } else {
        result = ldexpf(1.0f + (float)mant / 1024.0f, (int)exp - 15);
    }
    return sign ? -result : result;
}

/* Dequantize one Q4_K block (256 elements) */
static void dequant_q4k_block(const uint8_t* block, float* output) {
    /* Read d and dmin (half-precision) */
    uint16_t d_half, dmin_half;
    memcpy(&d_half, block, 2);
    memcpy(&dmin_half, block + 2, 2);
    float d = half_to_float(d_half);
    float dmin = half_to_float(dmin_half);

    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    /* 8 sub-blocks of 32 elements each */
    for (int j = 0; j < 8; j++) {
        /* Extract 6-bit scale and 6-bit min for this sub-block */
        uint8_t sc, m;
        if (j < 4) {
            sc = scales[j] & 0x3F;
            m = scales[j + 4] & 0x3F;
        } else {
            sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        }

        float scale = d * sc;
        float min = dmin * m;

        /* Dequant 32 elements from nibbles */
        for (int k = 0; k < 16; k++) {
            uint8_t byte = qs[j * 16 + k];
            output[j * 32 + k]      = scale * (float)(byte & 0x0F) - min;
            output[j * 32 + k + 16] = scale * (float)(byte >> 4) - min;
        }
    }
}

/* Dequantize n_blocks of Q4_K data to float32 */
void bpd_dequant_q4k(const uint8_t* qdata, float* output, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        dequant_q4k_block(qdata + b * Q4K_BYTES, output + b * Q4K_BLOCK_SIZE);
    }
}

/* Quantized matrix-vector multiply: output[M] = qweight[M, K] @ x[K]
 * qweight is Q4_K quantized (K must be multiple of 256)
 * x is float32
 * output is float32
 */
void bpd_qmatmul_q4k(const uint8_t* qweight, const float* x, float* output,
                       int M, int K) {
    int n_blocks_per_row = K / Q4K_BLOCK_SIZE;
    float dequant_buf[Q4K_BLOCK_SIZE];

    for (int row = 0; row < M; row++) {
        float sum = 0.0f;
        const uint8_t* row_data = qweight + row * n_blocks_per_row * Q4K_BYTES;

        for (int b = 0; b < n_blocks_per_row; b++) {
            /* Dequant one block */
            dequant_q4k_block(row_data + b * Q4K_BYTES, dequant_buf);

            /* Dot product with x */
            const float* x_ptr = x + b * Q4K_BLOCK_SIZE;
            for (int k = 0; k < Q4K_BLOCK_SIZE; k++) {
                sum += dequant_buf[k] * x_ptr[k];
            }
        }
        output[row] = sum;
    }
}

/* Full dequant + matmul: output[M, N] = dequant(qA[M, K]) @ B[K, N]
 * For cases where both matmul dimensions are large
 */
void bpd_qmm_q4k(const uint8_t* qA, const float* B, float* C,
                   int M, int N, int K) {
    int n_blocks_per_row = K / Q4K_BLOCK_SIZE;
    float dequant_buf[Q4K_BLOCK_SIZE];

    for (int row = 0; row < M; row++) {
        const uint8_t* row_data = qA + row * n_blocks_per_row * Q4K_BYTES;

        /* Initialize output row to zero */
        for (int col = 0; col < N; col++) C[row * N + col] = 0.0f;

        for (int b = 0; b < n_blocks_per_row; b++) {
            dequant_q4k_block(row_data + b * Q4K_BYTES, dequant_buf);
            int k_start = b * Q4K_BLOCK_SIZE;

            for (int k = 0; k < Q4K_BLOCK_SIZE; k++) {
                float a_val = dequant_buf[k];
                for (int col = 0; col < N; col++) {
                    C[row * N + col] += a_val * B[(k_start + k) * N + col];
                }
            }
        }
    }
}
