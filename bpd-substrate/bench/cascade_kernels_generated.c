// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// =================================================================
// cascade_kernels_generated.c — auto-generated, do not edit.
//
// One C function per valid cascade(SW, ILP, CD, CB) instantiation.
// Generator: bench/generate_cascade_kernels.py
// Parameter space: lib/reduction_kernel.pl reduction_pattern/4
// Total kernels: 160
//
// Each kernel sums an array of `n` floats. The sweep harness
// (bench/verify_cascade_sweep.py) compiles this file and runs each
// instantiation against PyTorch CPU at multiple input sizes to
// determine which (SW, ILP, CD, CB) matches PyTorch's bit pattern.
// =================================================================

float cascade_sum_simd1_ilp1_depth1_base0(const float* data, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += data[i];
    return s;
}

float cascade_sum_simd1_ilp1_depth2_base16(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=2, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[2][1] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp1_depth2_base32(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=2, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[2][1] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp1_depth2_base64(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=2, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[2][1] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp1_depth4_base16(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=4, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[4][1] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp1_depth4_base32(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=4, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[4][1] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp1_depth4_base64(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=4, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[4][1] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp1_depth8_base16(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=8, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[8][1] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp1_depth8_base32(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=8, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[8][1] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp1_depth8_base64(const float* data, int n) {
    // cascade(SW=1, ILP=1, CD=8, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 1;
    int simd_processed = size_ilp * 1;
    float acc[8][1] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 1;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 1;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth1_base0(const float* data, int n) {
    float acc[2] = {0};
    int ilp_size = n / 2;
    for (int i = 0; i < ilp_size; ++i) {
        acc[0] += data[i * 2 + 0];
        acc[1] += data[i * 2 + 1];
    }
    for (int i = ilp_size * 2; i < n; ++i) acc[0] += data[i];
    acc[0] += acc[1];
    return acc[0];
}

float cascade_sum_simd1_ilp2_depth2_base16(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=2, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[2][2] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth2_base32(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=2, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[2][2] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth2_base64(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=2, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[2][2] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth4_base16(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=4, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[4][2] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth4_base32(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=4, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[4][2] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth4_base64(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=4, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[4][2] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth8_base16(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=8, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[8][2] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth8_base32(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=8, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[8][2] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp2_depth8_base64(const float* data, int n) {
    // cascade(SW=1, ILP=2, CD=8, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 2;
    int simd_processed = size_ilp * 2;
    float acc[8][2] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 2;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 2;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth1_base0(const float* data, int n) {
    float acc[4] = {0};
    int ilp_size = n / 4;
    for (int i = 0; i < ilp_size; ++i) {
        acc[0] += data[i * 4 + 0];
        acc[1] += data[i * 4 + 1];
        acc[2] += data[i * 4 + 2];
        acc[3] += data[i * 4 + 3];
    }
    for (int i = ilp_size * 4; i < n; ++i) acc[0] += data[i];
    acc[0] += acc[1];
    acc[0] += acc[2];
    acc[0] += acc[3];
    return acc[0];
}

float cascade_sum_simd1_ilp4_depth2_base16(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=2, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[2][4] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth2_base32(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=2, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[2][4] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth2_base64(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=2, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[2][4] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth4_base16(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=4, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[4][4] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth4_base32(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=4, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[4][4] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth4_base64(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=4, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[4][4] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth8_base16(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=8, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[8][4] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth8_base32(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=8, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[8][4] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp4_depth8_base64(const float* data, int n) {
    // cascade(SW=1, ILP=4, CD=8, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 4;
    int simd_processed = size_ilp * 4;
    float acc[8][4] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth1_base0(const float* data, int n) {
    float acc[8] = {0};
    int ilp_size = n / 8;
    for (int i = 0; i < ilp_size; ++i) {
        acc[0] += data[i * 8 + 0];
        acc[1] += data[i * 8 + 1];
        acc[2] += data[i * 8 + 2];
        acc[3] += data[i * 8 + 3];
        acc[4] += data[i * 8 + 4];
        acc[5] += data[i * 8 + 5];
        acc[6] += data[i * 8 + 6];
        acc[7] += data[i * 8 + 7];
    }
    for (int i = ilp_size * 8; i < n; ++i) acc[0] += data[i];
    acc[0] += acc[1];
    acc[0] += acc[2];
    acc[0] += acc[3];
    acc[0] += acc[4];
    acc[0] += acc[5];
    acc[0] += acc[6];
    acc[0] += acc[7];
    return acc[0];
}

float cascade_sum_simd1_ilp8_depth2_base16(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=2, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[2][8] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth2_base32(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=2, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[2][8] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth2_base64(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=2, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[2][8] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth4_base16(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=4, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[4][8] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth4_base32(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=4, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[4][8] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth4_base64(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=4, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[4][8] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth8_base16(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=8, CB=16)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[8][8] = {{0}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth8_base32(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=8, CB=32)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[8][8] = {{0}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd1_ilp8_depth8_base64(const float* data, int n) {
    // cascade(SW=1, ILP=8, CD=8, CB=64)

    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / 8;
    int simd_processed = size_ilp * 8;
    float acc[8][8] = {{0}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += base[ilp_lane];
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            acc[0][ilp_lane] += base[ilp_lane];
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }
    }

    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        acc[0][0] += acc[0][k];
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    final_acc += acc[0][0];
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth1_base0(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=1, CB=0) — SIMD ILP-only
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[1][4] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth2_base16(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=2, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[2][1][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth2_base32(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=2, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[2][1][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth2_base64(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=2, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[2][1][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth4_base16(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=4, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[4][1][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth4_base32(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=4, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[4][1][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth4_base64(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=4, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[4][1][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth8_base16(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=8, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[8][1][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth8_base32(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=8, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[8][1][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp1_depth8_base64(const float* data, int n) {
    // cascade(SW=4, ILP=1, CD=8, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 4;
    float acc[8][1][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 4;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 4;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth1_base0(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=1, CB=0) — SIMD ILP-only
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[2][4] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth2_base16(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=2, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[2][2][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth2_base32(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=2, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[2][2][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth2_base64(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=2, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[2][2][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth4_base16(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=4, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[4][2][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth4_base32(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=4, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[4][2][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth4_base64(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=4, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[4][2][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth8_base16(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=8, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[8][2][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth8_base32(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=8, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[8][2][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp2_depth8_base64(const float* data, int n) {
    // cascade(SW=4, ILP=2, CD=8, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 4;
    float acc[8][2][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth1_base0(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=1, CB=0) — SIMD ILP-only
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[4][4] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth2_base16(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=2, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[2][4][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth2_base32(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=2, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[2][4][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth2_base64(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=2, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[2][4][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth4_base16(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=4, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[4][4][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth4_base32(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=4, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[4][4][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth4_base64(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=4, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[4][4][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth8_base16(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=8, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[8][4][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth8_base32(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=8, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[8][4][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp4_depth8_base64(const float* data, int n) {
    // cascade(SW=4, ILP=4, CD=8, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 4;
    float acc[8][4][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth1_base0(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=1, CB=0) — SIMD ILP-only
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[8][4] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth2_base16(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=2, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[2][8][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth2_base32(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=2, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[2][8][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth2_base64(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=2, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[2][8][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth4_base16(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=4, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[4][8][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth4_base32(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=4, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[4][8][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth4_base64(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=4, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[4][8][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth8_base16(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=8, CB=16)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[8][8][4] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth8_base32(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=8, CB=32)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[8][8][4] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd4_ilp8_depth8_base64(const float* data, int n) {
    // cascade(SW=4, ILP=8, CD=8, CB=64)
    if (n < 4) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 4;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 4;
    float acc[8][8][4] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 4;
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 4;
            for (int s = 0; s < 4; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 4; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 4;
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 4; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 4; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth1_base0(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=1, CB=0) — SIMD ILP-only
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[1][8] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth2_base16(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=2, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[2][1][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth2_base32(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=2, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[2][1][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth2_base64(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=2, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[2][1][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth4_base16(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=4, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[4][1][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth4_base32(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=4, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[4][1][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth4_base64(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=4, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[4][1][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth8_base16(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=8, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[8][1][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth8_base32(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=8, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[8][1][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp1_depth8_base64(const float* data, int n) {
    // cascade(SW=8, ILP=1, CD=8, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 8;
    float acc[8][1][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 8;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 8;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth1_base0(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=1, CB=0) — SIMD ILP-only
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[2][8] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth2_base16(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=2, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[2][2][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth2_base32(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=2, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[2][2][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth2_base64(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=2, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[2][2][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth4_base16(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=4, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[4][2][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth4_base32(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=4, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[4][2][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth4_base64(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=4, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[4][2][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth8_base16(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=8, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[8][2][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth8_base32(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=8, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[8][2][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp2_depth8_base64(const float* data, int n) {
    // cascade(SW=8, ILP=2, CD=8, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 8;
    float acc[8][2][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth1_base0(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=1, CB=0) — SIMD ILP-only
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[4][8] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth2_base16(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=2, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[2][4][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth2_base32(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=2, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[2][4][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth2_base64(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=2, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[2][4][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth4_base16(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=4, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[4][4][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth4_base32(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=4, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[4][4][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth4_base64(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=4, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[4][4][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth8_base16(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=8, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[8][4][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth8_base32(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=8, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[8][4][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp4_depth8_base64(const float* data, int n) {
    // cascade(SW=8, ILP=4, CD=8, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 8;
    float acc[8][4][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth1_base0(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=1, CB=0) — SIMD ILP-only
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[8][8] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth2_base16(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=2, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[2][8][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth2_base32(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=2, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[2][8][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth2_base64(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=2, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[2][8][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth4_base16(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=4, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[4][8][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth4_base32(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=4, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[4][8][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth4_base64(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=4, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[4][8][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth8_base16(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=8, CB=16)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[8][8][8] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth8_base32(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=8, CB=32)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[8][8][8] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd8_ilp8_depth8_base64(const float* data, int n) {
    // cascade(SW=8, ILP=8, CD=8, CB=64)
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 8;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 8;
    float acc[8][8][8] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 8;
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 8; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth1_base0(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=1, CB=0) — SIMD ILP-only
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[1][16] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth2_base16(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=2, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[2][1][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth2_base32(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=2, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[2][1][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth2_base64(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=2, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[2][1][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth4_base16(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=4, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[4][1][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth4_base32(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=4, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[4][1][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth4_base64(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=4, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[4][1][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth8_base16(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=8, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[8][1][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth8_base32(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=8, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[8][1][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp1_depth8_base64(const float* data, int n) {
    // cascade(SW=16, ILP=1, CD=8, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 1;
    int simd_processed = vec_size * 16;
    float acc[8][1][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 16;
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 16;
        for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 1; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 1; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 1; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth1_base0(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=1, CB=0) — SIMD ILP-only
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[2][16] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth2_base16(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=2, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[2][2][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth2_base32(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=2, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[2][2][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth2_base64(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=2, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[2][2][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth4_base16(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=4, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[4][2][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth4_base32(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=4, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[4][2][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth4_base64(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=4, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[4][2][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth8_base16(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=8, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[8][2][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth8_base32(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=8, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[8][2][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp2_depth8_base64(const float* data, int n) {
    // cascade(SW=16, ILP=2, CD=8, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 2;
    int simd_processed = vec_size * 16;
    float acc[8][2][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 2; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 2; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 2; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth1_base0(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=1, CB=0) — SIMD ILP-only
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[4][16] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth2_base16(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=2, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[2][4][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth2_base32(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=2, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[2][4][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth2_base64(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=2, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[2][4][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth4_base16(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=4, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[4][4][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth4_base32(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=4, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[4][4][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth4_base64(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=4, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[4][4][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth8_base16(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=8, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[8][4][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth8_base32(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=8, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[8][4][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp4_depth8_base64(const float* data, int n) {
    // cascade(SW=16, ILP=4, CD=8, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 4;
    int simd_processed = vec_size * 16;
    float acc[8][4][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 64;
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 64;
        for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 4; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 4; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth1_base0(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=1, CB=0) — SIMD ILP-only
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }
    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[8][16] = {{0}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[ilp_lane][s] += src[s];
            }
        }
    }

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][s] += src[s];
        }
    }

    // ILP collapse
    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][s] += acc[k][s];
        }
    }

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth2_base16(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=2, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[2][8][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth2_base32(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=2, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[2][8][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth2_base64(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=2, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[2][8][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 2; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth4_base16(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=4, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[4][8][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth4_base32(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=4, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[4][8][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth4_base64(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=4, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[4][8][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 4; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth8_base16(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=8, CB=16)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[8][8][16] = {{{0}}};
    int level_step = 16;
    int level_mask = 15;
    int lp = 4;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth8_base32(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=8, CB=32)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[8][8][16] = {{{0}}};
    int level_step = 32;
    int level_mask = 31;
    int lp = 5;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

float cascade_sum_simd16_ilp8_depth8_base64(const float* data, int n) {
    // cascade(SW=16, ILP=8, CD=8, CB=64)
    if (n < 16) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    int vec_size = n / 16;
    int size_ilp = vec_size / 8;
    int simd_processed = vec_size * 16;
    float acc[8][8][16] = {{{0}}};
    int level_step = 64;
    int level_mask = 63;
    int lp = 6;

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 128;
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                const float* src = base + ilp_lane * 16;
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += src[s];
                }
            }
        }
        // Cascade promotion
        for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 128;
        for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
            const float* src = base + ilp_lane * 16;
            for (int s = 0; s < 16; ++s) {
                acc[0][ilp_lane][s] += src[s];
            }
        }
    }

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < 8; ++level) {
            for (int ilp_lane = 0; ilp_lane < 8; ++ilp_lane) {
                for (int s = 0; s < 16; ++s) {
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }
            }
    }

    for (int v = size_ilp * 8; v < vec_size; ++v) {
        const float* src = data + v * 16;
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += src[s];
        }
    }
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1

    for (int k = 1; k < 8; ++k) {
        for (int s = 0; s < 16; ++s) {
            acc[0][0][s] += acc[0][k][s];
        }
    }

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 16; ++s) {
        final_acc += acc[0][0][s];
    }
    return final_acc;
}

// Dispatch table: cascade_dispatch[i] is the i-th kernel function pointer.
// cascade_dispatch_names[i] is the matching string name.
// cascade_dispatch_count is the total number of kernels.

typedef float (*cascade_kernel_fn)(const float* data, int n);

const int cascade_dispatch_count = 160;

cascade_kernel_fn cascade_dispatch[] = {
    cascade_sum_simd1_ilp1_depth1_base0,
    cascade_sum_simd1_ilp1_depth2_base16,
    cascade_sum_simd1_ilp1_depth2_base32,
    cascade_sum_simd1_ilp1_depth2_base64,
    cascade_sum_simd1_ilp1_depth4_base16,
    cascade_sum_simd1_ilp1_depth4_base32,
    cascade_sum_simd1_ilp1_depth4_base64,
    cascade_sum_simd1_ilp1_depth8_base16,
    cascade_sum_simd1_ilp1_depth8_base32,
    cascade_sum_simd1_ilp1_depth8_base64,
    cascade_sum_simd1_ilp2_depth1_base0,
    cascade_sum_simd1_ilp2_depth2_base16,
    cascade_sum_simd1_ilp2_depth2_base32,
    cascade_sum_simd1_ilp2_depth2_base64,
    cascade_sum_simd1_ilp2_depth4_base16,
    cascade_sum_simd1_ilp2_depth4_base32,
    cascade_sum_simd1_ilp2_depth4_base64,
    cascade_sum_simd1_ilp2_depth8_base16,
    cascade_sum_simd1_ilp2_depth8_base32,
    cascade_sum_simd1_ilp2_depth8_base64,
    cascade_sum_simd1_ilp4_depth1_base0,
    cascade_sum_simd1_ilp4_depth2_base16,
    cascade_sum_simd1_ilp4_depth2_base32,
    cascade_sum_simd1_ilp4_depth2_base64,
    cascade_sum_simd1_ilp4_depth4_base16,
    cascade_sum_simd1_ilp4_depth4_base32,
    cascade_sum_simd1_ilp4_depth4_base64,
    cascade_sum_simd1_ilp4_depth8_base16,
    cascade_sum_simd1_ilp4_depth8_base32,
    cascade_sum_simd1_ilp4_depth8_base64,
    cascade_sum_simd1_ilp8_depth1_base0,
    cascade_sum_simd1_ilp8_depth2_base16,
    cascade_sum_simd1_ilp8_depth2_base32,
    cascade_sum_simd1_ilp8_depth2_base64,
    cascade_sum_simd1_ilp8_depth4_base16,
    cascade_sum_simd1_ilp8_depth4_base32,
    cascade_sum_simd1_ilp8_depth4_base64,
    cascade_sum_simd1_ilp8_depth8_base16,
    cascade_sum_simd1_ilp8_depth8_base32,
    cascade_sum_simd1_ilp8_depth8_base64,
    cascade_sum_simd4_ilp1_depth1_base0,
    cascade_sum_simd4_ilp1_depth2_base16,
    cascade_sum_simd4_ilp1_depth2_base32,
    cascade_sum_simd4_ilp1_depth2_base64,
    cascade_sum_simd4_ilp1_depth4_base16,
    cascade_sum_simd4_ilp1_depth4_base32,
    cascade_sum_simd4_ilp1_depth4_base64,
    cascade_sum_simd4_ilp1_depth8_base16,
    cascade_sum_simd4_ilp1_depth8_base32,
    cascade_sum_simd4_ilp1_depth8_base64,
    cascade_sum_simd4_ilp2_depth1_base0,
    cascade_sum_simd4_ilp2_depth2_base16,
    cascade_sum_simd4_ilp2_depth2_base32,
    cascade_sum_simd4_ilp2_depth2_base64,
    cascade_sum_simd4_ilp2_depth4_base16,
    cascade_sum_simd4_ilp2_depth4_base32,
    cascade_sum_simd4_ilp2_depth4_base64,
    cascade_sum_simd4_ilp2_depth8_base16,
    cascade_sum_simd4_ilp2_depth8_base32,
    cascade_sum_simd4_ilp2_depth8_base64,
    cascade_sum_simd4_ilp4_depth1_base0,
    cascade_sum_simd4_ilp4_depth2_base16,
    cascade_sum_simd4_ilp4_depth2_base32,
    cascade_sum_simd4_ilp4_depth2_base64,
    cascade_sum_simd4_ilp4_depth4_base16,
    cascade_sum_simd4_ilp4_depth4_base32,
    cascade_sum_simd4_ilp4_depth4_base64,
    cascade_sum_simd4_ilp4_depth8_base16,
    cascade_sum_simd4_ilp4_depth8_base32,
    cascade_sum_simd4_ilp4_depth8_base64,
    cascade_sum_simd4_ilp8_depth1_base0,
    cascade_sum_simd4_ilp8_depth2_base16,
    cascade_sum_simd4_ilp8_depth2_base32,
    cascade_sum_simd4_ilp8_depth2_base64,
    cascade_sum_simd4_ilp8_depth4_base16,
    cascade_sum_simd4_ilp8_depth4_base32,
    cascade_sum_simd4_ilp8_depth4_base64,
    cascade_sum_simd4_ilp8_depth8_base16,
    cascade_sum_simd4_ilp8_depth8_base32,
    cascade_sum_simd4_ilp8_depth8_base64,
    cascade_sum_simd8_ilp1_depth1_base0,
    cascade_sum_simd8_ilp1_depth2_base16,
    cascade_sum_simd8_ilp1_depth2_base32,
    cascade_sum_simd8_ilp1_depth2_base64,
    cascade_sum_simd8_ilp1_depth4_base16,
    cascade_sum_simd8_ilp1_depth4_base32,
    cascade_sum_simd8_ilp1_depth4_base64,
    cascade_sum_simd8_ilp1_depth8_base16,
    cascade_sum_simd8_ilp1_depth8_base32,
    cascade_sum_simd8_ilp1_depth8_base64,
    cascade_sum_simd8_ilp2_depth1_base0,
    cascade_sum_simd8_ilp2_depth2_base16,
    cascade_sum_simd8_ilp2_depth2_base32,
    cascade_sum_simd8_ilp2_depth2_base64,
    cascade_sum_simd8_ilp2_depth4_base16,
    cascade_sum_simd8_ilp2_depth4_base32,
    cascade_sum_simd8_ilp2_depth4_base64,
    cascade_sum_simd8_ilp2_depth8_base16,
    cascade_sum_simd8_ilp2_depth8_base32,
    cascade_sum_simd8_ilp2_depth8_base64,
    cascade_sum_simd8_ilp4_depth1_base0,
    cascade_sum_simd8_ilp4_depth2_base16,
    cascade_sum_simd8_ilp4_depth2_base32,
    cascade_sum_simd8_ilp4_depth2_base64,
    cascade_sum_simd8_ilp4_depth4_base16,
    cascade_sum_simd8_ilp4_depth4_base32,
    cascade_sum_simd8_ilp4_depth4_base64,
    cascade_sum_simd8_ilp4_depth8_base16,
    cascade_sum_simd8_ilp4_depth8_base32,
    cascade_sum_simd8_ilp4_depth8_base64,
    cascade_sum_simd8_ilp8_depth1_base0,
    cascade_sum_simd8_ilp8_depth2_base16,
    cascade_sum_simd8_ilp8_depth2_base32,
    cascade_sum_simd8_ilp8_depth2_base64,
    cascade_sum_simd8_ilp8_depth4_base16,
    cascade_sum_simd8_ilp8_depth4_base32,
    cascade_sum_simd8_ilp8_depth4_base64,
    cascade_sum_simd8_ilp8_depth8_base16,
    cascade_sum_simd8_ilp8_depth8_base32,
    cascade_sum_simd8_ilp8_depth8_base64,
    cascade_sum_simd16_ilp1_depth1_base0,
    cascade_sum_simd16_ilp1_depth2_base16,
    cascade_sum_simd16_ilp1_depth2_base32,
    cascade_sum_simd16_ilp1_depth2_base64,
    cascade_sum_simd16_ilp1_depth4_base16,
    cascade_sum_simd16_ilp1_depth4_base32,
    cascade_sum_simd16_ilp1_depth4_base64,
    cascade_sum_simd16_ilp1_depth8_base16,
    cascade_sum_simd16_ilp1_depth8_base32,
    cascade_sum_simd16_ilp1_depth8_base64,
    cascade_sum_simd16_ilp2_depth1_base0,
    cascade_sum_simd16_ilp2_depth2_base16,
    cascade_sum_simd16_ilp2_depth2_base32,
    cascade_sum_simd16_ilp2_depth2_base64,
    cascade_sum_simd16_ilp2_depth4_base16,
    cascade_sum_simd16_ilp2_depth4_base32,
    cascade_sum_simd16_ilp2_depth4_base64,
    cascade_sum_simd16_ilp2_depth8_base16,
    cascade_sum_simd16_ilp2_depth8_base32,
    cascade_sum_simd16_ilp2_depth8_base64,
    cascade_sum_simd16_ilp4_depth1_base0,
    cascade_sum_simd16_ilp4_depth2_base16,
    cascade_sum_simd16_ilp4_depth2_base32,
    cascade_sum_simd16_ilp4_depth2_base64,
    cascade_sum_simd16_ilp4_depth4_base16,
    cascade_sum_simd16_ilp4_depth4_base32,
    cascade_sum_simd16_ilp4_depth4_base64,
    cascade_sum_simd16_ilp4_depth8_base16,
    cascade_sum_simd16_ilp4_depth8_base32,
    cascade_sum_simd16_ilp4_depth8_base64,
    cascade_sum_simd16_ilp8_depth1_base0,
    cascade_sum_simd16_ilp8_depth2_base16,
    cascade_sum_simd16_ilp8_depth2_base32,
    cascade_sum_simd16_ilp8_depth2_base64,
    cascade_sum_simd16_ilp8_depth4_base16,
    cascade_sum_simd16_ilp8_depth4_base32,
    cascade_sum_simd16_ilp8_depth4_base64,
    cascade_sum_simd16_ilp8_depth8_base16,
    cascade_sum_simd16_ilp8_depth8_base32,
    cascade_sum_simd16_ilp8_depth8_base64,
};

const char* cascade_dispatch_names[] = {
    "cascade_sum_simd1_ilp1_depth1_base0",
    "cascade_sum_simd1_ilp1_depth2_base16",
    "cascade_sum_simd1_ilp1_depth2_base32",
    "cascade_sum_simd1_ilp1_depth2_base64",
    "cascade_sum_simd1_ilp1_depth4_base16",
    "cascade_sum_simd1_ilp1_depth4_base32",
    "cascade_sum_simd1_ilp1_depth4_base64",
    "cascade_sum_simd1_ilp1_depth8_base16",
    "cascade_sum_simd1_ilp1_depth8_base32",
    "cascade_sum_simd1_ilp1_depth8_base64",
    "cascade_sum_simd1_ilp2_depth1_base0",
    "cascade_sum_simd1_ilp2_depth2_base16",
    "cascade_sum_simd1_ilp2_depth2_base32",
    "cascade_sum_simd1_ilp2_depth2_base64",
    "cascade_sum_simd1_ilp2_depth4_base16",
    "cascade_sum_simd1_ilp2_depth4_base32",
    "cascade_sum_simd1_ilp2_depth4_base64",
    "cascade_sum_simd1_ilp2_depth8_base16",
    "cascade_sum_simd1_ilp2_depth8_base32",
    "cascade_sum_simd1_ilp2_depth8_base64",
    "cascade_sum_simd1_ilp4_depth1_base0",
    "cascade_sum_simd1_ilp4_depth2_base16",
    "cascade_sum_simd1_ilp4_depth2_base32",
    "cascade_sum_simd1_ilp4_depth2_base64",
    "cascade_sum_simd1_ilp4_depth4_base16",
    "cascade_sum_simd1_ilp4_depth4_base32",
    "cascade_sum_simd1_ilp4_depth4_base64",
    "cascade_sum_simd1_ilp4_depth8_base16",
    "cascade_sum_simd1_ilp4_depth8_base32",
    "cascade_sum_simd1_ilp4_depth8_base64",
    "cascade_sum_simd1_ilp8_depth1_base0",
    "cascade_sum_simd1_ilp8_depth2_base16",
    "cascade_sum_simd1_ilp8_depth2_base32",
    "cascade_sum_simd1_ilp8_depth2_base64",
    "cascade_sum_simd1_ilp8_depth4_base16",
    "cascade_sum_simd1_ilp8_depth4_base32",
    "cascade_sum_simd1_ilp8_depth4_base64",
    "cascade_sum_simd1_ilp8_depth8_base16",
    "cascade_sum_simd1_ilp8_depth8_base32",
    "cascade_sum_simd1_ilp8_depth8_base64",
    "cascade_sum_simd4_ilp1_depth1_base0",
    "cascade_sum_simd4_ilp1_depth2_base16",
    "cascade_sum_simd4_ilp1_depth2_base32",
    "cascade_sum_simd4_ilp1_depth2_base64",
    "cascade_sum_simd4_ilp1_depth4_base16",
    "cascade_sum_simd4_ilp1_depth4_base32",
    "cascade_sum_simd4_ilp1_depth4_base64",
    "cascade_sum_simd4_ilp1_depth8_base16",
    "cascade_sum_simd4_ilp1_depth8_base32",
    "cascade_sum_simd4_ilp1_depth8_base64",
    "cascade_sum_simd4_ilp2_depth1_base0",
    "cascade_sum_simd4_ilp2_depth2_base16",
    "cascade_sum_simd4_ilp2_depth2_base32",
    "cascade_sum_simd4_ilp2_depth2_base64",
    "cascade_sum_simd4_ilp2_depth4_base16",
    "cascade_sum_simd4_ilp2_depth4_base32",
    "cascade_sum_simd4_ilp2_depth4_base64",
    "cascade_sum_simd4_ilp2_depth8_base16",
    "cascade_sum_simd4_ilp2_depth8_base32",
    "cascade_sum_simd4_ilp2_depth8_base64",
    "cascade_sum_simd4_ilp4_depth1_base0",
    "cascade_sum_simd4_ilp4_depth2_base16",
    "cascade_sum_simd4_ilp4_depth2_base32",
    "cascade_sum_simd4_ilp4_depth2_base64",
    "cascade_sum_simd4_ilp4_depth4_base16",
    "cascade_sum_simd4_ilp4_depth4_base32",
    "cascade_sum_simd4_ilp4_depth4_base64",
    "cascade_sum_simd4_ilp4_depth8_base16",
    "cascade_sum_simd4_ilp4_depth8_base32",
    "cascade_sum_simd4_ilp4_depth8_base64",
    "cascade_sum_simd4_ilp8_depth1_base0",
    "cascade_sum_simd4_ilp8_depth2_base16",
    "cascade_sum_simd4_ilp8_depth2_base32",
    "cascade_sum_simd4_ilp8_depth2_base64",
    "cascade_sum_simd4_ilp8_depth4_base16",
    "cascade_sum_simd4_ilp8_depth4_base32",
    "cascade_sum_simd4_ilp8_depth4_base64",
    "cascade_sum_simd4_ilp8_depth8_base16",
    "cascade_sum_simd4_ilp8_depth8_base32",
    "cascade_sum_simd4_ilp8_depth8_base64",
    "cascade_sum_simd8_ilp1_depth1_base0",
    "cascade_sum_simd8_ilp1_depth2_base16",
    "cascade_sum_simd8_ilp1_depth2_base32",
    "cascade_sum_simd8_ilp1_depth2_base64",
    "cascade_sum_simd8_ilp1_depth4_base16",
    "cascade_sum_simd8_ilp1_depth4_base32",
    "cascade_sum_simd8_ilp1_depth4_base64",
    "cascade_sum_simd8_ilp1_depth8_base16",
    "cascade_sum_simd8_ilp1_depth8_base32",
    "cascade_sum_simd8_ilp1_depth8_base64",
    "cascade_sum_simd8_ilp2_depth1_base0",
    "cascade_sum_simd8_ilp2_depth2_base16",
    "cascade_sum_simd8_ilp2_depth2_base32",
    "cascade_sum_simd8_ilp2_depth2_base64",
    "cascade_sum_simd8_ilp2_depth4_base16",
    "cascade_sum_simd8_ilp2_depth4_base32",
    "cascade_sum_simd8_ilp2_depth4_base64",
    "cascade_sum_simd8_ilp2_depth8_base16",
    "cascade_sum_simd8_ilp2_depth8_base32",
    "cascade_sum_simd8_ilp2_depth8_base64",
    "cascade_sum_simd8_ilp4_depth1_base0",
    "cascade_sum_simd8_ilp4_depth2_base16",
    "cascade_sum_simd8_ilp4_depth2_base32",
    "cascade_sum_simd8_ilp4_depth2_base64",
    "cascade_sum_simd8_ilp4_depth4_base16",
    "cascade_sum_simd8_ilp4_depth4_base32",
    "cascade_sum_simd8_ilp4_depth4_base64",
    "cascade_sum_simd8_ilp4_depth8_base16",
    "cascade_sum_simd8_ilp4_depth8_base32",
    "cascade_sum_simd8_ilp4_depth8_base64",
    "cascade_sum_simd8_ilp8_depth1_base0",
    "cascade_sum_simd8_ilp8_depth2_base16",
    "cascade_sum_simd8_ilp8_depth2_base32",
    "cascade_sum_simd8_ilp8_depth2_base64",
    "cascade_sum_simd8_ilp8_depth4_base16",
    "cascade_sum_simd8_ilp8_depth4_base32",
    "cascade_sum_simd8_ilp8_depth4_base64",
    "cascade_sum_simd8_ilp8_depth8_base16",
    "cascade_sum_simd8_ilp8_depth8_base32",
    "cascade_sum_simd8_ilp8_depth8_base64",
    "cascade_sum_simd16_ilp1_depth1_base0",
    "cascade_sum_simd16_ilp1_depth2_base16",
    "cascade_sum_simd16_ilp1_depth2_base32",
    "cascade_sum_simd16_ilp1_depth2_base64",
    "cascade_sum_simd16_ilp1_depth4_base16",
    "cascade_sum_simd16_ilp1_depth4_base32",
    "cascade_sum_simd16_ilp1_depth4_base64",
    "cascade_sum_simd16_ilp1_depth8_base16",
    "cascade_sum_simd16_ilp1_depth8_base32",
    "cascade_sum_simd16_ilp1_depth8_base64",
    "cascade_sum_simd16_ilp2_depth1_base0",
    "cascade_sum_simd16_ilp2_depth2_base16",
    "cascade_sum_simd16_ilp2_depth2_base32",
    "cascade_sum_simd16_ilp2_depth2_base64",
    "cascade_sum_simd16_ilp2_depth4_base16",
    "cascade_sum_simd16_ilp2_depth4_base32",
    "cascade_sum_simd16_ilp2_depth4_base64",
    "cascade_sum_simd16_ilp2_depth8_base16",
    "cascade_sum_simd16_ilp2_depth8_base32",
    "cascade_sum_simd16_ilp2_depth8_base64",
    "cascade_sum_simd16_ilp4_depth1_base0",
    "cascade_sum_simd16_ilp4_depth2_base16",
    "cascade_sum_simd16_ilp4_depth2_base32",
    "cascade_sum_simd16_ilp4_depth2_base64",
    "cascade_sum_simd16_ilp4_depth4_base16",
    "cascade_sum_simd16_ilp4_depth4_base32",
    "cascade_sum_simd16_ilp4_depth4_base64",
    "cascade_sum_simd16_ilp4_depth8_base16",
    "cascade_sum_simd16_ilp4_depth8_base32",
    "cascade_sum_simd16_ilp4_depth8_base64",
    "cascade_sum_simd16_ilp8_depth1_base0",
    "cascade_sum_simd16_ilp8_depth2_base16",
    "cascade_sum_simd16_ilp8_depth2_base32",
    "cascade_sum_simd16_ilp8_depth2_base64",
    "cascade_sum_simd16_ilp8_depth4_base16",
    "cascade_sum_simd16_ilp8_depth4_base32",
    "cascade_sum_simd16_ilp8_depth4_base64",
    "cascade_sum_simd16_ilp8_depth8_base16",
    "cascade_sum_simd16_ilp8_depth8_base32",
    "cascade_sum_simd16_ilp8_depth8_base64",
};
